//! SpeechEnded → ASR → LLM (streaming) → TTS pipeline.
//!
//! Shape of one turn:
//!
//! ```text
//! orchestrator ──POST /inference──────► ASR (whisper.cpp)
//!              ──POST /api/chat (ndjson)► LLM (ollama)
//!                  ├─ delta tokens accumulate into a sentence buffer
//!                  └─ each completed sentence ──POST /speak──► TTS
//! ```
//!
//! Per-stage concurrency is bounded by a `Semaphore` so the orchestrator
//! degrades predictably (drop with warning) instead of queuing unboundedly
//! if VAD fires faster than the backends can keep up. The TTS permit is
//! held for the *whole turn* (across N /speak calls) so two consecutive
//! sentences from the same turn don't race for the same slot.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use base64::Engine;
use reqwest::multipart;
use serde::Serialize;
use serde_json::json;
use tokio::sync::{mpsc, Semaphore};
use tracing::{info, warn};

use crate::config::{AsrConfig, LlmConfig, ResultSinkConfig, TtsConfig};
use crate::state::WakeMachine;
use wav_utils::wav_from_pcm_s16le_mono;

/// LLM chat request body for ollama's `/api/chat`.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// Per-stage HTTP client + backpressure permit.
pub struct Backends {
    pub http: reqwest::Client,
    pub asr: AsrConfig,
    pub llm: LlmConfig,
    pub tts: TtsConfig,
    pub result_sink: ResultSinkConfig,
    pub asr_inflight: Arc<Semaphore>,
    pub llm_inflight: Arc<Semaphore>,
    pub tts_inflight: Arc<Semaphore>,
    /// Deadline before which all VAD events (SS *and* SE) are dropped
    /// at the service.rs boundary. Set when run_turn finishes its TTS
    /// stage so the audio-io playback-ring tail can drain without the
    /// assistant's own voice being picked up by the mic and firing a
    /// new turn. `None` = no quiet window currently active.
    pub tts_quiet_until: Arc<Mutex<Option<Instant>>>,
}

impl Backends {
    pub fn new(
        asr: AsrConfig,
        llm: LlmConfig,
        tts: TtsConfig,
        result_sink: ResultSinkConfig,
    ) -> Result<Self> {
        // Client timeout is disabled at the client level; per-request
        // timeouts are applied in the individual POST builders below so
        // that slower models (ASR on `large-v3-turbo`, LLM on a big
        // prompt) can have independent budgets.
        let http = reqwest::Client::builder()
            .build()
            .context("building reqwest client")?;
        let asr_inflight = Arc::new(Semaphore::new(asr.max_inflight as usize));
        let llm_inflight = Arc::new(Semaphore::new(llm.max_inflight as usize));
        // tts.max_inflight is 0 only when tts is disabled (validated in
        // Config::validate). Use max(1) so the Semaphore stays well-formed
        // either way — try_acquire is gated on tts.url being set anyway.
        let tts_inflight = Arc::new(Semaphore::new(tts.max_inflight.max(1) as usize));
        Ok(Self {
            http,
            asr,
            llm,
            tts,
            result_sink,
            asr_inflight,
            llm_inflight,
            tts_inflight,
            tts_quiet_until: Arc::new(Mutex::new(None)),
        })
    }

    /// True when the configured `tts.tail_quiet_ms` window is still
    /// active — service.rs uses this to drop VAD events that would
    /// otherwise be the assistant's own voice echoing back through
    /// the mic during the audio-io playback tail.
    pub fn in_tts_quiet_window(&self) -> bool {
        match *self.tts_quiet_until.lock().expect("tts_quiet poisoned") {
            Some(t) => t > Instant::now(),
            None => false,
        }
    }
}

/// Best-effort POST to the configured `result_sink.url`. Silent no-op
/// when the sink is unconfigured. Failures log a warning but never
/// propagate — the pipeline must keep running even if a downstream
/// observer is offline.
pub async fn forward_to_result_sink(backends: &Backends, payload: &serde_json::Value) {
    if backends.result_sink.url.is_empty() {
        return;
    }
    let res = backends
        .http
        .post(&backends.result_sink.url)
        .json(payload)
        .timeout(std::time::Duration::from_millis(
            backends.result_sink.timeout_ms,
        ))
        .send()
        .await;
    match res {
        Ok(resp) => {
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                warn!(
                    target: "orch::sink",
                    "result_sink POST -> {}: {}",
                    status,
                    body.chars().take(200).collect::<String>()
                );
            }
        }
        Err(e) => warn!(target: "orch::sink", "result_sink POST failed: {e:#}"),
    }
}

/// Run one full SpeechEnded → ASR → LLM → sink → TTS turn. Logs the
/// assistant reply on success; logs and swallows errors at each stage
/// so one bad utterance can't bring the service down.
///
/// Barge-in (`WakeWordDetected` mid-turn with `barge_in=true`) is
/// driven from `service.rs`: it calls `JoinHandle::abort()` on this
/// task's spawn, which immediately cancels every await below — the
/// in-flight ASR/LLM/TTS HTTP responses are dropped (closing the
/// connection so the upstream stops producing), the mpsc sentence
/// channel is closed (so the consumer task exits and releases the
/// turn-scoped TTS permit), and the ASR/LLM permits drop with the
/// run_turn locals. The polling `wake.pipeline_still_active()`
/// checks below remain as belt-and-braces for the no-barge_in path
/// (where the running turn must finish but downstream stages can
/// still notice and skip), and to short-circuit cleanly if abort
/// hasn't landed yet at a stage boundary.
pub async fn run_turn(
    backends: Arc<Backends>,
    wake: Arc<WakeMachine>,
    pcm: Vec<u8>,
    sample_rate: u32,
) {
    // ASR stage
    let asr_permit = match backends.asr_inflight.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            warn!(
                target: "orch::pipeline",
                "ASR at capacity ({} in flight); dropping utterance",
                backends.asr.max_inflight
            );
            return;
        }
    };

    let wav = wav_from_pcm_s16le_mono(&pcm, sample_rate);
    info!(
        target: "orch::pipeline",
        bytes = wav.len(),
        sample_rate,
        "transcribing utterance"
    );

    let text = match asr_transcribe(&backends, wav).await {
        Ok(t) => t,
        Err(e) => {
            warn!(target: "orch::pipeline", "ASR failed: {e:#}");
            drop(asr_permit);
            return;
        }
    };
    drop(asr_permit);

    if !wake.pipeline_still_active() {
        info!(target: "orch::pipeline", "barge-in detected after ASR; aborting turn (LLM / sink / TTS skipped)");
        return;
    }

    if text.is_empty() {
        info!(target: "orch::pipeline", "ASR returned empty text; skipping LLM");
        return;
    }
    // Whisper hallucination guard. Whisper fills ambiguous near-
    // silence with stock YouTube end-of-video phrases ("ご視聴あり
    // がとうございました", "(拍手)", "Thanks for watching", etc.) —
    // none of which are plausible voice-agent inputs. Treat as an
    // empty transcription so the LLM never sees them and TTS doesn't
    // speak a response to nothing the operator said. Substring match
    // catches punctuated and trailing-text variants in one rule each.
    if let Some(matched) = backends
        .asr
        .hallucination_blacklist
        .iter()
        .find(|p| text.contains(p.as_str()))
    {
        info!(
            target: "orch::pipeline",
            text = %text,
            matched = %matched,
            "ASR returned a known whisper hallucination; skipping LLM"
        );
        return;
    }
    info!(target: "orch::pipeline", text = %text, "transcribed");

    // LLM stage (streaming).
    //
    // Pipeline shape (pipelined, sentence-granular):
    //   LLM /api/chat (stream=true) ─emit sentence─► mpsc ─► TTS consumer
    //                                                          │
    //                                              POST /speak (serial)
    //
    // While the LLM is still generating sentence N+1, the consumer is
    // already pacing sentence N's audio out to the streamer. First
    // audio reaches the user when the *first* sentence boundary is
    // hit, instead of when the *last* token is generated. The TTS
    // semaphore is held once for the whole turn (not per sentence) so
    // two consecutive sentences from the same turn don't race for the
    // permit and skip themselves.
    let llm_permit = match backends.llm_inflight.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            warn!(
                target: "orch::pipeline",
                "LLM at capacity ({} in flight); dropping utterance",
                backends.llm.max_inflight
            );
            return;
        }
    };

    // Channel buffers up to 8 sentences. Tuned to absorb a slow TTS
    // step (synthesis can be a few hundred ms) without back-pressuring
    // the LLM read loop on a fast model — but bounded so a runaway
    // generation can't OOM the orchestrator.
    let (sentence_tx, sentence_rx) = mpsc::channel::<String>(8);

    // Acquire the TTS permit once per turn. None when at capacity or
    // when TTS is disabled — consumer still drains the channel either
    // way (so the LLM read loop never wedges) but skips the actual
    // POST /speak.
    let tts_permit = if !backends.tts.url.is_empty() {
        match backends.tts_inflight.clone().try_acquire_owned() {
            Ok(p) => Some(p),
            Err(_) => {
                warn!(
                    target: "orch::pipeline",
                    "TTS at capacity ({} in flight); turn will skip /speak",
                    backends.tts.max_inflight
                );
                None
            }
        }
    } else {
        None
    };

    let consumer = spawn_tts_consumer(
        backends.clone(),
        wake.clone(),
        sentence_rx,
        tts_permit,
    );

    let reply_result = llm_chat_streaming(&backends, &wake, &text, sentence_tx).await;
    // Sender dropped here → channel closes once consumer drains;
    // awaiting the consumer ensures every /speak HTTP has returned
    // before we move on (so the next turn's run_turn can't open a
    // new /speak while this one is still pushing PCM frames).
    let _ = consumer.await;
    drop(llm_permit);

    // Convert "we sent the last PCM frame to the WS" into "the
    // speaker actually fell silent" by asking tts-streamer to do the
    // EOS/drained handshake against audio-io. /finalize's response
    // is the precise silent moment. Skipped when finalize_url is
    // empty (tail_quiet_ms then has to absorb audio-io's playback
    // ring drain time as well).
    if !backends.tts.finalize_url.is_empty() {
        tts_finalize(&backends).await;
    }

    // Open the post-TTS VAD quiet window. With the drain handshake
    // above, this only needs to cover VAD's silence hangover (~200 ms
    // for hang_frames=10) plus a propagation safety margin — VAD will
    // fire SE that long after the audio actually ended, and that SE
    // would otherwise dispatch an echo turn. service.rs::events
    // checks `backends.in_tts_quiet_window()` before forwarding any
    // SS/SE to the wake machine.
    if !backends.tts.url.is_empty() && backends.tts.tail_quiet_ms > 0 {
        let until = Instant::now() + Duration::from_millis(backends.tts.tail_quiet_ms);
        *backends.tts_quiet_until.lock().expect("tts_quiet poisoned") = Some(until);
        info!(
            target: "orch::pipeline",
            quiet_ms = backends.tts.tail_quiet_ms,
            "TTS drained; opening VAD quiet window"
        );
    }

    let reply = match reply_result {
        Ok(r) => r,
        Err(e) => {
            warn!(target: "orch::pipeline", "LLM failed: {e:#}");
            return;
        }
    };

    if !wake.pipeline_still_active() {
        info!(target: "orch::pipeline", "barge-in detected after LLM; aborting turn (sink skipped)");
        return;
    }

    info!(
        target: "orch::pipeline",
        user = %text,
        assistant = %reply,
        "turn complete"
    );

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let payload = json!({
        "name": "TurnCompleted",
        "user": text,
        "assistant": reply,
        "ts": ts,
    });
    forward_to_result_sink(&backends, &payload).await;
}

/// TTS consumer task: drains the sentence channel, calling
/// `tts_speak_inner` serially per sentence while the turn is still
/// active. Holds the turn-level TTS permit until the channel closes.
fn spawn_tts_consumer(
    backends: Arc<Backends>,
    wake: Arc<WakeMachine>,
    mut rx: mpsc::Receiver<String>,
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(sentence) = rx.recv().await {
            // Drain the rest after barge-in so the LLM sender can
            // finish without blocking, but skip the actual /speak.
            // tts_stop() (fired from on_wake → service.rs) has
            // already cancelled any in-flight /speak — we just stop
            // queuing new ones.
            if !wake.pipeline_still_active() {
                continue;
            }
            if permit.is_none() || backends.tts.url.is_empty() || sentence.is_empty() {
                continue;
            }
            tts_speak_inner(&backends, &sentence).await;
        }
        drop(permit);
    })
}

/// Inner /speak POST. Permit management is the caller's responsibility
/// — see `spawn_tts_consumer` (per-turn permit) for the streaming path.
async fn tts_speak_inner(backends: &Backends, text: &str) {
    let body = json!({ "text": text });
    let res = backends
        .http
        .post(&backends.tts.url)
        .json(&body)
        .timeout(std::time::Duration::from_millis(backends.tts.timeout_ms))
        .send()
        .await;
    match res {
        Ok(resp) => {
            let status = resp.status();
            // 499 is what tts-streamer returns when /stop cancelled
            // an in-flight /speak (barge-in). It's the *expected*
            // outcome for the cancelled turn — log at info, not warn,
            // so it doesn't pollute production logs every time the
            // user interrupts the assistant.
            if status.is_success() || status.as_u16() == 499 {
                info!(target: "orch::pipeline", "TTS ok ({status})");
            } else {
                let body = resp.text().await.unwrap_or_default();
                warn!(
                    target: "orch::pipeline",
                    "TTS responded {}: {}",
                    status,
                    body.chars().take(200).collect::<String>()
                );
            }
        }
        Err(e) => warn!(target: "orch::pipeline", "TTS POST failed: {e:#}"),
    }
}

/// POST `tts.finalize_url` to wait for tts-streamer's drain
/// handshake with audio-io. Returns when audio-io reports its
/// playback ring is empty (= the speaker fell silent). Caller
/// should have already drained every per-sentence /speak before
/// calling this. Failures are logged but never propagate — the
/// quiet window will still be opened immediately afterward, just
/// without the precise silent-moment anchor.
pub async fn tts_finalize(backends: &Backends) {
    if backends.tts.finalize_url.is_empty() {
        return;
    }
    let started = Instant::now();
    let res = backends
        .http
        .post(&backends.tts.finalize_url)
        .timeout(std::time::Duration::from_millis(backends.tts.timeout_ms))
        .send()
        .await;
    let elapsed_ms = started.elapsed().as_millis();
    match res {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                info!(
                    target: "orch::pipeline",
                    elapsed_ms,
                    "TTS /finalize ok ({status})"
                );
            } else {
                let body = resp.text().await.unwrap_or_default();
                warn!(
                    target: "orch::pipeline",
                    elapsed_ms,
                    "TTS /finalize responded {}: {}",
                    status,
                    body.chars().take(200).collect::<String>()
                );
            }
        }
        Err(e) => warn!(
            target: "orch::pipeline",
            elapsed_ms,
            "TTS /finalize POST failed: {e:#}"
        ),
    }
}

/// POST `tts.stop_url` to cancel an in-flight `/speak` on the
/// streamer. No-op when stop_url is empty (barge-in disabled or
/// tts-streamer doesn't expose `/stop`). Failures are logged but
/// never propagate — the wake state has already transitioned, so
/// failing the stop POST shouldn't block the next turn.
pub async fn tts_stop(backends: &Backends) {
    if backends.tts.stop_url.is_empty() {
        return;
    }
    let res = backends
        .http
        .post(&backends.tts.stop_url)
        .timeout(std::time::Duration::from_millis(backends.tts.timeout_ms))
        .send()
        .await;
    match res {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                info!(target: "orch::pipeline", "TTS stop ok ({status})");
            } else {
                let body = resp.text().await.unwrap_or_default();
                warn!(
                    target: "orch::pipeline",
                    "TTS stop responded {}: {}",
                    status,
                    body.chars().take(200).collect::<String>()
                );
            }
        }
        Err(e) => warn!(target: "orch::pipeline", "TTS stop POST failed: {e:#}"),
    }
}

async fn asr_transcribe(backends: &Backends, wav: Vec<u8>) -> Result<String> {
    let part = multipart::Part::bytes(wav)
        .file_name("utterance.wav")
        .mime_str("audio/wav")?;
    let form = multipart::Form::new()
        .part("file", part)
        .text("response_format", "json")
        .text("temperature", "0");
    let resp = backends
        .http
        .post(&backends.asr.url)
        .multipart(form)
        .timeout(std::time::Duration::from_millis(backends.asr.timeout_ms))
        .send()
        .await
        .context("POST /inference")?;
    let status = resp.status();
    let body = resp.text().await.context("read /inference body")?;
    if !status.is_success() {
        anyhow::bail!("ASR responded {status}: {body}");
    }
    // whisper-server with response_format=json returns {"text": "..."}.
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
    let text = parsed
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| body.trim().to_string());
    Ok(text)
}

/// Streaming /api/chat. Reads ollama's ndjson response one line at a
/// time, accumulates `message.content` deltas, and emits each sentence
/// (split on `。 ！ ？ ! ? \n`) into `sentence_tx` as soon as it
/// completes. The full assistant reply is returned at the end for
/// logging + sink forwarding.
///
/// Barge-in: a `wake.pipeline_still_active() == false` between
/// sentences (or between deltas) returns early, dropping the response
/// stream and closing the underlying connection — so ollama stops
/// generating tokens we'd never use.
async fn llm_chat_streaming(
    backends: &Backends,
    wake: &WakeMachine,
    user_text: &str,
    sentence_tx: mpsc::Sender<String>,
) -> Result<String> {
    // Prepend system prompt when configured. ollama's /api/chat
    // normalises {role:"system"} into the model's chat template
    // regardless of which model is loaded — empty string here means
    // "send only the user turn" (v0.x behaviour).
    let mut messages = Vec::with_capacity(2);
    if !backends.llm.system_prompt.is_empty() {
        messages.push(ChatMessage {
            role: "system",
            content: &backends.llm.system_prompt,
        });
    }
    messages.push(ChatMessage {
        role: "user",
        content: user_text,
    });
    let body = ChatRequest {
        model: &backends.llm.model,
        messages,
        stream: true,
    };
    let llm_started = Instant::now();
    let mut resp = backends
        .http
        .post(&backends.llm.url)
        .json(&body)
        .timeout(std::time::Duration::from_millis(backends.llm.timeout_ms))
        .send()
        .await
        .context("POST /api/chat")?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("LLM responded {status}: {text}");
    }

    // Byte-level accumulator (chunks may split mid-line); newline (0x0A)
    // is ASCII and never appears mid-UTF-8 codepoint, so splitting at
    // `\n` is always at a valid string boundary.
    //
    // Hard cap so a malformed upstream that streams without ever
    // emitting `\n` can't drag the orchestrator down with unbounded
    // memory growth. ollama in practice emits one ndjson line per
    // delta token (typically <1 KB), so even a long verbose reply
    // stays well under this limit. Hitting the cap is "the response
    // shape isn't ndjson" — bail and let the turn skip TTS.
    const LLM_STREAM_BUF_MAX: usize = 1 << 20; // 1 MiB
    let mut byte_buf: Vec<u8> = Vec::new();
    // Per-sentence text accumulator: deltas append here and we drain
    // each completed sentence out into the channel.
    let mut sentence_buf = String::new();
    let mut full_reply = String::new();
    // Diagnostic timing for the streaming path. TTFB = time to first
    // body chunk (covers prompt eval). TTFS = time to first sentence
    // boundary (covers prompt eval + token generation up to the first
    // terminator). Together they pinpoint whether a slow turn is the
    // model warming up vs. the model generating verbose preamble
    // before any sentence break.
    let mut first_chunk_logged = false;
    let mut first_sentence_logged = false;

    'outer: loop {
        let chunk = resp
            .chunk()
            .await
            .context("read /api/chat stream")?;
        let chunk = match chunk {
            Some(c) => c,
            None => break, // EOF without explicit done:true — flush below.
        };
        if !first_chunk_logged {
            info!(
                target: "orch::pipeline",
                ttfb_ms = llm_started.elapsed().as_millis(),
                "LLM first chunk received"
            );
            first_chunk_logged = true;
        }
        byte_buf.extend_from_slice(&chunk);
        if byte_buf.len() > LLM_STREAM_BUF_MAX {
            anyhow::bail!(
                "LLM stream produced {} bytes without a newline (cap {}); aborting turn",
                byte_buf.len(),
                LLM_STREAM_BUF_MAX
            );
        }

        while let Some(nl_pos) = byte_buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = byte_buf.drain(..=nl_pos).collect();
            let line = match std::str::from_utf8(&raw[..nl_pos]) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }
            let parsed: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    warn!(target: "orch::pipeline", "skipping unparseable LLM stream line: {e}");
                    continue;
                }
            };

            let delta = parsed
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("");
            if !delta.is_empty() {
                sentence_buf.push_str(delta);
                full_reply.push_str(delta);

                // Drain every completed sentence from the buffer.
                while let Some(end) = find_sentence_end(&sentence_buf) {
                    let remainder = sentence_buf.split_off(end);
                    let sentence = std::mem::replace(&mut sentence_buf, remainder)
                        .trim()
                        .to_string();
                    if sentence.is_empty() {
                        continue;
                    }
                    if !wake.pipeline_still_active() {
                        // Drop response → connection closes →
                        // ollama stops generating. No further
                        // sentences emitted.
                        return Ok(full_reply);
                    }
                    if !first_sentence_logged {
                        info!(
                            target: "orch::pipeline",
                            ttfs_ms = llm_started.elapsed().as_millis(),
                            "LLM first sentence emitted"
                        );
                        first_sentence_logged = true;
                    }
                    if sentence_tx.send(sentence).await.is_err() {
                        // Consumer gone — abandon stream.
                        return Ok(full_reply);
                    }
                }
            }

            if parsed
                .get("done")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                break 'outer;
            }
        }
    }

    // Flush trailing partial sentence (final chunk had no terminator).
    let tail = sentence_buf.trim().to_string();
    if !tail.is_empty() && wake.pipeline_still_active() {
        let _ = sentence_tx.send(tail).await;
    }
    Ok(full_reply.trim().to_string())
}

/// Find the first sentence-terminator in `s` and return the byte
/// offset *after* it (so `s[..end]` is the sentence including its
/// terminator). Returns None if no terminator is present yet.
///
/// Terminator policy:
/// * Full-width `。 ！ ？` and prosodic breaks `、 …`, plus `\n`,
///   are unconditional terminators. The Japanese comma `、` is safe
///   because it never appears mid-numeric, and VOICEVOX inserts a
///   short prosodic pause at it so flushing per-`、` shortens
///   time-to-first-audio without warping cadence.
/// * ASCII `. ! ?` are terminators *only when followed by whitespace*.
///   The lookahead lets English-only LLM replies stream sentence-by-
///   sentence — without an ASCII rule, a model that answers in pure
///   English produces zero terminators and `sentence_buf` accumulates
///   the whole turn before the trailing flush, defeating streaming.
///   The whitespace gate keeps numerics like "1.5" and host names
///   like "api.example.com" intact. Abbreviations followed by a
///   space ("Mr. Smith", "etc. and") still split — accepted v1
///   trade-off because the resulting TTS just gets a small extra
///   pause where the period is, which sounds like a beat in fluent
///   reading.
/// * ASCII `,` deliberately stays out — English clausal commas
///   ("a, b, and c") would each become a separate TTS unit with a
///   wrong-feeling break.
/// * A bare ASCII terminator at end-of-buffer (no lookahead char)
///   does *not* split — the trailing flush at end-of-stream emits
///   the final partial sentence.
fn find_sentence_end(s: &str) -> Option<usize> {
    let mut iter = s.char_indices().peekable();
    while let Some((i, ch)) = iter.next() {
        match ch {
            '。' | '、' | '！' | '？' | '…' | '\n' => {
                return Some(i + ch.len_utf8());
            }
            '!' | '?' | '.' => {
                if let Some(&(_, next)) = iter.peek() {
                    if next.is_whitespace() {
                        return Some(i + ch.len_utf8());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Decode `audio_base64` to raw PCM bytes.
pub fn decode_audio(b64: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .context("decode audio_base64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_audio_roundtrip() {
        let raw = vec![1u8, 2, 3, 4, 5];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
        assert_eq!(decode_audio(&b64).unwrap(), raw);
    }

    #[test]
    fn find_sentence_end_detects_each_terminator() {
        // Multibyte: 。！？、… are 3 bytes each. ASCII !? and \n are 1.
        assert_eq!(find_sentence_end("こんにちは。world"), Some("こんにちは。".len()));
        assert_eq!(find_sentence_end("やあ！ next"), Some("やあ！".len()));
        assert_eq!(find_sentence_end("元気？ next"), Some("元気？".len()));
        assert_eq!(find_sentence_end("えーと、それで"), Some("えーと、".len()));
        assert_eq!(find_sentence_end("うーん…続き"), Some("うーん…".len()));
        // ASCII !? followed by whitespace → split (the common English
        // sentence-end shape; matches v0 behaviour).
        assert_eq!(find_sentence_end("hi! next"), Some(3));
        assert_eq!(find_sentence_end("hi? next"), Some(3));
        assert_eq!(find_sentence_end("line1\nline2"), Some(6));
        assert_eq!(find_sentence_end("no terminator yet"), None);
        // ASCII `,` stays non-terminator (English clausal commas would
        // produce wrong-feeling prosodic breaks via VOICEVOX).
        assert_eq!(find_sentence_end("price 1,000 yen"), None);
        assert_eq!(find_sentence_end("a, b, and c"), None);
        assert_eq!(find_sentence_end(""), None);
    }

    #[test]
    fn find_sentence_end_ascii_period_requires_whitespace_after() {
        // The English-streaming rule. `.` followed by space terminates;
        // `.` mid-numeric or mid-identifier does not. Without this,
        // English LLM replies never stream — `sentence_buf` accumulates
        // the entire turn until the trailing flush.
        assert_eq!(find_sentence_end("Hello. World"), Some(6));
        assert_eq!(find_sentence_end("Done.\nNext"), Some(5));
        // Numerics / host names / file extensions stay intact.
        assert_eq!(find_sentence_end("about 1.5 meters"), None);
        assert_eq!(find_sentence_end("api.example.com"), None);
        assert_eq!(find_sentence_end("file.txt is here"), None);
        // Same gate applies to ASCII ! and ?.
        assert_eq!(find_sentence_end("Hi!World"), None);
        assert_eq!(find_sentence_end("Why?Yes"), None);
        // Bare terminator at end-of-buffer doesn't split — the trailing
        // flush at end-of-stream emits the final partial sentence.
        assert_eq!(find_sentence_end("Done."), None);
        assert_eq!(find_sentence_end("Done!"), None);
        assert_eq!(find_sentence_end("Done?"), None);
    }

    #[test]
    fn find_sentence_end_returns_first_terminator() {
        // Two sentences in one string: end of the *first* is what we want
        // so the caller can drain one sentence at a time.
        let s = "前。後！";
        let end = find_sentence_end(s).unwrap();
        assert_eq!(&s[..end], "前。");
    }

    #[test]
    fn in_tts_quiet_window_respects_deadline() {
        // Build a minimal Backends; only the tts_quiet_until field
        // matters for this test, the rest can be defaults.
        let backends = Backends::new(
            crate::config::AsrConfig::default(),
            crate::config::LlmConfig::default(),
            crate::config::TtsConfig::default(),
            crate::config::ResultSinkConfig::default(),
        )
        .unwrap();

        // Initial state: no quiet window has been opened yet.
        assert!(!backends.in_tts_quiet_window());

        // Future deadline → window is active.
        *backends.tts_quiet_until.lock().unwrap() =
            Some(Instant::now() + Duration::from_millis(50));
        assert!(backends.in_tts_quiet_window());

        // Past deadline → window has lapsed (the field is left set
        // intentionally; service.rs never bothers to clear it because
        // a stale Instant in the past is a no-op for the comparison).
        *backends.tts_quiet_until.lock().unwrap() =
            Some(Instant::now() - Duration::from_millis(1));
        assert!(!backends.in_tts_quiet_window());

        // Cleared (None) → no window.
        *backends.tts_quiet_until.lock().unwrap() = None;
        assert!(!backends.in_tts_quiet_window());
    }

    #[test]
    fn in_tts_quiet_window_boundary_exactly_now_is_not_active() {
        // The check is `t > Instant::now()` (strict greater-than),
        // so a deadline equal to "now" is treated as already lapsed.
        // Pin this in a test so a future refactor to >= doesn't slip
        // in unnoticed (would silently widen the echo-suppression
        // window by one tick).
        let backends = Backends::new(
            crate::config::AsrConfig::default(),
            crate::config::LlmConfig::default(),
            crate::config::TtsConfig::default(),
            crate::config::ResultSinkConfig::default(),
        )
        .unwrap();
        let now = Instant::now();
        *backends.tts_quiet_until.lock().unwrap() = Some(now);
        // By the time `in_tts_quiet_window` reads `Instant::now()`
        // again, even nanoseconds have elapsed, so `t > Instant::now()`
        // must be false.
        assert!(!backends.in_tts_quiet_window());
    }

    // --- llm_chat_streaming integration tests ------------------------------
    //
    // These bind a local tokio TCP listener and write a hand-rolled HTTP/1.1
    // response, mocking ollama's `/api/chat` ndjson stream. The point is to
    // exercise the *parsing* path (chunk re-assembly, ndjson splitting,
    // delta accumulation, sentence drain) end-to-end on the real reqwest
    // stack, since that's where streaming bugs hide. Spinning up axum
    // would be heavier and leak more deps into dev-deps.

    use crate::config::{AsrConfig, LlmConfig, ResultSinkConfig, TtsConfig, WakeConfig};
    use crate::state::WakeMachine;
    use std::sync::Arc as StdArc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Spawn a one-shot HTTP server that accepts a single connection,
    /// reads (and discards) the request, and writes back a fixed body
    /// with the given content-type. Returns the bound address.
    async fn spawn_mock_llm(body: Vec<u8>) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            // Read and discard the request headers + body. We only
            // need to drain enough that the client's POST completes;
            // the actual content is irrelevant for the parser test.
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf).await;
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = sock.write_all(header.as_bytes()).await;
            let _ = sock.write_all(&body).await;
            let _ = sock.shutdown().await;
        });
        addr
    }

    /// Build a minimal `Backends` whose llm.url points at `addr`.
    /// All other backend URLs stay empty so no live calls are made.
    fn backends_with_llm_addr(addr: std::net::SocketAddr) -> StdArc<Backends> {
        let asr = AsrConfig::default();
        let mut llm = LlmConfig::default();
        llm.url = format!("http://{addr}/api/chat");
        // Long enough that the test never trips it but short enough
        // that a hung mock fails fast.
        llm.timeout_ms = 5_000;
        let tts = TtsConfig::default();
        let result_sink = ResultSinkConfig::default();
        StdArc::new(Backends::new(asr, llm, tts, result_sink).unwrap())
    }

    /// `WakeMachine` in always-listening mode so `pipeline_still_active()`
    /// returns true throughout — otherwise the streaming function would
    /// short-circuit on the very first sentence.
    fn loose_wake() -> StdArc<WakeMachine> {
        let mut cfg = WakeConfig::default();
        cfg.required = false;
        StdArc::new(WakeMachine::new(&cfg))
    }

    #[tokio::test]
    async fn llm_chat_streaming_drains_japanese_sentences_in_order() {
        // Three deltas split across line boundaries. The middle delta
        // crosses a sentence boundary (ends mid-sentence) to make sure
        // the per-line drain correctly accumulates across deltas.
        let body = concat!(
            r#"{"message":{"content":"こんにちは"}}"#, "\n",
            r#"{"message":{"content":"。今日は"}}"#, "\n",
            r#"{"message":{"content":"いい天気ですね。"}}"#, "\n",
            r#"{"done":true}"#, "\n",
        ).as_bytes().to_vec();
        let addr = spawn_mock_llm(body).await;
        let backends = backends_with_llm_addr(addr);
        let wake = loose_wake();
        let (tx, mut rx) = mpsc::channel::<String>(8);
        let reply = llm_chat_streaming(&backends, &wake, "test", tx).await.unwrap();
        let mut sentences = vec![];
        while let Some(s) = rx.recv().await {
            sentences.push(s);
        }
        assert_eq!(sentences, vec!["こんにちは。", "今日はいい天気ですね。"]);
        assert_eq!(reply, "こんにちは。今日はいい天気ですね。");
    }

    #[tokio::test]
    async fn llm_chat_streaming_drains_english_sentences_on_period_then_space() {
        // English-only reply: no Japanese terminators ever appear, so
        // streaming is fully driven by the ASCII period/?/! +
        // whitespace rule. Without that rule sentence_buf would
        // accumulate the entire reply until the trailing flush.
        let body = concat!(
            r#"{"message":{"content":"Hello world."}}"#, "\n",
            r#"{"message":{"content":" How are you?"}}"#, "\n",
            r#"{"message":{"content":" I am fine!"}}"#, "\n",
            r#"{"done":true}"#, "\n",
        ).as_bytes().to_vec();
        let addr = spawn_mock_llm(body).await;
        let backends = backends_with_llm_addr(addr);
        let wake = loose_wake();
        let (tx, mut rx) = mpsc::channel::<String>(8);
        let _reply = llm_chat_streaming(&backends, &wake, "test", tx).await.unwrap();
        let mut sentences = vec![];
        while let Some(s) = rx.recv().await {
            sentences.push(s);
        }
        assert_eq!(
            sentences,
            vec!["Hello world.", "How are you?", "I am fine!"]
        );
    }

    #[tokio::test]
    async fn llm_chat_streaming_preserves_numerics_with_commas_and_periods() {
        // Regression for the comma-was-a-terminator bug: "1,000" must
        // arrive as one sentence, not two prosodic units. Same for
        // "1.5" (already covered by the find_sentence_end test, but
        // we re-check it via the streaming path because that's where
        // operators see the wrong-prosody symptom).
        let body = concat!(
            r#"{"message":{"content":"値段は1,000円で、サイズは1.5"}}"#, "\n",
            r#"{"message":{"content":"メートルです。"}}"#, "\n",
            r#"{"done":true}"#, "\n",
        ).as_bytes().to_vec();
        let addr = spawn_mock_llm(body).await;
        let backends = backends_with_llm_addr(addr);
        let wake = loose_wake();
        let (tx, mut rx) = mpsc::channel::<String>(8);
        let _reply = llm_chat_streaming(&backends, &wake, "test", tx).await.unwrap();
        let mut sentences = vec![];
        while let Some(s) = rx.recv().await {
            sentences.push(s);
        }
        // Japanese `、` *is* a terminator (prosodic break that
        // VOICEVOX renders cleanly), so we expect a split there.
        // The ASCII `,` and `.` inside numerics must NOT split.
        assert_eq!(
            sentences,
            vec!["値段は1,000円で、", "サイズは1.5メートルです。"]
        );
    }

    #[tokio::test]
    async fn llm_chat_streaming_skips_unparseable_lines() {
        // Forward-compat: garbage line in the middle of a stream
        // shouldn't kill the turn — log + skip and keep parsing.
        let body = concat!(
            r#"{"message":{"content":"はい、"}}"#, "\n",
            "garbage not json\n",
            r#"{"message":{"content":"了解しました。"}}"#, "\n",
            r#"{"done":true}"#, "\n",
        ).as_bytes().to_vec();
        let addr = spawn_mock_llm(body).await;
        let backends = backends_with_llm_addr(addr);
        let wake = loose_wake();
        let (tx, mut rx) = mpsc::channel::<String>(8);
        let reply = llm_chat_streaming(&backends, &wake, "test", tx).await.unwrap();
        let mut sentences = vec![];
        while let Some(s) = rx.recv().await {
            sentences.push(s);
        }
        assert_eq!(sentences, vec!["はい、", "了解しました。"]);
        assert_eq!(reply, "はい、了解しました。");
    }

    #[tokio::test]
    async fn llm_chat_streaming_caps_unbounded_buffer_without_newlines() {
        // Pathological upstream: 2 MiB of body without ever sending
        // a newline. The orchestrator must bail before swallowing
        // the whole thing — otherwise a malformed LLM could OOM us.
        let body = vec![b'x'; 2 * 1024 * 1024];
        let addr = spawn_mock_llm(body).await;
        let backends = backends_with_llm_addr(addr);
        let wake = loose_wake();
        let (tx, _rx) = mpsc::channel::<String>(8);
        let err = llm_chat_streaming(&backends, &wake, "test", tx)
            .await
            .expect_err("expected bail on unbounded body");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("without a newline"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn llm_chat_streaming_flushes_trailing_partial_sentence() {
        // Last delta ends without a terminator; the function should
        // still emit it as a tail sentence so VOICEVOX speaks it.
        let body = concat!(
            r#"{"message":{"content":"続きの一文"}}"#, "\n",
            r#"{"done":true}"#, "\n",
        ).as_bytes().to_vec();
        let addr = spawn_mock_llm(body).await;
        let backends = backends_with_llm_addr(addr);
        let wake = loose_wake();
        let (tx, mut rx) = mpsc::channel::<String>(8);
        let reply = llm_chat_streaming(&backends, &wake, "test", tx).await.unwrap();
        let mut sentences = vec![];
        while let Some(s) = rx.recv().await {
            sentences.push(s);
        }
        assert_eq!(sentences, vec!["続きの一文"]);
        assert_eq!(reply, "続きの一文");
    }
}
