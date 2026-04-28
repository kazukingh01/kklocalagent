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

use std::sync::Arc;

use anyhow::{Context, Result};
use base64::Engine;
use reqwest::multipart;
use serde::Serialize;
use serde_json::json;
use tokio::sync::{mpsc, Semaphore};
use tracing::{info, warn};

use crate::config::{AsrConfig, LlmConfig, ResultSinkConfig, TtsConfig};
use crate::state::WakeMachine;

/// Wrap a raw little-endian 16-bit mono PCM buffer in a 44-byte WAV
/// header. Duplicated from voice-activity-detection's helper — the two
/// modules are small enough that sharing a crate isn't worth the
/// workspace churn yet.
fn wav_from_pcm_s16le_mono(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let byte_rate = sample_rate * 2;
    let data_size = pcm.len() as u32;
    let chunk_size = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // PCM subchunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

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
        })
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
    // awaiting the consumer ensures the last /speak finishes before
    // we return (so the next turn's run_turn can't open a new /speak
    // while this one's tail is still pacing out).
    let _ = consumer.await;
    drop(llm_permit);

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
    let mut byte_buf: Vec<u8> = Vec::new();
    // Per-sentence text accumulator: deltas append here and we drain
    // each completed sentence out into the channel.
    let mut sentence_buf = String::new();
    let mut full_reply = String::new();

    'outer: loop {
        let chunk = resp
            .chunk()
            .await
            .context("read /api/chat stream")?;
        let chunk = match chunk {
            Some(c) => c,
            None => break, // EOF without explicit done:true — flush below.
        };
        byte_buf.extend_from_slice(&chunk);

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
/// terminator). Terminators: `。 ！ ？ ! ? \n`. Returns None if no
/// terminator is present yet.
fn find_sentence_end(s: &str) -> Option<usize> {
    for (i, ch) in s.char_indices() {
        if matches!(ch, '。' | '！' | '？' | '!' | '?' | '\n') {
            return Some(i + ch.len_utf8());
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
    fn wav_header_is_44_bytes_plus_data() {
        let pcm = vec![0u8; 320];
        let wav = wav_from_pcm_s16le_mono(&pcm, 16000);
        assert_eq!(wav.len(), 44 + 320);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        // Data chunk marker at offset 36.
        assert_eq!(&wav[36..40], b"data");
    }

    #[test]
    fn decode_audio_roundtrip() {
        let raw = vec![1u8, 2, 3, 4, 5];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
        assert_eq!(decode_audio(&b64).unwrap(), raw);
    }

    #[test]
    fn find_sentence_end_detects_each_terminator() {
        // 。 (3 bytes), ！ (3 bytes), ？ (3 bytes), ! and ? (1 byte), \n (1 byte).
        assert_eq!(find_sentence_end("こんにちは。world"), Some("こんにちは。".len()));
        assert_eq!(find_sentence_end("やあ！ next"), Some("やあ！".len()));
        assert_eq!(find_sentence_end("元気？ next"), Some("元気？".len()));
        assert_eq!(find_sentence_end("hi! next"), Some(3));
        assert_eq!(find_sentence_end("hi? next"), Some(3));
        assert_eq!(find_sentence_end("line1\nline2"), Some(6));
        assert_eq!(find_sentence_end("no terminator yet"), None);
        assert_eq!(find_sentence_end(""), None);
    }

    #[test]
    fn find_sentence_end_returns_first_terminator() {
        // Two sentences in one string: end of the *first* is what we want
        // so the caller can drain one sentence at a time.
        let s = "前。後！";
        let end = find_sentence_end(s).unwrap();
        assert_eq!(&s[..end], "前。");
    }
}
