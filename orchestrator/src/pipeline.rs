use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use base64::Engine;
use reqwest::multipart;
use serde::Serialize;
use serde_json::json;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, info, warn};

use crate::config::{AsrConfig, LlmConfig, ResultSinkConfig, TtsConfig};
use crate::state::WakeMachine;
use wav_utils::wav_from_pcm_s16le_mono;

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

pub struct Backends {
    pub http: reqwest::Client,
    pub asr: AsrConfig,
    pub llm: LlmConfig,
    pub tts: TtsConfig,
    pub result_sink: ResultSinkConfig,
    pub asr_inflight: Arc<Semaphore>,
    pub llm_inflight: Arc<Semaphore>,
    pub tts_inflight: Arc<Semaphore>,
    pub tts_quiet_until: Arc<Mutex<Option<Instant>>>,
}

impl Backends {
    pub fn new(
        asr: AsrConfig,
        llm: LlmConfig,
        tts: TtsConfig,
        result_sink: ResultSinkConfig,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .build()
            .context("building reqwest client")?;
        let asr_inflight = Arc::new(Semaphore::new(asr.max_inflight as usize));
        let llm_inflight = Arc::new(Semaphore::new(llm.max_inflight as usize));
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

    pub fn in_tts_quiet_window(&self) -> bool {
        match *self.tts_quiet_until.lock().expect("tts_quiet poisoned") {
            Some(t) => t > Instant::now(),
            None => false,
        }
    }
}

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

/// Barge-in is driven from `service.rs` via `JoinHandle::abort()` on this
/// task: every await below cancels, the in-flight HTTP responses drop
/// (closing connections so upstreams stop producing), the mpsc sentence
/// channel closes (consumer exits and releases the turn-scoped TTS
/// permit), and the ASR/LLM permits drop with the locals. The polling
/// `wake.pipeline_still_active()` checks remain as belt-and-braces for
/// the no-barge_in path and for the gap before abort lands.
pub async fn run_turn(
    backends: Arc<Backends>,
    wake: Arc<WakeMachine>,
    mut pcm: Vec<u8>,
    sample_rate: u32,
) {
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

    // Whisper rejects inputs <1000 ms outright ("input is too short"), so
    // pad short utterances with silence; 1200 ms gives margin over the
    // hard floor and whisper handles trailing zeros without hallucinating.
    const MIN_ASR_MS: usize = 1200;
    let min_bytes = (sample_rate as usize) * 2 * MIN_ASR_MS / 1000;
    if pcm.len() < min_bytes {
        let pad = min_bytes - pcm.len();
        pcm.resize(min_bytes, 0);
        info!(
            target: "orch::pipeline",
            pad_bytes = pad,
            original_ms = pcm.len().saturating_sub(pad) * 1000 / (sample_rate as usize * 2),
            "padded utterance to {}ms (whisper rejects <1000ms inputs)",
            MIN_ASR_MS,
        );
    }

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

    let (sentence_tx, sentence_rx) = mpsc::channel::<String>(8);

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
    // Sender dropped here → channel closes; awaiting the consumer ensures
    // every /speak has returned before the next turn can open a new one.
    let _ = consumer.await;
    drop(llm_permit);

    if !backends.tts.finalize_url.is_empty() {
        tts_finalize(&backends).await;
    }

    open_tts_quiet_window(&backends);

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

/// TTS consumer: drains the sentence channel serially, holding the
/// turn-level TTS permit until the channel closes. First sentence goes to
/// `/speak` (resets the streamer's burst budget), the rest to `/append`
/// (reuses it — issue #16); `append_url` is validated at startup so the
/// continuation branch is always wired.
fn spawn_tts_consumer(
    backends: Arc<Backends>,
    wake: Arc<WakeMachine>,
    mut rx: mpsc::Receiver<String>,
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut is_first = true;
        while let Some(sentence) = rx.recv().await {
            // After barge-in, keep draining (so the LLM sender never
            // blocks) but skip the actual /speak — tts_stop() has already
            // cancelled the in-flight one.
            if !wake.pipeline_still_active() {
                continue;
            }
            if permit.is_none() || backends.tts.url.is_empty() || sentence.is_empty() {
                continue;
            }
            let (api_label, target_url) = if is_first {
                ("speak", &backends.tts.url)
            } else {
                ("append", &backends.tts.append_url)
            };
            tts_speak_inner(&backends, api_label, target_url, &sentence).await;
            is_first = false;
        }
        drop(permit);
    })
}

async fn tts_speak_inner(backends: &Backends, api: &str, url: &str, text: &str) {
    info!(
        target: "orch::pipeline",
        api,
        url,
        chars = text.chars().count(),
        "TTS POST: {:?}",
        text.chars().take(40).collect::<String>()
    );
    let body = json!({ "text": text });
    let res = backends
        .http
        .post(url)
        .json(&body)
        .timeout(std::time::Duration::from_millis(backends.tts.timeout_ms))
        .send()
        .await;
    match res {
        Ok(resp) => {
            let status = resp.status();
            // 499 = tts-streamer's "cancelled by /stop" (barge-in) — the
            // expected outcome for the cancelled turn, so info not warn.
            if status.is_success() || status.as_u16() == 499 {
                info!(target: "orch::pipeline", api, "TTS ok ({status})");
            } else {
                let body = resp.text().await.unwrap_or_default();
                warn!(
                    target: "orch::pipeline",
                    api,
                    "TTS responded {}: {}",
                    status,
                    body.chars().take(200).collect::<String>()
                );
            }
        }
        Err(e) => warn!(target: "orch::pipeline", api, "TTS POST failed: {e:#}"),
    }
}

pub fn open_tts_quiet_window(backends: &Backends) {
    if !backends.tts.url.is_empty() && backends.tts.tail_quiet_ms > 0 {
        let until = Instant::now() + Duration::from_millis(backends.tts.tail_quiet_ms);
        *backends.tts_quiet_until.lock().expect("tts_quiet poisoned") = Some(until);
        info!(
            target: "orch::pipeline",
            quiet_ms = backends.tts.tail_quiet_ms,
            "TTS drained; opening VAD quiet window"
        );
    }
}

pub async fn tts_wake_ack(backends: &Backends) {
    let text = backends.tts.wake_ack_text.trim();
    if text.is_empty() || backends.tts.url.is_empty() {
        return;
    }
    tts_speak_inner(backends, "wake-ack", &backends.tts.url, text).await;
    // Same tail protection as a normal turn: the fixed post-wake SE dropout
    // misses the ack's echo when synthesis/playback runs late, so wait for
    // the speaker to actually drain, then gate VAD for tail_quiet from there.
    // A user SS eaten by this window is harmless — SE alone dispatches from
    // ArmedAfterWake and carries the utterance audio.
    tts_finalize(backends).await;
    open_tts_quiet_window(backends);
}

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
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
    let text = parsed
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| body.trim().to_string());
    Ok(text)
}

async fn llm_chat_streaming(
    backends: &Backends,
    wake: &WakeMachine,
    user_text: &str,
    sentence_tx: mpsc::Sender<String>,
) -> Result<String> {
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

    // Byte-level accumulator (chunks may split mid-line); `\n` (0x0A)
    // never appears mid-UTF-8 codepoint, so splitting there is always a
    // valid string boundary. Hard cap so a malformed upstream that never
    // emits `\n` can't grow unboundedly — ollama emits one ndjson line
    // per delta token, so real replies stay far under it.
    const LLM_STREAM_BUF_MAX: usize = 1 << 20;
    let mut byte_buf: Vec<u8> = Vec::new();
    let mut sentence_buf = String::new();
    let mut full_reply = String::new();
    let mut first_chunk_logged = false;
    let mut first_sentence_logged = false;
    // Pending `<...` span (including the `<`), buffered until its `>`
    // arrives. A span that closes within ANGLE_SPAN_MAX_CHARS is markup
    // (e.g. gemma's <|think|> reasoning) and is dropped before TTS; one
    // that grows past the cap is NOT markup (lone `<` in maths or an
    // emoticon) and is flushed back as literal text, so an unmatched `<`
    // can never mute the rest of the stream. Persists across deltas /
    // chunks / ndjson lines because a span can straddle them.
    const ANGLE_SPAN_MAX_CHARS: usize = 256;
    let mut angle_buf = String::new();

    'outer: loop {
        let chunk = resp
            .chunk()
            .await
            .context("read /api/chat stream")?;
        let chunk = match chunk {
            Some(c) => c,
            None => break,
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
                let mut cleaned = String::with_capacity(delta.len());
                for ch in delta.chars() {
                    if !angle_buf.is_empty() {
                        angle_buf.push(ch);
                        if ch == '>' {
                            debug!(
                                target: "orch::pipeline",
                                span = %angle_buf,
                                "dropped <...> span before TTS"
                            );
                            angle_buf.clear();
                        } else if angle_buf.chars().count() > ANGLE_SPAN_MAX_CHARS {
                            cleaned.push_str(&angle_buf);
                            angle_buf.clear();
                        }
                    } else if ch == '<' {
                        angle_buf.push(ch);
                    } else {
                        cleaned.push(ch);
                    }
                }
                if !cleaned.is_empty() {
                    sentence_buf.push_str(&cleaned);
                    full_reply.push_str(&cleaned);

                    while let Some(end) = find_sentence_end(&sentence_buf) {
                        let remainder = sentence_buf.split_off(end);
                        let sentence = std::mem::replace(&mut sentence_buf, remainder)
                            .trim()
                            .to_string();
                        if sentence.is_empty() {
                            continue;
                        }
                        if !wake.pipeline_still_active() {
                            // Drop response → connection closes → ollama
                            // stops generating.
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
                            return Ok(full_reply);
                        }
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

    if !angle_buf.is_empty() {
        sentence_buf.push_str(&angle_buf);
        full_reply.push_str(&angle_buf);
        angle_buf.clear();
    }
    let tail = sentence_buf.trim().to_string();
    if !tail.is_empty() && wake.pipeline_still_active() {
        let _ = sentence_tx.send(tail).await;
    }
    Ok(full_reply.trim().to_string())
}

/// Policy: `。 ！ ？ 、 … \n` terminate unconditionally (`、` is safe —
/// never mid-numeric, and VOICEVOX renders a clean prosodic pause there).
/// ASCII `. ! ?` terminate *only when followed by whitespace*: without an
/// ASCII rule pure-English replies never stream (the whole turn waits for
/// the trailing flush), while the whitespace gate keeps "1.5" and
/// "api.example.com" intact. ASCII `,` deliberately stays out — English
/// clausal commas would become wrong-feeling TTS breaks. A bare ASCII
/// terminator at end-of-buffer doesn't split (no lookahead char); the
/// end-of-stream flush emits it.
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
        assert_eq!(find_sentence_end("こんにちは。world"), Some("こんにちは。".len()));
        assert_eq!(find_sentence_end("やあ！ next"), Some("やあ！".len()));
        assert_eq!(find_sentence_end("元気？ next"), Some("元気？".len()));
        assert_eq!(find_sentence_end("えーと、それで"), Some("えーと、".len()));
        assert_eq!(find_sentence_end("うーん…続き"), Some("うーん…".len()));
        assert_eq!(find_sentence_end("hi! next"), Some(3));
        assert_eq!(find_sentence_end("hi? next"), Some(3));
        assert_eq!(find_sentence_end("line1\nline2"), Some(6));
        assert_eq!(find_sentence_end("no terminator yet"), None);
        assert_eq!(find_sentence_end("price 1,000 yen"), None);
        assert_eq!(find_sentence_end("a, b, and c"), None);
        assert_eq!(find_sentence_end(""), None);
    }

    #[test]
    fn find_sentence_end_ascii_period_requires_whitespace_after() {
        assert_eq!(find_sentence_end("Hello. World"), Some(6));
        assert_eq!(find_sentence_end("Done.\nNext"), Some(5));
        assert_eq!(find_sentence_end("about 1.5 meters"), None);
        assert_eq!(find_sentence_end("api.example.com"), None);
        assert_eq!(find_sentence_end("file.txt is here"), None);
        assert_eq!(find_sentence_end("Hi!World"), None);
        assert_eq!(find_sentence_end("Why?Yes"), None);
        assert_eq!(find_sentence_end("Done."), None);
        assert_eq!(find_sentence_end("Done!"), None);
        assert_eq!(find_sentence_end("Done?"), None);
    }

    #[test]
    fn find_sentence_end_returns_first_terminator() {
        let s = "前。後！";
        let end = find_sentence_end(s).unwrap();
        assert_eq!(&s[..end], "前。");
    }

    #[test]
    fn in_tts_quiet_window_respects_deadline() {
        let backends = Backends::new(
            crate::config::AsrConfig::default(),
            crate::config::LlmConfig::default(),
            crate::config::TtsConfig::default(),
            crate::config::ResultSinkConfig::default(),
        )
        .unwrap();

        assert!(!backends.in_tts_quiet_window());

        *backends.tts_quiet_until.lock().unwrap() =
            Some(Instant::now() + Duration::from_millis(50));
        assert!(backends.in_tts_quiet_window());

        *backends.tts_quiet_until.lock().unwrap() =
            Some(Instant::now() - Duration::from_millis(1));
        assert!(!backends.in_tts_quiet_window());

        *backends.tts_quiet_until.lock().unwrap() = None;
        assert!(!backends.in_tts_quiet_window());
    }

    #[test]
    fn open_tts_quiet_window_sets_deadline_only_when_tts_configured() {
        let mut tts = crate::config::TtsConfig::default();
        tts.url = "http://tts/speak".into();
        tts.tail_quiet_ms = 400;
        let backends = Backends::new(
            crate::config::AsrConfig::default(),
            crate::config::LlmConfig::default(),
            tts,
            crate::config::ResultSinkConfig::default(),
        )
        .unwrap();
        open_tts_quiet_window(&backends);
        assert!(backends.in_tts_quiet_window());

        // url empty → no window
        let backends = Backends::new(
            crate::config::AsrConfig::default(),
            crate::config::LlmConfig::default(),
            crate::config::TtsConfig::default(),
            crate::config::ResultSinkConfig::default(),
        )
        .unwrap();
        open_tts_quiet_window(&backends);
        assert!(!backends.in_tts_quiet_window());

        // tail_quiet_ms = 0 → no window
        let mut tts = crate::config::TtsConfig::default();
        tts.url = "http://tts/speak".into();
        tts.tail_quiet_ms = 0;
        let backends = Backends::new(
            crate::config::AsrConfig::default(),
            crate::config::LlmConfig::default(),
            tts,
            crate::config::ResultSinkConfig::default(),
        )
        .unwrap();
        open_tts_quiet_window(&backends);
        assert!(!backends.in_tts_quiet_window());
    }

    #[test]
    fn in_tts_quiet_window_boundary_exactly_now_is_not_active() {
        let backends = Backends::new(
            crate::config::AsrConfig::default(),
            crate::config::LlmConfig::default(),
            crate::config::TtsConfig::default(),
            crate::config::ResultSinkConfig::default(),
        )
        .unwrap();
        let now = Instant::now();
        *backends.tts_quiet_until.lock().unwrap() = Some(now);
        assert!(!backends.in_tts_quiet_window());
    }

    use crate::config::{AsrConfig, LlmConfig, ResultSinkConfig, TtsConfig, WakeConfig};
    use crate::state::WakeMachine;
    use std::sync::Arc as StdArc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn spawn_mock_llm(body: Vec<u8>) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
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

    fn backends_with_llm_addr(addr: std::net::SocketAddr) -> StdArc<Backends> {
        let asr = AsrConfig::default();
        let mut llm = LlmConfig::default();
        llm.url = format!("http://{addr}/api/chat");
        llm.timeout_ms = 5_000;
        let tts = TtsConfig::default();
        let result_sink = ResultSinkConfig::default();
        StdArc::new(Backends::new(asr, llm, tts, result_sink).unwrap())
    }

    fn loose_wake() -> StdArc<WakeMachine> {
        let mut cfg = WakeConfig::default();
        cfg.required = false;
        StdArc::new(WakeMachine::new(&cfg))
    }

    #[tokio::test]
    async fn llm_chat_streaming_drains_japanese_sentences_in_order() {
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
        // arrive as one sentence, not two prosodic units.
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
        assert_eq!(
            sentences,
            vec!["値段は1,000円で、", "サイズは1.5メートルです。"]
        );
    }

    #[tokio::test]
    async fn llm_chat_streaming_skips_unparseable_lines() {
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

    #[tokio::test]
    async fn llm_chat_streaming_drops_angle_span_but_keeps_unclosed_tail() {
        let body = concat!(
            r#"{"message":{"content":"<|th"}}"#, "\n",
            r#"{"message":{"content":"ink|>今日は晴れ。3<5 だよ。"}}"#, "\n",
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
        assert_eq!(sentences, vec!["今日は晴れ。", "3<5 だよ。"]);
        assert_eq!(reply, "今日は晴れ。3<5 だよ。");
    }

    #[tokio::test]
    async fn llm_chat_streaming_reemits_overlong_angle_span_verbatim() {
        let text = format!("注意{}{}終わり。", "<", "あ".repeat(300));
        let body = format!(
            "{}\n{}\n",
            serde_json::json!({"message": {"content": text}}),
            r#"{"done":true}"#,
        ).into_bytes();
        let addr = spawn_mock_llm(body).await;
        let backends = backends_with_llm_addr(addr);
        let wake = loose_wake();
        let (tx, mut rx) = mpsc::channel::<String>(8);
        let reply = llm_chat_streaming(&backends, &wake, "test", tx).await.unwrap();
        let mut sentences = vec![];
        while let Some(s) = rx.recv().await {
            sentences.push(s);
        }
        assert_eq!(reply, text);
        assert_eq!(sentences.concat(), text);
    }
}
