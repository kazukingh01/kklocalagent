//! SpeechEnded → ASR → LLM pipeline.
//!
//! Shape of one turn, per #4 §5:
//!
//! ```text
//! orchestrator ──POST /inference──► ASR (whisper.cpp)
//!              ──POST /api/chat───► LLM (ollama)
//!              (logs the assistant response)
//! ```
//!
//! Per-stage concurrency is bounded by a `Semaphore` so the orchestrator
//! degrades predictably (drop with warning) instead of queuing unboundedly
//! if VAD fires faster than the backends can keep up.

use std::sync::Arc;

use anyhow::{Context, Result};
use base64::Engine;
use reqwest::multipart;
use serde::Serialize;
use serde_json::json;
use tokio::sync::Semaphore;
use tracing::{info, warn};

use crate::config::{AsrConfig, LlmConfig, ResultSinkConfig, TtsConfig};

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

/// Run one full SpeechEnded → ASR → LLM turn. Logs the assistant reply
/// on success; logs and swallows errors at each stage so one bad
/// utterance can't bring the service down.
pub async fn run_turn(backends: Arc<Backends>, pcm: Vec<u8>, sample_rate: u32) {
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

    if text.is_empty() {
        info!(target: "orch::pipeline", "ASR returned empty text; skipping LLM");
        return;
    }
    info!(target: "orch::pipeline", text = %text, "transcribed");

    // LLM stage
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

    let reply = match llm_chat(&backends, &text).await {
        Ok(r) => r,
        Err(e) => {
            warn!(target: "orch::pipeline", "LLM failed: {e:#}");
            drop(llm_permit);
            return;
        }
    };
    drop(llm_permit);

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

    // TTS stage. Disabled when tts.url is empty — degrades to "log
    // only" so dev runs without `tts-streamer` still complete the turn.
    // Empty replies (rare; would mean LLM said literally nothing) skip
    // TTS rather than POST an empty string the streamer would reject.
    if !backends.tts.url.is_empty() && !reply.is_empty() {
        tts_speak(&backends, &reply).await;
    }
}

async fn tts_speak(backends: &Backends, text: &str) {
    let permit = match backends.tts_inflight.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            warn!(
                target: "orch::pipeline",
                "TTS at capacity ({} in flight); skipping speak",
                backends.tts.max_inflight
            );
            return;
        }
    };

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
    drop(permit);
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

async fn llm_chat(backends: &Backends, user_text: &str) -> Result<String> {
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
        stream: false,
    };
    let resp = backends
        .http
        .post(&backends.llm.url)
        .json(&body)
        .timeout(std::time::Duration::from_millis(backends.llm.timeout_ms))
        .send()
        .await
        .context("POST /api/chat")?;
    let status = resp.status();
    let text = resp.text().await.context("read /api/chat body")?;
    if !status.is_success() {
        anyhow::bail!("LLM responded {status}: {text}");
    }
    // ollama /api/chat with stream=false returns one JSON object:
    //   {"model":"...","message":{"role":"assistant","content":"..."},
    //    "done":true,...}
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse /api/chat response: {text}"))?;
    let content = parsed
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    Ok(content)
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
}
