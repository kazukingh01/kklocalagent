use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use futures_util::StreamExt;
use serde::Serialize;
use tokio::sync::Semaphore;
use tokio_tungstenite::tungstenite::protocol::Message;
use tracing::{info, warn};
use webrtc_vad::{SampleRate, Vad, VadMode};

use crate::config::{Config, DiagConfig, SinkMode};
use crate::detector::{Event, SpeechFsm};

/// Accumulates RMS energy and speech_ratio over a window of frames and logs
/// a summary line each time the window fills. Enabled only in debug mode so
/// normal operation is quiet.
struct Diag {
    enabled: bool,
    window_frames: u32,
    frames: u32,
    voiced: u32,
    /// Running sum of per-frame mean-square values across the current window.
    /// Window RMS = sqrt(this / frames).
    sum_of_frame_mean_sq: u64,
}

impl Diag {
    fn new(cfg: &DiagConfig) -> Self {
        Self {
            enabled: cfg.enabled,
            window_frames: cfg.window_frames.max(1),
            frames: 0,
            voiced: 0,
            sum_of_frame_mean_sq: 0,
        }
    }

    fn record(&mut self, samples: &[i16], is_speech: bool) {
        if !self.enabled {
            return;
        }
        // Mean square of this frame — summing squares of i16 into u64 is safe
        // for any reasonable frame length (320 * 32768^2 ≈ 3.4e11 ≪ u64 max).
        let sumsq: u64 = samples
            .iter()
            .map(|s| {
                let v = *s as i64;
                (v * v) as u64
            })
            .sum();
        self.sum_of_frame_mean_sq += sumsq / samples.len() as u64;
        self.frames += 1;
        if is_speech {
            self.voiced += 1;
        }
        if self.frames >= self.window_frames {
            let mean_sq = self.sum_of_frame_mean_sq as f64 / self.frames as f64;
            let rms = mean_sq.sqrt() as u32;
            let ratio = self.voiced as f32 / self.frames as f32;
            info!(
                target: "vad::diag",
                "[diag] rms={rms} speech_ratio={ratio:.2} ({}/{} voiced)",
                self.voiced,
                self.frames,
            );
            self.frames = 0;
            self.voiced = 0;
            self.sum_of_frame_mean_sq = 0;
        }
    }
}

pub async fn run(config: Config) -> Result<()> {
    let cfg = Arc::new(config);
    let http = Arc::new(
        reqwest::Client::builder()
            .timeout(Duration::from_millis(cfg.sink.asr_timeout_ms))
            .build()
            .context("build reqwest client")?,
    );
    // Bounds the number of in-flight asr-direct POSTs. Excess utterances
    // are dropped with a warning rather than queued so backpressure is
    // visible in logs instead of presenting as silent timeouts. Survives
    // WS reconnects so we don't lose backpressure state on a flap.
    let asr_inflight = Arc::new(Semaphore::new(cfg.sink.asr_max_inflight as usize));
    let mut shutdown = Box::pin(shutdown_signal());
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => {
                info!("shutdown signal received");
                return Ok(());
            }
            result = connect_and_run(cfg.clone(), http.clone(), asr_inflight.clone()) => {
                match result {
                    Ok(()) => info!("WS session ended cleanly; reconnecting"),
                    Err(e) => warn!(error = %e, "WS session error; reconnecting"),
                }
            }
        }
        tokio::select! {
            biased;
            _ = &mut shutdown => {
                info!("shutdown signal received");
                return Ok(());
            }
            _ = tokio::time::sleep(Duration::from_millis(cfg.source.reconnect_ms)) => {}
        }
    }
}

async fn connect_and_run(
    cfg: Arc<Config>,
    http: Arc<reqwest::Client>,
    asr_inflight: Arc<Semaphore>,
) -> Result<()> {
    info!(url = %cfg.source.mic_url, "connecting to audio-io /mic");
    let (ws, _) = tokio_tungstenite::connect_async(&cfg.source.mic_url)
        .await
        .context("WS connect")?;
    info!("connected; starting VAD loop");

    let mut vad = make_vad(cfg.detector.aggressiveness, cfg.detector.sample_rate)?;
    let mut fsm = SpeechFsm::new(
        cfg.detector.start_frames,
        cfg.detector.hang_frames,
        cfg.detector.max_utterance_frames,
    );
    let mut diag = Diag::new(&cfg.diag);

    let bytes_per_frame = cfg.detector.bytes_per_frame();
    let samples_per_frame = cfg.detector.samples_per_frame();
    let sr = cfg.detector.sample_rate;
    let mode = cfg.sink.mode;
    let log_audio = cfg.sink.log_audio_in_event;
    let asr_url = cfg.sink.asr_url.clone();

    // TODO: if audio-io enables WS ping/pong keepalive, the server will drop
    // us because tokio-tungstenite doesn't auto-respond to Pings — the write
    // half has to echo them. Today we rely on the reconnect loop to recover.
    let (_write, mut read) = ws.split();
    // Carry-over buffer: audio-io sends 20 ms frames but the WebSocket layer
    // may merge or split them, so we re-slice at bytes_per_frame boundaries.
    let mut scratch: Vec<u8> = Vec::with_capacity(bytes_per_frame * 4);
    let mut samples: Vec<i16> = Vec::with_capacity(samples_per_frame);

    while let Some(msg) = read.next().await {
        let msg = msg.context("ws read")?;
        let payload = match msg {
            Message::Binary(b) => b,
            Message::Close(_) => {
                info!("peer closed WS");
                return Ok(());
            }
            _ => continue,
        };
        scratch.extend_from_slice(&payload);
        while scratch.len() >= bytes_per_frame {
            let frame: Vec<u8> = scratch.drain(..bytes_per_frame).collect();
            samples.clear();
            samples.extend(
                frame
                    .chunks_exact(2)
                    .map(|p| i16::from_le_bytes([p[0], p[1]])),
            );
            let is_speech = vad
                .is_voice_segment(&samples)
                .map_err(|_| anyhow!("vad classify: frame length mismatch"))?;
            diag.record(&samples, is_speech);
            if let Some(event) = fsm.push_frame(&frame, is_speech) {
                handle_event(
                    &event,
                    fsm.utterance_buffer(),
                    sr,
                    mode,
                    log_audio,
                    &http,
                    &asr_url,
                    &asr_inflight,
                );
            }
        }
    }
    Ok(())
}

fn make_vad(aggressiveness: u8, sample_rate: u32) -> Result<Vad> {
    let rate = match sample_rate {
        8000 => SampleRate::Rate8kHz,
        16000 => SampleRate::Rate16kHz,
        32000 => SampleRate::Rate32kHz,
        48000 => SampleRate::Rate48kHz,
        other => anyhow::bail!("unsupported sample rate: {other}"),
    };
    let mode = match aggressiveness {
        0 => VadMode::Quality,
        1 => VadMode::LowBitrate,
        2 => VadMode::Aggressive,
        3 => VadMode::VeryAggressive,
        other => anyhow::bail!("aggressiveness must be 0..=3, got {other}"),
    };
    Ok(Vad::new_with_rate_and_mode(rate, mode))
}

#[derive(Serialize)]
struct EventEnvelope<'a> {
    #[serde(flatten)]
    event: &'a Event,
    ts: f64,
    sample_rate: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_base64: Option<String>,
}

fn handle_event(
    event: &Event,
    utterance: &[u8],
    sample_rate: u32,
    mode: SinkMode,
    log_audio: bool,
    http: &Arc<reqwest::Client>,
    asr_url: &str,
    asr_inflight: &Arc<Semaphore>,
) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let audio_base64 = if log_audio && matches!(event, Event::SpeechEnded { .. }) {
        Some(base64::engine::general_purpose::STANDARD.encode(utterance))
    } else {
        None
    };
    let env = EventEnvelope {
        event,
        ts,
        sample_rate,
        audio_base64,
    };
    let json = serde_json::to_string(&env).unwrap_or_else(|_| "<serialize error>".into());

    match mode {
        SinkMode::DryRun => {
            // Dry-run sink: pretend to POST to the orchestrator and just log.
            info!(target: "vad::sink", "[orchestrator-stub <-] {json}");
        }
        SinkMode::Orchestrator => {
            // TODO: real HTTP POST to cfg.sink.orchestrator_url once the
            // orchestrator crate exists.
            warn!(target: "vad::sink", "orchestrator sink not yet implemented; event dropped: {json}");
        }
        SinkMode::AsrDirect => {
            info!(target: "vad::sink", "[event] {json}");
            if let Event::SpeechEnded { .. } = event {
                // try_acquire_owned returns Err iff every permit is held —
                // i.e. asr_max_inflight POSTs are already in flight. Drop
                // this utterance with a warning rather than queuing so the
                // operator sees backpressure instead of silent timeouts.
                let permit = match asr_inflight.clone().try_acquire_owned() {
                    Ok(p) => p,
                    Err(_) => {
                        warn!(
                            target: "vad::asr",
                            "ASR busy (asr_max_inflight reached); dropping utterance",
                        );
                        return;
                    }
                };
                let wav = wav_from_pcm_s16le_mono(utterance, sample_rate);
                let client = http.clone();
                let url = asr_url.to_string();
                tokio::spawn(async move {
                    let _permit = permit; // released on drop after the POST
                    match post_wav_to_asr(&client, &url, wav).await {
                        Ok(text) => info!(
                            target: "vad::asr",
                            "[asr <-] transcription: {text:?}"
                        ),
                        Err(e) => warn!(target: "vad::asr", "ASR call failed: {e:#}"),
                    }
                });
            }
        }
    }
}

/// Wrap a raw little-endian 16-bit mono PCM buffer in a 44-byte WAV header.
/// whisper.cpp's `/inference` endpoint accepts WAV uploads via multipart.
fn wav_from_pcm_s16le_mono(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let chunk_size = 36 + data_len;
    let byte_rate = sample_rate * 2; // mono * 2 bytes
    let block_align: u16 = 2;
    let bits_per_sample: u16 = 16;

    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&1u16.to_le_bytes()); // channels
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

async fn post_wav_to_asr(
    client: &reqwest::Client,
    url: &str,
    wav: Vec<u8>,
) -> Result<String> {
    let part = reqwest::multipart::Part::bytes(wav)
        .file_name("utterance.wav")
        .mime_str("audio/wav")
        .context("set wav mime")?;
    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("response_format", "json")
        .text("temperature", "0");
    let resp = client
        .post(url)
        .multipart(form)
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

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut sig) = signal(SignalKind::terminate()) {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::Event;
    use serde_json::Value;

    fn parse(json: &str) -> Value {
        serde_json::from_str(json).unwrap()
    }

    fn envelope_json(event: &Event, sample_rate: u32, audio_base64: Option<String>) -> String {
        let env = EventEnvelope {
            event,
            ts: 1_744_284_000.5,
            sample_rate,
            audio_base64,
        };
        serde_json::to_string(&env).unwrap()
    }

    #[test]
    fn speech_started_wire_format() {
        let ev = Event::SpeechStarted { frame_index: 123 };
        let v = parse(&envelope_json(&ev, 16000, None));
        assert_eq!(v["name"], "SpeechStarted");
        assert_eq!(v["frame_index"], 123);
        assert_eq!(v["sample_rate"], 16000);
        assert!(v["ts"].is_number());
        assert!(v.get("audio_base64").is_none(), "audio_base64 must be omitted when None");
    }

    #[test]
    fn speech_ended_wire_format_without_audio() {
        let ev = Event::SpeechEnded {
            frame_index: 167,
            duration_frames: 45,
            audio_len_bytes: 28_800,
        };
        let v = parse(&envelope_json(&ev, 16000, None));
        assert_eq!(v["name"], "SpeechEnded");
        assert_eq!(v["frame_index"], 167);
        assert_eq!(v["duration_frames"], 45);
        assert_eq!(v["audio_len_bytes"], 28_800);
        assert!(v.get("audio_base64").is_none());
    }

    #[test]
    fn speech_ended_includes_audio_base64_when_set() {
        let ev = Event::SpeechEnded {
            frame_index: 10,
            duration_frames: 5,
            audio_len_bytes: 4,
        };
        let v = parse(&envelope_json(&ev, 16000, Some("AAAA".into())));
        assert_eq!(v["audio_base64"], "AAAA");
    }

    #[test]
    fn wav_header_layout() {
        let pcm = vec![0u8; 6400]; // 200 ms @ 16 kHz s16 mono
        let wav = wav_from_pcm_s16le_mono(&pcm, 16000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        // fmt chunk size = 16
        assert_eq!(u32::from_le_bytes(wav[16..20].try_into().unwrap()), 16);
        // PCM format
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1);
        // mono
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        // sample rate
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        // byte rate = sample_rate * channels * bytes_per_sample
        assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 32_000);
        // block align
        assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 2);
        // bits per sample
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(
            u32::from_le_bytes(wav[40..44].try_into().unwrap()),
            pcm.len() as u32
        );
        assert_eq!(wav.len(), 44 + pcm.len());
    }
}
