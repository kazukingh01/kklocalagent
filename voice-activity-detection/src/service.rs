use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use futures_util::StreamExt;
use serde::Serialize;
use tokio::sync::Semaphore;
use tokio_tungstenite::tungstenite::protocol::Message;
use tracing::{debug, info, warn};
use webrtc_vad::{SampleRate, Vad, VadMode};

use crate::config::{Config, DiagConfig, SinkMode};
use crate::detector::{Event, SpeechFsm};
use wav_utils::wav_from_pcm_s16le_mono;

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
            // Debug-level: a per-second summary line is too chatty
            // for production INFO. Operators investigating false
            // triggers can re-enable it via
            // `RUST_LOG=info,vad::diag=debug`.
            debug!(
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
        // Per-request timeouts (asr_timeout_ms / orchestrator_timeout_ms)
        // are applied at each POST builder so the two stages have
        // independent budgets.
        reqwest::Client::builder()
            .build()
            .context("build reqwest client")?,
    );
    // Bounds the number of in-flight asr-direct POSTs. Excess utterances
    // are dropped with a warning rather than queued so backpressure is
    // visible in logs instead of presenting as silent timeouts. Survives
    // WS reconnects so we don't lose backpressure state on a flap.
    let asr_inflight = Arc::new(Semaphore::new(cfg.sink.asr_max_inflight as usize));
    let orchestrator_inflight =
        Arc::new(Semaphore::new(cfg.sink.orchestrator_max_inflight as usize));
    let mut shutdown = Box::pin(shutdown_signal());
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => {
                info!("shutdown signal received");
                return Ok(());
            }
            result = connect_and_run(
                cfg.clone(),
                http.clone(),
                asr_inflight.clone(),
                orchestrator_inflight.clone(),
            ) => {
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
    orchestrator_inflight: Arc<Semaphore>,
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
    // Optional denoiser. Built once and held across frames so the
    // RNNoise state and rubato FFT plans amortise across the whole
    // session. None when `detector.denoise = false` — the audio
    // path stays byte-identical to the previous build in that case.
    let mut denoiser = if cfg.detector.denoise {
        if cfg.detector.sample_rate != 16_000 {
            anyhow::bail!(
                "detector.denoise requires sample_rate=16000 (got {})",
                cfg.detector.sample_rate
            );
        }
        info!("denoise: nnnoiseless (RNNoise) enabled, frame={}", samples_per_frame);
        Some(crate::denoise::Denoiser::new(samples_per_frame).context("init denoiser")?)
    } else {
        None
    };
    let sr = cfg.detector.sample_rate;
    let mode = cfg.sink.mode;
    let log_audio = cfg.sink.log_audio_in_event;
    let asr_url = cfg.sink.asr_url.clone();
    let asr_timeout_ms = cfg.sink.asr_timeout_ms;
    let orchestrator_url = cfg.sink.orchestrator_url.clone();
    let orchestrator_timeout_ms = cfg.sink.orchestrator_timeout_ms;

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
            let mut frame: Vec<u8> = scratch.drain(..bytes_per_frame).collect();
            samples.clear();
            samples.extend(
                frame
                    .chunks_exact(2)
                    .map(|p| i16::from_le_bytes([p[0], p[1]])),
            );
            // RNNoise pre-stage. Mutates `samples` in place; we then
            // re-encode the cleaned PCM back into `frame` so the
            // utterance buffer that goes to ASR (via fsm.push_frame
            // → fsm.utterance_buffer()) carries the denoised audio
            // too — Whisper sees a less ambiguous signal, fewer
            // hallucinations on near-silence.
            if let Some(d) = denoiser.as_mut() {
                d.process(&mut samples).context("denoise frame")?;
                for (i, s) in samples.iter().enumerate() {
                    let bytes = s.to_le_bytes();
                    frame[i * 2] = bytes[0];
                    frame[i * 2 + 1] = bytes[1];
                }
            }
            let is_speech = vad
                .is_voice_segment(&samples)
                .map_err(|_| anyhow!("vad classify: frame length mismatch"))?;
            diag.record(&samples, is_speech);
            if let Some(event) = fsm.push_frame(&frame, is_speech) {
                // RMS-energy gate on SpeechEnded: drop utterances
                // whose buffered audio is below `min_utterance_rms_dbfs`
                // (negative dBFS, threshold disabled when >= 0). The
                // typical hallucination trigger is "VAD spuriously
                // armed on near-silence + sent the empty buffer to
                // ASR + Whisper filled the void with `(拍手)` /
                // `ご視聴ありがとうございました`". Gating below e.g.
                // -45 dBFS catches that without rejecting legitimate
                // quiet speech (real voice is typically -10..-25).
                if cfg.detector.min_utterance_rms_dbfs < 0.0 {
                    if let Event::SpeechEnded { .. } = &event {
                        let buf = fsm.utterance_buffer();
                        let r = rms_dbfs_i16le(buf);
                        if r < cfg.detector.min_utterance_rms_dbfs {
                            warn!(
                                target: "vad::sink",
                                event = "SpeechEnded",
                                reason = "below_rms_gate",
                                rms_dbfs = r,
                                threshold = cfg.detector.min_utterance_rms_dbfs,
                                bytes = buf.len(),
                                "dropped utterance: too quiet (likely ambient noise → would trigger Whisper hallucination)"
                            );
                            continue;
                        }
                    }
                }
                handle_event(
                    &event,
                    fsm.utterance_buffer(),
                    sr,
                    mode,
                    log_audio,
                    &http,
                    &asr_url,
                    asr_timeout_ms,
                    &asr_inflight,
                    &orchestrator_url,
                    orchestrator_timeout_ms,
                    &orchestrator_inflight,
                );
            }
        }
    }
    Ok(())
}

/// Compute RMS dBFS of a little-endian s16 PCM buffer. 0 dBFS =
/// full-scale i16; real voice is typically -10..-25, room ambient
/// is -50..-60. Returns `f32::NEG_INFINITY` on empty input or true
/// digital silence. ~1 µs per 320-sample (20 ms) frame so calling
/// it once per SpeechEnded is free.
fn rms_dbfs_i16le(buf: &[u8]) -> f32 {
    let n = buf.len() / 2;
    if n == 0 {
        return f32::NEG_INFINITY;
    }
    let mut sum_sq: f64 = 0.0;
    for chunk in buf.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f64 / 32768.0;
        sum_sq += s * s;
    }
    let mean_sq = sum_sq / n as f64;
    if mean_sq <= 0.0 {
        return f32::NEG_INFINITY;
    }
    20.0 * (mean_sq.sqrt() as f32).log10()
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

#[allow(clippy::too_many_arguments)]
fn handle_event(
    event: &Event,
    utterance: &[u8],
    sample_rate: u32,
    mode: SinkMode,
    log_audio: bool,
    http: &Arc<reqwest::Client>,
    asr_url: &str,
    asr_timeout_ms: u64,
    asr_inflight: &Arc<Semaphore>,
    orchestrator_url: &str,
    orchestrator_timeout_ms: u64,
    orchestrator_inflight: &Arc<Semaphore>,
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
            info!(target: "vad::sink", "[orchestrator <-] {} bytes",
                  json.len());
            // try_acquire_owned drops events when orchestrator_max_inflight
            // POSTs are already in flight — same backpressure pattern as
            // asr-direct, scoped to a separate semaphore so a slow ASR
            // doesn't starve VAD-event delivery.
            let permit = match orchestrator_inflight.clone().try_acquire_owned() {
                Ok(p) => p,
                Err(_) => {
                    warn!(
                        target: "vad::sink",
                        "orchestrator busy (orchestrator_max_inflight reached); dropping event",
                    );
                    return;
                }
            };
            let client = http.clone();
            let url = orchestrator_url.to_string();
            let body = json.clone();
            let timeout_ms = orchestrator_timeout_ms;
            tokio::spawn(async move {
                let _permit = permit;
                match post_json_to_orchestrator(&client, &url, body, timeout_ms).await {
                    Ok(()) => {}
                    Err(e) => warn!(target: "vad::sink", "orchestrator POST failed: {e:#}"),
                }
            });
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
                let timeout_ms = asr_timeout_ms;
                tokio::spawn(async move {
                    let _permit = permit; // released on drop after the POST
                    match post_wav_to_asr(&client, &url, wav, timeout_ms).await {
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

async fn post_wav_to_asr(
    client: &reqwest::Client,
    url: &str,
    wav: Vec<u8>,
    timeout_ms: u64,
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
        .timeout(Duration::from_millis(timeout_ms))
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

async fn post_json_to_orchestrator(
    client: &reqwest::Client,
    url: &str,
    body: String,
    timeout_ms: u64,
) -> Result<()> {
    let resp = client
        .post(url)
        .header("content-type", "application/json")
        .body(body)
        .timeout(Duration::from_millis(timeout_ms))
        .send()
        .await
        .context("POST /events")?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!(
            "orchestrator responded {status}: {}",
            text.chars().take(200).collect::<String>()
        );
    }
    Ok(())
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
    fn rms_dbfs_silence_is_neg_infinity() {
        let zeros = vec![0u8; 320 * 2];
        assert_eq!(rms_dbfs_i16le(&zeros), f32::NEG_INFINITY);
    }

    #[test]
    fn rms_dbfs_full_scale_sine_is_near_minus_three() {
        // A sine that swings to ±i16::MAX has RMS = peak / sqrt(2),
        // which in dBFS is -3.01. 320 samples at 16 kHz = 20 ms; one
        // 440 Hz cycle is ~36 samples so the buffer averages cleanly
        // to the analytical RMS.
        let mut buf = Vec::with_capacity(320 * 2);
        for i in 0..320 {
            let s = ((i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16_000.0).sin()
                * (i16::MAX as f32)) as i16;
            buf.extend_from_slice(&s.to_le_bytes());
        }
        let r = rms_dbfs_i16le(&buf);
        assert!(
            (r - (-3.01)).abs() < 0.5,
            "rms={r} expected ~-3.01 dBFS for full-scale sine"
        );
    }

    #[test]
    fn rms_dbfs_quiet_noise_below_minus_forty() {
        // ±100 LSB pseudo-random "ambient noise". -100/32768 ≈ -50 dBFS.
        let mut buf = Vec::with_capacity(320 * 2);
        for i in 0..320 {
            let s = (((i * 73) % 200) as i16) - 100;
            buf.extend_from_slice(&s.to_le_bytes());
        }
        let r = rms_dbfs_i16le(&buf);
        assert!(r < -40.0 && r > -60.0, "rms={r} expected -40..-60 dBFS");
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

    // wav_header_layout — moved to `wav-utils/src/lib.rs::tests` along
    // with the function it covered. No need for a duplicate assertion
    // here now that VAD imports the shared crate.
}
