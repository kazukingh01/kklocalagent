use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use futures_util::StreamExt;
use serde::Serialize;
use tokio_tungstenite::tungstenite::protocol::Message;
use tracing::{info, warn};
use webrtc_vad::{SampleRate, Vad, VadMode};

use crate::config::Config;
use crate::detector::{Event, SpeechFsm};

pub async fn run(config: Config) -> Result<()> {
    let cfg = Arc::new(config);
    let mut shutdown = Box::pin(shutdown_signal());
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => {
                info!("shutdown signal received");
                return Ok(());
            }
            result = connect_and_run(cfg.clone()) => {
                match result {
                    Ok(()) => info!("WS session ended cleanly; reconnecting"),
                    Err(e) => warn!(error = %e, "WS session error; reconnecting"),
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(cfg.source.reconnect_ms)).await;
    }
}

async fn connect_and_run(cfg: Arc<Config>) -> Result<()> {
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

    let bytes_per_frame = cfg.detector.bytes_per_frame();
    let samples_per_frame = cfg.detector.samples_per_frame();
    let sr = cfg.detector.sample_rate;
    let dry_run = cfg.sink.dry_run;
    let include_audio = cfg.sink.include_audio_in_event;

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
                .map_err(|e| anyhow!("vad classify: {e:?}"))?;
            if let Some(event) = fsm.push_frame(&frame, is_speech) {
                emit_event(&event, &fsm, sr, dry_run, include_audio);
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

fn emit_event(
    event: &Event,
    fsm: &SpeechFsm,
    sample_rate: u32,
    dry_run: bool,
    include_audio: bool,
) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let audio_base64 = if include_audio && matches!(event, Event::SpeechEnded { .. }) {
        Some(base64::engine::general_purpose::STANDARD.encode(fsm.utterance_buffer()))
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
    if dry_run {
        // Dry-run sink: pretend to POST to the orchestrator and just log. This
        // is the only sink implemented in v0.1.
        info!(target: "vad::sink", "[orchestrator-stub <-] {json}");
    } else {
        // TODO: real HTTP POST to cfg.sink.orchestrator_url once the
        // orchestrator crate exists.
        warn!(target: "vad::sink", "live sink not yet implemented; event dropped: {json}");
    }
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
