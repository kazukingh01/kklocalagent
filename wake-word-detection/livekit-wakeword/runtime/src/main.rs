//! livekit-wakeword runtime — drop-in replacement for the openwakeword
//! Python shim. Same wire contract: subscribe to audio-io's `/mic`
//! WebSocket, run wake-word inference, POST `WakeWordDetected` to the
//! orchestrator's `/events`, expose `/health` for compose's
//! `service_healthy` gate.
//!
//! Pipeline:
//!     ws_client ──pcm──▶ detector ──detection──▶ event_sink
//!                                  ▲
//!                                  └─ ring buffer + WakeWordModel::predict
//!
//! Each stage is a Tokio task; mpsc channels carry frames between them.
//! `health_server` reads two `AtomicBool`s reflecting model-load /
//! ws-connected state without coupling to the data path.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use tokio::signal;
use tokio::sync::mpsc;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

mod config;
mod detector;
mod event_sink;
mod health;
mod wakeword;
mod ws_client;

#[derive(Debug, Clone)]
pub struct Detection {
    pub model: String,
    pub score: f32,
    pub ts: f64,
}

/// One PCM frame received from audio-io's `/mic?ts=1` WebSocket. The
/// 8-byte header carries the wall-clock time of the frame's *last*
/// sample (epoch ns). Held alongside the samples so the detector can
/// compute end-to-end lag against `SystemTime::now()` without relying
/// on chunk-arrival time (which hides intra-audio-io / broadcast / NW
/// delay).
#[derive(Debug, Clone)]
pub struct MicFrame {
    pub end_epoch_ns: u64,
    pub samples: Vec<i16>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,livekit_wakeword_runtime=info")),
        )
        .with_target(true)
        .init();

    let cfg = config::Config::from_env()?;
    info!(?cfg, "starting livekit-wakeword runtime");

    let model_loaded = Arc::new(AtomicBool::new(false));
    let ws_connected = Arc::new(AtomicBool::new(false));

    // ws_client → detector: 32 frames ≈ 640 ms back-pressure tolerance
    // before audio-io's broadcast channel starts dropping (it's sized
    // ~1.28 s upstream).
    let (pcm_tx, pcm_rx) = mpsc::channel::<MicFrame>(32);
    // detector → event_sink: cooldown caps the rate at <1 detection /
    // 2 s, so 8 is generous.
    let (det_tx, det_rx) = mpsc::channel::<Detection>(8);

    let h_health = tokio::spawn(health::serve(
        cfg.listen_addr,
        health::HealthFlags {
            model_loaded: Arc::clone(&model_loaded),
            ws_connected: Arc::clone(&ws_connected),
        },
    ));
    let h_event = tokio::spawn(event_sink::run(cfg.clone(), det_rx));
    let h_detector = tokio::spawn(detector::run(
        cfg.clone(),
        pcm_rx,
        det_tx,
        Arc::clone(&model_loaded),
    ));
    let h_ws = tokio::spawn(ws_client::run(
        cfg.mic_url.clone(),
        pcm_tx,
        Arc::clone(&ws_connected),
    ));

    let shutdown = async {
        let ctrl_c = signal::ctrl_c();
        #[cfg(unix)]
        let term = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler")
                .recv()
                .await;
        };
        #[cfg(not(unix))]
        let term = std::future::pending::<()>();
        tokio::select! {
            _ = ctrl_c => info!("ctrl-c received, shutting down"),
            _ = term => info!("sigterm received, shutting down"),
        }
    };

    tokio::select! {
        _ = shutdown => {}
        r = h_ws => log_task_exit("ws_client", r),
        r = h_detector => log_task_exit("detector", r),
        r = h_event => log_task_exit("event_sink", r),
        r = h_health => log_task_exit("health", r),
    }

    Ok(())
}

fn log_task_exit(name: &str, r: Result<Result<()>, tokio::task::JoinError>) {
    match r {
        Ok(Ok(())) => info!("{name} task exited"),
        Ok(Err(e)) => error!("{name} task error: {e:?}"),
        Err(e) => error!("{name} task panicked: {e:?}"),
    }
}
