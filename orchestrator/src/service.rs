//! HTTP server: `POST /events` + `GET /health`.
//!
//! `/events` is the single entry point for all upstream producers (VAD,
//! wake-word-detection, future sources). Events are dispatched by `name`
//! on a best-effort basis — unknown names are logged and acknowledged
//! (forward-compat: producers can add new event types before the
//! orchestrator learns about them).

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use tokio::net::TcpListener;
use tracing::{info, warn};

use crate::config::Config;
use crate::events::EventEnvelope;
use crate::pipeline::{self, forward_to_result_sink, Backends};

#[derive(Clone)]
struct AppState {
    backends: Arc<Backends>,
}

pub async fn run(config: Config) -> Result<()> {
    let backends = Arc::new(Backends::new(
        config.asr.clone(),
        config.llm.clone(),
        config.tts.clone(),
        config.result_sink.clone(),
    )?);
    let state = AppState { backends };

    let app = Router::new()
        .route("/health", get(health))
        .route("/events", post(events))
        .with_state(state);

    let addr: std::net::SocketAddr = config
        .server
        .listen
        .parse()
        .with_context(|| format!("parse server.listen {:?}", config.server.listen))?;
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    info!(%addr, "orchestrator listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve")?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    // v0.1: liveness only. We don't probe ASR/LLM here because the
    // orchestrator is designed to *degrade* (log + drop) when a backend
    // is down rather than refuse incoming events.
    (StatusCode::OK, Json(json!({"ok": true})))
}

async fn events(
    State(state): State<AppState>,
    Json(ev): Json<EventEnvelope>,
) -> impl IntoResponse {
    // Log a compact view of the event before dispatching. Never log the
    // full `audio_base64` blob (it's big and the content is in the PCM,
    // not interesting as text).
    info!(
        target: "orch::events",
        name = %ev.name,
        ts = ?ev.ts,
        sample_rate = ?ev.sample_rate,
        audio_bytes = ?ev.audio_base64.as_ref().map(|s| s.len()),
        "received event"
    );

    match ev.name.as_str() {
        "SpeechStarted" => {
            // v0.1: no state machine yet — just log.
            info!(
                target: "orch::events",
                frame_index = ?ev.frame_index,
                "speech started"
            );
        }
        "SpeechEnded" => {
            if !ev.has_utterance_audio() {
                info!(
                    target: "orch::events",
                    "SpeechEnded without audio_base64 — skipping pipeline"
                );
            } else if let Err(e) = dispatch_utterance(&state, &ev) {
                warn!(target: "orch::events", "dispatch failed: {e:#}");
            }
        }
        "WakeWordDetected" => {
            // v0.1 is VAD-triggered (always-listening); wake is logged
            // here so the later state-machine work (#4 §9 step 10) can
            // promote it to Armed without a schema change. The full
            // event is also forwarded to result_sink so external
            // observers (and the integration test) see it.
            info!(
                target: "orch::events",
                model = ?ev.model,
                score = ?ev.score,
                "wake word detected"
            );
            let payload = json!({
                "name": "WakeWordDetected",
                "model": ev.model,
                "score": ev.score,
                "ts": ev.ts,
            });
            let backends = state.backends.clone();
            tokio::spawn(async move {
                forward_to_result_sink(&backends, &payload).await;
            });
        }
        other => {
            info!(target: "orch::events", name = %other, "unhandled event");
        }
    }
    (StatusCode::OK, Json(json!({"ok": true})))
}

fn dispatch_utterance(state: &AppState, ev: &EventEnvelope) -> Result<()> {
    // Defensive: has_utterance_audio() has already established these.
    let b64 = ev
        .audio_base64
        .as_deref()
        .context("audio_base64 missing after has_utterance_audio()")?;
    let sample_rate = ev.sample_rate.context("sample_rate missing")?;
    let pcm = pipeline::decode_audio(b64)?;

    // Run the ASR→LLM turn off the request-response path so the HTTP
    // caller (VAD) isn't blocked for the full transcription+chat
    // latency, and so a backend hang can't starve incoming events.
    let backends = state.backends.clone();
    tokio::spawn(async move {
        pipeline::run_turn(backends, pcm, sample_rate).await;
    });
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
