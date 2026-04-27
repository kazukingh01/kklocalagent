//! HTTP server: `POST /events` + `GET /health`.
//!
//! `/events` is the single entry point for all upstream producers (VAD,
//! wake-word-detection, future sources). Events are dispatched by `name`
//! on a best-effort basis — unknown names are logged and acknowledged
//! (forward-compat: producers can add new event types before the
//! orchestrator learns about them).
//!
//! v1.0 introduces wake-gated dispatch (see `state::WakeMachine`):
//! `SpeechEnded` events run the pipeline only when preceded by a
//! recent `WakeWordDetected`. Set `wake.required = false` to fall
//! back to v0.1 always-listening behaviour.

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
use crate::pipeline::{self, forward_to_result_sink, tts_stop, Backends};
use crate::state::{DispatchOutcome, SpeechStartedOutcome, WakeMachine, WakeResult};

#[derive(Clone)]
struct AppState {
    backends: Arc<Backends>,
    wake: Arc<WakeMachine>,
}

pub async fn run(config: Config) -> Result<()> {
    let backends = Arc::new(Backends::new(
        config.asr.clone(),
        config.llm.clone(),
        config.tts.clone(),
        config.result_sink.clone(),
    )?);
    let wake = Arc::new(WakeMachine::new(&config.wake));
    let state = AppState { backends, wake };

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
    info!(%addr, wake_required = config.wake.required, "orchestrator listening");

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
            // SpeechStarted is the *cancel* trigger for armed-window
            // timers (per v1.0 spec). on_speech_started() handles the
            // state transition (ArmedAfter* → Listening); here we
            // just turn its outcome into a structured log line so
            // the operator can see exactly why each VAD frame was
            // accepted or dropped. Drop logs share `event=` /
            // `reason=` fields with SpeechEnded drops below — a
            // single `grep 'reason='` finds every dropped VAD event.
            match state.wake.on_speech_started() {
                SpeechStartedOutcome::Bypass => {
                    info!(
                        target: "orch::events",
                        frame_index = ?ev.frame_index,
                        "speech started (always-listening; no gate)"
                    );
                }
                SpeechStartedOutcome::Listening => {
                    info!(
                        target: "orch::events",
                        frame_index = ?ev.frame_index,
                        "speech started: listening for SpeechEnded"
                    );
                }
                SpeechStartedOutcome::DroppedIdle => {
                    warn!(
                        target: "orch::events",
                        event = "SpeechStarted",
                        reason = "not_armed",
                        frame_index = ?ev.frame_index,
                        ts = ?ev.ts,
                        "VAD event dropped: not armed (say the wake word first)"
                    );
                }
                SpeechStartedOutcome::DroppedAlreadyListening => {
                    info!(
                        target: "orch::events",
                        frame_index = ?ev.frame_index,
                        "speech started: already listening (duplicate SS, ignored)"
                    );
                }
                SpeechStartedOutcome::DroppedInTurn => {
                    warn!(
                        target: "orch::events",
                        event = "SpeechStarted",
                        reason = "in_turn",
                        frame_index = ?ev.frame_index,
                        ts = ?ev.ts,
                        "VAD event dropped: turn in progress"
                    );
                }
                SpeechStartedOutcome::WakeWindowExpired => {
                    warn!(
                        target: "orch::events",
                        event = "SpeechStarted",
                        reason = "wake_window_expired",
                        frame_index = ?ev.frame_index,
                        ts = ?ev.ts,
                        "VAD event dropped: wake window expired before SpeechStarted"
                    );
                }
                SpeechStartedOutcome::TurnWindowExpired => {
                    warn!(
                        target: "orch::events",
                        event = "SpeechStarted",
                        reason = "turn_window_expired",
                        frame_index = ?ev.frame_index,
                        ts = ?ev.ts,
                        "VAD event dropped: turn-followup window expired"
                    );
                }
            }
        }
        "SpeechEnded" => {
            if !ev.has_utterance_audio() {
                warn!(
                    target: "orch::events",
                    event = "SpeechEnded",
                    reason = "no_audio",
                    ts = ?ev.ts,
                    "VAD event dropped: missing audio_base64"
                );
            } else {
                // Wake gate: drop SpeechEnded events that don't pass
                // the gate. Each outcome is logged with a distinct
                // `reason=` so a particular dropped utterance can be
                // traced to the exact branch.
                match state.wake.try_dispatch() {
                    DispatchOutcome::Run(guard) => {
                        if let Err(e) = dispatch_utterance(&state, &ev, guard) {
                            warn!(target: "orch::events", "dispatch failed: {e:#}");
                        }
                    }
                    DispatchOutcome::NotArmed => {
                        warn!(
                            target: "orch::events",
                            event = "SpeechEnded",
                            reason = "not_armed",
                            ts = ?ev.ts,
                            "VAD event dropped: no recent WakeWordDetected (say the wake word first, or set wake.required=false)"
                        );
                    }
                    DispatchOutcome::InTurn => {
                        warn!(
                            target: "orch::events",
                            event = "SpeechEnded",
                            reason = "in_turn",
                            ts = ?ev.ts,
                            "VAD event dropped: turn in progress"
                        );
                    }
                    DispatchOutcome::WakeWindowExpired => {
                        warn!(
                            target: "orch::events",
                            event = "SpeechEnded",
                            reason = "wake_window_expired",
                            ts = ?ev.ts,
                            "VAD event dropped: wake window expired"
                        );
                    }
                    DispatchOutcome::TurnWindowExpired => {
                        warn!(
                            target: "orch::events",
                            event = "SpeechEnded",
                            reason = "turn_window_expired",
                            ts = ?ev.ts,
                            "VAD event dropped: turn-followup window expired"
                        );
                    }
                }
            }
        }
        "WakeWordDetected" => {
            // Always forward to result_sink — observers (the v0.1
            // assertion harness, future activity logs) want every
            // wake event regardless of state.
            let payload = json!({
                "name": "WakeWordDetected",
                "model": ev.model,
                "score": ev.score,
                "ts": ev.ts,
            });
            let backends_for_sink = state.backends.clone();
            tokio::spawn(async move {
                forward_to_result_sink(&backends_for_sink, &payload).await;
            });

            match state.wake.on_wake() {
                WakeResult::Bypass => {
                    info!(
                        target: "orch::events",
                        model = ?ev.model,
                        score = ?ev.score,
                        "wake word detected (always-listening; no gate)"
                    );
                }
                WakeResult::Armed => {
                    info!(
                        target: "orch::events",
                        model = ?ev.model,
                        score = ?ev.score,
                        "wake word detected: armed"
                    );
                }
                WakeResult::ArmedBusy => {
                    info!(
                        target: "orch::events",
                        model = ?ev.model,
                        score = ?ev.score,
                        "wake word detected mid-turn: armed for next utterance (barge_in disabled)"
                    );
                }
                WakeResult::BargeIn => {
                    info!(
                        target: "orch::events",
                        model = ?ev.model,
                        score = ?ev.score,
                        "wake word detected mid-turn: barge-in — cancelling current TTS"
                    );
                    let backends_for_stop = state.backends.clone();
                    tokio::spawn(async move {
                        tts_stop(&backends_for_stop).await;
                    });
                }
            }
        }
        other => {
            info!(target: "orch::events", name = %other, "unhandled event");
        }
    }
    (StatusCode::OK, Json(json!({"ok": true})))
}

fn dispatch_utterance(
    state: &AppState,
    ev: &EventEnvelope,
    guard: crate::state::ProcessingGuard,
) -> Result<()> {
    // Defensive: has_utterance_audio() has already established these.
    let b64 = ev
        .audio_base64
        .as_deref()
        .context("audio_base64 missing after has_utterance_audio()")?;
    let sample_rate = ev.sample_rate.context("sample_rate missing")?;
    let pcm = pipeline::decode_audio(b64)?;

    // Move the guard into the spawned task so the wake machine stays
    // in `Processing` until the pipeline finishes — barge-in detection
    // (WakeResult::BargeIn) and the "drop second SpeechEnded mid-turn"
    // semantics both depend on this.
    //
    // Pass `wake` into run_turn too: the pipeline polls
    // `wake.is_in_turn()` between stages, so a barge-in flips the
    // state to Armed and the next stage's HTTP is skipped.
    let backends = state.backends.clone();
    let wake = state.wake.clone();
    tokio::spawn(async move {
        pipeline::run_turn(backends, wake, pcm, sample_rate).await;
        drop(guard);
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
