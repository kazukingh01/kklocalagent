use std::borrow::Cow;

use axum::extract::ws::{close_code, CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use crate::playback::PlaybackMessage;
use crate::state::AppState;

pub async fn ws_mic(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_mic(socket, state))
}

async fn handle_mic(mut socket: WebSocket, state: AppState) {
    let mut rx = state.mic_tx.subscribe();
    info!("mic ws: client connected");
    loop {
        tokio::select! {
            msg = rx.recv() => match msg {
                Ok(frame) => {
                    if socket.send(Message::Binary(frame.to_vec())).await.is_err() {
                        break;
                    }
                }
                Err(RecvError::Lagged(n)) => {
                    warn!("mic ws: lagged {n} frames (client too slow)");
                }
                Err(RecvError::Closed) => break,
            },
            inbound = socket.recv() => match inbound {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(e)) => {
                    debug!("mic ws recv err: {e}");
                    break;
                }
                _ => {}
            }
        }
    }
    info!("mic ws: client disconnected");
}

pub async fn ws_spk(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_spk(socket, state))
}

async fn handle_spk(mut socket: WebSocket, state: AppState) {
    info!("spk ws: client connected");
    let spk_tx = {
        let guard = state.spk_tx.lock().await;
        match guard.as_ref() {
            Some(tx) => tx.clone(),
            None => {
                warn!("spk ws: playback not running; closing");
                let _ = socket
                    .send(Message::Close(Some(CloseFrame {
                        code: close_code::POLICY,
                        reason: Cow::Borrowed("playback not running"),
                    })))
                    .await;
                return;
            }
        }
    };

    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Binary(data)) => {
                if data.len() % 2 != 0 {
                    warn!(
                        len = data.len(),
                        "spk ws: rejecting odd-length frame (s16le requires even bytes)"
                    );
                    let _ = socket
                        .send(Message::Close(Some(CloseFrame {
                            code: close_code::INVALID,
                            reason: Cow::Borrowed("odd-length frame (s16le requires even bytes)"),
                        })))
                        .await;
                    break;
                }
                if spk_tx
                    .send(PlaybackMessage::Frame(Bytes::from(data)))
                    .await
                    .is_err()
                {
                    warn!("spk ws: playback task gone; closing");
                    break;
                }
            }
            Ok(Message::Text(text)) => {
                // EOS/drain handshake. Client sends `{"type":"eos"}`
                // after the last PCM frame; we forward an Eos marker
                // through the producer task, wait for it to confirm
                // the cpal ring is empty, then echo
                // `{"type":"drained"}` back so the client knows the
                // speaker has *actually* finished. Lets the
                // orchestrator stop guessing the audio-tail with a
                // tail_quiet_ms timeout and trust the WS handshake
                // as the precise boundary instead.
                //
                // Unknown types are logged + ignored — keeps the
                // protocol forward-compatible if we add other control
                // messages later (e.g. mid-stream priority hints).
                let parsed: Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(text = %text, err = %e, "spk ws: ignoring non-json text frame");
                        continue;
                    }
                };
                let kind = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if kind != "eos" {
                    debug!(kind = %kind, "spk ws: ignoring unknown control message");
                    continue;
                }
                let (drain_done_tx, drain_done_rx) = oneshot::channel::<()>();
                if spk_tx
                    .send(PlaybackMessage::Eos {
                        drain_done: drain_done_tx,
                    })
                    .await
                    .is_err()
                {
                    warn!("spk ws: playback task gone before drain handshake; closing");
                    break;
                }
                // The producer task fires drain_done either when the
                // ring is empty (normal path) or when /spk/stop's
                // flush yanked everything (cancellation). RecvError
                // on the oneshot means the producer task itself
                // exited — treat it the same as a successful drain
                // so the client doesn't hang on the WS forever.
                let _ = drain_done_rx.await;
                let payload = json!({"type": "drained"}).to_string();
                if socket.send(Message::Text(payload)).await.is_err() {
                    debug!("spk ws: client closed before drained reply");
                    break;
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(_) => {}
            Err(e) => {
                debug!("spk ws recv err: {e}");
                break;
            }
        }
    }
    info!("spk ws: client disconnected");
}
