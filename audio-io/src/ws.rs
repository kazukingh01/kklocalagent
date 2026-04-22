use std::borrow::Cow;

use axum::extract::ws::{close_code, CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use bytes::Bytes;
use tokio::sync::broadcast::error::RecvError;
use tracing::{debug, info, warn};

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
                if spk_tx.send(Bytes::from(data)).await.is_err() {
                    warn!("spk ws: playback task gone; closing");
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
