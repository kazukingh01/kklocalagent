use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Instant;

use axum::extract::ws::{close_code, CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::response::IntoResponse;
use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use crate::playback::PlaybackMessage;
use crate::state::AppState;

pub async fn ws_mic(
    ws: WebSocketUpgrade,
    Query(params): Query<HashMap<String, String>>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let with_ts = matches!(params.get("ts").map(String::as_str), Some("1"));
    ws.on_upgrade(move |socket| handle_mic(socket, state, with_ts))
}

async fn handle_mic(mut socket: WebSocket, state: AppState, with_ts: bool) {
    let aec = state.config.aec.enabled;
    let mut rx = if aec {
        state.mic_aec_tx.subscribe()
    } else {
        state.mic_tx.subscribe()
    };
    info!(with_ts, aec, "mic ws: client connected");
    loop {
        tokio::select! {
            msg = rx.recv() => match msg {
                Ok((ts_ns, frame)) => {
                    let payload = if with_ts {
                        let mut buf = Vec::with_capacity(8 + frame.len());
                        buf.extend_from_slice(&ts_ns.to_le_bytes());
                        buf.extend_from_slice(&frame);
                        buf
                    } else {
                        frame.to_vec()
                    };
                    if socket.send(Message::Binary(payload)).await.is_err() {
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

pub async fn ws_spk(
    ws: WebSocketUpgrade,
    Query(params): Query<HashMap<String, String>>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let track_id: usize = params
        .get("track")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    ws.on_upgrade(move |socket| handle_spk(socket, state, track_id))
}

async fn handle_spk(mut socket: WebSocket, state: AppState, track_id: usize) {
    info!(track_id, "spk ws: client connected");
    let mut pcm_frames_received: u64 = 0;
    let (spk_tx, close) = {
        let guard = state.spk_tracks.lock().await;
        match guard.get(track_id) {
            Some(t) => (t.sender.clone(), t.close.clone()),
            None => {
                warn!(
                    track_id,
                    n_tracks = guard.len(),
                    "spk ws: track id out of range or playback not running; closing"
                );
                let _ = socket
                    .send(Message::Close(Some(CloseFrame {
                        code: close_code::POLICY,
                        reason: Cow::Borrowed("invalid track or playback not running"),
                    })))
                    .await;
                return;
            }
        }
    };

    // Hardware-vs-system clock drift: baseline cpal's hardware-clock-paced
    // counters at connect, diff at disconnect against wall-clock elapsed.
    let drift_baseline = {
        let guard = state.handles.lock().await;
        guard.playback.get(track_id).map(|h| {
            let (cb, samples) = h.stats().snapshot();
            (
                Instant::now(),
                cb,
                samples,
                h.native_rate(),
                h.native_channels(),
                h.stats(),
            )
        })
    };

    // The `Notified` future is created once and pinned so it stays registered
    // across select iterations (no lost wakeup); `biased` checks close first.
    let close_notified = close.notified();
    tokio::pin!(close_notified);
    loop {
        let msg = tokio::select! {
            biased;
            _ = &mut close_notified => {
                info!(track_id, "spk ws: closed by /spk/stop");
                let _ = socket
                    .send(Message::Close(Some(CloseFrame {
                        code: close_code::NORMAL,
                        reason: Cow::Borrowed("stopped by /spk/stop"),
                    })))
                    .await;
                break;
            }
            recv = socket.recv() => recv,
        };
        let Some(msg) = msg else { break };
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
                let frame = Bytes::from(data);
                // NB: the AEC far-end reference is NOT teed here but in the
                // playback output callback, so it aligns with the speaker
                // output instead of leading it by the playback-ring residency.
                if spk_tx.send(PlaybackMessage::Frame(frame)).await.is_err() {
                    warn!("spk ws: playback task gone; closing");
                    break;
                }
                pcm_frames_received = pcm_frames_received.saturating_add(1);
            }
            Ok(Message::Text(text)) => {
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
                // drain_done fires when the ring empties or /spk/stop flushed
                // it; a oneshot RecvError (producer task exited) is treated as
                // drained so the client doesn't hang forever.
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
    if let Some((start, cb0, samples0, native_rate, native_channels, stats)) = drift_baseline {
        if pcm_frames_received == 0 {
            // Skip the drift line: cpal's sample counter advances on silence
            // too, so "drift" for a zero-PCM session is meaningless.
        } else {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            let (cb1, samples1) = stats.snapshot();
            let callbacks = cb1.saturating_sub(cb0);
            let consumed = samples1.saturating_sub(samples0);
            let frames_per_sec = native_rate as u64 * native_channels.max(1) as u64;
            let consumed_ms = if frames_per_sec > 0 {
                consumed * 1000 / frames_per_sec
            } else {
                0
            };
            let drift_ms = consumed_ms as i64 - elapsed_ms as i64;
            info!(
                track_id,
                elapsed_ms,
                callbacks,
                consumed_samples = consumed,
                consumed_ms,
                drift_ms,
                pcm_frames_received,
                "spk ws session: cpal hw-clock consumed vs wall elapsed"
            );
        }
    }
    info!(track_id, "spk ws: client disconnected");
}
