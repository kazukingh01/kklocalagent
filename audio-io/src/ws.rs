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
    // `?ts=1` opts the client into an 8-byte little-endian u64 header
    // (epoch ns of the frame's *last* sample) prepended to each PCM
    // frame. Default behavior is unchanged so existing consumers (VAD,
    // openwakeword shim, tests) keep working without modification.
    let with_ts = matches!(params.get("ts").map(String::as_str), Some("1"));
    ws.on_upgrade(move |socket| handle_mic(socket, state, with_ts))
}

async fn handle_mic(mut socket: WebSocket, state: AppState, with_ts: bool) {
    // AEC (issue #20) is a single switch: when `aec.enabled`, `/mic` serves
    // the echo-cancelled stream; otherwise the raw mic. No per-connection
    // opt-in — every consumer (VAD, wwd) transparently gets whichever the
    // host config selected, so enabling AEC needs no client/compose change.
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
    // `?track=N` picks one of the parallel playback streams (default 0).
    // 0 keeps the existing TTS-streamer client working unmodified.
    let track_id: usize = params
        .get("track")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    ws.on_upgrade(move |socket| handle_spk(socket, state, track_id))
}

async fn handle_spk(mut socket: WebSocket, state: AppState, track_id: usize) {
    info!(track_id, "spk ws: client connected");
    // Tracks whether this WS actually pushed any PCM. cpal is always
    // running (consuming silence when no producer), so a drift report
    // for a zero-PCM session would still log a number — but it would
    // be measuring host scheduling jitter against the device crystal,
    // not anything related to /spk. Suppress that case so the log
    // line is unambiguously "this session's PCM throughput vs hw clock".
    let mut pcm_frames_received: u64 = 0;
    let spk_tx = {
        let guard = state.spk_tracks.lock().await;
        match guard.get(track_id) {
            Some(t) => t.sender.clone(),
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

    // Hardware-vs-system clock drift snapshot. We baseline cpal's
    // hardware-clock-paced counters at connect and diff at disconnect;
    // the wall-clock duration of the session is the system-clock
    // reference. The drift is per-track (each cpal stream has its own
    // hardware-clock-paced sample counter).
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
                let frame = Bytes::from(data);
                // Tee a copy to the AEC far-end mixer (issue #20). Cheap:
                // `Bytes` is refcounted, and with AEC disabled the broadcast
                // has no subscriber so `send` is a no-op error we ignore.
                let _ = state.ref_in_tx.send((track_id, frame.clone()));
                if spk_tx.send(PlaybackMessage::Frame(frame)).await.is_err() {
                    warn!("spk ws: playback task gone; closing");
                    break;
                }
                pcm_frames_received = pcm_frames_received.saturating_add(1);
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
    if let Some((start, cb0, samples0, native_rate, native_channels, stats)) = drift_baseline {
        if pcm_frames_received == 0 {
            // No /spk traffic this session → skip the drift line. cpal's
            // sample counter advances on silence too (the consumer task
            // pushes zeros when the producer is idle), and reporting that
            // as "drift_ms" is meaningless for /spk diagnosis.
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
