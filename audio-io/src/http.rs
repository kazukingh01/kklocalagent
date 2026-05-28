use std::collections::HashMap;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use cpal::traits::{DeviceTrait, HostTrait};
use serde::Serialize;
use tracing::{error, info, warn};

use crate::service;
use crate::state::AppState;

pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

#[derive(Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub default: bool,
}

#[derive(Serialize)]
pub struct Devices {
    pub inputs: Vec<DeviceInfo>,
    pub outputs: Vec<DeviceInfo>,
}

pub async fn devices() -> impl IntoResponse {
    match list_devices() {
        Ok(d) => (StatusCode::OK, Json(d)).into_response(),
        Err(e) => {
            error!("devices: {e:?}");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

fn list_devices() -> anyhow::Result<Devices> {
    let host = cpal::default_host();
    let default_in = host.default_input_device().and_then(|d| d.name().ok());
    let default_out = host.default_output_device().and_then(|d| d.name().ok());

    // cpal does not expose stable device IDs, so we match on name. When two
    // devices share a name (e.g. multiple identical USB mics on Windows) only
    // the first is flagged as default — there is exactly one default device.
    let mut inputs = Vec::new();
    let mut default_in_used = false;
    for dev in host.input_devices()? {
        let name = dev.name().unwrap_or_default();
        let default = !default_in_used && Some(&name) == default_in.as_ref();
        if default {
            default_in_used = true;
        }
        inputs.push(DeviceInfo { name, default });
    }
    let mut outputs = Vec::new();
    let mut default_out_used = false;
    for dev in host.output_devices()? {
        let name = dev.name().unwrap_or_default();
        let default = !default_out_used && Some(&name) == default_out.as_ref();
        if default {
            default_out_used = true;
        }
        outputs.push(DeviceInfo { name, default });
    }
    Ok(Devices { inputs, outputs })
}

pub async fn start(State(state): State<AppState>) -> impl IntoResponse {
    match service::start_services(&state).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "status": "started" }))).into_response(),
        Err(e) => {
            error!("start: {e:?}");
            // anyhow carries the root cause in the chain. Walk it so a
            // nested device-lookup error still maps to 404, not 500.
            let msg = format!("{e:#}");
            let status = if msg.contains("already in progress") {
                StatusCode::CONFLICT
            } else if msg.contains("not found") || msg.contains("no default") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, e.to_string()).into_response()
        }
    }
}

pub async fn stop(State(state): State<AppState>) -> impl IntoResponse {
    match service::stop_services(&state).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "status": "stopped" }))).into_response(),
        Err(e) => {
            error!("stop: {e:?}");
            (StatusCode::CONFLICT, e.to_string()).into_response()
        }
    }
}

pub async fn spk_stop(
    Query(params): Query<HashMap<String, String>>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    // `?track=N` stops just that track; absent = stop all tracks
    // (treated as a full audio reset — useful as a global barge-in that
    // takes out any TTS-streamer and any agent-side file playback in
    // one shot).
    let requested: Option<usize> = params.get("track").and_then(|s| s.parse().ok());
    let tracks = state.spk_tracks.lock().await;
    match requested {
        Some(idx) => match tracks.get(idx) {
            Some(t) => {
                t.flush.trigger();
                info!(track = idx, "spk_stop: flush signaled");
                (
                    StatusCode::OK,
                    Json(serde_json::json!({ "status": "flushed", "track": idx })),
                )
                    .into_response()
            }
            None => {
                warn!(
                    track = idx,
                    n_tracks = tracks.len(),
                    "spk_stop: track out of range"
                );
                (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "status": "no such track",
                        "track": idx,
                        "n_tracks": tracks.len(),
                    })),
                )
                    .into_response()
            }
        },
        None => {
            for t in tracks.iter() {
                t.flush.trigger();
            }
            info!(
                n_tracks = tracks.len(),
                "spk_stop: flush signaled (all tracks)"
            );
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "flushed",
                    "n_tracks": tracks.len(),
                })),
            )
                .into_response()
        }
    }
}
