use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use cpal::traits::{DeviceTrait, HostTrait};
use serde::Serialize;
use tracing::{error, info};

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

    let mut inputs = Vec::new();
    for dev in host.input_devices()? {
        let name = dev.name().unwrap_or_default();
        let default = Some(&name) == default_in.as_ref();
        inputs.push(DeviceInfo { name, default });
    }
    let mut outputs = Vec::new();
    for dev in host.output_devices()? {
        let name = dev.name().unwrap_or_default();
        let default = Some(&name) == default_out.as_ref();
        outputs.push(DeviceInfo { name, default });
    }
    Ok(Devices { inputs, outputs })
}

pub async fn start(State(state): State<AppState>) -> impl IntoResponse {
    match service::start_services(&state).await {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "status": "started" }))).into_response(),
        Err(e) => {
            error!("start: {e:?}");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

pub async fn stop(State(state): State<AppState>) -> impl IntoResponse {
    service::stop_services(&state).await;
    Json(serde_json::json!({ "status": "stopped" }))
}

pub async fn spk_stop(State(state): State<AppState>) -> impl IntoResponse {
    state.flush.trigger();
    info!("spk_stop: flush signaled");
    Json(serde_json::json!({ "status": "flushed" }))
}
