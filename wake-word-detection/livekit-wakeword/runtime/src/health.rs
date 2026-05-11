//! GET /health on `WW_LISTEN`. Returns 200 + `{"ok":true}` only when
//! both the model is loaded and the mic WS is currently connected — same
//! contract the openwakeword shim exposes, so compose's existing
//! `service_healthy` gate behaves identically across implementations.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{extract::State, http::StatusCode, response::Json, routing::get, Router};
use serde::Serialize;
use tokio::net::TcpListener;
use tracing::info;

#[derive(Clone)]
pub struct HealthFlags {
    pub model_loaded: Arc<AtomicBool>,
    pub ws_connected: Arc<AtomicBool>,
}

#[derive(Serialize)]
struct HealthResp {
    ok: bool,
}

async fn health_handler(State(flags): State<HealthFlags>) -> (StatusCode, Json<HealthResp>) {
    let ok = flags.model_loaded.load(Ordering::Relaxed)
        && flags.ws_connected.load(Ordering::Relaxed);
    let status = if ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(HealthResp { ok }))
}

pub async fn serve(addr: SocketAddr, flags: HealthFlags) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(health_handler))
        .with_state(flags);
    let listener = TcpListener::bind(addr).await?;
    info!(%addr, "http server listening");
    axum::serve(listener, app).await?;
    Ok(())
}
