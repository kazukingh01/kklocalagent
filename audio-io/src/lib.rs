pub mod capture;
pub mod config;
pub mod framer;
pub mod http;
pub mod playback;
pub mod service;
pub mod state;
pub mod ws;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, Mutex};
use tracing::info;

use crate::config::Config;
use crate::state::{AppState, FlushSignals, ServiceHandles};

pub async fn run(config: Config) -> Result<()> {
    config.validate().context("invalid config")?;
    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .context("parsing bind address")?;
    let state = build_state(config.clone());

    if config.runtime.autostart {
        service::start_services(&state)
            .await
            .context("autostart services")?;
    }

    let app = build_router(state);

    let listener = TcpListener::bind(addr).await?;
    info!(addr = %listener.local_addr()?, "audio-io listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

pub fn build_state(config: Config) -> AppState {
    let (mic_tx, _) = broadcast::channel(config.runtime.mic_broadcast_frames.max(1) as usize);
    AppState {
        config: Arc::new(config),
        mic_tx,
        spk_tx: Arc::new(Mutex::new(None)),
        flush: Arc::new(FlushSignals::new()),
        handles: Arc::new(Mutex::new(ServiceHandles::default())),
    }
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(http::health))
        .route("/devices", get(http::devices))
        .route("/start", post(http::start))
        .route("/stop", post(http::stop))
        .route("/spk/stop", post(http::spk_stop))
        .route("/mic", get(ws::ws_mic))
        .route("/spk", get(ws::ws_spk))
        .with_state(state)
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
