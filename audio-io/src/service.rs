use anyhow::{anyhow, Context, Result};
use tracing::info;

use crate::capture::start_capture;
use crate::playback::start_playback;
use crate::state::{AppState, ServiceHandles};

pub async fn start_services(state: &AppState) -> Result<()> {
    let mut handles = state
        .handles
        .try_lock()
        .map_err(|_| anyhow!("start/stop already in progress"))?;
    if handles.capture.is_some() || handles.playback.is_some() {
        info!("start_services: restarting existing services");
        drop_inner(&mut handles);
        *state.spk_tx.lock().await = None;
    }

    let capture = start_capture(
        &state.config.input.device,
        state.config.audio.clone(),
        state.mic_tx.clone(),
    )
    .context("start_capture")?;
    handles.capture = Some(capture);

    let playback = start_playback(
        &state.config.output.device,
        state.config.audio.clone(),
        state.config.runtime.playback_buffer_ms,
        state.flush.clone(),
    )
    .context("start_playback")?;
    *state.spk_tx.lock().await = Some(playback.sender());
    handles.playback = Some(playback);

    info!("services started");
    Ok(())
}

pub async fn stop_services(state: &AppState) -> Result<()> {
    let mut handles = state
        .handles
        .try_lock()
        .map_err(|_| anyhow!("start/stop already in progress"))?;
    drop_inner(&mut handles);
    *state.spk_tx.lock().await = None;
    info!("services stopped");
    Ok(())
}

fn drop_inner(handles: &mut ServiceHandles) {
    handles.capture = None;
    handles.playback = None;
}
