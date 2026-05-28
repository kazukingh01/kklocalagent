use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tracing::info;

use crate::capture::start_capture;
use crate::playback::start_playback;
use crate::state::{AppState, FlushSignals, PlaybackTrack, ServiceHandles};

pub async fn start_services(state: &AppState) -> Result<()> {
    let mut handles = state
        .handles
        .try_lock()
        .map_err(|_| anyhow!("start/stop already in progress"))?;
    if handles.capture.is_some() || !handles.playback.is_empty() {
        info!("start_services: restarting existing services");
        drop_inner(&mut handles);
        state.spk_tracks.lock().await.clear();
    }

    let capture = start_capture(
        &state.config.input.device,
        state.config.audio.clone(),
        state.mic_tx.clone(),
    )
    .context("start_capture")?;
    handles.capture = Some(capture);

    // Open `playback_tracks` independent cpal output streams against the
    // same device. WASAPI shared mode mixes them at the OS layer, so
    // callers (TTS-streamer on track 0, agent on track 1, ...) can push
    // PCM in parallel without per-sample stomping or contention.
    let n_tracks = state.config.runtime.playback_tracks as usize;
    let mut new_tracks = Vec::with_capacity(n_tracks);
    let mut new_handles = Vec::with_capacity(n_tracks);
    for track_id in 0..n_tracks {
        let flush = Arc::new(FlushSignals::new());
        let playback = start_playback(
            track_id,
            &state.config.output.device,
            state.config.audio.clone(),
            state.config.runtime.playback_buffer_ms,
            flush.clone(),
        )
        .with_context(|| format!("start_playback (track {track_id})"))?;
        new_tracks.push(PlaybackTrack {
            sender: playback.sender(),
            flush,
        });
        new_handles.push(playback);
    }
    *state.spk_tracks.lock().await = new_tracks;
    handles.playback = new_handles;

    info!(n_tracks, "services started");
    Ok(())
}

pub async fn stop_services(state: &AppState) -> Result<()> {
    let mut handles = state
        .handles
        .try_lock()
        .map_err(|_| anyhow!("start/stop already in progress"))?;
    drop_inner(&mut handles);
    state.spk_tracks.lock().await.clear();
    info!("services stopped");
    Ok(())
}

fn drop_inner(handles: &mut ServiceHandles) {
    handles.capture = None;
    // Drop in reverse order so a track's logger task doesn't see the
    // producer task abort mid-Tick (purely defensive — Drop on each is
    // independent — but keeps log order tidy).
    handles.playback.clear();
}
