use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tracing::info;

use crate::aec::{aec_task, reference_mixer_task, Aec};
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

    // Acoustic echo cancellation (issue #20). When enabled, spin up the
    // reference mixer (sums the far-end `/spk` tracks) and the AEC task
    // (cancels that far-end out of the mic; `/mic` then serves it). Both
    // subscribe to broadcast channels that already exist on AppState, so the
    // `/spk` tee and `/mic` handler don't depend on these tasks being up.
    if state.config.aec.enabled {
        let spf = state.config.audio.samples_per_frame();
        let mixer = tokio::spawn(reference_mixer_task(
            state.ref_in_tx.subscribe(),
            state.ref_tx.clone(),
            spf,
            n_tracks,
            state.config.audio.frame_ms,
        ));
        let aec = Aec::new(
            state.config.audio.sample_rate,
            state.config.aec.filter_length_ms,
            state.config.aec.initial_delay_ms,
        );
        let canceller = tokio::spawn(aec_task(
            state.mic_tx.subscribe(),
            state.ref_tx.subscribe(),
            state.mic_aec_tx.clone(),
            aec,
        ));
        handles.aec_tasks = vec![mixer, canceller];
        info!(
            backend = %state.config.aec.backend,
            filter_length_ms = state.config.aec.filter_length_ms,
            initial_delay_ms = state.config.aec.initial_delay_ms,
            "aec enabled"
        );
    }

    info!(n_tracks, aec = state.config.aec.enabled, "services started");
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
    // tokio JoinHandles detach on drop, so abort the AEC tasks explicitly.
    for t in handles.aec_tasks.drain(..) {
        t.abort();
    }
}
