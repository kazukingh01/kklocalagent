use std::sync::Arc;

use anyhow::Context;
use tracing::info;

use crate::aec::{aec_task, reference_mixer_task, Aec, EchoCanceller};
use crate::capture::start_capture;
use crate::error::AudioError;
use crate::playback::start_playback;
use crate::state::{AppState, FlushSignals, PlaybackTrack, ServiceHandles};

pub async fn start_services(state: &AppState) -> Result<(), AudioError> {
    let mut handles = state.handles.try_lock().map_err(|_| AudioError::Busy)?;
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
    .context("start_capture")
    .map_err(AudioError::from_chain)?;
    handles.capture = Some(capture);

    // Independent cpal output streams against the same device; WASAPI shared
    // mode mixes them at the OS layer.
    let n_tracks = state.config.runtime.playback_tracks as usize;
    let ref_tap = if state.config.aec.enabled {
        Some(state.ref_in_tx.clone())
    } else {
        None
    };
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
            ref_tap.clone(),
        )
        .with_context(|| format!("start_playback (track {track_id})"))
        .map_err(AudioError::from_chain)?;
        new_tracks.push(PlaybackTrack {
            sender: playback.sender(),
            flush,
            close: Arc::new(tokio::sync::Notify::new()),
        });
        new_handles.push(playback);
    }
    *state.spk_tracks.lock().await = new_tracks;
    handles.playback = new_handles;

    if state.config.aec.enabled {
        let spf = state.config.audio.samples_per_frame();
        let mixer = tokio::spawn(reference_mixer_task(
            state.ref_in_tx.subscribe(),
            state.ref_tx.clone(),
            state.config.audio.sample_rate,
            spf,
            n_tracks,
            state.config.audio.frame_ms,
        ));
        let sr = state.config.audio.sample_rate;
        let flen = state.config.aec.filter_length_ms;
        let canceller: Box<dyn EchoCanceller> = match state.config.aec.backend.as_str() {
            "nlms" => Box::new(Aec::new(sr, flen)),
            #[cfg(feature = "speex")]
            "speex" => Box::new(crate::speex::SpeexAec::new(sr, spf, flen)),
            other => return Err(AudioError::UnsupportedBackend(other.to_string())),
        };
        let aec_handle = tokio::spawn(aec_task(
            state.mic_tx.subscribe(),
            state.ref_tx.subscribe(),
            state.mic_aec_tx.clone(),
            sr,
            canceller,
        ));
        handles.aec_tasks = vec![mixer, aec_handle];
        info!(
            backend = %state.config.aec.backend,
            filter_length_ms = flen,
            "aec enabled"
        );
    }

    info!(n_tracks, aec = state.config.aec.enabled, "services started");
    Ok(())
}

pub async fn stop_services(state: &AppState) -> Result<(), AudioError> {
    let mut handles = state.handles.try_lock().map_err(|_| AudioError::Busy)?;
    drop_inner(&mut handles);
    state.spk_tracks.lock().await.clear();
    info!("services stopped");
    Ok(())
}

fn drop_inner(handles: &mut ServiceHandles) {
    handles.capture = None;
    handles.playback.clear();
    // tokio JoinHandles detach on drop, so abort the AEC tasks explicitly.
    for t in handles.aec_tasks.drain(..) {
        t.abort();
    }
}
