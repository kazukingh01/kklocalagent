use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::{broadcast, mpsc, Mutex};

use crate::capture::CaptureHandle;
use crate::config::Config;
use crate::playback::{PlaybackHandle, PlaybackMessage};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    /// The `u64` is the wall-clock epoch-ns of the frame's *last* sample,
    /// captured right after the cpal callback assembles the frame; on the wire
    /// only with `?ts=1`.
    pub mic_tx: broadcast::Sender<(u64, Bytes)>,
    /// Vec index = track id (= `?track=N` on the /spk WS); empty while stopped.
    pub spk_tracks: Arc<Mutex<Vec<PlaybackTrack>>>,
    /// AEC far-end ingress (issue #20), `(track_id, pcm)`. Tapped from each
    /// playback track *after* the playback ring so the reference is on the same
    /// wall clock as the sound leaving the speaker.
    pub ref_in_tx: broadcast::Sender<(usize, Bytes)>,
    /// Mixed far-end reference (16 kHz mono s16le, gap-free). The AEC pairs it
    /// to the mic by count (one far sample per near sample), so no timestamp.
    pub ref_tx: broadcast::Sender<Bytes>,
    /// Echo-cancelled mic; when `aec.enabled` the `/mic` WS serves this instead
    /// of `mic_tx`.
    pub mic_aec_tx: broadcast::Sender<(u64, Bytes)>,
    pub handles: Arc<Mutex<ServiceHandles>>,
}

#[derive(Clone)]
pub struct PlaybackTrack {
    pub sender: mpsc::Sender<PlaybackMessage>,
    pub flush: Arc<FlushSignals>,
    /// `/spk/stop` must also *close* the track's active `/spk` WS, not just
    /// flush the ring — a continuously streaming client would otherwise refill
    /// the ring and keep playing.
    pub close: Arc<tokio::sync::Notify>,
}

pub struct FlushSignals {
    pub producer: AtomicBool,
    pub consumer: AtomicBool,
}

impl FlushSignals {
    pub fn new() -> Self {
        Self {
            producer: AtomicBool::new(false),
            consumer: AtomicBool::new(false),
        }
    }

    pub fn trigger(&self) {
        self.producer.store(true, Ordering::Relaxed);
        self.consumer.store(true, Ordering::Relaxed);
    }
}

impl Default for FlushSignals {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct ServiceHandles {
    pub capture: Option<CaptureHandle>,
    /// Vec index = track id, in lockstep with `AppState.spk_tracks`.
    pub playback: Vec<PlaybackHandle>,
    /// Reference-mixer + AEC tasks (issue #20). tokio `JoinHandle`s detach on
    /// drop, so these are aborted explicitly on stop/restart.
    pub aec_tasks: Vec<tokio::task::JoinHandle<()>>,
}
