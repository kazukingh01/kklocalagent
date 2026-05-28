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
    /// Per-mic-frame broadcast. The `u64` is the wall-clock time of the
    /// frame's *last* sample as nanoseconds since UNIX epoch, captured at
    /// `dispatch` (i.e. immediately after the cpal callback finishes
    /// assembling the frame). Subscribers that don't care about timing
    /// can ignore the first field; the WS handler exposes it on the wire
    /// only when the client requests it via `?ts=1`.
    pub mic_tx: broadcast::Sender<(u64, Bytes)>,
    /// Per-playback-track endpoints. Vec index = track id (= `?track=N`
    /// on the /spk WS). Empty Vec while services are stopped; populated
    /// at `service::start_services` with `config.runtime.playback_tracks`
    /// entries, each pointing at an independent cpal output stream that
    /// the Windows audio engine mixes at the OS layer.
    pub spk_tracks: Arc<Mutex<Vec<PlaybackTrack>>>,
    /// AEC far-end ingress (issue #20). `/spk` tees every playback frame here
    /// as `(track_id, pcm)`; the reference mixer task sums the tracks. A
    /// broadcast with no subscribers (AEC disabled / not started) makes the
    /// `/spk` tee a cheap no-op. Always present so the WS handler needn't
    /// branch; only consumed when the mixer task is running.
    pub ref_in_tx: broadcast::Sender<(usize, Bytes)>,
    /// Mixed far-end reference (16 kHz mono s16le, gap-free) published by the
    /// reference mixer task and consumed by the AEC task. ts = epoch ns at
    /// mix time, same clock as `mic_tx`.
    pub ref_tx: broadcast::Sender<(u64, Bytes)>,
    /// Echo-cancelled mic, published by the AEC task. When `aec.enabled`,
    /// the `/mic` WS serves this instead of `mic_tx`; otherwise it has no
    /// producer and `/mic` serves the raw `mic_tx`.
    pub mic_aec_tx: broadcast::Sender<(u64, Bytes)>,
    pub handles: Arc<Mutex<ServiceHandles>>,
}

/// One playback track's user-facing handles. The producer task and cpal
/// thread own clones of these (the sender via the channel, the flush
/// signal via `Arc::clone`), and so do the `/spk` / `/spk/stop` HTTP
/// handlers — `spk_tracks` lets them route by track id.
#[derive(Clone)]
pub struct PlaybackTrack {
    pub sender: mpsc::Sender<PlaybackMessage>,
    pub flush: Arc<FlushSignals>,
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
    /// Vec index = track id. Populated in lockstep with
    /// `AppState.spk_tracks`; both are emptied on `stop_services`.
    pub playback: Vec<PlaybackHandle>,
    /// Reference-mixer + AEC tokio tasks (issue #20), present only when
    /// `aec.enabled`. tokio `JoinHandle`s detach on drop, so `drop_inner`
    /// aborts these explicitly on stop/restart.
    pub aec_tasks: Vec<tokio::task::JoinHandle<()>>,
}
