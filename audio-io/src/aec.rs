//! Acoustic echo cancellation, run inside audio-io (issue #20, 方式B).
//!
//! The far-end reference is tapped where cpal actually *consumes* each
//! playback frame (the output callback), NOT at the `/spk` WS ingress: an
//! earlier design teed the raw WS bytes before the playback ring, so the echo
//! lagged the reference by the whole ring residency — beyond the filter
//! window, killing all cancellation.

mod mixer;
mod nlms;

use std::collections::VecDeque;

use bytes::Bytes;
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;
use tracing::{debug, info, warn};

use crate::pcm::{bytes_to_i16, i16_to_bytes};

pub use mixer::ReferenceMixer;
pub use nlms::Aec;

/// Per-frame diagnostics for the `aec stats` log line; backends that don't
/// expose a field leave it at 0.
#[derive(Default, Clone, Copy)]
pub struct CancellerStats {
    /// Applied bulk pre-delay (ms).
    pub delay_ms: u32,
    /// Total estimated echo delay (ms).
    pub peak_ms: u32,
    /// Measured residual-echo ratio.
    pub erl: f32,
    /// Residual-suppressor gain (1.0 = none).
    pub nlp_gain: f32,
    /// L2 norm of the adaptive weights (convergence indicator).
    pub w_l2: f32,
}

/// A pluggable acoustic echo canceller, selected by `aec.backend`.
pub trait EchoCanceller: Send {
    /// Cancel the echo of `far` from `near` (one frame each, equal length).
    fn process_frame(&mut self, near: &[i16], far: &[i16]) -> Vec<i16>;
    fn stats(&self) -> CancellerStats {
        CancellerStats::default()
    }
}

fn sum_sq(samples: &[i16]) -> f64 {
    samples.iter().map(|&s| (s as f64) * (s as f64)).sum()
}

/// Drains per-track far-end frames into the [`ReferenceMixer`] and publishes
/// one mixed frame every `frame_ms` on `ref_tx`.
pub async fn reference_mixer_task(
    mut ref_in_rx: broadcast::Receiver<(usize, Bytes)>,
    ref_tx: broadcast::Sender<Bytes>,
    sample_rate: u32,
    samples_per_frame: usize,
    n_tracks: usize,
    frame_ms: u32,
) {
    let mut mixer = ReferenceMixer::new(sample_rate, samples_per_frame, n_tracks);
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(frame_ms as u64));
    // Report backlog drops ~once per second so a playback clock outrunning the
    // mix timer is visible rather than silent.
    let warn_every = (1000 / frame_ms.max(1)).max(1) as u64;
    let mut ticks: u64 = 0;
    let mut last_dropped: u64 = 0;
    info!(
        samples_per_frame,
        n_tracks, frame_ms, "reference mixer started"
    );
    loop {
        tokio::select! {
            inbound = ref_in_rx.recv() => match inbound {
                Ok((track_id, bytes)) => mixer.push(track_id, &bytes),
                Err(RecvError::Lagged(n)) => {
                    warn!("reference mixer: lagged {n} far-end frames");
                }
                Err(RecvError::Closed) => break,
            },
            _ = interval.tick() => {
                let frame = mixer.tick();
                let _ = ref_tx.send(Bytes::from(frame));
                ticks += 1;
                if ticks.is_multiple_of(warn_every) && mixer.dropped() > last_dropped {
                    warn!(
                        dropped = mixer.dropped() - last_dropped,
                        "reference mixer: per-track backlog over cap, dropped oldest \
                         far samples (playback clock outrunning the mix timer?)"
                    );
                    last_dropped = mixer.dropped();
                }
            }
        }
    }
    info!("reference mixer exiting");
}

/// Pairs near (mic) and far (mixed reference) by *count*, not timestamp: both
/// are real-time 16 kHz streams that start together, so the constant far-path
/// latency plus the acoustic delay becomes a tap inside the filter window,
/// which the adaptive filter finds. An earlier version aligned by wall-clock
/// timestamps and dropped far older than the current near frame; the far path
/// is a few frames more latent than the mic path, so its correctly-old frames
/// were discarded wholesale (far_rms→0, zero cancellation). Counting tolerates
/// that latency; underflow (far momentarily behind) feeds silence.
pub async fn aec_task(
    mut mic_rx: broadcast::Receiver<(u64, Bytes)>,
    mut ref_rx: broadcast::Receiver<Bytes>,
    aec_tx: broadcast::Sender<(u64, Bytes)>,
    sample_rate: u32,
    mut canceller: Box<dyn EchoCanceller>,
) {
    info!("aec task started");
    // Runaway guard only (~250 ms): the speaker→mic bulk delay is handled
    // *inside* the Aec by its measured pre-delay, not by holding samples here,
    // so at steady state this cap never fires.
    let max_backlog = (sample_rate as usize / 4).max(1);
    let mut far_pending: VecDeque<i16> = VecDeque::new();

    let log_every = (sample_rate as usize / 2).max(1); // ~0.5 s of samples
    let mut near_sq = 0.0f64;
    let mut far_sq = 0.0f64;
    let mut resid_sq = 0.0f64;
    let mut acc_samples = 0usize;
    let mut acc_dropped = 0usize;
    let mut acc_padded = 0usize;

    loop {
        tokio::select! {
            far = ref_rx.recv() => match far {
                Ok(bytes) => {
                    far_pending.extend(bytes_to_i16(&bytes));
                    if far_pending.len() > max_backlog {
                        let excess = far_pending.len() - max_backlog;
                        far_pending.drain(..excess);
                        acc_dropped += excess;
                    }
                }
                Err(RecvError::Lagged(n)) => {
                    warn!("aec: lagged {n} far-end frames");
                    far_pending.clear();
                }
                Err(RecvError::Closed) => break,
            },
            near = mic_rx.recv() => match near {
                Ok((ts, bytes)) => {
                    let near = bytes_to_i16(&bytes);
                    let mut far = Vec::with_capacity(near.len());
                    let mut padded = 0usize;
                    for _ in 0..near.len() {
                        match far_pending.pop_front() {
                            Some(s) => far.push(s),
                            None => {
                                far.push(0);
                                padded += 1;
                            }
                        }
                    }
                    let cleaned = canceller.process_frame(&near, &far);

                    near_sq += sum_sq(&near);
                    far_sq += sum_sq(&far);
                    resid_sq += sum_sq(&cleaned);
                    acc_samples += near.len();
                    acc_padded += padded;
                    if acc_samples >= log_every {
                        let denom = acc_samples as f64;
                        let near_rms = (near_sq / denom).sqrt();
                        let far_rms = (far_sq / denom).sqrt();
                        let resid_rms = (resid_sq / denom).sqrt();
                        // Diagnostics: enable with RUST_LOG=audio_io::aec=debug;
                        // gated on actual playback/speech so idle is silent.
                        if near_rms > 30.0 || far_rms > 30.0 {
                            // erle_db is the backend-agnostic comparison metric.
                            let erle_db = 20.0 * (near_rms / resid_rms.max(1.0)).log10();
                            let s = canceller.stats();
                            debug!(
                                near_rms = near_rms as i64,
                                far_rms = far_rms as i64,
                                resid_rms = resid_rms as i64,
                                erle_db = format!("{erle_db:.1}"),
                                delay_ms = s.delay_ms,
                                peak_ms = s.peak_ms,
                                erl = format!("{:.3}", s.erl),
                                w_l2 = format!("{:.3}", s.w_l2),
                                nlp_g = format!("{:.2}", s.nlp_gain),
                                far_buf = far_pending.len(),
                                dropped = acc_dropped,
                                padded = acc_padded,
                                "aec stats"
                            );
                        }
                        near_sq = 0.0;
                        far_sq = 0.0;
                        resid_sq = 0.0;
                        acc_samples = 0;
                        acc_dropped = 0;
                        acc_padded = 0;
                    }

                    let _ = aec_tx.send((ts, Bytes::from(i16_to_bytes(&cleaned))));
                }
                Err(RecvError::Lagged(n)) => {
                    warn!("aec: lagged {n} mic frames");
                }
                Err(RecvError::Closed) => break,
            }
        }
    }
    info!("aec task exiting");
}
