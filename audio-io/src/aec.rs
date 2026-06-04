//! Acoustic echo cancellation, run inside audio-io (issue #20, 方式B).
//!
//! Both ends of the echo problem live in this process: the near-end is the
//! mic capture (`mic_tx`), the far-end is whatever audio-io itself is playing
//! out the speaker — i.e. the mix of every `/spk` track (TTS on track 0,
//! `play_audio_file` on track 1, ...). Doing the cancellation here, once,
//! lets every consumer (VAD, wake-word-detection) receive the echo-cancelled
//! stream from `/mic` once enabled, with no per-client change — flipping
//! `[aec] enabled` swaps what `/mic` serves (raw vs cancelled).
//!
//! **Where the far-end is tapped matters.** The reference is captured at the
//! point cpal actually *consumes* each playback frame (the output callback in
//! [`crate::playback`]), NOT at the `/spk` WS ingress. Tapping after the
//! playback ring buffer means the reference is on the same wall clock as the
//! sound leaving the speaker, so the only residual delay the filter must model
//! is the small DAC→air→mic→ADC path (tens of ms) — not the ~hundreds of ms
//! the ring can hold. (An earlier design teed the raw WS bytes before the ring;
//! the echo then lagged the reference by the whole ring residency, which
//! exceeded the filter window and killed all cancellation.)
//!
//! Three pieces:
//! * Each playback track's output callback emits its consumed PCM as a 16 kHz
//!   mono frame tagged with the wall-clock consumption time.
//! * [`ReferenceMixer`] sums those per-track frames into one continuous 16 kHz
//!   mono stream (silence when nothing plays — the adaptive filter needs a
//!   gap-free far-end timeline) and carries the play timestamp through.
//! * [`Aec`] is a pure-Rust normalized-LMS (NLMS) adaptive filter. Pure Rust
//!   (no native dep) so it cross-compiles to the mingw Windows target with
//!   zero extra build setup; the `backend` config field leaves room for a
//!   `speex`/`webrtc` swap later. [`aec_task`] time-aligns far to near using
//!   the carried timestamps (capture and consumption share the system clock),
//!   so the echo always lands inside the filter's `[0, filter_length_ms]`
//!   window regardless of when each stream started — no fixed delay hint
//!   needed.

mod mixer;
mod nlms;

use std::collections::VecDeque;

use bytes::Bytes;
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;
use tracing::{debug, info, warn};

use crate::pcm::{bytes_to_i16, epoch_ns, i16_to_bytes};

pub use mixer::ReferenceMixer;
pub use nlms::Aec;

/// Per-frame diagnostics surfaced in the `aec stats` log line. Backend-specific
/// fields are best-effort: a backend that doesn't expose them leaves them at 0.
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

/// A pluggable acoustic echo canceller, selected by `aec.backend`. Both the
/// pure-Rust [`Aec`] and (behind the `speex` feature) the Speex DSP backend
/// implement it, so the rest of the pipeline ([`aec_task`]) is backend-agnostic.
/// `Send` so the canceller can live in the spawned AEC task.
pub trait EchoCanceller: Send {
    /// Cancel the echo of `far` from `near` (one frame each, equal length) and
    /// return the cleaned near frame.
    fn process_frame(&mut self, near: &[i16], far: &[i16]) -> Vec<i16>;
    /// Optional per-frame diagnostics (default: none).
    fn stats(&self) -> CancellerStats {
        CancellerStats::default()
    }
}

/// Sum of squares as f64 (energy), for RMS-based AEC diagnostics.
fn sum_sq(samples: &[i16]) -> f64 {
    samples.iter().map(|&s| (s as f64) * (s as f64)).sum()
}

/// Drains per-track far-end frames into the [`ReferenceMixer`] and publishes
/// one mixed frame every `frame_ms` on `ref_tx`. Runs until both inputs close
/// (services stopped / handle aborted).
pub async fn reference_mixer_task(
    mut ref_in_rx: broadcast::Receiver<(usize, u64, Bytes)>,
    ref_tx: broadcast::Sender<(u64, Bytes)>,
    sample_rate: u32,
    samples_per_frame: usize,
    n_tracks: usize,
    frame_ms: u32,
) {
    let mut mixer = ReferenceMixer::new(sample_rate, samples_per_frame, n_tracks);
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(frame_ms as u64));
    info!(
        samples_per_frame,
        n_tracks, frame_ms, "reference mixer started"
    );
    loop {
        tokio::select! {
            inbound = ref_in_rx.recv() => match inbound {
                Ok((track_id, ts, bytes)) => mixer.push(track_id, ts, &bytes),
                Err(RecvError::Lagged(n)) => {
                    warn!("reference mixer: lagged {n} far-end frames");
                }
                Err(RecvError::Closed) => break,
            },
            _ = interval.tick() => {
                let (ts, frame) = mixer.tick();
                // Before the first playback frame the timeline has no anchor;
                // fall back to wall-clock so silence frames still carry a sane
                // (monotonic-ish) timestamp.
                let ts = ts.unwrap_or_else(epoch_ns);
                // No subscribers (AEC task gone) → send errors, ignored.
                let _ = ref_tx.send((ts, Bytes::from(frame)));
            }
        }
    }
    info!("reference mixer exiting");
}

/// Subscribes to the raw mic (`mic_rx`, near-end) and the mixed far-end
/// (`ref_rx`), pairs them sample-for-sample in arrival order, runs each near
/// frame through [`Aec`], and publishes the echo-cancelled mic on `aec_tx`
/// (served as `/mic` when enabled).
///
/// Pairing is by *count*, not timestamp. Now that the reference is tapped at
/// playback consumption (not the `/spk` ingress), near and far are both
/// real-time 16 kHz streams that start together, so feeding one far sample per
/// near sample lines them up to within the far path's small, *constant* extra
/// latency. That constant offset (plus the acoustic delay) just becomes a tap
/// inside the filter window — the adaptive filter finds it. An earlier version
/// aligned by the carried wall-clock timestamps and dropped far whose timestamp
/// was older than the current near frame; but the far path (consumption tap →
/// mixer's timer → broadcast) is a few frames more latent than the mic path, so
/// its correctly-old frames kept arriving *after* the matching near had already
/// been emitted and were discarded wholesale (far_rms→0, zero cancellation).
/// Counting tolerates that latency. `far_pending` is capped so a burst can't
/// push the echo past the window; underflow (far momentarily behind) feeds
/// silence.
pub async fn aec_task(
    mut mic_rx: broadcast::Receiver<(u64, Bytes)>,
    mut ref_rx: broadcast::Receiver<(u64, Bytes)>,
    aec_tx: broadcast::Sender<(u64, Bytes)>,
    sample_rate: u32,
    mut canceller: Box<dyn EchoCanceller>,
) {
    info!("aec task started");
    // Cap the far backlog (pipeline jitter only): near and far are count-paired
    // 1:1, so the residency here should stay tiny. The actual speaker→mic bulk
    // delay is handled *inside* the Aec by its measured pre-delay, not by
    // holding samples here, so this cap is just a runaway guard — drop the
    // oldest beyond ~250 ms. At steady state it never fires.
    let max_backlog = (sample_rate as usize / 4).max(1);
    let mut far_pending: VecDeque<i16> = VecDeque::new();

    // --- Periodic diagnostics (enable with RUST_LOG=audio_io::aec=info) ---
    // From one ~0.5 s line:
    //   far_rms  — is the reference flowing? (0 ⇒ tap broken / nothing played)
    //   erle_db  — echo removed (near_rms vs resid_rms)
    //   peak_ms  — the delay the filter locked onto; must be < window
    //   dropped  — far dropped by the backlog cap (should stay ~0)
    //   padded   — near samples that found far_pending empty (far behind)
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
                Ok((_ts, bytes)) => {
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
                        // DEBUG level: a per-0.5 s tuning/diagnostic line, off
                        // under the default `info` filter. Enable when needed
                        // with `RUST_LOG=audio_io::aec=debug`. Also gated on
                        // actual playback/speech so an idle session is silent.
                        if near_rms > 30.0 || far_rms > 30.0 {
                            // erle_db is the backend-agnostic comparison metric
                            // (works for nlms and speex alike).
                            let erle_db = 20.0 * (near_rms / resid_rms.max(1.0)).log10();
                            let s = canceller.stats(); // backend-specific extras
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
