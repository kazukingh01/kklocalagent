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

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;
use tracing::{info, warn};

/// NLMS step size. 0 < mu < 2 for stability; 0.3 is a conservative value
/// that converges in a few hundred ms without ringing on a 16 kHz stream.
const NLMS_MU: f32 = 0.3;
/// Per-tap regularization floor. The NLMS denominator is `reg + ||far||²`
/// where `reg = NLMS_REG_PER_TAP * num_taps`. A tiny absolute floor (the old
/// 1e-6) is catastrophic: with a *quiet* far-end (||far||² near zero) under a
/// loud near-end, the step `mu*e/denom` explodes, the weights run to ±inf, and
/// `inf - inf = NaN` poisons the filter — every output sample then casts to 0
/// (`f32::NAN as i16 == 0`), i.e. total silence (observed in the wild). Scaling
/// the floor with the filter length keeps the denominator sane at low energy.
/// 1e-4/tap is tuned to stop the blowup while staying small enough not to
/// throttle convergence once real far-end audio is flowing (≈11 dB ERLE in the
/// offline harness vs ≈7 dB at 1e-3/tap).
const NLMS_REG_PER_TAP: f32 = 1e-4;
/// Leakage factor: weights decay by `(1 - NLMS_LEAK)` each sample. Bleeds off
/// the slow drift that an un-regularized NLMS accumulates, bounding the filter
/// so a bad patch can't grow without limit. Tiny enough not to hurt steady
/// cancellation.
const NLMS_LEAK: f32 = 1e-5;

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn bytes_to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|p| i16::from_le_bytes([p[0], p[1]]))
        .collect()
}

fn i16_to_bytes(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Sums the per-track far-end PCM into one 16 kHz mono s16le stream, carrying
/// the wall-clock *play* time of the emitted samples.
///
/// Track frames arrive asynchronously (one `push` per playback output
/// callback, any length); the mixer buffers per track and emits fixed
/// `samples_per_frame` slots on [`tick`](Self::tick). A slot with no buffered
/// samples for a track contributes silence, so the output is gap-free even
/// when nothing plays — which the adaptive filter relies on for a continuous
/// far-end timeline.
///
/// `timeline_ts` tracks the wall-clock play time of the *next* sample to be
/// emitted (front of the mixed timeline). It is re-anchored from each incoming
/// frame's timestamp (so it stays locked to the playback hardware clock, not a
/// free-running software timer) and advanced one frame per [`tick`]. With a
/// single active track this is exact; with several simultaneous tracks the
/// most-recently-pushed track wins the anchor — fine, since they share one
/// output device clock and so are within a callback of each other.
pub struct ReferenceMixer {
    samples_per_frame: usize,
    sample_period_ns: u64,
    tracks: Vec<VecDeque<i16>>,
    timeline_ts: Option<u64>,
}

impl ReferenceMixer {
    pub fn new(sample_rate: u32, samples_per_frame: usize, n_tracks: usize) -> Self {
        Self {
            samples_per_frame,
            sample_period_ns: 1_000_000_000 / sample_rate.max(1) as u64,
            tracks: (0..n_tracks).map(|_| VecDeque::new()).collect(),
            timeline_ts: None,
        }
    }

    /// Append a track's incoming s16le bytes. `last_sample_ts` is the
    /// wall-clock play time of the frame's *last* sample. Out-of-range track
    /// ids are ignored (defensive — the playback tap already supplies a valid
    /// track id).
    pub fn push(&mut self, track_id: usize, last_sample_ts: u64, bytes: &[u8]) {
        if let Some(buf) = self.tracks.get_mut(track_id) {
            buf.extend(bytes_to_i16(bytes));
            // Front-of-buffer play time = last sample's time minus the span of
            // the samples queued ahead of it. Re-anchors the shared timeline.
            let ahead = (buf.len() as u64).saturating_sub(1);
            self.timeline_ts = Some(last_sample_ts.saturating_sub(ahead * self.sample_period_ns));
        }
    }

    /// Emit one mixed frame (`samples_per_frame` samples, s16le) plus the
    /// play timestamp of its first sample (`None` until the first push). Pulls
    /// up to one frame from each track buffer (missing samples = silence) and
    /// sums with saturation so two simultaneous tracks never wrap around.
    pub fn tick(&mut self) -> (Option<u64>, Vec<u8>) {
        let ts = self.timeline_ts;
        let n = self.samples_per_frame;
        let mut acc = vec![0i32; n];
        for buf in &mut self.tracks {
            for slot in acc.iter_mut() {
                if let Some(s) = buf.pop_front() {
                    *slot += s as i32;
                }
            }
        }
        let mixed: Vec<i16> = acc
            .into_iter()
            .map(|v| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
            .collect();
        // Advance the timeline by the frame we just emitted so silence ticks
        // keep the clock moving; the next push re-anchors it precisely.
        if let Some(t) = self.timeline_ts.as_mut() {
            *t = t.saturating_add(n as u64 * self.sample_period_ns);
        }
        (ts, i16_to_bytes(&mixed))
    }
}

/// Normalized-LMS adaptive echo canceller operating on 16 kHz mono s16le.
///
/// `process_frame(near, far)` returns `near` with the linear echo of `far`
/// subtracted. The adaptive filter spans `num_taps` samples starting at **zero
/// delay**, so it models the echo wherever it actually lands within that window
/// — taps ahead of the true echo simply adapt toward zero. This is deliberate:
/// an earlier version pre-shifted the far-end by a fixed `initial_delay_ms`
/// bulk delay, which silently killed all cancellation whenever the real echo
/// delay was *smaller* than that hint (e.g. realtime-paced playback, where the
/// speaker→mic delay is only tens of ms). Covering `[0, num_taps]` removes that
/// fragile assumption; `initial_delay_ms` now just *extends* the window to also
/// reach larger bulk delays. The filter adapts continuously, so steady
/// speaker→mic echo is cancelled while uncorrelated near-end speech passes
/// through.
pub struct Aec {
    weights: VecDeque<f32>,
    /// Filter input history, newest at the front, paired index-for-index
    /// with `weights`.
    far_hist: VecDeque<f32>,
    /// Running sum of squares of `far_hist`, maintained incrementally for the
    /// NLMS normalization denominator.
    energy: f32,
    num_taps: usize,
    /// Regularization constant in the NLMS denominator (`= NLMS_REG_PER_TAP *
    /// num_taps`), precomputed so the per-sample hot loop stays a single add.
    reg: f32,
}

impl Aec {
    /// The filter window covers `initial_delay_ms + filter_length_ms` from zero
    /// delay: `filter_length_ms` is the echo tail to model, `initial_delay_ms`
    /// is extra head-room for a larger bulk transport delay. Either alone works;
    /// their sum is the maximum echo delay (in ms) the filter can still cancel.
    pub fn new(sample_rate: u32, filter_length_ms: u32, initial_delay_ms: u32) -> Self {
        let window_ms = (initial_delay_ms + filter_length_ms) as usize;
        let num_taps = ((sample_rate as usize * window_ms) / 1000).max(1);
        Self {
            weights: VecDeque::from(vec![0.0; num_taps]),
            far_hist: VecDeque::from(vec![0.0; num_taps]),
            energy: 0.0,
            num_taps,
            reg: NLMS_REG_PER_TAP * num_taps as f32,
        }
    }

    /// Zero the adaptive state. Called when the filter has gone non-finite
    /// (numerical blowup) so it relearns from scratch rather than emitting
    /// silence forever.
    fn reset_filter(&mut self) {
        for w in self.weights.iter_mut() {
            *w = 0.0;
        }
        for h in self.far_hist.iter_mut() {
            *h = 0.0;
        }
        self.energy = 0.0;
    }

    fn push_far(&mut self, x: f32) {
        // Maintain far_hist as a fixed-length newest-at-front window and keep
        // `energy` in sync without re-summing the whole window each sample.
        if let Some(old) = self.far_hist.pop_back() {
            self.energy -= old * old;
        }
        self.far_hist.push_front(x);
        self.energy += x * x;
        if self.energy < 0.0 {
            // Guard against f32 drift turning the running sum slightly
            // negative after many subtractions.
            self.energy = 0.0;
        }
    }

    /// near/far must be the same length (one 20 ms frame each).
    pub fn process_frame(&mut self, near: &[i16], far: &[i16]) -> Vec<i16> {
        let mut out = Vec::with_capacity(near.len());
        for (i, &d) in near.iter().enumerate() {
            let x = far.get(i).copied().unwrap_or(0) as f32 / 32768.0;
            self.push_far(x);

            // Estimated echo = w · far_hist.
            let y: f32 = self
                .weights
                .iter()
                .zip(self.far_hist.iter())
                .map(|(w, h)| w * h)
                .sum();
            let d_f = d as f32 / 32768.0;
            let e = d_f - y;

            // Numerical safety net: if the estimate has gone non-finite, the
            // filter has diverged — reset it and pass the raw near sample
            // through this frame rather than casting NaN to a silent 0.
            if !e.is_finite() {
                self.reset_filter();
                out.push(d);
                continue;
            }

            // NLMS weight update with leakage:
            //   w := (1 - leak) * w + mu * e * far_hist / (reg + ||far_hist||²)
            // `reg` (scaled with the filter length) keeps the step bounded when
            // the far-end energy dips toward zero; leakage bleeds off drift.
            let g = NLMS_MU * e / (self.reg + self.energy);
            for (w, h) in self.weights.iter_mut().zip(self.far_hist.iter()) {
                *w = (1.0 - NLMS_LEAK) * *w + g * h;
            }

            out.push((e.clamp(-1.0, 1.0) * 32767.0) as i16);
        }
        out
    }

    pub fn num_taps(&self) -> usize {
        self.num_taps
    }

    /// Diagnostics: `(L2 norm of all weights, index of the largest-magnitude
    /// tap, that tap's value)`. The peak tap is the filter's current estimate
    /// of the dominant echo delay *in samples* — if the filter has locked on,
    /// `peak_tap * 1000 / sample_rate` is roughly the speaker→mic delay in ms.
    /// A peak pinned at the last tap (or a near-zero L2 that never grows) means
    /// the true echo sits at/after the window edge and the filter can't reach
    /// it.
    pub fn weight_stats(&self) -> (f32, usize, f32) {
        let mut l2 = 0.0f32;
        let mut peak_idx = 0usize;
        let mut peak_abs = 0.0f32;
        for (i, w) in self.weights.iter().enumerate() {
            l2 += w * w;
            if w.abs() > peak_abs {
                peak_abs = w.abs();
                peak_idx = i;
            }
        }
        (l2.sqrt(), peak_idx, peak_abs)
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
                let ts = ts.unwrap_or_else(now_ns);
                // No subscribers (AEC task gone) → send errors, ignored.
                let _ = ref_tx.send((ts, Bytes::from(frame)));
            }
        }
    }
    info!("reference mixer exiting");
}

/// Time-aligns the far-end reference to a near (mic) frame and returns one far
/// sample per near sample, on the near frame's timeline.
///
/// Both timestamps are wall-clock (capture stamps near at the input callback,
/// the playback tap stamps far at the output callback), so we feed the far
/// sample that was *played* at the same instant each near sample was
/// *captured*. The real echo — far played `d` ms earlier (DAC→air→mic) — then
/// lands at tap `d` inside the filter window, wherever `d` happens to be, so
/// the constant offset between when each stream started no longer matters.
///
/// `far_front_ts` is the play time of `far_buf.front()`. Far older than the
/// near frame start is dropped (its echo is already past); if far lags (front
/// newer than the near start) the unmatched head is padded with silence.
/// Returns `(far, dropped, padded)`: the aligned far samples, how many stale
/// far samples were dropped to catch up to the near frame, and how many head
/// samples were silence-padded because far was lagging. The two counts are
/// diagnostics — at steady state both should be ~0 (far tracks near exactly).
fn align_far(
    far_buf: &mut VecDeque<i16>,
    far_front_ts: &mut Option<u64>,
    near_ts: u64,
    n: usize,
    period_ns: u64,
) -> (Vec<i16>, usize, usize) {
    let mut far = Vec::with_capacity(n);
    let near_first_ts = near_ts.saturating_sub((n as u64).saturating_sub(1) * period_ns);
    let mut dropped = 0usize;
    match *far_front_ts {
        // No far timeline yet (nothing has played) → reference is silence.
        None => {
            far.resize(n, 0);
            (far, 0, n)
        }
        Some(mut fts) => {
            // Drop far strictly older than this near frame's start: its echo
            // belongs to already-processed near frames.
            while fts + period_ns <= near_first_ts {
                if far_buf.pop_front().is_none() {
                    fts = near_first_ts;
                    break;
                }
                fts += period_ns;
                dropped += 1;
            }
            // Far lagging: front is newer than the near start, so the head of
            // this frame has no reference yet → silence-pad it.
            let lead = if fts > near_first_ts {
                ((fts - near_first_ts) / period_ns) as usize
            } else {
                0
            };
            let pad = lead.min(n);
            far.resize(pad, 0);
            for _ in pad..n {
                far.push(far_buf.pop_front().unwrap_or(0));
            }
            let consumed = (n - pad) as u64;
            *far_front_ts = if far_buf.is_empty() {
                None
            } else {
                Some(fts + consumed * period_ns)
            };
            (far, dropped, pad)
        }
    }
}

/// Subscribes to the raw mic (`mic_rx`, near-end) and the mixed far-end
/// (`ref_rx`), time-aligns the two streams by their carried wall-clock
/// timestamps (see [`align_far`]), runs each near frame through [`Aec`], and
/// publishes the echo-cancelled mic on `aec_tx` (served as `/mic` when
/// enabled). Near frames drive the clock; far samples are buffered and aligned
/// to each near frame, padding with silence if the far-end momentarily lags.
pub async fn aec_task(
    mut mic_rx: broadcast::Receiver<(u64, Bytes)>,
    mut ref_rx: broadcast::Receiver<(u64, Bytes)>,
    aec_tx: broadcast::Sender<(u64, Bytes)>,
    sample_rate: u32,
    mut aec: Aec,
) {
    info!(num_taps = aec.num_taps(), "aec task started");
    let period_ns = 1_000_000_000 / sample_rate.max(1) as u64;
    let mut far_buf: VecDeque<i16> = VecDeque::new();
    // Play time of far_buf.front(); None when the buffer is empty / unanchored.
    let mut far_front_ts: Option<u64> = None;

    // --- Periodic diagnostics (enable with RUST_LOG=audio_io::aec=info) ---
    // Accumulated over ~1 s so an operator can answer, from one log line:
    //   far_rms   — is the reference actually flowing? (0 ⇒ tap broken/idle)
    //   erle_db   — how much echo is being removed (near_rms vs resid_rms)
    //   peak_ms   — the delay the filter locked onto; must be < window
    //   drop/pad  — alignment health (large/persistent ⇒ skew or clock drift)
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
                Ok((ts, bytes)) => {
                    // Re-anchor the front timestamp to the playback clock every
                    // frame: this frame's first sample plays at `ts` and sits
                    // `far_buf.len()` samples behind the current front, so the
                    // front plays at `ts - len*period`. Re-anchoring (vs only
                    // when empty) tracks any drift between the playback and
                    // capture hardware clocks instead of accumulating it.
                    let behind = far_buf.len() as u64;
                    far_front_ts = Some(ts.saturating_sub(behind * period_ns));
                    far_buf.extend(bytes_to_i16(&bytes));
                }
                Err(RecvError::Lagged(n)) => {
                    // A gap breaks timeline continuity; reset so the next frame
                    // re-anchors rather than splicing across the hole.
                    warn!("aec: lagged {n} far-end frames");
                    far_buf.clear();
                    far_front_ts = None;
                }
                Err(RecvError::Closed) => break,
            },
            near = mic_rx.recv() => match near {
                Ok((ts, bytes)) => {
                    let near = bytes_to_i16(&bytes);
                    let (far, dropped, padded) =
                        align_far(&mut far_buf, &mut far_front_ts, ts, near.len(), period_ns);
                    let cleaned = aec.process_frame(&near, &far);

                    near_sq += sum_sq(&near);
                    far_sq += sum_sq(&far);
                    resid_sq += sum_sq(&cleaned);
                    acc_samples += near.len();
                    acc_dropped += dropped;
                    acc_padded += padded;
                    if acc_samples >= log_every {
                        let denom = acc_samples as f64;
                        let near_rms = (near_sq / denom).sqrt();
                        let far_rms = (far_sq / denom).sqrt();
                        let resid_rms = (resid_sq / denom).sqrt();
                        // Only log when something is actually playing/speaking,
                        // so an idle session doesn't spam a line every 0.5 s.
                        if near_rms > 30.0 || far_rms > 30.0 {
                            let erle_db = 20.0 * (near_rms / resid_rms.max(1.0)).log10();
                            let (w_l2, peak_tap, peak_val) = aec.weight_stats();
                            let peak_ms = peak_tap as u32 * 1000 / sample_rate.max(1);
                            info!(
                                near_rms = near_rms as i64,
                                far_rms = far_rms as i64,
                                resid_rms = resid_rms as i64,
                                erle_db = format!("{erle_db:.1}"),
                                peak_tap,
                                peak_ms,
                                peak_val = format!("{peak_val:.3}"),
                                w_l2 = format!("{w_l2:.3}"),
                                far_buf = far_buf.len(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[i16]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
        (sum / samples.len() as f64).sqrt()
    }

    // 16 kHz sample period in ns; used by the timestamp/alignment tests.
    const P: u64 = 62_500;

    #[test]
    fn mixer_passthrough_single_track() {
        let mut m = ReferenceMixer::new(16000, 4, 2);
        let frame = i16_to_bytes(&[100, -200, 300, -400]);
        m.push(0, 0, &frame);
        assert_eq!(bytes_to_i16(&m.tick().1), vec![100, -200, 300, -400]);
    }

    #[test]
    fn mixer_sums_two_tracks() {
        let mut m = ReferenceMixer::new(16000, 4, 2);
        m.push(0, 0, &i16_to_bytes(&[100, 100, 100, 100]));
        m.push(1, 0, &i16_to_bytes(&[50, -50, 50, -50]));
        assert_eq!(bytes_to_i16(&m.tick().1), vec![150, 50, 150, 50]);
    }

    #[test]
    fn mixer_saturates_instead_of_wrapping() {
        let mut m = ReferenceMixer::new(16000, 2, 2);
        m.push(0, 0, &i16_to_bytes(&[30000, -30000]));
        m.push(1, 0, &i16_to_bytes(&[30000, -30000]));
        // 60000 / -60000 must clamp to i16 range, not wrap.
        assert_eq!(bytes_to_i16(&m.tick().1), vec![i16::MAX, i16::MIN]);
    }

    #[test]
    fn mixer_emits_silence_when_idle() {
        let mut m = ReferenceMixer::new(16000, 4, 2);
        let (ts, frame) = m.tick();
        assert_eq!(ts, None); // no anchor before the first push
        assert_eq!(bytes_to_i16(&frame), vec![0, 0, 0, 0]);
    }

    #[test]
    fn mixer_carries_residual_across_ticks() {
        // Push 6 samples into a 4-wide mixer: first tick takes 4, second takes
        // the remaining 2 then pads with silence.
        let mut m = ReferenceMixer::new(16000, 4, 1);
        m.push(0, 0, &i16_to_bytes(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(bytes_to_i16(&m.tick().1), vec![1, 2, 3, 4]);
        assert_eq!(bytes_to_i16(&m.tick().1), vec![5, 6, 0, 0]);
    }

    #[test]
    fn mixer_timestamp_anchors_to_play_time() {
        // Last sample played at t = 1_000_000 ns; the front (first) sample
        // played 3 periods earlier. The emit ts is that front time.
        let mut m = ReferenceMixer::new(16000, 4, 1);
        let last = 1_000_000u64;
        m.push(0, last, &i16_to_bytes(&[1, 2, 3, 4]));
        assert_eq!(m.tick().0, Some(last - 3 * P));
        // The now-idle tick advances the timeline by one frame (4 samples).
        assert_eq!(m.tick().0, Some(last - 3 * P + 4 * P));
    }

    #[test]
    fn align_far_aligns_by_timestamp() {
        // Far front plays at t=0; near frame's first sample is at t=0 too →
        // the front far samples are taken 1:1 with no skew.
        let mut far_buf: VecDeque<i16> = VecDeque::from(vec![10, 20, 30, 40]);
        let mut fts = Some(0u64);
        // n=2, near_ts = P → near_first = 0.
        let (far, dropped, padded) = align_far(&mut far_buf, &mut fts, P, 2, P);
        assert_eq!(far, vec![10, 20]);
        assert_eq!((dropped, padded), (0, 0));
        assert_eq!(fts, Some(2 * P)); // remaining [30,40] front now at t=2P
    }

    #[test]
    fn align_far_drops_stale_far() {
        // Far front at t=0 but the near frame doesn't start until t=3P, so the
        // three oldest far samples (their echo is already past) are dropped.
        let mut far_buf: VecDeque<i16> = VecDeque::from(vec![1, 2, 3, 4, 5, 6]);
        let mut fts = Some(0u64);
        let (far, dropped, padded) = align_far(&mut far_buf, &mut fts, 4 * P, 2, P); // near_first = 3P
        assert_eq!(far, vec![4, 5]);
        assert_eq!((dropped, padded), (3, 0)); // three stale samples dropped
    }

    #[test]
    fn align_far_pads_when_far_lags() {
        // Far front is newer (t=2P) than the near frame start (t=0): the head
        // of the frame has no reference yet and must be silence-padded.
        let mut far_buf: VecDeque<i16> = VecDeque::from(vec![7, 8]);
        let mut fts = Some(2 * P);
        let (far, dropped, padded) = align_far(&mut far_buf, &mut fts, 3 * P, 4, P); // near_first = 0
        assert_eq!(far, vec![0, 0, 7, 8]);
        assert_eq!((dropped, padded), (0, 2)); // two head samples silence-padded
    }

    // Deterministic pseudo-random far-end (LCG) so the test needs no rng dep.
    fn lcg(seed: &mut u64) -> i16 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 48) as i16) / 4 // bounded amplitude
    }

    #[test]
    fn aec_cancels_pure_echo() {
        // far-end = noise; near = far delayed by D samples and attenuated
        // (a pure linear echo, no near-end voice). After the filter adapts,
        // the residual RMS should fall well below the echo RMS (high ERLE).
        let rate = 16000;
        let delay = 50; // samples of echo path delay
        let mut aec = Aec::new(rate, 80, delay as u32 * 1000 / rate); // ~3ms delay hint
        let mut seed = 0x1234_5678u64;

        let frame_len = 320;
        let mut last_echo_rms = 0.0;
        let mut last_resid_rms = 0.0;
        // Carry the echo delay across frame boundaries.
        let mut echo_delay_line: VecDeque<i16> = VecDeque::from(vec![0i16; delay]);

        for _ in 0..200 {
            let far: Vec<i16> = (0..frame_len).map(|_| lcg(&mut seed)).collect();
            // Build the echo: near[i] = 0.5 * far_delayed[i].
            let mut near = Vec::with_capacity(frame_len);
            for &f in &far {
                echo_delay_line.push_back(f);
                let delayed = echo_delay_line.pop_front().unwrap_or(0);
                near.push((delayed as f32 * 0.5) as i16);
            }
            let resid = aec.process_frame(&near, &far);
            last_echo_rms = rms(&near);
            last_resid_rms = rms(&resid);
        }
        assert!(
            last_echo_rms > 50.0,
            "echo too quiet to test: {last_echo_rms}"
        );
        let erle = 20.0 * (last_echo_rms / last_resid_rms.max(1.0)).log10();
        assert!(
            erle > 12.0,
            "expected >12 dB echo reduction, got {erle:.1} dB (echo={last_echo_rms:.0}, resid={last_resid_rms:.0})"
        );
    }

    #[test]
    fn aec_cancels_when_echo_delay_far_below_hint() {
        // Regression for the "echo passes through untouched" bug: the real
        // echo delay (here ~1 ms) is much SMALLER than initial_delay_ms (120
        // ms). The old code pre-shifted the far-end by the full hint, putting
        // the echo *before* the filter window → zero cancellation. With the
        // window anchored at delay 0 the filter must still find and cancel it.
        let rate = 16000;
        let delay = 16; // ~1 ms echo, far below the 40 ms hint below
        let mut aec = Aec::new(rate, 60, 40); // window = 100 ms, hint ≫ echo
        let mut seed = 0xfeed_face_u64;
        let frame_len = 320;
        let mut echo_delay_line: VecDeque<i16> = VecDeque::from(vec![0i16; delay]);
        let mut last_echo_rms = 0.0;
        let mut last_resid_rms = 0.0;

        for _ in 0..200 {
            let far: Vec<i16> = (0..frame_len).map(|_| lcg(&mut seed)).collect();
            let mut near = Vec::with_capacity(frame_len);
            for &f in &far {
                echo_delay_line.push_back(f);
                let delayed = echo_delay_line.pop_front().unwrap_or(0);
                near.push((delayed as f32 * 0.5) as i16);
            }
            let resid = aec.process_frame(&near, &far);
            last_echo_rms = rms(&near);
            last_resid_rms = rms(&resid);
        }
        let erle = 20.0 * (last_echo_rms / last_resid_rms.max(1.0)).log10();
        assert!(
            erle > 12.0,
            "echo not cancelled when delay ≪ hint: {erle:.1} dB (echo={last_echo_rms:.0}, resid={last_resid_rms:.0})"
        );
    }

    #[test]
    fn aec_preserves_uncorrelated_near_voice() {
        // far-end = noise echoed into near, PLUS a near-end tone uncorrelated
        // with the far-end. The tone (the human voice we must keep) should
        // survive cancellation.
        let rate = 16000;
        let delay = 40;
        let mut aec = Aec::new(rate, 80, delay as u32 * 1000 / rate);
        let mut seed = 0xdead_beefu64;
        let frame_len = 320;
        let mut echo_delay_line: VecDeque<i16> = VecDeque::from(vec![0i16; delay]);
        let mut t: f32 = 0.0;
        let mut last_voice_only_rms = 0.0;
        let mut last_resid_rms = 0.0;

        for _ in 0..200 {
            let far: Vec<i16> = (0..frame_len).map(|_| lcg(&mut seed)).collect();
            let mut near = Vec::with_capacity(frame_len);
            let mut voice_only = Vec::with_capacity(frame_len);
            for &f in &far {
                echo_delay_line.push_back(f);
                let delayed = echo_delay_line.pop_front().unwrap_or(0);
                let voice = (3000.0 * (t * 0.05).sin()) as i16; // ~127 Hz tone
                t += 1.0;
                voice_only.push(voice);
                near.push((delayed as f32 * 0.5) as i16 + voice);
            }
            let resid = aec.process_frame(&near, &far);
            last_voice_only_rms = rms(&voice_only);
            last_resid_rms = rms(&resid);
        }
        // The residual should retain most of the near-end voice energy — i.e.
        // be on the same order as the voice alone, not driven to ~0.
        assert!(
            last_resid_rms > last_voice_only_rms * 0.5,
            "near voice was suppressed: resid={last_resid_rms:.0}, voice={last_voice_only_rms:.0}"
        );
    }

    #[test]
    fn aec_does_not_collapse_on_quiet_far_loud_near() {
        // Regression for the divergence bug the user hit (output went fully
        // zero — `xxd` showed all 0x00). The trigger is a *quiet* far-end
        // (tiny but non-zero ||far||²) under a *loud* near-end: the old
        // absolute 1e-6 denominator floor let `mu*e/denom` explode, the
        // weights ran to ±inf, and `inf - inf = NaN` cast to 0 every sample.
        // Here there's almost no real echo (far is near-silent), so a correct
        // filter must just pass the loud near through — not annihilate it.
        let rate = 16000;
        let mut aec = Aec::new(rate, 60, 3);
        let frame_len = 320;
        let mut t: f32 = 0.0;
        let mut resid_acc: Vec<i16> = Vec::new();
        let mut near_acc: Vec<i16> = Vec::new();

        for _ in 0..200 {
            // Quiet far (constant 30 ≈ -60 dBFS): ||far||² stays tiny but
            // non-zero — the pathological denominator regime.
            let far = vec![30i16; frame_len];
            // Loud near-end voice, uncorrelated with the far-end.
            let near: Vec<i16> = (0..frame_len)
                .map(|_| {
                    t += 1.0;
                    (20000.0 * (t * 0.05).sin()) as i16
                })
                .collect();
            let resid = aec.process_frame(&near, &far);
            resid_acc.extend_from_slice(&resid);
            near_acc.extend_from_slice(&near);
        }

        // After settling, the residual must still carry the near voice — i.e.
        // the filter neither diverged to NaN→0 nor over-suppressed.
        let tail = resid_acc.len() / 2;
        let resid_rms = rms(&resid_acc[tail..]);
        let near_rms = rms(&near_acc[tail..]);
        assert!(
            resid_rms > near_rms * 0.5,
            "filter collapsed (NaN→silence regression?): resid={resid_rms:.0}, near={near_rms:.0}"
        );
    }
}
