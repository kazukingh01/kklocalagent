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
use tracing::{debug, info, warn};

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

// --- Residual echo suppressor (post-NLP), with adaptive ERL ---
// The linear NLMS alone reaches only ~10-15 dB ERLE on a real speaker→mic path
// (speaker nonlinearity, long reverb tail), which leaves the echo audible. A
// Wiener-style gain on the residual removes the rest. Rather than a fixed,
// hand-tuned "how much echo is left" knob (which had to be re-tuned per room /
// speaker / volume), the suppressor *measures* the leftover-echo ratio from the
// signal itself (see `erl` in [`Aec`]) so it adapts to the environment.
//
/// Over-subtraction applied to the measured residual-echo ratio. >1 digs a bit
/// past the estimate so echo-only output is firmly inaudible; the value trades
/// echo depth against how much an overlapping (double-talk) talker is dented.
const NLP_OVERSUB: f64 = 2.0;
/// Initial residual-echo ratio (residual energy / echo-estimate energy) before
/// anything has been measured.
const NLP_ERL_INIT: f32 = 0.1;
/// ERL tracker: fast pull toward a *lower* observed ratio (echo-only stretches
/// reveal the true leftover-echo floor) and a slow drift back up (so it can
/// recover when the path genuinely worsens, and double-talk — which inflates the
/// ratio — can't corrupt the estimate). A minimum-statistics style follower.
const NLP_ERL_TRACK_DOWN: f32 = 0.2;
const NLP_ERL_TRACK_UP: f32 = 0.002;
/// Clamp range for the ERL estimate.
const NLP_ERL_MIN: f32 = 0.003;
const NLP_ERL_MAX: f32 = 1.0;
/// Lowest gain the suppressor applies, so echo-only output is strongly
/// attenuated but never fully muted (keeps a natural floor; ≈ -30 dB).
const NLP_GAIN_FLOOR: f32 = 0.03;
/// Far-end power (mean x² with x in [-1,1]) below which nothing is playing, so
/// there is no echo to suppress and the gain is released to unity.
const NLP_FAR_FLOOR: f64 = 1e-6;
/// Per-frame smoothing for the suppression gain. Attack (toward more
/// suppression) is brisk so echo is caught within a couple of 20 ms frames.
const NLP_ATTACK: f32 = 0.6;
/// Release back toward unity is SLOW while the far-end is still active, so a
/// brief residual-echo spike or a gap between far-end words can't pop the gain
/// back up and leak echo mid-playback; it only fully releases once playback
/// stops (then [`NLP_RELEASE_IDLE`] restores normal mic capture promptly).
const NLP_RELEASE_FAR_ACTIVE: f32 = 0.12;
const NLP_RELEASE_IDLE: f32 = 0.7;

// --- Bulk-delay estimator (envelope cross-correlation) ---
// The echo arrives some bulk delay after the reference (playback/device
// buffering + acoustic flight + capture buffering). Rather than make the
// adaptive filter long enough to *span* that delay — which differs per machine
// and forces a hand-set window — we estimate the delay and pre-delay the
// reference so the echo lands at the *start* of a compact, fixed-length filter
// that then only has to model the room reverb tail. Works for any bulk delay up
// to `DELAY_MAX_MS` with the same small filter, so no per-environment knob.
//
/// Envelope decimation factor (16 kHz → 2 kHz): delay needs coarse timing, not
/// fine phase, so correlating decimated |signal| envelopes is cheap and robust.
const DELAY_DECIM: usize = 8;
/// Largest bulk delay searched/handled (ms). Generous enough for deep playback
/// buffers on any host.
const DELAY_MAX_MS: u32 = 500;
/// Correlation window (ms of recent audio compared at each lag).
const DELAY_WINDOW_MS: u32 = 256;
/// How often the delay is re-estimated (ms).
const DELAY_ESTIMATE_MS: u32 = 250;
/// Minimum normalized correlation to trust an estimate (rejects double-talk /
/// no-echo frames, which don't correlate cleanly).
const DELAY_CONFIDENCE: f32 = 0.6;
/// Mean-envelope floor (|x| in [0,1]) the far-end must exceed in the
/// correlation window before a delay is estimated at all — quiet/transition
/// frames correlate to noise and produced spurious jumps.
const DELAY_ENV_FLOOR: f32 = 0.01;
/// Consecutive agreeing confident estimates required before the delay is
/// (re)locked. Stops a single spurious frame from moving the pre-delay.
const DELAY_LOCK_COUNT: u32 = 3;
/// How close (ms) successive estimates must be to count as "agreeing". Loose
/// enough that a few-ms jitter still locks (the filter head-room absorbs the
/// slack), tight enough to reject the wild spurious jumps.
const DELAY_AGREE_MS: u32 = 16;
/// Once locked, only re-lock when a *new* stable estimate differs by more than
/// this. Small drift is left for the filter's head-room to absorb, so the
/// expensive filter reset happens at most a handful of times per session.
const DELAY_RELOCK_MS: u32 = 64;
/// Head-room left ahead of the estimated echo inside the filter, so estimation
/// error and frame-level jitter still land within the taps.
const DELAY_HEAD_MARGIN_MS: u32 = 32;

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

fn push_capped(q: &mut VecDeque<f32>, v: f32, cap: usize) {
    if q.len() >= cap {
        q.pop_front();
    }
    q.push_back(v);
}

/// Envelope cross-correlation bulk-delay estimator (see the `DELAY_*` consts).
/// Fed every (far, near) sample, it keeps decimated `|·|` envelopes and
/// periodically reports the lag (in full-rate samples) at which `near` best
/// matches a delayed `far` — the speaker→mic bulk delay — but only when the
/// normalized correlation is confident, so double-talk / no-echo frames are
/// ignored rather than producing a bogus delay.
struct DelayEstimator {
    far_env: VecDeque<f32>,
    near_env: VecDeque<f32>,
    cap: usize,
    max_lag: usize,
    window: usize,
    far_acc: f32,
    near_acc: f32,
    acc_n: usize,
    interval: usize,
    since: usize,
    agree: usize,  // estimates within this many samples count as agreeing
    relock: usize, // re-lock threshold (full-rate samples)
    // Lock state: only a delay confirmed `DELAY_LOCK_COUNT` times in a row is
    // applied, and once applied it is held until a new stable estimate differs
    // by more than `relock` — so the filter is reset at most a few times, not
    // every estimate (the bug that thrashed convergence).
    locked: Option<usize>,
    cand: usize,
    cand_n: u32,
}

impl DelayEstimator {
    fn new(sample_rate: u32) -> Self {
        let dec_rate = (sample_rate as usize / DELAY_DECIM).max(1);
        let max_lag = (dec_rate * DELAY_MAX_MS as usize / 1000).max(1);
        let window = (dec_rate * DELAY_WINDOW_MS as usize / 1000).max(1);
        let cap = max_lag + window + 1;
        Self {
            far_env: VecDeque::new(),
            near_env: VecDeque::new(),
            cap,
            max_lag,
            window,
            far_acc: 0.0,
            near_acc: 0.0,
            acc_n: 0,
            interval: (sample_rate as usize * DELAY_ESTIMATE_MS as usize / 1000).max(1),
            since: 0,
            agree: (sample_rate as usize * DELAY_AGREE_MS as usize / 1000).max(DELAY_DECIM),
            relock: (sample_rate as usize * DELAY_RELOCK_MS as usize / 1000).max(1),
            locked: None,
            cand: 0,
            cand_n: 0,
        }
    }

    /// Feed one full-rate (far, near) pair. Returns a delay (full-rate samples)
    /// only when it (re)locks — i.e. rarely — so the caller resets the filter
    /// at most a handful of times.
    fn push(&mut self, far: f32, near: f32) -> Option<usize> {
        self.far_acc += far.abs();
        self.near_acc += near.abs();
        self.acc_n += 1;
        if self.acc_n >= DELAY_DECIM {
            let inv = 1.0 / self.acc_n as f32;
            push_capped(&mut self.far_env, self.far_acc * inv, self.cap);
            push_capped(&mut self.near_env, self.near_acc * inv, self.cap);
            self.far_acc = 0.0;
            self.near_acc = 0.0;
            self.acc_n = 0;
        }
        self.since += 1;
        if self.since >= self.interval {
            self.since = 0;
            return self.estimate_now();
        }
        None
    }

    /// Raw confident estimate this tick (full-rate samples), or None when the
    /// far-end is too quiet or the correlation isn't confident.
    fn raw_estimate(&self) -> Option<usize> {
        if self.near_env.len() < self.window + self.max_lag {
            return None;
        }
        let near: Vec<f32> = self.near_env.iter().copied().collect();
        let far: Vec<f32> = self.far_env.iter().copied().collect();
        let nlen = near.len();
        let n0 = nlen - self.window; // start of the most-recent `window` of near
        let mut near_e = 0.0f32;
        let mut far_recent = 0.0f32; // mean |far| over the freshest window (gate)
        for j in 0..self.window {
            near_e += near[n0 + j] * near[n0 + j];
            far_recent += far[n0 + j];
        }
        // Gate: don't estimate unless the far-end is actually playing.
        if near_e <= 1e-9 || far_recent / self.window as f32 <= DELAY_ENV_FLOOR {
            return None;
        }
        let mut best_lag = 0usize;
        let mut best_corr = 0.0f32;
        for lag in 0..=self.max_lag.min(n0) {
            let f0 = n0 - lag;
            let mut dot = 0.0f32;
            let mut far_e = 0.0f32;
            for j in 0..self.window {
                let fv = far[f0 + j];
                dot += near[n0 + j] * fv;
                far_e += fv * fv;
            }
            if far_e <= 1e-9 {
                continue;
            }
            let corr = dot / (near_e * far_e).sqrt();
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
        if best_corr >= DELAY_CONFIDENCE {
            Some(best_lag * DELAY_DECIM)
        } else {
            None
        }
    }

    fn estimate_now(&mut self) -> Option<usize> {
        let raw = self.raw_estimate()?;
        // Build confidence: agreeing values (within `agree`) must recur
        // `DELAY_LOCK_COUNT` times before they can move the pre-delay.
        if raw.abs_diff(self.cand) <= self.agree {
            self.cand_n += 1;
        } else {
            self.cand = raw;
            self.cand_n = 1;
        }
        if self.cand_n < DELAY_LOCK_COUNT {
            return None;
        }
        // Confirmed. Lock on first acquisition, or re-lock only on a large move.
        match self.locked {
            None => {
                self.locked = Some(self.cand);
                Some(self.cand)
            }
            Some(cur) if self.cand.abs_diff(cur) > self.relock => {
                self.locked = Some(self.cand);
                Some(self.cand)
            }
            _ => None,
        }
    }
}

/// Normalized-LMS adaptive echo canceller operating on 16 kHz mono s16le.
///
/// `process_frame(near, far)` returns `near` with the echo of `far` removed in
/// three stages, all self-tuning so there are no per-environment knobs:
///
/// 1. **Bulk-delay compensation** — a [`DelayEstimator`] measures the
///    speaker→mic delay and the reference is pre-delayed by it, so the echo
///    lands at the front of the filter no matter how deep the host's buffers or
///    how distant the mic. The adaptive filter therefore only needs to be long
///    enough for the room *reverb tail* (`num_taps`), not the whole transport
///    delay.
/// 2. **Linear NLMS** — a `num_taps` adaptive filter subtracts the linear echo;
///    it adapts continuously so a steady echo path is cancelled while
///    uncorrelated near-end speech passes through.
/// 3. **Adaptive residual suppressor** — a Wiener-style gain (see the `NLP_*`
///    constants) removes the leftover echo the linear stage can't reach. Its
///    strength is *measured* from the signal (the `erl` ratio learned during
///    echo-only stretches), so it adapts to room / speaker / volume instead of
///    needing a hand-set level.
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
    /// Smoothed residual-echo-suppressor gain (1.0 = pass-through). Updated once
    /// per frame from the frame's far / echo-estimate / residual energies and
    /// applied to every residual sample of that frame.
    nlp_g: f32,
    /// Adaptively measured residual-echo ratio (residual energy / echo-estimate
    /// energy during echo-only stretches). Drives the suppressor instead of a
    /// hand-set strength, so it tracks the room / speaker / volume on its own.
    erl: f32,
    sample_rate: u32,
    /// Estimates the bulk speaker→mic delay so the reference can be pre-delayed
    /// onto the front of the compact filter (no per-environment delay knob).
    estimator: DelayEstimator,
    /// Pre-delay line on the reference: holds `predelay_len` samples so the
    /// filter only has to model the reverb tail, not the bulk transport delay.
    predelay: VecDeque<f32>,
    predelay_len: usize,
    /// Head-room (samples) kept ahead of the estimated echo so estimation error
    /// and frame-level jitter still land within the filter taps.
    head_margin: usize,
}

impl Aec {
    /// `filter_length_ms` is the reverb tail the adaptive filter models. It no
    /// longer has to span the bulk transport delay — that is measured and
    /// removed by a pre-delay (see [`DelayEstimator`]) — so a compact filter
    /// works for any speaker→mic delay up to `DELAY_MAX_MS`.
    pub fn new(sample_rate: u32, filter_length_ms: u32) -> Self {
        let num_taps = ((sample_rate as usize * filter_length_ms as usize) / 1000).max(1);
        Self {
            weights: VecDeque::from(vec![0.0; num_taps]),
            far_hist: VecDeque::from(vec![0.0; num_taps]),
            energy: 0.0,
            num_taps,
            reg: NLMS_REG_PER_TAP * num_taps as f32,
            nlp_g: 1.0,
            erl: NLP_ERL_INIT,
            sample_rate,
            estimator: DelayEstimator::new(sample_rate),
            predelay: VecDeque::new(),
            predelay_len: 0,
            head_margin: (sample_rate as usize * DELAY_HEAD_MARGIN_MS as usize / 1000),
        }
    }

    /// Apply a (re)locked bulk delay: pre-delay the reference by
    /// `delay - head_margin` so the echo lands `head_margin` into the filter,
    /// then reset the filter to re-converge around the new alignment. The
    /// estimator only (re)locks rarely, so this disruptive reset is rare.
    fn apply_delay(&mut self, delay_samples: usize) {
        let target = delay_samples.saturating_sub(self.head_margin);
        if target == self.predelay_len {
            return;
        }
        self.predelay_len = target;
        self.predelay = VecDeque::from(vec![0.0; target]);
        self.reset_filter();
    }

    /// Zero the adaptive state. Called when the filter has gone non-finite
    /// (numerical blowup) or the pre-delay changed, so it relearns from scratch
    /// rather than emitting silence forever / fighting a stale alignment.
    fn reset_filter(&mut self) {
        for w in self.weights.iter_mut() {
            *w = 0.0;
        }
        for h in self.far_hist.iter_mut() {
            *h = 0.0;
        }
        self.energy = 0.0;
    }

    /// Pre-delay one reference sample: push it in, return the delayed one.
    fn predelay_sample(&mut self, x: f32) -> f32 {
        if self.predelay_len == 0 {
            return x;
        }
        self.predelay.push_back(x);
        self.predelay.pop_front().unwrap_or(0.0)
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
        // First pass: linear NLMS. Collect the float residual per sample plus
        // the frame energies the residual suppressor needs.
        let mut resid = Vec::with_capacity(near.len());
        let mut sum_xx = 0.0f64; // far energy
        let mut sum_yy = 0.0f64; // echo-estimate energy
        let mut sum_ee = 0.0f64; // residual energy
        let mut new_delay: Option<usize> = None;
        for (i, &d) in near.iter().enumerate() {
            let x_raw = far.get(i).copied().unwrap_or(0) as f32 / 32768.0;
            let d_f = d as f32 / 32768.0;
            // Measure the bulk delay on the RAW (un-pre-delayed) streams.
            if let Some(delay) = self.estimator.push(x_raw, d_f) {
                new_delay = Some(delay);
            }
            // Pre-delay the reference so the echo lands near the filter's front.
            let x = self.predelay_sample(x_raw);
            self.push_far(x);

            // Estimated echo = w · far_hist.
            let y: f32 = self
                .weights
                .iter()
                .zip(self.far_hist.iter())
                .map(|(w, h)| w * h)
                .sum();
            let e = d_f - y;

            // Numerical safety net: if the estimate has gone non-finite, the
            // filter has diverged — reset it and pass the raw near sample
            // through this sample rather than casting NaN to a silent 0.
            if !e.is_finite() {
                self.reset_filter();
                resid.push(d_f);
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

            sum_xx += (x * x) as f64;
            sum_yy += (y * y) as f64;
            sum_ee += (e * e) as f64;
            resid.push(e);
        }

        // Re-align the pre-delay once per frame if a new bulk delay was measured
        // (deferred out of the sample loop so this frame stays self-consistent).
        if let Some(delay) = new_delay {
            self.apply_delay(delay);
        }

        // Second pass: residual echo suppressor. One smoothed gain for the
        // whole frame, then apply it.
        let gain = self.update_nlp_gain(sum_xx, sum_yy, sum_ee, near.len());
        resid
            .into_iter()
            .map(|e| ((e * gain).clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }

    /// Update and return the residual-echo-suppressor gain for this frame.
    ///
    /// First the residual-echo ratio `erl` is tracked from the data: during
    /// echo-only stretches `sum_ee / sum_yy` reveals how much echo the linear
    /// filter leaves, so `erl` is pulled down quickly toward low observed ratios
    /// and drifts up only slowly (double-talk inflates the ratio but, rising
    /// slowly, can't corrupt the estimate). This replaces the old fixed,
    /// per-environment strength knob.
    ///
    /// Then Wiener-style: the leftover echo is estimated as `erl * NLP_OVERSUB *
    /// sum_yy`; whatever residual remains beyond it is treated as near-end
    /// speech to keep. The gain is `near / residual`, floored so output is never
    /// fully muted, and smoothed across frames. Self-gating: when nothing is
    /// playing (`sum_xx` below the floor) or the filter predicts no echo
    /// (`sum_yy` ≈ 0, e.g. no acoustic coupling), the target is unity, so normal
    /// mic capture is untouched.
    ///
    /// Release is slow while the far-end is active (hangover) so a residual
    /// spike or an inter-word gap can't bounce the gain back up and leak echo
    /// mid-playback; once playback stops it releases promptly.
    fn update_nlp_gain(&mut self, sum_xx: f64, sum_yy: f64, sum_ee: f64, n: usize) -> f32 {
        if n == 0 {
            return self.nlp_g;
        }
        let far_pow = sum_xx / n as f64;
        let far_active = far_pow >= NLP_FAR_FLOOR;
        // Only meaningful once the filter actually predicts an echo: right after
        // a reset `sum_yy ≈ 0`, where `sum_ee / sum_yy` would explode and drag
        // the ERL estimate to its ceiling.
        let echo_predicted = sum_yy / n as f64 >= NLP_FAR_FLOOR;

        // Track the residual-echo ratio (minimum-statistics style). The ratio is
        // clamped to the ERL ceiling so a transient can't take a huge step.
        if far_active && echo_predicted {
            let ratio = ((sum_ee / sum_yy) as f32).clamp(0.0, NLP_ERL_MAX);
            let coef = if ratio < self.erl {
                NLP_ERL_TRACK_DOWN
            } else {
                NLP_ERL_TRACK_UP
            };
            self.erl = (self.erl + coef * (ratio - self.erl)).clamp(NLP_ERL_MIN, NLP_ERL_MAX);
        }

        let target = if !far_active {
            1.0
        } else {
            let resid_echo = self.erl as f64 * NLP_OVERSUB * sum_yy;
            let near_pow = (sum_ee - resid_echo).max(0.0);
            ((near_pow / (sum_ee + 1e-9)) as f32).clamp(0.0, 1.0)
        };
        let target = target.max(NLP_GAIN_FLOOR);
        let coef = if target < self.nlp_g {
            NLP_ATTACK
        } else if far_active {
            NLP_RELEASE_FAR_ACTIVE
        } else {
            NLP_RELEASE_IDLE
        };
        self.nlp_g += coef * (target - self.nlp_g);
        self.nlp_g
    }

    pub fn num_taps(&self) -> usize {
        self.num_taps
    }

    /// Current residual-suppressor gain (1.0 = no suppression). For diagnostics.
    pub fn nlp_gain(&self) -> f32 {
        self.nlp_g
    }

    /// Current measured residual-echo ratio. For diagnostics.
    pub fn erl(&self) -> f32 {
        self.erl
    }

    /// Current applied bulk pre-delay in milliseconds. For diagnostics.
    pub fn delay_ms(&self) -> u32 {
        (self.predelay_len as u32 * 1000) / self.sample_rate.max(1)
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

impl EchoCanceller for Aec {
    fn process_frame(&mut self, near: &[i16], far: &[i16]) -> Vec<i16> {
        Aec::process_frame(self, near, far)
    }

    fn stats(&self) -> CancellerStats {
        let (w_l2, peak_tap, _) = self.weight_stats();
        let delay_ms = self.delay_ms();
        CancellerStats {
            delay_ms,
            peak_ms: delay_ms + peak_tap as u32 * 1000 / self.sample_rate.max(1),
            erl: self.erl(),
            nlp_gain: self.nlp_gain(),
            w_l2,
        }
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

    // Deterministic pseudo-random far-end (LCG) so the test needs no rng dep.
    fn lcg(seed: &mut u64) -> i16 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 48) as i16) / 4 // bounded amplitude
    }

    #[test]
    fn delay_estimator_finds_bulk_delay() {
        // near = far delayed by a known bulk delay; the estimator should report
        // it within one decimation step.
        let rate = 16000;
        let true_delay = 1600; // 100 ms
        let mut est = DelayEstimator::new(rate);
        let mut seed = 0xabcd_1234u64;
        let mut line: VecDeque<f32> = VecDeque::from(vec![0.0; true_delay]);
        let mut found: Option<usize> = None;
        // ~2 s of audio, well past the estimator warm-up.
        for _ in 0..rate * 2 {
            let far = lcg(&mut seed) as f32 / 32768.0;
            line.push_back(far);
            let near = 0.5 * line.pop_front().unwrap_or(0.0);
            if let Some(d) = est.push(far, near) {
                found = Some(d);
            }
        }
        let d = found.expect("estimator never produced a confident estimate");
        let err = (d as i64 - true_delay as i64).unsigned_abs() as usize;
        assert!(
            err <= DELAY_DECIM,
            "estimated delay {d} too far from {true_delay} (err {err})"
        );
    }

    #[test]
    fn aec_cancels_pure_echo() {
        // far-end = noise; near = far delayed by D samples and attenuated
        // (a pure linear echo, no near-end voice). After the filter adapts,
        // the residual RMS should fall well below the echo RMS (high ERLE).
        let rate = 16000;
        let delay = 50; // samples of echo path delay
        let mut aec = Aec::new(rate, 80);
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
    fn aec_suppressor_deepens_echo_only_cancellation() {
        // Echo-only (no near speech): the linear filter removes the bulk, then
        // the residual suppressor should drive its gain toward the floor and
        // push total ERLE well past what the linear stage reaches alone — this
        // is the piece that makes played audio actually inaudible, not just
        // quieter.
        let rate = 16000;
        let delay = 40;
        let mut aec = Aec::new(rate, 80);
        let mut seed = 0x0bad_c0de_u64;
        let frame_len = 320;
        let mut echo_delay_line: VecDeque<i16> = VecDeque::from(vec![0i16; delay]);
        let mut last_echo_rms = 0.0;
        let mut last_resid_rms = 0.0;
        for _ in 0..300 {
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
            erle > 24.0,
            "suppressor should push echo-only ERLE well past linear-only, got {erle:.1} dB"
        );
        assert!(
            aec.nlp_gain() < 0.2,
            "suppressor gain should be low on echo-only, got {:.2}",
            aec.nlp_gain()
        );
    }

    #[test]
    fn aec_cancels_short_delay_echo() {
        // A very short echo delay (~1 ms) must be cancelled with no delay hint:
        // the bulk-delay estimator reports ~0 (below the head margin), so the
        // compact filter models it directly from tap 0.
        let rate = 16000;
        let delay = 16; // ~1 ms echo
        let mut aec = Aec::new(rate, 60);
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
            "short-delay echo not cancelled: {erle:.1} dB (echo={last_echo_rms:.0}, resid={last_resid_rms:.0})"
        );
    }

    #[test]
    fn aec_preserves_uncorrelated_near_voice() {
        // A near-end talker overlapping playback (double-talk) must survive.
        // First a stretch of echo-only audio lets the adaptive ERL learn the
        // true leftover-echo floor; then a near-end tone is mixed in (a barge-in
        // burst) and must still come through — the suppressor must not mistake
        // it for residual echo and gate it away.
        let rate = 16000;
        let delay = 40;
        let mut aec = Aec::new(rate, 80);
        let mut seed = 0xdead_beefu64;
        let frame_len = 320;
        let mut echo_delay_line: VecDeque<i16> = VecDeque::from(vec![0i16; delay]);
        let mut t: f32 = 0.0;
        let mut last_voice_only_rms = 0.0;
        let mut last_resid_rms = 0.0;

        // Phase 1: echo only (no near voice) — the ERL tracker learns the floor.
        for _ in 0..200 {
            let far: Vec<i16> = (0..frame_len).map(|_| lcg(&mut seed)).collect();
            let mut near = Vec::with_capacity(frame_len);
            for &f in &far {
                echo_delay_line.push_back(f);
                let delayed = echo_delay_line.pop_front().unwrap_or(0);
                near.push((delayed as f32 * 0.5) as i16);
            }
            aec.process_frame(&near, &far);
        }
        // Phase 2: short double-talk burst — near voice mixed into the echo.
        for _ in 0..20 {
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
        let mut aec = Aec::new(rate, 60);
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
