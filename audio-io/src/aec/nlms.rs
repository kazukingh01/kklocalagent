use std::collections::VecDeque;

use crate::pcm::{f32_to_i16, i16_to_f32};

use super::{CancellerStats, EchoCanceller};

/// 0 < mu < 2 for stability; 0.3 converges in a few hundred ms without ringing.
const NLMS_MU: f32 = 0.3;
/// The old absolute 1e-6 denominator floor was catastrophic: a quiet far-end
/// under a loud near-end made the step `mu*e/denom` explode, weights ran to
/// ±inf, `inf - inf = NaN` poisoned the filter and every output cast to 0
/// (`f32::NAN as i16 == 0`) — total silence, observed in the wild. 1e-4/tap
/// stops the blowup without throttling convergence (≈11 dB ERLE offline vs
/// ≈7 dB at 1e-3/tap).
const NLMS_REG_PER_TAP: f32 = 1e-4;
/// Bleeds off the slow weight drift an un-regularized NLMS accumulates.
const NLMS_LEAK: f32 = 1e-5;

// Residual echo suppressor: the linear NLMS alone reaches only ~10-15 dB ERLE
// on a real speaker→mic path, leaving the echo audible. A Wiener-style gain
// removes the rest; its strength is *measured* from the signal (`erl`) instead
// of a hand-tuned per-room/speaker/volume knob.
//
/// >1 digs past the measured estimate so echo-only output is firmly inaudible;
/// trades echo depth against how much a double-talk talker is dented.
const NLP_OVERSUB: f64 = 2.0;
const NLP_ERL_INIT: f32 = 0.1;
/// ERL tracker (minimum-statistics style): fast pull toward a *lower* observed
/// ratio (echo-only stretches reveal the true floor), slow drift back up so
/// double-talk — which inflates the ratio — can't corrupt the estimate.
const NLP_ERL_TRACK_DOWN: f32 = 0.2;
const NLP_ERL_TRACK_UP: f32 = 0.002;
const NLP_ERL_MIN: f32 = 0.003;
const NLP_ERL_MAX: f32 = 1.0;
const NLP_GAIN_FLOOR: f32 = 0.03;
const NLP_FAR_FLOOR: f64 = 1e-6;
const NLP_ATTACK: f32 = 0.6;
/// Release is SLOW while the far-end is active, so a residual spike or a gap
/// between far-end words can't pop the gain back up and leak echo
/// mid-playback; once playback stops NLP_RELEASE_IDLE restores capture fast.
const NLP_RELEASE_FAR_ACTIVE: f32 = 0.12;
const NLP_RELEASE_IDLE: f32 = 0.7;

// Bulk-delay estimator: rather than making the adaptive filter long enough to
// *span* the per-machine playback/capture/acoustic delay, estimate it and
// pre-delay the reference, so the echo lands at the start of a compact filter
// that only models the room reverb tail.
//
/// Envelope decimation (16 kHz → 2 kHz): delay needs coarse timing, not fine
/// phase, so correlating decimated |signal| envelopes is cheap and robust.
const DELAY_DECIM: usize = 8;
const DELAY_MAX_MS: u32 = 500;
const DELAY_WINDOW_MS: u32 = 256;
const DELAY_ESTIMATE_MS: u32 = 250;
/// Minimum normalized correlation to trust an estimate (rejects double-talk /
/// no-echo frames, which don't correlate cleanly).
const DELAY_CONFIDENCE: f32 = 0.6;
/// Mean-envelope floor the far-end must exceed before estimating at all —
/// quiet/transition frames correlate to noise and produced spurious jumps.
const DELAY_ENV_FLOOR: f32 = 0.01;
/// Consecutive agreeing confident estimates required before (re)locking, so a
/// single spurious frame can't move the pre-delay.
const DELAY_LOCK_COUNT: u32 = 3;
const DELAY_AGREE_MS: u32 = 16;
/// Once locked, re-lock only on a move larger than this; small drift is left
/// for the filter head-room, so the expensive filter reset stays rare.
const DELAY_RELOCK_MS: u32 = 64;
/// Head-room ahead of the estimated echo inside the filter, so estimation
/// error and frame-level jitter still land within the taps.
const DELAY_HEAD_MARGIN_MS: u32 = 32;

fn push_capped(q: &mut VecDeque<f32>, v: f32, cap: usize) {
    if q.len() >= cap {
        q.pop_front();
    }
    q.push_back(v);
}

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
    agree: usize,
    relock: usize,
    // Only a delay confirmed DELAY_LOCK_COUNT times in a row is applied, then
    // held until a stable estimate differs by more than `relock` — resetting
    // the filter on every estimate was the bug that thrashed convergence.
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

    fn raw_estimate(&self) -> Option<usize> {
        if self.near_env.len() < self.window + self.max_lag {
            return None;
        }
        let near: Vec<f32> = self.near_env.iter().copied().collect();
        let far: Vec<f32> = self.far_env.iter().copied().collect();
        let nlen = near.len();
        let n0 = nlen - self.window;
        let mut near_e = 0.0f32;
        let mut far_recent = 0.0f32;
        for j in 0..self.window {
            near_e += near[n0 + j] * near[n0 + j];
            far_recent += far[n0 + j];
        }
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
        if raw.abs_diff(self.cand) <= self.agree {
            self.cand_n += 1;
        } else {
            self.cand = raw;
            self.cand_n = 1;
        }
        if self.cand_n < DELAY_LOCK_COUNT {
            return None;
        }
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

/// Normalized-LMS adaptive echo canceller (16 kHz mono s16le), three
/// self-tuning stages: (1) bulk-delay compensation — the measured speaker→mic
/// delay pre-delays the reference so the filter only spans the reverb tail;
/// (2) linear NLMS subtraction; (3) adaptive Wiener-style residual suppressor
/// whose strength is measured from the signal (`erl`), not hand-set.
pub struct Aec {
    weights: VecDeque<f32>,
    /// Filter input history, newest at the front, paired index-for-index
    /// with `weights`.
    far_hist: VecDeque<f32>,
    /// Incrementally maintained `||far_hist||²`, the NLMS denominator.
    energy: f32,
    num_taps: usize,
    reg: f32,
    nlp_g: f32,
    /// Measured residual-echo ratio (residual energy / echo-estimate energy
    /// during echo-only stretches); drives the suppressor strength.
    erl: f32,
    sample_rate: u32,
    estimator: DelayEstimator,
    predelay: VecDeque<f32>,
    predelay_len: usize,
    head_margin: usize,
}

impl Aec {
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

    /// Pre-delay the reference by `delay - head_margin` so the echo lands
    /// `head_margin` into the filter, then reset to re-converge on the new
    /// alignment (rare — the estimator only (re)locks rarely).
    fn apply_delay(&mut self, delay_samples: usize) {
        let target = delay_samples.saturating_sub(self.head_margin);
        if target == self.predelay_len {
            return;
        }
        self.predelay_len = target;
        self.predelay = VecDeque::from(vec![0.0; target]);
        self.reset_filter();
    }

    /// Zero the adaptive state on numerical blowup or pre-delay change, so the
    /// filter relearns rather than emitting silence forever / fighting a stale
    /// alignment.
    fn reset_filter(&mut self) {
        for w in self.weights.iter_mut() {
            *w = 0.0;
        }
        for h in self.far_hist.iter_mut() {
            *h = 0.0;
        }
        self.energy = 0.0;
    }

    fn predelay_sample(&mut self, x: f32) -> f32 {
        if self.predelay_len == 0 {
            return x;
        }
        self.predelay.push_back(x);
        self.predelay.pop_front().unwrap_or(0.0)
    }

    fn push_far(&mut self, x: f32) {
        if let Some(old) = self.far_hist.pop_back() {
            self.energy -= old * old;
        }
        self.far_hist.push_front(x);
        self.energy += x * x;
        if self.energy < 0.0 {
            // f32 drift can turn the running sum slightly negative.
            self.energy = 0.0;
        }
    }

    pub fn process_frame(&mut self, near: &[i16], far: &[i16]) -> Vec<i16> {
        let mut resid = Vec::with_capacity(near.len());
        let mut sum_xx = 0.0f64;
        let mut sum_yy = 0.0f64;
        let mut sum_ee = 0.0f64;
        let mut new_delay: Option<usize> = None;
        for (i, &d) in near.iter().enumerate() {
            let x_raw = i16_to_f32(far.get(i).copied().unwrap_or(0));
            let d_f = i16_to_f32(d);
            // Measure the bulk delay on the RAW (un-pre-delayed) streams.
            if let Some(delay) = self.estimator.push(x_raw, d_f) {
                new_delay = Some(delay);
            }
            let x = self.predelay_sample(x_raw);
            self.push_far(x);

            let y: f32 = self
                .weights
                .iter()
                .zip(self.far_hist.iter())
                .map(|(w, h)| w * h)
                .sum();
            let e = d_f - y;

            // Non-finite means the filter diverged: reset and pass the raw near
            // sample through rather than casting NaN to a silent 0.
            if !e.is_finite() {
                self.reset_filter();
                resid.push(d_f);
                continue;
            }

            // NLMS weight update with leakage:
            //   w := (1 - leak) * w + mu * e * far_hist / (reg + ||far_hist||²)
            let g = NLMS_MU * e / (self.reg + self.energy);
            for (w, h) in self.weights.iter_mut().zip(self.far_hist.iter()) {
                *w = (1.0 - NLMS_LEAK) * *w + g * h;
            }

            sum_xx += (x * x) as f64;
            sum_yy += (y * y) as f64;
            sum_ee += (e * e) as f64;
            resid.push(e);
        }

        // Re-align deferred out of the sample loop so this frame stays
        // self-consistent.
        if let Some(delay) = new_delay {
            self.apply_delay(delay);
        }

        let gain = self.update_nlp_gain(sum_xx, sum_yy, sum_ee, near.len());
        resid.into_iter().map(|e| f32_to_i16(e * gain)).collect()
    }

    /// Residual-suppressor gain for this frame. `erl` is tracked from the data
    /// (echo-only stretches reveal `sum_ee / sum_yy`; double-talk inflates it
    /// but rises too slowly to corrupt the estimate). Wiener-style: leftover
    /// echo ≈ `erl * NLP_OVERSUB * sum_yy`, residual beyond that is near-end
    /// speech to keep; gain = near / residual, floored and smoothed. When
    /// nothing plays or the filter predicts no echo, the target is unity.
    fn update_nlp_gain(&mut self, sum_xx: f64, sum_yy: f64, sum_ee: f64, n: usize) -> f32 {
        if n == 0 {
            return self.nlp_g;
        }
        let far_pow = sum_xx / n as f64;
        let far_active = far_pow >= NLP_FAR_FLOOR;
        // Right after a reset `sum_yy ≈ 0`, where `sum_ee / sum_yy` would
        // explode and drag the ERL estimate to its ceiling.
        let echo_predicted = sum_yy / n as f64 >= NLP_FAR_FLOOR;

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

    pub fn nlp_gain(&self) -> f32 {
        self.nlp_g
    }

    pub fn erl(&self) -> f32 {
        self.erl
    }

    pub fn delay_ms(&self) -> u32 {
        (self.predelay_len as u32 * 1000) / self.sample_rate.max(1)
    }

    /// The peak tap is the dominant echo delay in samples; a peak pinned at the
    /// last tap (or a near-zero L2 that never grows) means the true echo sits
    /// at/after the window edge and the filter can't reach it.
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

    fn lcg(seed: &mut u64) -> i16 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 48) as i16) / 4
    }

    #[test]
    fn delay_estimator_finds_bulk_delay() {
        let rate = 16000;
        let true_delay = 1600;
        let mut est = DelayEstimator::new(rate);
        let mut seed = 0xabcd_1234u64;
        let mut line: VecDeque<f32> = VecDeque::from(vec![0.0; true_delay]);
        let mut found: Option<usize> = None;
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
        let rate = 16000;
        let delay = 50;
        let mut aec = Aec::new(rate, 80);
        let mut seed = 0x1234_5678u64;

        let frame_len = 320;
        let mut last_echo_rms = 0.0;
        let mut last_resid_rms = 0.0;
        let mut echo_delay_line: VecDeque<i16> = VecDeque::from(vec![0i16; delay]);

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
        let rate = 16000;
        let delay = 16;
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
        let rate = 16000;
        let delay = 40;
        let mut aec = Aec::new(rate, 80);
        let mut seed = 0xdead_beefu64;
        let frame_len = 320;
        let mut echo_delay_line: VecDeque<i16> = VecDeque::from(vec![0i16; delay]);
        let mut t: f32 = 0.0;
        let mut last_voice_only_rms = 0.0;
        let mut last_resid_rms = 0.0;

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
        for _ in 0..20 {
            let far: Vec<i16> = (0..frame_len).map(|_| lcg(&mut seed)).collect();
            let mut near = Vec::with_capacity(frame_len);
            let mut voice_only = Vec::with_capacity(frame_len);
            for &f in &far {
                echo_delay_line.push_back(f);
                let delayed = echo_delay_line.pop_front().unwrap_or(0);
                let voice = (3000.0 * (t * 0.05).sin()) as i16;
                t += 1.0;
                voice_only.push(voice);
                near.push((delayed as f32 * 0.5) as i16 + voice);
            }
            let resid = aec.process_frame(&near, &far);
            last_voice_only_rms = rms(&voice_only);
            last_resid_rms = rms(&resid);
        }
        assert!(
            last_resid_rms > last_voice_only_rms * 0.5,
            "near voice was suppressed: resid={last_resid_rms:.0}, voice={last_voice_only_rms:.0}"
        );
    }

    #[test]
    fn aec_does_not_collapse_on_quiet_far_loud_near() {
        // Regression for the divergence bug hit in the wild (output went fully
        // zero): a quiet far-end (tiny non-zero ||far||²) under a loud near-end
        // let the old absolute 1e-6 denominator floor blow up the weights to
        // ±inf → NaN → every sample cast to 0. A correct filter must pass the
        // loud near through — not annihilate it.
        let rate = 16000;
        let mut aec = Aec::new(rate, 60);
        let frame_len = 320;
        let mut t: f32 = 0.0;
        let mut resid_acc: Vec<i16> = Vec::new();
        let mut near_acc: Vec<i16> = Vec::new();

        for _ in 0..200 {
            let far = vec![30i16; frame_len];
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

        let tail = resid_acc.len() / 2;
        let resid_rms = rms(&resid_acc[tail..]);
        let near_rms = rms(&near_acc[tail..]);
        assert!(
            resid_rms > near_rms * 0.5,
            "filter collapsed (NaN→silence regression?): resid={resid_rms:.0}, near={near_rms:.0}"
        );
    }
}
