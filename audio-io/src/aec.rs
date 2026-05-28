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
//! Two pieces:
//! * [`ReferenceMixer`] sums the per-track far-end PCM into one continuous
//!   16 kHz mono stream (silence when nothing plays — the adaptive filter
//!   needs a gap-free far-end timeline).
//! * [`Aec`] is a pure-Rust normalized-LMS (NLMS) adaptive filter. Pure Rust
//!   (no native dep) so it cross-compiles to the mingw Windows target with
//!   zero extra build setup; the `backend` config field leaves room for a
//!   `speex`/`webrtc` swap later. Because near and far share one clock here,
//!   the bulk delay is known from config (`initial_delay_ms`) and the filter
//!   only has to model the room tail (`filter_length_ms`) — no delay
//!   estimator required.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;
use tracing::{info, warn};

/// NLMS step size. 0 < mu < 2 for stability; 0.3 is a conservative value
/// that converges in a few hundred ms without ringing on a 16 kHz stream.
const NLMS_MU: f32 = 0.3;
/// Regularization added to the far-end energy denominator so a silent
/// far-end (energy ≈ 0) doesn't blow the update up.
const NLMS_EPS: f32 = 1e-6;

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

/// Sums the per-track far-end PCM into one 16 kHz mono s16le stream.
///
/// Track frames arrive asynchronously (one `push` per `/spk` WS frame, any
/// length); the mixer buffers per track and emits fixed `samples_per_frame`
/// slots on [`tick`](Self::tick). A slot with no buffered samples for a track
/// contributes silence, so the output is gap-free even when nothing plays —
/// which the adaptive filter relies on for a continuous far-end timeline.
pub struct ReferenceMixer {
    samples_per_frame: usize,
    tracks: Vec<VecDeque<i16>>,
}

impl ReferenceMixer {
    pub fn new(samples_per_frame: usize, n_tracks: usize) -> Self {
        Self {
            samples_per_frame,
            tracks: (0..n_tracks).map(|_| VecDeque::new()).collect(),
        }
    }

    /// Append a track's incoming s16le bytes. Out-of-range track ids are
    /// ignored (defensive — `/spk` already validates the track id).
    pub fn push(&mut self, track_id: usize, bytes: &[u8]) {
        if let Some(buf) = self.tracks.get_mut(track_id) {
            buf.extend(bytes_to_i16(bytes));
        }
    }

    /// Emit one mixed frame (`samples_per_frame` samples, s16le). Pulls up to
    /// one frame from each track buffer (missing samples = silence) and sums
    /// with saturation so two simultaneous tracks never wrap around.
    pub fn tick(&mut self) -> Vec<u8> {
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
        i16_to_bytes(&mixed)
    }
}

/// Normalized-LMS adaptive echo canceller operating on 16 kHz mono s16le.
///
/// `process_frame(near, far)` returns `near` with the linear echo of `far`
/// subtracted. The far-end is delayed by `delay_samples` (bulk transport
/// delay) before entering a `num_taps`-long adaptive filter that models the
/// room tail; the filter adapts continuously toward whatever minimizes the
/// residual, so steady speaker→mic echo is cancelled while a person talking
/// into the mic (uncorrelated with the far-end) passes through.
pub struct Aec {
    weights: VecDeque<f32>,
    /// Filter input history, newest at the front, paired index-for-index
    /// with `weights`.
    far_hist: VecDeque<f32>,
    /// Bulk-delay line: raw far samples wait here before reaching the filter.
    far_delay: VecDeque<f32>,
    /// Running sum of squares of `far_hist`, maintained incrementally for the
    /// NLMS normalization denominator.
    energy: f32,
    num_taps: usize,
}

impl Aec {
    pub fn new(sample_rate: u32, filter_length_ms: u32, initial_delay_ms: u32) -> Self {
        let num_taps = ((sample_rate as usize * filter_length_ms as usize) / 1000).max(1);
        let delay_samples = (sample_rate as usize * initial_delay_ms as usize) / 1000;
        Self {
            weights: VecDeque::from(vec![0.0; num_taps]),
            far_hist: VecDeque::from(vec![0.0; num_taps]),
            far_delay: VecDeque::from(vec![0.0; delay_samples]),
            energy: 0.0,
            num_taps,
        }
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
            let raw_far = far.get(i).copied().unwrap_or(0) as f32 / 32768.0;
            self.far_delay.push_back(raw_far);
            let x = self.far_delay.pop_front().unwrap_or(0.0);
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

            // NLMS weight update: w += mu * e * far_hist / (eps + ||far_hist||²).
            let g = NLMS_MU * e / (NLMS_EPS + self.energy);
            for (w, h) in self.weights.iter_mut().zip(self.far_hist.iter()) {
                *w += g * h;
            }

            out.push((e.clamp(-1.0, 1.0) * 32767.0) as i16);
        }
        out
    }

    pub fn num_taps(&self) -> usize {
        self.num_taps
    }
}

/// Drains per-track far-end frames into the [`ReferenceMixer`] and publishes
/// one mixed frame every `frame_ms` on `ref_tx`. Runs until both inputs close
/// (services stopped / handle aborted).
pub async fn reference_mixer_task(
    mut ref_in_rx: broadcast::Receiver<(usize, Bytes)>,
    ref_tx: broadcast::Sender<(u64, Bytes)>,
    samples_per_frame: usize,
    n_tracks: usize,
    frame_ms: u32,
) {
    let mut mixer = ReferenceMixer::new(samples_per_frame, n_tracks);
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(frame_ms as u64));
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
                // No subscribers (AEC task gone) → send errors, ignored.
                let _ = ref_tx.send((now_ns(), Bytes::from(frame)));
            }
        }
    }
    info!("reference mixer exiting");
}

/// Subscribes to the raw mic (`mic_rx`, near-end) and the mixed far-end
/// (`ref_rx`), runs each near frame through [`Aec`], and publishes the
/// echo-cancelled mic on `aec_tx` (served as `/mic` when enabled). Near frames drive
/// the clock; far samples are buffered and pulled to match each near frame's
/// length (padding with silence if the far-end momentarily lags).
pub async fn aec_task(
    mut mic_rx: broadcast::Receiver<(u64, Bytes)>,
    mut ref_rx: broadcast::Receiver<(u64, Bytes)>,
    aec_tx: broadcast::Sender<(u64, Bytes)>,
    mut aec: Aec,
) {
    info!(num_taps = aec.num_taps(), "aec task started");
    let mut far_pending: VecDeque<i16> = VecDeque::new();
    loop {
        tokio::select! {
            far = ref_rx.recv() => match far {
                Ok((_ts, bytes)) => far_pending.extend(bytes_to_i16(&bytes)),
                Err(RecvError::Lagged(n)) => {
                    warn!("aec: lagged {n} far-end frames");
                }
                Err(RecvError::Closed) => break,
            },
            near = mic_rx.recv() => match near {
                Ok((ts, bytes)) => {
                    let near = bytes_to_i16(&bytes);
                    let far: Vec<i16> = (0..near.len())
                        .map(|_| far_pending.pop_front().unwrap_or(0))
                        .collect();
                    let cleaned = aec.process_frame(&near, &far);
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

    #[test]
    fn mixer_passthrough_single_track() {
        let mut m = ReferenceMixer::new(4, 2);
        let frame = i16_to_bytes(&[100, -200, 300, -400]);
        m.push(0, &frame);
        assert_eq!(bytes_to_i16(&m.tick()), vec![100, -200, 300, -400]);
    }

    #[test]
    fn mixer_sums_two_tracks() {
        let mut m = ReferenceMixer::new(4, 2);
        m.push(0, &i16_to_bytes(&[100, 100, 100, 100]));
        m.push(1, &i16_to_bytes(&[50, -50, 50, -50]));
        assert_eq!(bytes_to_i16(&m.tick()), vec![150, 50, 150, 50]);
    }

    #[test]
    fn mixer_saturates_instead_of_wrapping() {
        let mut m = ReferenceMixer::new(2, 2);
        m.push(0, &i16_to_bytes(&[30000, -30000]));
        m.push(1, &i16_to_bytes(&[30000, -30000]));
        // 60000 / -60000 must clamp to i16 range, not wrap.
        assert_eq!(bytes_to_i16(&m.tick()), vec![i16::MAX, i16::MIN]);
    }

    #[test]
    fn mixer_emits_silence_when_idle() {
        let mut m = ReferenceMixer::new(4, 2);
        assert_eq!(bytes_to_i16(&m.tick()), vec![0, 0, 0, 0]);
    }

    #[test]
    fn mixer_carries_residual_across_ticks() {
        // Push 6 samples into a 4-wide mixer: first tick takes 4, second takes
        // the remaining 2 then pads with silence.
        let mut m = ReferenceMixer::new(4, 1);
        m.push(0, &i16_to_bytes(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(bytes_to_i16(&m.tick()), vec![1, 2, 3, 4]);
        assert_eq!(bytes_to_i16(&m.tick()), vec![5, 6, 0, 0]);
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
}
