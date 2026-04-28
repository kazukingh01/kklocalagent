//! RNNoise-based pre-VAD denoiser.
//!
//! nnnoiseless processes 480-sample 48 kHz frames. Our pipeline
//! delivers 320-sample 16 kHz frames (20 ms each from audio-io's
//! /mic). Per VAD frame we therefore:
//!
//!   320 samples @ 16 kHz  ──► (rubato FFT upsample)  ──► 960 @ 48 kHz
//!   960 @ 48 kHz          ──► (2× nnnoiseless::DenoiseFeatures
//!                                process_frame, each on 480
//!                                samples)              ──► 960 @ 48 kHz
//!   960 @ 48 kHz          ──► (rubato FFT downsample) ──► 320 @ 16 kHz
//!
//! Cost is dominated by the two RNNoise calls (~50 µs each on x86)
//! and the two FFT resamples (similar magnitude). Total <0.5% of
//! one core at the 50 frames-per-second pipeline rate. The model
//! itself (~85 KB) is baked into the nnnoiseless crate so there's
//! no separate weight file to ship.
//!
//! The denoised frames replace the originals in both VAD's
//! classifier input *and* the utterance buffer that goes to ASR,
//! so Whisper sees the cleaner audio too — buys back some of the
//! Japanese-silence-hallucination problem ("ご視聴ありがとう…")
//! that whisper.cpp tends to fall into on noisy near-silence.

use anyhow::{Context, Result};
use nnnoiseless::DenoiseState;
use rubato::{FftFixedInOut, Resampler};

/// Sample count nnnoiseless wants per `process_frame` call (480 @ 48 kHz = 10 ms).
const RNNOISE_FRAME: usize = 480;

/// Source pipeline: 16 kHz mono.
const SRC_RATE: usize = 16_000;
/// nnnoiseless target rate.
const DST_RATE: usize = 48_000;

/// Stateful per-stream denoiser. Hold one of these alongside the
/// VAD instance and feed every 20 ms i16 frame through `process`.
pub struct Denoiser {
    upsampler: FftFixedInOut<f32>,
    downsampler: FftFixedInOut<f32>,
    /// Box because DenoiseState owns large internal buffers (≈ a
    /// few KB) and `nnnoiseless::DenoiseState::new()` returns it
    /// pre-boxed for that reason.
    rnnoise: Box<DenoiseState<'static>>,
    /// Reusable scratch buffers, sized once at construction.
    src_f32: Vec<f32>,
    dst_f32: Vec<f32>,
    rnn_in: Vec<f32>,
    rnn_out: Vec<f32>,
}

impl Denoiser {
    /// Build a denoiser sized to handle `frame_samples` 16 kHz mono
    /// samples per call (typically 320 = 20 ms). Both resamplers are
    /// FFT-based with chunk size matching the VAD frame so the FFT
    /// plans cache and reuse on every iteration.
    pub fn new(frame_samples: usize) -> Result<Self> {
        // 16 kHz → 48 kHz: chunk_size_in = frame_samples (320),
        // chunk_size_out = frame_samples * 3 (960). FftFixedInOut
        // computes the output size from the rate ratio, so we just
        // give it the input chunk and the rates; output emerges
        // automatically.
        let upsampler = FftFixedInOut::<f32>::new(SRC_RATE, DST_RATE, frame_samples, 1)
            .context("build upsampler 16k→48k")?;
        let downsampler =
            FftFixedInOut::<f32>::new(DST_RATE, SRC_RATE, frame_samples * 3, 1)
                .context("build downsampler 48k→16k")?;
        Ok(Self {
            upsampler,
            downsampler,
            rnnoise: DenoiseState::new(),
            src_f32: vec![0.0; frame_samples],
            dst_f32: vec![0.0; frame_samples * 3],
            rnn_in: vec![0.0; RNNOISE_FRAME],
            rnn_out: vec![0.0; RNNOISE_FRAME],
        })
    }

    /// Denoise one VAD frame in-place. Input/output length must be
    /// identical (typically 320 samples). Errors propagate if the
    /// resampler hits a size mismatch — should never happen in
    /// steady state since both sides are fixed size.
    pub fn process(&mut self, samples: &mut [i16]) -> Result<()> {
        // i16 → f32 in [-1, 1]. nnnoiseless internally wants f32
        // *not* normalised (it expects raw 16-bit values cast to
        // float, range ±32768), but rubato wants normalised f32 and
        // the conversion to/from is symmetric so we keep one
        // representation throughout: scale up before the RNNoise
        // call, scale down after.
        for (dst, src) in self.src_f32.iter_mut().zip(samples.iter()) {
            *dst = *src as f32 / 32768.0;
        }

        // 16 k → 48 k. FftFixedInOut::process takes &[Vec<f32>] (one
        // vec per channel) and returns Vec<Vec<f32>>. We're mono so
        // a single-element wrapper is fine.
        let upsampled = self
            .upsampler
            .process(&[self.src_f32.clone()], None)
            .context("upsample 16→48")?;
        let mono_48 = &upsampled[0];

        // RNNoise: 2× 480-sample frames per VAD frame (960 = 20 ms
        // at 48 k). DenoiseFeatures expects the input scaled to
        // ±32768 (signed-16 range as f32), so undo the rubato
        // normalisation at this boundary.
        debug_assert_eq!(mono_48.len(), RNNOISE_FRAME * 2);
        for chunk_idx in 0..2 {
            let off = chunk_idx * RNNOISE_FRAME;
            for (i, s) in mono_48[off..off + RNNOISE_FRAME].iter().enumerate() {
                self.rnn_in[i] = s * 32768.0;
            }
            // process_frame returns the model's voice-activity
            // probability for this 10 ms window; we ignore it (our
            // own webrtc-vad classifier downstream is the source
            // of truth for SS/SE).
            self.rnnoise.process_frame(&mut self.rnn_out, &self.rnn_in);
            for (i, s) in self.rnn_out.iter().enumerate() {
                self.dst_f32[off + i] = s / 32768.0;
            }
        }

        // 48 k → 16 k.
        let downsampled = self
            .downsampler
            .process(&[self.dst_f32.clone()], None)
            .context("downsample 48→16")?;
        let mono_16 = &downsampled[0];

        // f32 → i16 with saturation. RNNoise should keep amplitudes
        // bounded, but a clipped voice frame after gain is still
        // possible with very loud input — saturate cleanly to avoid
        // i16-cast wraparound (which would manifest as a loud click).
        for (dst, src) in samples.iter_mut().zip(mono_16.iter()) {
            *dst = (src * 32768.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_silence_returns_silence_shaped_output() {
        // Smoke: feed pure silence, get back something the same length
        // (320 samples) without panicking. RNNoise on silence may emit
        // very small near-zero values due to FFT-roundtrip artefacts —
        // that's fine, we just want shape stability.
        let mut d = Denoiser::new(320).unwrap();
        let mut frame = vec![0i16; 320];
        d.process(&mut frame).unwrap();
        assert_eq!(frame.len(), 320);
    }

    #[test]
    fn process_synthetic_tone_preserves_length() {
        // Feed a 440 Hz sine (mid-band, well within voice range so
        // RNNoise should mostly preserve it) and confirm length holds.
        // Amplitude survival isn't checked precisely — RNNoise is
        // ML-based and applies gain, not a pure passthrough.
        let mut d = Denoiser::new(320).unwrap();
        let mut frame: Vec<i16> = (0..320)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                ((t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 16_000.0) as i16
            })
            .collect();
        d.process(&mut frame).unwrap();
        assert_eq!(frame.len(), 320);
    }
}
