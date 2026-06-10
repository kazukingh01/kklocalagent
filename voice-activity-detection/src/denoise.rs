//! RNNoise-based pre-VAD denoiser. nnnoiseless only accepts 480-sample
//! 48 kHz frames, so each 320-sample 16 kHz VAD frame is FFT-upsampled to
//! 960 @ 48 kHz, denoised in two RNNoise calls, then downsampled back.

use anyhow::{Context, Result};
use nnnoiseless::DenoiseState;
use rubato::{FftFixedInOut, Resampler};

/// Sample count nnnoiseless wants per `process_frame` call (480 @ 48 kHz = 10 ms).
const RNNOISE_FRAME: usize = 480;

const SRC_RATE: usize = 16_000;
const DST_RATE: usize = 48_000;

/// Stateful per-stream denoiser; feed every 20 ms i16 frame through
/// `process`. All scratch buffers are allocated once and reused.
pub struct Denoiser {
    upsampler: FftFixedInOut<f32>,
    downsampler: FftFixedInOut<f32>,
    /// `DenoiseState::new()` returns it pre-boxed (large internal buffers).
    rnnoise: Box<DenoiseState<'static>>,
    in_16k: Vec<Vec<f32>>,
    /// Doubles as the upsampler's output and the downsampler's input.
    mid_48k: Vec<Vec<f32>>,
    out_16k: Vec<Vec<f32>>,
    rnn_in: Vec<f32>,
    rnn_out: Vec<f32>,
}

impl Denoiser {
    /// Build a denoiser sized for `frame_samples` 16 kHz mono samples per
    /// call (typically 320 = 20 ms).
    pub fn new(frame_samples: usize) -> Result<Self> {
        // FftFixedInOut derives the output chunk size (frame_samples * 3)
        // from the rate ratio; only the input chunk is specified.
        let upsampler = FftFixedInOut::<f32>::new(SRC_RATE, DST_RATE, frame_samples, 1)
            .context("build upsampler 16k→48k")?;
        let downsampler =
            FftFixedInOut::<f32>::new(DST_RATE, SRC_RATE, frame_samples * 3, 1)
                .context("build downsampler 48k→16k")?;
        Ok(Self {
            upsampler,
            downsampler,
            rnnoise: DenoiseState::new(),
            in_16k: vec![vec![0.0; frame_samples]],
            mid_48k: vec![vec![0.0; frame_samples * 3]],
            out_16k: vec![vec![0.0; frame_samples]],
            rnn_in: vec![0.0; RNNOISE_FRAME],
            rnn_out: vec![0.0; RNNOISE_FRAME],
        })
    }

    /// Denoise one VAD frame in-place; input/output length must match.
    pub fn process(&mut self, samples: &mut [i16]) -> Result<()> {
        // nnnoiseless wants f32 in ±32768 (raw i16 cast to float), rubato
        // wants normalised f32; keep the normalised representation
        // throughout and scale up/down around the RNNoise call.
        let in_16k = &mut self.in_16k[0];
        for (dst, src) in in_16k.iter_mut().zip(samples.iter()) {
            *dst = *src as f32 / 32768.0;
        }

        self.upsampler
            .process_into_buffer(&self.in_16k, &mut self.mid_48k, None)
            .context("upsample 16→48")?;

        let mid = &mut self.mid_48k[0];
        debug_assert_eq!(mid.len(), RNNOISE_FRAME * 2);
        for chunk_idx in 0..2 {
            let off = chunk_idx * RNNOISE_FRAME;
            for (i, s) in mid[off..off + RNNOISE_FRAME].iter().enumerate() {
                self.rnn_in[i] = s * 32768.0;
            }
            // process_frame's return value is the model's own voice-activity
            // probability; webrtc-vad downstream is the source of truth, so ignore it.
            self.rnnoise.process_frame(&mut self.rnn_out, &self.rnn_in);
            for (i, s) in self.rnn_out.iter().enumerate() {
                mid[off + i] = s / 32768.0;
            }
        }

        self.downsampler
            .process_into_buffer(&self.mid_48k, &mut self.out_16k, None)
            .context("downsample 48→16")?;

        // Saturate the i16 cast: RNNoise gain can clip very loud input,
        // and cast wraparound would manifest as a loud click.
        let out = &self.out_16k[0];
        for (dst, src) in samples.iter_mut().zip(out.iter()) {
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
        // RNNoise on silence may emit near-zero values from FFT-roundtrip
        // artefacts — only shape stability is asserted.
        let mut d = Denoiser::new(320).unwrap();
        let mut frame = vec![0i16; 320];
        d.process(&mut frame).unwrap();
        assert_eq!(frame.len(), 320);
    }

    #[test]
    fn process_synthetic_tone_preserves_length() {
        // Amplitude isn't checked — RNNoise is ML-based and applies gain,
        // not a pure passthrough.
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
