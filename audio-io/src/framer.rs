use anyhow::Result;
use cpal::{FromSample, Sample};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use tracing::error;

use crate::pcm::f32_to_i16;

fn make_resampler(input_rate: u32, output_rate: u32, chunk_size: usize) -> Result<SincFixedIn<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    Ok(SincFixedIn::<f32>::new(
        output_rate as f64 / input_rate as f64,
        1.0,
        params,
        chunk_size,
        1,
    )?)
}

fn resample_chunk_for(rate: u32) -> usize {
    (rate as usize).div_ceil(100).max(160)
}

pub struct CaptureFramer {
    native_channels: usize,
    target_samples_per_frame: usize,
    resampler: Option<SincFixedIn<f32>>,
    resample_chunk: usize,
    mono_buf: Vec<f32>,
    resampled_buf: Vec<f32>,
}

impl CaptureFramer {
    pub fn new(
        native_rate: u32,
        native_channels: u16,
        target_rate: u32,
        target_samples_per_frame: usize,
    ) -> Result<Self> {
        let (resampler, resample_chunk) = if native_rate != target_rate {
            let chunk = resample_chunk_for(native_rate);
            (Some(make_resampler(native_rate, target_rate, chunk)?), chunk)
        } else {
            (None, target_samples_per_frame)
        };
        Ok(Self {
            native_channels: native_channels.max(1) as usize,
            target_samples_per_frame,
            resampler,
            resample_chunk,
            mono_buf: Vec::with_capacity(resample_chunk * 4),
            resampled_buf: Vec::with_capacity(target_samples_per_frame * 4),
        })
    }

    pub fn push<T>(&mut self, data: &[T]) -> Vec<Vec<u8>>
    where
        T: Sample,
        f32: FromSample<T>,
    {
        self.downmix(data, |v| f32::from_sample(*v));
        self.emit()
    }

    pub fn push_f32(&mut self, data: &[f32]) -> Vec<Vec<u8>> {
        self.push(data)
    }

    fn downmix<T, F: Fn(&T) -> f32>(&mut self, data: &[T], to_f32: F) {
        let ch = self.native_channels;
        for chunk in data.chunks(ch) {
            let sum: f32 = chunk.iter().map(&to_f32).sum();
            self.mono_buf.push(sum / ch as f32);
        }
    }

    fn emit(&mut self) -> Vec<Vec<u8>> {
        if let Some(resampler) = self.resampler.as_mut() {
            while self.mono_buf.len() >= self.resample_chunk {
                let input_chunk: Vec<f32> =
                    self.mono_buf.drain(..self.resample_chunk).collect();
                match resampler.process(&[input_chunk], None) {
                    Ok(output) => self.resampled_buf.extend_from_slice(&output[0]),
                    Err(e) => error!("capture resampler error (chunk dropped): {e}"),
                }
            }
        } else {
            self.resampled_buf.append(&mut self.mono_buf);
        }
        let mut frames = Vec::new();
        while self.resampled_buf.len() >= self.target_samples_per_frame {
            let mut bytes = Vec::with_capacity(self.target_samples_per_frame * 2);
            for s in self.resampled_buf.drain(..self.target_samples_per_frame) {
                bytes.extend_from_slice(&f32_to_i16(s).to_le_bytes());
            }
            frames.push(bytes);
        }
        frames
    }
}

pub struct PlaybackFramer {
    native_channels: usize,
    resampler: Option<SincFixedIn<f32>>,
    resample_chunk: usize,
    mono_buf: Vec<f32>,
    resampled_buf: Vec<f32>,
}

impl PlaybackFramer {
    pub fn new(source_rate: u32, native_rate: u32, native_channels: u16) -> Result<Self> {
        let (resampler, resample_chunk) = if source_rate != native_rate {
            let chunk = resample_chunk_for(source_rate);
            (Some(make_resampler(source_rate, native_rate, chunk)?), chunk)
        } else {
            (None, 0)
        };
        Ok(Self {
            native_channels: native_channels.max(1) as usize,
            resampler,
            resample_chunk,
            mono_buf: Vec::new(),
            resampled_buf: Vec::new(),
        })
    }

    pub fn flush(&mut self) {
        self.mono_buf.clear();
        self.resampled_buf.clear();
        if let Some(r) = self.resampler.as_mut() {
            r.reset();
        }
    }

    pub fn push_s16le(&mut self, bytes: &[u8]) -> Vec<f32> {
        for pair in bytes.chunks_exact(2) {
            let v = i16::from_le_bytes([pair[0], pair[1]]);
            self.mono_buf.push(v as f32 / 32768.0);
        }
        if let Some(resampler) = self.resampler.as_mut() {
            while self.mono_buf.len() >= self.resample_chunk {
                let input_chunk: Vec<f32> =
                    self.mono_buf.drain(..self.resample_chunk).collect();
                match resampler.process(&[input_chunk], None) {
                    Ok(output) => self.resampled_buf.extend_from_slice(&output[0]),
                    Err(e) => error!("playback resampler error (chunk dropped): {e}"),
                }
            }
        } else {
            self.resampled_buf.append(&mut self.mono_buf);
        }
        let mut out = Vec::with_capacity(self.resampled_buf.len() * self.native_channels);
        for s in self.resampled_buf.drain(..) {
            for _ in 0..self.native_channels {
                out.push(s);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_passthrough_when_rates_match() {
        let mut f = CaptureFramer::new(16000, 1, 16000, 320).unwrap();
        let input: Vec<f32> = (0..640).map(|i| (i as f32 / 640.0) - 0.5).collect();
        let frames = f.push_f32(&input);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].len(), 640);
        assert_eq!(frames[1].len(), 640);
    }

    #[test]
    fn capture_downmixes_stereo() {
        let mut f = CaptureFramer::new(16000, 2, 16000, 320).unwrap();
        let input: Vec<f32> = vec![0.25; 640];
        let frames = f.push_f32(&input);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), 640);
    }

    #[test]
    fn capture_resamples_48k_to_16k() {
        let mut f = CaptureFramer::new(48000, 1, 16000, 320).unwrap();
        let input = vec![0.0f32; 48000];
        let frames = f.push_f32(&input);
        assert!(frames.len() >= 45, "got {} frames", frames.len());
        for fr in &frames {
            assert_eq!(fr.len(), 640);
        }
    }

    #[test]
    fn playback_passthrough_when_rates_match() {
        let mut p = PlaybackFramer::new(16000, 16000, 2).unwrap();
        let bytes = vec![0u8; 640];
        let out = p.push_s16le(&bytes);
        assert_eq!(out.len(), 640);
    }

    #[test]
    fn playback_resamples_16k_to_48k() {
        let mut p = PlaybackFramer::new(16000, 48000, 1).unwrap();
        let bytes = vec![0u8; 16000 * 2];
        let out = p.push_s16le(&bytes);
        assert!(out.len() >= 44000, "got {} samples", out.len());
    }

    #[test]
    fn playback_flush_clears_residual() {
        let mut p = PlaybackFramer::new(16000, 48000, 1).unwrap();
        let partial = vec![0x10u8; 32];
        let out = p.push_s16le(&partial);
        assert!(
            out.len() < 100,
            "expected little-to-no output before chunk fills; got {}",
            out.len()
        );
        p.flush();
        let bytes = vec![0u8; 16000 * 2];
        let out_after = p.push_s16le(&bytes);
        assert!(out_after.len() >= 44000, "got {} samples", out_after.len());
    }

    #[test]
    fn capture_accumulates_across_non_boundary_pushes() {
        let mut f = CaptureFramer::new(48000, 1, 16000, 320).unwrap();
        let total: Vec<f32> = vec![0.0; 48000];
        let mut frames_total = 0;
        for chunk in total.chunks(777) {
            frames_total += f.push_f32(chunk).len();
        }
        assert!(
            frames_total >= 45,
            "split pushes produced {frames_total} frames — residual not carried?"
        );
    }
}
