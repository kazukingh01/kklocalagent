use anyhow::Result;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

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
    // ~10ms of input; rubato requires a fixed chunk per call.
    ((rate as usize + 99) / 100).max(160)
}

/// Converts native-format interleaved mic samples into s16le mono frames
/// at a fixed target sample rate. Accumulates inputs and emits zero or more
/// complete frames per push.
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

    pub fn push_f32(&mut self, data: &[f32]) -> Vec<Vec<u8>> {
        self.downmix(data, |v| *v);
        self.emit()
    }

    pub fn push_i16(&mut self, data: &[i16]) -> Vec<Vec<u8>> {
        self.downmix(data, |v| *v as f32 / 32768.0);
        self.emit()
    }

    pub fn push_u16(&mut self, data: &[u16]) -> Vec<Vec<u8>> {
        self.downmix(data, |v| (*v as f32 - 32768.0) / 32768.0);
        self.emit()
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
                let output = resampler
                    .process(&[input_chunk], None)
                    .expect("resampler process");
                self.resampled_buf.extend_from_slice(&output[0]);
            }
        } else {
            self.resampled_buf.append(&mut self.mono_buf);
        }
        let mut frames = Vec::new();
        while self.resampled_buf.len() >= self.target_samples_per_frame {
            let mut bytes = Vec::with_capacity(self.target_samples_per_frame * 2);
            for s in self.resampled_buf.drain(..self.target_samples_per_frame) {
                let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            frames.push(bytes);
        }
        frames
    }
}

/// Converts incoming s16le mono bytes at `source_rate` into interleaved f32
/// samples at the native output rate × channel count. Output is what the
/// cpal callback will stream to the device.
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

    /// Accept s16le bytes and return interleaved native-format f32 samples.
    pub fn push_s16le(&mut self, bytes: &[u8]) -> Vec<f32> {
        for pair in bytes.chunks_exact(2) {
            let v = i16::from_le_bytes([pair[0], pair[1]]);
            self.mono_buf.push(v as f32 / 32768.0);
        }
        if let Some(resampler) = self.resampler.as_mut() {
            while self.mono_buf.len() >= self.resample_chunk {
                let input_chunk: Vec<f32> =
                    self.mono_buf.drain(..self.resample_chunk).collect();
                let output = resampler
                    .process(&[input_chunk], None)
                    .expect("resampler process");
                self.resampled_buf.extend_from_slice(&output[0]);
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
        // Push 640 mono f32 samples → expect 2 frames of 640 bytes each.
        let input: Vec<f32> = (0..640).map(|i| (i as f32 / 640.0) - 0.5).collect();
        let frames = f.push_f32(&input);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].len(), 640);
        assert_eq!(frames[1].len(), 640);
    }

    #[test]
    fn capture_downmixes_stereo() {
        let mut f = CaptureFramer::new(16000, 2, 16000, 320).unwrap();
        // 320 stereo pairs = 640 f32 → 320 mono samples → 1 frame.
        let input: Vec<f32> = vec![0.25; 640];
        let frames = f.push_f32(&input);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), 640);
    }

    #[test]
    fn capture_resamples_48k_to_16k() {
        let mut f = CaptureFramer::new(48000, 1, 16000, 320).unwrap();
        // Feed ~1 second of silence — should produce many frames.
        let input = vec![0.0f32; 48000];
        let frames = f.push_f32(&input);
        // Expect ~50 frames of 640 bytes; allow a little slack for resampler warmup.
        assert!(frames.len() >= 45, "got {} frames", frames.len());
        for fr in &frames {
            assert_eq!(fr.len(), 640);
        }
    }

    #[test]
    fn playback_passthrough_when_rates_match() {
        let mut p = PlaybackFramer::new(16000, 16000, 2).unwrap();
        // 320 samples of s16le = 640 bytes → 320 * 2 = 640 f32 (stereo interleaved).
        let bytes = vec![0u8; 640];
        let out = p.push_s16le(&bytes);
        assert_eq!(out.len(), 640);
    }

    #[test]
    fn playback_resamples_16k_to_48k() {
        let mut p = PlaybackFramer::new(16000, 48000, 1).unwrap();
        let bytes = vec![0u8; 16000 * 2]; // 1 second
        let out = p.push_s16le(&bytes);
        assert!(out.len() >= 44000, "got {} samples", out.len());
    }
}
