//! Speex DSP echo-cancellation backend, behind the `speex` Cargo feature.
//! `aec-rs` compiles the vendored speexdsp from source via cmake + bindgen, so
//! the build host needs `cmake`, a C compiler, and `libclang` — hence opt-in.

use aec_rs::{Aec as SpeexInner, AecConfig as SpeexInnerConfig};

use crate::aec::{CancellerStats, EchoCanceller};

pub struct SpeexAec {
    inner: SpeexInner,
    frame_size: usize,
}

// SAFETY: the underlying Speex states are raw C pointers (`!Send`), but this
// canceller is only ever touched from the single AEC task.
unsafe impl Send for SpeexAec {}

impl SpeexAec {
    /// Speex has no separate delay compensation — its filter must span the bulk
    /// speaker→mic delay *and* the reverb tail — so the filter length is floored
    /// at 200 ms regardless of the (nlms-oriented) `filter_length_ms` setting.
    pub fn new(sample_rate: u32, samples_per_frame: usize, filter_length_ms: u32) -> Self {
        let flen_ms = filter_length_ms.max(200);
        let filter_length = (sample_rate as i32 * flen_ms as i32) / 1000;
        let config = SpeexInnerConfig {
            frame_size: samples_per_frame,
            filter_length,
            sample_rate,
            enable_preprocess: true, // residual echo suppression + denoise
        };
        Self {
            inner: SpeexInner::new(&config),
            frame_size: samples_per_frame,
        }
    }
}

impl EchoCanceller for SpeexAec {
    fn process_frame(&mut self, near: &[i16], far: &[i16]) -> Vec<i16> {
        // Speex requires exactly `frame_size` samples per call; on mismatch
        // pass the mic through.
        if near.len() != self.frame_size || far.len() != self.frame_size {
            return near.to_vec();
        }
        let mut out = vec![0i16; self.frame_size];
        // rec = near (mic), echo = far (reference); writes the cleaned near.
        self.inner.cancel_echo(near, far, &mut out);
        out
    }

    fn stats(&self) -> CancellerStats {
        CancellerStats {
            nlp_gain: 1.0,
            ..CancellerStats::default()
        }
    }
}
