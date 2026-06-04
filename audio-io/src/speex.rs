//! Speex DSP echo-cancellation backend (`aec.backend = "speex"`), behind the
//! `speex` Cargo feature.
//!
//! Wraps the [`aec-rs`](https://crates.io/crates/aec-rs) crate, which bundles
//! Speex DSP: a frequency-domain (MDF) adaptive echo canceller plus the Speex
//! preprocessor for residual-echo suppression and denoise. It is offered as an
//! alternative to the built-in pure-Rust [`crate::aec::Aec`] so the two can be
//! A/B compared by flipping one config line.
//!
//! Build with `--features speex`. `aec-rs` compiles the vendored speexdsp from
//! source via cmake + bindgen, so the build host needs `cmake`, a C compiler,
//! and `libclang` for the target (this is why it is opt-in rather than always
//! on — the default build stays dependency-free and cross-compiles trivially).

use aec_rs::{Aec as SpeexInner, AecConfig as SpeexInnerConfig};

use crate::aec::{CancellerStats, EchoCanceller};

/// Speex DSP echo canceller wrapped to the [`EchoCanceller`] interface.
pub struct SpeexAec {
    inner: SpeexInner,
    frame_size: usize,
}

// The underlying Speex states are raw C pointers (`!Send`), but this canceller
// is only ever touched from the single AEC task, so handing it to that task is
// sound.
unsafe impl Send for SpeexAec {}

impl SpeexAec {
    /// `samples_per_frame` is the fixed frame Speex will process (must match the
    /// frames fed by `aec_task`). Speex has no separate delay compensation — its
    /// filter has to span the bulk speaker→mic delay *and* the reverb tail — so
    /// the filter length is floored at 200 ms regardless of the (nlms-oriented)
    /// `filter_length_ms` tail setting.
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
        // Speex requires exactly `frame_size` samples per call; on any size
        // mismatch (shouldn't happen in steady state) pass the mic through.
        if near.len() != self.frame_size || far.len() != self.frame_size {
            return near.to_vec();
        }
        let mut out = vec![0i16; self.frame_size];
        // rec = near (mic), echo = far (reference); writes the cleaned near.
        self.inner.cancel_echo(near, far, &mut out);
        out
    }

    fn stats(&self) -> CancellerStats {
        // Speex doesn't expose its internal filter/ERL state; the backend-
        // agnostic erle_db (computed in aec_task from RMS) is the comparison
        // metric. Report nlp_gain = 1.0 to signal "n/a, not the nlms path".
        CancellerStats {
            nlp_gain: 1.0,
            ..CancellerStats::default()
        }
    }
}
