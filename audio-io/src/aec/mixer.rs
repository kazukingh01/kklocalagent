//! Far-end reference mixer (see the parent module for the AEC overview): sums
//! the per-track playback PCM into one gap-free 16 kHz mono timeline.

use std::collections::VecDeque;

use crate::pcm::{bytes_to_i16, i16_to_bytes};

/// Sums the per-track far-end PCM into one 16 kHz mono s16le stream.
///
/// Track frames arrive asynchronously (one `push` per playback output
/// callback, any length); the mixer buffers per track and emits fixed
/// `samples_per_frame` slots on [`tick`](Self::tick). A slot with no buffered
/// samples for a track contributes silence, so the output is gap-free even
/// when nothing plays — which the adaptive filter relies on for a continuous
/// far-end timeline.
///
/// Each track buffer is capped at `max_track_samples`: the mix timer drains
/// `samples_per_frame` per tick at wall-clock rate, while the playback tap
/// fills at the audio-hardware rate, so a device clock running faster than the
/// timer would otherwise grow a track buffer without bound during continuous
/// playback. Excess beyond the cap drops the oldest samples (counted in
/// [`dropped`](Self::dropped) so the task can log it).
pub struct ReferenceMixer {
    samples_per_frame: usize,
    /// Per-track backlog cap (samples). Anything beyond this drops oldest.
    max_track_samples: usize,
    tracks: Vec<VecDeque<i16>>,
    /// Cumulative far samples dropped by the per-track cap (runaway guard).
    dropped: u64,
}

impl ReferenceMixer {
    pub fn new(sample_rate: u32, samples_per_frame: usize, n_tracks: usize) -> Self {
        Self {
            samples_per_frame,
            // ~250 ms, matching aec_task's far-backlog cap. Floored at one
            // frame so a freshly-pushed frame is never dropped on arrival.
            max_track_samples: (sample_rate as usize / 4).max(samples_per_frame),
            tracks: (0..n_tracks).map(|_| VecDeque::new()).collect(),
            dropped: 0,
        }
    }

    /// Append a track's incoming s16le bytes, dropping the oldest if the track
    /// backlog exceeds the cap. Out-of-range track ids are ignored (defensive —
    /// the playback tap already supplies a valid track id).
    pub fn push(&mut self, track_id: usize, bytes: &[u8]) {
        let cap = self.max_track_samples;
        let mut dropped = 0;
        if let Some(buf) = self.tracks.get_mut(track_id) {
            buf.extend(bytes_to_i16(bytes));
            if buf.len() > cap {
                dropped = buf.len() - cap;
                buf.drain(..dropped);
            }
        }
        self.dropped += dropped as u64;
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

    /// Cumulative far samples dropped by the per-track backlog cap. Monotonic;
    /// the task logs the delta periodically.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_passthrough_single_track() {
        let mut m = ReferenceMixer::new(16000, 4, 2);
        let frame = i16_to_bytes(&[100, -200, 300, -400]);
        m.push(0, &frame);
        assert_eq!(bytes_to_i16(&m.tick()), vec![100, -200, 300, -400]);
    }

    #[test]
    fn mixer_sums_two_tracks() {
        let mut m = ReferenceMixer::new(16000, 4, 2);
        m.push(0, &i16_to_bytes(&[100, 100, 100, 100]));
        m.push(1, &i16_to_bytes(&[50, -50, 50, -50]));
        assert_eq!(bytes_to_i16(&m.tick()), vec![150, 50, 150, 50]);
    }

    #[test]
    fn mixer_saturates_instead_of_wrapping() {
        let mut m = ReferenceMixer::new(16000, 2, 2);
        m.push(0, &i16_to_bytes(&[30000, -30000]));
        m.push(1, &i16_to_bytes(&[30000, -30000]));
        // 60000 / -60000 must clamp to i16 range, not wrap.
        assert_eq!(bytes_to_i16(&m.tick()), vec![i16::MAX, i16::MIN]);
    }

    #[test]
    fn mixer_emits_silence_when_idle() {
        let mut m = ReferenceMixer::new(16000, 4, 2);
        assert_eq!(bytes_to_i16(&m.tick()), vec![0, 0, 0, 0]);
    }

    #[test]
    fn mixer_carries_residual_across_ticks() {
        // Push 6 samples into a 4-wide mixer: first tick takes 4, second takes
        // the remaining 2 then pads with silence.
        let mut m = ReferenceMixer::new(16000, 4, 1);
        m.push(0, &i16_to_bytes(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(bytes_to_i16(&m.tick()), vec![1, 2, 3, 4]);
        assert_eq!(bytes_to_i16(&m.tick()), vec![5, 6, 0, 0]);
    }

    #[test]
    fn mixer_caps_track_backlog_and_counts_drops() {
        // Cap is sample_rate/4 = 4 here; pushing 6 samples drops the oldest 2.
        let mut m = ReferenceMixer::new(16, 2, 1);
        m.push(0, &i16_to_bytes(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(m.dropped(), 2);
        // The two oldest (1, 2) were dropped; the cap window kept 3..=6.
        assert_eq!(bytes_to_i16(&m.tick()), vec![3, 4]);
        assert_eq!(bytes_to_i16(&m.tick()), vec![5, 6]);
    }
}
