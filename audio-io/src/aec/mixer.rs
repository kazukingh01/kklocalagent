//! Far-end reference mixer (see the parent module for the AEC overview): sums
//! the per-track playback PCM into one gap-free 16 kHz mono timeline, carrying
//! the wall-clock play timestamp.

use std::collections::VecDeque;

use crate::pcm::{bytes_to_i16, i16_to_bytes};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
