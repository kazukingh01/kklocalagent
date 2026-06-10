use std::collections::VecDeque;

use crate::pcm::{bytes_to_i16, i16_to_bytes};

/// Each track buffer is capped: the mix timer drains at wall-clock rate while
/// the playback tap fills at the audio-hardware rate, so a device clock running
/// faster than the timer would otherwise grow a buffer without bound.
pub struct ReferenceMixer {
    samples_per_frame: usize,
    max_track_samples: usize,
    tracks: Vec<VecDeque<i16>>,
    dropped: u64,
}

impl ReferenceMixer {
    pub fn new(sample_rate: u32, samples_per_frame: usize, n_tracks: usize) -> Self {
        Self {
            samples_per_frame,
            max_track_samples: (sample_rate as usize / 4).max(samples_per_frame),
            tracks: (0..n_tracks).map(|_| VecDeque::new()).collect(),
            dropped: 0,
        }
    }

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
        assert_eq!(bytes_to_i16(&m.tick()), vec![i16::MAX, i16::MIN]);
    }

    #[test]
    fn mixer_emits_silence_when_idle() {
        let mut m = ReferenceMixer::new(16000, 4, 2);
        assert_eq!(bytes_to_i16(&m.tick()), vec![0, 0, 0, 0]);
    }

    #[test]
    fn mixer_carries_residual_across_ticks() {
        let mut m = ReferenceMixer::new(16000, 4, 1);
        m.push(0, &i16_to_bytes(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(bytes_to_i16(&m.tick()), vec![1, 2, 3, 4]);
        assert_eq!(bytes_to_i16(&m.tick()), vec![5, 6, 0, 0]);
    }

    #[test]
    fn mixer_caps_track_backlog_and_counts_drops() {
        let mut m = ReferenceMixer::new(16, 2, 1);
        m.push(0, &i16_to_bytes(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(m.dropped(), 2);
        assert_eq!(bytes_to_i16(&m.tick()), vec![3, 4]);
        assert_eq!(bytes_to_i16(&m.tick()), vec![5, 6]);
    }
}
