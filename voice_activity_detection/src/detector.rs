use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silent,
    Speaking,
}

/// Events emitted by [`SpeechFsm`]. Serialized with `{"name": "...", ...}`
/// shape so the envelope is flat on the wire.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "name")]
pub enum Event {
    SpeechStarted {
        frame_index: u64,
    },
    SpeechEnded {
        frame_index: u64,
        duration_frames: u64,
        audio_len_bytes: usize,
    },
}

/// Two-state speech segmenter driven by per-frame `is_speech` decisions from a
/// VAD implementation. The FSM is deliberately decoupled from the VAD itself
/// so its logic can be unit-tested with plain booleans.
///
/// While in `Speaking` state every received frame is appended to an internal
/// utterance buffer; on `SpeechEnded` that buffer is exposed via
/// [`SpeechFsm::utterance_buffer`] and reset on the next start.
pub struct SpeechFsm {
    state: State,
    voiced_run: u32,
    silent_run: u32,
    start_frames: u32,
    hang_frames: u32,
    max_utterance_frames: u32,
    utterance: Vec<u8>,
    utterance_frames: u64,
    frame_index: u64,
}

impl SpeechFsm {
    pub fn new(start_frames: u32, hang_frames: u32, max_utterance_frames: u32) -> Self {
        Self {
            state: State::Silent,
            voiced_run: 0,
            silent_run: 0,
            start_frames,
            hang_frames,
            max_utterance_frames,
            utterance: Vec::new(),
            utterance_frames: 0,
            frame_index: 0,
        }
    }

    pub fn push_frame(&mut self, pcm: &[u8], is_speech: bool) -> Option<Event> {
        let idx = self.frame_index;
        self.frame_index += 1;
        match self.state {
            State::Silent => {
                self.voiced_run = if is_speech { self.voiced_run + 1 } else { 0 };
                if self.voiced_run >= self.start_frames {
                    self.state = State::Speaking;
                    self.silent_run = 0;
                    self.utterance.clear();
                    self.utterance.extend_from_slice(pcm);
                    self.utterance_frames = 1;
                    return Some(Event::SpeechStarted { frame_index: idx });
                }
                None
            }
            State::Speaking => {
                self.utterance.extend_from_slice(pcm);
                self.utterance_frames += 1;
                self.silent_run = if is_speech { 0 } else { self.silent_run + 1 };
                let force_end = self.utterance_frames >= self.max_utterance_frames as u64;
                if self.silent_run >= self.hang_frames || force_end {
                    let ev = Event::SpeechEnded {
                        frame_index: idx + 1,
                        duration_frames: self.utterance_frames,
                        audio_len_bytes: self.utterance.len(),
                    };
                    self.state = State::Silent;
                    self.voiced_run = 0;
                    self.silent_run = 0;
                    return Some(ev);
                }
                None
            }
        }
    }

    pub fn utterance_buffer(&self) -> &[u8] {
        &self.utterance
    }

    pub fn is_speaking(&self) -> bool {
        self.state == State::Speaking
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Vec<u8> {
        vec![0u8; 640]
    }

    #[test]
    fn stays_silent_on_silence() {
        let mut fsm = SpeechFsm::new(3, 20, 1500);
        for _ in 0..100 {
            assert!(fsm.push_frame(&frame(), false).is_none());
        }
    }

    #[test]
    fn starts_after_start_frames_consecutive_voiced() {
        let mut fsm = SpeechFsm::new(3, 20, 1500);
        assert!(fsm.push_frame(&frame(), true).is_none());
        assert!(fsm.push_frame(&frame(), true).is_none());
        assert!(matches!(
            fsm.push_frame(&frame(), true),
            Some(Event::SpeechStarted { .. })
        ));
    }

    #[test]
    fn voiced_run_resets_on_silence() {
        let mut fsm = SpeechFsm::new(3, 20, 1500);
        fsm.push_frame(&frame(), true);
        fsm.push_frame(&frame(), true);
        fsm.push_frame(&frame(), false); // reset
        assert!(fsm.push_frame(&frame(), true).is_none());
        assert!(fsm.push_frame(&frame(), true).is_none());
        assert!(matches!(
            fsm.push_frame(&frame(), true),
            Some(Event::SpeechStarted { .. })
        ));
    }

    #[test]
    fn ends_after_hang_frames_silence() {
        let mut fsm = SpeechFsm::new(2, 3, 1500);
        // voiced_run=1 — no event yet
        assert!(fsm.push_frame(&frame(), true).is_none());
        // voiced_run=2 → SpeechStarted, utterance_frames=1
        assert!(matches!(
            fsm.push_frame(&frame(), true),
            Some(Event::SpeechStarted { .. })
        ));
        // Two more voiced frames → frames=3
        fsm.push_frame(&frame(), true);
        fsm.push_frame(&frame(), true);
        // Two silent frames → silent_run=2, frames=5
        fsm.push_frame(&frame(), false);
        fsm.push_frame(&frame(), false);
        // Third silent frame → silent_run=3 ≥ hang_frames → SpeechEnded
        let ev = fsm.push_frame(&frame(), false);
        let Some(Event::SpeechEnded {
            duration_frames,
            audio_len_bytes,
            ..
        }) = ev
        else {
            panic!("expected SpeechEnded, got {ev:?}");
        };
        assert_eq!(duration_frames, 6);
        assert_eq!(audio_len_bytes, 6 * 640);
    }

    #[test]
    fn max_utterance_forces_end() {
        let mut fsm = SpeechFsm::new(1, 100, 5);
        // SpeechStarted, utterance_frames=1
        fsm.push_frame(&frame(), true);
        // Feed continuous speech; no natural hang termination can fire.
        let mut ended_at = None;
        for i in 0..10 {
            if let Some(ev) = fsm.push_frame(&frame(), true) {
                ended_at = Some((i, ev));
                break;
            }
        }
        let Some((i, Event::SpeechEnded { duration_frames, .. })) = ended_at else {
            panic!("expected forced end, got {ended_at:?}");
        };
        // 1 start frame + 4 loop iterations → frames=5, cap reached
        assert_eq!(i, 3);
        assert_eq!(duration_frames, 5);
    }

    #[test]
    fn restart_after_end_works() {
        let mut fsm = SpeechFsm::new(1, 2, 1500);
        fsm.push_frame(&frame(), true); // start #1
        fsm.push_frame(&frame(), false);
        fsm.push_frame(&frame(), false); // end #1
        assert!(!fsm.is_speaking());
        let ev = fsm.push_frame(&frame(), true);
        assert!(matches!(ev, Some(Event::SpeechStarted { .. })));
    }
}
