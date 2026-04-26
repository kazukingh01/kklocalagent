//! Wake-gated state machine for the v1.0 pipeline.
//!
//! ```text
//!     ┌────────────────────────────── WakeWord ──────────────────────────────┐
//!     │                                                                      │
//!     ▼                                                                      │
//!   Idle ── WakeWord ──► Armed ── SpeechEnded ──► Processing ── done ──► Idle
//!                          │                          │
//!                          │  arm_window expires      │  WakeWord (barge_in)
//!                          │                          │     → cancel TTS
//!                          ▼                          ▼
//!                         Idle                      Armed
//! ```
//!
//! Why a hand-rolled state struct rather than a typestate enum: the
//! state is shared across `axum` request handlers (each event arrives
//! on its own task), and request bodies don't carry the typestate. A
//! `Mutex<Inner>` is the simplest correct way to express "pop the
//! Armed window when SpeechEnded fires; reject otherwise" under
//! concurrent dispatch. Lock contention is essentially nil because
//! events are sparse (one per utterance, not per audio frame).
//!
//! Always-listening mode (`required = false`) bypasses the gate
//! entirely and behaves like v0.1: every SpeechEnded triggers a
//! pipeline run. Barge-in is also gated on `required` because
//! barge-in is meaningless when there's no Armed/Processing
//! distinction.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::config::WakeConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Armed,
    Processing,
}

struct Inner {
    phase: Phase,
    armed_until: Option<Instant>,
}

pub struct WakeMachine {
    required: bool,
    arm_window: Duration,
    barge_in: bool,
    inner: Mutex<Inner>,
}

/// What the caller should do in response to a WakeWordDetected event.
/// The state has already been mutated by the time this is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeResult {
    /// Always-listening mode (`wake.required = false`). The state
    /// machine is bypassed; the caller logs the event and moves on.
    Bypass,
    /// First wake or wake during Idle/Armed: window armed.
    Armed,
    /// Wake during Processing with `barge_in = true`: caller should
    /// cancel the in-flight TTS (`tts.stop_url`) and treat the next
    /// SpeechEnded as a fresh turn.
    BargeIn,
    /// Wake during Processing with `barge_in = false`: window armed,
    /// but the caller must NOT interrupt the current turn. The next
    /// SpeechEnded after the current turn finishes will be accepted.
    ArmedBusy,
}

/// Drop-on-completion guard returned by `try_dispatch`. Holds an
/// `Arc<WakeMachine>` (rather than a borrow) so it can be `move`d
/// into `tokio::spawn` for the duration of a turn — once the spawned
/// task ends, `Drop` transitions Processing back to Idle.
///
/// Held by the pipeline runner; never inspected directly.
pub struct ProcessingGuard {
    machine: Arc<WakeMachine>,
}

impl Drop for ProcessingGuard {
    fn drop(&mut self) {
        self.machine.complete();
    }
}

impl WakeMachine {
    pub fn new(cfg: &WakeConfig) -> Self {
        Self {
            required: cfg.required,
            arm_window: Duration::from_millis(cfg.arm_window_ms),
            barge_in: cfg.barge_in,
            inner: Mutex::new(Inner {
                phase: Phase::Idle,
                armed_until: None,
            }),
        }
    }

    /// Whether barge-in TTS cancellation should be attempted on
    /// `WakeResult::BargeIn`. The caller checks this before deciding
    /// whether to fire the `tts.stop_url` POST.
    pub fn barge_in_enabled(&self) -> bool {
        self.barge_in
    }

    /// React to a WakeWordDetected event. See `WakeResult`.
    pub fn on_wake(&self) -> WakeResult {
        if !self.required {
            return WakeResult::Bypass;
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        let was_processing = g.phase == Phase::Processing;
        let until = Instant::now() + self.arm_window;
        if was_processing {
            if self.barge_in {
                // Promote: drop Processing back to Armed so the next
                // SpeechEnded (the barging utterance) is accepted.
                // The ProcessingGuard held by the running pipeline
                // still exists — when it drops at end of turn it'll
                // observe Armed and leave it alone (see `complete`).
                g.phase = Phase::Armed;
                g.armed_until = Some(until);
                WakeResult::BargeIn
            } else {
                // Arm for after the current turn finishes. Don't
                // touch `phase` — the running pipeline owns it.
                g.armed_until = Some(until);
                WakeResult::ArmedBusy
            }
        } else {
            g.phase = Phase::Armed;
            g.armed_until = Some(until);
            WakeResult::Armed
        }
    }

    /// React to a SpeechEnded event with utterance audio. Returns
    /// `Some(ProcessingGuard)` if the caller should run the pipeline,
    /// `None` if the event should be dropped (Idle, or arm window
    /// expired). The guard's `Drop` releases the state back to Idle.
    pub fn try_dispatch(self: &Arc<Self>) -> Option<ProcessingGuard> {
        if !self.required {
            // Always-listening: no Processing state. Each turn is
            // independent; the guard's complete() is a no-op in this
            // mode (see `complete`).
            return Some(ProcessingGuard { machine: self.clone() });
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        let now = Instant::now();
        match (g.phase, g.armed_until) {
            (Phase::Armed, Some(t)) if t > now => {
                g.phase = Phase::Processing;
                g.armed_until = None;
                Some(ProcessingGuard { machine: self.clone() })
            }
            (Phase::Armed, _) => {
                // Window expired between wake and SpeechEnded. Reset.
                g.phase = Phase::Idle;
                g.armed_until = None;
                None
            }
            _ => None,
        }
    }

    /// Called from ProcessingGuard::drop. Transitions Processing →
    /// Idle. If a barge-in already promoted us to Armed mid-turn,
    /// leave that alone — the new arm window is owned by the next
    /// utterance.
    fn complete(&self) {
        if !self.required {
            return;
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        if g.phase == Phase::Processing {
            g.phase = Phase::Idle;
            g.armed_until = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(required: bool, ms: u64, barge: bool) -> WakeConfig {
        WakeConfig { required, arm_window_ms: ms, barge_in: barge }
    }

    fn mk(c: WakeConfig) -> Arc<WakeMachine> {
        Arc::new(WakeMachine::new(&c))
    }

    #[test]
    fn idle_drops_speech_ended() {
        let m = mk(cfg(true, 1000, true));
        assert!(m.try_dispatch().is_none());
    }

    #[test]
    fn wake_then_speech_dispatches_then_idles() {
        let m = mk(cfg(true, 1000, true));
        assert_eq!(m.on_wake(), WakeResult::Armed);
        let permit = m.try_dispatch();
        assert!(permit.is_some());
        // Second SpeechEnded without another wake: dropped.
        assert!(m.try_dispatch().is_none());
        drop(permit);
        // After turn completes: still idle, next SpeechEnded dropped.
        assert!(m.try_dispatch().is_none());
    }

    #[test]
    fn arm_window_expires() {
        let m = mk(cfg(true, 1, true)); // 1 ms window
        assert_eq!(m.on_wake(), WakeResult::Armed);
        std::thread::sleep(Duration::from_millis(10));
        assert!(m.try_dispatch().is_none());
    }

    #[test]
    fn always_listening_passes_through() {
        let m = mk(cfg(false, 1000, true));
        assert_eq!(m.on_wake(), WakeResult::Bypass);
        // No wake needed.
        assert!(m.try_dispatch().is_some());
        assert!(m.try_dispatch().is_some());
    }

    #[test]
    fn barge_in_returns_bargein_during_processing() {
        let m = mk(cfg(true, 1000, true));
        m.on_wake();
        let _permit = m.try_dispatch().unwrap();
        // Processing now. Wake again:
        assert_eq!(m.on_wake(), WakeResult::BargeIn);
    }

    #[test]
    fn no_barge_in_returns_armedbusy_during_processing() {
        let m = mk(cfg(true, 1000, false));
        m.on_wake();
        let _permit = m.try_dispatch().unwrap();
        assert_eq!(m.on_wake(), WakeResult::ArmedBusy);
    }

    #[test]
    fn turn_completion_returns_to_idle() {
        let m = mk(cfg(true, 1000, true));
        m.on_wake();
        let permit = m.try_dispatch().unwrap();
        drop(permit);
        // New wake → arm again; the next dispatch should succeed.
        m.on_wake();
        assert!(m.try_dispatch().is_some());
    }
}
