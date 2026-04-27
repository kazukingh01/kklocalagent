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

/// Outcome of `WakeMachine::try_dispatch`. The variants distinguish
/// the *reason* a SpeechEnded was rejected so the caller can log
/// each case under its own message — important for diagnosing why
/// a particular utterance was dropped (whisper-hallucination noise
/// vs. spoken-while-still-replying vs. arm window timed out).
pub enum DispatchOutcome {
    /// Accept the SpeechEnded and run the pipeline. Caller holds the
    /// guard for the duration of the turn.
    Run(ProcessingGuard),
    /// Wake gate is empty: no recent WakeWordDetected. Drop silently
    /// (this is the normal path for noise / hallucinations under
    /// `wake.required=true`).
    NotArmed,
    /// A turn is already in flight. Drop without disturbing the
    /// current Turn — wake-during-Processing is what triggers
    /// barge-in instead, via on_wake().
    InTurn,
    /// A wake word was received recently but more than `arm_window_ms`
    /// ago. Drop and reset to Idle.
    ArmExpired,
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
    /// `DispatchOutcome::Run(guard)` when the caller should run the
    /// pipeline, or one of three drop reasons. The guard's `Drop`
    /// releases the state back to Idle on turn completion.
    pub fn try_dispatch(self: &Arc<Self>) -> DispatchOutcome {
        if !self.required {
            // Always-listening: no Processing state. Each turn is
            // independent; the guard's complete() is a no-op in this
            // mode (see `complete`).
            return DispatchOutcome::Run(ProcessingGuard { machine: self.clone() });
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        let now = Instant::now();
        match (g.phase, g.armed_until) {
            (Phase::Armed, Some(t)) if t > now => {
                g.phase = Phase::Processing;
                g.armed_until = None;
                DispatchOutcome::Run(ProcessingGuard { machine: self.clone() })
            }
            (Phase::Armed, _) => {
                // Window expired between wake and SpeechEnded. Reset.
                g.phase = Phase::Idle;
                g.armed_until = None;
                DispatchOutcome::ArmExpired
            }
            (Phase::Processing, _) => DispatchOutcome::InTurn,
            (Phase::Idle, _) => DispatchOutcome::NotArmed,
        }
    }

    /// Whether a turn is currently being processed. Used by the
    /// SpeechStarted handler to log "ignored — turn in progress"
    /// instead of the regular "speech started", so the operator can
    /// see at a glance which VAD frames the orchestrator deliberately
    /// passed over. No-op when `required=false` — without the gate
    /// there's no Processing phase to be in.
    pub fn is_in_turn(&self) -> bool {
        if !self.required {
            return false;
        }
        let g = self.inner.lock().expect("wake state poisoned");
        g.phase == Phase::Processing
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

    fn run_guard(o: DispatchOutcome) -> Option<ProcessingGuard> {
        match o {
            DispatchOutcome::Run(g) => Some(g),
            _ => None,
        }
    }

    fn is_run(o: &DispatchOutcome) -> bool {
        matches!(o, DispatchOutcome::Run(_))
    }

    #[test]
    fn idle_drops_speech_ended() {
        let m = mk(cfg(true, 1000, true));
        assert!(matches!(m.try_dispatch(), DispatchOutcome::NotArmed));
    }

    #[test]
    fn wake_then_speech_dispatches_then_idles() {
        let m = mk(cfg(true, 1000, true));
        assert_eq!(m.on_wake(), WakeResult::Armed);
        let permit = run_guard(m.try_dispatch());
        assert!(permit.is_some());
        // Second SpeechEnded without another wake while still
        // Processing → InTurn drop (distinct from NotArmed).
        assert!(matches!(m.try_dispatch(), DispatchOutcome::InTurn));
        drop(permit);
        // After turn completes: back to Idle, next SE dropped as NotArmed.
        assert!(matches!(m.try_dispatch(), DispatchOutcome::NotArmed));
    }

    #[test]
    fn arm_window_expires() {
        let m = mk(cfg(true, 1, true)); // 1 ms window
        assert_eq!(m.on_wake(), WakeResult::Armed);
        std::thread::sleep(Duration::from_millis(10));
        assert!(matches!(m.try_dispatch(), DispatchOutcome::ArmExpired));
    }

    #[test]
    fn always_listening_passes_through() {
        let m = mk(cfg(false, 1000, true));
        assert_eq!(m.on_wake(), WakeResult::Bypass);
        // No wake needed.
        assert!(is_run(&m.try_dispatch()));
        assert!(is_run(&m.try_dispatch()));
    }

    #[test]
    fn barge_in_returns_bargein_during_processing() {
        let m = mk(cfg(true, 1000, true));
        m.on_wake();
        let _permit = run_guard(m.try_dispatch()).unwrap();
        // Processing now. Wake again:
        assert_eq!(m.on_wake(), WakeResult::BargeIn);
    }

    #[test]
    fn no_barge_in_returns_armedbusy_during_processing() {
        let m = mk(cfg(true, 1000, false));
        m.on_wake();
        let _permit = run_guard(m.try_dispatch()).unwrap();
        assert_eq!(m.on_wake(), WakeResult::ArmedBusy);
    }

    #[test]
    fn turn_completion_returns_to_idle() {
        let m = mk(cfg(true, 1000, true));
        m.on_wake();
        let permit = run_guard(m.try_dispatch()).unwrap();
        drop(permit);
        // New wake → arm again; the next dispatch should succeed.
        m.on_wake();
        assert!(is_run(&m.try_dispatch()));
    }

    #[test]
    fn is_in_turn_tracks_processing_phase() {
        let m = mk(cfg(true, 1000, true));
        assert!(!m.is_in_turn(), "Idle");
        m.on_wake();
        assert!(!m.is_in_turn(), "Armed");
        let permit = run_guard(m.try_dispatch()).unwrap();
        assert!(m.is_in_turn(), "Processing");
        drop(permit);
        assert!(!m.is_in_turn(), "Idle after turn");
    }

    #[test]
    fn is_in_turn_false_in_loose_mode() {
        // required=false bypasses the gate entirely; there's no
        // Processing phase, so is_in_turn always reports false even
        // while a notional turn is running.
        let m = mk(cfg(false, 1000, true));
        let _permit = run_guard(m.try_dispatch()).unwrap();
        assert!(!m.is_in_turn());
    }
}
