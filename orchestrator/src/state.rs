//! Wake-gated state machine.
//!
//! ```text
//!   ┌──────────── (window expires) ───────────┐
//!   ▼                                          │
//!  Idle ─Wake─► ArmedAfterWake ─SS─► Listening ─SE─► Processing
//!                  │  ▲                  │               │
//!                  │  └── Wake (refresh) ┤               │
//!                  │                     │               │
//!                  │  ◄─── Wake (restart from Listening) ┤
//!                  │                                     │
//!                  │     (turn ends)                     │
//!                  │                       ◄─────────────┘
//!   Idle ◄─(window expires)─ ArmedAfterTurn ─SS─► Listening ─...─┐
//!     ▲                          │  ▲                              │
//!     │                          │  └─── Wake → ArmedAfterWake ────┤
//!     └─────── (loop continues) ─┘                                  │
//!                                                                   │
//!   Processing ── Wake (barge_in=true)  → ArmedAfterWake + tts /stop
//!              ── Wake (barge_in=false) → stay Processing,
//!                                          flag pending_wake_after_turn
//!                                          (so complete() goes to
//!                                          ArmedAfterWake, not Turn)
//! ```
//!
//! SpeechStarted is a *phase-only* trigger: ArmedAfter* → Listening, but
//! `armed_until` keeps ticking and is only reset by `complete()` — never
//! by a bare SS. This stops noise-driven SSs from parking state in
//! Listening forever when their SE is filtered upstream by VAD's RMS gate.
//! Always-listening mode (`required = false`) bypasses the gate entirely.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::config::WakeConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    ArmedAfterWake,
    ArmedAfterTurn,
    /// Waiting for SpeechEnded. The armed_until timer from the preceding
    /// ArmedAfter* phase keeps ticking — see the module doc.
    Listening,
    Processing,
}

struct Inner {
    phase: Phase,
    armed_until: Option<Instant>,
    /// Set on a mid-Processing wake with `barge_in=false`: the next
    /// `complete()` goes to ArmedAfterWake instead of ArmedAfterTurn,
    /// honouring the wake the operator pressed mid-reply.
    pending_wake_after_turn: bool,
    /// Monotonic turn id, bumped on every transition into `Processing`.
    /// `complete()` no-ops on a stale id so a barge-in-aborted turn's
    /// guard Drop can't roll back the next turn's state.
    turn_generation: u64,
    /// Instant of the most recent WakeWordDetected; `try_dispatch()` drops
    /// SEs within `post_wake_se_dropout` of it (VAD echoing the wake word
    /// itself). Cleared on successful dispatch so a legit follow-up SE
    /// isn't gated by a stale timestamp.
    last_wake_at: Option<Instant>,
}

pub struct WakeMachine {
    required: bool,
    wake_window: Duration,
    turn_followup_window: Duration,
    barge_in: bool,
    /// `None` (configured 0) disables the post-wake SE dropout check.
    post_wake_se_dropout: Option<Duration>,
    inner: Mutex<Inner>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeResult {
    /// Always-listening (`required=false`); no state change.
    Bypass,
    /// Transitioned (or refreshed) to ArmedAfterWake.
    Armed,
    /// Mid-Processing wake with barge_in=true: caller must cancel TTS;
    /// state already flipped to ArmedAfterWake.
    BargeIn,
    /// Mid-Processing wake with barge_in=false: phase stays Processing;
    /// `complete()` will go to ArmedAfterWake instead of ArmedAfterTurn.
    ArmedBusy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechStartedOutcome {
    Bypass,
    Listening,
    DroppedIdle,
    DroppedAlreadyListening,
    DroppedInTurn,
    WakeWindowExpired,
    TurnWindowExpired,
}

pub enum DispatchOutcome {
    /// Pipeline should run; caller holds the guard for the turn's life.
    Run(ProcessingGuard),
    NotArmed,
    InTurn,
    WakeWindowExpired,
    TurnWindowExpired,
    /// Listening's inherited armed_until expired before SE arrived (noise
    /// drove an SS but the real SE never landed). State reset to Idle.
    ListeningWindowExpired,
    /// SE arrived within `post_wake_se_dropout` of the most recent wake —
    /// almost certainly VAD echoing the wake word's own audio. State is
    /// left armed so the operator's actual utterance still dispatches.
    DroppedTooSoonAfterWake,
}

/// Drop-on-completion guard: dropping it transitions Processing →
/// ArmedAfter{Turn,Wake}. Carries the `turn_generation` it was minted
/// under so a dropped-after-abort guard detects that a different turn now
/// owns Processing and refuses to mutate state.
pub struct ProcessingGuard {
    machine: Arc<WakeMachine>,
    generation: u64,
}

impl Drop for ProcessingGuard {
    fn drop(&mut self) {
        self.machine.complete(self.generation);
    }
}

impl WakeMachine {
    pub fn new(cfg: &WakeConfig) -> Self {
        Self {
            required: cfg.required,
            wake_window: Duration::from_millis(cfg.wake_window_ms),
            turn_followup_window: Duration::from_millis(cfg.turn_followup_window_ms),
            barge_in: cfg.barge_in,
            post_wake_se_dropout: (cfg.post_wake_se_dropout_ms > 0)
                .then(|| Duration::from_millis(cfg.post_wake_se_dropout_ms)),
            inner: Mutex::new(Inner {
                phase: Phase::Idle,
                armed_until: None,
                pending_wake_after_turn: false,
                turn_generation: 0,
                last_wake_at: None,
            }),
        }
    }

    pub fn barge_in_enabled(&self) -> bool {
        self.barge_in
    }

    /// React to a WakeWordDetected event.
    pub fn on_wake(&self) -> WakeResult {
        if !self.required {
            return WakeResult::Bypass;
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        let now = Instant::now();
        // Stamp last_wake_at even mid-Processing under barge_in=false —
        // the dropout check keys off this regardless of phase, so a
        // wake-word echo SE arriving 300 ms later is still gated.
        g.last_wake_at = Some(now);
        let until = now + self.wake_window;
        match g.phase {
            Phase::Processing => {
                if self.barge_in {
                    g.phase = Phase::ArmedAfterWake;
                    g.armed_until = Some(until);
                    g.pending_wake_after_turn = false;
                    WakeResult::BargeIn
                } else {
                    // Don't disturb the running pipeline, but make sure
                    // the operator's wake isn't lost.
                    g.pending_wake_after_turn = true;
                    WakeResult::ArmedBusy
                }
            }
            // Any other state → ArmedAfterWake. Listening → ArmedAfterWake
            // is "wake word said mid-utterance — scratch that, start over":
            // we can't clear VAD's in-progress buffer, but its eventual SE
            // lands inside `post_wake_se_dropout` and gets dropped.
            _ => {
                g.phase = Phase::ArmedAfterWake;
                g.armed_until = Some(until);
                g.pending_wake_after_turn = false;
                WakeResult::Armed
            }
        }
    }

    /// React to a SpeechStarted event (phase-only transition; see module doc).
    pub fn on_speech_started(&self) -> SpeechStartedOutcome {
        if !self.required {
            return SpeechStartedOutcome::Bypass;
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        let now = Instant::now();
        match g.phase {
            Phase::ArmedAfterWake => match g.armed_until {
                Some(t) if t > now => {
                    // Preserve armed_until — noise-driven SSs must not
                    // park state in Listening past the original window.
                    g.phase = Phase::Listening;
                    SpeechStartedOutcome::Listening
                }
                _ => {
                    g.phase = Phase::Idle;
                    g.armed_until = None;
                    SpeechStartedOutcome::WakeWindowExpired
                }
            },
            Phase::ArmedAfterTurn => match g.armed_until {
                Some(t) if t > now => {
                    g.phase = Phase::Listening;
                    SpeechStartedOutcome::Listening
                }
                _ => {
                    g.phase = Phase::Idle;
                    g.armed_until = None;
                    SpeechStartedOutcome::TurnWindowExpired
                }
            },
            Phase::Idle => SpeechStartedOutcome::DroppedIdle,
            Phase::Listening => SpeechStartedOutcome::DroppedAlreadyListening,
            Phase::Processing => SpeechStartedOutcome::DroppedInTurn,
        }
    }

    /// React to a SpeechEnded event with utterance audio. Lenient: an SE
    /// arriving directly in `ArmedAfter*` without a preceding SS is
    /// accepted while the timer holds — real VAD always sends SS first,
    /// but harness tests synthesise events without one.
    pub fn try_dispatch(self: &Arc<Self>) -> DispatchOutcome {
        if !self.required {
            // Loose mode never bumps generation; complete() is a no-op
            // anyway, so the guard just records 0.
            return DispatchOutcome::Run(ProcessingGuard {
                machine: self.clone(),
                generation: 0,
            });
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        let now = Instant::now();
        // Wake-word echo dropout: VAD fires SE ~300 ms after the wake for
        // the wake word's own audio; without this gate that SE becomes a
        // turn whose ASR text *is* the wake word. Phase is left untouched
        // so a follow-up SE within the wake window still dispatches.
        if let Some(dropout) = self.post_wake_se_dropout {
            if let Some(t) = g.last_wake_at {
                if now.saturating_duration_since(t) < dropout {
                    return DispatchOutcome::DroppedTooSoonAfterWake;
                }
            }
        }
        // Fresh generation per Processing transition: a previously-aborted
        // turn's guard carries a stale id and no-ops on Drop.
        let mint_guard = |g: &mut Inner| {
            g.phase = Phase::Processing;
            g.armed_until = None;
            g.turn_generation = g.turn_generation.wrapping_add(1);
            // Clear the wake stamp so a later follow-up SE isn't gated by
            // an ancient wake.
            g.last_wake_at = None;
            ProcessingGuard {
                machine: self.clone(),
                generation: g.turn_generation,
            }
        };
        match g.phase {
            Phase::Listening => match g.armed_until {
                Some(t) if t > now => DispatchOutcome::Run(mint_guard(&mut g)),
                // Inherited window expired while waiting for SE — the
                // noise-driven-SS case falls to Idle.
                Some(_) => {
                    g.phase = Phase::Idle;
                    g.armed_until = None;
                    DispatchOutcome::ListeningWindowExpired
                }
                // No timer set shouldn't happen in normal flow; be lenient.
                None => DispatchOutcome::Run(mint_guard(&mut g)),
            },
            Phase::ArmedAfterWake => match g.armed_until {
                Some(t) if t > now => DispatchOutcome::Run(mint_guard(&mut g)),
                _ => {
                    g.phase = Phase::Idle;
                    g.armed_until = None;
                    DispatchOutcome::WakeWindowExpired
                }
            },
            Phase::ArmedAfterTurn => match g.armed_until {
                Some(t) if t > now => DispatchOutcome::Run(mint_guard(&mut g)),
                _ => {
                    g.phase = Phase::Idle;
                    g.armed_until = None;
                    DispatchOutcome::TurnWindowExpired
                }
            },
            Phase::Idle => DispatchOutcome::NotArmed,
            Phase::Processing => DispatchOutcome::InTurn,
        }
    }

    /// Called from ProcessingGuard::drop: Processing → ArmedAfterTurn (or
    /// ArmedAfterWake if a barge_in=false wake landed mid-turn). A stale
    /// `gen` (barge-in aborted this turn and a new one already owns
    /// Processing) skips — otherwise the aborted turn's late drop would
    /// clobber the live one's phase.
    fn complete(&self, gen: u64) {
        if !self.required {
            return;
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        if g.turn_generation != gen {
            return;
        }
        if g.phase == Phase::Processing {
            let now = Instant::now();
            if g.pending_wake_after_turn {
                g.phase = Phase::ArmedAfterWake;
                g.armed_until = Some(now + self.wake_window);
                g.pending_wake_after_turn = false;
            } else {
                g.phase = Phase::ArmedAfterTurn;
                g.armed_until = Some(now + self.turn_followup_window);
                g.pending_wake_after_turn = false;
            }
        }
    }

    /// Whether the pipeline should keep sending HTTP downstream; false
    /// after a barge-in flipped state mid-turn. Loose mode: always true.
    pub fn pipeline_still_active(&self) -> bool {
        if !self.required {
            return true;
        }
        let g = self.inner.lock().expect("wake state poisoned");
        g.phase == Phase::Processing
    }

    /// Whether a turn is currently Processing. Loose mode: always false.
    pub fn is_in_turn(&self) -> bool {
        if !self.required {
            return false;
        }
        let g = self.inner.lock().expect("wake state poisoned");
        g.phase == Phase::Processing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(required: bool, wake_ms: u64, turn_ms: u64, barge: bool) -> WakeConfig {
        // Dropout disabled (0) so scenarios that fire SE immediately
        // after wake still dispatch; dropout has its own tests below.
        WakeConfig {
            required,
            wake_window_ms: wake_ms,
            turn_followup_window_ms: turn_ms,
            barge_in: barge,
            post_wake_se_dropout_ms: 0,
        }
    }

    fn cfg_dropout(wake_ms: u64, dropout_ms: u64) -> WakeConfig {
        WakeConfig {
            required: true,
            wake_window_ms: wake_ms,
            turn_followup_window_ms: 10_000,
            barge_in: true,
            post_wake_se_dropout_ms: dropout_ms,
        }
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

    #[test]
    fn idle_drops_speech_ended() {
        let m = mk(cfg(true, 1000, 1000, true));
        assert!(matches!(m.try_dispatch(), DispatchOutcome::NotArmed));
    }

    #[test]
    fn idle_drops_speech_started() {
        let m = mk(cfg(true, 1000, 1000, true));
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::DroppedIdle);
    }

    #[test]
    fn wake_then_ss_then_se_dispatches_via_listening() {
        let m = mk(cfg(true, 1000, 1000, true));
        assert_eq!(m.on_wake(), WakeResult::Armed);
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn wake_then_se_lenient_dispatch() {
        // SE without a preceding SS is accepted while the window holds.
        let m = mk(cfg(true, 1000, 1000, true));
        m.on_wake();
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn wake_window_expires() {
        let m = mk(cfg(true, 1, 10_000, true));
        m.on_wake();
        std::thread::sleep(Duration::from_millis(10));
        assert!(matches!(m.on_speech_started(), SpeechStartedOutcome::WakeWindowExpired));
    }

    #[test]
    fn turn_end_arms_after_turn_window() {
        let m = mk(cfg(true, 1000, 1000, true));
        m.on_wake();
        let g = run_guard(m.try_dispatch()).unwrap();
        drop(g); // turn completes → ArmedAfterTurn
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
    }

    #[test]
    fn turn_followup_window_expires() {
        let m = mk(cfg(true, 1000, 1, true)); // 1 ms follow-up window
        m.on_wake();
        let g = run_guard(m.try_dispatch()).unwrap();
        drop(g); // turn completes → ArmedAfterTurn (1 ms window)
        std::thread::sleep(Duration::from_millis(10));
        assert!(matches!(
            m.on_speech_started(),
            SpeechStartedOutcome::TurnWindowExpired
        ));
    }

    #[test]
    fn wake_during_armed_after_turn_resets_to_armed_after_wake() {
        let m = mk(cfg(true, 1000, 10_000, true));
        m.on_wake();
        let g = run_guard(m.try_dispatch()).unwrap();
        drop(g); // ArmedAfterTurn now
        assert_eq!(m.on_wake(), WakeResult::Armed);
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
    }

    #[test]
    fn always_listening_passes_through() {
        let m = mk(cfg(false, 1000, 1000, true));
        assert_eq!(m.on_wake(), WakeResult::Bypass);
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Bypass);
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn barge_in_returns_bargein_during_processing() {
        let m = mk(cfg(true, 1000, 1000, true));
        m.on_wake();
        let _g = run_guard(m.try_dispatch()).unwrap();
        assert_eq!(m.on_wake(), WakeResult::BargeIn);
    }

    #[test]
    fn no_barge_in_returns_armedbusy_during_processing() {
        let m = mk(cfg(true, 1000, 1000, false));
        m.on_wake();
        let _g = run_guard(m.try_dispatch()).unwrap();
        assert_eq!(m.on_wake(), WakeResult::ArmedBusy);
    }

    #[test]
    fn no_barge_in_pending_wake_drives_completion_to_armed_after_wake() {
        let m = mk(cfg(true, 1000, 10_000, false));
        m.on_wake();
        let g = run_guard(m.try_dispatch()).unwrap();
        assert_eq!(m.on_wake(), WakeResult::ArmedBusy);
        drop(g); // turn ends → ArmedAfterWake (pending wake), not ArmedAfterTurn
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
    }

    #[test]
    fn is_in_turn_tracks_processing_phase() {
        let m = mk(cfg(true, 1000, 1000, true));
        assert!(!m.is_in_turn());
        m.on_wake();
        assert!(!m.is_in_turn());
        let g = run_guard(m.try_dispatch()).unwrap();
        assert!(m.is_in_turn());
        drop(g);
        assert!(!m.is_in_turn()); // Now ArmedAfterTurn, not Processing
    }

    #[test]
    fn se_within_post_wake_dropout_is_dropped() {
        let m = mk(cfg_dropout(5_000, 800));
        m.on_wake();
        assert!(matches!(
            m.try_dispatch(),
            DispatchOutcome::DroppedTooSoonAfterWake
        ));
        // State must remain armed for the operator's real follow-up.
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
    }

    #[test]
    fn se_after_post_wake_dropout_dispatches() {
        let m = mk(cfg_dropout(5_000, 50));
        m.on_wake();
        std::thread::sleep(Duration::from_millis(60));
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn dropout_zero_disables_check() {
        let m = mk(cfg_dropout(5_000, 0));
        m.on_wake();
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn dropout_does_not_gate_followup_after_turn() {
        // Dispatch clears last_wake_at; without the clear, every
        // armed-after-turn dispatch would race the dropout window.
        let m = mk(cfg_dropout(5_000, 50));
        m.on_wake();
        std::thread::sleep(Duration::from_millis(60)); // past 50 ms dropout
        let g = run_guard(m.try_dispatch()).expect("first dispatch");
        drop(g); // → ArmedAfterTurn, last_wake_at now cleared
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn wake_during_listening_restarts_into_armed_after_wake() {
        // "Scratch that, start over": a wake mid-utterance resets to
        // ArmedAfterWake; the in-progress utterance's eventual SE gets
        // dropped by post_wake_se_dropout.
        let m = mk(cfg(true, 5_000, 10_000, true));
        m.on_wake();
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
        assert_eq!(m.on_wake(), WakeResult::Armed);
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
    }

    #[test]
    fn listening_inherits_armed_timer_and_expires() {
        // Regression: armed_until used to be cleared on SS, so a
        // noise-driven SS parked state in Listening with no timeout and a
        // gate-passing SE minutes later still dispatched.
        let m = mk(cfg(true, 50, 10_000, true));
        m.on_wake();
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
        std::thread::sleep(Duration::from_millis(60));
        assert!(matches!(
            m.try_dispatch(),
            DispatchOutcome::ListeningWindowExpired
        ));
        assert!(matches!(m.try_dispatch(), DispatchOutcome::NotArmed));
    }

    #[test]
    fn listening_within_window_still_dispatches() {
        let m = mk(cfg(true, 1_000, 10_000, true));
        m.on_wake();
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn turn_followup_listening_expires_when_se_never_arrives() {
        // The bug from the field: turn ends → ArmedAfterTurn → noise SS →
        // Listening; the follow-up window must still expire.
        let m = mk(cfg(true, 1_000, 50, true));
        m.on_wake();
        let g = run_guard(m.try_dispatch()).unwrap();
        drop(g); // → ArmedAfterTurn (50 ms window)
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
        std::thread::sleep(Duration::from_millis(60));
        assert!(matches!(
            m.try_dispatch(),
            DispatchOutcome::ListeningWindowExpired
        ));
    }

    #[test]
    fn stale_guard_drop_does_not_disturb_new_turn() {
        // Regression: a barge-in-aborted turn's guard Drop can land after
        // a new turn is already Processing; without generation tracking it
        // would roll the new turn back to ArmedAfterTurn.
        let m = mk(cfg(true, 1000, 1000, true));
        m.on_wake();
        let old_guard = run_guard(m.try_dispatch()).unwrap();
        assert_eq!(m.on_wake(), WakeResult::BargeIn);
        let _new_guard = run_guard(m.try_dispatch()).unwrap();
        assert!(m.is_in_turn());
        drop(old_guard);
        assert!(m.is_in_turn(), "stale guard's drop must not roll back the live turn");
    }
}
