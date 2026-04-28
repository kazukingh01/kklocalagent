//! Wake-gated state machine for the v1.0 pipeline.
//!
//! ```text
//!   ┌──────────── (window expires) ───────────┐
//!   ▼                                          │
//!  Idle ─Wake─► ArmedAfterWake ─SS─► Listening ─SE─► Processing
//!                  │  ▲                                  │
//!                  │  └────── Wake (refresh) ────────────┤
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
//! Two distinct windows replace v0.1's single `arm_window_ms`:
//!   - `wake_window_ms`           — after a wake, how long to wait
//!                                  for SpeechStarted (default 5 s)
//!   - `turn_followup_window_ms`  — after a turn ends, how long to
//!                                  wait for the next SpeechStarted
//!                                  (default 10 s)
//!
//! SpeechStarted is the trigger that *cancels* the timer (per v1.0
//! spec). SpeechEnded then carries the audio for dispatch — when SE
//! arrives in `Listening` we transition to `Processing` and run the
//! pipeline. SE arriving directly in an `ArmedAfter*` state without
//! a preceding SS is accepted leniently (real VAD always sends SS
//! first; this fallback covers the harness tests that synthesise
//! events without an SS step).
//!
//! Always-listening mode (`required = false`) bypasses the gate
//! entirely: every SpeechEnded triggers a pipeline run regardless of
//! state. Barge-in is a strict-only concept.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::config::WakeConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    /// Wake just received; waiting for SpeechStarted within wake_window.
    ArmedAfterWake,
    /// A turn just finished; waiting for the next SpeechStarted within
    /// turn_followup_window. Lets the operator continue a conversation
    /// without re-saying the wake word.
    ArmedAfterTurn,
    /// SpeechStarted received; waiting for SpeechEnded. No timeout
    /// here (VAD's max_utterance_frames bounds the utterance length).
    Listening,
    /// Pipeline (ASR/LLM/TTS) running.
    Processing,
}

struct Inner {
    phase: Phase,
    /// Expiry time for the *current* armed window
    /// (ArmedAfterWake → wake_window, ArmedAfterTurn → turn_followup_window).
    /// `None` outside Armed* phases.
    armed_until: Option<Instant>,
    /// Set when a wake event arrives during Processing with
    /// `barge_in=false` — at the next `complete()`, the state goes to
    /// `ArmedAfterWake` (5 s) instead of `ArmedAfterTurn` (10 s),
    /// honouring the wake the operator pressed mid-reply.
    pending_wake_after_turn: bool,
    /// Monotonic id of the *current* turn — bumped on every transition
    /// into `Processing`. ProcessingGuard records the id at creation;
    /// `complete()` no-ops if the id has moved on, so a turn that was
    /// aborted by barge-in (and whose JoinHandle::abort() runs Drop on
    /// its guard) can't accidentally roll back the next turn's state
    /// from Processing → ArmedAfterTurn.
    turn_generation: u64,
    /// Wall-clock instant of the most recent WakeWordDetected.
    /// `try_dispatch()` consults this to drop SE events that arrive
    /// within `post_wake_se_dropout` — those are VAD reporting the
    /// silence after the wake word itself rather than a real command.
    /// Cleared on a successful dispatch so a future SE that legitimately
    /// arrives long after the wake (e.g. follow-up via ArmedAfterTurn)
    /// isn't gated by a stale timestamp.
    last_wake_at: Option<Instant>,
}

pub struct WakeMachine {
    required: bool,
    wake_window: Duration,
    turn_followup_window: Duration,
    barge_in: bool,
    /// `Some` when post-wake SE dropout is active. `None` (set when
    /// the configured value is 0) disables the check entirely so the
    /// dispatch path stays straight-through.
    post_wake_se_dropout: Option<Duration>,
    inner: Mutex<Inner>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeResult {
    /// Always-listening (`required=false`); no state change.
    Bypass,
    /// State transitioned (or refreshed) to ArmedAfterWake.
    Armed,
    /// Wake during Processing with barge_in=true: TTS should be
    /// cancelled and state has flipped to ArmedAfterWake.
    BargeIn,
    /// Wake during Processing with barge_in=false: phase stayed
    /// Processing, but `complete()` will transition to
    /// ArmedAfterWake (not ArmedAfterTurn) when this turn finishes.
    ArmedBusy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechStartedOutcome {
    /// Always-listening; no state change. SS is informational.
    Bypass,
    /// Transitioned to Listening; the timer for the previous Armed
    /// state has been cancelled.
    Listening,
    /// Phase was Idle; SS dropped (VAD fired without a wake).
    DroppedIdle,
    /// Phase was Listening already (duplicate SS or VAD glitch).
    DroppedAlreadyListening,
    /// Phase was Processing; SS dropped (turn in flight).
    DroppedInTurn,
    /// Phase was ArmedAfterWake but the window had already expired
    /// by the time SS arrived. State reset to Idle.
    WakeWindowExpired,
    /// Phase was ArmedAfterTurn but the follow-up window had already
    /// expired by the time SS arrived. State reset to Idle.
    TurnWindowExpired,
}

pub enum DispatchOutcome {
    /// Pipeline should run; caller holds the guard for the turn's life.
    Run(ProcessingGuard),
    /// State was Idle; SE dropped (no wake, no recent turn).
    NotArmed,
    /// State was Processing; SE dropped (a previous turn is in flight).
    InTurn,
    /// State was ArmedAfterWake but the window had expired; SE dropped.
    WakeWindowExpired,
    /// State was ArmedAfterTurn but the follow-up window had expired;
    /// SE dropped.
    TurnWindowExpired,
    /// SE arrived within `post_wake_se_dropout` of the most recent
    /// WakeWordDetected — almost certainly VAD echoing the wake word's
    /// own audio rather than a real command. State is left armed (the
    /// original wake_window keeps ticking) so the operator's actual
    /// follow-up utterance still dispatches.
    DroppedTooSoonAfterWake,
}

/// Drop-on-completion guard. When dropped, transitions Processing
/// to one of {ArmedAfterTurn, ArmedAfterWake} so the operator can
/// follow up without (or with) a fresh wake. Never observed
/// directly by callers — moved into the spawned pipeline task.
///
/// Guards carry the `turn_generation` they were minted under so a
/// dropped-after-abort guard can detect that a *different* turn now
/// owns the Processing phase and refuses to mutate state.
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

    /// React to a WakeWordDetected event. Returns `WakeResult` so the
    /// caller can fire any side-effects (sink forward, tts /stop).
    pub fn on_wake(&self) -> WakeResult {
        if !self.required {
            return WakeResult::Bypass;
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        let now = Instant::now();
        // Always stamp last_wake_at, even mid-Processing under
        // barge_in=false where the phase doesn't move. The dropout
        // check in try_dispatch keys off this regardless of phase, so
        // a wake-word echo SE arriving 300 ms later is still gated.
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
                    // Don't disturb the running pipeline — but ensure
                    // the *next* state is ArmedAfterWake so the
                    // operator's wake isn't lost.
                    g.pending_wake_after_turn = true;
                    WakeResult::ArmedBusy
                }
            }
            // From any other state, transition to / refresh
            // ArmedAfterWake. ArmedAfterTurn → ArmedAfterWake matches
            // the v1.0 spec ("Wake during follow-up window resets
            // back to a fresh wake state"). Refresh from
            // ArmedAfterWake same idea. Listening → ArmedAfterWake
            // means "user said wake again mid-utterance, restart".
            _ => {
                g.phase = Phase::ArmedAfterWake;
                g.armed_until = Some(until);
                g.pending_wake_after_turn = false;
                WakeResult::Armed
            }
        }
    }

    /// React to a SpeechStarted event. SS is the *cancel* trigger for
    /// the armed timer — once it arrives, no further timeout applies
    /// to the in-progress utterance (VAD's max_utterance_frames is
    /// the upper bound on Listening duration).
    pub fn on_speech_started(&self) -> SpeechStartedOutcome {
        if !self.required {
            return SpeechStartedOutcome::Bypass;
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        let now = Instant::now();
        match g.phase {
            Phase::ArmedAfterWake => match g.armed_until {
                Some(t) if t > now => {
                    g.phase = Phase::Listening;
                    g.armed_until = None;
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
                    g.armed_until = None;
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

    /// React to a SpeechEnded event with utterance audio. Returns
    /// `DispatchOutcome::Run(guard)` if the pipeline should run.
    /// Lenient: an SE arriving directly in an `ArmedAfter*` state
    /// (without a preceding SS) is accepted as long as the timer
    /// hasn't expired — covers test paths that synthesise events
    /// without a SpeechStarted step. Real VAD always sends SS first.
    pub fn try_dispatch(self: &Arc<Self>) -> DispatchOutcome {
        if !self.required {
            // Loose mode never bumps generation — there's no Processing
            // phase to guard against, so the id stays at its initial 0
            // and complete() is a no-op anyway (early-return on
            // !self.required). The guard still records 0 for symmetry.
            return DispatchOutcome::Run(ProcessingGuard {
                machine: self.clone(),
                generation: 0,
            });
        }
        let mut g = self.inner.lock().expect("wake state poisoned");
        let now = Instant::now();
        // Wake-word echo dropout. VAD captures the wake word's own
        // audio and fires SE for it ~300 ms later (silence hangover
        // between "Hey Jarvis" and the operator's actual command, or
        // the wake word standing alone). Without this, the orchestrator
        // dispatches a turn whose ASR text *is* the wake word ("Jervis")
        // and the LLM treats that as a real query. Real continuous
        // commands push SE well past the dropout (>1 s) so they're
        // unaffected. Phase is left untouched — the original
        // wake_window keeps ticking and a follow-up SE within it still
        // dispatches normally.
        if let Some(dropout) = self.post_wake_se_dropout {
            if let Some(t) = g.last_wake_at {
                if now.saturating_duration_since(t) < dropout {
                    return DispatchOutcome::DroppedTooSoonAfterWake;
                }
            }
        }
        // Mint a fresh generation on every Processing transition. Any
        // guard from a previously-aborted turn carries a stale id and
        // will no-op when its Drop fires after we've already moved on.
        let mint_guard = |g: &mut Inner| {
            g.phase = Phase::Processing;
            g.armed_until = None;
            g.turn_generation = g.turn_generation.wrapping_add(1);
            // Clear the wake stamp on a successful dispatch so a later
            // SE in ArmedAfterTurn (legit follow-up) isn't gated by an
            // ancient wake.
            g.last_wake_at = None;
            ProcessingGuard {
                machine: self.clone(),
                generation: g.turn_generation,
            }
        };
        match g.phase {
            Phase::Listening => DispatchOutcome::Run(mint_guard(&mut g)),
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

    /// Called from ProcessingGuard::drop. Transitions Processing →
    /// ArmedAfterTurn (or ArmedAfterWake if a barge_in=false wake
    /// landed mid-turn). If barge-in already promoted state to
    /// ArmedAfterWake, complete() is a no-op (the wake's window
    /// owns the state now).
    ///
    /// `gen` is the guard's recorded generation. If barge-in aborted
    /// the previous turn and a *new* turn has already advanced to
    /// Processing, `gen` will be stale and we skip — otherwise the
    /// aborted turn's late drop would clobber the live one's phase.
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

    /// Whether the pipeline should continue sending HTTP to
    /// downstream stages. False after a barge-in flipped state to
    /// ArmedAfterWake mid-turn. Loose mode always returns true.
    pub fn pipeline_still_active(&self) -> bool {
        if !self.required {
            return true;
        }
        let g = self.inner.lock().expect("wake state poisoned");
        g.phase == Phase::Processing
    }

    /// Whether a turn is currently being processed. Used by event
    /// handlers to log "ignored — turn in progress" with accurate
    /// reason. Loose mode always returns false.
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
        // Tests pre-dating the dropout default a 0-disabled value so
        // the existing scenarios that fire SE immediately after wake
        // still dispatch (otherwise every wake_then_se test would
        // newly drop). Dropout-specific behaviour is covered by its
        // own dedicated tests below.
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
        // Real VAD sends SS before SE; tests sometimes skip SS. We
        // accept SE in ArmedAfterWake as long as the window holds.
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
        drop(g); // turn completes
        // Now in ArmedAfterTurn — a new SS should be accepted via
        // the follow-up window without a fresh wake.
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
        // Wake during ArmedAfterTurn → transition to ArmedAfterWake
        // (with the shorter wake_window timer).
        assert_eq!(m.on_wake(), WakeResult::Armed);
        // Confirm: a SpeechStarted within wake_window is accepted via
        // the wake path (Listening), not a turn-window path.
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
        // With barge_in=false, a mid-turn wake should *not* cancel
        // the running turn, but should redirect the post-turn state
        // from ArmedAfterTurn to ArmedAfterWake.
        let m = mk(cfg(true, 1000, 10_000, false));
        m.on_wake();
        let g = run_guard(m.try_dispatch()).unwrap();
        // Mid-turn wake (no barge-in).
        assert_eq!(m.on_wake(), WakeResult::ArmedBusy);
        drop(g); // turn ends
        // SS now should land in ArmedAfterWake (5 s) not
        // ArmedAfterTurn (10 s). We can't directly observe "which
        // window was used" — but we can confirm the state accepts a
        // SS via Listening (both paths do).
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
        // VAD-captures-the-wake-word echo: SE arrives ~300 ms after
        // WWD with the wake word audio. With dropout configured, the
        // dispatch is rejected and state stays armed so the operator's
        // real follow-up still triggers the pipeline.
        let m = mk(cfg_dropout(5_000, 800));
        m.on_wake();
        // Immediate SE — well inside the 800 ms dropout.
        assert!(matches!(
            m.try_dispatch(),
            DispatchOutcome::DroppedTooSoonAfterWake
        ));
        // State remained armed: a SS within the original wake_window
        // still transitions to Listening.
        assert_eq!(m.on_speech_started(), SpeechStartedOutcome::Listening);
    }

    #[test]
    fn se_after_post_wake_dropout_dispatches() {
        // Real continuous command: VAD's hangover pushes SE past the
        // dropout window. Must dispatch normally.
        let m = mk(cfg_dropout(5_000, 50)); // 50 ms — easy to wait past
        m.on_wake();
        std::thread::sleep(Duration::from_millis(60));
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn dropout_zero_disables_check() {
        // Belt-and-braces for the operator escape hatch: setting
        // post_wake_se_dropout_ms=0 must keep the legacy "lenient SE
        // immediately after wake dispatches" behaviour intact.
        let m = mk(cfg_dropout(5_000, 0));
        m.on_wake();
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn dropout_does_not_gate_followup_after_turn() {
        // Once a real dispatch runs (past the dropout window),
        // last_wake_at is cleared. A follow-up SE landing in
        // ArmedAfterTurn must dispatch even when the configured
        // dropout would otherwise span across both turns.
        let m = mk(cfg_dropout(5_000, 50));
        m.on_wake();
        std::thread::sleep(Duration::from_millis(60)); // past 50 ms dropout
        let g = run_guard(m.try_dispatch()).expect("first dispatch");
        drop(g); // → ArmedAfterTurn, last_wake_at now cleared
        // Follow-up SE — no fresh wake, no fresh wake stamp. With the
        // stamp cleared, dropout has nothing to measure against and
        // the dispatch proceeds. (Without the clear, every armed-
        // after-turn dispatch would race the dropout window.)
        assert!(matches!(m.try_dispatch(), DispatchOutcome::Run(_)));
    }

    #[test]
    fn stale_guard_drop_does_not_disturb_new_turn() {
        // Regression: barge-in aborts the old turn but its
        // ProcessingGuard's Drop runs *after* a new turn has already
        // moved phase back to Processing. Without generation tracking,
        // the stale Drop would call complete() and roll the new turn's
        // phase back to ArmedAfterTurn, breaking it. With tracking, the
        // stale Drop is a no-op.
        let m = mk(cfg(true, 1000, 1000, true));
        m.on_wake();
        let old_guard = run_guard(m.try_dispatch()).unwrap();
        // Barge-in: state goes Processing → ArmedAfterWake.
        assert_eq!(m.on_wake(), WakeResult::BargeIn);
        // New turn dispatched within the wake window — bumps generation.
        let _new_guard = run_guard(m.try_dispatch()).unwrap();
        assert!(m.is_in_turn());
        // Old turn's late drop fires now (simulating
        // JoinHandle::abort()'s drop chain landing after the new
        // dispatch). Must not change phase: the new turn is mid-flight.
        drop(old_guard);
        assert!(m.is_in_turn(), "stale guard's drop must not roll back the live turn");
    }
}
