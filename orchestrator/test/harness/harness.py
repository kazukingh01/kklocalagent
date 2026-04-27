"""Test harness for the orchestrator.

Runs in a single sidecar container alongside the orchestrator under
test. Hosts four mock HTTP backends — the same shape the real
modules expose — so the orchestrator's outbound calls land on
known-shape responses we control:

    :9100  POST /inference     (whisper-style ASR)
    :9200  POST /api/chat      (ollama-style LLM)
    :9300  POST /speak         (tts-streamer speak)
    :9300  POST /stop          (tts-streamer stop)
    :9400  POST /sink          (result_sink forward target)

The same process also drives the orchestrator from the *upstream*
side, impersonating the modules that POST events at it (VAD,
wake-word-detection, future producers). Each test case sends a
specific event sequence and asserts which mock endpoints fired (and
how many times, and with what payload).

Why one harness rather than separate driver+mock containers:
- the mocks must observe what the orchestrator sent in *response to*
  the events we drove — keeping both sides in one process means
  there's no IPC to plumb between "what I sent" and "what landed".
- the barge-in test holds /speak open via asyncio.Event so /stop can
  release it; that coordination is local to one process.

Invocation (test.sh restarts the orchestrator container between
flavors and exec's this for each):

    python harness.py --orch-url http://orch-under-test:7000 \\
                      --flavor strict | loose | no-barge
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
from typing import Any, Awaitable, Callable

import aiohttp
from aiohttp import web

# Mock backend ports — must match what test.sh passes to the
# orchestrator container's ORCH_*_URL env vars. Centralised here so a
# port change touches one place.
ASR_PORT = 9100
LLM_PORT = 9200
TTS_PORT = 9300
SINK_PORT = 9400

# Strict orchestrator's two windows (must agree with test.sh's
# ORCH_WAKE_WINDOW_MS / ORCH_TURN_FOLLOWUP_WINDOW_MS for the strict
# flavor). Used by the "window expires" scenarios to know how long
# to sleep before the next event.
WAKE_WINDOW_MS = 2000
TURN_FOLLOWUP_WINDOW_MS = 2000

# How long to wait after POSTing to /events before reading mock
# counters. The orchestrator returns 200 immediately and runs the
# pipeline on a tokio::spawn task — without a settle interval we'd
# read counters before the spawn drains.
SETTLE_SEC = 0.5

log = logging.getLogger("harness")


# --- mock backends ---------------------------------------------------


class Mocks:
    """Counters + per-call payload capture for each downstream
    endpoint. Reset between tests so each starts from zero."""

    def __init__(self) -> None:
        self.asr_calls: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.tts_speak_calls: list[dict[str, Any]] = []
        self.tts_stop_calls: list[dict[str, Any]] = []
        self.sink_calls: list[dict[str, Any]] = []

        # ASR / LLM response controls. Tests flip these to simulate
        # backend failure modes (real whisper crashing, ollama
        # returning a 503, etc.) and verify the orchestrator's
        # fall-through behaviour.
        self.asr_status = 200
        self.asr_text = "hello mock"
        self.llm_status = 200
        self.llm_content = "OK from mock LLM"
        self.tts_speak_status = 200
        self.sink_status = 200

        # Barge-in coordination knobs. When set, the corresponding
        # mock handler awaits its release event before responding — so
        # the test can observe a "stage in flight" state and fire a
        # WakeWordDetected at exactly the moment we want.
        # `*_in_progress` is set as the handler enters the wait,
        # `*_release` is set by the test (or another handler) to let
        # the mock respond.
        self.asr_blocks = False
        self.asr_release = asyncio.Event()
        self.asr_in_progress = asyncio.Event()

        self.llm_blocks = False
        self.llm_release = asyncio.Event()
        self.llm_in_progress = asyncio.Event()

        self.speak_blocks = False
        self.speak_release = asyncio.Event()
        self.speak_in_progress = asyncio.Event()

    def reset(self) -> None:
        self.asr_calls.clear()
        self.llm_calls.clear()
        self.tts_speak_calls.clear()
        self.tts_stop_calls.clear()
        self.sink_calls.clear()
        self.asr_status = 200
        self.asr_text = "hello mock"
        self.llm_status = 200
        self.llm_content = "OK from mock LLM"
        self.tts_speak_status = 200
        self.sink_status = 200
        self.asr_blocks = False
        self.asr_release.clear()
        self.asr_in_progress.clear()
        self.llm_blocks = False
        self.llm_release.clear()
        self.llm_in_progress.clear()
        self.speak_blocks = False
        self.speak_release.clear()
        self.speak_in_progress.clear()

    # --- handlers ---

    async def asr_handler(self, request: web.Request) -> web.Response:
        # whisper.cpp speaks multipart; record the file size + that
        # response_format=json was sent (orchestrator's expected shape).
        reader = await request.multipart()
        size = 0
        had_response_format = False
        async for part in reader:
            if part.name == "file":
                buf = await part.read()
                size = len(buf)
            elif part.name == "response_format":
                had_response_format = (await part.read()).decode() == "json"
        self.asr_calls.append(
            {"file_bytes": size, "response_format_json": had_response_format}
        )
        # Block before responding when the test sets `asr_blocks` so
        # a barge-in scenario can fire WakeWordDetected at the moment
        # ASR is "in flight" (deterministic stand-in for slow whisper).
        if self.asr_blocks:
            self.asr_in_progress.set()
            try:
                await asyncio.wait_for(self.asr_release.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        if self.asr_status != 200:
            return web.json_response({"error": "mock"}, status=self.asr_status)
        return web.json_response({"text": self.asr_text})

    async def llm_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.llm_calls.append(body)
        # Same block-before-respond pattern as ASR — used by the
        # "barge-in during LLM" scenario.
        if self.llm_blocks:
            self.llm_in_progress.set()
            try:
                await asyncio.wait_for(self.llm_release.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        if self.llm_status != 200:
            return web.json_response({"error": "mock"}, status=self.llm_status)
        # Orchestrator always sends `stream: true` against /api/chat.
        # Mirror ollama's ndjson contract: one line per delta plus a
        # final `{done: true}` line. We emit the entire `llm_content`
        # in a single delta — sufficient for tests asserting on the
        # full reply, and keeps the mock simple. The real ollama emits
        # one line per token; the orchestrator's parser handles either.
        if body.get("stream"):
            response = web.StreamResponse(status=200)
            response.content_type = "application/x-ndjson"
            await response.prepare(request)
            if self.llm_content:
                line = json.dumps(
                    {
                        "model": body.get("model"),
                        "message": {"role": "assistant", "content": self.llm_content},
                        "done": False,
                    }
                ) + "\n"
                await response.write(line.encode())
            final = json.dumps(
                {
                    "model": body.get("model"),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                }
            ) + "\n"
            await response.write(final.encode())
            await response.write_eof()
            return response
        # Non-streaming path kept for completeness — no current caller
        # exercises it, but a plain JSON response is the right thing
        # for any future client that POSTs `stream: false`.
        return web.json_response(
            {
                "model": body.get("model"),
                "message": {"role": "assistant", "content": self.llm_content},
                "done": True,
            }
        )

    async def tts_speak_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.tts_speak_calls.append(body)
        if self.speak_blocks:
            self.speak_in_progress.set()
            try:
                await asyncio.wait_for(self.speak_release.wait(), timeout=5.0)
                # Released by /stop → mirror the real streamer's
                # cancelled-as-499 response so the orchestrator logs
                # it at info, not warn.
                return web.json_response(
                    {"ok": False, "cancelled": True}, status=499
                )
            except asyncio.TimeoutError:
                return web.json_response({"ok": True})
        if self.tts_speak_status != 200:
            return web.json_response(
                {"error": "mock"}, status=self.tts_speak_status
            )
        return web.json_response({"ok": True})

    async def tts_stop_handler(self, _: web.Request) -> web.Response:
        self.tts_stop_calls.append({})
        self.speak_release.set()
        return web.json_response({"ok": True, "cancelled": True})

    async def sink_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.sink_calls.append(body)
        if self.sink_status != 200:
            return web.json_response({"error": "mock"}, status=self.sink_status)
        return web.json_response({"ok": True})


# --- event builders --------------------------------------------------
#
# Each helper builds a JSON envelope shaped the way the corresponding
# real producer sends it. Naming the helpers after the producer
# (`vad_speech_ended`, `wake_word_detected`) makes test reads match
# how a debugger would describe the runtime situation.


def vad_speech_started(frame: int = 0) -> dict[str, Any]:
    return {
        "name": "SpeechStarted",
        "ts": time.time(),
        "frame_index": frame,
        "sample_rate": 16000,
    }


def vad_speech_ended(audio: bytes | None = b"") -> dict[str, Any]:
    """SpeechEnded as VAD emits it — including audio_base64 unless
    `audio` is None (which exercises the orchestrator's
    has_utterance_audio() pre-gate skip)."""
    env = {
        "name": "SpeechEnded",
        "ts": time.time(),
        "sample_rate": 16000,
        "end_frame_index": 100,
        "duration_frames": 50,
    }
    if audio is not None:
        # Default: 100 ms of silent PCM. The mock ASR doesn't decode
        # it, so the contents are irrelevant — only the size shape.
        if audio == b"":
            audio = b"\x00" * (16000 * 2 // 10)
        env["audio_base64"] = base64.b64encode(audio).decode("ascii")
        env["utterance_bytes"] = len(audio)
    return env


def wake_word_detected(model: str = "alexa", score: float = 0.95) -> dict[str, Any]:
    return {
        "name": "WakeWordDetected",
        "ts": time.time(),
        "model": model,
        "score": score,
    }


def unknown_event() -> dict[str, Any]:
    """Future producer the orchestrator hasn't learned about yet —
    forward-compat: orchestrator should ack with 200 and log."""
    return {
        "name": "SomeFutureEvent",
        "ts": time.time(),
        "extra_field": "not in the schema",
    }


# --- helpers ---------------------------------------------------------


async def post_event(
    session: aiohttp.ClientSession, orch_url: str, env: dict[str, Any]
) -> int:
    """POST /events on the orchestrator. Returns the status code so
    forward-compat tests can assert on it directly (most tests just
    require 200)."""
    async with session.post(
        f"{orch_url}/events", json=env, timeout=aiohttp.ClientTimeout(total=5)
    ) as r:
        return r.status


async def wait_for_health(
    session: aiohttp.ClientSession, url: str, deadline: float
) -> None:
    while time.monotonic() < deadline:
        try:
            async with session.get(
                f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)
            ) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(0.5)
    raise AssertionError(f"orchestrator at {url} never became healthy")


def expect(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


# --- test cases ------------------------------------------------------
#
# A "test" here = a coroutine taking (session, orch_url, mocks) that
# asserts behaviour for one specific input pattern. Tests are grouped
# by which orchestrator config they require; test.sh restarts orch
# between groups.

Test = Callable[[aiohttp.ClientSession, str, Mocks], Awaitable[None]]


# --- strict flavor (default v1.0): wake.required=true, barge_in=true ---


async def strict_drop_se_without_wake(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """VAD fires SpeechEnded on background noise — the orchestrator
    must drop it because no wake word preceded it. No backend calls,
    no sink event."""
    mocks.reset()
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 0, "ASR (no wake)")
    expect(len(mocks.llm_calls), 0, "LLM (no wake)")
    expect(len(mocks.tts_speak_calls), 0, "TTS speak (no wake)")
    expect(len(mocks.sink_calls), 0, "sink (no wake)")


async def strict_system_prompt_prepended(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """When ORCH_LLM_SYSTEM_PROMPT is set, the orchestrator must
    prepend a {role:"system"} message before the user turn. Verifies
    role, content, and order — catches regressions where the system
    message slot ends up after `user` (some clients do this and the
    LLM ignores it) or with the wrong role string."""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.llm_calls), 1, "LLM")
    msgs = mocks.llm_calls[0]["messages"]
    if len(msgs) != 2:
        raise AssertionError(f"messages: expected 2 (system+user), got {len(msgs)}: {msgs}")
    if msgs[0].get("role") != "system":
        raise AssertionError(f"messages[0].role: expected 'system', got {msgs[0].get('role')!r}")
    if not msgs[0].get("content"):
        raise AssertionError("messages[0].content is empty — system_prompt didn't propagate")
    if msgs[1].get("role") != "user":
        raise AssertionError(f"messages[1].role: expected 'user', got {msgs[1].get('role')!r}")
    if msgs[1].get("content") != mocks.asr_text:
        raise AssertionError(
            f"messages[1].content: expected ASR text {mocks.asr_text!r}, got {msgs[1].get('content')!r}"
        )


async def strict_happy_path(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """wake-word detects "alexa", VAD fires SpeechEnded → full pipeline.
    Verifies that the text the mock ASR returned is exactly what the
    mock LLM saw, and the LLM's reply is what the mock TTS got
    (the orchestrator chained stages, not fired in parallel)."""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR")
    expect(len(mocks.llm_calls), 1, "LLM")
    expect(len(mocks.tts_speak_calls), 1, "TTS speak")

    saw_text = mocks.llm_calls[0]["messages"][-1]["content"]
    if saw_text != mocks.asr_text:
        raise AssertionError(
            f"LLM saw {saw_text!r}, expected ASR output {mocks.asr_text!r}"
        )
    spoke_text = mocks.tts_speak_calls[0]["text"]
    if spoke_text != mocks.llm_content:
        raise AssertionError(
            f"TTS spoke {spoke_text!r}, expected LLM reply {mocks.llm_content!r}"
        )

    # Both the wake event and the turn completion should reach the
    # result sink (analytics / activity log path).
    sink_names = sorted(s["name"] for s in mocks.sink_calls)
    if sink_names != ["TurnCompleted", "WakeWordDetected"]:
        raise AssertionError(f"sink saw {sink_names}")


async def strict_se_during_turn_dropped(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """A second SpeechEnded that arrives WHILE the previous turn is
    still mid-pipeline must be dropped (no second ASR/LLM/TTS run).
    Distinct from `arm_is_single_use`: that one tests "after the turn
    finishes, the arm is consumed"; this one tests "during the turn
    the gate refuses everything regardless of arm state". The mock
    TTS holds /speak open via speak_blocks so we can observe the
    Processing phase deterministically."""
    mocks.reset()
    mocks.speak_blocks = True

    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    try:
        await asyncio.wait_for(mocks.speak_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("TTS /speak never entered the blocking phase")

    # Now the orchestrator is in Processing. Send another SE.
    # Even with no fresh wake the second SE goes into the gate, which
    # must return InTurn (drop without affecting the running turn).
    await post_event(session, orch, vad_speech_ended())

    # Release the held /speak so the original turn finishes cleanly.
    mocks.speak_release.set()
    await asyncio.sleep(SETTLE_SEC)

    # Exactly one of each — the second SE was dropped.
    expect(len(mocks.asr_calls), 1, "ASR (mid-turn SE dropped)")
    expect(len(mocks.llm_calls), 1, "LLM (mid-turn SE dropped)")
    expect(len(mocks.tts_speak_calls), 1, "TTS speak (mid-turn SE dropped)")


async def strict_followup_within_window_dispatches(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """After a Turn ends, the next SpeechEnded within
    `turn_followup_window_ms` must dispatch a *second* full pipeline
    run *without* a fresh wake — that's the v1.0 follow-up
    semantics. (Replaces the v0.x "single-use arm" scenario which
    expected the second SE to be dropped.)"""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    # Let the first turn drain so state transitions Processing →
    # ArmedAfterTurn (the test fixture window is 2 s — plenty).
    await asyncio.sleep(SETTLE_SEC)
    # Now in ArmedAfterTurn. A SpeechEnded *without* a fresh wake
    # should still dispatch via the follow-up window.
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 2, "ASR (follow-up dispatched)")
    expect(len(mocks.llm_calls), 2, "LLM (follow-up dispatched)")
    expect(len(mocks.tts_speak_calls), 2, "TTS (follow-up dispatched)")


async def strict_followup_window_expires(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """If no SpeechEnded arrives within
    `turn_followup_window_ms` of a turn ending, the state returns
    to Idle and the next SE is dropped (operator must re-wake)."""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)  # turn drains → ArmedAfterTurn
    # Sleep past the turn-followup window.
    await asyncio.sleep(TURN_FOLLOWUP_WINDOW_MS / 1000.0 + 0.5)
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR (follow-up window expired)")
    expect(len(mocks.llm_calls), 1, "LLM (follow-up window expired)")


async def strict_wake_during_followup_resets_to_armed_after_wake(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """A WakeWordDetected during the post-turn follow-up window
    transitions state back to ArmedAfterWake (resetting the timer
    to wake_window_ms). Spec: "B 中の WWD は A に遷移".

    Observable consequence: a second wake mid-followup must still
    forward to sink, and the next SpeechEnded must dispatch via the
    refreshed window. We don't directly observe "which window was
    used", but we *do* observe that wake_calls_to_sink increments
    and the dispatch succeeds."""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)  # ArmedAfterTurn
    # A second wake during ArmedAfterTurn → ArmedAfterWake.
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 2, "ASR (wake-during-followup re-arms)")
    wakes = sum(1 for s in mocks.sink_calls if s["name"] == "WakeWordDetected")
    expect(wakes, 2, "sink WakeWordDetected (both wakes forwarded)")


async def strict_wake_window_expires(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """A wake that's gone stale by the time SpeechEnded arrives must
    not dispatch — the user said the wake word, then went silent
    long enough that we shouldn't trust the next noise as "the
    intended utterance". Specifically tests the post-wake window
    (5 s default; test fixture sets 2 s)."""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await asyncio.sleep(WAKE_WINDOW_MS / 1000.0 + 0.5)
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 0, "ASR (wake window expired)")
    expect(len(mocks.llm_calls), 0, "LLM (wake window expired)")


async def strict_wake_always_to_sink(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """Every WakeWordDetected forwards to result_sink — even when no
    follow-up SpeechEnded arrives. Activity logs care about every
    wake regardless of whether it produced a turn."""
    mocks.reset()
    for _ in range(3):
        await post_event(session, orch, wake_word_detected())
        await asyncio.sleep(0.05)
    await asyncio.sleep(SETTLE_SEC)
    wakes = sum(1 for s in mocks.sink_calls if s["name"] == "WakeWordDetected")
    expect(wakes, 3, "sink WakeWordDetected count")
    expect(len(mocks.asr_calls), 0, "ASR (wake-only)")


async def strict_barge_in_during_asr_aborts(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """A wake event arriving while ASR is in flight must abort the
    rest of the turn — no LLM call, no sink TurnCompleted forward,
    no TTS speak. The original ASR HTTP request still completes (we
    don't cancel awaits mid-request), so asr_calls counts 1, but
    everything downstream gets zero. Verifies the post-ASR
    `wake.is_in_turn()` check in run_turn."""
    mocks.reset()
    mocks.asr_blocks = True

    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    try:
        await asyncio.wait_for(mocks.asr_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("ASR /inference never entered the blocking phase")

    # Now the orchestrator is in Processing, blocked inside ASR.
    # Fire barge-in — orch flips state Processing → Armed and POSTs
    # tts /stop (no-op because no /speak yet).
    await post_event(session, orch, wake_word_detected())

    # Release ASR so run_turn can reach its post-ASR check.
    mocks.asr_release.set()
    await asyncio.sleep(SETTLE_SEC)

    expect(len(mocks.asr_calls), 1, "ASR (was in flight, completed)")
    expect(len(mocks.llm_calls), 0, "LLM (post-ASR abort)")
    expect(len(mocks.tts_speak_calls), 0, "TTS (post-ASR abort)")
    turns = sum(1 for s in mocks.sink_calls if s["name"] == "TurnCompleted")
    expect(turns, 0, "sink TurnCompleted (post-ASR abort)")


async def strict_barge_in_during_llm_aborts(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """Same as the ASR variant but the wake fires while the LLM is
    in flight. ASR has already completed, so asr_calls=1 and
    llm_calls=1, but TTS and the TurnCompleted sink forward are both
    skipped — the post-LLM `wake.is_in_turn()` check catches it."""
    mocks.reset()
    mocks.llm_blocks = True

    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    try:
        await asyncio.wait_for(mocks.llm_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("LLM /api/chat never entered the blocking phase")

    # Barge-in mid-LLM.
    await post_event(session, orch, wake_word_detected())

    mocks.llm_release.set()
    await asyncio.sleep(SETTLE_SEC)

    expect(len(mocks.asr_calls), 1, "ASR (completed pre-barge-in)")
    expect(len(mocks.llm_calls), 1, "LLM (was in flight, completed)")
    expect(len(mocks.tts_speak_calls), 0, "TTS (post-LLM abort)")
    turns = sum(1 for s in mocks.sink_calls if s["name"] == "TurnCompleted")
    expect(turns, 0, "sink TurnCompleted (post-LLM abort)")


async def strict_barge_in_cancels_tts(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """User says "alexa" again while the assistant is still replying
    — the orchestrator must POST /stop on tts-streamer to cut the
    reply, and re-arm for the next utterance."""
    mocks.reset()
    mocks.speak_blocks = True

    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    try:
        await asyncio.wait_for(mocks.speak_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("TTS /speak never entered the blocking phase")

    # Second wake mid-/speak triggers barge-in.
    await post_event(session, orch, wake_word_detected())
    await asyncio.sleep(SETTLE_SEC)

    expect(len(mocks.tts_stop_calls), 1, "TTS /stop (barge-in)")
    expect(len(mocks.tts_speak_calls), 1, "TTS /speak (barge-in)")
    expect(len(mocks.asr_calls), 1, "ASR (barge-in)")
    expect(len(mocks.llm_calls), 1, "LLM (barge-in)")


async def strict_unknown_event_is_acked(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """A future producer adds a new event name before the orchestrator
    learns about it. The orchestrator must ack with 200 and trigger
    no downstream calls — forward-compat guard for upstream rollouts
    that lead the orchestrator's release schedule."""
    mocks.reset()
    status = await post_event(session, orch, unknown_event())
    expect(status, 200, "POST /events status (unknown event)")
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 0, "ASR (unknown event)")
    expect(len(mocks.llm_calls), 0, "LLM (unknown event)")
    expect(len(mocks.sink_calls), 0, "sink (unknown event)")


async def strict_speech_started_alone(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """SpeechStarted is informational — orchestrator should log it
    but never fire backend calls (utterance audio comes with
    SpeechEnded). Crash-safety: even with no following SpeechEnded,
    the orchestrator stays healthy."""
    mocks.reset()
    await post_event(session, orch, vad_speech_started(frame=42))
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 0, "ASR (SpeechStarted alone)")
    expect(len(mocks.llm_calls), 0, "LLM (SpeechStarted alone)")
    expect(len(mocks.sink_calls), 0, "sink (SpeechStarted alone)")


async def strict_se_without_audio_preserves_arm(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """A SpeechEnded missing audio_base64 (e.g., VAD config glitch)
    is skipped *before* reaching the wake gate. The arm window
    therefore stays intact — the next *valid* SpeechEnded inside the
    window still dispatches. Catches the regression where the
    "audio missing" log accidentally consumed the arm."""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended(audio=None))
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 0, "ASR (audio missing)")

    # Arm should still be live.
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR after recovery")


async def strict_asr_500_blocks_pipeline(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """ASR backend returns 500 — orchestrator must short-circuit:
    no LLM call, no TTS, no TurnCompleted to sink. Wake forward to
    sink still happens (it ran before the failure)."""
    mocks.reset()
    mocks.asr_status = 500
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR was tried")
    expect(len(mocks.llm_calls), 0, "LLM (ASR failed)")
    expect(len(mocks.tts_speak_calls), 0, "TTS (ASR failed)")
    turns = sum(1 for s in mocks.sink_calls if s["name"] == "TurnCompleted")
    expect(turns, 0, "sink TurnCompleted (ASR failed)")


async def strict_asr_empty_text_skips_llm(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """ASR returned 200 but with empty `text` (silence misclassified
    as speech). orchestrator skips LLM/TTS — there's no user input
    to respond to. Same fall-through as a 500, just a different
    branch in pipeline.rs."""
    mocks.reset()
    mocks.asr_text = ""
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR was tried")
    expect(len(mocks.llm_calls), 0, "LLM (ASR returned empty)")
    expect(len(mocks.tts_speak_calls), 0, "TTS (ASR returned empty)")


async def strict_llm_500_blocks_tts(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """LLM backend returns 500 — orchestrator skips TTS and the
    TurnCompleted forward. No half-turn artifacts in the activity
    log."""
    mocks.reset()
    mocks.llm_status = 500
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR was tried")
    expect(len(mocks.llm_calls), 1, "LLM was tried")
    expect(len(mocks.tts_speak_calls), 0, "TTS (LLM failed)")
    turns = sum(1 for s in mocks.sink_calls if s["name"] == "TurnCompleted")
    expect(turns, 0, "sink TurnCompleted (LLM failed)")


async def strict_llm_empty_reply_skips_tts(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """LLM returned 200 with empty `content`. The orchestrator still
    forwards TurnCompleted (the turn *did* complete, just with an
    empty assistant reply) but skips TTS — there's nothing to speak
    and the streamer would 400 on an empty text."""
    mocks.reset()
    mocks.llm_content = ""
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR")
    expect(len(mocks.llm_calls), 1, "LLM")
    expect(len(mocks.tts_speak_calls), 0, "TTS (empty reply)")
    turns = sum(1 for s in mocks.sink_calls if s["name"] == "TurnCompleted")
    expect(turns, 1, "sink TurnCompleted (still emitted)")


async def strict_tts_500_does_not_block_turn(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """TTS backend errors — the assistant reply is already in the
    sink (TurnCompleted forwarded *before* the speak attempt), so
    failing TTS is a best-effort warning. The user just doesn't hear
    the reply, but the activity log is intact."""
    mocks.reset()
    mocks.tts_speak_status = 500
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.tts_speak_calls), 1, "TTS was tried")
    turns = sum(1 for s in mocks.sink_calls if s["name"] == "TurnCompleted")
    expect(turns, 1, "sink TurnCompleted (despite TTS fail)")


async def strict_sink_500_does_not_break_pipeline(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """result_sink errors — orchestrator logs a warning but the next
    turn must still run. Sink is an observer; it must not be in the
    critical path."""
    mocks.reset()
    mocks.sink_status = 500
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    # All four downstream stages still fired.
    expect(len(mocks.asr_calls), 1, "ASR (sink down)")
    expect(len(mocks.llm_calls), 1, "LLM (sink down)")
    expect(len(mocks.tts_speak_calls), 1, "TTS (sink down)")


# --- loose flavor: wake.required=false (always-listening) ----------


async def loose_se_alone_dispatches(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """v0.1 fallback path: with the wake gate disabled, every
    SpeechEnded triggers the pipeline directly. Useful for hosts
    where the wake-word model isn't available."""
    mocks.reset()
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR (loose)")
    expect(len(mocks.llm_calls), 1, "LLM (loose)")
    expect(len(mocks.tts_speak_calls), 1, "TTS (loose)")


# --- no-barge flavor: wake.required=true, barge_in=false ------------


async def no_barge_mid_turn_wake_does_not_cancel(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """barge_in=false: a fresh wake during TTS does NOT cut the reply
    — used by deployments that prefer "let the assistant finish, the
    user will wait" UX over interruptibility. /stop must not fire."""
    mocks.reset()
    mocks.speak_blocks = True

    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    try:
        await asyncio.wait_for(mocks.speak_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("TTS /speak never entered the blocking phase")

    await post_event(session, orch, wake_word_detected())
    await asyncio.sleep(SETTLE_SEC)

    expect(len(mocks.tts_stop_calls), 0, "TTS /stop (barge_in=false)")

    # Release the held /speak so the harness shuts down cleanly.
    mocks.speak_release.set()
    await asyncio.sleep(SETTLE_SEC)


# --- groups -----------------------------------------------------------

GROUPS: dict[str, list[tuple[str, Test]]] = {
    "strict": [
        ("drop SE without wake", strict_drop_se_without_wake),
        ("system prompt prepended as {role:'system'}", strict_system_prompt_prepended),
        ("happy path: wake → SE → full pipeline", strict_happy_path),
        ("SE during in-flight turn is dropped", strict_se_during_turn_dropped),
        ("follow-up SE within turn window dispatches", strict_followup_within_window_dispatches),
        ("turn-followup window expires", strict_followup_window_expires),
        ("wake during follow-up resets to wake-window", strict_wake_during_followup_resets_to_armed_after_wake),
        ("wake window expires", strict_wake_window_expires),
        ("WakeWordDetected always forwarded to sink", strict_wake_always_to_sink),
        ("barge-in during ASR aborts pipeline", strict_barge_in_during_asr_aborts),
        ("barge-in during LLM aborts pipeline", strict_barge_in_during_llm_aborts),
        ("barge-in cancels in-flight TTS", strict_barge_in_cancels_tts),
        ("unknown event name is acked (forward-compat)", strict_unknown_event_is_acked),
        ("SpeechStarted alone is logged only", strict_speech_started_alone),
        ("SE without audio preserves arm window", strict_se_without_audio_preserves_arm),
        ("ASR 500 blocks pipeline (no LLM/TTS/Turn)", strict_asr_500_blocks_pipeline),
        ("ASR empty text skips LLM", strict_asr_empty_text_skips_llm),
        ("LLM 500 blocks TTS + Turn", strict_llm_500_blocks_tts),
        ("LLM empty reply: Turn forwarded, TTS skipped", strict_llm_empty_reply_skips_tts),
        ("TTS 500 doesn't block TurnCompleted", strict_tts_500_does_not_block_turn),
        ("sink 500 doesn't break pipeline", strict_sink_500_does_not_break_pipeline),
    ],
    "loose": [
        ("always-listening: SE without wake dispatches", loose_se_alone_dispatches),
    ],
    "no-barge": [
        ("mid-turn wake doesn't cancel TTS", no_barge_mid_turn_wake_does_not_cancel),
    ],
}


# --- runner ----------------------------------------------------------


async def start_mock_servers(mocks: Mocks) -> list[web.AppRunner]:
    runners: list[web.AppRunner] = []
    for port, routes in [
        (ASR_PORT, [("POST", "/inference", mocks.asr_handler)]),
        (LLM_PORT, [("POST", "/api/chat", mocks.llm_handler)]),
        (
            TTS_PORT,
            [
                ("POST", "/speak", mocks.tts_speak_handler),
                ("POST", "/stop", mocks.tts_stop_handler),
            ],
        ),
        (SINK_PORT, [("POST", "/sink", mocks.sink_handler)]),
    ]:
        app = web.Application()
        for method, path, handler in routes:
            app.router.add_route(method, path, handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        runners.append(runner)
        log.info("mock listening on :%d", port)
    return runners


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orch-url", required=True)
    ap.add_argument(
        "--flavor", required=True, choices=sorted(GROUPS.keys()),
        help="which test group to run; matches the orchestrator's wake config",
    )
    args = ap.parse_args()

    mocks = Mocks()
    runners = await start_mock_servers(mocks)

    deadline = time.monotonic() + 30.0
    async with aiohttp.ClientSession() as session:
        log.info("waiting for orch /health at %s", args.orch_url)
        await wait_for_health(session, args.orch_url, deadline)
        log.info("orch healthy; running flavor=%s", args.flavor)

        passed: list[str] = []
        failed: list[tuple[str, str]] = []
        for label, fn in GROUPS[args.flavor]:
            log.info("--- %s ---", label)
            try:
                await fn(session, args.orch_url, mocks)
            except AssertionError as e:
                log.error("FAIL: %s", e)
                log.error(
                    "  asr=%d llm=%d tts_speak=%d tts_stop=%d sink=%d",
                    len(mocks.asr_calls), len(mocks.llm_calls),
                    len(mocks.tts_speak_calls), len(mocks.tts_stop_calls),
                    len(mocks.sink_calls),
                )
                failed.append((label, str(e)))
                continue
            log.info("PASS")
            passed.append(label)

    for r in runners:
        await r.cleanup()

    log.info("=" * 60)
    log.info("flavor=%s  PASSED %d / %d", args.flavor, len(passed), len(passed) + len(failed))
    if failed:
        for label, err in failed:
            log.error("  FAIL %s: %s", label, err)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    sys.exit(asyncio.run(main()))
