"""Test harness for the orchestrator: hosts mock backends (ASR / LLM /
TTS / sink) and drives the orchestrator's /events from the same process.

One process (not separate driver+mock containers) because the barge-in
tests hold /speak open via asyncio.Event so /stop can release it — that
coordination must stay local.
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

# Must match what test.sh passes to the orchestrator's ORCH_*_URL env vars.
ASR_PORT = 9100
LLM_PORT = 9200
TTS_PORT = 9300
SINK_PORT = 9400

# Must agree with test.sh's ORCH_WAKE_WINDOW_MS / ORCH_TURN_FOLLOWUP_WINDOW_MS
# for the strict flavor.
WAKE_WINDOW_MS = 2000
TURN_FOLLOWUP_WINDOW_MS = 2000

# Wait after POSTing /events before reading mock counters: the orchestrator
# returns 200 immediately and runs the pipeline on a tokio::spawn task.
SETTLE_SEC = 0.5

log = logging.getLogger("harness")


class Mocks:
    """Counters + per-call payload capture for each downstream endpoint."""

    def __init__(self) -> None:
        self.asr_calls: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.tts_speak_calls: list[dict[str, Any]] = []
        self.tts_stop_calls: list[dict[str, Any]] = []
        self.sink_calls: list[dict[str, Any]] = []

        self.asr_status = 200
        self.asr_text = "hello mock"
        self.llm_status = 200
        self.llm_content = "OK from mock LLM"
        self.tts_speak_status = 200
        self.sink_status = 200

        # Barge-in coordination: when `*_blocks` is set, the mock handler
        # awaits `*_release` before responding so a test can fire a wake at
        # an exact "stage in flight" moment; `*_in_progress` is set as the
        # handler enters the wait.
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

    async def asr_handler(self, request: web.Request) -> web.Response:
        # whisper.cpp speaks multipart.
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
        if self.llm_blocks:
            self.llm_in_progress.set()
            try:
                await asyncio.wait_for(self.llm_release.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        if self.llm_status != 200:
            return web.json_response({"error": "mock"}, status=self.llm_status)
        # Mirror ollama's ndjson contract: one line per delta plus a final
        # `{done: true}` line. The whole `llm_content` goes in one delta;
        # real ollama emits one line per token — the parser handles either.
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
        # Non-streaming path: no current caller exercises it.
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
                # Released by /stop → mirror the real streamer's cancelled-
                # as-499 response so the orchestrator logs info, not warn.
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


def vad_speech_started(frame: int = 0) -> dict[str, Any]:
    return {
        "name": "SpeechStarted",
        "ts": time.time(),
        "frame_index": frame,
        "sample_rate": 16000,
    }


def vad_speech_ended(audio: bytes | None = b"") -> dict[str, Any]:
    """SpeechEnded as VAD emits it; audio=None omits audio_base64
    (exercises the has_utterance_audio() pre-gate skip)."""
    env = {
        "name": "SpeechEnded",
        "ts": time.time(),
        "sample_rate": 16000,
        "end_frame_index": 100,
        "duration_frames": 50,
    }
    if audio is not None:
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
    """Event name the orchestrator hasn't learned about (forward-compat)."""
    return {
        "name": "SomeFutureEvent",
        "ts": time.time(),
        "extra_field": "not in the schema",
    }


async def post_event(
    session: aiohttp.ClientSession, orch_url: str, env: dict[str, Any]
) -> int:
    """POST /events on the orchestrator; returns the status code."""
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


Test = Callable[[aiohttp.ClientSession, str, Mocks], Awaitable[None]]


async def strict_drop_se_without_wake(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """SpeechEnded with no preceding wake must be dropped entirely."""
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
    """ORCH_LLM_SYSTEM_PROMPT must arrive as {role:"system"} *before* the
    user turn — catches the regression where the system message lands after
    `user` (some clients do this and the LLM ignores it)."""
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
    """wake → SE → full pipeline; payload equality across stages proves the
    orchestrator chained ASR→LLM→TTS rather than firing in parallel."""
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

    sink_names = sorted(s["name"] for s in mocks.sink_calls)
    if sink_names != ["TurnCompleted", "WakeWordDetected"]:
        raise AssertionError(f"sink saw {sink_names}")


async def strict_se_during_turn_dropped(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """A second SpeechEnded arriving WHILE a turn is mid-pipeline must be
    dropped regardless of arm state. speak_blocks holds /speak open so the
    Processing phase is observable deterministically."""
    mocks.reset()
    mocks.speak_blocks = True

    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    try:
        await asyncio.wait_for(mocks.speak_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("TTS /speak never entered the blocking phase")

    # Orchestrator is now in Processing; the gate must return InTurn for
    # this SE (drop without affecting the running turn).
    await post_event(session, orch, vad_speech_ended())

    mocks.speak_release.set()
    await asyncio.sleep(SETTLE_SEC)

    expect(len(mocks.asr_calls), 1, "ASR (mid-turn SE dropped)")
    expect(len(mocks.llm_calls), 1, "LLM (mid-turn SE dropped)")
    expect(len(mocks.tts_speak_calls), 1, "TTS speak (mid-turn SE dropped)")


async def strict_followup_within_window_dispatches(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """SpeechEnded within `turn_followup_window_ms` of a turn ending must
    dispatch *without* a fresh wake. (Replaces the v0.x "single-use arm"
    scenario, which expected the second SE to be dropped.)"""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)  # turn drains → ArmedAfterTurn
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 2, "ASR (follow-up dispatched)")
    expect(len(mocks.llm_calls), 2, "LLM (follow-up dispatched)")
    expect(len(mocks.tts_speak_calls), 2, "TTS (follow-up dispatched)")


async def strict_followup_window_expires(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """After `turn_followup_window_ms` with no SpeechEnded, state returns
    to Idle and the next SE is dropped."""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)  # turn drains → ArmedAfterTurn
    await asyncio.sleep(TURN_FOLLOWUP_WINDOW_MS / 1000.0 + 0.5)
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR (follow-up window expired)")
    expect(len(mocks.llm_calls), 1, "LLM (follow-up window expired)")


async def strict_wake_during_followup_resets_to_armed_after_wake(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """Wake during the follow-up window → ArmedAfterWake (timer resets to
    wake_window_ms). Spec: "B 中の WWD は A に遷移". Which window was used
    isn't directly observable; we assert the sink forward + dispatch."""
    mocks.reset()
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)  # ArmedAfterTurn
    # Second wake during ArmedAfterTurn → ArmedAfterWake.
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 2, "ASR (wake-during-followup re-arms)")
    wakes = sum(1 for s in mocks.sink_calls if s["name"] == "WakeWordDetected")
    expect(wakes, 2, "sink WakeWordDetected (both wakes forwarded)")


async def strict_wake_window_expires(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """A stale wake (older than the post-wake window) must not dispatch."""
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
    """Every WakeWordDetected forwards to result_sink, turn or no turn."""
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
    """Wake while ASR is in flight must abort the rest of the turn (no
    LLM/TTS/TurnCompleted). asr_calls still counts 1: the mock appends at
    request entry, before its release-event await — the orchestrator drops
    the connection on abort, but the append has already fired."""
    mocks.reset()
    mocks.asr_blocks = True

    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    try:
        await asyncio.wait_for(mocks.asr_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("ASR /inference never entered the blocking phase")

    # Orchestrator is blocked inside ASR. Barge-in flips Processing → Armed
    # and POSTs tts /stop (no-op because no /speak yet).
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
    """ASR variant with the wake mid-LLM: asr_calls=1 and llm_calls=1, but
    the task abort cancels the streaming /api/chat response (closing the
    connection so ollama would stop generating) before any sentence
    reaches the TTS consumer."""
    mocks.reset()
    mocks.llm_blocks = True

    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    try:
        await asyncio.wait_for(mocks.llm_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("LLM /api/chat never entered the blocking phase")

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
    """Wake during /speak must POST tts /stop and re-arm."""
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

    expect(len(mocks.tts_stop_calls), 1, "TTS /stop (barge-in)")
    expect(len(mocks.tts_speak_calls), 1, "TTS /speak (barge-in)")
    expect(len(mocks.asr_calls), 1, "ASR (barge-in)")
    expect(len(mocks.llm_calls), 1, "LLM (barge-in)")


async def strict_unknown_event_is_acked(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """Unknown event names must be acked with 200 and trigger nothing —
    forward-compat guard for upstream rollouts that lead the orchestrator."""
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
    """SpeechStarted is informational — no backend calls, stays healthy."""
    mocks.reset()
    await post_event(session, orch, vad_speech_started(frame=42))
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 0, "ASR (SpeechStarted alone)")
    expect(len(mocks.llm_calls), 0, "LLM (SpeechStarted alone)")
    expect(len(mocks.sink_calls), 0, "sink (SpeechStarted alone)")


async def strict_se_without_audio_preserves_arm(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """A SpeechEnded missing audio_base64 is skipped *before* the wake
    gate, leaving the arm window intact. Catches the regression where the
    "audio missing" path accidentally consumed the arm."""
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
    """ASR 500 → short-circuit: no LLM, no TTS, no TurnCompleted."""
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
    """ASR 200 with empty `text` (silence misclassified as speech) skips
    LLM/TTS."""
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
    """LLM 500 → no TTS, no TurnCompleted (no half-turn artifacts)."""
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
    """LLM 200 with empty `content`: TurnCompleted is still forwarded but
    TTS is skipped — the streamer would 400 on empty text."""
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
    """TTS failure is best-effort: TurnCompleted is forwarded *before* the
    speak attempt, so the activity log stays intact."""
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
    """result_sink is an observer — its failure must not block the turn."""
    mocks.reset()
    mocks.sink_status = 500
    await post_event(session, orch, wake_word_detected())
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR (sink down)")
    expect(len(mocks.llm_calls), 1, "LLM (sink down)")
    expect(len(mocks.tts_speak_calls), 1, "TTS (sink down)")


async def loose_se_alone_dispatches(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """Wake gate disabled: every SpeechEnded dispatches directly."""
    mocks.reset()
    await post_event(session, orch, vad_speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR (loose)")
    expect(len(mocks.llm_calls), 1, "LLM (loose)")
    expect(len(mocks.tts_speak_calls), 1, "TTS (loose)")


async def no_barge_mid_turn_wake_does_not_cancel(
    session: aiohttp.ClientSession, orch: str, mocks: Mocks
) -> None:
    """barge_in=false: a wake during TTS must NOT fire /stop."""
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

    mocks.speak_release.set()
    await asyncio.sleep(SETTLE_SEC)


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
