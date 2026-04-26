"""Offline scenario test for the orchestrator's v1.0 control flow.

Boots three orchestrator containers (strict / no-barge / loose) wired
to a single set of in-process mock backends, then drives each orch
through a sequence of `/events` POSTs and asserts which downstream
endpoints were hit (and how many times). No real models — every
assertion is a count, a shape check, or "exact text reached LLM".

Why per-config orch instances rather than reconfiguring one: the
orchestrator reads its config at startup and there's no runtime
toggle. Three lightweight services with different env are simpler
than building a control-plane just for tests, and compose's image
caching means the Rust build runs once across them.

The mocks live in *this* container (shared via :9100..:9400) so
counter state is local — orch instances are stateless w.r.t. mocks.
A scenario calls `mocks.reset()` before sending events so each test
starts from a clean slate.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
import time
from typing import Any, Awaitable, Callable

import aiohttp
from aiohttp import web

# Orch service URLs (compose network resolution).
ORCH_STRICT_URL = os.environ.get("ORCH_STRICT_URL", "http://orch-strict:7000")
ORCH_NO_BARGE_URL = os.environ.get("ORCH_NO_BARGE_URL", "http://orch-no-barge:7000")
ORCH_LOOSE_URL = os.environ.get("ORCH_LOOSE_URL", "http://orch-loose:7000")

# Mock backend ports (in this container).
ASR_PORT = 9100
LLM_PORT = 9200
TTS_PORT = 9300
SINK_PORT = 9400

# strict orch's arm window — must match compose ORCH_WAKE_ARM_WINDOW_MS.
# Used by the "expiry" scenario to know how long to sleep.
ARM_WINDOW_MS = int(os.environ.get("ARM_WINDOW_MS", "2000"))

# Settle interval after a POST /events before reading mock counters.
# `dispatch_utterance` returns 200 to the caller before the spawned
# pipeline task completes — without a sleep here, scenarios race the
# orchestrator's tokio runtime and assert on stale counters.
SETTLE_SEC = float(os.environ.get("SCENARIOS_SETTLE_SEC", "0.5"))

log = logging.getLogger("scenarios")


# --- mock backends ---------------------------------------------------


class Mocks:
    """In-process counters for the four downstream endpoints the
    orchestrator can hit. Every test mutates these by sending events
    to an orch and then reads them after `settle()`."""

    def __init__(self) -> None:
        self.asr_calls: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.tts_speak_calls: list[dict[str, Any]] = []
        self.tts_stop_calls: list[dict[str, Any]] = []
        self.sink_calls: list[dict[str, Any]] = []

        # Barge-in coordination: when `speak_blocks` is True, /speak
        # awaits this event before responding. /stop sets it (so the
        # blocked /speak returns 499). Default behaviour returns 200
        # instantly so non-barge scenarios complete in milliseconds.
        self.speak_blocks = False
        self.speak_release = asyncio.Event()
        self.speak_in_progress = asyncio.Event()  # set while /speak handler is awaiting release

        self.fake_transcript = "hello mock"
        self.fake_reply = "OK from mock LLM"

    def reset(self) -> None:
        self.asr_calls.clear()
        self.llm_calls.clear()
        self.tts_speak_calls.clear()
        self.tts_stop_calls.clear()
        self.sink_calls.clear()
        self.speak_blocks = False
        self.speak_release.clear()
        self.speak_in_progress.clear()

    # --- handlers ---

    async def asr_handler(self, request: web.Request) -> web.Response:
        # Drain multipart but only record the file size; the
        # orchestrator wraps PCM in a 44-byte WAV header.
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
        return web.json_response({"text": self.fake_transcript})

    async def llm_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.llm_calls.append(body)
        return web.json_response(
            {
                "model": body.get("model"),
                "message": {"role": "assistant", "content": self.fake_reply},
                "done": True,
            }
        )

    async def tts_speak_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.tts_speak_calls.append(body)
        if self.speak_blocks:
            self.speak_in_progress.set()
            try:
                # 5 s upper bound — generous vs the 2 s arm window so
                # barge-in scenarios always race well inside it.
                await asyncio.wait_for(self.speak_release.wait(), timeout=5.0)
                # Released by /stop → emulate the real streamer's
                # cancelled response.
                return web.json_response(
                    {"ok": False, "cancelled": True}, status=499
                )
            except asyncio.TimeoutError:
                # Release never came; treat as a timeout (the test
                # would already have failed by now).
                return web.json_response({"ok": True})
        return web.json_response({"ok": True})

    async def tts_stop_handler(self, _: web.Request) -> web.Response:
        self.tts_stop_calls.append({})
        # Release the blocked /speak (no-op if not blocking).
        self.speak_release.set()
        return web.json_response({"ok": True, "cancelled": True})

    async def sink_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.sink_calls.append(body)
        return web.json_response({"ok": True})


# --- helpers ---------------------------------------------------------


def speech_ended(audio_b64: str | None = None) -> dict[str, Any]:
    if audio_b64 is None:
        # 100 ms of silent PCM — orchestrator wraps in WAV header
        # before forwarding to ASR. The mock doesn't decode it.
        pcm = b"\x00" * (16000 * 2 // 10)
        audio_b64 = base64.b64encode(pcm).decode("ascii")
    return {
        "name": "SpeechEnded",
        "ts": time.time(),
        "sample_rate": 16000,
        "audio_base64": audio_b64,
        "end_frame_index": 100,
        "duration_frames": 50,
    }


def speech_ended_no_audio() -> dict[str, Any]:
    return {
        "name": "SpeechEnded",
        "ts": time.time(),
        "sample_rate": 16000,
        "end_frame_index": 100,
        "duration_frames": 50,
    }


def wake() -> dict[str, Any]:
    return {
        "name": "WakeWordDetected",
        "ts": time.time(),
        "model": "alexa",
        "score": 0.95,
    }


async def post_event(
    session: aiohttp.ClientSession, orch_url: str, env: dict[str, Any]
) -> None:
    async with session.post(
        f"{orch_url}/events", json=env, timeout=aiohttp.ClientTimeout(total=5)
    ) as r:
        if r.status != 200:
            body = await r.text()
            raise AssertionError(
                f"POST /events {env['name']} -> {r.status}: {body[:200]}"
            )


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


def expect_at_least(actual: int, minimum: int, label: str) -> None:
    if actual < minimum:
        raise AssertionError(f"{label}: expected >= {minimum}, got {actual}")


# --- scenarios -------------------------------------------------------

# Each scenario is `async (session, mocks) -> None`. It must call
# mocks.reset() at the top, send events, settle, then assert.

Scenario = Callable[[aiohttp.ClientSession, Mocks], Awaitable[None]]


async def scenario_drop_without_wake(
    session: aiohttp.ClientSession, mocks: Mocks
) -> None:
    """SpeechEnded without a preceding wake → fully dropped: no ASR,
    no LLM, no TTS calls, no sink Turn event. The corresponding
    info-level "dropped: not armed" line in orchestrator logs is the
    expected path for whisper-hallucination noise (#4 §10)."""
    mocks.reset()
    await post_event(session, ORCH_STRICT_URL, speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 0, "ASR calls (no wake)")
    expect(len(mocks.llm_calls), 0, "LLM calls (no wake)")
    expect(len(mocks.tts_speak_calls), 0, "TTS speak calls (no wake)")
    # No sink either — neither Wake nor Turn happened.
    expect(len(mocks.sink_calls), 0, "sink calls (no wake)")


async def scenario_wake_then_speech(
    session: aiohttp.ClientSession, mocks: Mocks
) -> None:
    """The happy path: WakeWordDetected → SpeechEnded → ASR → LLM →
    TTS speak. The wake event is forwarded to result_sink immediately;
    the Turn event is forwarded after the pipeline completes. The text
    that reached the LLM must equal what the mock ASR returned —
    proves the orchestrator chained the stages, didn't fire in
    parallel."""
    mocks.reset()
    await post_event(session, ORCH_STRICT_URL, wake())
    await post_event(session, ORCH_STRICT_URL, speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR calls")
    expect(len(mocks.llm_calls), 1, "LLM calls")
    expect(len(mocks.tts_speak_calls), 1, "TTS speak calls")

    llm_user_text = mocks.llm_calls[0]["messages"][-1]["content"]
    if llm_user_text != mocks.fake_transcript:
        raise AssertionError(
            f"LLM user text: expected {mocks.fake_transcript!r}, got {llm_user_text!r}"
        )
    tts_text = mocks.tts_speak_calls[0]["text"]
    if tts_text != mocks.fake_reply:
        raise AssertionError(
            f"TTS text: expected {mocks.fake_reply!r}, got {tts_text!r}"
        )

    sink_names = [s["name"] for s in mocks.sink_calls]
    if "WakeWordDetected" not in sink_names:
        raise AssertionError(f"sink missing WakeWordDetected: {sink_names}")
    if "TurnCompleted" not in sink_names:
        raise AssertionError(f"sink missing TurnCompleted: {sink_names}")


async def scenario_second_speech_dropped(
    session: aiohttp.ClientSession, mocks: Mocks
) -> None:
    """A single wake event is single-use: the second SpeechEnded after
    the same wake (no fresh wake in between) must be dropped. Catches
    the regression where arm_until isn't cleared on dispatch."""
    mocks.reset()
    await post_event(session, ORCH_STRICT_URL, wake())
    await post_event(session, ORCH_STRICT_URL, speech_ended())
    # Settle so the first turn finishes before the second SE arrives —
    # otherwise the in-flight Processing state would also gate the
    # second SE, and we wouldn't be testing the "single-use arm"
    # property in isolation.
    await asyncio.sleep(SETTLE_SEC)
    await post_event(session, ORCH_STRICT_URL, speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR calls (second SE dropped)")
    expect(len(mocks.llm_calls), 1, "LLM calls (second SE dropped)")
    expect(len(mocks.tts_speak_calls), 1, "TTS speak calls (second SE dropped)")


async def scenario_arm_window_expires(
    session: aiohttp.ClientSession, mocks: Mocks
) -> None:
    """A SpeechEnded that arrives more than arm_window_ms after the
    last wake must be dropped — wake is *time-bounded*, not a sticky
    "always armed once primed" flag."""
    mocks.reset()
    await post_event(session, ORCH_STRICT_URL, wake())
    # Arm window in compose is 2 s; sleep 1 s past it.
    await asyncio.sleep(ARM_WINDOW_MS / 1000.0 + 1.0)
    await post_event(session, ORCH_STRICT_URL, speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 0, "ASR calls (window expired)")
    expect(len(mocks.llm_calls), 0, "LLM calls (window expired)")


async def scenario_wake_always_forwarded(
    session: aiohttp.ClientSession, mocks: Mocks
) -> None:
    """Every WakeWordDetected goes to result_sink, regardless of state
    — observers (activity log, future analytics) need every wake even
    when no follow-up utterance arrives."""
    mocks.reset()
    for _ in range(3):
        await post_event(session, ORCH_STRICT_URL, wake())
        await asyncio.sleep(0.05)
    await asyncio.sleep(SETTLE_SEC)
    wake_count = sum(1 for s in mocks.sink_calls if s["name"] == "WakeWordDetected")
    expect(wake_count, 3, "sink WakeWordDetected count")
    expect(len(mocks.asr_calls), 0, "ASR calls (wake-only)")
    expect(len(mocks.llm_calls), 0, "LLM calls (wake-only)")


async def scenario_barge_in(
    session: aiohttp.ClientSession, mocks: Mocks
) -> None:
    """While TTS is mid-speak, a fresh wake must POST /stop to cancel
    the in-flight reply (and re-arm the state machine for the barging
    utterance). The mock TTS holds /speak open until /stop releases
    it — a deterministic stand-in for the real streamer's
    asyncio.Task.cancel() path."""
    mocks.reset()
    mocks.speak_blocks = True

    await post_event(session, ORCH_STRICT_URL, wake())
    await post_event(session, ORCH_STRICT_URL, speech_ended())

    # Wait until the orchestrator has actually entered the TTS speak
    # call — otherwise Wake2 might race the dispatch and arrive while
    # state is still Armed (no barge-in).
    try:
        await asyncio.wait_for(mocks.speak_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("TTS /speak never entered the blocking phase")

    # Now the orchestrator is in Processing. Fire Wake2.
    await post_event(session, ORCH_STRICT_URL, wake())

    # Give the orchestrator's spawned tts_stop() task time to land.
    await asyncio.sleep(SETTLE_SEC)

    expect(len(mocks.tts_speak_calls), 1, "TTS speak calls (barge-in)")
    expect(len(mocks.tts_stop_calls), 1, "TTS stop calls (barge-in)")
    # ASR + LLM should still have been called once each (the original
    # turn ran to the TTS step before being cut).
    expect(len(mocks.asr_calls), 1, "ASR calls (barge-in)")
    expect(len(mocks.llm_calls), 1, "LLM calls (barge-in)")


async def scenario_no_barge_in(
    session: aiohttp.ClientSession, mocks: Mocks
) -> None:
    """With wake.barge_in=false (orch-no-barge), a mid-turn wake event
    must NOT cancel the current TTS — instead it just re-arms for the
    *next* SpeechEnded (the old reply finishes normally)."""
    mocks.reset()
    mocks.speak_blocks = True

    await post_event(session, ORCH_NO_BARGE_URL, wake())
    await post_event(session, ORCH_NO_BARGE_URL, speech_ended())

    try:
        await asyncio.wait_for(mocks.speak_in_progress.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        raise AssertionError("TTS /speak never entered the blocking phase")

    await post_event(session, ORCH_NO_BARGE_URL, wake())
    await asyncio.sleep(SETTLE_SEC)

    # No /stop call — the orch should NOT have tried to cancel.
    expect(len(mocks.tts_stop_calls), 0, "TTS stop calls (no-barge)")
    # /speak is still blocked. Release it so the test cleans up.
    mocks.speak_release.set()
    await asyncio.sleep(SETTLE_SEC)


async def scenario_always_listening(
    session: aiohttp.ClientSession, mocks: Mocks
) -> None:
    """With wake.required=false (orch-loose), every SpeechEnded
    triggers the pipeline — no wake needed. v0.1 fallback for tests
    and for hosts where the wake-word model isn't available."""
    mocks.reset()
    await post_event(session, ORCH_LOOSE_URL, speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR calls (always-listening)")
    expect(len(mocks.llm_calls), 1, "LLM calls (always-listening)")
    expect(len(mocks.tts_speak_calls), 1, "TTS speak calls (always-listening)")


async def scenario_speech_ended_without_audio(
    session: aiohttp.ClientSession, mocks: Mocks
) -> None:
    """SpeechEnded missing audio_base64 — even after a valid wake — is
    skipped before reaching the wake gate. has_utterance_audio() short-
    circuits the dispatch with an info-level log; the wake window
    therefore stays armed for the next utterance."""
    mocks.reset()
    await post_event(session, ORCH_STRICT_URL, wake())
    await post_event(session, ORCH_STRICT_URL, speech_ended_no_audio())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 0, "ASR calls (no audio)")
    expect(len(mocks.llm_calls), 0, "LLM calls (no audio)")

    # The wake window should still be armed — sending a real
    # SpeechEnded now should dispatch.
    await post_event(session, ORCH_STRICT_URL, speech_ended())
    await asyncio.sleep(SETTLE_SEC)
    expect(len(mocks.asr_calls), 1, "ASR calls after recoverable miss")


SCENARIOS: list[tuple[str, Scenario]] = [
    ("drop without wake", scenario_drop_without_wake),
    ("wake then speech (happy path)", scenario_wake_then_speech),
    ("second speech without re-wake is dropped", scenario_second_speech_dropped),
    ("arm window expires", scenario_arm_window_expires),
    ("WakeWordDetected always forwarded to sink", scenario_wake_always_forwarded),
    ("barge-in cancels in-flight TTS", scenario_barge_in),
    ("no-barge: mid-turn wake doesn't cancel", scenario_no_barge_in),
    ("always-listening: SE without wake works", scenario_always_listening),
    ("SpeechEnded without audio_base64 is skipped pre-gate", scenario_speech_ended_without_audio),
]


# --- runner ----------------------------------------------------------


async def start_mock_server(
    name: str, port: int, routes: list[tuple[str, str, Callable]]
) -> web.AppRunner:
    app = web.Application()
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info("mock %s listening on :%d", name, port)
    return runner


async def main() -> int:
    mocks = Mocks()

    runners = [
        await start_mock_server(
            "asr", ASR_PORT, [("POST", "/inference", mocks.asr_handler)]
        ),
        await start_mock_server(
            "llm", LLM_PORT, [("POST", "/api/chat", mocks.llm_handler)]
        ),
        await start_mock_server(
            "tts", TTS_PORT,
            [
                ("POST", "/speak", mocks.tts_speak_handler),
                ("POST", "/stop", mocks.tts_stop_handler),
            ],
        ),
        await start_mock_server(
            "sink", SINK_PORT, [("POST", "/sink", mocks.sink_handler)]
        ),
    ]

    deadline = time.monotonic() + 30.0
    async with aiohttp.ClientSession() as session:
        for name, url in [
            ("strict", ORCH_STRICT_URL),
            ("no-barge", ORCH_NO_BARGE_URL),
            ("loose", ORCH_LOOSE_URL),
        ]:
            log.info("waiting for orch-%s health at %s", name, url)
            await wait_for_health(session, url, deadline)
            log.info("orch-%s healthy", name)

        passed: list[str] = []
        failed: list[tuple[str, str]] = []
        for label, fn in SCENARIOS:
            log.info("--- scenario: %s ---", label)
            try:
                await fn(session, mocks)
            except AssertionError as e:
                log.error("FAIL %r: %s", label, e)
                log.error("  asr_calls=%d llm_calls=%d tts_speak=%d tts_stop=%d sink=%d",
                          len(mocks.asr_calls), len(mocks.llm_calls),
                          len(mocks.tts_speak_calls), len(mocks.tts_stop_calls),
                          len(mocks.sink_calls))
                failed.append((label, str(e)))
                continue
            log.info("PASS %r", label)
            passed.append(label)

    for r in runners:
        await r.cleanup()

    log.info("=" * 60)
    log.info("PASSED %d / %d", len(passed), len(SCENARIOS))
    if failed:
        log.error("FAILED %d:", len(failed))
        for label, err in failed:
            log.error("  - %s: %s", label, err)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    sys.exit(asyncio.run(main()))
