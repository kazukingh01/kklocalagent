"""Integration test driver / sink.

Receives forwarded events from the orchestrator's `result_sink` and
asserts the expected sequence happened end-to-end:

  1. WakeWordDetected fired (proves wake-word-detection ran on the
     stream and reached the orchestrator).
  2. TurnCompleted fired with non-empty `user` and `assistant` (proves
     VAD → orchestrator → ASR → LLM all worked).
  3. spk-sink received non-zero bytes on its /spk WS (proves the
     orchestrator's TTS stage synthesized the assistant reply and
     streamed it to the audio-io edge).

Exits 0 when all three observed within ASSERT_TIMEOUT_SEC, 1 on
timeout. Logs every received event so failures are diagnosable from
container logs alone.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any

import aiohttp
from aiohttp import web

PORT = int(os.environ.get("ASSERT_PORT", "9300"))
TIMEOUT_SEC = float(os.environ.get("ASSERT_TIMEOUT_SEC", "240"))
SPK_SINK_URL = os.environ.get("SPK_SINK_URL", "http://spk-sink:7011/stats")
SPK_POLL_SEC = float(os.environ.get("SPK_POLL_SEC", "1.0"))
SPK_MIN_BYTES = int(os.environ.get("SPK_MIN_BYTES", "640"))  # at least one frame

log = logging.getLogger("assert")

wake_word_seen = asyncio.Event()
turn_completed_seen = asyncio.Event()
tts_received = asyncio.Event()
last_turn: dict[str, Any] | None = None
last_wake: dict[str, Any] | None = None
last_spk_stats: dict[str, Any] | None = None


async def sink_handler(request: web.Request) -> web.Response:
    global last_turn, last_wake
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        log.warning("malformed JSON on /sink: %s", e)
        return web.json_response({"ok": False}, status=400)
    name = body.get("name")
    log.info("sink received: %s", body if name != "TurnCompleted"
             else {**body, "user": (body.get("user") or "")[:80],
                   "assistant": (body.get("assistant") or "")[:80]})
    if name == "WakeWordDetected":
        last_wake = body
        wake_word_seen.set()
    elif name == "TurnCompleted":
        last_turn = body
        turn_completed_seen.set()
    return web.json_response({"ok": True})


async def poll_spk_sink() -> None:
    """Poll the spk-sink's /stats. Sets `tts_received` once non-silent
    frames have arrived. Logs every distinct change so timing of when
    audio reached the edge is visible in container logs."""
    global last_spk_stats
    last_logged_bytes = -1
    async with aiohttp.ClientSession() as sess:
        while True:
            try:
                async with sess.get(SPK_SINK_URL, timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        last_spk_stats = await r.json()
                        nb = int(last_spk_stats.get("non_silent_bytes", 0))
                        if nb != last_logged_bytes:
                            log.info("spk-sink: %s", last_spk_stats)
                            last_logged_bytes = nb
                        if nb >= SPK_MIN_BYTES and not tts_received.is_set():
                            log.info(
                                "spk-sink threshold met: non_silent_bytes=%d ≥ %d",
                                nb, SPK_MIN_BYTES,
                            )
                            tts_received.set()
            except Exception as e:  # noqa: BLE001
                # First few polls may race the sink coming up.
                log.debug("spk-sink poll failed: %s", e)
            await asyncio.sleep(SPK_POLL_SEC)


async def main() -> int:
    app = web.Application()
    app.router.add_post("/sink", sink_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info(
        "assert sink listening on :%d/sink (timeout=%.0fs, spk_sink=%s)",
        PORT, TIMEOUT_SEC, SPK_SINK_URL,
    )

    poll_task = asyncio.create_task(poll_spk_sink())

    deadline = time.monotonic() + TIMEOUT_SEC
    try:
        await asyncio.wait_for(
            asyncio.gather(
                wake_word_seen.wait(),
                turn_completed_seen.wait(),
                tts_received.wait(),
            ),
            timeout=TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        log.error(
            "FAIL: timeout after %.0fs — wake_word=%s, turn_completed=%s, tts=%s",
            TIMEOUT_SEC,
            wake_word_seen.is_set(),
            turn_completed_seen.is_set(),
            tts_received.is_set(),
        )
        if last_spk_stats:
            log.error("  last spk-sink stats: %s", last_spk_stats)
        poll_task.cancel()
        return 1

    poll_task.cancel()

    assert last_turn is not None
    user = (last_turn.get("user") or "").strip()
    assistant = (last_turn.get("assistant") or "").strip()
    if not user:
        log.error("FAIL: TurnCompleted.user was empty")
        return 1
    if not assistant:
        log.error("FAIL: TurnCompleted.assistant was empty")
        return 1

    log.info("PASS: end-to-end pipeline completed")
    log.info("  wake-word: model=%s score=%.3f",
             last_wake.get("model") if last_wake else "?",
             float((last_wake or {}).get("score") or 0.0))
    log.info("  user (ASR transcript): %r", user)
    log.info("  assistant (LLM reply, first 200 chars): %r", assistant[:200])
    log.info("  tts: %s", last_spk_stats)
    log.info("  elapsed: %.1fs of %.0fs budget",
             TIMEOUT_SEC - max(0.0, deadline - time.monotonic()),
             TIMEOUT_SEC)
    await runner.cleanup()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    sys.exit(asyncio.run(main()))
