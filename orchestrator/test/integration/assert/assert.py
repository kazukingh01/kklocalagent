"""Integration test driver / sink.

Receives forwarded events from the orchestrator's `result_sink` and
asserts the expected sequence happened end-to-end:

  1. WakeWordDetected fired (proves wake-word-detection ran on the
     stream and reached the orchestrator).
  2. TurnCompleted fired with non-empty `user` and `assistant` (proves
     VAD → orchestrator → ASR → LLM all worked).

Exits 0 on both, 1 on timeout. Logs every received event so failures
are diagnosable from container logs alone.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any

from aiohttp import web

PORT = int(os.environ.get("ASSERT_PORT", "9300"))
TIMEOUT_SEC = float(os.environ.get("ASSERT_TIMEOUT_SEC", "180"))

log = logging.getLogger("assert")

wake_word_seen = asyncio.Event()
turn_completed_seen = asyncio.Event()
last_turn: dict[str, Any] | None = None
last_wake: dict[str, Any] | None = None


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


async def main() -> int:
    app = web.Application()
    app.router.add_post("/sink", sink_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info("assert sink listening on :%d/sink (timeout=%.0fs)", PORT, TIMEOUT_SEC)

    deadline = time.monotonic() + TIMEOUT_SEC
    try:
        await asyncio.wait_for(
            asyncio.gather(wake_word_seen.wait(), turn_completed_seen.wait()),
            timeout=TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        log.error(
            "FAIL: timeout after %.0fs — wake_word=%s, turn_completed=%s",
            TIMEOUT_SEC,
            wake_word_seen.is_set(),
            turn_completed_seen.is_set(),
        )
        return 1

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
