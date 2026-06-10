"""spk-sink: integration-test stub for audio-io's /spk endpoint — counts
received WS frames and exposes the counters at GET /stats."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import websockets
from aiohttp import web

WS_HOST = os.environ.get("WS_HOST", "0.0.0.0")
WS_PORT = int(os.environ.get("WS_PORT", "7010"))
HTTP_HOST = os.environ.get("HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "7011"))

# audio-io wire format (must match): 16 kHz s16le mono, 20 ms = 640 B
EXPECTED_FRAME_BYTES = 640

log = logging.getLogger("spk-sink")

stats = {
    "connections": 0,
    "frames": 0,
    "bytes": 0,
    "non_silent_bytes": 0,
}


async def ws_handler(ws) -> None:
    stats["connections"] += 1
    log.info("ws: client connected (n=%d)", stats["connections"])
    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                stats["frames"] += 1
                stats["bytes"] += len(msg)
                if any(b != 0 for b in msg):
                    stats["non_silent_bytes"] += len(msg)
            else:
                # Text frames aren't part of the audio-io wire format.
                pass
    except websockets.ConnectionClosed:
        pass
    log.info("ws: disconnected; running totals=%s", stats)


async def stats_handler(_: web.Request) -> web.Response:
    return web.json_response(stats)


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def main() -> None:
    app = web.Application()
    app.router.add_get("/stats", stats_handler)
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HTTP_HOST, HTTP_PORT)
    await site.start()
    log.info("http listening on :%d (/stats, /health)", HTTP_PORT)

    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        log.info("ws listening on :%d (any path)", WS_PORT)
        await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    asyncio.run(main())
