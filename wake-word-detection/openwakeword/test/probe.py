"""Smoke test probe.

Dual-role sidecar:
  * WS server on :9100 — streams data/alexa_test.wav as 20ms s16le mono
    16kHz PCM frames (matching audio-io's /mic format), padded with
    leading/trailing silence and looped so the model has enough context.
  * HTTP server on :9200 — receives POST /events from wake-word-detection.
    When a WakeWordDetected event lands, the probe logs "PASS" and
    exits 0. If PROBE_TIMEOUT_SEC elapses first, it exits 1.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import wave
from typing import Any

import websockets
from aiohttp import web

WAV_PATH = os.environ.get("PROBE_WAV", "/data/alexa_test.wav")
WS_PORT = int(os.environ.get("PROBE_WS_PORT", "9100"))
HTTP_PORT = int(os.environ.get("PROBE_HTTP_PORT", "9200"))
TIMEOUT_SEC = float(os.environ.get("PROBE_TIMEOUT_SEC", "60"))

# 20ms frame at 16kHz s16le mono = 320 samples × 2 bytes
FRAME_BYTES = 640
SILENCE_FRAME = b"\x00" * FRAME_BYTES
# predict_clip in openWakeWord defaults to 1s padding; mirror that so
# short WAVs (alexa_test.wav is 0.625s) have enough context.
LEAD_SILENCE_FRAMES = 50   # 1s
TAIL_SILENCE_FRAMES = 50   # 1s
LOOP_COUNT = 3

log = logging.getLogger("probe")

detected = asyncio.Event()
received_payload: dict[str, Any] | None = None


def load_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"need 16kHz, got {w.getframerate()}"
        assert w.getnchannels() == 1, f"need mono, got {w.getnchannels()}"
        assert w.getsampwidth() == 2, f"need 16-bit, got {w.getsampwidth() * 8}"
        return w.readframes(w.getnframes())


async def ws_handler(ws: websockets.WebSocketServerProtocol) -> None:
    log.info("ws client connected")
    pcm = load_pcm(WAV_PATH)
    try:
        # Feed real-time at 20ms/frame to match what audio-io would do
        # off a live mic; openWakeWord's sliding window expects this cadence.
        async def send_frames(frames_iter):
            for frame in frames_iter:
                if detected.is_set():
                    return
                await ws.send(frame)
                await asyncio.sleep(0.02)

        # Pad → WAV → pad; loop a few times to give the model multiple
        # chances. Bail early once the HTTP side has seen an event.
        for loop_i in range(LOOP_COUNT):
            if detected.is_set():
                break
            log.info("ws: streaming loop %d/%d", loop_i + 1, LOOP_COUNT)
            await send_frames([SILENCE_FRAME] * LEAD_SILENCE_FRAMES)
            # Pad the final frame to FRAME_BYTES if the WAV length isn't
            # a multiple.
            wav_frames = [
                pcm[i : i + FRAME_BYTES].ljust(FRAME_BYTES, b"\x00")
                for i in range(0, len(pcm), FRAME_BYTES)
            ]
            await send_frames(wav_frames)
            await send_frames([SILENCE_FRAME] * TAIL_SILENCE_FRAMES)
    except websockets.ConnectionClosed:
        log.info("ws client disconnected")


async def events_handler(request: web.Request) -> web.Response:
    global received_payload
    try:
        received_payload = await request.json()
    except Exception as e:  # noqa: BLE001
        log.warning("bad JSON on /events: %s", e)
        return web.json_response({"ok": False}, status=400)
    log.info("received /events: %s", received_payload)
    if received_payload.get("name") == "WakeWordDetected":
        detected.set()
    return web.json_response({"ok": True})


async def main() -> int:
    app = web.Application()
    app.router.add_post("/events", events_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
    await site.start()
    log.info("http on :%d/events", HTTP_PORT)

    ws_server = await websockets.serve(ws_handler, "0.0.0.0", WS_PORT, max_size=None)
    log.info("ws on :%d", WS_PORT)

    try:
        await asyncio.wait_for(detected.wait(), timeout=TIMEOUT_SEC)
        log.info("PASS: detected %s (score=%.3f)",
                 received_payload.get("model"),
                 float(received_payload.get("score", 0.0)))
        return 0
    except asyncio.TimeoutError:
        log.error("FAIL: no WakeWordDetected in %.1fs", TIMEOUT_SEC)
        return 1
    finally:
        ws_server.close()
        await ws_server.wait_closed()
        await runner.cleanup()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    sys.exit(asyncio.run(main()))
