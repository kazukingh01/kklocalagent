"""
HTTP shim that turns a text request into VOICEVOX synthesis + audio-io
playback streaming. The orchestrator POSTs `{"text": "..."}` here on
every successful turn; the shim handles the engine handshake, the
24 kHz → 16 kHz resample, and the framed WS push to audio-io's `/spk`.

Endpoints:
    POST /speak  body: {"text": "..."}     → streams to SPK_URL
    GET  /health                            → 200 once boot finishes

Env (compose-friendly):
    VOICEVOX_URL    base URL of the VOICEVOX engine (default: http://text-to-speech:50021)
    VOICEVOX_SPEAKER speaker id (default: 3 = ずんだもん ノーマル)
    SPK_URL         WS URL to push frames to (e.g. ws://<windows-host>:7010/spk)
    AUDIO_IO_BASE   HTTP base for /start + /spk/stop (e.g. http://<windows-host>:7010)
    HOST / PORT     bind address (default 0.0.0.0:7070)

Concurrency:
    /speak is serialised behind an asyncio.Lock so two overlapping turns
    don't interleave their PCM frames on the same /spk channel. Excess
    requests block until the previous synth+stream finishes — matches
    the orchestrator's max_inflight=1 semantics on the LLM side.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from typing import Optional

import httpx
import websockets
from aiohttp import web

VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://text-to-speech:50021")
VOICEVOX_SPEAKER = int(os.environ.get("VOICEVOX_SPEAKER", "3"))
SPK_URL = os.environ.get("SPK_URL", "")
AUDIO_IO_BASE = os.environ.get("AUDIO_IO_BASE", "")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "7070"))

# audio-io wire format — must match audio-io README:
#   s16le, 16 kHz, mono, 20 ms/frame = 640 bytes/frame.
SAMPLE_RATE = 16000
FRAME_MS = 20
BYTES_PER_FRAME = SAMPLE_RATE // 1000 * FRAME_MS * 2  # 640
FRAME_PERIOD = FRAME_MS / 1000.0

log = logging.getLogger("tts-streamer")

# Tracks the currently running speak task so /stop can cancel it
# (barge-in). Exactly one task at a time — a new /speak cancels the
# previous task before starting, matching audio-io's "newest wins"
# playback semantics.
CURRENT_TASK: asyncio.Task | None = None
TASK_LOCK = asyncio.Lock()


async def synthesize(client: httpx.AsyncClient, text: str, speaker: int) -> bytes:
    q = await client.post("/audio_query", params={"text": text, "speaker": speaker})
    q.raise_for_status()
    r = await client.post(
        "/synthesis",
        params={"speaker": speaker},
        content=q.content,
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.content


def to_pcm_16k_mono(wav_bytes: bytes) -> bytes:
    """Decode arbitrary WAV → raw s16le 16 kHz mono via an ffmpeg pipe.

    Spawned per-call rather than kept alive: the call cost is dwarfed by
    the synthesis time and ffmpeg's own startup is ~10 ms.
    """
    p = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", "pipe:0",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
            "-f", "s16le", "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        check=False,
    )
    if p.returncode != 0:
        # Common cause: VOICEVOX returned a JSON error body (e.g. unknown
        # speaker id) instead of WAV. Surface ffmpeg's diagnostic so the
        # actual cause is visible without per-step printf debugging.
        raise RuntimeError(
            f"ffmpeg rc={p.returncode}: {p.stderr.decode(errors='replace').strip()}"
        )
    return p.stdout


async def push_to_spk(spk_url: str, pcm: bytes) -> int:
    """Stream `pcm` to audio-io's /spk WS in 20 ms frames, paced.

    Returns total bytes sent (including the trailing zero-pad if any but
    excluding the post-utterance silence). Pacing matches FRAME_MS so
    audio-io's playback ring drains naturally rather than buffering the
    whole utterance.
    """
    sent = 0
    async with websockets.connect(spk_url) as ws:
        for i in range(0, len(pcm) - BYTES_PER_FRAME + 1, BYTES_PER_FRAME):
            await ws.send(pcm[i : i + BYTES_PER_FRAME])
            sent += BYTES_PER_FRAME
            await asyncio.sleep(FRAME_PERIOD)
        rem = len(pcm) - sent
        if rem:
            tail = pcm[sent:] + b"\x00" * (BYTES_PER_FRAME - rem)
            await ws.send(tail)
            sent += BYTES_PER_FRAME
            await asyncio.sleep(FRAME_PERIOD)
        # Brief trailing silence so the device buffer flushes the last
        # samples before the WS close.
        for _ in range(10):
            await ws.send(b"\x00" * BYTES_PER_FRAME)
            await asyncio.sleep(FRAME_PERIOD)
    return sent


async def speak_one(text: str) -> dict:
    """One full synthesize + stream pass. Reuses module-level config."""
    if not SPK_URL:
        raise web.HTTPInternalServerError(reason="SPK_URL is not configured")

    log.info("speak: text=%r speaker=%d", text[:80], VOICEVOX_SPEAKER)
    async with httpx.AsyncClient(base_url=VOICEVOX_URL, timeout=60.0) as engine:
        wav = await synthesize(engine, text, VOICEVOX_SPEAKER)

    # ffmpeg is sync; offload so we don't block the event loop while
    # resampling several seconds of audio.
    pcm = await asyncio.to_thread(to_pcm_16k_mono, wav)
    duration_s = len(pcm) / (SAMPLE_RATE * 2)
    log.info("synthesized: wav=%dB → pcm=%dB (%.2fs)", len(wav), len(pcm), duration_s)

    # Idempotent /start so the playback pipeline is up before the WS
    # write. Failures here are non-fatal — older audio-io builds without
    # /start still accept frames.
    if AUDIO_IO_BASE:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                await c.post(f"{AUDIO_IO_BASE}/start")
        except Exception as e:  # noqa: BLE001
            log.warning("POST /start failed (%s); continuing", e)

    sent = await push_to_spk(SPK_URL, pcm)
    log.info("streamed: %dB to %s", sent, SPK_URL)
    return {"ok": True, "wav_bytes": len(wav), "pcm_bytes": len(pcm),
            "sent_bytes": sent, "duration_s": round(duration_s, 3)}


# --- HTTP server ---------------------------------------------------------

async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def speak(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        return web.json_response({"ok": False, "error": f"bad json: {e}"}, status=400)
    text: Optional[str] = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return web.json_response(
            {"ok": False, "error": "text must be a non-empty string"}, status=400
        )

    global CURRENT_TASK
    # Cancel any in-flight task before starting the new one. Two /speak
    # calls overlapping is the barge-in case (orchestrator preferring
    # the new utterance over the old reply); failing the previous task
    # with CancelledError is exactly what we want.
    async with TASK_LOCK:
        if CURRENT_TASK is not None and not CURRENT_TASK.done():
            CURRENT_TASK.cancel()
        task = asyncio.create_task(speak_one(text.strip()))
        CURRENT_TASK = task

    try:
        result = await task
        return web.json_response(result)
    except asyncio.CancelledError:
        # 499 Client Closed Request — surfacing the cancel as a
        # non-success status lets the orchestrator log it as
        # "cancelled" without conflating it with synth/network errors.
        log.info("speak cancelled (barge-in)")
        return web.json_response(
            {"ok": False, "cancelled": True}, status=499
        )
    except Exception as e:  # noqa: BLE001
        log.error("speak failed: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=502)


async def stop(_: web.Request) -> web.Response:
    """Cancel any in-flight /speak task and tell audio-io to drop
    pending playback. Idempotent — safe to call when nothing is
    speaking. Used by orchestrator's barge-in path."""
    global CURRENT_TASK
    cancelled = False
    async with TASK_LOCK:
        if CURRENT_TASK is not None and not CURRENT_TASK.done():
            CURRENT_TASK.cancel()
            cancelled = True

    # Drop already-buffered playback. The cancel above stops further
    # frames from being WS-pushed, but audio-io still has a few
    # hundred ms in its ring — /spk/stop drains it for a clean cut.
    if AUDIO_IO_BASE:
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                await c.post(f"{AUDIO_IO_BASE}/spk/stop")
        except Exception as e:  # noqa: BLE001
            log.warning("POST /spk/stop failed: %s", e)
    return web.json_response({"ok": True, "cancelled": cancelled})


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log.info(
        "tts-streamer starting: voicevox=%s spk=%s speaker=%d",
        VOICEVOX_URL, SPK_URL or "(unset)", VOICEVOX_SPEAKER,
    )
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/speak", speak)
    app.router.add_post("/stop", stop)
    # access_log=None suppresses aiohttp's per-request INFO line. The
    # Dockerfile HEALTHCHECK pings GET /health every 10 s; without
    # this each ping logs a `"GET /health 200 OK"` line that drowns
    # out real /speak / /stop traffic. /speak and /stop already log
    # their own lifecycle messages (synthesised: ..., speak cancelled
    # (barge-in)) at module level, so dropping the access log doesn't
    # hide anything diagnostic.
    web.run_app(
        app, host=HOST, port=PORT, access_log=None, print=lambda *_: None
    )


if __name__ == "__main__":
    main()
