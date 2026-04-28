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
    "Newest wins" — at most one speak task runs at a time, and a fresh
    /speak cancels the in-flight one before starting (TASK_LOCK guards
    the cancel→replace, not the body). Matches the barge-in case where
    the orchestrator prefers the new utterance over the old reply: the
    cancelled task surfaces as 499 Client Closed Request so the caller
    can distinguish "interrupted" from synth/network errors. The
    orchestrator's per-turn TTS permit normally prevents overlapping
    /speak in the no-barge-in path, so this cancel-on-overlap mostly
    fires during /stop or barge-in.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import struct
import sys
import wave
from typing import Optional

import httpx
import websockets
from aiohttp import web

VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://text-to-speech:50021")
VOICEVOX_SPEAKER = int(os.environ.get("VOICEVOX_SPEAKER", "3"))
# Multiplier applied to the AudioQuery's `speedScale` between
# /audio_query and /synthesis. 1.0 = VOICEVOX default; 1.5 = 50%
# faster ("brisker, less patient" voice agent); 0.8 = slower /
# clearer for accessibility. VOICEVOX accepts roughly 0.5–2.0.
VOICEVOX_SPEED_SCALE = float(os.environ.get("VOICEVOX_SPEED_SCALE", "1.0"))
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
    # /audio_query's response is the canonical input to /synthesis.
    # We mutate three fields and leave every other key untouched so a
    # future VOICEVOX schema change doesn't trip us up:
    #   - speedScale: env-controlled "brisker / slower" voice agent.
    #   - outputSamplingRate: ask VOICEVOX to render at 16 kHz so we
    #     don't have to resample on the wire (audio-io expects 16 k).
    #   - outputStereoToMono: belt-and-braces; VOICEVOX is mono today
    #     but explicit beats implicit when the downstream WS is mono.
    # Together these eliminate the previous ffmpeg pipe stage —
    # synthesis output is already exactly the format /spk wants and
    # we just strip the WAV header.
    params = json.loads(q.content)
    params["outputSamplingRate"] = SAMPLE_RATE
    params["outputStereoToMono"] = True
    if VOICEVOX_SPEED_SCALE != 1.0:
        params["speedScale"] = VOICEVOX_SPEED_SCALE
    body = json.dumps(params).encode()
    r = await client.post(
        "/synthesis",
        params={"speaker": speaker},
        content=body,
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.content


def to_pcm_16k_mono(wav_bytes: bytes) -> bytes:
    """Strip the WAV header from a 16 kHz mono s16le payload returned
    by VOICEVOX's /synthesis (with outputSamplingRate / outputStereoToMono
    set in the AudioQuery JSON — see ``synthesize`` above).

    Validates the WAV header against the expected (16 kHz, mono, s16le)
    shape so a future VOICEVOX upgrade that silently drops support for
    `outputSamplingRate` doesn't ship subtly-wrong audio downstream:
    we'd rather error loudly than play 24 kHz audio at 16 kHz speed.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            n_frames = w.getnframes()
            if sr != SAMPLE_RATE or ch != 1 or sw != 2:
                raise RuntimeError(
                    f"VOICEVOX returned unexpected WAV: rate={sr} channels={ch} "
                    f"sample_width={sw} (expected {SAMPLE_RATE}/1/2). "
                    "The engine may have ignored outputSamplingRate / "
                    "outputStereoToMono in the AudioQuery — check the "
                    "VOICEVOX engine version."
                )
            return w.readframes(n_frames)
    except (wave.Error, EOFError, struct.error) as e:
        # VOICEVOX returns a JSON error body (e.g. unknown speaker id)
        # with content-type audio/wav, so the wave parser fails cleanly.
        # Surface a useful diagnostic instead of a low-level struct error.
        head = wav_bytes[:200].decode(errors="replace") if wav_bytes else "(empty)"
        raise RuntimeError(
            f"failed to parse VOICEVOX response as WAV ({e}); "
            f"head={head!r}"
        ) from e


async def push_to_spk(spk_url: str, pcm: bytes) -> int:
    """Stream `pcm` to audio-io's /spk WS in 20 ms frames, paced.

    Returns when the last frame has been sent — does NOT wait for
    audio-io's playback ring to drain. The drain handshake is
    factored out into the separate /finalize endpoint so an
    intermediate sentence in a multi-sentence turn (e.g. "はい、")
    doesn't block the next /speak: audio-io's ring continues
    feeding the cpal callback while the next sentence's PCM is
    being pushed in, so the speaker hears continuous audio with no
    inter-sentence silence gaps. The orchestrator calls /finalize
    once after the last sentence's /speak.
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
    return sent


async def speak_one(text: str) -> dict:
    """One full synthesize + stream pass. Reuses module-level config."""
    if not SPK_URL:
        raise web.HTTPInternalServerError(reason="SPK_URL is not configured")

    log.info("speak: text=%r speaker=%d", text[:80], VOICEVOX_SPEAKER)
    async with httpx.AsyncClient(base_url=VOICEVOX_URL, timeout=60.0) as engine:
        wav = await synthesize(engine, text, VOICEVOX_SPEAKER)

    # WAV header parsing only — VOICEVOX already returned 16 kHz mono
    # s16le, so this is a header-strip + length validation, not a
    # resample. Cheap enough to run on the event loop directly.
    pcm = to_pcm_16k_mono(wav)
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


async def finalize(_: web.Request) -> web.Response:
    """Wait for audio-io's playback ring to fully drain.

    Called by the orchestrator once after all per-sentence /speak
    posts for a turn have completed. Opens a fresh WS to /spk,
    sends `{"type":"eos"}`, and awaits the matching
    `{"type":"drained"}` reply from audio-io — that reply fires
    only when audio-io's cpal output ring has been fully consumed
    by the device, so /finalize's HTTP return is the precise
    moment the speaker fell silent. The orchestrator uses this
    boundary to open its post-TTS VAD quiet window without
    inheriting audio-io's playback_buffer_ms as a guess.
    """
    if not SPK_URL:
        return web.json_response(
            {"ok": False, "error": "SPK_URL not configured"}, status=500
        )
    start = asyncio.get_event_loop().time()
    try:
        async with websockets.connect(SPK_URL) as ws:
            await ws.send(json.dumps({"type": "eos"}))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if isinstance(raw, (bytes, bytearray)):
                    continue
                if json.loads(raw).get("type") == "drained":
                    break
    except Exception as e:  # noqa: BLE001
        log.error("finalize failed: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=502)
    elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
    log.info("finalize: drained after %.0f ms", elapsed_ms)
    return web.json_response({"ok": True, "drained_ms": round(elapsed_ms, 1)})


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
    app.router.add_post("/finalize", finalize)
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
