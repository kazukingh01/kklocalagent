"""Smoke test probe.

Runs three roles in one process:

* Mock ASR at POST :9100/inference — accepts multipart (file +
  response_format + temperature), returns {"text": "こんにちは"}.
* Mock LLM at POST :9200/api/chat — accepts ollama-shaped JSON, returns
  {"model": ..., "message": {"role": "assistant", "content": "OK, 了解しました。"}, "done": true}.
* Test driver — waits for orchestrator /health to be 200, then POSTs a
  synthesized SpeechEnded envelope to orchestrator:7000/events with a
  short PCM s16le mono payload, then waits until both mocks have been
  hit.

Exits 0 if both mocks were hit within PROBE_TIMEOUT_SEC and the LLM
received the exact text the ASR returned. Exits 1 on timeout or
mismatch.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Any

import aiohttp
from aiohttp import web

ORCHESTRATOR_URL = os.environ.get("PROBE_ORCH_URL", "http://orchestrator:7000")
ASR_PORT = int(os.environ.get("PROBE_ASR_PORT", "9100"))
LLM_PORT = int(os.environ.get("PROBE_LLM_PORT", "9200"))
TIMEOUT_SEC = float(os.environ.get("PROBE_TIMEOUT_SEC", "30"))
FAKE_TRANSCRIPT = "こんにちは"
FAKE_REPLY = "OK, 了解しました。"

log = logging.getLogger("probe")

asr_hit = asyncio.Event()
llm_hit = asyncio.Event()
# What text the mock LLM saw — we assert it matches the ASR's canned
# response, proving the orchestrator plumbed ASR→LLM in the right order
# rather than just firing both in parallel.
llm_seen_text: list[str] = []


async def asr_handler(request: web.Request) -> web.Response:
    # whisper.cpp server speaks multipart; we don't need to parse the
    # file bytes — just confirm the call shape and record the hit.
    reader = await request.multipart()
    saw_file = False
    saw_response_format = False
    async for part in reader:
        if part.name == "file":
            # Drain the part; size isn't asserted (orchestrator wraps
            # raw PCM in a 44-byte WAV header, so minimum is 44).
            buf = await part.read()
            if len(buf) >= 44 and buf[:4] == b"RIFF":
                saw_file = True
        elif part.name == "response_format":
            text = (await part.read()).decode("utf-8")
            saw_response_format = text == "json"
    if not (saw_file and saw_response_format):
        log.warning(
            "ASR handler: malformed request (file=%s, response_format=%s)",
            saw_file,
            saw_response_format,
        )
        return web.json_response({"error": "bad request"}, status=400)
    log.info("mock ASR hit; returning canned transcript %r", FAKE_TRANSCRIPT)
    asr_hit.set()
    return web.json_response({"text": FAKE_TRANSCRIPT})


async def llm_handler(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        log.warning("LLM handler: bad JSON: %s", e)
        return web.json_response({"error": "bad json"}, status=400)
    messages = body.get("messages") or []
    last = messages[-1] if messages else {}
    user_text = last.get("content", "")
    llm_seen_text.append(user_text)
    log.info("mock LLM hit; received user text %r", user_text)
    llm_hit.set()
    return web.json_response({
        "model": body.get("model", "unknown"),
        "message": {"role": "assistant", "content": FAKE_REPLY},
        "done": True,
    })


async def start_mock_server(port: int, routes: list[tuple[str, str, Any]]) -> web.AppRunner:
    app = web.Application()
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info("mock server listening on :%d", port)
    return runner


async def wait_for_health(session: aiohttp.ClientSession, url: str, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


def build_speech_ended_envelope() -> dict:
    # 100 ms of silence at 16kHz s16le mono — enough for the orchestrator
    # to wrap in WAV and POST. The mock ASR doesn't actually transcribe
    # it, so the contents don't matter.
    pcm = b"\x00" * (16000 * 2 // 10)
    return {
        "name": "SpeechEnded",
        "end_frame_index": 10,
        "duration_frames": 10,
        "utterance_bytes": len(pcm),
        "ts": time.time(),
        "sample_rate": 16000,
        "audio_base64": base64.b64encode(pcm).decode("ascii"),
    }


async def main() -> int:
    # Boot mock backends before orchestrator tries to reach them.
    asr_runner = await start_mock_server(
        ASR_PORT, [("POST", "/inference", asr_handler)]
    )
    llm_runner = await start_mock_server(
        LLM_PORT, [("POST", "/api/chat", llm_handler)]
    )

    deadline = time.monotonic() + TIMEOUT_SEC
    async with aiohttp.ClientSession() as session:
        log.info("waiting for orchestrator /health at %s", ORCHESTRATOR_URL)
        if not await wait_for_health(session, ORCHESTRATOR_URL, deadline):
            log.error("FAIL: orchestrator /health never reached 200")
            return 1
        log.info("orchestrator healthy; sending SpeechEnded")
        envelope = build_speech_ended_envelope()
        async with session.post(
            f"{ORCHESTRATOR_URL}/events",
            json=envelope,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            log.info("POST /events -> %s", r.status)
            if r.status != 200:
                body = await r.text()
                log.error("FAIL: /events returned %s: %s", r.status, body[:200])
                return 1

        # Wait for the pipeline to propagate through both backends.
        try:
            await asyncio.wait_for(
                asyncio.gather(asr_hit.wait(), llm_hit.wait()),
                timeout=max(1.0, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            log.error(
                "FAIL: timeout waiting for mocks — asr_hit=%s, llm_hit=%s",
                asr_hit.is_set(),
                llm_hit.is_set(),
            )
            return 1

    if not llm_seen_text or llm_seen_text[0] != FAKE_TRANSCRIPT:
        log.error(
            "FAIL: LLM was supposed to see %r, saw %r",
            FAKE_TRANSCRIPT,
            llm_seen_text,
        )
        return 1

    log.info("PASS: orchestrator plumbed SpeechEnded → ASR → LLM in order")
    log.info("  ASR returned: %s", FAKE_TRANSCRIPT)
    log.info("  LLM received: %s", llm_seen_text[0])
    log.info("  (canned LLM reply: %s)", FAKE_REPLY)

    await asr_runner.cleanup()
    await llm_runner.cleanup()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    sys.exit(asyncio.run(main()))
