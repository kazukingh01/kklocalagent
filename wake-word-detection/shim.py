"""Wake-word detection shim.

Consumes s16le-mono-16kHz PCM from audio-io's /mic WebSocket, feeds it
to openWakeWord in 80ms chunks, and POSTs a WakeWordDetected event to
the orchestrator when any configured model's score crosses the
threshold. Also serves /health so compose can gate `service_healthy` on
both model load and WS connection.

Two sink modes (mirrors `voice-activity-detection`'s `SinkMode`):

* ``orchestrator`` (default) — real POST to ``WW_ORCHESTRATOR_URL``.
  Used by the offline smoke (POSTs to a probe sink) and by the
  production compose (POSTs to the orchestrator).
* ``dry-run`` — skip the POST and log the would-be envelope instead.
  Used by the online manual test against a live audio-io: lets you
  speak into a real mic and verify wake-word detection works without
  needing an orchestrator stack running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import websockets
from aiohttp import ClientSession, ClientTimeout, web
from openwakeword.model import Model

VALID_SINK_MODES = ("orchestrator", "dry-run")

# audio-io emits s16le mono @ 16kHz. openWakeWord wants chunks that are
# multiples of 80ms (1280 samples = 2560 bytes) for best efficiency.
FRAME_BYTES = 2560

log = logging.getLogger("wwd")


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


class Shim:
    def __init__(self) -> None:
        self.mic_url = os.environ.get("WW_MIC_URL", "ws://audio-io:7010/mic")
        self.orchestrator_url = os.environ.get(
            "WW_ORCHESTRATOR_URL", "http://orchestrator:7000/events"
        )
        self.model_names = [
            m.strip() for m in os.environ.get("WW_MODELS", "alexa").split(",") if m.strip()
        ]
        self.threshold = env_float("WW_THRESHOLD", 0.5)
        self.cooldown = env_float("WW_COOLDOWN_SEC", 2.0)
        # Diagnostic: when > 0, periodically log the highest score
        # observed in each window so the operator can tune
        # `WW_THRESHOLD` based on actual mic / pronunciation /
        # ambient-noise behaviour. Set to 0 (default) to keep the
        # event-driven log cadence — the "wake never logs unless it
        # fires" semantics that suits production but hides why a
        # particular utterance didn't trigger. Typical values:
        #   5.0 — light diagnostic during threshold tuning
        #   1.0 — aggressive diagnostic for noisy mics
        self.peak_log_interval = env_float("WW_PEAK_LOG_INTERVAL_SEC", 0.0)
        # Floor below which peak-score logging is suppressed even when
        # peak_log_interval is on; avoids spamming the log with the
        # background-noise floor (~0.0–0.05 from openWakeWord's
        # tflite alexa model). Voice that produces scores above this
        # is interesting; below is treated as silence.
        self.peak_log_floor = env_float("WW_PEAK_LOG_FLOOR", 0.05)
        self.framework = os.environ.get("WW_INFERENCE_FRAMEWORK", "tflite")
        self.port = env_int("WW_PORT", 7030)
        self.sink_mode = os.environ.get("WW_SINK_MODE", "orchestrator").lower()
        if self.sink_mode not in VALID_SINK_MODES:
            raise SystemExit(
                f"WW_SINK_MODE must be one of {VALID_SINK_MODES}, got {self.sink_mode!r}"
            )

        self.model: Optional[Model] = None
        self.http: Optional[ClientSession] = None
        self.ws_connected = False
        self.last_fire_ts = 0.0
        self.buffer = bytearray()
        # Per-window peak-score tracker for the diagnostic log path.
        # Reset each time a window is logged.
        self.peak_score = 0.0
        self.peak_model = ""
        self.last_peak_log_ts = 0.0

    def load_model(self) -> None:
        log.info(
            "loading openWakeWord models=%s framework=%s sink=%s",
            self.model_names,
            self.framework,
            self.sink_mode,
        )
        # Model(...) resolves bare names (e.g. "alexa") against the
        # bundled pre-trained models baked into the image at build time.
        self.model = Model(
            wakeword_models=self.model_names,
            inference_framework=self.framework,
        )
        log.info("model loaded: %s", list(self.model.models.keys()))

    async def ws_loop(self) -> None:
        backoff = 1.0
        while True:
            try:
                log.info("connecting to %s", self.mic_url)
                async with websockets.connect(self.mic_url, max_size=None) as ws:
                    self.ws_connected = True
                    backoff = 1.0
                    log.info("mic connected")
                    async for msg in ws:
                        if isinstance(msg, (bytes, bytearray)):
                            self.process(bytes(msg))
            except Exception as e:  # noqa: BLE001 — log & reconnect on any failure
                self.ws_connected = False
                log.warning("mic ws error: %s; reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            else:
                self.ws_connected = False

    def process(self, pcm: bytes) -> None:
        # Re-buffer: audio-io emits 20ms frames but a single WS message
        # may carry several (and TCP coalesces), so slice into 80ms
        # chunks that openWakeWord prefers.
        self.buffer.extend(pcm)
        assert self.model is not None
        while len(self.buffer) >= FRAME_BYTES:
            chunk = bytes(self.buffer[:FRAME_BYTES])
            del self.buffer[:FRAME_BYTES]
            frame = np.frombuffer(chunk, dtype=np.int16)
            scores = self.model.predict(frame)
            now = time.time()
            if (now - self.last_fire_ts) < self.cooldown:
                continue
            # scores: {model_name: float}; take the best over configured models.
            best_name, best_score = max(scores.items(), key=lambda kv: kv[1])
            if best_score >= self.threshold:
                self.last_fire_ts = now
                asyncio.create_task(self.fire(best_name, float(best_score), now))

            # Diagnostic: periodically surface the highest score seen
            # within each window so the operator can tune
            # WW_THRESHOLD against real-world scores. Off by default
            # (peak_log_interval == 0) — production keeps the silent
            # event-driven cadence; threshold tuning sessions enable
            # via env.
            if self.peak_log_interval > 0:
                if best_score > self.peak_score:
                    self.peak_score = float(best_score)
                    self.peak_model = best_name
                if (now - self.last_peak_log_ts) >= self.peak_log_interval:
                    if self.peak_score >= self.peak_log_floor:
                        log.info(
                            "peak score over last %.1fs: model=%s score=%.3f (threshold=%.2f)",
                            self.peak_log_interval, self.peak_model,
                            self.peak_score, self.threshold,
                        )
                    self.peak_score = 0.0
                    self.peak_model = ""
                    self.last_peak_log_ts = now

    async def fire(self, name: str, score: float, ts: float) -> None:
        log.info("detected: model=%s score=%.3f", name, score)
        envelope = {
            "name": "WakeWordDetected",
            "model": name,
            "score": score,
            "ts": ts,
        }
        if self.sink_mode == "dry-run":
            # Online manual test path — surface the envelope without
            # needing a live orchestrator at the other end.
            log.info("[dry-run] would POST WakeWordDetected: %s",
                     json.dumps(envelope, ensure_ascii=False))
            return
        assert self.http is not None
        try:
            async with self.http.post(self.orchestrator_url, json=envelope) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    log.warning(
                        "POST /events -> %s: %s", resp.status, body[:200]
                    )
        except Exception as e:  # noqa: BLE001 — detection events are best-effort
            log.warning("POST /events failed: %s", e)

    async def health(self, _req: web.Request) -> web.Response:
        ok = self.model is not None and self.ws_connected
        return web.json_response({"ok": ok}, status=200 if ok else 503)

    async def start_http(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self.health)
        # access_log=None suppresses aiohttp's per-request INFO line.
        # The Dockerfile HEALTHCHECK pings GET /health every 10 s, so
        # the access logger fires that often by default — too chatty
        # for production. Other paths besides /health don't exist on
        # this server, so dropping the access log doesn't hide
        # anything diagnostic; failures are still surfaced as the
        # response status (503) the orchestrator's healthcheck
        # observes.
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        log.info("http server on :%d", self.port)


async def main() -> None:
    shim = Shim()
    # Synchronous model load is fine — openWakeWord's Model() is blocking
    # and we don't want to accept /mic frames until it's ready anyway.
    shim.load_model()
    shim.http = ClientSession(timeout=ClientTimeout(total=5))
    await shim.start_http()
    try:
        await shim.ws_loop()
    finally:
        await shim.http.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
