"""Wake-word detection shim: audio-io /mic PCM → openWakeWord →
WakeWordDetected POST to the orchestrator; serves /health for compose.
Sink modes: ``orchestrator`` (real POST) or ``dry-run`` (log only — lets the
online manual test run without an orchestrator stack).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import sys
import time
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import websockets
from aiohttp import ClientSession, ClientTimeout, web
from openwakeword.model import Model

VALID_SINK_MODES = ("orchestrator", "dry-run")

# openWakeWord wants chunks that are multiples of 80 ms
# (1280 samples = 2560 bytes) for best efficiency.
FRAME_BYTES = 2560
SAMPLE_RATE_HZ = 16000

# With ?ts=1, audio-io prepends a u64 LE epoch-ns of each frame's *last*
# sample (see audio-io/src/ws.rs).
TS_HEADER_BYTES = 8

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
        # > 0: periodically log the peak score per window for WW_THRESHOLD
        # tuning; 0 (default) keeps the production event-driven cadence.
        self.peak_log_interval = env_float("WW_PEAK_LOG_INTERVAL_SEC", 0.0)
        # Suppress peak logs below this — openWakeWord's tflite alexa model
        # has a ~0.0–0.05 background-noise floor.
        self.peak_log_floor = env_float("WW_PEAK_LOG_FLOOR", 0.05)
        self.framework = os.environ.get("WW_INFERENCE_FRAMEWORK", "tflite")
        listen_str = os.environ.get("WW_LISTEN", "0.0.0.0:7030")
        host, _, port_str = listen_str.rpartition(":")
        if not host or not port_str:
            raise SystemExit(f"WW_LISTEN must be host:port, got {listen_str!r}")
        self.listen_host = host
        self.listen_port = int(port_str)
        self.sink_mode = os.environ.get("WW_SINK_MODE", "orchestrator").lower()
        if self.sink_mode not in VALID_SINK_MODES:
            raise SystemExit(
                f"WW_SINK_MODE must be one of {VALID_SINK_MODES}, got {self.sink_mode!r}"
            )

        # Detect ?ts=1 from the URL; no auto-append, so the offline smoke
        # probe (raw PCM, no header) keeps working.
        try:
            qs = parse_qs(urlparse(self.mic_url).query)
            self.with_ts = qs.get("ts", [""])[0] == "1"
        except Exception:  # noqa: BLE001
            self.with_ts = False

        self.model: Optional[Model] = None
        self.http: Optional[ClientSession] = None
        self.ws_connected = False
        self.last_fire_ts = 0.0
        self.buffer = bytearray()
        # epoch-ns of the newest PCM byte in `self.buffer`; with the bytes
        # still buffered after a drain it yields per-predict end-to-end lag.
        self.last_frame_end_ns = 0
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
        # Model() resolves bare names (e.g. "alexa") against the pre-trained
        # models baked into the image at build time.
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
                            await self.process(bytes(msg))
            except Exception as e:  # noqa: BLE001 — log & reconnect on any failure
                self.ws_connected = False
                log.warning("mic ws error: %s; reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            else:
                self.ws_connected = False

    async def process(self, msg: bytes) -> None:
        # With ?ts=1 each frame is [8B u64 LE epoch-ns][s16le PCM]. The
        # buffer accumulates because audio-io emits 20 ms frames but
        # openWakeWord wants 80 ms chunks — four frames per predict.
        if self.with_ts:
            if len(msg) < TS_HEADER_BYTES:
                log.warning("mic ws: dropping short frame (no ts header), len=%d", len(msg))
                return
            (self.last_frame_end_ns,) = struct.unpack_from("<Q", msg, 0)
            self.buffer.extend(msg[TS_HEADER_BYTES:])
        else:
            self.buffer.extend(msg)
        assert self.model is not None
        while len(self.buffer) >= FRAME_BYTES:
            chunk = bytes(self.buffer[:FRAME_BYTES])
            del self.buffer[:FRAME_BYTES]
            # The drained window's last-sample time is last_frame_end_ns
            # minus the duration of whatever is still buffered (which is
            # all newer than the window).
            window_end_ns = 0
            if self.with_ts:
                remaining_samples = len(self.buffer) // 2
                window_end_ns = self.last_frame_end_ns - int(
                    remaining_samples * 1_000_000_000 / SAMPLE_RATE_HZ
                )
            frame = np.frombuffer(chunk, dtype=np.int16)
            # predict() is sync (ONNX can be tens of ms) and would block
            # /health + the WS read on the event loop; to_thread works because
            # tflite/onnxruntime release the GIL, and the serial await keeps
            # model access single-threaded.
            scores = await asyncio.to_thread(self.model.predict, frame)
            now = time.time()
            if self.with_ts and window_end_ns > 0:
                # capture→predict lag; the headline number for the 1 s
                # wake-word latency budget (mirrors the runtime's e2e_lag_ms).
                e2e_lag_ms = int(now * 1_000) - (window_end_ns // 1_000_000)
                log.debug("predict done e2e_lag_ms=%d", e2e_lag_ms)
            if (now - self.last_fire_ts) < self.cooldown:
                continue
            best_name, best_score = max(scores.items(), key=lambda kv: kv[1])
            if best_score >= self.threshold:
                self.last_fire_ts = now
                asyncio.create_task(self.fire(best_name, float(best_score), now))

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
        # access_log=None: the HEALTHCHECK pings /health every 10 s and
        # aiohttp's per-request INFO line is too chatty for production.
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.listen_host, self.listen_port)
        await site.start()
        log.info("http server on %s:%d", self.listen_host, self.listen_port)


async def main() -> None:
    shim = Shim()
    shim.load_model()
    shim.http = ClientSession(timeout=ClientTimeout(total=5))
    await shim.start_http()
    try:
        await shim.ws_loop()
    finally:
        await shim.http.close()


if __name__ == "__main__":
    # The compose stacks document WW_LOG_LEVEL=DEBUG as the way to surface
    # e2e_lag_ms etc.
    raw_level = os.environ.get("WW_LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(raw_level)
    if not isinstance(level, int):
        print(
            f"WW_LOG_LEVEL={raw_level!r} not recognised; defaulting to INFO",
            file=sys.stderr,
        )
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
