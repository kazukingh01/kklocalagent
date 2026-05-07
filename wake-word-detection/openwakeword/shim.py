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

# audio-io emits s16le mono @ 16kHz. openWakeWord wants chunks that are
# multiples of 80ms (1280 samples = 2560 bytes) for best efficiency.
FRAME_BYTES = 2560
SAMPLE_RATE_HZ = 16000

# When ?ts=1 is in WW_MIC_URL, audio-io prepends a u64 LE epoch-ns to
# each Binary frame (the wall-clock time of the frame's *last* sample).
# 8 bytes; documented in audio-io/src/ws.rs and mirrored in the
# livekit-wakeword Rust runtime.
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

        # Header opt-in: audio-io's `?ts=1` prepends a per-frame
        # epoch-ns header. Detect it from the configured URL so the
        # offline smoke probe (which sends raw PCM) still works
        # untouched while a real audio-io connection benefits from the
        # header. No auto-append — the URL is the source of truth.
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
        # Updated each WS message when `with_ts` is on. Holds the
        # epoch-ns of the most recent PCM byte we've appended to
        # `self.buffer`. Combined with how many bytes remain in the
        # buffer after a predict-chunk drain, it lets us compute the
        # end-to-end lag (capture → predict result) per prediction.
        self.last_frame_end_ns = 0
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
                            await self.process(bytes(msg))
            except Exception as e:  # noqa: BLE001 — log & reconnect on any failure
                self.ws_connected = False
                log.warning("mic ws error: %s; reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            else:
                self.ws_connected = False

    async def process(self, msg: bytes) -> None:
        # When ?ts=1 is in effect, each WS message is one audio-io
        # frame: [8B u64 LE epoch-ns of last sample][640B s16le PCM].
        # We strip the header, remember its timestamp, and feed only
        # the PCM into `self.buffer`. The buffer mechanism still
        # exists because audio-io emits 20 ms frames and openWakeWord
        # wants 80 ms chunks; we accumulate four frames per predict.
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
            # After draining FRAME_BYTES from the front (oldest), the
            # remaining buffer holds samples newer than the predict
            # window. So the time of the window's last sample is
            # last_frame_end_ns minus the duration of whatever's still
            # buffered. Only meaningful when with_ts is on.
            window_end_ns = 0
            if self.with_ts:
                remaining_samples = len(self.buffer) // 2
                window_end_ns = self.last_frame_end_ns - int(
                    remaining_samples * 1_000_000_000 / SAMPLE_RATE_HZ
                )
            frame = np.frombuffer(chunk, dtype=np.int16)
            # `model.predict` is a synchronous ML call — for tflite the
            # default `alexa` model it's ~1–2 ms, but ONNX or larger
            # models are tens of ms per chunk. Running it directly on
            # the event loop blocks /health, the fire-and-forget POST,
            # and the WS read for that long. `to_thread` puts it on the
            # default executor so the loop stays responsive; tflite /
            # onnxruntime release the GIL during inference, so the
            # other coros really do run in parallel. Single-threaded
            # access to the model is preserved because we await each
            # call serially — the worker pool sees one predict at a
            # time per shim instance.
            scores = await asyncio.to_thread(self.model.predict, frame)
            now = time.time()
            if self.with_ts and window_end_ns > 0:
                # End-to-end lag = "now" minus mic time of the last
                # sample in the predicted 80 ms window. Mirrors the
                # livekit-wakeword runtime's e2e_lag_ms diagnostic;
                # this is the headline number for the 1 s wake-word
                # latency budget. DEBUG so production stays quiet but
                # threshold-tuning sessions (RUST_LOG-equivalent
                # `--log-level DEBUG` via env) can see it.
                e2e_lag_ms = int(now * 1_000) - (window_end_ns // 1_000_000)
                log.debug("predict done e2e_lag_ms=%d", e2e_lag_ms)
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
        site = web.TCPSite(runner, self.listen_host, self.listen_port)
        await site.start()
        log.info("http server on %s:%d", self.listen_host, self.listen_port)


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
