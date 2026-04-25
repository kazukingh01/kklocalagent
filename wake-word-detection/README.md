# wake-word-detection

Thin Python shim around [openWakeWord](https://github.com/dscripka/openWakeWord)
(Apache-2.0). Subscribes to `audio-io`'s `/mic` WebSocket, feeds the
PCM stream to openWakeWord in 80ms chunks, and POSTs a
`WakeWordDetected` event to the orchestrator when any configured
model's score crosses the threshold.

## Why openWakeWord + Python (not Rust)

- Bundled English pre-trained models (`alexa`, `hey jarvis`,
  `hey mycroft`, `hey rhasspy`, `current weather`, `timers`) work out
  of the box — no custom training required for v0.1.
- Three-component architecture (melspectrogram → frozen Google
  speech_embedding backbone → small per-word classifier) gives strong
  accuracy on synthetic-only training data.
- Alternatives considered in #4 §9-7:
  - Picovoice Porcupine — commercial, license key required, clashes
    with the local-agent ethos.
  - rustpotter — Rust-native but no bundled English phrases; DTW mode
    needs user recordings, NN mode needs a trained model file.
  - Wyoming-openWakeWord (`rhasspy/wyoming-openwakeword`) — same
    underlying engine but adds a TCP+JSON protocol layer between us
    and the library, for no benefit in an HTTP-first architecture.

## Build

```bash
cd wake-word-detection
docker build -t kklocalagent/wake-word-detection .
```

The build pre-downloads the shared speech-embedding backbone + every
pre-trained classifier (~100 MB) so the container is self-contained
and the first `/mic` frame doesn't race a model pull.

## Run

```bash
docker run --rm \
    -e WW_MIC_URL=ws://host.docker.internal:7010/mic \
    -e WW_ORCHESTRATOR_URL=http://host.docker.internal:7000/events \
    -e WW_MODELS=alexa \
    -p 7030:7030 \
    kklocalagent/wake-word-detection
```

## Environment

| Name | Default | Notes |
|---|---|---|
| `WW_MIC_URL` | `ws://audio-io:7010/mic` | PCM source (s16le mono 16kHz binary frames). |
| `WW_ORCHESTRATOR_URL` | `http://orchestrator:7000/events` | POST target for `WakeWordDetected`. |
| `WW_MODELS` | `alexa` | Comma-separated. Bundled names: `alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`, `current_weather`, `timers`. |
| `WW_THRESHOLD` | `0.5` | Fire when any model score >= this value. |
| `WW_COOLDOWN_SEC` | `2.0` | Minimum interval between two fires (shim-side debounce). |
| `WW_INFERENCE_FRAMEWORK` | `tflite` | `tflite` (Linux default) or `onnx` (Windows-safe). |
| `WW_PORT` | `7030` | HTTP port for `/health`. |

## Event envelope

Matches the shape used by `voice-activity-detection` (`#[serde(tag = "name")]`):

```json
{
  "name": "WakeWordDetected",
  "model": "alexa",
  "score": 0.87,
  "ts": 1744284000.5
}
```

## Health

`GET /health` — `200 {"ok": true}` only when:

1. the openWakeWord model has finished loading, **and**
2. the `/mic` WebSocket is currently connected.

Otherwise returns `503`. The compose `HEALTHCHECK` hits this, so
`depends_on: condition: service_healthy` gates downstream on both
conditions together.

## Smoke test

See `test/README.md` — compose stack with a single-sidecar probe that
serves the WS (streaming `alexa_test.wav`) and receives the HTTP
event.
