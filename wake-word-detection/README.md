# wake-word-detection

Wake-word service for the voice agent. Two implementations live side by
side under this directory; the root `compose.yaml` picks one by hard-
coding its `build.context` (switching is expected to be rare — see
issue #12 for the migration plan).

| Subdir | Engine | Language | Status | Notes |
|---|---|---|---|---|
| [`openwakeword/`](./openwakeword/) | openWakeWord (tflite/onnx) | Python | **current default** | mature, English-biased, custom training requires the upstream notebook |
| [`livekit-wakeword/`](./livekit-wakeword/) | LiveKit conv-attention ONNX | Rust | **WIP** | pure-Rust inference via `ort-tract`, single-command custom training, ONNX is openWakeWord-compatible |

Both implementations expose the same wire contract to the orchestrator:

* subscribe to `audio-io`'s `/mic` WebSocket (s16le mono 16 kHz PCM, 80 ms chunks)
* POST `WakeWordDetected` events to `orchestrator:7000/events`
* expose `/health` on port `7030`

so swapping which one is built does not require touching the
orchestrator, agent, ASR, or TTS services.

## Switching implementations

Edit `compose.yaml`'s `wake-word-detection.build.context` to point at
the desired subdir, then rebuild:

```sh
docker compose build wake-word-detection
docker compose up -d wake-word-detection
```

The env vars consumed by each shim differ — see the per-subdir README
for the relevant `WW_*` variables.
