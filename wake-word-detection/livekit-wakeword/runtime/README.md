# livekit-wakeword runtime

Rust wake-word service backed by the [`livekit-wakeword`](https://crates.io/crates/livekit-wakeword)
crate's pure-Rust ONNX inference (`ort-tract`). Drop-in replacement for
[`../../openwakeword/`](../../openwakeword/): same wire contract with
audio-io, orchestrator, and compose's `service_healthy` gate.

## Pipeline

```
ws_client ──pcm──▶ detector ──detection──▶ event_sink
                       ▲
                       └─ ring buffer + WakeWordModel::predict
```

* `ws_client` subscribes to audio-io's `/mic` WebSocket. Each binary
  frame (s16le mono 16 kHz, ~20 ms = 320 samples) is decoded to
  `Vec<i16>` and forwarded to the detector.
* `detector` keeps a `WW_PREDICT_WINDOW_MS` (default 2000 ms) ring of
  i16 samples. Once the ring is full, predict is called every 80 ms (=
  the model's own embedding stride). `WakeWordModel::predict` is sync
  and needs `&mut self`, so the call lives inside `spawn_blocking` over
  an `Arc<Mutex<_>>`.
* `event_sink` POSTs `WakeWordDetected` envelopes to the orchestrator,
  or logs them in `dry-run` mode.
* `health` exposes `GET /health` returning `{"ok":bool}` based on
  model-loaded + ws-connected atomics.

Cooldown (`WW_COOLDOWN_SEC`) is enforced in the detector after predict
so a single utterance doesn't generate a burst of events.

## Environment

| Var | Default | Notes |
|---|---|---|
| `WW_MIC_URL` | `ws://audio-io:7010/mic` | source of s16le PCM |
| `WW_ORCHESTRATOR_URL` | `http://orchestrator:7000/events` | POST sink |
| `WW_MODEL_PATHS` | `/opt/models/hey_livekit.onnx` | classifier ONNX(s), comma-separated |
| `WW_THRESHOLD` | `0.5` | per-classifier score gate |
| `WW_COOLDOWN_SEC` | `2.0` | suppress duplicate fires within this window |
| `WW_PREDICT_WINDOW_MS` | `2000` | ring buffer length; predict() needs ≥2 s |
| `WW_LISTEN` | `0.0.0.0:7030` | health server bind |
| `WW_SINK_MODE` | `orchestrator` | `orchestrator` or `dry-run` |
| `WW_PEAK_LOG_INTERVAL_SEC` | `0` | >0 enables periodic peak-score diagnostic |
| `WW_PEAK_LOG_FLOOR` | `0.05` | suppress peak log below this score |

The Dockerfile sets `RUST_LOG`-style filtering via `EnvFilter`; default
is `info,livekit_wakeword_runtime=info`. Override with `RUST_LOG=debug`
for verbose detector output.

## Cutover from openwakeword

In root `compose.yaml`, change:

```yaml
wake-word-detection:
  build:
    context: ./wake-word-detection/openwakeword     # before
    context: ./wake-word-detection/livekit-wakeword/runtime  # after
```

and rename the env var on the same service:

```yaml
WW_MODELS: hey_jarvis           # before (openwakeword bare-name registry)
WW_MODEL_PATHS: /opt/models/hey_livekit.onnx   # after (file paths)
```

`WW_INFERENCE_FRAMEWORK` should be removed — this runtime has only one
backend. All other `WW_*` vars carry over unchanged.

## Tests

```sh
cargo test --release
```

Covers the small pure functions (peak tracker, cooldown predicate).
End-to-end testing is a future milestone (M2 — record real mic audio,
replay through both implementations, compare recall/FPPH).

## Models

The `Dockerfile` fetches the upstream `hey_livekit.onnx` fixture into
`/opt/models/` at build time. Once we publish our own classifiers
(custom-trained via [`../train/`](../train/)) to a GitHub release, the
fetch URL will move and `WW_MODEL_PATHS` will list the local file by
name.

`models/` in this directory is gitignored — no ONNX is committed.
