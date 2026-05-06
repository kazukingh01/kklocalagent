# livekit-wakeword runtime

## Build (test image)

```sh
cd wake-word-detection/livekit-wakeword/runtime
sudo docker build -t kklocalagent/wake-word-detection:test .
```

## Run (standalone, test mode)

The runtime needs the train-side feature ONNX (mel + embedding,
filenames `melspectrogram.onnx` and `embedding_model.onnx`) mounted
into the container so it uses the same feature extractor that produced
the trained classifier. The image defaults to `/opt/models` for both
the classifier and the feature ONNX, so a single bind mount covers it
all. Sync the train env first if you haven't (`cd ../train && uv sync`).

```sh
TRAIN_RES=$(realpath ../train/.venv/lib/python3.12/site-packages/livekit/wakeword/resources)

# One option: stage classifier + feature ONNX in $(pwd)/models, then
# mount that single dir to /opt/models. Another is to mount the train
# venv resources to /opt/models directly if it already contains the
# classifier you want to test.
mkdir -p models
cp "${TRAIN_RES}/melspectrogram.onnx" "${TRAIN_RES}/embedding_model.onnx" models/

sudo docker run --rm \
    --name ww-test \
    -v "$(pwd)/models:/opt/models:ro" \
    -e WW_MIC_URL=ws://$(ip route show | awk '/default/ {print $3}'):7010/mic \
    -e WW_SINK_MODE=dry-run \
    -e WW_MODEL_PATHS=/opt/models/my_phrase.onnx \
    -e WW_PEAK_LOG_INTERVAL_SEC=1.0 \
    -e WW_PEAK_LOG_FLOOR=0.01 \
    -e RUST_LOG=info,livekit_wakeword_runtime::detector=debug \
    -p 7030:7030 \
    kklocalagent/wake-word-detection:test
```

The image defaults `WW_FEATURE_ONNX_DIR=/opt/models`, so as long as
the two feature ONNX files exist at `/opt/models/{melspectrogram,
embedding_model}.onnx` no extra env var is needed. Override
individual paths with `WW_MEL_ONNX_PATH` / `WW_EMBEDDING_ONNX_PATH`
if the layout differs.

## Run (compose cutover from openwakeword)

```sh
# in compose.yaml -> services.wake-word-detection
#   build.context: ./wake-word-detection/livekit-wakeword/runtime
#   environment:
#     WW_MODEL_PATHS: /opt/models/hey_livekit.onnx
#   volumes:
#     - ./path/to/models:/opt/models:ro   # contains classifier + mel/embedding ONNX
#   (remove WW_MODELS, WW_INFERENCE_FRAMEWORK)

docker compose up -d --build wake-word-detection
```

## Health

```sh
curl -sfS http://localhost:7030/health
```

## Test

```sh
cargo test --release
```

## Logs

```sh
docker compose logs -f wake-word-detection
RUST_LOG=debug docker compose up wake-word-detection
```

## Environment

| Var | Default |
|---|---|
| `WW_MIC_URL` | `ws://audio-io:7010/mic` (the runtime appends `?ts=1` automatically; see below) |
| `WW_ORCHESTRATOR_URL` | `http://orchestrator:7000/events` |
| `WW_MODEL_PATHS` | `/opt/models/hey_livekit.onnx` |
| `WW_FEATURE_ONNX_DIR` | `/opt/models` (must contain `melspectrogram.onnx` + `embedding_model.onnx`) |
| `WW_MEL_ONNX_PATH` | unset → `${WW_FEATURE_ONNX_DIR}/melspectrogram.onnx` |
| `WW_EMBEDDING_ONNX_PATH` | unset → `${WW_FEATURE_ONNX_DIR}/embedding_model.onnx` |
| `WW_THRESHOLD` | `0.5` |
| `WW_COOLDOWN_SEC` | `2.0` |
| `WW_PREDICT_WINDOW_MS` | `2000` |
| `WW_LISTEN` | `0.0.0.0:7030` |
| `WW_SINK_MODE` | `orchestrator` |
| `WW_PEAK_LOG_INTERVAL_SEC` | `0` |
| `WW_PEAK_LOG_FLOOR` | `0.05` |

## End-to-end latency telemetry

The runtime requires audio-io's `?ts=1` mic protocol — each binary frame
is `[u64 LE epoch_ns of the frame's last sample][s16le PCM]`. The
runtime appends `?ts=1` to `WW_MIC_URL` automatically if absent.

Two diagnostics fall out of this:

* `chunk consumed` (TRACE) — `audio_lag_ms = now − frame.end_epoch_ns`
  per arriving frame. Steady-state freshness; rising values mean
  audio-io broadcast / Tokio scheduling / NW is backing up.
* `predict DONE` (DEBUG) — `e2e_lag_ms = now − window_end_epoch_ns`
  measured *after* the predict completes. This is the headline number
  for the "wake-word in 1 s" budget: capture → predict result.

Set `RUST_LOG=livekit_wakeword_runtime=debug` to see `e2e_lag_ms` per
prediction.
