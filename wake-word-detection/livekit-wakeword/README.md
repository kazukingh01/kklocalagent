# livekit-wakeword

## Runtime

#### build

```sh
cd ./runtime
sudo docker build -t kklocalagent/wake-word-detection:test .
``` 

#### load models

```bash
TRAIN_RES=$(realpath ../train/.venv/lib/python3.12/site-packages/livekit/wakeword/resources)
mkdir -p models
cp "${TRAIN_RES}/melspectrogram.onnx" "${TRAIN_RES}/embedding_model.onnx" ./models/
cp ../train/output/xxxxxx/xxxxxx.onnx ./models/
```

#### Online test

For WSL2

```bash
sudo docker run --rm \
    --name ww-test \
    -v "$(pwd)/models:/opt/models:ro" \
    -e WW_MIC_URL=ws://$(ip route show | awk '/default/ {print $3}'):7010/mic \
    -e WW_SINK_MODE=dry-run \
    -e WW_MODEL_PATHS=/opt/models/xxxxxx.onnx \
    -e WW_PEAK_LOG_INTERVAL_SEC=1.0 \
    -e WW_PEAK_LOG_FLOOR=0.01 \
    -e RUST_LOG=info,livekit_wakeword_runtime::detector=debug \
    -p 7030:7030 \
    kklocalagent/wake-word-detection:test
```

For Linux

```bash
sudo docker run --rm \
    --name ww-test --network=host \
    -v "$(pwd)/models:/opt/models:ro" \
    -e WW_MIC_URL=ws://127.0.0.1:7010/mic \
    -e WW_SINK_MODE=dry-run \
    -e WW_MODEL_PATHS=/opt/models/xxxxxx.onnx \
    -e WW_PEAK_LOG_INTERVAL_SEC=1.0 \
    -e WW_PEAK_LOG_FLOOR=0.01 \
    -e RUST_LOG=info,livekit_wakeword_runtime::detector=debug \
    -p 7030:7030 \
    kklocalagent/wake-word-detection:test
```

#### Envs

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
| `WW_PREDICT_INTERVAL_MS` | `100` (predict cadence; predictor task fires on this wallclock interval) |
| `WW_LISTEN` | `0.0.0.0:7030` |
| `WW_SINK_MODE` | `orchestrator` |
| `WW_PEAK_LOG_INTERVAL_SEC` | `0` |
| `WW_PEAK_LOG_FLOOR` | `0.05` |

## Train

#### Setup

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ./train/
uv sync
uv run livekit-wakeword setup --config configs/my_phrase.yaml
```

#### Training

```sh
cp configs/example.yaml configs/my_phrase.yaml
$EDITOR configs/my_phrase.yaml
bash train.sh configs/my_phrase.yaml
```

After training, you can get files.

```bash
livekit-wakeword/train$ ll output/xxxxxx/
-rw-rw-r-- 1 aaaaaa aaaaaa  178237  MM DD HH:MM xxxxxx.onnx
```

#### Evaluate

```sh
uv run python eval.py \
    --model output/my_phrase/my_phrase.onnx \
    --recordings ./my_recordings/
```

#### Re-pin dependencies

```sh
uv lock --upgrade
uv sync
```
