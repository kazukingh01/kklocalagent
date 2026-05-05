# livekit-wakeword training

## Setup

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run livekit-wakeword setup --config configs/my_phrase.yaml
```

## Train

```sh
cp configs/example.yaml configs/my_phrase.yaml
$EDITOR configs/my_phrase.yaml
bash train.sh configs/my_phrase.yaml
```

## Evaluate

```sh
uv run python eval.py \
    --model output/my_phrase/my_phrase.onnx \
    --recordings ./my_recordings/
```

## Re-pin dependencies

```sh
uv lock --upgrade
uv sync
```
