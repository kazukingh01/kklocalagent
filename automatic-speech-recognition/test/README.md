# test/ — vad → asr smoke tests

```text
[offline]
mic-stub ── ws://mic-stub:7010/mic ──► vad ── POST /inference (WAV) ──► asr

[online]
audio-io (Windows) ── ws://${WINDOWS_HOST}:7010/mic ──► vad ── POST /inference (WAV) ──► asr
```

## Setup

```bash
cd automatic-speech-recognition
bash ./fetch-models.sh ggml-small-q8_0.bin    # ~250 MB into ./models
```

```bash
# Synthesize a Japanese test wav into test/samples (~270 KB).
# Uses gTTS in a one-shot docker container; needs internet.
bash ./test/fetch-sample-ja.sh
cd ./test
```

## Test offline

```bash
sudo docker compose -f compose.offline.yaml up --build
```

## Test offline

```bash
echo "WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')" > .env
sudo docker compose -f compose.online.yaml up --build
```
