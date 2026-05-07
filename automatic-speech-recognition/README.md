# automatic-speech-recognition

whisper.cpp サーバを kklocalagent 用にパッケージしたもの。
`ghcr.io/ggml-org/whisper.cpp` を thin-wrap し `/inference` を公開。

# Test

```text
[offline]
mic-stub ── ws://mic-stub:7010/mic ──► vad ── POST /inference (WAV) ──► asr

[online]
audio-io (Windows) ── ws://${WINDOWS_HOST}:7010/mic ──► vad ── POST /inference (WAV) ──► asr
```

## Setup

```bash
bash ./fetch-models.sh ggml-small-q8_0.bin    # ~250 MB into ./models
```

```bash
# Synthesize a Japanese test wav into test/samples (~270 KB).
# Uses gTTS in a one-shot docker container; needs internet.
bash ./test/fetch-sample-ja.sh
```

## Offline

```bash
sudo docker compose -f ./test/compose.offline.yaml up --build
```

以下の log が表示されるはず

```text
kklocalagent-test-vad       | 2026-05-07T16:08:28.419211Z  INFO vad::asr: [asr <-] transcription: "こんにちは私はクロードです これは音声認識のテストです\n今日は良い天気ですね"
```

## Online

```bash
sudo WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}') docker compose -f ./test/compose.online.yaml up --build
```

```bash
sudo WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}') \
    docker compose -f ./test/compose.online.yaml -f ./test/compose.online.gpu.yaml \
    up --build
```

# Envs

| Var | Default |
|---|---|
| `WHISPER_MODEL` | `ggml-tiny.bin` (`/models` 下のファイル名) |
| `WHISPER_LANGUAGE` | `auto` (`ja` / `en` / `auto`) |
| `WHISPER_THREADS` | `4` |
