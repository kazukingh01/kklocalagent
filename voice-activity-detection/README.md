# voice-activity-detection

`audio-io` の `/mic` を購読し webrtc-vad で `SpeechStarted` /
`SpeechEnded` を出す Rust サービス。

## Build

```bash
cd voice-activity-detection
cargo build --release
```

## Run

`audio-io` を起動した状態で:

```bash
./target/release/voice-activity-detection --config config.example.toml
```

設定は `config.example.toml` を参照。

## Docker test

#### build

`wav-utils/` を一階層上から参照する path 依存があるので、build
context はリポジトリルートで固定。Dockerfile 単体を相対 path 化する
ことはできない (Docker は context より上を `COPY` できない)。

```sh
cd ../ && sudo docker build -t kklocalagent/voice-activity-detection:test \
    -f voice-activity-detection/Dockerfile . && cd ./voice-activity-detection/
```

#### Online test

For WSL2

```bash
sudo docker run --rm \
    --name vad-test \
    -e VAD_MIC_URL=ws://$(ip route show | awk '/default/ {print $3}'):7010/mic \
    -e VAD_SINK_MODE=dry-run \
    -e RUST_LOG=info,voice_activity_detection=debug \
    kklocalagent/voice-activity-detection:test
```

For Linux

```bash
sudo docker run --rm \
    --name vad-test --network=host \
    -e VAD_MIC_URL=ws://127.0.0.1:7010/mic \
    -e VAD_SINK_MODE=dry-run \
    -e RUST_LOG=info,voice_activity_detection=debug \
    kklocalagent/voice-activity-detection:test
```

#### Envs

| Var | Default |
|---|---|
| `VAD_MIC_URL` | `ws://audio-io:7010/mic` |
| `VAD_SINK_MODE` | `dry-run` (`dry-run` / `asr-direct` / `orchestrator`) |
| `VAD_ASR_URL` | `http://127.0.0.1:7040/inference` (`asr-direct` mode) |
| `VAD_ORCHESTRATOR_URL` | `http://orchestrator:7000/events` (`orchestrator` mode) |
| `VAD_AGGRESSIVENESS` | `2` (0..=3) |
| `VAD_START_FRAMES` | `3` |
| `VAD_HANG_FRAMES` | `20` |
| `VAD_DENOISE` | `false` |
| `VAD_MIN_UTTERANCE_RMS_DBFS` | `-45` (0 = disable) |
| `VAD_LOG_AUDIO_IN_EVENT` | `false` |
