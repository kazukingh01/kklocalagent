# text-to-speech / streamer

orchestrator ↔ VOICEVOX ↔ audio-io の glue。
`POST /speak {"text":...}` を受けて VOICEVOX で 16 kHz mono WAV を作り、
WAV header を剥がして audio-io の `/spk` WS に 20 ms (640 B) ペーシング
で push する Rust サービス。

旧 `tts-streamer/` (Python) の置き換え。エンドポイント・env 名は完全互換。

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness probe |
| POST | `/speak` | synthesize body の `text` を /spk に stream |
| POST | `/finalize` | audio-io の再生 ring drain を待つ (post-TTS VAD 静音窓の anchor) |
| POST | `/stop` | 進行中の /speak を abort + audio-io 側 buffer を drop |

`POST /speak` body:

```json
{ "text": "今日はいい天気だね。" }
```

Concurrent な /speak は **"newest wins"** で、新しいリクエストが
in-flight タスクを abort してから自分を起動する。abort された側は
**499 Client Closed Request** で返る (barge-in と通常エラーの区別)。

## Build

```bash
cd text-to-speech/streamer
cargo build --release
```

Docker:

```bash
docker build -t kklocalagent/tts-streamer .
```

## Run (standalone smoke)

```bash
docker run --rm -p 7070:7070 \
    -e VOICEVOX_URL=http://host.docker.internal:7060 \
    -e SPK_URL=ws://host.docker.internal:7010/spk \
    -e AUDIO_IO_BASE=http://host.docker.internal:7010 \
    kklocalagent/tts-streamer

# 別シェルで:
curl -sS -X POST http://127.0.0.1:7070/speak \
    -H 'Content-Type: application/json' \
    -d '{"text":"テスト発話なのだ。"}'
```

## Envs

| Var | Default | Notes |
|---|---|---|
| `VOICEVOX_URL` | `http://text-to-speech:50021` | engine base |
| `VOICEVOX_SPEAKER` | `3` (ずんだもん ノーマル) | per-style id; `curl /speakers` で列挙 |
| `VOICEVOX_SPEED_SCALE` | `1.0` | AudioQuery の `speedScale` (おおむね 0.5–2.0) |
| `SPK_URL` | _(unset, /speak は 500)_ | `ws://${WINDOWS_HOST}:7010/spk` |
| `AUDIO_IO_BASE` | _(unset, /start, /spk/stop はスキップ)_ | `http://${WINDOWS_HOST}:7010` |
| `HOST` / `PORT` | `0.0.0.0:7070` | bind |
| `RUST_LOG` | `info` | tracing filter |
