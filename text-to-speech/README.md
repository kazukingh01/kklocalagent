# text-to-speech

Voice agent の TTS 一式。engine と streamer の 2 層構成。

```
text-to-speech/
├── engine/        TTS 合成エンジン (将来複数実装を想定して dir で分離)
│   └── voicevox/    VOICEVOX (ずんだもん 等) — current default
└── streamer/      orchestrator ↔ engine ↔ audio-io の glue (Rust)
```

| Layer | Subdir | Status |
|---|---|---|
| engine | [`engine/voicevox/`](./engine/voicevox/) | current default |
| streamer | [`streamer/`](./streamer/) | Rust impl |

# Test

```bash
sudo WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}') \
    docker compose -f compose.voicevox.yaml up --build
```

```bash
curl -sS -X POST http://127.0.0.1:7070/speak \
      -H 'Content-Type: application/json' \
      -d '{"text":"おはよう"}'

  sleep 0.5

  # 割り込み — こっちが勝つ
  curl -sS -X POST http://127.0.0.1:7070/speak \
      -H 'Content-Type: application/json' \
      -d '{"text":"割り込みです。"}'
```

```bash
curl -s -X POST 'http://127.0.0.1:7060/audio_query?speaker=3' \
      --get --data-urlencode 'text=ぼくはずんだもんなのだ。今日はとてもいい天気なのだ。' \
    | curl -s -X POST 'http://127.0.0.1:7060/synthesis?speaker=3' \
        -H 'Content-Type: application/json' --data-binary @- \
    > /tmp/zundamon.wav
```

# API (streamer)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness probe |
| POST | `/speak` | body の `text` を VOICEVOX で合成し audio-io `/spk` に stream |
| POST | `/finalize` | audio-io 再生 ring の drain 待ち (post-TTS VAD 静音窓の anchor) |
| POST | `/stop` | 進行中の `/speak` を abort + audio-io 側 buffer drop |

並行 `/speak` は **newest wins**: 新しいリクエストが in-flight タスクを abort してから自分を起動。abort された側は **499 Client Closed Request** で返る (barge-in と通常エラーの区別)。

# Envs

## streamer (`text-to-speech/streamer/`)

| Var | Default | Notes |
|---|---|---|
| `VOICEVOX_URL` | `http://text-to-speech:50021` | engine base URL |
| `VOICEVOX_SPEAKER` | `3` (ずんだもん ノーマル) | per-style id; `curl /speakers` で列挙 |
| `VOICEVOX_SPEED_SCALE` | `1.0` | AudioQuery の `speedScale` (おおむね 0.5–2.0) |
| `SPK_URL` | _(unset → /speak は 500)_ | `ws://${WINDOWS_HOST}:7010/spk` |
| `AUDIO_IO_BASE` | _(unset → /start, /spk/stop はスキップ)_ | `http://${WINDOWS_HOST}:7010` |
| `WS_PACING_MS` | `500` | prebuffer 後の WS 送信間隔。500 = realtime。下げると wire を overrate して audio-io 側のハード時計ドリフトを補償 (例: `WS_PACING_MS=450` で約 11 % 速く) |
| `HOST` / `PORT` | `0.0.0.0` / `7070` | bind |
| `RUST_LOG` | `info` | tracing filter |

## compose (`compose.voicevox.yaml`)

| Var | Default | Notes |
|---|---|---|
| `WINDOWS_HOST` | `host.docker.internal` | audio-io が居るホスト。別マシンなら LAN IP |
| `VOICEVOX_VARIANT` | `cpu-latest` | engine image tag (`nvidia-latest` で GPU) |
