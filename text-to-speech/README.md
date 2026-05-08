# text-to-speech

Voice agent の TTS 一式。engine と streamer の 2 層構成。

```
text-to-speech/
├── engine/        TTS 合成エンジン (将来複数実装を想定して dir で分離)
│   └── voicevox/    VOICEVOX (ずんだもん 等) — current default
└── streamer/      orchestrator ↔ engine ↔ audio-io の glue
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
