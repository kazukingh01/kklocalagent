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

詳細・コマンドは各 subdir の README を参照。
