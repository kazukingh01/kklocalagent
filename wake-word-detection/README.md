# wake-word-detection

Voice agent の wake-word サービス。2つの実装が並存している。

| Subdir | Engine | Language | Status |
|---|---|---|---|
| [`openwakeword/`](./openwakeword/) | openWakeWord (tflite/onnx) | Python | current default |
| [`livekit-wakeword/`](./livekit-wakeword/) | LiveKit conv-attention ONNX | Rust | WIP |

ルートの `compose.yaml` の `wake-word-detection.build.context` で
どちらを使うか切り替える。詳細・コマンドは各 subdir の README を参照。
