# livekit-wakeword (WIP — issue #12)

Rust-native wake-word service backed by [`livekit-wakeword`](https://github.com/livekit/livekit-wakeword)'s
pure-Rust ONNX runtime (`ort-tract`). Replaces the Python
[`openwakeword/`](../openwakeword/) shim while preserving the wire
contract with the orchestrator (WS `/mic` subscription + POST
`/events` + `/health` on port 7030).

## Why

* Rust-native inference (no Python interpreter, no tflite / onnxruntime
  Python wheels)
* Single-command custom-phrase training in upstream's Python pipeline,
  output is a plain ONNX file consumed by the Rust crate here
* ONNX is openWakeWord-compatible, so models trained with either tool
  chain are interchangeable

## Status

Not implemented yet. See issue #12 for the migration plan. Until this
ships, the root `compose.yaml` builds [`openwakeword/`](../openwakeword/).
