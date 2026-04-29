# livekit-wakeword (issue #12)

Rust-native wake-word service backed by the
[`livekit-wakeword`](https://crates.io/crates/livekit-wakeword) Rust
crate's pure-Rust ONNX runtime. Replaces the Python
[`openwakeword/`](../openwakeword/) shim while preserving the wire
contract with the orchestrator (WS `/mic` subscription + POST
`/events` + `/health` on port 7030).

## Layout

| Subdir | Role |
|---|---|
| [`runtime/`](./runtime/) | Rust inference service (Cargo binary). Built into the compose `wake-word-detection` container. |
| [`train/`](./train/) | Custom-phrase training scripts (Python, GPU). Run on a separate host; output ONNX is the only artefact that comes back into the runtime image. |

## Why two subdirs

* The Rust crate (`livekit-wakeword` on crates.io) and the upstream
  Python training toolkit (`livekit/livekit-wakeword` on GitHub) are
  two distinct packages from LiveKit Inc. The runtime depends only on
  the crate; the training toolkit pulls TTS + torch + audio aug — many
  GB of deps that have no place inside the inference container.
* Inference is always-on; training is rare (per phrase, run once).
  Different lifecycles deserve different containers.

## Status

* `runtime/` — implemented with `WakeWordModel::predict` end-to-end, ws
  client, ring buffer, event sink, /health. Compiles cleanly; unit
  tests for the small pure functions pass.
* `train/` — config templates + thin wrapper + eval harness scaffolded.
  Actual training runs happen on a separate GPU host (see `train/README.md`).
* Cutover from openwakeword: change `compose.yaml`'s `build.context` to
  `./wake-word-detection/livekit-wakeword/runtime` and rename
  `WW_MODELS` → `WW_MODEL_PATHS`. Roll back by reverting the path.
