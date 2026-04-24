# voice-activity-detection

Voice activity detection service for kklocalagent. Subscribes to
`audio-io`'s `/mic` WebSocket, runs each 20 ms PCM frame through
[webrtc-vad](https://crates.io/crates/webrtc-vad), and emits
`SpeechStarted` / `SpeechEnded` events.

## Build

Uses the same toolchain as `audio-io` (no audio libraries required on
this side — only the WebSocket client).

```bash
# Linux / WSL2
cd voice-activity-detection
cargo build --release

# WSL2 -> Windows cross-compile (matching audio-io README)
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

## Run

`audio-io` must be running (defaults to `ws://127.0.0.1:7010/mic`).

```bash
# Dry-run / test mode (default) — events are logged instead of POSTed.
./target/release/voice-activity-detection --config config.example.toml

# Override the mic URL without a config file.
./target/release/voice-activity-detection \
    --mic-url ws://<windows-ip>:7010/mic
```

Expected output when you speak into the mic:

```
INFO voice_activity_detection::service: connecting to audio-io /mic url=ws://127.0.0.1:7010/mic
INFO voice_activity_detection::service: connected; starting VAD loop
INFO vad::sink: [orchestrator-stub <-] {"name":"SpeechStarted","frame_index":123,"ts":1744284000.12,"sample_rate":16000}
INFO vad::sink: [orchestrator-stub <-] {"name":"SpeechEnded","frame_index":167,"duration_frames":45,"audio_len_bytes":28800,"ts":1744284000.99,"sample_rate":16000}
```

`--sink-mode asr-direct` (or its shorthand `--live`) switches to "live"
mode where each `SpeechEnded` is also POSTed as a WAV upload to a
whisper.cpp `/inference` endpoint at `--asr-url` / `sink.asr_url`. This
lets the audio-io→vad→asr smoke test run without the orchestrator.

`--sink-mode orchestrator` is reserved for the not-yet-built orchestrator
and currently drops events with a warning.

## Configuration

See `config.example.toml` for all tunables. The ones you'll reach for most:

| key | default | effect |
|---|---|---|
| `detector.aggressiveness` | 2 | 0 = most permissive, 3 = strictest. Raise if the VAD fires on keyboard clicks, lower if it misses quiet speech. |
| `detector.start_frames` | 3 | Voiced frames required before `SpeechStarted`. Higher = fewer false triggers, more latency. |
| `detector.hang_frames` | 20 | Silent frames required before `SpeechEnded`. Higher = tolerates longer pauses mid-utterance, but longer wait before ASR can start. |
| `detector.max_utterance_frames` | 1500 (30 s) | Force-ends runaway utterances. |
| `sink.mode` | `"dry-run"` | `"dry-run"` (log only), `"asr-direct"` (POST WAV to whisper.cpp `/inference`), `"orchestrator"` (TODO). |
| `sink.asr_url` | `http://127.0.0.1:7040/inference` | Used in `asr-direct` mode. |
| `sink.asr_timeout_ms` | 30000 | HTTP timeout for `asr-direct` POSTs. Bump if you move to a heavier whisper model. |
| `sink.asr_max_inflight` | 1 | Cap on concurrent `asr-direct` POSTs. Excess utterances are dropped with a warning rather than queued. |
| `sink.log_audio_in_event` | false | Include base64 PCM in `SpeechEnded` *log* lines. Independent of `asr-direct`, which always uploads regardless. |

## Event wire format

Flat JSON, one event per log line / POST body:

```json
{"name":"SpeechStarted","frame_index":123,"ts":1744284000.12,"sample_rate":16000}
{"name":"SpeechEnded","frame_index":167,"duration_frames":45,"audio_len_bytes":28800,"ts":1744284000.99,"sample_rate":16000,"audio_base64":"..."}
```

`audio_base64` is only present when `sink.log_audio_in_event = true`.

## Tests

```bash
cargo test
```

Unit tests cover the FSM logic (no audio hardware or webrtc-vad needed)
and config validation.
