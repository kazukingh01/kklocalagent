# audio-io

Microphone capture & speaker playback service for kklocalagent.

## Responsibility

This module only owns physical audio I/O. It does **not** transcribe or
synthesize speech. It exposes mic PCM as a WebSocket stream and accepts
PCM on another WebSocket to play through the speaker.

- Capture: mic → resample/downmix → **s16le mono 16kHz, 20ms frames** → broadcast to subscribers
- Playback: accept s16le mono 16kHz PCM → resample/channel-map → native output device

## Supported platforms

| OS | Backend | Notes |
|---|---|---|
| Linux (Ubuntu) | ALSA (+ optional JACK) | native |
| Windows 10/11 | WASAPI | used when running WSL2 deployment — install on Windows side, **not** inside WSL2 |
| macOS | CoreAudio | works but not the primary target |

WSL2 cannot access Windows audio devices directly, so `audio-io` runs on
the Windows host and other services in WSL2 connect to it over TCP
(`ws://<windows-host-ip>:7010/mic` etc.).

## Build

```bash
# Linux / macOS native
cargo build --release

# Windows (cross from Linux requires mingw / cross toolchain;
# easier to build natively on Windows)
cargo build --release --target x86_64-pc-windows-msvc
```

## Run

```bash
./target/release/audio-io --config audio-io/config.example.toml
```

Or with env var:

```bash
AUDIO_IO_CONFIG=./config.toml ./target/release/audio-io
```

## HTTP API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/devices` | list available input/output devices |
| POST | `/start` | (re)start capture + playback |
| POST | `/stop` | stop capture + playback |
| POST | `/spk/stop` | drop all pending playback audio (barge-in) |
| GET | `/mic` | WebSocket — server streams captured PCM frames (binary) |
| GET | `/spk` | WebSocket — client sends PCM frames to play (binary) |

PCM format on both WebSockets: **s16le, 16kHz, mono, 20ms/frame = 640 bytes/frame**.

## Config

See `config.example.toml`. All fields optional — defaults are sane.

## Tests

```bash
cargo test
```

Integration tests exercise the HTTP routes with a loopback cpal (null)
config; no real audio hardware is required.
