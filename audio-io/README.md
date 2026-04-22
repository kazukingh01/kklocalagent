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

## Prerequisites

### Rust toolchain (all platforms)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Linux / WSL2 (Ubuntu, Debian)

ALSA development headers are required to build `cpal`:

```bash
sudo apt-get update
sudo apt-get install -y libasound2-dev pkg-config build-essential
```

Verify:

```bash
pkg-config --modversion alsa   # → 1.2.x
```

### Windows (native)

Install the MSVC C++ build tools (via Visual Studio Installer or the
standalone "Build Tools for Visual Studio"). No extra audio SDK needed —
cpal uses WASAPI which ships with Windows.

### macOS

No extra packages needed; cpal uses CoreAudio from the SDK bundled with
Xcode Command Line Tools (`xcode-select --install`).

## Build

### Native

```bash
# Linux / macOS / Windows (native)
cargo build --release
```

### Cross-compile to Windows from WSL2 / Ubuntu

Use the MinGW-w64 toolchain to produce a Windows `.exe` without needing a
Windows machine.

```bash
# one-time setup
sudo apt-get install -y mingw-w64
rustup target add x86_64-pc-windows-gnu

# build
cd audio-io
cargo build --release --target x86_64-pc-windows-gnu
```

Output: `target/x86_64-pc-windows-gnu/release/audio-io.exe`.

Note: the MinGW build may need `libgcc_s_seh-1.dll` and
`libwinpthread-1.dll` (from `/usr/x86_64-w64-mingw32/lib/`) next to the
`.exe` on the Windows host. If you want a single-file binary, pass
`-C target-feature=+crt-static` via `RUSTFLAGS` or add a `.cargo/config.toml`.

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
