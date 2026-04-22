# audio-io

Microphone capture & speaker playback service for kklocalagent.

## Build

### Ubuntu / WSL2 (Ubuntu)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

```bash
# one-time setup
sudo apt-get install -y mingw-w64
rustup target add x86_64-pc-windows-gnu

# build
cd audio-io

## For Windows
cargo build --release --target x86_64-pc-windows-gnu
## for Ubuntu
cargo build --release
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

## Run

### Start process

#### For Ubuntu

```bash
./target/release/audio-io --config audio-io/config.example.toml
```

#### For Windows

```bash
cp target/x86_64-pc-windows-gnu/release/audio-io.exe /mnt/c/Users/XXXX/Documents/
cp config.example.toml /mnt/c/Users/XXXX/Documents/config.local.toml
```

Open windows power shell.

```bash
$env:RUST_LOG = "info"
.\audio-io.exe --config .\config.local.toml
```

### Test record

Open shell ( ubuntu )

```bash
sudo apt update && sudo apt install -y ffmpeg
cargo install websocat # Install websocket program
websocat -b "ws://${WIN_HOST}:7010/mic" | head -c 96000 > mic.raw
ffmpeg -f s16le -ar 16000 -ac 1 -i mic.raw mic.wav -y
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
