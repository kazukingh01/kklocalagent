# voice_activity_detection

Voice activity detection service for kklocalagent.

Implementation: **webrtcvad** (Google WebRTC VAD, non-ML, lightweight C
extension). Accepts 10 / 20 / 30 ms frames at 8 / 16 / 32 / 48 kHz mono s16le —
matches audio-io's wire format (20 ms @ 16 kHz).

## Install

Ubuntu / WSL2:

```bash
cd voice_activity_detection
./install.sh
```

The script installs the build toolchain (`build-essential`, `python3-dev`),
installs [uv](https://docs.astral.sh/uv/) if missing, then runs `uv sync` to
build and install webrtcvad into `.venv/`. Finishes with a smoke test that
classifies 20 ms of silence.

## Run (placeholder)

Service entry point not yet implemented. For now, import works:

```bash
uv run python -c "import webrtcvad; print(webrtcvad.Vad(2))"
```
