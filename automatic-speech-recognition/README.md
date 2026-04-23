# automatic-speech-recognition

Whisper.cpp server packaged for kklocalagent. Wraps the upstream
`ghcr.io/ggml-org/whisper.cpp:server` (or `server-cuda`) image and
exposes the HTTP `/inference` endpoint on port 8080.

## Build

```bash
# CPU (default)
docker build -t kklocalagent/asr automatic-speech-recognition

# CUDA — requires NVIDIA Container Toolkit at run time
docker build --build-arg WHISPER_VARIANT=server-cuda \
    -t kklocalagent/asr:cuda automatic-speech-recognition
```

## Fetch a model

Models live in `automatic-speech-recognition/models/` (gitignored) and
are bind-mounted into the container at `/models`.

```bash
cd automatic-speech-recognition
./fetch-models.sh                    # ggml-tiny.bin (~75 MB)
./fetch-models.sh ggml-base.bin      # other names work too
./fetch-models.sh ggml-small.bin
```

## Run

```bash
docker run --rm -p 7040:8080 \
    -v "$(pwd)/automatic-speech-recognition/models:/models:ro" \
    -e WHISPER_MODEL=ggml-tiny.bin \
    kklocalagent/asr
```

## HTTP API

Inherited from whisper.cpp's `whisper-server`:

| Method | Path | Description |
|---|---|---|
| GET | `/` | health / info page |
| POST | `/inference` | multipart upload (`file=@audio.wav`) → JSON transcription |
| POST | `/load` | hot-swap model |

```bash
# Smoke test the running container.
curl http://127.0.0.1:7040/inference \
    -F file=@some-audio.wav \
    -F response_format=json
# -> {"text":"..."}
```

## End-to-end test (audio-io ↔ vad ↔ asr)

A self-contained compose stack lives under `test/` that wires a wav
playback stub (substituting for `audio-io`, which is Windows native) →
`voice-activity-detection` → this service. See `test/README.md`.
