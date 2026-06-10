#!/bin/sh

set -e

MODEL_PATH="/models/${WHISPER_MODEL:-ggml-tiny.bin}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "model file not found: $MODEL_PATH" >&2
    echo "run automatic-speech-recognition/fetch-models.sh first" >&2
    exit 1
fi

if [ -x /app/build/bin/whisper-server ]; then
    SERVER=/app/build/bin/whisper-server
else
    SERVER=$(command -v whisper-server || true)
fi

if [ -z "$SERVER" ]; then
    echo "whisper-server binary not found at /app/build/bin/whisper-server or in PATH" >&2
    exit 1
fi

exec "$SERVER" \
    --model "$MODEL_PATH" \
    --language "${WHISPER_LANGUAGE:-auto}" \
    --threads "${WHISPER_THREADS:-4}" \
    --host 0.0.0.0 \
    --port 8080 \
    "$@"
