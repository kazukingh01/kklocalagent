#!/bin/sh
# Wrapper around the upstream whisper.cpp server image. Resolves the
# model file from $WHISPER_MODEL and invokes the server binary on
# 0.0.0.0:8080. Extra args are forwarded to whisper-server.

set -e

MODEL_PATH="/models/${WHISPER_MODEL:-ggml-tiny.bin}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "model file not found: $MODEL_PATH" >&2
    echo "run automatic-speech-recognition/fetch-models.sh first" >&2
    exit 1
fi

# Locate the server binary. Upstream's image bundles all binaries under
# /app/build/bin; PATH lookups are a fallback for layout changes.
if [ -x /app/build/bin/whisper-server ]; then
    SERVER=/app/build/bin/whisper-server
else
    SERVER=$(command -v whisper-server || true)
fi

if [ -z "$SERVER" ]; then
    echo "whisper-server binary not found at /app/build/bin/whisper-server or in PATH" >&2
    exit 1
fi

THREADS="${WHISPER_THREADS:-$(nproc)}"

exec "$SERVER" \
    --model "$MODEL_PATH" \
    --language "${WHISPER_LANGUAGE:-auto}" \
    --threads "$THREADS" \
    --host 0.0.0.0 \
    --port 8080 \
    "$@"
