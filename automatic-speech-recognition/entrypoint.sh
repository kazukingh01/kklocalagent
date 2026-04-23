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

# Locate the server binary. Upstream's server image puts it at
# /app/server; PATH lookups are a fallback for layout changes.
if [ -x /app/server ]; then
    SERVER=/app/server
else
    SERVER=$(command -v whisper-server || command -v server || true)
fi

if [ -z "$SERVER" ]; then
    echo "whisper-server binary not found at /app/server or in PATH" >&2
    exit 1
fi

exec "$SERVER" \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8080 \
    "$@"
