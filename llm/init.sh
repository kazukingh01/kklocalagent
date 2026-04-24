#!/usr/bin/env bash
# Start `ollama serve` in the background, wait for the API to accept
# requests, then pull ${LLM_MODEL} so the first /api/chat against this
# container doesn't 404. Hand PID back to `ollama serve` via `wait` so
# SIGTERM propagates cleanly at container stop.
#
# Idempotent w.r.t. the model cache: if the model is already present in
# the mounted volume, `ollama pull` is a no-op and the API is serving
# within ~2 s.
set -euo pipefail

MODEL="${LLM_MODEL:-gemma3:4b}"

echo "[init] starting ollama serve (model=${MODEL})"
ollama serve &
SERVE_PID=$!

# /api/tags returns 200 as soon as the HTTP server binds, regardless of
# what's pulled. Typically ready within ~1 s.
for i in $(seq 1 60); do
    if curl -sfS -o /dev/null http://127.0.0.1:11434/api/tags; then
        echo "[init] ollama api ready after ${i}s"
        break
    fi
    if [ "${i}" = "60" ]; then
        echo "[init] ollama api did not come up in 60s" >&2
        kill "${SERVE_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo "[init] pulling model: ${MODEL}"
ollama pull "${MODEL}"
echo "[init] model ready: ${MODEL}"

wait "${SERVE_PID}"
