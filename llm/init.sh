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
# Filter out GIN per-request access logs ("[GIN] ... | 200 | ... POST /api/chat").
# `ollama serve` is backgrounded so $! is its PID (not grep's), keeping the
# `wait "${SERVE_PID}"` below as a clean signal-propagation point. Process
# substitution receives both stdout and stderr; --line-buffered so logs aren't
# held back waiting for a 4 KiB block to fill.
ollama serve > >(grep --line-buffered -v '^\[GIN\]') 2>&1 &
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

# Warm the model into VRAM so the first /api/chat from the
# orchestrator doesn't pay a 5–15 s cold-load tax (model load shows up
# in `runner started ... loading model ... ggml_cuda_init` in the
# llm logs and visibly stalls the first turn). Ollama's "load a model
# into memory" endpoint is /api/generate with `model` set and no
# prompt; the request blocks until the weights are mmap'd + uploaded
# to GPU. With `OLLAMA_KEEP_ALIVE=-1` set in compose.yaml, the model
# then sticks for the lifetime of the container.
#
# Sentinel file gates the HEALTHCHECK below. /api/show alone goes 200
# as soon as `ollama pull` completes (model present in the local
# registry — not necessarily resident in VRAM), which would let
# `depends_on: service_healthy` race the warmup. Touching this only
# after the warmup curl succeeds keeps the orchestrator from sending
# its first /api/chat until the model is actually hot.
echo "[init] warming model into VRAM"
curl -sfS -X POST http://127.0.0.1:11434/api/generate \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\"}" -o /dev/null
touch /tmp/llm-warm
echo "[init] model warmed"

wait "${SERVE_PID}"
