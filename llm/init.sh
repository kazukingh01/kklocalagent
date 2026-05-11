#!/usr/bin/env bash
# Start `ollama serve` in the background, wait for the API to accept
# requests, then make ${LLM_MODEL} available (either via `ollama pull`
# for library tags, or via `ollama create` from Hugging Face
# safetensors for the MTP variants). Hand PID back to `ollama serve`
# via `wait` so SIGTERM propagates cleanly at container stop.
#
# Idempotent w.r.t. the model cache: if the model is already present in
# the mounted volume, both paths short-circuit and the API is serving
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

# Model-name convention: `gemma4:<variant>-mtp` triggers the MTP build
# path. Anything else falls through to plain `ollama pull`.
#
# MTP needs target + drafter safetensors (not GGUF) because PR #15980
# only wires the DRAFT directive through Ollama's safetensors import
# path. The variants below mirror Google's HF org layout
# (https://huggingface.co/google).
case "${MODEL}" in
    *-mtp|*-mtp-*)
        case "${MODEL}" in
            gemma4:e4b-mtp)
                TARGET_REPO="google/gemma-4-E4B-it"
                DRAFT_REPO="google/gemma-4-E4B-it-assistant"
                NUM_SPEC_TOKENS=4
                ;;
            gemma4:e2b-mtp)
                TARGET_REPO="google/gemma-4-E2B-it"
                DRAFT_REPO="google/gemma-4-E2B-it-assistant"
                NUM_SPEC_TOKENS=2
                ;;
            gemma4:26b-mtp)
                TARGET_REPO="google/gemma-4-26B-A4B-it"
                DRAFT_REPO="google/gemma-4-26B-A4B-it-assistant"
                NUM_SPEC_TOKENS=4
                ;;
            gemma4:31b-mtp)
                TARGET_REPO="google/gemma-4-31B-it"
                DRAFT_REPO="google/gemma-4-31B-it-assistant"
                NUM_SPEC_TOKENS=4
                ;;
            *)
                echo "[init] unrecognised MTP model tag: ${MODEL}" >&2
                kill "${SERVE_PID}" 2>/dev/null || true
                exit 1
                ;;
        esac

        # Skip the build if `ollama list` already shows the tag — keeps
        # `docker compose restart` snappy after the first cold create.
        if ollama list 2>/dev/null | awk '{print $1}' | grep -Fxq "${MODEL}"; then
            echo "[init] MTP model already present: ${MODEL}"
        else
            if [ -z "${HF_TOKEN:-}" ]; then
                echo "[init] HF_TOKEN is required to build ${MODEL} (target + drafter are gated Gemma models on HF)" >&2
                kill "${SERVE_PID}" 2>/dev/null || true
                exit 1
            fi
            BUILD_DIR="/tmp/mtp-build-$$"
            mkdir -p "${BUILD_DIR}/target" "${BUILD_DIR}/draft"
            echo "[init] downloading target ${TARGET_REPO}"
            HF_TOKEN="${HF_TOKEN}" huggingface-cli download "${TARGET_REPO}" \
                --local-dir "${BUILD_DIR}/target" \
                --local-dir-use-symlinks False
            echo "[init] downloading drafter ${DRAFT_REPO}"
            HF_TOKEN="${HF_TOKEN}" huggingface-cli download "${DRAFT_REPO}" \
                --local-dir "${BUILD_DIR}/draft" \
                --local-dir-use-symlinks False
            cat > "${BUILD_DIR}/Modelfile" <<EOF
FROM ${BUILD_DIR}/target
DRAFT ${BUILD_DIR}/draft
PARAMETER num_speculative_tokens ${NUM_SPEC_TOKENS}
EOF
            echo "[init] ollama create --experimental ${MODEL}"
            ollama create --experimental "${MODEL}" -f "${BUILD_DIR}/Modelfile"
            # Cleanup: ollama create copied the weights into its own
            # blob store under /root/.ollama; the staging dir is no
            # longer needed and would otherwise waste ~10 GB per build.
            rm -rf "${BUILD_DIR}"
            echo "[init] MTP model ready: ${MODEL}"
        fi
        ;;
    *)
        echo "[init] pulling model: ${MODEL}"
        ollama pull "${MODEL}"
        echo "[init] model ready: ${MODEL}"
        ;;
esac

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
