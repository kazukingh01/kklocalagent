#!/usr/bin/env bash
set -euo pipefail

MODEL="${LLM_MODEL:-gemma3:4b}"

echo "[init] starting ollama serve (model=${MODEL})"
# Filter out GIN per-request access logs. Process substitution (not a
# pipe) so $! is `ollama serve`'s PID, keeping `wait "${SERVE_PID}"`
# below as a clean signal-propagation point; --line-buffered so logs
# aren't held back waiting for a 4 KiB block to fill.
ollama serve > >(grep --line-buffered -v '^\[GIN\]') 2>&1 &
SERVE_PID=$!

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

# MTP build path: `ollama create --experimental` only accepts
# safetensors directories under `FROM` — the GGUF-tag-as-FROM fallback
# was tested and rejected with "not a supported model directory".
case "${MODEL}" in
    gemma4:e4b-mtp|gemma4:e2b-mtp|gemma4:26b-mtp|gemma4:31b-mtp)
        case "${MODEL}" in
            gemma4:e4b-mtp)
                TARGET_REPO="google/gemma-4-E4B-it"
                DRAFT_REPO="google/gemma-4-E4B-it-assistant"
                ;;
            gemma4:e2b-mtp)
                TARGET_REPO="google/gemma-4-E2B-it"
                DRAFT_REPO="google/gemma-4-E2B-it-assistant"
                ;;
            gemma4:26b-mtp)
                TARGET_REPO="google/gemma-4-26B-A4B-it"
                DRAFT_REPO="google/gemma-4-26B-A4B-it-assistant"
                ;;
            gemma4:31b-mtp)
                TARGET_REPO="google/gemma-4-31B-it"
                DRAFT_REPO="google/gemma-4-31B-it-assistant"
                ;;
        esac

        if ollama list 2>/dev/null | awk '{print $1}' | grep -Fxq "${MODEL}"; then
            echo "[init] MTP model already present: ${MODEL}"
        else
            if [ -z "${HF_TOKEN:-}" ]; then
                echo "[init] HF_TOKEN is required to build ${MODEL} (target + drafter are gated Gemma models on HF)" >&2
                kill "${SERVE_PID}" 2>/dev/null || true
                exit 1
            fi
            BUILD_DIR="/tmp/mtp-build-$$"
            trap 'rm -rf "${BUILD_DIR:-}"' EXIT
            mkdir -p "${BUILD_DIR}/target" "${BUILD_DIR}/draft"
            echo "[init] downloading target ${TARGET_REPO}"
            HF_TOKEN="${HF_TOKEN}" huggingface-cli download "${TARGET_REPO}" \
                --local-dir "${BUILD_DIR}/target" \
                --local-dir-use-symlinks False
            echo "[init] downloading drafter ${DRAFT_REPO}"
            HF_TOKEN="${HF_TOKEN}" huggingface-cli download "${DRAFT_REPO}" \
                --local-dir "${BUILD_DIR}/draft" \
                --local-dir-use-symlinks False
            # `PARAMETER num_speculative_tokens` is rejected as
            # "unknown parameter" by `ollama create`'s allowlist, so
            # the Modelfile can't set it.
            cat > "${BUILD_DIR}/Modelfile" <<EOF
FROM ${BUILD_DIR}/target
DRAFT ${BUILD_DIR}/draft
EOF
            echo "[init] ollama create --experimental ${MODEL} (native BF16)"
            ollama create --experimental "${MODEL}" -f "${BUILD_DIR}/Modelfile"
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

# A *no-prompt* /api/generate only LOADS the weights into VRAM — it
# never runs a decode, so the very first inference still eats the
# one-off CUDA-graph capture + kernel/cuBLAS init (USE_GRAPHS=1 in the
# runner log) and visibly stalls the first turn even though the model
# is resident. A tiny prompt with `num_predict>0` forces a real
# prefill + decode so those graphs/kernels are captured during warmup.
#
# Sentinel gates the HEALTHCHECK: /api/show alone goes 200 as soon as
# `ollama pull` completes (present in the registry — not necessarily
# resident in VRAM), which would let `depends_on: service_healthy`
# race the warmup.
echo "[init] warming model into VRAM (load + first decode)"
curl -sfS -X POST http://127.0.0.1:11434/api/generate \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"warmup\",\"stream\":false,\"options\":{\"num_predict\":8}}" \
    -o /dev/null
touch /tmp/llm-warm
echo "[init] model warmed"

wait "${SERVE_PID}"
