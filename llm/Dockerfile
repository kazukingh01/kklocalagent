# 0.30.3 is the first stable release that supports the Gemma 4 12B
# encoder-free multimodal architecture (gemma4:12b); 0.30.4 still
# bundles the Gemma 4 MTP speculative decoding from the older 0.23.x
# pin. Bump alongside any newer model that needs updated runtime
# support.
ARG OLLAMA_TAG=0.30.4

FROM ollama/ollama:${OLLAMA_TAG}

# python3 + huggingface_hub is only used by the MTP path in init.sh:
# the target + drafter safetensors are gated Gemma weights on HF and
# need an authenticated download before `ollama create` can ingest
# them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages \
        "huggingface_hub[cli]>=0.25,<1.0"

# The HEALTHCHECK CMD below is shell-form, so ${LLM_MODEL} is expanded
# at probe time and runtime overrides are honored. gemma4:12b needs
# Ollama >= 0.30.3 (see OLLAMA_TAG above).
ENV LLM_MODEL=hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M

COPY init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh

# Two gates per probe:
#   1. /tmp/llm-warm sentinel — without it, /api/show goes 200 as soon
#      as `ollama pull` completes and `depends_on: service_healthy`
#      would race the warmup, leaving the orchestrator's first
#      /api/chat to pay the cold-load tax.
#   2. /api/show on the *specific* configured model — a mismatched
#      LLM_MODEL between this container and the orchestrator would 404
#      every chat.
# start_period is generous because cold-pulling gemma4:12b (~9 GB)
# can take 10+ minutes on a slow link.
HEALTHCHECK --interval=15s --timeout=5s --start-period=12000s --retries=40 \
    CMD test -f /tmp/llm-warm \
        && curl -sfS --connect-timeout 2 -o /dev/null -X POST \
            -H 'Content-Type: application/json' \
            -d "{\"name\":\"${LLM_MODEL}\"}" \
            http://localhost:11434/api/show \
        || exit 1

EXPOSE 11434

ENTRYPOINT ["/usr/local/bin/init.sh"]
