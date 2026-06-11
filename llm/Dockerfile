# 0.30.3 is the first stable release that supports the Gemma 4 12B
# encoder-free multimodal architecture (gemma4:12b); the library
# gemma4:12b-* tags ship configs with `requires: 0.30.5`, so the pin
# must be >= 0.30.5 to load them.
ARG OLLAMA_TAG=0.30.5

FROM ollama/ollama:${OLLAMA_TAG}

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages \
        "huggingface_hub[cli]>=0.25,<1.0"

ENV LLM_MODEL=gemma4:12b-it-q4_K_M

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
