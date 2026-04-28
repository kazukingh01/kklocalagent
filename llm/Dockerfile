# Wraps the upstream ollama image, baking in a startup-time model
# pull *and* warm-into-VRAM. Upstream's ENTRYPOINT is `/bin/ollama`;
# we replace it with init.sh which starts `ollama serve` in the
# background, waits for the API, pulls ${LLM_MODEL}, fires a no-prompt
# /api/generate to load the weights into GPU memory, touches a
# sentinel for the HEALTHCHECK, then waits on the serve PID so
# container signals propagate.
#
# Model cache lives at /root/.ollama — mount a named volume there so
# the first-boot pull persists across `docker compose down && up`.
ARG OLLAMA_TAG=latest

FROM ollama/ollama:${OLLAMA_TAG}

# curl is needed for the HEALTHCHECK probe and for init.sh to poll
# /api/tags before issuing the pull. Not guaranteed in upstream.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Default model per design v0.2 (#4 §9). Override at runtime via
# compose `environment:` — the HEALTHCHECK CMD below is a shell-form
# command, so ${LLM_MODEL} is expanded in the container at probe time
# and runtime overrides are honored.
ENV LLM_MODEL=gemma3:4b

COPY init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh

# Two gates per probe:
#   1. /tmp/llm-warm sentinel — touched by init.sh after the post-pull
#      warmup curl finishes (model mmap'd + uploaded to VRAM). Without
#      this, /api/show would go 200 as soon as `ollama pull` completes
#      and `depends_on: service_healthy` would race the warmup, leaving
#      the orchestrator's first /api/chat to pay the cold-load tax.
#   2. /api/show on the *specific* configured model — verifies the
#      pull succeeded for the right name (a mismatched LLM_MODEL
#      between this container and the orchestrator would 404 every
#      chat).
#
# start_period is generous because cold-pulling gemma3:4b (~3.3 GB)
# can take 10+ minutes on a slow link; retries*interval = ~10 min
# extra grace if the pull stalls.
HEALTHCHECK --interval=15s --timeout=5s --start-period=12000s --retries=40 \
    CMD test -f /tmp/llm-warm \
        && curl -sfS --connect-timeout 2 -o /dev/null -X POST \
            -H 'Content-Type: application/json' \
            -d "{\"name\":\"${LLM_MODEL}\"}" \
            http://localhost:11434/api/show \
        || exit 1

EXPOSE 11434

ENTRYPOINT ["/usr/local/bin/init.sh"]
