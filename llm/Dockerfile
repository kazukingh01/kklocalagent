# Wraps the upstream ollama image, baking in a startup-time model pull.
# Upstream's ENTRYPOINT is `/bin/ollama`; we replace it with init.sh
# which starts `ollama serve` in the background, waits for the API to
# come up, pulls ${LLM_MODEL}, then waits on the serve PID so container
# signals propagate.
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

# Probe /api/show for the *specific* configured model — unlike /api/tags
# (always 200 once serve is up), /api/show only returns 200 after the
# model has been pulled into the local registry. Gating
# `service_healthy` on this means dependents don't race the first-boot
# pull.
#
# start_period is generous because cold-pulling gemma3:4b (~3.3 GB) can
# take 10+ minutes on a slow link; retries*interval = ~10 min extra
# grace if the pull stalls.
HEALTHCHECK --interval=15s --timeout=5s --start-period=12000s --retries=40 \
    CMD curl -sfS --connect-timeout 2 -o /dev/null -X POST \
        -H 'Content-Type: application/json' \
        -d "{\"name\":\"${LLM_MODEL}\"}" \
        http://localhost:11434/api/show || exit 1

EXPOSE 11434

ENTRYPOINT ["/usr/local/bin/init.sh"]
