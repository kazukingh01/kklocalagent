# Wraps the upstream whisper.cpp image. Models live in /models
# (bind-mounted from ./models). Override WHISPER_VARIANT at build time
# to switch CPU/CUDA/intel/musa, and WHISPER_MODEL at run time to switch
# model. Upstream publishes one tag per backend (`main`, `main-cuda`,
# `main-musa`, `main-intel`) — the same image bundles whisper-cli and
# whisper-server, we just point our entrypoint at the server binary.
ARG WHISPER_VARIANT=main

FROM ghcr.io/ggml-org/whisper.cpp:${WHISPER_VARIANT}

ENV WHISPER_MODEL=ggml-tiny.bin
# Language code for whisper-server (`-l`). `auto` = detect; pin to a
# code (e.g. `ja`, `en`) when the deployment is single-language and you
# want lower latency / less misdetection.
ENV WHISPER_LANGUAGE=auto
# CPU threads used by whisper-server (`-t`). Default = 4 (matches
# whisper-server's own default). Bumping past the host's *physical*
# core count usually hurts whisper.cpp due to cache contention from
# Hyperthreading — pick a value with that in mind, not blindly nproc.

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# curl is for the HEALTHCHECK probe — the upstream image is slim and
# ships no HTTP client. whisper-server only starts listening *after* the
# model is fully loaded, so a successful TCP/HTTP response is a reliable
# "ready to accept inference" signal. We don't pass -f because any HTTP
# response (even 404 on `/`) means the server is up; only network errors
# return non-zero from `curl -s`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# start-period is generous because the larger ggml models can take 30+ s
# to mmap and warm up on first start. retries*interval = ~5 min total.
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s --retries=60 \
    CMD curl -s --connect-timeout 2 -o /dev/null http://localhost:8080/ || exit 1

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
