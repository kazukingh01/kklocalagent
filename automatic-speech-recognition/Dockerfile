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

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
