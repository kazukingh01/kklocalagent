# Wraps the upstream whisper.cpp server image. Models live in /models
# (bind-mounted from ./models). Override WHISPER_VARIANT at build time
# to switch CPU/CUDA, and WHISPER_MODEL at run time to switch model.
ARG WHISPER_VARIANT=server

FROM ghcr.io/ggml-org/whisper.cpp:${WHISPER_VARIANT}

ENV WHISPER_MODEL=ggml-tiny.bin

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
