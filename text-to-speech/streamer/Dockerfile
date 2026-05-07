# tts-streamer: HTTP shim that turns a text request from the
# orchestrator into VOICEVOX synthesis + audio-io playback streaming.
#
# Two-stage build: the builder pulls the toolchain + dependencies and
# emits a single static-ish binary; the runtime stage ships only the
# binary on top of debian-slim with ca-certificates (rustls roots).

FROM rust:1-slim-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release --bin tts-streamer

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/target/release/tts-streamer /usr/local/bin/tts-streamer

EXPOSE 7070

# Liveness only — the dependency chain (VOICEVOX ready, audio-io
# reachable) varies between production and test, so dependents wait
# on the more authoritative healthchecks of the engine/stub instead.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://127.0.0.1:7070/health || exit 1

ENTRYPOINT ["/usr/local/bin/tts-streamer"]
