# Build the orchestrator binary in one stage and ship a slim runtime
# image in the next. Mirrors the pattern used by
# voice-activity-detection/Dockerfile so the two Rust modules have a
# consistent shape.

FROM rust:1-slim-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# rustls-tls means no system openssl needed; pkg-config + build-essential
# still cover any transitive C deps.
RUN cargo build --release --bin orchestrator

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/target/release/orchestrator \
    /usr/local/bin/orchestrator

EXPOSE 7000

# Liveness probe — does not verify ASR/LLM reachability since the
# orchestrator degrades gracefully (logs + drops) when a backend is
# down. Gates compose `service_healthy` on the HTTP server being up.
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7000/health || exit 1

ENTRYPOINT ["/usr/local/bin/orchestrator"]
