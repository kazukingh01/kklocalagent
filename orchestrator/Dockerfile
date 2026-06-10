FROM rust:1-slim-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Build context is the repo root (compose.yaml `build.context: .`) so the
# path-dependency on `../wav-utils` resolves; /src mirrors the repo layout.
COPY wav-utils ./wav-utils
COPY orchestrator/Cargo.toml orchestrator/Cargo.lock ./orchestrator/
COPY orchestrator/src ./orchestrator/src
RUN cargo build --release --manifest-path orchestrator/Cargo.toml --bin orchestrator

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/orchestrator/target/release/orchestrator \
    /usr/local/bin/orchestrator

EXPOSE 7000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7000/health || exit 1

ENTRYPOINT ["/usr/local/bin/orchestrator"]
