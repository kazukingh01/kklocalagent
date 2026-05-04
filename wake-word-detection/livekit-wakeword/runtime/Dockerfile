# livekit-wakeword runtime: subscribes to audio-io /mic, runs the
# livekit-wakeword crate's pure-Rust ONNX inference (ort-tract), and
# POSTs WakeWordDetected to the orchestrator. Pattern mirrors the
# orchestrator and audio-io Dockerfiles for consistency.

FROM rust:1-slim-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# rustls-tls in reqwest means no system openssl is needed; pkg-config +
# build-essential cover any transitive C deps from the tract stack.
RUN cargo build --release --bin livekit-wakeword-runtime

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl wget \
    && rm -rf /var/lib/apt/lists/*

# Pre-fetch the upstream "hey livekit" classifier ONNX into the
# image. M3 will replace this with our own custom-trained models
# published as a GitHub release; for now the upstream fixture is
# enough to verify the wire path end-to-end.
ENV WW_MODELS_DIR=/opt/models
RUN mkdir -p ${WW_MODELS_DIR} \
    && wget -q -O ${WW_MODELS_DIR}/hey_livekit.onnx \
       https://raw.githubusercontent.com/livekit/rust-sdks/main/livekit-wakeword/tests/fixtures/hey_livekit.onnx

COPY --from=builder /src/target/release/livekit-wakeword-runtime \
    /usr/local/bin/livekit-wakeword-runtime

ENV WW_MIC_URL=ws://audio-io:7010/mic \
    WW_ORCHESTRATOR_URL=http://orchestrator:7000/events \
    WW_MODEL_PATHS=/opt/models/hey_livekit.onnx \
    WW_THRESHOLD=0.5 \
    WW_COOLDOWN_SEC=2.0 \
    WW_PREDICT_WINDOW_MS=2000 \
    WW_LISTEN=0.0.0.0:7030 \
    WW_SINK_MODE=orchestrator

EXPOSE 7030

# /health flips to 200 only after the model is loaded AND the /mic WS
# is connected — same contract as the openwakeword shim, so compose's
# `service_healthy` gate behaves identically across implementations.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7030/health || exit 1

ENTRYPOINT ["/usr/local/bin/livekit-wakeword-runtime"]
