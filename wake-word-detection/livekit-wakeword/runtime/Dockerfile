FROM rust:1-slim-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Stub-main pre-build so a `src/` edit doesn't bust the deps cache layer.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && echo 'fn main() {}' > src/main.rs \
    && cargo build --release --bin livekit-wakeword-runtime \
    && rm -rf src target/release/deps/livekit_wakeword_runtime* \
              target/release/livekit-wakeword-runtime*
COPY src ./src
# rustls-tls → no system openssl needed; ort load-dynamic → libonnxruntime.so
# is not needed at build time, only at run time (next stage).
RUN cargo build --release --bin livekit-wakeword-runtime

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl wget \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime CPU dylib, resolved via ldconfig at startup. ort 2.0.0-rc.11
# enforces onnxruntime >= 1.23.x at dylib load time, so 1.23.0 is the floor.
# Refresh both ARGs together when bumping (sha256sum the new tarball).
ARG ONNXRUNTIME_VERSION=1.23.0
ARG ONNXRUNTIME_SHA256=b6deea7f2e22c10c043019f294a0ea4d2a6c0ae52a009c34847640db75ec5580
RUN curl -sL -o /tmp/onnxruntime.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
    && echo "${ONNXRUNTIME_SHA256}  /tmp/onnxruntime.tgz" | sha256sum -c - \
    && tar -xzf /tmp/onnxruntime.tgz -C /tmp \
    && cp -P "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/"libonnxruntime.so* /usr/local/lib/ \
    && ldconfig \
    && rm -rf "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"* /tmp/onnxruntime.tgz

# Upstream "hey livekit" classifier as a wire-path fixture until M3 ships
# custom-trained models. Pinned to commit SHA + sha256 for reproducibility;
# refresh both values together when bumping.
ENV WW_MODELS_DIR=/opt/models
ARG HEY_LIVEKIT_COMMIT=2a87895dc2d2ec6d3e2044e85b654cefb22d4f49
ARG HEY_LIVEKIT_SHA256=e773490192400f19a9840acb909c2afc43f99f08ca6ae43552e582deea2cb560
RUN mkdir -p ${WW_MODELS_DIR} \
    && wget -q -O ${WW_MODELS_DIR}/hey_livekit.onnx \
       "https://raw.githubusercontent.com/livekit/rust-sdks/${HEY_LIVEKIT_COMMIT}/livekit-wakeword/tests/fixtures/hey_livekit.onnx" \
    && echo "${HEY_LIVEKIT_SHA256}  ${WW_MODELS_DIR}/hey_livekit.onnx" | sha256sum -c -

COPY --from=builder /src/target/release/livekit-wakeword-runtime \
    /usr/local/bin/livekit-wakeword-runtime

# WW_MODELS_DIR must hold the classifier ONNX(s) named in WW_MODELS *and*
# melspectrogram.onnx + embedding_model.onnx (filenames hardcoded).
ENV WW_MIC_URL=ws://audio-io:7010/mic?ts=1 \
    WW_ORCHESTRATOR_URL=http://orchestrator:7000/events \
    WW_MODELS=hey_livekit.onnx \
    WW_THRESHOLD=0.5 \
    WW_COOLDOWN_SEC=2.0 \
    WW_PREDICT_WINDOW_MS=2000 \
    WW_PREDICT_INTERVAL_MS=100 \
    WW_LISTEN=0.0.0.0:7030 \
    WW_SINK_MODE=orchestrator

EXPOSE 7030

RUN adduser --system --no-create-home --group --uid 10001 wwd
USER wwd

# /health is 200 only after model load AND /mic WS connect — same contract
# as the openwakeword shim, so compose's service_healthy gate is identical.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7030/health || exit 1

ENTRYPOINT ["/usr/local/bin/livekit-wakeword-runtime"]
