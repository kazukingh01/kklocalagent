# livekit-wakeword runtime: subscribes to audio-io /mic, runs ONNX
# inference via real onnxruntime (ort 2.0 with load-dynamic), POSTs
# WakeWordDetected to the orchestrator. Pattern mirrors the
# orchestrator and audio-io Dockerfiles for consistency.

FROM rust:1-slim-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# rustls-tls in reqwest means no system openssl is needed; pkg-config +
# build-essential cover any transitive C deps. ort uses load-dynamic
# so libonnxruntime.so is *not* needed at build time — only at run time
# (linked into the next stage below).
RUN cargo build --release --bin livekit-wakeword-runtime

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl wget \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime CPU shared library. `load-dynamic` resolves
# libonnxruntime.so via ORT_DYLIB_PATH or ldconfig at startup;
# /usr/local/lib + ldconfig is the simpler of the two. ort
# 2.0.0-rc.11 enforces onnxruntime >= 1.23.x at dylib load time
# ("expected version >= '1.23.x'"), so 1.23.0 is the floor.
ARG ONNXRUNTIME_VERSION=1.23.0
RUN curl -sL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
        | tar -xz -C /tmp \
    && cp -P "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/"libonnxruntime.so* /usr/local/lib/ \
    && ldconfig \
    && rm -rf "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"*

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

# WW_FEATURE_ONNX_DIR points at the directory containing the mel +
# embedding ONNX (filenames: melspectrogram.onnx, embedding_model.onnx).
# Defaults to /opt/models so feature ONNX sits alongside the classifier
# ONNX referenced by WW_MODEL_PATHS — one bind mount of the host's
# train uv venv resources directory (or a hand-curated models dir)
# covers everything. Override individually with WW_MEL_ONNX_PATH /
# WW_EMBEDDING_ONNX_PATH if the layout doesn't match.
ENV WW_MIC_URL=ws://audio-io:7010/mic?ts=1 \
    WW_ORCHESTRATOR_URL=http://orchestrator:7000/events \
    WW_MODEL_PATHS=/opt/models/hey_livekit.onnx \
    WW_FEATURE_ONNX_DIR=/opt/models \
    WW_THRESHOLD=0.5 \
    WW_COOLDOWN_SEC=2.0 \
    WW_PREDICT_WINDOW_MS=2000 \
    WW_PREDICT_INTERVAL_MS=100 \
    WW_LISTEN=0.0.0.0:7030 \
    WW_SINK_MODE=orchestrator

EXPOSE 7030

# /health flips to 200 only after the model is loaded AND the /mic WS
# is connected — same contract as the openwakeword shim, so compose's
# `service_healthy` gate behaves identically across implementations.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7030/health || exit 1

ENTRYPOINT ["/usr/local/bin/livekit-wakeword-runtime"]
