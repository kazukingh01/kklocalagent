# livekit-wakeword runtime: subscribes to audio-io /mic, runs ONNX
# inference via real onnxruntime (ort 2.0 with load-dynamic), POSTs
# WakeWordDetected to the orchestrator. Pattern mirrors the
# orchestrator and audio-io Dockerfiles for consistency.

FROM rust:1-slim-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Two-step copy so a `src/` edit doesn't bust the deps cache layer.
# Step 1: compile against a stub main with all dependencies registered;
# this layer is only invalidated when Cargo.{toml,lock} change.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && echo 'fn main() {}' > src/main.rs \
    && cargo build --release --bin livekit-wakeword-runtime \
    && rm -rf src target/release/deps/livekit_wakeword_runtime* \
              target/release/livekit-wakeword-runtime*
# Step 2: drop the real source in. Cargo only recompiles our own crate
# now; all transitive deps are reused from the cached layer above.
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
#
# Pinned to a commit SHA + sha256-verified so the image is bit-for-bit
# reproducible. Refresh both values together when bumping (commit hash
# from `gh api repos/livekit/rust-sdks/commits?path=...`; the sha256
# from `sha256sum hey_livekit.onnx` after fetching the new file).
ENV WW_MODELS_DIR=/opt/models
ARG HEY_LIVEKIT_COMMIT=2a87895dc2d2ec6d3e2044e85b654cefb22d4f49
ARG HEY_LIVEKIT_SHA256=e773490192400f19a9840acb909c2afc43f99f08ca6ae43552e582deea2cb560
RUN mkdir -p ${WW_MODELS_DIR} \
    && wget -q -O ${WW_MODELS_DIR}/hey_livekit.onnx \
       "https://raw.githubusercontent.com/livekit/rust-sdks/${HEY_LIVEKIT_COMMIT}/livekit-wakeword/tests/fixtures/hey_livekit.onnx" \
    && echo "${HEY_LIVEKIT_SHA256}  ${WW_MODELS_DIR}/hey_livekit.onnx" | sha256sum -c -

COPY --from=builder /src/target/release/livekit-wakeword-runtime \
    /usr/local/bin/livekit-wakeword-runtime

# WW_MODELS_DIR (set above) is the single directory containing both
# classifier ONNX(s) named in WW_MODELS *and* the upstream
# melspectrogram.onnx + embedding_model.onnx (filenames are hardcoded).
# One bind mount of the host's train uv venv resources directory (or
# a hand-curated models dir) covers everything.
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

# Drop privileges. The runtime only needs to read /opt/models (root-
# readable above) and bind to :7030 (>1024, no caps needed). adduser
# --system gives a UID without a home dir or login shell.
RUN adduser --system --no-create-home --group --uid 10001 wwd
USER wwd

# /health flips to 200 only after the model is loaded AND the /mic WS
# is connected — same contract as the openwakeword shim, so compose's
# `service_healthy` gate behaves identically across implementations.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7030/health || exit 1

ENTRYPOINT ["/usr/local/bin/livekit-wakeword-runtime"]
