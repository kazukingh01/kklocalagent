FROM rust:1-slim-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY wav-utils ./wav-utils
COPY voice-activity-detection/Cargo.toml voice-activity-detection/Cargo.lock \
    ./voice-activity-detection/
COPY voice-activity-detection/src ./voice-activity-detection/src
RUN cargo build --release --manifest-path voice-activity-detection/Cargo.toml \
    --bin voice-activity-detection

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/voice-activity-detection/target/release/voice-activity-detection \
    /usr/local/bin/voice-activity-detection
EXPOSE 7020
ENTRYPOINT ["/usr/local/bin/voice-activity-detection"]
