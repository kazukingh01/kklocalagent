# WHISPER_VARIANT=main uses upstream's CPU image as-is (already AVX2-only);
# main-cuda is built from source here because upstream's :main-cuda is
# compiled with GGML_NATIVE=ON on an AVX-512 CI host and SIGILLs on consumer
# Intel since Alder Lake (12th-gen+, AVX-512 fused off). GGML_AVX512=OFF
# caps the ISA at AVX2.

ARG WHISPER_VARIANT=main-cuda
ARG WHISPER_REF=v1.7.5

# Only consumed when WHISPER_VARIANT=main-cuda.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder-cuda
ARG WHISPER_REF
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# cmake 3.22's FindCUDAToolkit doesn't probe lib64/stubs/, so the
# `CUDA::cuda_driver` link fails ("undefined reference to cuGetErrorString").
# Two symlinks because the stub's SONAME is libcuda.so.1. Link-time only —
# the host-injected libcuda from nvidia-container-toolkit takes over at runtime.
RUN ln -s stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so \
    && ln -s stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1
# Build at /app/build (not /src/build) so the binary's build-tree
# RUNPATH still points at the right .so dirs after COPY into the
# runtime stage. CUDA archs cover Turing(75) through Hopper(90);
# CUDA 13 dropped 50/61/70 so whisper.cpp's defaults won't compile.
RUN git clone --depth 1 --branch "$WHISPER_REF" \
        https://github.com/ggml-org/whisper.cpp.git /app/src
WORKDIR /app/src
RUN cmake -B /app/build \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_F16C=ON -DGGML_FMA=ON \
        -DGGML_AVX512=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
        -DWHISPER_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /app/build -j"$(nproc)" --target whisper-server

# `base-${WHISPER_VARIANT}` is BuildKit's stage-by-arg pattern;
# only the selected variant's stage runs.

FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04 AS base-main-cuda
COPY --from=builder-cuda /app/build /app/build
# nvidia/cuda runtime base ships neither curl nor the OpenMP runtime
# that libggml-cpu links against; upstream's :main image already has both.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

FROM ghcr.io/ggml-org/whisper.cpp:main AS base-main
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

FROM base-${WHISPER_VARIANT}

ENV WHISPER_MODEL=ggml-tiny.bin
# whisper-server `-l`: `auto` = detect; pin (e.g. `ja`) for lower latency.
ENV WHISPER_LANGUAGE=auto
# WHISPER_THREADS (whisper-server `-t`) defaults to 4. Going past the
# *physical* core count usually hurts whisper.cpp — Hyperthreading cache contention.

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# start-period is generous because the larger ggml models can take
# 30+ s to mmap and warm up on first start. retries*interval = ~5 min.
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s --retries=60 \
    CMD curl -s --connect-timeout 2 -o /dev/null http://localhost:8080/ || exit 1

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
