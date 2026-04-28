# whisper.cpp wrapper. WHISPER_VARIANT picks the backend:
#   main      → upstream CPU image as-is (already AVX2-only, portable)
#   main-cuda → built from source here, NOT upstream's image
#
# Why we don't use upstream's :main-cuda: it's compiled with
# GGML_NATIVE=ON on a CI host that has AVX-512, and the resulting
# binary SIGILLs on every consumer Intel chip since Alder Lake
# (12th-gen+) where AVX-512 is fused off in production. Source-
# building with GGML_AVX512=OFF caps the ISA at AVX2 and makes the
# binary portable across that whole range. Upstream's :main (CPU)
# image is already built without AVX-512, so it doesn't need the
# same treatment.

ARG WHISPER_VARIANT=main-cuda
ARG WHISPER_REF=v1.7.5

# --- CUDA build (only consumed when WHISPER_VARIANT=main-cuda) ------
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder-cuda
ARG WHISPER_REF
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# cmake 3.22's FindCUDAToolkit doesn't probe lib64/stubs/, so the
# `CUDA::cuda_driver` target ggml-cuda links against resolves to
# nothing and the link fails with "undefined reference to
# cuGetErrorString". Two symlinks because the stub has SONAME
# libcuda.so.1: libggml-cuda.so records that as its NEEDED tag, and
# the subsequent whisper-server link recursively probes for
# libcuda.so.1 itself. Link-time only — the host-injected libcuda
# from nvidia-container-toolkit takes over at runtime.
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

# --- variant bases -------------------------------------------------
# `base-${WHISPER_VARIANT}` is BuildKit's stage-by-arg pattern;
# only the selected variant's stage runs.

FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04 AS base-main-cuda
COPY --from=builder-cuda /app/build /app/build
# nvidia/cuda runtime base ships neither curl nor the OpenMP runtime
# that libggml-cpu links against. Upstream's :main image already
# has both, hence we only do this for the CUDA path.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

FROM ghcr.io/ggml-org/whisper.cpp:main AS base-main
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# --- final ---------------------------------------------------------
FROM base-${WHISPER_VARIANT}

ENV WHISPER_MODEL=ggml-tiny.bin
# Language code for whisper-server (`-l`). `auto` = detect; pin to a
# code (e.g. `ja`, `en`) when the deployment is single-language and
# you want lower latency / less misdetection.
ENV WHISPER_LANGUAGE=auto
# CPU threads used by whisper-server (`-t`). Default = 4 (matches
# whisper-server's own default). Bumping past the host's *physical*
# core count usually hurts whisper.cpp due to cache contention from
# Hyperthreading — pick a value with that in mind, not blindly nproc.

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# start-period is generous because the larger ggml models can take
# 30+ s to mmap and warm up on first start. retries*interval = ~5 min.
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s --retries=60 \
    CMD curl -s --connect-timeout 2 -o /dev/null http://localhost:8080/ || exit 1

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
