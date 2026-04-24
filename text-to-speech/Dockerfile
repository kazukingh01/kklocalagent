# Wraps the upstream VOICEVOX engine image. The engine self-contains all
# voice models (including ずんだもん, speaker id 3) and exposes its HTTP
# API on 0.0.0.0:50021 via the upstream's default CMD — we just add a
# HEALTHCHECK so dependent services can wait on `condition: service_healthy`.
#
# Override VOICEVOX_VARIANT at build time to switch CPU / GPU:
#   --build-arg VOICEVOX_VARIANT=cpu-latest      (default)
#   --build-arg VOICEVOX_VARIANT=nvidia-latest   (CUDA)
ARG VOICEVOX_VARIANT=cpu-latest

FROM voicevox/voicevox_engine:${VOICEVOX_VARIANT}

# Probe /version via the curl already bundled in the upstream image.
# The engine binary is PyInstaller-packed so there's no system python3 —
# curl is the only HTTP client present. /version is the cheapest endpoint
# that requires the engine to be fully initialized (returns engine semver).
# start-period is generous because first boot lazy-mmaps the .onnx voice
# models. retries*interval = ~2 min total before unhealthy.
HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=24 \
    CMD curl -sfS --connect-timeout 2 -o /dev/null http://localhost:50021/version || exit 1

EXPOSE 50021
