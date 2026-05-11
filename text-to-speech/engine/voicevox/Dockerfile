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

# Probe via a bare TCP connect to :50021 rather than `curl /version`.
# The HTTP variant works but uvicorn (inside the upstream image) logs
# every request at INFO, so a 5 s healthcheck spams a `GET /version
# 200 OK` line every 5 s into production logs. Bash's built-in
# /dev/tcp opens the socket and closes it without sending any HTTP
# bytes, so uvicorn's access logger never fires. Liveness check is
# equivalent — uvicorn only binds the port once the engine is fully
# initialised (after lazy-mmap of the .onnx voice models).
# start-period is generous for that first-boot mmap.
# retries * interval = ~2 min total before unhealthy.
HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=24 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/50021' || exit 1

EXPOSE 50021
