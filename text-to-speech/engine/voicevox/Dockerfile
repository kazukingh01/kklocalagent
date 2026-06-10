ARG VOICEVOX_VARIANT=cpu-latest

FROM voicevox/voicevox_engine:${VOICEVOX_VARIANT}

# Bare TCP connect rather than `curl /version`: uvicorn logs every
# request at INFO, so an HTTP healthcheck spams the logs every 5 s.
# Liveness is equivalent — uvicorn only binds the port once the engine
# is fully initialised (lazy-mmap of the .onnx voice models, hence the
# generous start-period).
HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=24 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/50021' || exit 1

EXPOSE 50021
