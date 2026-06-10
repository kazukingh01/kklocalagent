ARG VOICEVOX_VARIANT=cpu-latest

FROM voicevox/voicevox_engine:${VOICEVOX_VARIANT}

# Bare TCP connect rather than `curl /version`: uvicorn inside the
# upstream image logs every request at INFO, so an HTTP healthcheck
# spams a `GET /version 200 OK` line every 5 s into production logs.
# Bash's /dev/tcp opens and closes the socket without sending HTTP
# bytes, so the access logger never fires. Liveness is equivalent —
# uvicorn only binds the port once the engine is fully initialised
# (after lazy-mmap of the .onnx voice models); start-period is
# generous for that first-boot mmap.
HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=24 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/50021' || exit 1

EXPOSE 50021
