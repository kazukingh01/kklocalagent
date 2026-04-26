# tts-streamer: HTTP shim that turns a text request from the
# orchestrator into VOICEVOX synthesis + audio-io playback streaming.
#
# The orchestrator POSTs `{"text": "..."}` to /speak after each turn;
# this image handles the engine handshake, the 24 kHz → 16 kHz resample,
# and the framed WS push to audio-io's /spk endpoint (which lives
# outside compose, on the Windows host, in the production stack).

FROM python:3.12-slim

# ffmpeg is for the 24 kHz → 16 kHz mono s16le resample. Pinned to the
# slim image's apt to keep the layer reasonable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        'aiohttp>=3.9,<4' \
        'httpx>=0.27,<0.29' \
        'websockets>=12,<14'

WORKDIR /app
COPY synthesize.py .

EXPOSE 7070

# Liveness only — the dependency chain (VOICEVOX ready, audio-io
# reachable) varies between production and test, so dependents wait on
# the more authoritative healthchecks of the engine/stub instead.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://127.0.0.1:7070/health || exit 1

ENTRYPOINT ["python", "-u", "synthesize.py"]
