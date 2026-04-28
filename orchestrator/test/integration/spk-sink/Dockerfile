# spk-sink: tiny WS receiver that mocks audio-io's /spk endpoint for
# the integration smoke. Counts received frames and exposes the count
# at GET /stats so the assert service can verify TTS audio actually
# reached the playback edge.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir 'aiohttp>=3.9,<4' 'websockets>=12,<14'

WORKDIR /app
COPY spk_sink.py .

EXPOSE 7010 7011

# Liveness only — the WS server binds simultaneously, so a healthy
# /stats response means /spk is also accepting connections.
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=10 \
    CMD curl -sfS --connect-timeout 2 http://127.0.0.1:7011/stats || exit 1

ENTRYPOINT ["python", "-u", "spk_sink.py"]
