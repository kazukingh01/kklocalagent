FROM python:3.11-slim-bookworm
WORKDIR /app
# aiohttp serves the /sink endpoint (server) and polls the spk-sink
# /stats endpoint (client) — same dependency for both directions.
RUN pip install --no-cache-dir 'aiohttp>=3.9,<4'
COPY assert.py /app/assert.py
ENTRYPOINT ["python", "-u", "/app/assert.py"]
