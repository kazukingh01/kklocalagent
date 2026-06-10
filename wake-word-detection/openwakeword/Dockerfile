FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download all openWakeWord models so the first /mic frame doesn't race
# a multi-hundred-MB model pull at runtime.
RUN python -c "import openwakeword; openwakeword.utils.download_models()"

COPY shim.py /app/shim.py

ENV WW_MIC_URL=ws://audio-io:7010/mic \
    WW_ORCHESTRATOR_URL=http://orchestrator:7000/events \
    WW_MODELS=alexa \
    WW_THRESHOLD=0.5 \
    WW_COOLDOWN_SEC=2.0 \
    WW_INFERENCE_FRAMEWORK=tflite \
    WW_LISTEN=0.0.0.0:7030

EXPOSE 7030

# /health is 200 only after model load AND /mic WS connect; gates
# `service_healthy` in compose.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7030/health || exit 1

ENTRYPOINT ["python", "-u", "/app/shim.py"]
