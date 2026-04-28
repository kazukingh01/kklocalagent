# Wake-word detection shim: subscribes to audio-io /mic, runs openWakeWord
# on the PCM stream, and POSTs a WakeWordDetected event to the orchestrator
# when any configured model's score crosses the threshold.
FROM python:3.11-slim-bookworm

# curl for the HEALTHCHECK probe; build-essential + libsndfile1 for wheels
# that occasionally fall through to source builds on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download the shared feature-extractor backbone + every pre-trained
# classifier so the image is self-contained and the first /mic frame
# doesn't race a multi-hundred-MB model pull. Cache lives at
# /usr/local/lib/python3.11/site-packages/openwakeword/resources/models.
RUN python -c "import openwakeword; openwakeword.utils.download_models()"

COPY shim.py /app/shim.py

ENV WW_MIC_URL=ws://audio-io:7010/mic \
    WW_ORCHESTRATOR_URL=http://orchestrator:7000/events \
    WW_MODELS=alexa \
    WW_THRESHOLD=0.5 \
    WW_COOLDOWN_SEC=2.0 \
    WW_INFERENCE_FRAMEWORK=tflite \
    WW_PORT=7030

EXPOSE 7030

# /health flips to 200 only after the openWakeWord model is loaded AND the
# /mic WebSocket is connected. Gates `service_healthy` in compose so
# dependents don't race the backend being ready.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7030/health || exit 1

ENTRYPOINT ["python", "-u", "/app/shim.py"]
