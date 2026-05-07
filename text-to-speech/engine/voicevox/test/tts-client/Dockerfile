FROM python:3.12-slim

# ffmpeg is for resampling the engine's 24 kHz WAV down to audio-io's
# 16 kHz s16le mono wire format in `stream` mode. Pinned to the slim
# image's apt to keep the layer small (~80 MB).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir 'httpx>=0.27,<0.29' 'websockets>=12,<14'

WORKDIR /app
COPY synthesize.py .

ENTRYPOINT ["python", "-u", "synthesize.py"]
