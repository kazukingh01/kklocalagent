# Integration-test variant of the audio-io stub. Built with the repo
# root as context so we can COPY fixtures from sibling modules
# (alexa_test.wav from wake-word-detection, jfk.wav from
# automatic-speech-recognition) without bind mounts.
#
#   docker compose build mic-stub
# is what compose.yaml runs.

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir 'websockets>=12,<14'

# jfk.wav is .gitignored by the asr module's test setup (only fetched
# on demand by automatic-speech-recognition/test/fetch-sample.sh). Pull
# it at build time so the integration smoke is self-contained.
RUN mkdir -p /samples && \
    curl -L --fail \
        -o /samples/jfk.wav \
        https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav

# alexa_test.wav is committed under wake-word-detection/test/data/.
COPY wake-word-detection/test/data/alexa_test.wav /samples/alexa.wav

# Reuse the upstream stub.py (which already supports SAMPLE_PATHS for
# multi-clip playback). One source of truth — the integration smoke
# benefits from any improvements made for the ASR smoke and vice versa.
COPY automatic-speech-recognition/test/mic-stub/stub.py /app/stub.py

WORKDIR /app
EXPOSE 7010

ENV HOST=0.0.0.0 PORT=7010 \
    SAMPLE_PATHS="/samples/alexa.wav,/samples/jfk.wav" \
    INTER_SAMPLE_SILENCE_MS=1500 \
    LOOP_DELAY_MS=3000 \
    LOOP_COUNT=1

CMD ["python", "-u", "/app/stub.py"]
