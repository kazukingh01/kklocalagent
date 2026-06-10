FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir 'websockets>=12,<14'

# jfk.wav is .gitignored in the asr module (fetched on demand by its
# fetch-sample.sh), so it cannot be COPYed; pull it at build time.
RUN mkdir -p /samples && \
    curl -L --fail \
        -o /samples/jfk.wav \
        https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav

COPY wake-word-detection/openwakeword/test/data/alexa_test.wav /samples/alexa.wav

COPY automatic-speech-recognition/test/mic-stub/stub.py /app/stub.py

WORKDIR /app
EXPOSE 7010

ENV HOST=0.0.0.0 PORT=7010 \
    SAMPLE_PATHS="/samples/alexa.wav,/samples/jfk.wav" \
    INTER_SAMPLE_SILENCE_MS=1500 \
    LOOP_DELAY_MS=3000 \
    LOOP_COUNT=1

CMD ["python", "-u", "/app/stub.py"]
