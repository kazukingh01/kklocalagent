#!/bin/sh
# Generate a Japanese test wav by running gTTS (Google Translate's TTS
# endpoint, accessed by the `gtts` Python package) inside a one-shot
# python:3.12-slim docker container, then converting to the wire format
# the mic-stub expects: s16le, 16 kHz, mono.
#
# Output: ./samples/test-ja.wav (~270 KB, ~13 s).
#
# Why TTS instead of a hosted recording: there is no widely-stable,
# CC-licensed JP wav at a known URL that whisper-large transcribes
# meaningfully. gTTS produces clean female JP speech that whisper
# transcribes nearly verbatim — fine for a smoke test.
#
# Implementation note: the docker container writes the wav to a
# bind-mounted host directory rather than streaming it over stdout.
# Stdout-piping is fragile — any pip/apt/network noise that escapes the
# `>/dev/null` redirects would corrupt the wav, producing an empty file
# that mic-stub fails to parse with EOFError.

set -e

DIR=$(cd "$(dirname "$0")" && pwd)
SAMPLES_DIR="$DIR/samples"
mkdir -p "$SAMPLES_DIR"

DEST="$SAMPLES_DIR/test-ja.wav"
TEXT="${TEXT:-こんにちは。私はクロードです。これは音声認識のテストです。今日はいい天気ですね。}"

is_valid_wav() {
    [ -s "$1" ] && [ "$(head -c 4 "$1" 2>/dev/null)" = "RIFF" ]
}

if [ -f "$DEST" ]; then
    if is_valid_wav "$DEST"; then
        echo "$DEST already exists (valid wav), skipping"
        exit 0
    fi
    echo "$DEST exists but is not a valid wav; regenerating"
    rm -f "$DEST"
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2
    exit 1
fi

echo "synthesizing → $DEST"
echo "          text: $TEXT"

sudo docker run --rm \
    -v "$SAMPLES_DIR":/out \
    -e TEXT="$TEXT" \
    --entrypoint sh \
    python:3.12-slim -c '
set -e
pip install --disable-pip-version-check -q gtts >/dev/null
apt-get update -qq >/dev/null
apt-get install -qq -y ffmpeg >/dev/null
python3 -c "
from gtts import gTTS
import os
gTTS(os.environ[\"TEXT\"], lang=\"ja\").save(\"/tmp/jp.mp3\")
"
ffmpeg -loglevel error -y -i /tmp/jp.mp3 \
    -ar 16000 -ac 1 -sample_fmt s16 /out/test-ja.wav
'

if ! is_valid_wav "$DEST"; then
    echo "generation failed (output is missing or not a valid RIFF/WAV file)" >&2
    rm -f "$DEST"
    exit 1
fi

echo "done ($(wc -c < "$DEST") bytes)"
