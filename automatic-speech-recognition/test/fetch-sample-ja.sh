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

set -e

DIR=$(cd "$(dirname "$0")" && pwd)
SAMPLES_DIR="$DIR/samples"
mkdir -p "$SAMPLES_DIR"

DEST="$SAMPLES_DIR/test-ja.wav"
TEXT="${TEXT:-こんにちは。私はクロードです。これは音声認識のテストです。今日はいい天気ですね。}"

if [ -f "$DEST" ]; then
    echo "$DEST already exists, skipping (delete it to regenerate)"
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2
    exit 1
fi

echo "synthesizing → $DEST"
echo "          text: $TEXT"

# Stream the wav to the host file. Pip / apt noise is funneled to stderr
# so it doesn't corrupt the wav payload on stdout.
docker run --rm -i --entrypoint sh python:3.12-slim -c "
{
    pip install --disable-pip-version-check -q gtts >/dev/null 2>&1
    apt-get update -qq >/dev/null 2>&1
    apt-get install -qq -y ffmpeg >/dev/null 2>&1
    python3 -c \"
from gtts import gTTS
import sys
gTTS(sys.argv[1], lang='ja').save('/tmp/jp.mp3')
\" \"$TEXT\"
} >&2
ffmpeg -loglevel error -i /tmp/jp.mp3 -ar 16000 -ac 1 -sample_fmt s16 -f wav pipe:1
" > "$DEST"

if [ ! -s "$DEST" ]; then
    echo "generation failed (empty file)" >&2
    rm -f "$DEST"
    exit 1
fi

echo "done ($(wc -c < "$DEST") bytes)"
