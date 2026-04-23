#!/bin/sh
# Download a small public-domain wav sample for the audio-io→vad→asr
# smoke test. Default = whisper.cpp's bundled JFK clip (~11 s, 16 kHz
# mono, public domain).

set -e

DIR=$(cd "$(dirname "$0")" && pwd)
SAMPLES_DIR="$DIR/samples"
mkdir -p "$SAMPLES_DIR"

DEST="$SAMPLES_DIR/jfk.wav"
URL="https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"

if [ -f "$DEST" ]; then
    echo "$DEST already exists, skipping"
    exit 0
fi

echo "downloading $URL"
echo "          -> $DEST"
curl -L --fail --progress-bar -o "$DEST" "$URL"
echo "done"
