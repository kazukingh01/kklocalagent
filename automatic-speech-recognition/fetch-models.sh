#!/bin/sh
# Download a whisper.cpp ggml model into ./models. Default = ggml-tiny.bin
# (~75 MB). Override by passing the file name as the first argument:
#   ./fetch-models.sh ggml-base.bin
#   ./fetch-models.sh ggml-small.bin

set -e

DIR=$(cd "$(dirname "$0")" && pwd)
MODELS_DIR="$DIR/models"
mkdir -p "$MODELS_DIR"

MODEL="${1:-ggml-tiny.bin}"
DEST="$MODELS_DIR/$MODEL"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${MODEL}"

if [ -f "$DEST" ]; then
    echo "$DEST already exists, skipping"
    exit 0
fi

echo "downloading $URL"
echo "          -> $DEST"
curl -L --fail --progress-bar -o "$DEST" "$URL"
echo "done"
