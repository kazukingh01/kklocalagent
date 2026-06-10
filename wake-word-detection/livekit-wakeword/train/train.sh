#!/usr/bin/env bash
# `uv run` materialises the .venv from uv.lock on demand — no manual
# activate needed. GPU recommended; CPU is hours+.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <config.yaml>" >&2
    exit 2
fi

CONFIG="$1"
if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

cd "$(dirname "$0")"

echo "[train.sh] running livekit-wakeword run $CONFIG_ABS"
exec uv run livekit-wakeword run "$CONFIG_ABS"
