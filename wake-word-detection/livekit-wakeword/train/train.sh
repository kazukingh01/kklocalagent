#!/usr/bin/env bash
# Thin wrapper around `livekit-wakeword run`. Exists so the entry
# point is stable even if the upstream CLI flag set evolves; we
# re-pin the wrapper, callers don't have to.
#
# Runs via `uv run` so the .venv is materialised from uv.lock on
# demand — no manual `source .venv/bin/activate` step required.
# GPU recommended; CPU is hours+.

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
