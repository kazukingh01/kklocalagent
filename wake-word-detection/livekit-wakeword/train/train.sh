#!/usr/bin/env bash
# Thin wrapper around `livekit-wakeword run`. Exists so the entry
# point is stable even if the upstream CLI flag set evolves; we
# re-pin the wrapper, callers don't have to.
#
# Expects `livekit-wakeword` already installed (pip install -r
# requirements.txt). GPU recommended; CPU is hours+.

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

# Resolve to an absolute path so `cd` later doesn't break it.
CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

cd "$(dirname "$0")"
mkdir -p out

echo "[train.sh] running livekit-wakeword run $CONFIG_ABS"
exec livekit-wakeword run "$CONFIG_ABS"
