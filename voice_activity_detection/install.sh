#!/usr/bin/env bash
# Install webrtcvad and its build prerequisites for the
# voice_activity_detection service.
#
# Targets: Ubuntu 22.04+ / WSL2 (Ubuntu). macOS / native Windows not covered —
# install build-essential / python3-dev equivalents manually there.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { printf '\033[1;32m[vad-install]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[vad-install]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. OS build deps -----------------------------------------------------
# webrtcvad is a C extension with no pre-built wheels on PyPI, so the
# interpreter needs a compiler toolchain and Python headers to build it.
if ! command -v apt-get >/dev/null 2>&1; then
    die "apt-get not found — this script targets Ubuntu / WSL2. Install build-essential and the matching python3 headers manually on other platforms, then rerun from the uv step."
fi

need_apt=()
dpkg -s build-essential >/dev/null 2>&1 || need_apt+=(build-essential)
dpkg -s python3-dev     >/dev/null 2>&1 || need_apt+=(python3-dev)
dpkg -s python3-venv    >/dev/null 2>&1 || need_apt+=(python3-venv)

if [ "${#need_apt[@]}" -gt 0 ]; then
    log "installing OS packages: ${need_apt[*]}"
    sudo apt-get update
    sudo apt-get install -y "${need_apt[@]}"
else
    log "OS build dependencies already present"
fi

# ---- 2. uv ----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv (https://docs.astral.sh/uv/)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv not on PATH — add \$HOME/.local/bin to PATH and rerun"
log "uv $(uv --version | awk '{print $2}')"

# ---- 3. Python deps -------------------------------------------------------
log "syncing Python dependencies (webrtcvad) via uv"
uv sync

# ---- 4. smoke test --------------------------------------------------------
log "smoke test: import webrtcvad and classify 20 ms of silence at 16 kHz"
uv run python -W ignore::UserWarning - <<'PY'
import webrtcvad
vad = webrtcvad.Vad(2)  # aggressiveness 0-3 (higher = more aggressive)
# 20 ms frame @ 16 kHz mono s16le = 320 samples * 2 B = 640 bytes of silence.
frame = b"\x00" * 640
assert vad.is_speech(frame, 16000) is False, "unexpected speech on pure silence"
print("webrtcvad OK — is_speech(silence)=False")
PY

log "done. webrtcvad is ready."
