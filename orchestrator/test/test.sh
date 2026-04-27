#!/usr/bin/env bash
# Orchestrator-only scenario tests.
#
# Stands up exactly one orchestrator container at a time (the system
# under test) plus one long-lived harness container that hosts mock
# backends *and* drives the orchestrator from the upstream side. The
# harness impersonates whatever module the test needs — VAD posting
# SpeechEnded, wake-word-detection posting WakeWordDetected, etc. —
# and asserts which mock backends fired in response.
#
# Three orchestrator config flavors are exercised in sequence:
#
#   strict    — wake.required=true,  arm=2 s, barge_in=true   (v1.0 default)
#   loose     — wake.required=false                           (v0.1 fallback)
#   no-barge  — wake.required=true,  barge_in=false           (uninterruptible reply)
#
# The harness keeps running across all three phases; only the
# orchestrator container is replaced when the config changes.
#
# Run:
#   ./test.sh
#
# Exit 0 on green; non-zero on the first failing flavor (strict
# group runs to completion before bailing — we want to know
# everything that broke, not just the first symptom).

set -euo pipefail

cd "$(dirname "$0")"

ORCH_IMAGE="kklocalagent/orchestrator-test"
HARNESS_IMAGE="kklocalagent/orchestrator-test-harness"
NETWORK="orch-test-net"

ORCH_NAME="orch-under-test"
HARNESS_NAME="orch-test-harness"

# Base env shared across all orchestrator flavors. Each flavor
# overrides only the wake-related vars.
ORCH_BASE_ENV=(
  -e "RUST_LOG=info,orchestrator=info"
  -e "ORCH_LISTEN=0.0.0.0:7000"
  -e "ORCH_ASR_URL=http://${HARNESS_NAME}:9100/inference"
  -e "ORCH_LLM_URL=http://${HARNESS_NAME}:9200/api/chat"
  -e "ORCH_LLM_MODEL=mock"
  # Short fixed prompt for the test — production uses the longer
  # voice-assistant persona via compose. The harness's
  # `system_prompt_prepended` scenario only checks that *some* non-
  # empty system content reaches the LLM in role:"system" before the
  # user turn, so the exact text doesn't matter.
  -e "ORCH_LLM_SYSTEM_PROMPT=You are a test assistant."
  -e "ORCH_TTS_URL=http://${HARNESS_NAME}:9300/speak"
  -e "ORCH_TTS_STOP_URL=http://${HARNESS_NAME}:9300/stop"
  -e "ORCH_RESULT_SINK_URL=http://${HARNESS_NAME}:9400/sink"
)

cleanup() {
  # Best-effort tear-down — `set +e` so a missing container doesn't
  # fail the trap and mask the real exit code.
  set +e
  docker rm -f "$ORCH_NAME" >/dev/null 2>&1
  docker rm -f "$HARNESS_NAME" >/dev/null 2>&1
  docker network rm "$NETWORK" >/dev/null 2>&1
}
trap cleanup EXIT

echo "=== building images ==="
# Orchestrator: same Dockerfile as production. We want this test to
# fail if the production binary does — no test-only build flags.
docker build -t "$ORCH_IMAGE" .. >/dev/null
docker build -t "$HARNESS_IMAGE" ./harness >/dev/null

echo "=== creating private network ==="
# Recreate from scratch so a stale network from an aborted prior
# run doesn't leak addresses or DNS entries.
docker network rm "$NETWORK" >/dev/null 2>&1 || true
docker network create "$NETWORK" >/dev/null

echo "=== starting harness (long-lived) ==="
docker run -d --name "$HARNESS_NAME" --network "$NETWORK" \
    "$HARNESS_IMAGE" >/dev/null

run_phase() {
  local flavor=$1
  shift
  local extra_env=("$@")

  echo
  echo "=== flavor: ${flavor} ==="

  # Replace any prior orch container — config is fixed at startup,
  # so flavor switches require a full restart.
  docker rm -f "$ORCH_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$ORCH_NAME" --network "$NETWORK" \
      "${ORCH_BASE_ENV[@]}" "${extra_env[@]}" \
      "$ORCH_IMAGE" >/dev/null

  # Wait for the Dockerfile HEALTHCHECK to flip to healthy. The orch
  # comes up in ~1 s but the healthcheck has a 15 s start_period so
  # we give it 30 s of slack here.
  local status="starting"
  local i
  for i in $(seq 1 30); do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$ORCH_NAME" 2>/dev/null || echo "starting")
    [[ "$status" == "healthy" ]] && break
    sleep 1
  done
  if [[ "$status" != "healthy" ]]; then
    echo "FAIL: orchestrator never became healthy (status=${status})"
    echo "--- orchestrator logs ---"
    docker logs "$ORCH_NAME" 2>&1 | tail -40
    return 1
  fi

  # Stream orchestrator container stdout in parallel with the harness
  # driver so both log streams appear interleaved on this script's
  # stdout in real time. Each line is prefixed with the source so a
  # mixed dump remains readable:
  #   [orch]    INFO orch::events: received event name=...
  #   [harness] PASS: ...
  # `sed -u` (unbuffered) and `python -u` are required — without
  # them stdio block-buffering when piped delays output by hundreds
  # of ms, which defeats "see what happened in real time".
  (docker logs -f "$ORCH_NAME" 2>&1 | sed -u 's/^/[orch]    /') &
  local log_pid=$!

  local exit_code=0
  # `set -o pipefail` (set at the top of the script) makes the pipe
  # exit non-zero when docker exec fails, even though sed succeeds.
  # `|| exit_code=$?` captures that without tripping `set -e`.
  docker exec "$HARNESS_NAME" python -u /app/harness.py \
      --orch-url "http://${ORCH_NAME}:7000" \
      --flavor "$flavor" 2>&1 \
      | sed -u 's/^/[harness] /' \
      || exit_code=$?

  # Stop the orch log follower before the next phase replaces the
  # container — otherwise the bg sed keeps writing into the new
  # container's logs and the prefixes lose their flavor context.
  kill "$log_pid" 2>/dev/null || true
  wait "$log_pid" 2>/dev/null || true

  return $exit_code
}

# v1.0 default: wake-gated, 2 s arm window, barge-in on. Most of the
# scenarios live here.
run_phase strict \
    -e "ORCH_WAKE_REQUIRED=true" \
    -e "ORCH_WAKE_ARM_WINDOW_MS=2000" \
    -e "ORCH_WAKE_BARGE_IN=true"

# v0.1 fallback: wake gating disabled (every SpeechEnded fires).
# Operator-flag for tests / hosts without the wake-word model.
run_phase loose \
    -e "ORCH_WAKE_REQUIRED=false"

# Variant: wake gated but barge-in off (assistant always finishes its
# reply before listening again).
run_phase no-barge \
    -e "ORCH_WAKE_REQUIRED=true" \
    -e "ORCH_WAKE_ARM_WINDOW_MS=2000" \
    -e "ORCH_WAKE_BARGE_IN=false"

echo
echo "=== ALL TESTS PASSED ==="
