#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ORCH_IMAGE="kklocalagent/orchestrator-test"
HARNESS_IMAGE="kklocalagent/orchestrator-test-harness"
NETWORK="orch-test-net"

ORCH_NAME="orch-under-test"
HARNESS_NAME="orch-test-harness"

ORCH_BASE_ENV=(
  -e "RUST_LOG=info,orchestrator=info"
  -e "ORCH_LISTEN=0.0.0.0:7000"
  -e "ORCH_ASR_URL=http://${HARNESS_NAME}:9100/inference"
  -e "ORCH_LLM_URL=http://${HARNESS_NAME}:9200/api/chat"
  -e "ORCH_LLM_MODEL=mock"
  -e "ORCH_LLM_SYSTEM_PROMPT=You are a test assistant."
  -e "ORCH_TTS_URL=http://${HARNESS_NAME}:9300/speak"
  -e "ORCH_TTS_STOP_URL=http://${HARNESS_NAME}:9300/stop"
  -e "ORCH_RESULT_SINK_URL=http://${HARNESS_NAME}:9400/sink"
  # Disable the post-wake SE dropout: the harness fires WakeWordDetected
  # and SpeechEnded back-to-back, which production's 800 ms default would
  # correctly classify as "VAD echoing the wake word" and drop.
  -e "ORCH_POST_WAKE_SE_DROPOUT_MS=0"
  # Disable the post-TTS VAD quiet window: scenarios run back-to-back in
  # one orch process, so the 500 ms window after a TTS-bearing turn
  # straddles the next scenario's first SE and drops it (manifests as
  # "expected ASR=1, got 0" right after `system_prompt_prepended`).
  -e "ORCH_TTS_TAIL_QUIET_MS=0"
)

cleanup() {
  set +e
  docker rm -f "$ORCH_NAME" >/dev/null 2>&1
  docker rm -f "$HARNESS_NAME" >/dev/null 2>&1
  docker network rm "$NETWORK" >/dev/null 2>&1
}
trap cleanup EXIT

echo "=== building images ==="
docker build -t "$ORCH_IMAGE" .. >/dev/null
docker build -t "$HARNESS_IMAGE" ./harness >/dev/null

echo "=== creating private network ==="
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

  docker rm -f "$ORCH_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$ORCH_NAME" --network "$NETWORK" \
      "${ORCH_BASE_ENV[@]}" "${extra_env[@]}" \
      "$ORCH_IMAGE" >/dev/null

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

  (docker logs -f "$ORCH_NAME" 2>&1 | sed -u 's/^/[orch]    /') &
  local log_pid=$!

  local exit_code=0
  docker exec "$HARNESS_NAME" python -u /app/harness.py \
      --orch-url "http://${ORCH_NAME}:7000" \
      --flavor "$flavor" 2>&1 \
      | sed -u 's/^/[harness] /' \
      || exit_code=$?

  kill "$log_pid" 2>/dev/null || true
  wait "$log_pid" 2>/dev/null || true

  return $exit_code
}

run_phase strict \
    -e "ORCH_WAKE_REQUIRED=true" \
    -e "ORCH_WAKE_WINDOW_MS=2000" \
    -e "ORCH_TURN_FOLLOWUP_WINDOW_MS=2000" \
    -e "ORCH_WAKE_BARGE_IN=true"

run_phase loose \
    -e "ORCH_WAKE_REQUIRED=false"

run_phase no-barge \
    -e "ORCH_WAKE_REQUIRED=true" \
    -e "ORCH_WAKE_WINDOW_MS=2000" \
    -e "ORCH_TURN_FOLLOWUP_WINDOW_MS=2000" \
    -e "ORCH_WAKE_BARGE_IN=false"

echo
echo "=== ALL TESTS PASSED ==="
