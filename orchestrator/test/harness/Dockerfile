# Test harness for the orchestrator. Hosts mock backends (ASR / LLM
# / TTS / sink) on :9100..:9400 and runs scenario test cases that
# POST events at the orchestrator under test.
#
# Long-lived (entrypoint = sleep infinity) so test.sh can `docker
# exec` the actual test runner with a different `--flavor` per
# orchestrator config without re-pulling the image each time.

FROM python:3.11-slim-bookworm
WORKDIR /app
RUN pip install --no-cache-dir 'aiohttp>=3.9,<4'
COPY harness.py /app/harness.py
CMD ["sleep", "infinity"]
