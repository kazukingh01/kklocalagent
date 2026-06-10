# Long-lived (CMD = sleep infinity) so test.sh can `docker exec` the
# test runner once per orchestrator config flavor.

FROM python:3.11-slim-bookworm
WORKDIR /app
RUN pip install --no-cache-dir 'aiohttp>=3.9,<4'
COPY harness.py /app/harness.py
CMD ["sleep", "infinity"]
