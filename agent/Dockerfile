# Agent API: ollama-compatible /api/chat backed by LangGraph.
FROM python:3.11-slim-bookworm

# curl for HEALTHCHECK. procps (uptime) / iproute2 (ip) are required by
# run_shell's default allowlist — without them the LLM's request turns
# into a "command not found" loop.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl procps iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py tools.py sandbox.py /app/
# テストハーネス (issue #19 step 2) を `docker compose exec` で叩けるように含める。
COPY experiments /app/experiments

# /data is a named volume in compose so conversation memory survives down/up.
ENV AGENT_OLLAMA_URL=http://llm:11434 \
    AGENT_MODEL=gemma4:e4b \
    AGENT_SYSTEM_PROMPT="" \
    AGENT_DB_PATH=/data/agent.sqlite \
    AGENT_SESSION_IDLE_SEC=600 \
    AGENT_PORT=7080

EXPOSE 7080

# LangGraph build + SqliteSaver.setup() happen before /health is wired,
# so any 200 means the agent is fully ready to serve /api/chat.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7080/health || exit 1

ENTRYPOINT ["python", "-u", "/app/app.py"]
