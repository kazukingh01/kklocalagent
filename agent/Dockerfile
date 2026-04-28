# Agent API: ollama-compatible /api/chat backed by LangGraph + SQLite
# checkpointer. Sits between the orchestrator and ollama so the
# orchestrator's pipeline.rs::llm_chat_streaming sees a familiar API
# while conversation memory (and future tool calls / MCP / approval)
# live here.
FROM python:3.11-slim-bookworm

# curl for the HEALTHCHECK probe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

# /data is mounted as a named volume in compose so conversation memory
# survives `docker compose down && up`. AGENT_DB_PATH defaults to a
# file inside it; override to relocate (e.g. testing against a tmpfs).
ENV AGENT_OLLAMA_URL=http://llm:11434 \
    AGENT_MODEL=gemma3:4b \
    AGENT_SYSTEM_PROMPT="" \
    AGENT_DB_PATH=/data/agent.sqlite \
    AGENT_SESSION_IDLE_SEC=600 \
    AGENT_PORT=7080

EXPOSE 7080

# /health flips to 200 once the aiohttp server is up; the LangGraph
# build + SqliteSaver.setup() happen before /health is wired so any
# 200 response means the agent is fully ready to serve /api/chat.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD curl -sfS --connect-timeout 2 http://localhost:7080/health || exit 1

ENTRYPOINT ["python", "-u", "/app/app.py"]
