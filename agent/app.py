"""Agent API: ollama-compatible /api/chat backed by LangGraph.

The orchestrator's `pipeline.rs::llm_chat_streaming` continues to POST
the same body it always sent to ollama:

    {"model": ..., "messages": [{"role":"system|user", ...}],
     "stream": true}

— but the URL now points here. We pull the latest user turn out of
that body, run a single-node LangGraph chat against an internally
configured `ChatOllama`, and stream the assistant's deltas back as
ndjson lines whose shape matches ollama's /api/chat output exactly:

    {"message":{"content":"<delta>"},"done":false}\\n
    ...
    {"done":true}\\n

The orchestrator's parser only consults `message.content` and `done`
(see pipeline.rs:650-696), so anything else in our envelope is
ignored — we don't need to fake the rest of ollama's surface.

Architecture decisions:

* **Conversation memory** lives in this service's SQLite checkpointer
  (`AGENT_DB_PATH`). The orchestrator stays stateless on chat history.
* **System prompt** moves here from `ORCH_LLM_SYSTEM_PROMPT`. It's
  injected at LLM-invoke time, *not* persisted in graph state, so
  changing `AGENT_SYSTEM_PROMPT` and restarting the agent takes
  effect on every existing thread's next turn without rewriting
  checkpoints.
* **Session id** = LangGraph `thread_id`. One operator per agent
  process, so we generate a single session at startup and rotate it
  after `AGENT_SESSION_IDLE_SEC` of no /api/chat traffic (assume the
  operator walked away; the next turn is a fresh conversation).
* **No tools / MCP / approval in v1** — the graph is one chat node so
  the orchestrator's existing barge-in path (HTTP cancellation drops
  the response stream → ChatOllama drops its httpx connection →
  ollama stops generating) still works without agent-side
  bookkeeping. Tools land in a follow-up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import AsyncIterator

import aiosqlite
from aiohttp import web
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

log = logging.getLogger("agent")

OLLAMA_BASE_URL = os.environ.get("AGENT_OLLAMA_URL", "http://llm:11434")
MODEL_NAME = os.environ.get("AGENT_MODEL", "gemma3:4b")
SYSTEM_PROMPT = os.environ.get("AGENT_SYSTEM_PROMPT", "")
DB_PATH = os.environ.get("AGENT_DB_PATH", "/data/agent.sqlite")
SESSION_IDLE_SEC = float(os.environ.get("AGENT_SESSION_IDLE_SEC", "600"))
PORT = int(os.environ.get("AGENT_PORT", "7080"))


class SessionManager:
    """Owns the *current* session id and rotates it after a configurable
    idle gap.

    The voice orchestrator never sends a session_id (ollama's /api/chat
    body has no field for it) — there's only one operator per agent
    instance, so we treat the agent process lifetime as the upper
    bound on session length and "no /api/chat traffic for
    `idle_seconds`" as the lower bound. A rotation = the next turn
    starts with empty conversation memory in the checkpointer because
    `thread_id` is fresh.

    `time.monotonic()` rather than `time.time()` so an NTP step can't
    accidentally rotate (or refuse to rotate) a session.
    """

    def __init__(self, idle_seconds: float) -> None:
        self.idle_seconds = idle_seconds
        self.current_session = uuid.uuid4().hex
        self.last_active = time.monotonic()
        self.lock = asyncio.Lock()
        log.info(
            "session opened: %s (idle rotate after %.0fs)",
            self.current_session, idle_seconds,
        )

    async def claim(self) -> str:
        async with self.lock:
            now = time.monotonic()
            if (now - self.last_active) > self.idle_seconds:
                old = self.current_session
                self.current_session = uuid.uuid4().hex
                log.info(
                    "session rotated (idle %.1fs): %s -> %s",
                    now - self.last_active, old, self.current_session,
                )
            self.last_active = now
            return self.current_session


def build_graph(llm: ChatOllama, checkpointer: AsyncSqliteSaver):
    """Single-node chat graph.

    The system prompt is injected at LLM-invoke time from the
    env-configured constant, *not* persisted in state. Two reasons:

    1. Restarting the agent with a new `AGENT_SYSTEM_PROMPT` should
       take effect on existing sessions' next turn. If we persisted
       the SystemMessage in state the old prompt would stick until the
       thread rotated.
    2. Persisted state grows by one message per turn; keeping the
       system prompt out of it means the checkpoint row size doesn't
       carry a copy of the prompt forever.

    state["messages"] therefore only ever holds Human/AI exchanges.
    """

    async def chat_node(state: MessagesState):
        messages = list(state["messages"])
        if SYSTEM_PROMPT:
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        # ainvoke (not astream) inside the node — LangGraph's
        # stream_mode="messages" surfaces token-level chunks from the
        # underlying ChatOllama anyway. The full AIMessage returned
        # here is what gets persisted to the checkpoint.
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


def extract_user_text(body: dict) -> str:
    """Pluck the last `role:user` content from an ollama-compatible
    /api/chat body.

    The orchestrator sends `messages: [system?, user]` but the system
    role belongs to the agent in this architecture, so we ignore
    everything except the last user entry. Defensive against the
    orchestrator one day sending multi-turn history (right now it
    only sends one user turn per call): we still want the *latest*
    user message, not the first.
    """
    messages = body.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
    return ""


async def stream_chat(graph, sessions: SessionManager, user_text: str
                      ) -> AsyncIterator[dict]:
    session_id = await sessions.claim()
    config = {"configurable": {"thread_id": session_id}}
    input_state = {"messages": [HumanMessage(content=user_text)]}
    # stream_mode="messages" yields (message_chunk, metadata) tuples
    # for every LLM token chunk — exactly what we need to forward
    # delta-by-delta to the orchestrator's existing ndjson parser.
    async for chunk, _meta in graph.astream(
        input_state, config=config, stream_mode="messages"
    ):
        # `chunk.content` is `str` for plain text streams (today's
        # ChatOllama output), but newer langchain message types can
        # surface a `list[ContentBlock]` for multimodal / tool-call
        # deltas. The orchestrator's parser expects a string, so we
        # only forward str chunks — once tools are bound the list
        # branch will need its own handling.
        if (
            isinstance(chunk, AIMessageChunk)
            and isinstance(chunk.content, str)
            and chunk.content
        ):
            yield {"message": {"content": chunk.content}, "done": False}
    yield {"done": True}


async def chat_handler(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        return web.json_response(
            {"error": f"invalid JSON: {e}"}, status=400
        )
    user_text = extract_user_text(body)
    if not user_text:
        return web.json_response(
            {"error": "no user-role message with non-empty content"},
            status=400,
        )

    graph = request.app["graph"]
    sessions: SessionManager = request.app["sessions"]
    log.info("chat: text=%r", user_text[:120])

    # Stream ndjson back. If the client disconnects mid-stream
    # (orchestrator barge-in: JoinHandle::abort drops the reqwest
    # response on the orchestrator side), the next aiohttp write
    # raises, the generator's `async for` propagates the cancel, and
    # the LangGraph astream is dropped — which closes ChatOllama's
    # httpx connection to ollama and stops token generation upstream.
    # No explicit cancellation plumbing needed.
    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "application/x-ndjson"},
    )
    await resp.prepare(request)
    try:
        async for chunk in stream_chat(graph, sessions, user_text):
            line = json.dumps(chunk, ensure_ascii=False) + "\n"
            await resp.write(line.encode())
    except ConnectionResetError as e:
        # Client (orchestrator) dropped the response mid-stream — usual
        # cause is a barge-in: the orchestrator's JoinHandle::abort drops
        # its reqwest response, our next aiohttp write raises. The
        # connection is already gone so there's nothing to send back.
        log.info("chat stream cancelled: %s", type(e).__name__)
    except asyncio.CancelledError:
        # Re-raise per asyncio's cancellation contract — swallowing it
        # would break aiohttp's task lifecycle. write_eof() below would
        # also fail on a cancelled connection, so skip cleanup and let
        # the framework unwind.
        log.info("chat stream cancelled: CancelledError")
        raise
    except Exception as e:  # noqa: BLE001
        log.error("chat stream failed: %s", e)
    await resp.write_eof()
    return resp


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def session_handler(request: web.Request) -> web.Response:
    """Diagnostic: report the current session id and seconds since the
    last /api/chat. Useful for verifying idle rotation without grepping
    logs (`curl agent:7080/session` after waiting past
    `AGENT_SESSION_IDLE_SEC` should return a freshly-rotated id on the
    next chat)."""
    sessions: SessionManager = request.app["sessions"]
    async with sessions.lock:
        now = time.monotonic()
        return web.json_response({
            "session_id": sessions.current_session,
            "idle_sec": round(now - sessions.last_active, 2),
            "rotate_after_sec": sessions.idle_seconds,
        })


async def amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    log.info(
        "agent: ollama=%s model=%s db=%s",
        OLLAMA_BASE_URL, MODEL_NAME, DB_PATH,
    )

    # ChatOllama streams tokens from ollama via httpx. temperature=0
    # matches the orchestrator's previous /api/chat config (no temp
    # was sent, ollama's default for streaming is 0.8 — we override
    # to keep voice-agent replies deterministic for the same input).
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=MODEL_NAME,
        temperature=0,
    )

    # Open the checkpoint DB once and keep it open for the process
    # lifetime. AsyncSqliteSaver.setup() is idempotent — creates the
    # schema on first run, no-ops thereafter.
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = await aiosqlite.connect(DB_PATH)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()

    graph = build_graph(llm, saver)
    sessions = SessionManager(SESSION_IDLE_SEC)

    app = web.Application()
    app["graph"] = graph
    app["sessions"] = sessions
    app.router.add_get("/health", health_handler)
    app.router.add_get("/session", session_handler)
    app.router.add_post("/api/chat", chat_handler)

    # access_log=None: HEALTHCHECK pings /health every 10 s. Without
    # this, every probe shows up as a 200 line drowning out real
    # /api/chat traffic. /api/chat already logs its own lifecycle
    # ("chat: text=...", "chat stream cancelled", "chat stream failed")
    # so dropping the access log doesn't hide anything diagnostic.
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info("agent listening on :%d", PORT)

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await conn.close()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
