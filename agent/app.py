"""Agent API: ollama-compatible /api/chat backed by LangGraph.

The orchestrator's `pipeline.rs::llm_chat_streaming` continues to POST
the same body it always sent to ollama:

    {"model": ..., "messages": [{"role":"system|user", ...}],
     "stream": true}

— but the URL now points here. We pull the latest user turn out of
that body, run a LangGraph chat against an internally configured
`ChatOllama`, and stream the assistant's deltas back as ndjson lines
whose shape matches ollama's /api/chat output exactly:

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
* **Tools (issue #19)** are gated behind `AGENT_TOOLS_ENABLED`. When
  on, the chat graph is replaced with `create_react_agent` (LLM ↔
  tool loop). When off, the legacy single-node graph is used so
  pre-tools behaviour is preserved bit-for-bit. The stream filter
  drops tool-call deltas (would TTS structured data otherwise) and
  injects a per-tool "filler ack" chunk (e.g. "ちょっと検索してみるね")
  before slow tools so the user doesn't sit through a silent gap.
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
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from tools import TOOL_ACK_PHRASES, all_tools

log = logging.getLogger("agent")

OLLAMA_BASE_URL = os.environ.get("AGENT_OLLAMA_URL", "http://llm:11434")
MODEL_NAME = os.environ.get("AGENT_MODEL", "gemma3:4b")
SYSTEM_PROMPT = os.environ.get("AGENT_SYSTEM_PROMPT", "")
DB_PATH = os.environ.get("AGENT_DB_PATH", "/data/agent.sqlite")
SESSION_IDLE_SEC = float(os.environ.get("AGENT_SESSION_IDLE_SEC", "600"))
PORT = int(os.environ.get("AGENT_PORT", "7080"))

# Feature flag for issue #19. Default off so this PR is a no-op for
# anyone who doesn't opt in.
TOOLS_ENABLED = os.environ.get("AGENT_TOOLS_ENABLED", "false").lower() in ("1", "true", "yes")
# Caps the number of graph steps per turn (agent_node → tools → agent_node
# → ... ). 6 ≈ 3 tool calls in a chain. Beyond this we surface a fallback
# voice line rather than loop forever. issue #19 オープン項目 #6 の retry
# リミット。
RECURSION_LIMIT = int(os.environ.get("AGENT_TOOL_RECURSION_LIMIT", "6"))
# Recovery line spoken when the ReAct loop exceeds RECURSION_LIMIT
# (LLM kept trying tools but never produced a final answer).
RECURSION_FALLBACK_TEXT = os.environ.get(
    "AGENT_TOOL_FALLBACK_TEXT",
    "うまくできませんでした、すみません。",
)
# Suffix appended to AGENT_SYSTEM_PROMPT *only when tools are enabled* so we
# don't tell the LLM about tools that aren't wired. The phrasing matches the
# voice-agent persona (タメ口 / 短文 / TTS 向き). Override entirely with
# AGENT_TOOL_SYSTEM_SUFFIX if you want different guidance.
TOOL_SYSTEM_SUFFIX = os.environ.get(
    "AGENT_TOOL_SYSTEM_SUFFIX",
    " 使える機能があるときは自然に使って答えて。"
    "機能が失敗したら別の方法を試して、それでも無理なら正直に「できなかった」と伝えて。",
)


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


def build_legacy_graph(llm: ChatOllama, checkpointer: AsyncSqliteSaver):
    """Single-node chat graph (pre-tools fallback).

    Kept verbatim from the original implementation so AGENT_TOOLS_ENABLED=false
    deployments behave exactly as before issue #19.

    The system prompt is injected at LLM-invoke time from the env-configured
    constant, *not* persisted in state. Two reasons:

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


def build_react_graph(llm: ChatOllama, checkpointer: AsyncSqliteSaver):
    """ReAct (agent_node ↔ tools) graph for AGENT_TOOLS_ENABLED=true.

    `create_react_agent` handles tool binding, the agent-vs-tools router,
    and the loop back to agent_node after each tool result. We only
    customise the prompt (system prompt + tool-usage guidance) so the
    voice-agent persona is preserved while the LLM learns it has tools.

    Tools live in `tools.py::all_tools()`. Adding one there + entering
    its name in `TOOL_ACK_PHRASES` is enough to wire it — no changes
    needed here.
    """
    tools = all_tools()
    prompt = (SYSTEM_PROMPT + TOOL_SYSTEM_SUFFIX) if SYSTEM_PROMPT else TOOL_SYSTEM_SUFFIX.strip()
    # `prompt=` is injected as a SystemMessage on every LLM call, same
    # pattern as the legacy graph — i.e. not persisted in state. So
    # restarting with a different AGENT_SYSTEM_PROMPT still takes effect
    # on the next turn.
    return create_react_agent(
        llm,
        tools,
        prompt=prompt,
        checkpointer=checkpointer,
    )


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
    """Drive the graph for one turn and yield ndjson-shaped chunks.

    Three sources of LLM-side AIMessageChunk are interleaved when tools
    are on (one only, when tools are off):

      1. plain content tokens          → forward as `message.content` delta
      2. tool-call deltas              → DROP (TTS-ing structured JSON
                                          would be gibberish). Side
                                          effect: emit a one-shot ack
                                          chunk on the *first* time we
                                          see each new tool name so the
                                          user hears a filler line
                                          while the tool runs.
      3. ToolMessage (tool result)     → not an AIMessageChunk; the
                                          isinstance() filter drops it.

    `recursion_limit` caps the agent→tools→agent loop count per turn.
    On overflow we yield `RECURSION_FALLBACK_TEXT` as the final spoken
    line — better than leaving the speaker silent.
    """
    session_id = await sessions.claim()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    input_state = {"messages": [HumanMessage(content=user_text)]}

    # Tracks the most recent tool name we've spoken an ack for. Reset
    # per-call (= per-turn) so a fresh turn re-acks. Two consecutive
    # tool calls of *different* names within the same turn each get
    # their own ack; two of the *same* name don't re-ack (rare and
    # would feel repetitive on the speaker).
    last_acked_tool: str | None = None

    # Whether any *real* LLM-produced text was yielded (acks don't count).
    # If a turn ends with this still False — e.g. the LLM called a tool,
    # got back a `[denied]` ToolMessage, and silently stopped without
    # generating an apology — we emit `RECURSION_FALLBACK_TEXT` at the
    # end so the operator never hears total silence. Observed with
    # gemma4:e4b when allowlist-rejected shell commands feed back.
    real_content_yielded = False

    try:
        async for chunk, _meta in graph.astream(
            input_state, config=config, stream_mode="messages"
        ):
            if not isinstance(chunk, AIMessageChunk):
                # ToolMessage / SystemMessage etc — not for TTS.
                continue

            # Tool-call delta: each partial chunk lists 1+ tool_call_chunks
            # whose `name` may be partial early on (Ollama streams it
            # token-by-token). We only emit ack when a *complete* known
            # tool name appears in TOOL_ACK_PHRASES — partial names like
            # "get_" will simply miss the dict lookup and skip.
            if chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    name = tc.get("name")
                    if name and name != last_acked_tool:
                        ack = TOOL_ACK_PHRASES.get(name)
                        if ack:
                            yield {"message": {"content": ack}, "done": False}
                        last_acked_tool = name
                # Suppress the structured-data delta itself.
                continue

            # `chunk.content` is `str` for plain text streams (today's
            # ChatOllama output). Newer message types could surface a
            # `list[ContentBlock]` for multimodal — we ignore those
            # because the orchestrator's parser expects str.
            if isinstance(chunk.content, str) and chunk.content:
                real_content_yielded = True
                yield {"message": {"content": chunk.content}, "done": False}
    except GraphRecursionError:
        # ReAct loop ran past `recursion_limit` without producing a
        # final assistant message — e.g., the LLM kept calling a tool
        # that kept failing. Speak the fallback line so the user isn't
        # left wondering whether the agent crashed.
        log.warning(
            "recursion limit %d reached for session %s; emitting fallback",
            RECURSION_LIMIT, session_id,
        )
        yield {"message": {"content": RECURSION_FALLBACK_TEXT}, "done": False}
        real_content_yielded = True

    if not real_content_yielded:
        # Stream ended normally but the LLM produced no text — typically
        # after a tool error returned `[denied]` / `[error]` content that
        # the LLM decided not to comment on. Voice agent must always say
        # SOMETHING; emit the fallback so the speaker isn't dead.
        log.warning(
            "no LLM text yielded for session %s; emitting fallback",
            session_id,
        )
        yield {"message": {"content": RECURSION_FALLBACK_TEXT}, "done": False}

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
    # With tools enabled, asyncio-native tool implementations get the
    # same CancelledError so subprocess.kill() / aiohttp.close() fire
    # automatically. No explicit cancellation plumbing needed.
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
            "tools_enabled": TOOLS_ENABLED,
        })


async def amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    log.info(
        "agent: ollama=%s model=%s db=%s tools=%s recursion_limit=%d",
        OLLAMA_BASE_URL, MODEL_NAME, DB_PATH, TOOLS_ENABLED, RECURSION_LIMIT,
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

    graph = build_react_graph(llm, saver) if TOOLS_ENABLED else build_legacy_graph(llm, saver)
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
