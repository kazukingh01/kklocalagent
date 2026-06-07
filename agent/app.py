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
  on, the chat graph is a hand-rolled ReAct loop (agent ↔ tools) — see
  `build_react_graph`; a "terminal" tool like stop_audio ends the turn
  without a second LLM call. When off, the legacy single-node graph is
  used so pre-tools behaviour is preserved bit-for-bit. The stream
  filter drops tool-call deltas (would TTS structured data otherwise)
  and injects a per-tool "filler ack" chunk (e.g. "ちょっと検索してみるね")
  when the tool is invoked so the user doesn't sit through a silent gap.
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
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOL_ACK_PHRASES, all_tools

log = logging.getLogger("agent")

OLLAMA_BASE_URL = os.environ.get("AGENT_OLLAMA_URL", "http://llm:11434")
MODEL_NAME = os.environ.get("AGENT_MODEL", "gemma3:4b")
SYSTEM_PROMPT = os.environ.get("AGENT_SYSTEM_PROMPT", "")
DB_PATH = os.environ.get("AGENT_DB_PATH", "/data/agent.sqlite")
SESSION_IDLE_SEC = float(os.environ.get("AGENT_SESSION_IDLE_SEC", "600"))
PORT = int(os.environ.get("AGENT_PORT", "7080"))

# Feature flag for issue #19. The in-code default is off so a bare
# `import app` with no env set stays a no-op, but compose.yaml passes
# AGENT_TOOLS_ENABLED=true — deployed stacks therefore run with tools
# ON by default (set it false in .env to fall back to the legacy
# single-node chat graph).
TOOLS_ENABLED = os.environ.get("AGENT_TOOLS_ENABLED", "false").lower() in ("1", "true", "yes")
# Caps the number of graph steps per turn (agent_node → tools → agent_node
# → ... ). 6 ≈ 3 tool calls in a chain. Beyond this we surface a fallback
# voice line rather than loop forever. issue #19 オープン項目 #6 の retry
# リミット。
RECURSION_LIMIT = int(os.environ.get("AGENT_TOOL_RECURSION_LIMIT", "6"))
# Trim the accumulated chat history to this many (estimated) tokens before
# each LLM call. Without it, a long tool-heavy conversation grows the
# checkpointed history until it fills ollama's context window — the prompt
# then leaves no room to generate, the reply truncates to empty, and the
# agent gets stuck emitting its fallback line every turn (only recovering on
# session idle-rotation / restart). Kept well under OLLAMA_CONTEXT_LENGTH
# (default 16384) since the system prompt + tool schemas + generation room
# are *on top* of this budget (they are not part of state["messages"]).
MAX_HISTORY_TOKENS = int(os.environ.get("AGENT_MAX_HISTORY_TOKENS", "4096"))
# Tools whose *successful* result ends the turn with a fixed confirmation (their
# TOOL_ACK_PHRASES line) instead of a second LLM call. They are instant and
# deterministic — re-invoking the model just to paraphrase "done" only adds a
# whole model-call of latency. A failed one falls back to the LLM so it can
# explain. See build_react_graph's terminal-tool routing.
TERMINAL_TOOLS = {"stop_audio"}
# Log every LLM turn's raw output (tool_calls / content / additional_kwargs incl.
# any reasoning) at INFO so tool-calling can be debugged — e.g. did the model
# actually emit `stop_audio`, or just reason+text? Set AGENT_LOG_LLM_RAW=false to
# quiet it once things are working.
LOG_LLM_RAW = os.environ.get("AGENT_LOG_LLM_RAW", "true").lower() in ("1", "true", "yes")
# AGENT_REASONING — gemma4-style "thinking" mode selector. There are TWO
# distinct ways a model can think, and they are NOT interchangeable across
# models, so this is a 3-way mode rather than a bool:
#
#   0 = off          no thinking. We send think=FALSE (explicitly — omitting
#                    it lets a thinking-capable backend default to ON) and add
#                    no <|think|> token. Safe on every model.
#   1 = native       ollama's native `think=true` (ChatOllama reasoning=True).
#                    The thinking is kept out of `content` (it lands in
#                    additional_kwargs), so it never reaches TTS. Works only
#                    on models that advertise native thinking to ollama
#                    (e.g. the `gemma4:e4b` library tag). On an imported HF
#                    GGUF that lacks it, `think=true` returns HTTP 400.
#   2 = prompt-token gemma4's other mechanism: prepend AGENT_THINK_TOKEN
#                    (default <|think|>) to the system prompt. We send
#                    think=false to suppress native thinking, and the prompt
#                    token drives the reasoning instead. For GGUFs that reject
#                    native think (e.g. hf.co/unsloth/gemma-4-12b-it-GGUF).
#                    NOTE: in this mode the reasoning may surface inline in
#                    `content`; the orchestrator strips <...> spans before TTS
#                    so it isn't spoken.
#
# CAVEAT (modes 1 & 2): thinking adds latency and has been observed to
# suppress tool calls on gemma4 12B (the model reasons + replies with text
# instead of calling e.g. stop_audio). Use 0 if tool-calling regresses.
# Back-compat: on/true/yes → 1, off/false/no → 0. Default 1.
def _reasoning_mode() -> int:
    v = os.environ.get("AGENT_REASONING", "1").strip().lower()
    if v in ("0", "off", "false", "no", "none"):
        return 0
    if v == "2":
        return 2
    # "1" / "on" / "true" / "yes" / "default" / "model" / anything → native
    return 1


REASONING_MODE = _reasoning_mode()
# ChatOllama `reasoning=`: True for native mode (1) → think=true. Modes 0 and 2
# send think=FALSE (NOT None). This matters: None omits the `think` field, and a
# thinking-capable backend (recent ollama / llama.cpp peg-gemma4) then falls
# back to its DEFAULT, which is thinking ON — so AGENT_REASONING=0 would still
# reason. think=false disables it explicitly, and is safe even on a GGUF that
# can't think (the "does not support thinking" 400 only fires on think=TRUE —
# i.e. when you ask it TO think; "don't think" is trivially honoured). Mode 2
# suppresses native think and reasons via the THINK_PREFIX prompt token instead.
REASONING = True if REASONING_MODE == 1 else False
# Token that flips gemma4 into thinking when placed at the START of the system
# prompt (mode 2). Overridable so a model that spells it differently can be
# accommodated without a code change. Empty prefix in modes 0/1.
THINK_TOKEN = os.environ.get("AGENT_THINK_TOKEN", "<|think|>")
THINK_PREFIX = (THINK_TOKEN + "\n") if REASONING_MODE == 2 else ""
# Recovery line spoken whenever a turn would otherwise end with no
# spoken text — either the ReAct loop exceeded RECURSION_LIMIT, or the
# stream finished after a failed/denied tool without the LLM producing
# any reply. A voice agent must always say *something*.
FALLBACK_TEXT = os.environ.get(
    "AGENT_TOOL_FALLBACK_TEXT",
    "うまくできませんでした、すみません。",
)
# Action tools have a binary, deterministic outcome (the audio played or it
# didn't; the timer was set or it wasn't). gemma4 has been observed to IGNORE
# an [error] ToolMessage and falsely confirm success — e.g. reply 「再生したよ」
# after audio-io was unreachable and the tool returned `[error] … audio was
# NOT played`. For these tools we do NOT trust the model to report failure:
# when one returns [error]/[denied] the chat stream suppresses the model's
# (untrustworthy) reply and speaks a deterministic honest line instead. The
# system-prompt nudge below is defence-in-depth; this dict is the guarantee.
# Info/query tools (run_shell, read_file, web_search, system_health,
# check_timers) are intentionally excluded — there the model's interpretation
# of the result (summarising, retrying, explaining an error) is the point.
TOOL_FAIL_PHRASES: dict[str, str] = {
    "play_audio_file": "ごめん、音声を再生できなかった。",
    "stop_audio": "ごめん、音声を止められなかった。",
    "start_timer": "ごめん、タイマーをかけられなかった。",
    "cancel_timer": "ごめん、タイマーを止められなかった。",
}
ACTION_TOOLS = frozenset(TOOL_FAIL_PHRASES)
# Suffix appended to AGENT_SYSTEM_PROMPT *only when tools are enabled* so we
# don't tell the LLM about tools that aren't wired. The phrasing matches the
# voice-agent persona (タメ口 / 短文 / TTS 向き). Override entirely with
# AGENT_TOOL_SYSTEM_SUFFIX if you want different guidance.
# Empty string (not just unset) also falls back to the default. compose.yaml
# passes the env through with an empty fallback (`${VAR:-}`) so the container
# always sees the var; without `or`, a blank .env would silently disable the
# entire tool-use prompt.
TOOL_SYSTEM_SUFFIX = os.environ.get("AGENT_TOOL_SYSTEM_SUFFIX") or (
    " You have tools available — use them naturally when they help."
    " Tool results are private to you; the user can't see or hear them,"
    " so don't refer to them with deictic words like 'this', 'these',"
    " or 'as written there' — instead, restate the relevant content in"
    " your own words and speak it out. When asked for a list, pick a few"
    " representative items and name them (you don't have to read"
    " everything)."
    # --- Override the 1-2 sentence brevity rule for the tool-using path.
    # The base system prompt is tuned for chit-chat; multi-step requests
    # (find → check → act) need room to run several tool calls in a
    # single turn without speaking between them. Observed failure: model
    # said "じゃあ中身見てみるよ" and ended the turn, leaving the task
    # half-done.
    " The 1-2 sentence limit in the base prompt applies only to your"
    " final spoken reply once the task is done. While you're working,"
    " call as many tools as you need — silently. Do NOT announce intent"
    " (\"I'll check X\", \"まず~してみるよ\") and then stop the turn."
    " If you say you'll do something, the next thing in the same turn"
    " must be the tool call that does it, not the end of your response."
    # --- Step further: don't bounce questions back the user can't answer
    # any better than you can.
    " Before asking the user a clarifying question, check whether you"
    " can answer it yourself with a tool — today's date (use run_shell"
    " `date`), what files exist in a directory (`ls`), the contents of"
    " a file (read_file). Only ask the user about things only they know"
    " (their intent, their preference, ambiguous wording)."
    # --- Pre-action checklist for file/audio paths. Reduces the
    # \"relative path → tool failure\" loop we saw in production.
    " When acting on a file or directory: first establish the absolute"
    " path (combine the share root with the relative path the user"
    " referred to), then confirm the file or directory exists with"
    " `ls`, then perform the action. Don't guess a path and call the"
    " action tool hoping it works."
    # --- Hypothesize-and-retry instead of giving up on the first error.
    " If a tool returns `[error]` or `[denied]`, form one hypothesis"
    " about the cause (wrong path? relative instead of absolute? typo"
    " in filename?) and retry with a corrected input once before"
    " telling the user it failed. Don't loop more than 2-3 times on"
    " the same tool — if that doesn't work, honestly say you couldn't"
    " do it."
    # --- Never fabricate success after a failed action. (Backed by a
    # deterministic guard in the chat stream — see TOOL_FAIL_PHRASES — but
    # state it here too so the model doesn't even try.)
    " NEVER claim an action worked when its tool returned `[error]` or"
    " `[denied]`. Do not say 「再生したよ」「セットしたよ」「止めたよ」 or"
    " similar after a failure — if it failed, say plainly that it didn't"
    " work."
    # --- Tiny CoT nudge. We don't have explicit thinking tokens for
    # Gemma so this is the prompt-level equivalent.
    " For requests that involve multiple steps (find a file, then play"
    " it; look something up, then act on it), briefly plan the steps"
    " in your head before calling the first tool, then execute all"
    " the steps in one turn — only speak to the user once everything"
    " is done (or you've genuinely hit a wall)."
)

# --- Tool-use few-shot ------------------------------------------------------
# gemma4 12B is weak at *deciding to call* a tool (it tends to answer in text
# or fabricate success). The strongest fix is to show it real example turns —
# Human → AI(tool_calls) → Tool(result) → AI(reply) — so it sees the exact
# tool-call shape it should emit, not just an English description. These are
# injected as a fixed prefix AFTER the system message and BEFORE the live
# history (so the prompt-cache prefix stays stable), and are NOT persisted to
# the checkpoint. Controlled by:
#   AGENT_TOOL_FEWSHOT       on/off (default off in code; compose sets on)
#   AGENT_TOOL_FEWSHOT_FILE  optional JSON path overriding the builtin examples
# A turn is {"role": user|assistant|tool, "content": str,
#            "tool_calls": [{"name","args"}]?, "name": str?}. tool_call ids are
# auto-assigned (fs0, fs1, …) and matched to the following tool turn(s) in
# order, so authors never write ids by hand.
_FEWSHOT_DEFAULT: list[dict] = [
    {"role": "user", "content": "3分はかって"},
    {"role": "assistant", "tool_calls": [{"name": "start_timer", "args": {"seconds": 180}}]},
    {"role": "tool", "name": "start_timer", "content": "タイマー#1 を3分でセットしたよ"},
    {"role": "assistant", "content": "3分でセットしたよ。"},

    {"role": "user", "content": "今タイマーいくつ動いてる？"},
    {"role": "assistant", "tool_calls": [{"name": "check_timers", "args": {}}]},
    {"role": "tool", "name": "check_timers", "content": "タイマー#1 残り2分30秒"},
    {"role": "assistant", "content": "1個動いてて、残り2分半くらいだよ。"},

    {"role": "user", "content": "タイマー全部止めて"},
    {"role": "assistant", "tool_calls": [{"name": "cancel_timer", "args": {"which": "全部"}}]},
    {"role": "tool", "name": "cancel_timer", "content": "2件キャンセルした"},
    {"role": "assistant", "content": "全部止めたよ。"},

    {"role": "user", "content": "音止めて"},
    {"role": "assistant", "tool_calls": [{"name": "stop_audio", "args": {}}]},
    {"role": "tool", "name": "stop_audio", "content": "停止した"},
    {"role": "assistant", "content": "止めたよ。"},

    {"role": "user", "content": "今日の天気は？"},
    {"role": "assistant", "tool_calls": [{"name": "web_search", "args": {"query": "今日の天気"}}]},
    {"role": "tool", "name": "web_search", "content": "東京は晴れ、最高22度の見込み。"},
    {"role": "assistant", "content": "東京は晴れで、最高22度くらいみたいだよ。"},
]


def _fewshot_turns_to_messages(turns: list[dict]) -> list:
    """Convert raw few-shot turn dicts into LangChain messages, auto-assigning
    and matching tool_call ids (fs0, fs1, …). Linear examples only: each tool
    turn binds to the oldest still-unmatched tool_call (FIFO)."""
    msgs: list = []
    pending_ids: list[str] = []
    counter = 0
    for t in turns:
        role = t.get("role")
        if role == "user":
            msgs.append(HumanMessage(content=t.get("content", "")))
        elif role == "assistant":
            raw_tcs = t.get("tool_calls") or []
            if raw_tcs:
                lc_tcs = []
                for tc in raw_tcs:
                    tid = f"fs{counter}"
                    counter += 1
                    pending_ids.append(tid)
                    lc_tcs.append({
                        "name": tc["name"],
                        "args": tc.get("args", {}),
                        "id": tid,
                        "type": "tool_call",
                    })
                msgs.append(AIMessage(content=t.get("content", ""), tool_calls=lc_tcs))
            else:
                msgs.append(AIMessage(content=t.get("content", "")))
        elif role == "tool":
            tid = pending_ids.pop(0) if pending_ids else f"fs{counter}"
            msgs.append(ToolMessage(
                content=t.get("content", ""),
                name=t.get("name"),
                tool_call_id=tid,
            ))
        else:
            log.warning("few-shot: skipping turn with unknown role %r", role)
    return msgs


def _build_fewshot() -> list:
    """Build the fixed few-shot message prefix from env. Empty when tools are
    off or AGENT_TOOL_FEWSHOT is not enabled. A bad AGENT_TOOL_FEWSHOT_FILE
    warns and falls back to the builtin examples rather than crashing."""
    if not TOOLS_ENABLED:
        return []
    if os.environ.get("AGENT_TOOL_FEWSHOT", "off").strip().lower() not in (
        "on", "true", "1", "yes"
    ):
        return []
    turns, source = _FEWSHOT_DEFAULT, "builtin"
    path = os.environ.get("AGENT_TOOL_FEWSHOT_FILE", "").strip()
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("top-level JSON must be a list of turn objects")
            turns, source = data, "file"
        except Exception as e:  # noqa: BLE001
            log.warning(
                "few-shot file %r unusable (%s: %s); using builtin examples",
                path, type(e).__name__, e,
            )
    msgs = _fewshot_turns_to_messages(turns)
    log.info(
        "agent: tool few-shot enabled: %d turns -> %d messages (source=%s)",
        len(turns), len(msgs), source,
    )
    return msgs


FEWSHOT_MESSAGES = _build_fewshot()


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


def _approx_tokens(messages: list) -> int:
    """Cheap local token estimate for [`_trim_history`], no `/tokenize` round-trip.

    ~0.5 tokens/char + per-message overhead. This under-counts dense CJK, so
    the budget ([`MAX_HISTORY_TOKENS`]) is deliberately set well under the
    context window to keep a safety margin even when the estimate is low.
    """
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += len(content) // 2 + 8
    return total


def _trim_history(messages: list) -> list:
    """Keep the most recent messages within [`MAX_HISTORY_TOKENS`].

    `start_on="human"` guarantees the kept window begins on a user turn so a
    tool-call AIMessage and its ToolMessage result are never split (an orphan
    ToolMessage would be an invalid prompt). The system prompt is added by the
    caller, on top of this budget, so `include_system=False` here.
    """
    return trim_messages(
        messages,
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter=_approx_tokens,
        strategy="last",
        include_system=False,
        start_on="human",
        allow_partial=False,
    )


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
        messages = _trim_history(list(state["messages"]))
        # THINK_PREFIX (<|think|>) is empty unless AGENT_REASONING=2.
        sys_content = THINK_PREFIX + SYSTEM_PROMPT
        if sys_content:
            messages = [SystemMessage(content=sys_content), *messages]
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
    """ReAct (agent ↔ tools) graph for AGENT_TOOLS_ENABLED=true.

    Hand-rolled instead of `create_react_agent` so we can add one thing the
    prebuilt agent can't: when a *terminal* tool (`TERMINAL_TOOLS`, e.g.
    stop_audio) runs successfully, end the turn WITHOUT a second LLM call.
    Those tools are instant + deterministic, so re-invoking the model just to
    paraphrase "done" adds a whole model-call of latency for nothing; the
    user-facing confirmation comes from the tool's `TOOL_ACK_PHRASES` line
    (spoken/streamed when the tool is invoked). A failed terminal tool, and
    every non-terminal tool, loop back to the LLM as usual.

    The LLM call lives in `agent_node`, which injects the system prompt (not
    persisted in state — so AGENT_SYSTEM_PROMPT changes take effect next turn)
    and trims history to MAX_HISTORY_TOKENS, keeping system + tools first so the
    prompt-cache prefix is stable. `ainvoke` inside the node still streams token
    deltas via stream_mode="messages".
    """
    tools = all_tools()
    system_text = (SYSTEM_PROMPT + TOOL_SYSTEM_SUFFIX) if SYSTEM_PROMPT else TOOL_SYSTEM_SUFFIX.strip()
    # Prepend <|think|> (mode 2 only; empty otherwise). Goes FIRST so it's at
    # the very start of the system prompt, which is where gemma4 expects it.
    system_text = THINK_PREFIX + system_text
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesState):
        messages = _trim_history(list(state["messages"]))
        # Fixed prefix: system prompt, then the tool-use few-shot, then the live
        # (trimmed) history. Few-shot sits BEFORE history and is not persisted,
        # so the cache prefix [system, *few-shot] stays stable across turns.
        prefix: list = []
        if system_text:
            prefix.append(SystemMessage(content=system_text))
        prefix.extend(FEWSHOT_MESSAGES)
        messages = [*prefix, *messages]
        response = await llm_with_tools.ainvoke(messages)
        if LOG_LLM_RAW:
            # Raw output so we can see whether the model emitted a tool_call
            # (e.g. stop_audio) or just text/reasoning. additional_kwargs is
            # where ollama/gemma4 reasoning content lands.
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            log.info(
                "llm output: tool_calls=%s | content=%r | additional_kwargs=%r",
                [(tc.get("name"), tc.get("args")) for tc in (response.tool_calls or [])],
                content[:1000],
                dict(response.additional_kwargs or {}),
            )
        return {"messages": [response]}

    def route_after_tools(state: MessagesState) -> str:
        # Look at the ToolMessages this tools step just appended (the trailing
        # run). If a terminal tool succeeded (_safe_invoke prefixes failures
        # with [error]/[denied]), end via terminal_reply; otherwise loop back.
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                break
            if msg.name in TERMINAL_TOOLS and not str(msg.content).startswith(
                ("[error]", "[denied]")
            ):
                return "terminal_reply"
        return "agent"

    def terminal_reply(state: MessagesState):
        # Fixed assistant message so the persisted history stays a well-formed
        # ReAct exchange. NOT re-streamed (stream_mode="messages" only surfaces
        # LLM output) — the spoken/printed confirmation already came from the
        # tool's ack phrase, so we reuse that same phrase here for consistency.
        name = next(
            (
                m.name
                for m in reversed(state["messages"])
                if isinstance(m, ToolMessage) and m.name in TERMINAL_TOOLS
            ),
            None,
        )
        text = (TOOL_ACK_PHRASES.get(name) if name else None) or "はい。"
        return {"messages": [AIMessage(content=text)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("terminal_reply", terminal_reply)
    builder.add_edge(START, "agent")
    # agent → "tools" when the LLM emitted tool_calls, else → END.
    builder.add_conditional_edges("agent", tools_condition)
    # tools → terminal_reply (end, no 2nd LLM) on a successful terminal tool,
    # else back to agent.
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "terminal_reply": "terminal_reply"},
    )
    builder.add_edge("terminal_reply", END)
    return builder.compile(checkpointer=checkpointer)


async def warm_system_prefix() -> None:
    """Prime ollama's KV cache with the system-prompt prefix the graphs send.

    Best-effort, run as a background task at startup. The model is already
    resident (llm/init.sh warms it on load + first decode), but the very first
    *real* turn still has to prefill the long system prompt — `AGENT_SYSTEM_PROMPT`
    (+ the tool guidance when tools are on, + the bound tool schemas). Sending
    one inference here with the *exact* same effective system text and tool set
    leaves that prefix cached, so the first real turn only evaluates the user
    message. We rebuild the system text the same way the graphs do — legacy =
    `SYSTEM_PROMPT`; react = `SYSTEM_PROMPT + TOOL_SYSTEM_SUFFIX` with `all_tools()`
    bound — so the cached prefix matches token-for-token.

    Never fatal: on any error the first turn just pays the prefill as before.
    The wake-word flow means no real turn arrives before this finishes.
    """
    if TOOLS_ENABLED:
        system_text = (SYSTEM_PROMPT + TOOL_SYSTEM_SUFFIX) if SYSTEM_PROMPT else TOOL_SYSTEM_SUFFIX.strip()
    else:
        system_text = SYSTEM_PROMPT
    # Match the real graph's system prompt exactly (incl. mode-2 <|think|>
    # prefix) so the warmed prompt-cache prefix lines up with live turns.
    system_text = THINK_PREFIX + system_text
    messages = []
    if system_text:
        messages.append(SystemMessage(content=system_text))
    # Same few-shot prefix the react agent_node injects, so the warmed cache
    # prefix [system, *few-shot] matches live turns exactly. Empty unless
    # AGENT_TOOL_FEWSHOT is on (and tools enabled).
    messages.extend(FEWSHOT_MESSAGES)
    messages.append(HumanMessage(content="ウォームアップ"))
    # Dedicated capped client: a couple of tokens fill the prefix KV and warm
    # the decode path without generating a real reply. Bind the same tools so
    # the request — and thus the cached prefix — matches the react path exactly.
    warm_llm = ChatOllama(
        base_url=OLLAMA_BASE_URL, model=MODEL_NAME, temperature=0, num_predict=2,
        reasoning=REASONING,
    )
    target = warm_llm.bind_tools(all_tools()) if TOOLS_ENABLED else warm_llm
    # ollama may still be loading right after this container starts; retry briefly.
    for attempt in range(1, 11):
        try:
            await target.ainvoke(messages)
            log.info(
                "agent: system-prefix warmup done (attempt=%d tools=%s sys_chars=%d)",
                attempt, TOOLS_ENABLED, len(system_text),
            )
            return
        except Exception as exc:  # noqa: BLE001 — warmup is strictly best-effort
            if attempt == 10:
                log.warning("agent: system-prefix warmup gave up: %s", exc)
                return
            await asyncio.sleep(2)


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
    On overflow we yield `FALLBACK_TEXT` as the final spoken
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
    # generating an apology — we emit `FALLBACK_TEXT` at the
    # end so the operator never hears total silence. Observed with
    # gemma4:e4b when allowlist-rejected shell commands feed back.
    real_content_yielded = False
    # Did the *most recent* tool call succeed? `_safe_invoke` prefixes
    # failures with `[error]` / `[denied]`, so anything else means the
    # action went through. For action-only commands ("play this wav",
    # "run this shell") gemma3:4b often produces zero follow-up text
    # after a successful tool — that's expected, not a failure, so we
    # must NOT speak the apology fallback in that case (the user already
    # heard the ack and the action happened). Only emit fallback when
    # the silence really does follow a failure.
    last_tool_succeeded: bool | None = None
    # Name of the most recent ACTION tool (play/stop/timer) that returned
    # [error]/[denied] and hasn't since succeeded on a retry. When still set at
    # end of turn, we speak a deterministic failure line and suppress the
    # model's reply — which for these tools can falsely claim success (gemma4
    # ignoring the [error]). See TOOL_FAIL_PHRASES / ACTION_TOOLS.
    failed_action_tool: str | None = None

    try:
        async for chunk, _meta in graph.astream(
            input_state, config=config, stream_mode="messages"
        ):
            if isinstance(chunk, ToolMessage):
                content = chunk.content if isinstance(chunk.content, str) else ""
                failed = content.startswith(("[error]", "[denied]"))
                last_tool_succeeded = not failed
                # Track action-tool failure so we can override the model's
                # final reply with a deterministic honest line. A later
                # SUCCESS of the *same* tool (retry with a fixed arg) clears
                # it; a different tool succeeding does not mask this failure.
                if chunk.name in ACTION_TOOLS:
                    if failed:
                        failed_action_tool = chunk.name
                    elif chunk.name == failed_action_tool:
                        failed_action_tool = None
                continue
            if not isinstance(chunk, AIMessageChunk):
                # SystemMessage etc — not for TTS.
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
                # If an action tool failed this turn, the model's final text
                # is untrustworthy (it may claim success despite the [error]).
                # Drop it; a deterministic failure line is spoken after the
                # loop instead.
                if failed_action_tool is not None:
                    continue
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
        yield {"message": {"content": FALLBACK_TEXT}, "done": False}
        real_content_yielded = True
        # Recursion fallback already spoke; don't also emit the action line.
        failed_action_tool = None

    if failed_action_tool is not None:
        # Deterministic honest failure for an action tool — bypasses the LLM
        # entirely so a false 「再生したよ」 can never reach the speaker. The
        # model's own reply (if any) was suppressed above.
        fail_line = TOOL_FAIL_PHRASES.get(failed_action_tool) or FALLBACK_TEXT
        log.info(
            "action tool %s failed for session %s; speaking deterministic "
            "failure line (model reply suppressed)",
            failed_action_tool, session_id,
        )
        yield {"message": {"content": fail_line}, "done": False}
        real_content_yielded = True

    if not real_content_yielded and last_tool_succeeded is not True:
        # Stream ended normally but the LLM produced no text — typically
        # after a tool error returned `[denied]` / `[error]` content that
        # the LLM decided not to comment on. Voice agent must always say
        # SOMETHING; emit the fallback so the speaker isn't dead. We skip
        # this when the last tool *succeeded* — for action-only commands
        # the user already heard the ack and the action happened, so
        # apologising would contradict reality.
        log.warning(
            "no LLM text yielded for session %s; emitting fallback",
            session_id,
        )
        yield {"message": {"content": FALLBACK_TEXT}, "done": False}

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
    # write_eof() can itself raise on a client that already went away — a
    # barge-in resets the orchestrator→agent connection mid-stream, the
    # `except ConnectionResetError` above logs that as a normal cancel, but
    # the terminating chunk here would then throw a *second* reset that
    # aiohttp surfaces as an ERROR + traceback. Guard it so an aborted turn
    # stays quiet. (The CancelledError branch re-raises before reaching here.)
    try:
        await resp.write_eof()
    except ConnectionResetError:
        pass
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
        "agent: ollama=%s model=%s db=%s tools=%s recursion_limit=%d "
        "reasoning_mode=%d (native_think=%s think_prefix=%r)",
        OLLAMA_BASE_URL, MODEL_NAME, DB_PATH, TOOLS_ENABLED, RECURSION_LIMIT,
        REASONING_MODE, REASONING is True, THINK_PREFIX,
    )

    # ChatOllama streams tokens from ollama via httpx. temperature=0
    # matches the orchestrator's previous /api/chat config (no temp
    # was sent, ollama's default for streaming is 0.8 — we override
    # to keep voice-agent replies deterministic for the same input).
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=MODEL_NAME,
        temperature=0,
        # reasoning は AGENT_REASONING (mode 0/1/2) 由来。True=native think
        # (mode 1) のみ、mode 0/2 は None で think を送らない。詳細は
        # _reasoning_mode / THINK_PREFIX の定義を参照。
        reasoning=REASONING,
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

    # Prime ollama's KV cache with the system-prompt prefix in the background
    # so the first real turn doesn't re-prefill it. Best-effort; serving has
    # already started, and the wake-word flow means no real turn lands first.
    warmup_task = asyncio.create_task(warm_system_prefix())

    try:
        await asyncio.Event().wait()
    finally:
        warmup_task.cancel()
        await runner.cleanup()
        await conn.close()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
