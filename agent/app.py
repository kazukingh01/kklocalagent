"""The orchestrator's parser only consults `message.content` and `done`
(pipeline.rs:650-696), so the rest of ollama's response envelope is not faked.
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

from tools import TOOL_ACK_PHRASES, all_tools, set_reset_memory_hook

log = logging.getLogger("agent")

OLLAMA_BASE_URL = os.environ.get("AGENT_OLLAMA_URL", "http://llm:11434")
MODEL_NAME = os.environ.get("AGENT_MODEL", "gemma3:4b")
SYSTEM_PROMPT = os.environ.get("AGENT_SYSTEM_PROMPT", "")
DB_PATH = os.environ.get("AGENT_DB_PATH", "/data/agent.sqlite")
SESSION_IDLE_SEC = float(os.environ.get("AGENT_SESSION_IDLE_SEC", "600"))
PORT = int(os.environ.get("AGENT_PORT", "7080"))

TOOLS_ENABLED = os.environ.get("AGENT_TOOLS_ENABLED", "false").lower() in ("1", "true", "yes")
RECURSION_LIMIT = int(os.environ.get("AGENT_TOOL_RECURSION_LIMIT", "6"))
# Without trimming, a long tool-heavy conversation grows the checkpointed
# history until it fills ollama's context window — the reply truncates to
# empty and the agent gets stuck emitting its fallback line every turn.
# Kept well under OLLAMA_CONTEXT_LENGTH (default 16384) since the system
# prompt + tool schemas + generation room are on TOP of this budget.
MAX_HISTORY_TOKENS = int(os.environ.get("AGENT_MAX_HISTORY_TOKENS", "4096"))
TERMINAL_TOOLS = {"stop_audio", "reset_memory"}
LOG_LLM_RAW = os.environ.get("AGENT_LOG_LLM_RAW", "true").lower() in ("1", "true", "yes")
# AGENT_REASONING — 3-way thinking mode (the two mechanisms are NOT
# interchangeable across models):
#   0 = off          send think=FALSE explicitly (omitting it lets a
#                    thinking-capable backend default to ON). Safe everywhere.
#   1 = native       ollama think=true (ChatOllama reasoning=True). Thinking
#                    lands in additional_kwargs, never reaches TTS. Only on
#                    models advertising native thinking (e.g. gemma4:e4b);
#                    an imported HF GGUF without it returns HTTP 400.
#   2 = prompt-token prepend AGENT_THINK_TOKEN (<|think|>) to the system
#                    prompt, with think=false. For GGUFs that reject native
#                    think. Reasoning may surface inline in `content`; the
#                    orchestrator strips <...> spans before TTS.
# CAVEAT (modes 1 & 2): thinking has been observed to suppress tool calls on
# gemma4 12B. Use 0 if tool-calling regresses.
def _reasoning_mode() -> int:
    v = os.environ.get("AGENT_REASONING", "0").strip().lower()
    if v in ("0", "off", "false", "no", "none"):
        return 0
    if v == "2":
        return 2
    return 1


REASONING_MODE = _reasoning_mode()
# Modes 0/2 send think=FALSE, NOT None: None omits the `think` field and a
# thinking-capable backend then defaults to thinking ON — AGENT_REASONING=0
# would still reason. think=false is safe even on a GGUF that can't think
# (the "does not support thinking" 400 only fires on think=TRUE).
REASONING = True if REASONING_MODE == 1 else False
THINK_TOKEN = os.environ.get("AGENT_THINK_TOKEN", "<|think|>")
THINK_PREFIX = (THINK_TOKEN + "\n") if REASONING_MODE == 2 else ""
FALLBACK_TEXT = os.environ.get(
    "AGENT_TOOL_FALLBACK_TEXT",
    "うまくできませんでした、すみません。",
)
# gemma4 has been observed to IGNORE an [error] ToolMessage and falsely
# confirm success (e.g. 「再生したよ」 after audio-io was unreachable). For
# these action tools the chat stream suppresses the model's reply on
# [error]/[denied] and speaks a deterministic honest line instead.
# Info/query tools are intentionally excluded — there the model's
# interpretation of the result is the point.
TOOL_FAIL_PHRASES: dict[str, str] = {
    "play_audio_file": "ごめん、音声を再生できなかった。",
    "stop_audio": "ごめん、音声を止められなかった。",
    "start_timer": "ごめん、タイマーをかけられなかった。",
    "cancel_timer": "ごめん、タイマーを止められなかった。",
}
ACTION_TOOLS = frozenset(TOOL_FAIL_PHRASES)
# `or` (not a get() default): compose.yaml passes the env with an empty
# fallback (`${VAR:-}`), so a blank .env would otherwise silently disable
# the entire tool-use prompt.
TOOL_SYSTEM_SUFFIX = os.environ.get("AGENT_TOOL_SYSTEM_SUFFIX") or (
    " You have tools available — use them naturally when they help."
    " For any action request (resetting memory, timers, playing/stopping"
    " audio, telling the time, web search, file operations, etc.), you must"
    " call the corresponding tool to actually perform it. Never report"
    " completion or acknowledgement — saying things like 'Done' or 'I'll do"
    " that now' — without calling the tool. Only report the result after you"
    " have received the tool's output."
    " Tool results are private to you; the user can't see or hear them,"
    " so don't refer to them with deictic words like 'this', 'these',"
    " or 'as written there' — instead, restate the relevant content in"
    " your own words and speak it out. When asked for a list, pick a few"
    " representative items and name them (you don't have to read"
    " everything)."
    # Observed failure: model said "じゃあ中身見てみるよ" and ended the
    # turn, leaving the task half-done.
    " The 1-2 sentence limit in the base prompt applies only to your"
    " final spoken reply once the task is done. While you're working,"
    " call as many tools as you need — silently. Do NOT announce intent"
    " (\"I'll check X\", \"まず~してみるよ\") and then stop the turn."
    " If you say you'll do something, the next thing in the same turn"
    " must be the tool call that does it, not the end of your response."
    " Before asking the user a clarifying question, check whether you"
    " can answer it yourself with a tool — today's date (use run_shell"
    " `date`), what files exist in a directory (`ls`), the contents of"
    " a file (read_file). Only ask the user about things only they know"
    " (their intent, their preference, ambiguous wording)."
    # Reduces the "relative path → tool failure" loop seen in production.
    " When acting on a file or directory: first establish the absolute"
    " path (combine the share root with the relative path the user"
    " referred to), then confirm the file or directory exists with"
    " `ls`, then perform the action. Don't guess a path and call the"
    " action tool hoping it works."
    " If a tool returns `[error]` or `[denied]`, form one hypothesis"
    " about the cause (wrong path? relative instead of absolute? typo"
    " in filename?) and retry with a corrected input once before"
    " telling the user it failed. Don't loop more than 2-3 times on"
    " the same tool — if that doesn't work, honestly say you couldn't"
    " do it."
    " NEVER claim an action worked when its tool returned `[error]` or"
    " `[denied]`. Do not say 「再生したよ」「セットしたよ」「止めたよ」 or"
    " similar after a failure — if it failed, say plainly that it didn't"
    " work."
    " For requests that involve multiple steps (find a file, then play"
    " it; look something up, then act on it), briefly plan the steps"
    " in your head before calling the first tool, then execute all"
    " the steps in one turn — only speak to the user once everything"
    " is done (or you've genuinely hit a wall)."
)

# AGENT_TOOL_FEWSHOT defaults off: the fake-history examples were observed
# to be read as conversation facts and answer-copied instead of triggering
# the tool call.
_FEWSHOT_DEFAULT: list[dict] = [
    {"role": "user", "content": "タイマー3分"},
    {"role": "assistant", "tool_calls": [{"name": "start_timer", "args": {"seconds": 180}}]},
    {"role": "tool", "name": "start_timer", "content": "タイマー#1 を3分でセットしたよ"},

    {"role": "user", "content": "今タイマーいくつ動いてる？"},
    {"role": "assistant", "tool_calls": [{"name": "check_timers", "args": {}}]},
    {"role": "tool", "name": "check_timers", "content": "タイマー#1 残り2分30秒"},
    {"role": "assistant", "content": "1個動いてて、残り2分半くらいだよ。"},

    {"role": "user", "content": "タイマー全部止めて"},
    {"role": "assistant", "tool_calls": [{"name": "cancel_timer", "args": {"which": "全部"}}]},
    {"role": "tool", "name": "cancel_timer", "content": "2件キャンセルした"},
    {"role": "assistant", "content": "全部止めたよ。"},

    {"role": "user", "content": "音声止めて"},
    {"role": "assistant", "tool_calls": [{"name": "stop_audio", "args": {}}]},
    {"role": "tool", "name": "stop_audio", "content": "停止した"},

    {"role": "user", "content": "今日の天気は？"},
    {"role": "assistant", "tool_calls": [{"name": "web_search", "args": {"query": "今日の天気"}}]},
    {"role": "tool", "name": "web_search", "content": "AAは晴れ、最高BB度の見込み。"},
    {"role": "assistant", "content": "AAは晴れで、最高BB度くらいみたいだよ。"},

    {"role": "user", "content": "今日って何日だっけ？"},
    {"role": "assistant", "tool_calls": [{"name": "run_shell", "args": {"command": "date"}}]},
    {"role": "tool", "name": "run_shell", "content": "AA BB  X HH:MM:SS JST YYYY"},
    {"role": "assistant", "content": "今日X月X日だよ。"},

    {"role": "user", "content": "今何時？"},
    {"role": "assistant", "tool_calls": [{"name": "run_shell", "args": {"command": "date"}}]},
    {"role": "tool", "name": "run_shell", "content": "AA BB  X HH:MM:SS JST YYYY"},
    {"role": "assistant", "content": "時刻はHH時MM分だよ。"},

    {"role": "user", "content": "news フォルダの中に何がある？"},
    {"role": "assistant", "tool_calls": [{"name": "run_shell", "args": {"command": "ls /workspace/share/news"}}]},
    {"role": "tool", "name": "run_shell", "content": "20260607_morning.wav"},
    {"role": "assistant", "content": "音声ファイルが１つ入ってるよ"},

    {"role": "user", "content": "shareのメモ読んで"},
    {"role": "assistant", "tool_calls": [{"name": "run_shell", "args": {"command": "ls /workspace/share"}}]},
    {"role": "tool", "name": "run_shell", "content": "memo.txt\nnews/"},
    {"role": "assistant", "tool_calls": [{"name": "read_file", "args": {"path": "/workspace/share/xxxxxx.txt"}}]},
    {"role": "tool", "name": "read_file", "content": "XXXXXXXXXXXXXXXXXX"},
    {"role": "assistant", "content": "メモには、XXXXXXXXXXXXXXXXXXって書いてあるよ。"},

    {"role": "user", "content": "〇〇の音声再生して"},
    {"role": "assistant", "tool_calls": [{"name": "play_audio_file", "args": {"paths": ["/workspace/share/news/xxxxxxxxxxxx.wav"]}}]},
    {"role": "tool", "name": "play_audio_file", "content": "再生を開始した"},
    {"role": "assistant", "content": "音声を流すね。"},

    {"role": "user", "content": "〇〇の音声全部再生して"},
    {"role": "assistant", "tool_calls": [{"name": "play_audio_file", "args": {"paths": ["/workspace/share/news/*.wav"]}}]},
    {"role": "tool", "name": "play_audio_file", "content": "再生を開始した"},
    {"role": "assistant", "content": "音声を流すね。"},

    {"role": "user", "content": "システムチェックして"},
    {"role": "assistant", "tool_calls": [{"name": "system_health", "args": {}}]},
    {"role": "tool", "name": "system_health", "content": "agent OK / llm OK / tts OK / audio-io 応答なし"},
    {"role": "assistant", "content": "audio-io が応答してないみたい。それ以外は正常だよ"},

    {"role": "user", "content": "記憶リセットして"},
    {"role": "assistant", "tool_calls": [{"name": "reset_memory", "args": {}}]},
    {"role": "tool", "name": "reset_memory", "content": "会話の記憶をリセットした"},
    {"role": "assistant", "content": "記憶をリセットしたよ。"},
]


def _fewshot_turns_to_messages(turns: list[dict]) -> list:
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


def _fmt_msgs_for_log(messages: list) -> str:
    out: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            out.append(f"Tool[{m.name}]={c[:300]!r}")
        elif isinstance(m, AIMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            tcs = [(tc.get("name"), tc.get("args")) for tc in (m.tool_calls or [])]
            out.append(f"AI(content={c[:200]!r}, tool_calls={tcs})")
        elif isinstance(m, HumanMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            out.append(f"Human={c[:200]!r}")
        elif isinstance(m, SystemMessage):
            out.append(f"System(chars={len(str(m.content))})")
        else:
            out.append(f"{type(m).__name__}={str(getattr(m, 'content', ''))[:120]!r}")
    return " | ".join(out)


SHARE_PRIMER_ENABLED = os.environ.get("AGENT_CONTEXT_SHARE", "off").strip().lower() in (
    "on", "true", "1", "yes"
)
SHARE_PRIMER_DIR = os.environ.get("AGENT_CONTEXT_SHARE_DIR", "/workspace/share")
SHARE_PRIMER_REFRESH_SEC = float(os.environ.get("AGENT_CONTEXT_SHARE_REFRESH_SEC", "300"))
SHARE_PRIMER_MAX_DIRS = int(os.environ.get("AGENT_CONTEXT_SHARE_MAX_DIRS", "20"))


def _build_share_primer() -> str:
    if not SHARE_PRIMER_ENABLED:
        return ""
    root = os.path.abspath(SHARE_PRIMER_DIR)
    if not os.path.isdir(root):
        return ""
    entries: list[tuple[str, int, float, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        try:
            mtime = os.stat(dirpath).st_mtime
        except OSError:
            mtime = 0.0
        counts: dict[str, int] = {}
        for f in filenames:
            ext = os.path.splitext(f)[1].lower().lstrip(".") or "noext"
            counts[ext] = counts.get(ext, 0) + 1
        if counts:
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            summary = ", ".join(f"{e}×{c}" for e, c in items[:8])
            if len(items) > 8:
                summary += f", +{len(items) - 8} more"
        else:
            summary = "—"
        entries.append((rel, depth, mtime, summary))
    if not entries:
        return ""
    # Only leaf dirs consume the cap; ancestors are shown for free. Otherwise
    # a parent's mtime (bumped whenever a child is added) would crowd out the
    # very leaves we want.
    rels = {e[0] for e in entries}

    def _has_subdir(rel: str) -> bool:
        if rel == ".":
            return any(o != "." for o in rels)
        prefix = rel + os.sep
        return any(o != rel and o.startswith(prefix) for o in rels)

    leaves = [e for e in entries if not _has_subdir(e[0])]
    truncated = len(leaves) > SHARE_PRIMER_MAX_DIRS
    if truncated:
        newest = sorted(leaves, key=lambda e: e[2], reverse=True)[:SHARE_PRIMER_MAX_DIRS]
        keep = {e[0] for e in newest}
        for rel in list(keep):
            parts = rel.split(os.sep)
            for i in range(1, len(parts)):
                keep.add(os.sep.join(parts[:i]))
        keep.add(".")
        entries = [e for e in entries if e[0] in keep]
    lines: list[str] = []
    for rel, depth, _mtime, summary in entries:
        name = (os.path.basename(root) or "share") if rel == "." else os.path.basename(rel)
        lines.append(f"{'  ' * depth}{name}/  {summary}")
    if truncated:
        lines.append(
            f"… (newest {SHARE_PRIMER_MAX_DIRS} of {len(leaves)} content dirs by mtime; older omitted)"
        )
    body = "\n".join(lines)
    return (
        f"\n\n## 共有フォルダ ({root}) の構成"
        "（ディレクトリのみ。各 dir は直下ファイルを「拡張子×件数」で要約）\n"
        f"{body}\n"
        "これは現在の中身の参考。実際に再生/読み込みする前に、必要なら "
        "ls / read_file で正確なファイル名・パスを確認すること。"
    )


SHARE_PRIMER = _build_share_primer()


async def _refresh_share_primer_loop() -> None:
    global SHARE_PRIMER
    while True:
        await asyncio.sleep(SHARE_PRIMER_REFRESH_SEC)
        try:
            new = await asyncio.to_thread(_build_share_primer)
        except Exception as e:  # noqa: BLE001
            log.warning("share primer refresh failed: %s", e)
            continue
        if new != SHARE_PRIMER:
            SHARE_PRIMER = new
            log.info("share primer refreshed (%d chars)", len(new))


def _compose_system_text() -> str:
    if TOOLS_ENABLED:
        base = (SYSTEM_PROMPT + TOOL_SYSTEM_SUFFIX) if SYSTEM_PROMPT else TOOL_SYSTEM_SUFFIX.strip()
    else:
        base = SYSTEM_PROMPT
    return THINK_PREFIX + base


def _log_full_prompt_preview() -> None:
    if TOOLS_ENABLED:
        sys_content = _compose_system_text() + SHARE_PRIMER
        fewshot = FEWSHOT_MESSAGES
    else:
        sys_content = _compose_system_text()
        fewshot = []
    parts = [
        "================= LLM PROMPT PREVIEW (startup) =================",
        f"[settings] tools={TOOLS_ENABLED} reasoning_mode={REASONING_MODE} "
        f"think_prefix={THINK_PREFIX!r} context_share={SHARE_PRIMER_ENABLED} "
        f"few_shot={'on' if fewshot else 'off'}({len(fewshot)} msgs)",
        "----------------------- SystemMessage -------------------------",
        sys_content if sys_content else "(empty)",
    ]
    if TOOLS_ENABLED:
        tools = all_tools()
        parts.append(
            f"------------- tools ({len(tools)}) → native `tools` field (bind_tools) -------------"
        )
        for t in tools:
            try:
                args = ", ".join(
                    f"{k}:{(v or {}).get('type', '?')}" for k, v in (t.args or {}).items()
                )
            except Exception:  # noqa: BLE001
                args = "?"
            parts.append(f"• {t.name}({args})")
            parts.append(f"    {t.description}")
    if fewshot:
        parts.append(f"----------------- few-shot ({len(fewshot)} messages) -----------------")
        for m in fewshot:
            if isinstance(m, ToolMessage):
                parts.append(f"[tool:{m.name}] {m.content}")
            elif isinstance(m, AIMessage):
                tcs = [(tc.get("name"), tc.get("args")) for tc in (m.tool_calls or [])]
                parts.append(f"[assistant] content={m.content!r} tool_calls={tcs}")
            elif isinstance(m, HumanMessage):
                parts.append(f"[user] {m.content}")
            else:
                parts.append(f"[{type(m).__name__}] {getattr(m, 'content', '')}")
    parts.append("-------- (then at request time: live history + current user) --------")
    parts.append("===============================================================")
    log.info("LLM prompt preview:\n%s", "\n".join(parts))


class SessionManager:
    """The orchestrator never sends a session_id (ollama's /api/chat body has
    no field for it); one operator per agent instance."""

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

    async def reset(self) -> str:
        """The current turn keeps running on its already-claimed id, so the
        "reset" request itself stays on the old, now-abandoned thread."""
        async with self.lock:
            old = self.current_session
            self.current_session = uuid.uuid4().hex
            self.last_active = time.monotonic()
            log.info("session reset (manual): %s -> %s", old, self.current_session)
            return self.current_session


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3000 <= o <= 0x9FFF     # CJK punct/symbols, hiragana, katakana, kanji (+Ext A)
        or 0xF900 <= o <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF00 <= o <= 0xFFEF  # full-width forms
    )


def _approx_tokens(messages: list) -> int:
    """The earlier flat 0.5 tok/char UNDER-counted Japanese ~2x: history the
    estimate thought fit MAX_HISTORY_TOKENS was really ~double that, so the
    system prompt + tool schemas (sent on TOP of this budget) overflowed
    OLLAMA_CONTEXT_LENGTH and ollama silently truncated them from the front —
    which broke tool calling.
    """
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        cjk = sum(_is_cjk(ch) for ch in content)
        # CJK ~1 tok/char; the rest (Latin/digits/punct) ~1/3 tok/char.
        total += cjk + (len(content) - cjk) // 3 + 8
    return total


def _trim_history(messages: list) -> list:
    """`start_on="human"` so a tool-call AIMessage and its ToolMessage result
    are never split (an orphan ToolMessage would be an invalid prompt)."""
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
    async def chat_node(state: MessagesState):
        messages = _trim_history(list(state["messages"]))
        sys_content = THINK_PREFIX + SYSTEM_PROMPT
        if sys_content:
            messages = [SystemMessage(content=sys_content), *messages]
        # ainvoke (not astream): LangGraph's stream_mode="messages" surfaces
        # token-level chunks from the underlying ChatOllama anyway.
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


def build_react_graph(llm: ChatOllama, checkpointer: AsyncSqliteSaver):
    """Hand-rolled instead of `create_react_agent` for one thing the prebuilt
    agent can't do: a successful *terminal* tool (TERMINAL_TOOLS) ends the
    turn WITHOUT a second LLM call — the confirmation comes from the tool's
    TOOL_ACK_PHRASES line."""
    tools = all_tools()
    system_text = _compose_system_text()
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesState):
        history = _trim_history(list(state["messages"]))
        if FEWSHOT_MESSAGES:
            last_user = next(
                (i for i in range(len(history) - 1, -1, -1)
                 if isinstance(history[i], HumanMessage)),
                None,
            )
            if last_user is None:
                body = [*FEWSHOT_MESSAGES, *history]
            else:
                body = [*history[:last_user], *FEWSHOT_MESSAGES, *history[last_user:]]
        else:
            body = list(history)
        sys_content = system_text + SHARE_PRIMER
        messages = ([SystemMessage(content=sys_content)] if sys_content else []) + body
        if LOG_LLM_RAW:
            log.info(
                "llm input: [system + %d few-shot before last user] + %s",
                len(FEWSHOT_MESSAGES),
                _fmt_msgs_for_log(history),
            )
        response = await llm_with_tools.ainvoke(messages)
        if LOG_LLM_RAW:
            # NOTE: the model's *raw* text (tool-call tokens before ollama
            # parses them) is consumed server-side and NOT returned over
            # /api/chat — to see it, enable the LLM server's verbose log.
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            log.info(
                "llm output: tool_calls=%s | content=%r | additional_kwargs=%r | response_metadata=%r",
                [(tc.get("name"), tc.get("args")) for tc in (response.tool_calls or [])],
                content[:1000],
                dict(response.additional_kwargs or {}),
                dict(getattr(response, "response_metadata", {}) or {}),
            )
        return {"messages": [response]}

    def route_after_tools(state: MessagesState) -> str:
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                break
            if msg.name in TERMINAL_TOOLS and not str(msg.content).startswith(
                ("[error]", "[denied]")
            ):
                return "terminal_reply"
        return "agent"

    def terminal_reply(state: MessagesState):
        # Fixed AIMessage so the persisted history stays a well-formed ReAct
        # exchange. NOT re-streamed (stream_mode="messages" only surfaces LLM
        # output) — the spoken confirmation already came from the ack phrase.
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
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "terminal_reply": "terminal_reply"},
    )
    builder.add_edge("terminal_reply", END)
    return builder.compile(checkpointer=checkpointer)


async def warm_system_prefix() -> None:
    """The message assembly must match the real graphs token-for-token (same
    system text, few-shot position, bound tools) or the cached prefix won't
    line up with the first live turn."""
    system_text = _compose_system_text() + SHARE_PRIMER
    messages = []
    if system_text:
        messages.append(SystemMessage(content=system_text))
    messages.extend(FEWSHOT_MESSAGES)
    messages.append(HumanMessage(content="ウォームアップ"))
    warm_llm = ChatOllama(
        base_url=OLLAMA_BASE_URL, model=MODEL_NAME, temperature=0, num_predict=2,
        reasoning=REASONING,
    )
    target = warm_llm.bind_tools(all_tools()) if TOOLS_ENABLED else warm_llm
    for attempt in range(1, 11):
        try:
            await target.ainvoke(messages)
            log.info(
                "agent: system-prefix warmup done (attempt=%d tools=%s sys_chars=%d)",
                attempt, TOOLS_ENABLED, len(system_text),
            )
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == 10:
                log.warning("agent: system-prefix warmup gave up: %s", exc)
                return
            await asyncio.sleep(2)


def extract_user_text(body: dict) -> str:
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
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    input_state = {"messages": [HumanMessage(content=user_text)]}

    last_acked_tool: str | None = None

    # Real LLM text yielded (acks don't count)? If still False at end of turn
    # we emit FALLBACK_TEXT so the operator never hears total silence —
    # observed with gemma4:e4b when allowlist-rejected shell commands feed
    # back and the model silently stops.
    real_content_yielded = False
    # gemma3:4b often produces zero follow-up text after a SUCCESSFUL action
    # tool — expected, not a failure, so the apology fallback must not fire
    # then (the user already heard the ack and the action happened).
    last_tool_succeeded: bool | None = None
    failed_action_tool: str | None = None

    try:
        async for chunk, _meta in graph.astream(
            input_state, config=config, stream_mode="messages"
        ):
            if isinstance(chunk, ToolMessage):
                content = chunk.content if isinstance(chunk.content, str) else ""
                failed = content.startswith(("[error]", "[denied]"))
                last_tool_succeeded = not failed
                if chunk.name in ACTION_TOOLS:
                    if failed:
                        failed_action_tool = chunk.name
                    elif chunk.name == failed_action_tool:
                        failed_action_tool = None
                continue
            if not isinstance(chunk, AIMessageChunk):
                continue

            # Ollama streams the tool name token-by-token, so early chunks may
            # carry a PARTIAL name ("get_") — those simply miss the
            # TOOL_ACK_PHRASES lookup; only a complete known name acks.
            if chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    name = tc.get("name")
                    if name and name != last_acked_tool:
                        ack = TOOL_ACK_PHRASES.get(name)
                        if ack:
                            yield {"message": {"content": ack}, "done": False}
                        last_acked_tool = name
                continue

            if isinstance(chunk.content, str) and chunk.content:
                if failed_action_tool is not None:
                    continue
                real_content_yielded = True
                yield {"message": {"content": chunk.content}, "done": False}
    except GraphRecursionError:
        log.warning(
            "recursion limit %d reached for session %s; emitting fallback",
            RECURSION_LIMIT, session_id,
        )
        yield {"message": {"content": FALLBACK_TEXT}, "done": False}
        real_content_yielded = True
        failed_action_tool = None

    if failed_action_tool is not None:
        fail_line = TOOL_FAIL_PHRASES.get(failed_action_tool) or FALLBACK_TEXT
        log.info(
            "action tool %s failed for session %s; speaking deterministic "
            "failure line (model reply suppressed)",
            failed_action_tool, session_id,
        )
        yield {"message": {"content": fail_line}, "done": False}
        real_content_yielded = True

    if not real_content_yielded and last_tool_succeeded is not True:
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

    # Barge-in cancellation chain: orchestrator's JoinHandle::abort drops its
    # reqwest response → our next aiohttp write raises → the generator's
    # `async for` propagates the cancel → the LangGraph astream is dropped,
    # closing ChatOllama's httpx connection and stopping generation upstream.
    # asyncio-native tools get the same CancelledError, so no explicit
    # cancellation plumbing is needed.
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
        log.info("chat stream cancelled: %s", type(e).__name__)
    except asyncio.CancelledError:
        log.info("chat stream cancelled: CancelledError")
        raise
    except Exception as e:  # noqa: BLE001
        log.error("chat stream failed: %s", e)
    # write_eof() can throw a SECOND reset after a barge-in, which aiohttp
    # surfaces as an ERROR + traceback. Guard it so an aborted turn stays quiet.
    try:
        await resp.write_eof()
    except ConnectionResetError:
        pass
    return resp


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def session_handler(request: web.Request) -> web.Response:
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
    if LOG_LLM_RAW:
        _log_full_prompt_preview()

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=MODEL_NAME,
        temperature=0,
        reasoning=REASONING,
    )

    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = await aiosqlite.connect(DB_PATH)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()

    graph = build_react_graph(llm, saver) if TOOLS_ENABLED else build_legacy_graph(llm, saver)
    sessions = SessionManager(SESSION_IDLE_SEC)
    set_reset_memory_hook(sessions.reset)

    app = web.Application()
    app["graph"] = graph
    app["sessions"] = sessions
    app.router.add_get("/health", health_handler)
    app.router.add_get("/session", session_handler)
    app.router.add_post("/api/chat", chat_handler)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info("agent listening on :%d", PORT)

    warmup_task = asyncio.create_task(warm_system_prefix())
    primer_task = (
        asyncio.create_task(_refresh_share_primer_loop())
        if SHARE_PRIMER_ENABLED
        else None
    )
    if SHARE_PRIMER_ENABLED:
        log.info(
            "share primer enabled: dir=%s refresh=%.0fs (initial %d chars)",
            SHARE_PRIMER_DIR, SHARE_PRIMER_REFRESH_SEC, len(SHARE_PRIMER),
        )

    try:
        await asyncio.Event().wait()
    finally:
        warmup_task.cancel()
        if primer_task is not None:
            primer_task.cancel()
        await runner.cleanup()
        await conn.close()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
