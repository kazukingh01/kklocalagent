# agent

LangGraph-backed chat shim that fronts ollama with an ollama-compatible
`POST /api/chat` API. The orchestrator keeps talking the same wire
format it always did, but conversation memory, system prompt, and
(future) tool calls / MCP / approval live here instead of in the Rust
pipeline.

## Why an extra hop?

The Rust orchestrator was POSTing directly to ollama and re-sending
the whole prompt every turn — no chat history, no tool semantics, no
permission gating. Issue #10 wires a Python LangGraph layer so:

* conversation memory persists across turns (SQLite checkpointer)
* the system prompt becomes a single env var on this service
  (changing it doesn't require rebuilding the Rust container)
* tool calls / MCP / approval can be added node-by-node in the
  graph without changing the orchestrator at all (the agent stays
  ollama-compatible on the wire)

## API

`POST /api/chat` — ollama-compatible. Body:

```json
{"model": "...", "messages": [{"role":"user","content":"..."}], "stream": true}
```

Response: ndjson stream:

```
{"message":{"content":"<delta>"},"done":false}
...
{"done":true}
```

The orchestrator's parser only consults `message.content` and `done`,
so the agent doesn't need to fake the rest of ollama's surface.

`GET /health` — liveness, returns `{"ok":true}`.

`GET /session` — diagnostic. Returns the current session id, seconds
since the last /api/chat, and the configured idle-rotation window.
Use to verify rotation is working without grepping logs:

```
curl http://agent:7080/session
{"session_id":"abc...","idle_sec":42.1,"rotate_after_sec":600}
```

## Session model

There's exactly one logical operator per agent process (one mic, one
operator). The agent generates a single `session_id` (= LangGraph
`thread_id`) at startup and rotates it after `AGENT_SESSION_IDLE_SEC`
of no /api/chat traffic. A rotation = the next turn starts with empty
conversation memory in the checkpointer.

`thread_id` is the only correlation handle into the SQLite DB, so
operationally:

* "operator walks away for 10 min, comes back" → fresh thread.
* "operator runs straight through 5 turns over 30 s" → same thread.

## Environment

| Variable | Default | What it does |
|---|---|---|
| `AGENT_OLLAMA_URL` | `http://llm:11434` | base URL of the ollama service |
| `AGENT_MODEL` | `gemma3:4b` | model name passed to ChatOllama |
| `AGENT_SYSTEM_PROMPT` | `""` | prepended at LLM invoke time on every turn (not persisted in state) |
| `AGENT_DB_PATH` | `/data/agent.sqlite` | SQLite checkpoint DB path |
| `AGENT_SESSION_IDLE_SEC` | `600` | rotate session after this many idle seconds |
| `AGENT_PORT` | `7080` | bind port |

## v1 scope

* one chat node, no tools, no MCP, no approval
* SQLite checkpointer for conversation memory
* env-driven system prompt (injected at invoke time, not persisted)
* idle-rotated session id

Tools / MCP / approval are explicit follow-ups in issue #10.
