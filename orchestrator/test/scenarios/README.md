# test/scenarios/ — orchestrator control-flow scenarios

Offline scenario suite for the orchestrator's v1.0 wake-gated state
machine. Boots three orchestrator instances (`strict`, `no-barge`,
`loose`) wired to a single set of in-process mock backends, then
drives each through a known sequence of `/events` POSTs and asserts
which downstream HTTP endpoints fire (and how many times).

Complements the `state::tests` Rust unit tests: those check the
state machine's logic directly; this checks the *HTTP plumbing* —
that the gate decisions actually translate into POSTs (or absences
of POSTs) on the right backends. No real models, no GPU, no
network fixtures — just shape-and-count assertions.

```text
              ┌──────────────────────────────────────┐
              │           scenarios runner           │
              │  (mock ASR/LLM/TTS/sink + driver)    │
              └──────────────────────────────────────┘
                  │   ▲ POSTs           ▼ POSTs
                  │   │                 │
                  │   │ /events         │ /inference, /api/chat,
                  │   │                 │ /speak, /stop, /sink
                  ▼   │                 │
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ orch-strict│  │orch-no-barg│  │ orch-loose │
   └────────────┘  └────────────┘  └────────────┘
```

## What each scenario asserts

| # | orch    | event sequence                              | expected backend hits                |
|---|---------|---------------------------------------------|--------------------------------------|
| 1 | strict  | SE                                          | 0 ASR, 0 LLM, 0 TTS, 0 sink          |
| 2 | strict  | Wake → SE                                   | 1 ASR, 1 LLM, 1 /speak, sink WW + Turn |
| 3 | strict  | Wake → SE → SE                              | 1 ASR, 1 LLM, 1 /speak (2nd SE drop) |
| 4 | strict  | Wake → sleep 3 s → SE                       | 0 ASR (window expired)               |
| 5 | strict  | Wake × 3                                    | 3 sink WW, 0 ASR/LLM/TTS             |
| 6 | strict  | Wake → SE → (mid /speak) Wake               | 1 ASR, 1 LLM, 1 /speak, **1 /stop**  |
| 7 | no-barge| Wake → SE → (mid /speak) Wake               | 1 ASR, 1 LLM, 1 /speak, **0 /stop**  |
| 8 | loose   | SE (no wake)                                | 1 ASR, 1 LLM, 1 /speak               |
| 9 | strict  | Wake → SE without audio_base64 → real SE    | 0+1 ASR, 0+1 LLM (1st skipped pre-gate, window remains armed) |

Scenario 6 deterministically drives the barge-in path by holding the
mock TTS `/speak` open until `/stop` releases it — a stand-in for
the real streamer's `asyncio.Task.cancel()` flow. Without the
holding, `/speak` would return instantly and barge-in would race the
turn's natural completion.

## Run

```bash
cd orchestrator/test/scenarios
docker compose up --build --abort-on-container-exit --exit-code-from scenarios
```

Cold first run: ~1 min for the orchestrator Rust build (cached from
the integration smoke if you've run that). Subsequent runs: ~5 s for
the actual scenarios — the orchs only need to boot once.

Expected output ends with:

```text
============================================================
PASSED 9 / 9
```

Exit 0 on green; exit 1 on the first failed scenario.

## Tear down

```bash
docker compose down
```

No volumes — fully ephemeral.

## When to add a scenario

Anything you'd otherwise verify with `tail -f` and `grep` in
production logs: "did orchestrator drop X under Y conditions?". Add
a scenario rather than a manual test, and the gate stays green
across refactors. Specifically, add one when:

- A new event `name` is introduced (covers dispatch routing).
- A new config flag changes the gate decision (covers env wiring).
- A bug surfaces in production ("orchestrator called LLM when it
  shouldn't have") — write the scenario *first*, see it fail, then
  fix the orchestrator.
