# test/ — orchestrator smoke test

Mock-backed smoke. Validates that the orchestrator correctly plumbs
`POST /events` → ASR `/inference` → LLM `/api/chat`, without pulling
any real ML model.

```text
probe (mock ASR :9100 + mock LLM :9200 + test driver)
    │
    └─POST /events (SpeechEnded + base64 PCM)─► orch :7000
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
       POST /inference (mock ASR)                                        POST /api/chat (mock LLM)
       → {"text":"こんにちは"}                                             → {"message":{"content":"OK, 了解しました。"}}
                 │                                                                 │
                 └─────────────────────► probe asserts both mocks hit IN ORDER ◄───┘
```

The probe also asserts the mock LLM saw the exact string the mock ASR
returned — proving the orchestrator actually chained the two stages
rather than firing them in parallel.

## Run

```bash
sudo docker compose up --build --abort-on-container-exit --exit-code-from probe
```

First build compiles the orchestrator binary (Rust, ~1–2 min cold). A
second run reuses the cached image layer — only re-runs the probe.

Expected output:

```text
kklocalagent-test-orch        | ... orchestrator: orchestrator listening addr=0.0.0.0:7000
kklocalagent-test-orch-probe  | ... probe: mock server listening on :9100
kklocalagent-test-orch-probe  | ... probe: mock server listening on :9200
kklocalagent-test-orch-probe  | ... probe: orchestrator healthy; sending SpeechEnded
kklocalagent-test-orch        | ... orch::events: received event name=SpeechEnded ...
kklocalagent-test-orch        | ... orch::pipeline: transcribing utterance bytes=3244
kklocalagent-test-orch-probe  | ... probe: mock ASR hit; returning canned transcript 'こんにちは'
kklocalagent-test-orch        | ... orch::pipeline: transcribed text=こんにちは
kklocalagent-test-orch-probe  | ... probe: mock LLM hit; received user text 'こんにちは'
kklocalagent-test-orch        | ... orch::pipeline: turn complete user=こんにちは assistant=OK, 了解しました。
kklocalagent-test-orch-probe  | ... probe: PASS: orchestrator plumbed SpeechEnded → ASR → LLM in order
kklocalagent-test-orch-probe exited with code 0
```

## Tear down

```bash
docker compose down
```

No persistent volumes.

## What this smoke does NOT cover

- Real whisper.cpp transcription accuracy — `automatic-speech-recognition/test/` does that.
- Real ollama chat quality — `llm/test/` does that.
- VAD → orchestrator wiring — future follow-up PR (VAD currently has
  an `Orchestrator` sink mode stubbed with a warn-and-drop TODO).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `orch` build fails at `cargo build` | registry mirror down / no network at build time | re-run; or set a different registry via `CARGO_NET_...` env. |
| Probe reports `timeout waiting for mocks — asr_hit=false` | ORCH_ASR_URL misaligned, or orchestrator unable to resolve `probe` | check `docker compose logs orch` — a `connection refused` there means the mock isn't bound yet; compose's `depends_on` starts `orch` first so this is expected for a brief moment, and VAD/probe rely on the pipeline being async. |
| Probe reports `FAIL: LLM was supposed to see 'こんにちは', saw '...'` | orchestrator is passing raw audio / wrong field to LLM | likely a regression in `src/pipeline.rs::llm_chat` — check the `ChatMessage { role: "user", content: … }` path. |
