# orchestrator

Central event hub for kklocalagent. Receives events from upstream
producers (VAD, wake-word-detection) over HTTP and drives the
ASR → LLM pipeline on completed utterances.

Scope is v0.1 per #4 §9 step 3: **VAD event in → ASR → LLM → log**.
The state machine (§8) and barge-in are deferred to step 10.

## Design

- **Stateless HTTP** on port 7000.
- `POST /events` is the single ingress for all upstream producers. The
  envelope format follows VAD's existing shape
  (`#[serde(tag = "name")]`), loosely typed so new event types from
  upstream don't break the orchestrator before it learns about them.
- On `SpeechEnded` with `audio_base64`, the orchestrator:
  1. base64-decodes the utterance,
  2. wraps the raw PCM s16le in a 44-byte WAV header,
  3. POSTs multipart to ASR's `/inference` endpoint (whisper-server API),
  4. takes the returned `text`,
  5. POSTs it as a user turn to LLM's `/api/chat` (ollama API,
     `stream: false`),
  6. logs the assistant reply.
- `SpeechStarted` / `WakeWordDetected` are received and logged only.
  The state machine work (step 10) will promote `WakeWordDetected` to
  an `Armed` transition without a schema change.
- **Per-stage backpressure** via `tokio::sync::Semaphore`. Default
  `max_inflight = 1` for both ASR and LLM — matches whisper-server's
  single-request behaviour and avoids head-of-line blocking on ollama.
  Excess utterances are dropped with a warning rather than queued.

## Build

```bash
cd orchestrator
docker build -t kklocalagent/orchestrator .
```

## Run (standalone)

```bash
docker run --rm \
    -e ORCH_ASR_URL=http://127.0.0.1:7040/inference \
    -e ORCH_LLM_URL=http://127.0.0.1:7050/api/chat \
    -e ORCH_LLM_MODEL=gemma3:4b \
    -p 7000:7000 \
    kklocalagent/orchestrator
```

## Configuration

TOML file + env-var / CLI overrides. See `config.example.toml` for the
full shape and defaults. The env vars supported by the binary:

| Env | Overrides | Default |
|---|---|---|
| `ORCH_CONFIG` | path to TOML file | — |
| `ORCH_LISTEN` | `server.listen` | `0.0.0.0:7000` |
| `ORCH_ASR_URL` | `asr.url` | `http://automatic-speech-recognition:8080/inference` |
| `ORCH_LLM_URL` | `llm.url` | `http://llm:11434/api/chat` |
| `ORCH_LLM_MODEL` | `llm.model` | `gemma3:4b` |
| `RUST_LOG` | tracing filter | `info` |

## Event envelope

`POST /events` accepts JSON with a `name` discriminator:

```json
{
  "name": "SpeechEnded",
  "end_frame_index": 123,
  "duration_frames": 50,
  "utterance_bytes": 32000,
  "ts": 1744284000.5,
  "sample_rate": 16000,
  "audio_base64": "...base64-encoded PCM s16le mono..."
}
```

The response is always `200 {"ok": true}` — the pipeline runs off-thread
so the HTTP caller (VAD) isn't blocked on the full transcribe+chat
round trip.

Unknown `name` values are logged and acknowledged (forward-compat for
producers that add new event types).

## Health

`GET /health` — `200 {"ok": true}` as long as the HTTP server is up.
The probe deliberately does **not** check ASR/LLM reachability because
the orchestrator degrades gracefully (logs + drops) when a backend is
down; making `/health` depend on backend status would cascade a single
slow ollama startup into the compose-wide `service_healthy` gate.

## Smoke test

See `test/README.md` — mocked ASR / LLM, no real models pulled.

## VAD wiring

VAD today emits events but the `orchestrator` sink mode is stubbed
(`src/service.rs:249` in `voice-activity-detection` — a `TODO` waiting
for this module). Wiring VAD to POST to this orchestrator is a
follow-up PR; the envelope format is already compatible (loose
deserialisation via `events::EventEnvelope`).
