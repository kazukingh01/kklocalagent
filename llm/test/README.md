# test/ — llm smoke test

One compose stack that stands up the `llm` module on CPU with a small
model (`gemma3:1b`, ~815 MB) and fires a single `/api/chat` request via
a `curlimages/curl` sidecar once the model is pulled.

No audio-io / Windows dependency — purely HTTP.

```text
probe ──POST /api/chat──► llm (ollama, CPU) ──► stdout JSON response
```

## Run

```bash
cd llm/test
docker compose up --build
```

Expected output (after the first-boot pull of `gemma3:1b`):

```text
kklocalagent-test-llm        | [init] pulling model: gemma3:1b
kklocalagent-test-llm        | [init] model ready: gemma3:1b
kklocalagent-test-llm-probe  | [probe] POST /api/chat model=gemma3:1b
kklocalagent-test-llm-probe  | {"model":"gemma3:1b","created_at":"...","message":{"role":"assistant","content":"東京です。"}, ...}
```

## Tear down

```bash
# Keep the model cache — next `up` starts in seconds.
docker compose down

# Wipe the model cache along with the stack.
docker compose down -v
```

## Interactive probing

The engine is published on host port **7050**, so you can poke it
directly while the stack is up:

```bash
# List installed models
curl -sS http://127.0.0.1:7050/api/tags | jq .

# Chat (non-streaming)
curl -sS -X POST http://127.0.0.1:7050/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"..."}],"stream":false}'

# Chat (streaming — one JSON per line, `-N` disables curl buffering)
curl -N -sS -X POST http://127.0.0.1:7050/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"..."}],"stream":true}'
```

## Changing the model

The model name is defined once at the top of `compose.yaml` as a YAML
anchor (`x-model: &model "gemma3:1b"`) and referenced by both the `llm`
service's `LLM_MODEL` env and the probe's `MODEL` env. To switch:

1. Edit the `x-model` value in `compose.yaml`.
2. `docker compose up --build`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| compose hangs at `llm is starting` for >1 min | first-boot model pull in progress | tail `docker compose logs llm` — expect `[init] pulling model: gemma3:1b` followed by download progress. ~815 MB over a slow link can take several minutes |
| `llm` eventually goes `unhealthy` | pull stalled / network failure | check `docker compose logs llm` for network errors. Healthcheck budget is `start_period=1200s` + `retries*interval=600s` ≈ 30 min total |
| probe exits with HTTP 404 / "model not found" | model name mismatch between `LLM_MODEL` and the probe's JSON | both are `gemma3:1b` in the shipped compose — verify you didn't change only one side |
| probe output looks garbled / not Japanese | 1B-parameter model quality limit | expected; this stack exists to verify API wiring, not response quality. Use `gemma3:4b` via the root compose (GPU) for production quality |
| `docker: Error response from daemon: ... bind: address already in use` on port 7050 | another service on the host has 7050 | change the `ports:` mapping in `compose.yaml` (e.g. `7051:11434`) or stop the conflicting process |
