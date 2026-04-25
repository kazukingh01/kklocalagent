# test/ — llm smoke test

```text
probe ──POST /api/chat──► llm (ollama, CPU) ──► stdout JSON response
```

```bash
cd llm/test
docker compose up --build
```

## Interactive probing

```bash
# List installed models
curl -sS http://127.0.0.1:7050/api/tags | jq .

# Chat (non-streaming)
curl -sS -X POST http://127.0.0.1:7050/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"今日はどんな気分？"}],"stream":false}'

# Chat (streaming — one JSON per line, `-N` disables curl buffering)
curl -N -sS -X POST http://127.0.0.1:7050/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"..."}],"stream":true}'
```
