# test/ — llm smoke test

```text
probe ──POST /api/chat──► llm (ollama, CPU) ──► stdout JSON response
```

```bash
sudo docker compose -f compose.cpu.yaml up --build
sudo docker compose -f compose.gpu.yaml up --build
```

## Interactive probing

```bash
# List installed models
curl -sS http://127.0.0.1:7050/api/tags | jq .

# Chat (non-streaming)
curl -sS -X POST http://127.0.0.1:7050/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"今日はどんな気分？"}],"stream":false}'

curl -sS -X POST http://127.0.0.1:7050/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"hf.co/mmnga-o/llm-jp-4-8b-thinking-gguf:Q8_0","messages":[{"role":"user","content":"今日はどんな気分？"}],"stream":false}'

# Chat (streaming — one JSON per line, `-N` disables curl buffering)
curl -N -sS -X POST http://127.0.0.1:7050/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"..."}],"stream":true}'
```
