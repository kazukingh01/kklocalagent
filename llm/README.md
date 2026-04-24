# llm

Ollama packaged for kklocalagent. Wraps `ollama/ollama:${OLLAMA_TAG}`
(default `latest`) and auto-pulls `${LLM_MODEL}` at container start so
the first request doesn't 404.

## Build

```bash
cd llm
docker build -t kklocalagent/llm .
```

## Run (standalone)

```bash
docker run --rm -p 7050:11434 \
    -v ollama-data:/root/.ollama \
    kklocalagent/llm
```

The container reports **healthy only after** the configured model has
been pulled. First boot for `gemma3:4b` (~3.3 GB) takes 10+ min on a
slow link; the `ollama-data` volume persists the weights so subsequent
boots start in seconds.

Override the model at runtime:

```bash
docker run --rm -p 7050:11434 \
    -e LLM_MODEL=qwen2.5:7b \
    -v ollama-data:/root/.ollama \
    kklocalagent/llm
```

## GPU

The upstream image auto-detects NVIDIA — no Dockerfile changes needed.
In the root `compose.yaml`, expose the GPU via:

```yaml
services:
  llm:
    build: ./llm
    deploy:
      resources:
        reservations:
          devices:
            - {driver: nvidia, count: 1, capabilities: [gpu]}
```

NVIDIA Container Toolkit + WSL2 GPU support must be present on the host.

## API

Ollama exposes its native HTTP API at port 11434 (published as 7050):

```bash
# Chat (non-streaming)
curl -sS -X POST http://127.0.0.1:7050/api/chat \
    -H 'Content-Type: application/json' \
    -d '{
          "model": "gemma3:4b",
          "messages": [{"role": "user", "content": "こんにちは"}],
          "stream": false
        }'

# Chat (streaming — one JSON per line)
curl -N -sS -X POST http://127.0.0.1:7050/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma3:4b","messages":[{"role":"user","content":"..."}],"stream":true}'

# List installed models
curl -sS http://127.0.0.1:7050/api/tags | jq .
```

An OpenAI-compatible shim is also exposed at `/v1/chat/completions`
(same port) if a client speaks the OpenAI API.

## Smoke test

See `test/README.md` — a CPU-only compose stack using `gemma3:1b` and a
`curlimages/curl` sidecar that POSTs `/api/chat` once the engine is
healthy.
