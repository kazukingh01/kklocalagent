# kklocalagent

## Server ( the other ) - Client ( audio-io ) 

```bash
sudo ufw allow from 172.25.0.0/16 to any port 7010 proto tcp comment 'audio-io tunnel for kklocalagent containers'
sudo ufw reload
```

```bash
ssh home -R 0.0.0.0:7010:$(ip route show | awk '/default/ {print $3}'):7010
```

```bash
sudo docker compose -f compose.yaml -f compose.cpu.yaml up -d --build

sudo WW_MODELS=tanuki.onnx WINDOWS_HOST=$(sudo docker network inspect kklocalagent_default --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}') docker compose -f compose.yaml up --build
```

## LLM Chat-mode

```bash
sudo docker compose -f compose.yaml up -d --build llm agent
sudo docker compose exec agent python /app/experiments/chat_repl.py
sudo docker compose down
```
