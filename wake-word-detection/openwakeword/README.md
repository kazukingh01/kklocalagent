# openwakeword

[openWakeWord](https://github.com/dscripka/openWakeWord) を Python で
薄くラップした shim。`/mic` WS から PCM を受け取り、スコアが閾値を
超えたら orchestrator に `WakeWordDetected` を POST する。

英語の事前学習モデル (`alexa`, `hey_jarvis`, `hey_mycroft`,
`hey_rhasspy`, `current_weather`, `timers`) 同梱。

## Environment

| Name | Default |
|---|---|
| `WW_MIC_URL` | `ws://audio-io:7010/mic` |
| `WW_ORCHESTRATOR_URL` | `http://orchestrator:7000/events` |
| `WW_MODELS` | `alexa` |
| `WW_THRESHOLD` | `0.5` |
| `WW_COOLDOWN_SEC` | `2.0` |
| `WW_INFERENCE_FRAMEWORK` | `tflite` |
| `WW_PORT` | `7030` |
| `WW_SINK_MODE` | `orchestrator` |

## Test

### Offline

```bash
sudo docker compose -f ./test/compose.offline.yaml up \
  --build --abort-on-container-exit --exit-code-from probe
```

以下の結果が出ていればOK

```text
kklocalagent-test-wwd-probe  | 2026-05-07 06:56:20,928 INFO probe: received /events: {'name': 'WakeWordDetected', 'model': 'alexa', 'score': 0.9938252568244934, 'ts': 1778136980.926639}
```

### Online

```bash
sudo WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}') docker compose -f ./test/compose.online.yaml up --build
```

声を出して以下を検知できればOK

```text
kklocalagent-test-wwd-online  | 2026-05-07 06:59:05,731 INFO wwd: detected: model=alexa score=0.720
kklocalagent-test-wwd-online  | 2026-05-07 06:59:05,731 INFO wwd: [dry-run] would POST WakeWordDetected: {"name": "WakeWordDetected", "model": "alexa", "score": 0.7195968627929688, "ts": 1778137145.7311957}