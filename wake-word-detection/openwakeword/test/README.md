# test/ — wake-word-detection smoke tests

```bash
cd wake-word-detection/test
sudo docker compose -f compose.offline.yaml up --build \
    --abort-on-container-exit --exit-code-from probe
```

```bash
sudo WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}') docker compose -f compose.online.yaml up --build
```

Speak "alexa" into the Windows microphone. Expected lines in the
container log:

```text
kklocalagent-test-wwd-online | ... wwd: loading openWakeWord models=['alexa'] framework=tflite sink=dry-run
kklocalagent-test-wwd-online | ... wwd: model loaded: ['alexa']
kklocalagent-test-wwd-online | ... wwd: mic connected
kklocalagent-test-wwd-online | ... wwd: detected: model=alexa score=0.987
kklocalagent-test-wwd-online | ... wwd: [dry-run] would POST WakeWordDetected: {"name":"WakeWordDetected","model":"alexa","score":0.987...,"ts":...}
```
