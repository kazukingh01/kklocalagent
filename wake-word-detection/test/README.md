# test/ — wake-word-detection smoke tests

Two compose stacks, mirrored on the
`automatic-speech-recognition/test/` pattern:

| File | Audio source | Sink | When to use |
|---|---|---|---|
| `compose.offline.yaml` | probe replays `alexa_test.wav` over WS | probe HTTP sink + assert | regression smoke (no mic, no Windows) |
| `compose.online.yaml` | live audio-io on Windows | dry-run (logged only) | manual mic test, verifies real-mic path |

## Offline (automated)

Self-contained — no microphone, no orchestrator — `probe` plays both
roles the production stack splits across `audio-io` and the
orchestrator.

```text
probe (WS server :9100)  ──PCM frames──► wwd ──POST /events──► probe (HTTP sink :9200)
                                                                         │
                                                                   "PASS" + exit 0
```

```bash
cd wake-word-detection/test
sudo docker compose -f compose.offline.yaml up --build \
    --abort-on-container-exit --exit-code-from probe
```

Expected output once the build finishes (first build pulls ~20 MB of
openWakeWord models into the `wwd` image; `alexa_test.wav` is
redistributed from openWakeWord's own test data per `data/NOTICE`):

```text
kklocalagent-test-wwd        | ... wwd: loading openWakeWord models=['alexa'] framework=tflite sink=orchestrator
kklocalagent-test-wwd        | ... wwd: model loaded: ['alexa']
kklocalagent-test-wwd        | ... wwd: mic connected
kklocalagent-test-wwd-probe  | ... probe: ws: streaming loop 1/3
kklocalagent-test-wwd        | ... wwd: detected: model=alexa score=0.994
kklocalagent-test-wwd-probe  | ... probe: received /events: {'name': 'WakeWordDetected', 'model': 'alexa', 'score': 0.994..., 'ts': ...}
kklocalagent-test-wwd-probe  | ... probe: PASS: detected alexa (score=0.994)
kklocalagent-test-wwd-probe exited with code 0
```

Tear down:

```bash
sudo docker compose -f compose.offline.yaml down
```

No persistent volumes — the two built images (`test-wwd`, `test-probe`)
hold everything, nothing to clean up on the host.

## Online (manual, live mic)

Subscribes to the Windows-native `audio-io` so your microphone drives
detection in real time. Runs the shim in **dry-run sink mode** —
detections are logged ("`[dry-run] would POST WakeWordDetected: ...`")
rather than POSTed to an orchestrator (no orchestrator is part of this
stack). The full compose with a real orchestrator wires
`WW_SINK_MODE=orchestrator` instead.

### Prerequisites

1. `audio-io.exe` running on Windows with `[server] host = "0.0.0.0"`
   and the default port 7010 (see `../../audio-io/README.md`).
2. `WINDOWS_HOST` reachable from the WSL2 docker network. Default
   `host.docker.internal` works on Docker Desktop and on docker-ce
   with the WSL2 backend; override via `test/.env` if not.

### Run

```bash
cd wake-word-detection/test
sudo docker compose -f compose.online.yaml up --build
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

Stop with `Ctrl-C`, then:

```bash
sudo docker compose -f compose.online.yaml down
```

Unlike the offline stack, this one runs indefinitely — there's no
`exit-code-from` because the test is "did the developer hear/see a
detection in the log?" rather than an automated assert.

### Tuning while connected

To raise sensitivity (lower threshold) without rebuilding, override
the env vars on the same `up` command:

```bash
sudo WW_THRESHOLD=0.3 WW_COOLDOWN_SEC=1.0 \
    docker compose -f compose.online.yaml up
```

(env-substitutes via the `${VAR:-default}` pattern in `compose.online.yaml`.)

## Why this fixture (`alexa_test.wav`)

- Redistributed from openWakeWord's own `tests/data/` (Apache-2.0, see
  `data/NOTICE`).
- Matches the most-accurate bundled pre-trained model (`alexa`).
- 0.625s at 16kHz mono 16-bit — under the model's sliding-window
  length by itself, so `probe.py` pads ~1s of silence on each side and
  loops 3× to give the model multiple detection chances.

## Changing the model

`alexa` is the default because the offline-test WAV covers it. To
smoke-test a different bundled model:

1. (Offline only) Replace `data/alexa_test.wav` with a WAV of the new
   phrase (16kHz, mono, 16-bit). `hey_mycroft_test.wav` is also in
   openWakeWord's test data and works out of the box.
2. Update `wwd.environment.WW_MODELS` in the compose file you're
   running (`compose.offline.yaml` or `compose.online.yaml`).
3. Re-run with `--build` so the new env is picked up.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Offline: `FAIL: no WakeWordDetected in 60s` | model threshold too high, or WAV not landing at 16kHz | `sudo docker compose logs wwd` — if scores stay < 0.5, lower `WW_THRESHOLD` to `0.3`. Verify the WAV with `python3 -c "import wave; w=wave.open('data/alexa_test.wav'); print(w.getframerate(), w.getnchannels(), w.getsampwidth())"` — expect `16000 1 2`. |
| Online: `mic ws error: getaddrinfo failed` | `host.docker.internal` not resolving | confirm WSL2 / Docker Desktop networking; try setting `WINDOWS_HOST=192.168.x.y` (the Windows host IP visible from WSL) in `test/.env`. |
| Online: connects but no detections when speaking | mic too quiet / wrong device on Windows side | check audio-io's logs on Windows for the captured device; lower `WW_THRESHOLD` to 0.3 to confirm wiring; if scores stay near 0 the mic isn't actually being sampled. |
| Offline: `wwd` logs `mic ws error: ... Name or service not known` at startup | probe hasn't bound :9100 yet | the probe has a short `pip install` step; the shim retries with backoff, so this is benign. If it persists past ~30s, check `docker compose logs probe`. |
| `ai-edge-litert` / `tflite` import fails on `wwd` build | wheels missing for the base image | switch to ONNX: `wwd.environment.WW_INFERENCE_FRAMEWORK=onnx`. |
| Build downloads hundreds of MB each time | layer caching invalidated by editing earlier layers | keep `COPY requirements.txt` + `pip install` + `download_models` layers above `COPY shim.py` — editing the shim alone should reuse the heavy cached layer. |
