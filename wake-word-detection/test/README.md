# test/ — wake-word-detection smoke test

Single compose stack that verifies end-to-end: PCM over WebSocket →
openWakeWord detection → HTTP event POST.

```text
probe (WS server :9100)  ──PCM frames──► wwd ──POST /events──► probe (HTTP sink :9200)
                                                                         │
                                                                   "PASS" + exit 0
```

No microphone, no orchestrator — `probe` plays both roles the
production stack splits across `audio-io` and the orchestrator.

## Run

```bash
cd wake-word-detection/test
sudo docker compose up --build --abort-on-container-exit --exit-code-from probe
```

Expected output once the build finishes (first build pulls ~20 MB of
openWakeWord models into the `wwd` image; `alexa_test.wav` is
redistributed from openWakeWord's own test data per `data/NOTICE`):

```text
kklocalagent-test-wwd        | ... wwd: loading openWakeWord models=['alexa'] framework=tflite
kklocalagent-test-wwd        | ... wwd: model loaded: ['alexa']
kklocalagent-test-wwd        | ... wwd: mic connected
kklocalagent-test-wwd-probe  | ... probe: ws: streaming loop 1/3
kklocalagent-test-wwd        | ... wwd: detected: model=alexa score=0.994
kklocalagent-test-wwd-probe  | ... probe: received /events: {'name': 'WakeWordDetected', 'model': 'alexa', 'score': 0.994..., 'ts': ...}
kklocalagent-test-wwd-probe  | ... probe: PASS: detected alexa (score=0.994)
kklocalagent-test-wwd-probe exited with code 0
```

## Tear down

```bash
docker compose down
```

No persistent volumes — the two built images (`test-wwd`, `test-probe`)
hold everything, nothing to clean up on the host.

## Why this fixture (`alexa_test.wav`)

- Redistributed from openWakeWord's own `tests/data/` (Apache-2.0, see
  `data/NOTICE`).
- Matches the most-accurate bundled pre-trained model (`alexa`).
- 0.625s at 16kHz mono 16-bit — under the model's sliding-window
  length by itself, so `probe.py` pads ~1s of silence on each side and
  loops 3× to give the model multiple detection chances.

## Changing the model

`alexa` is the default because the test WAV covers it. To smoke-test a
different bundled model:

1. Replace `data/alexa_test.wav` with a WAV of the new phrase (16kHz,
   mono, 16-bit). `hey_mycroft_test.wav` is also in openWakeWord's
   test data and works out of the box.
2. Update `wwd.environment.WW_MODELS` in `compose.yaml`.
3. `docker compose up --build --abort-on-container-exit --exit-code-from probe`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FAIL: no WakeWordDetected in 60s` | model threshold too high, or WAV not landing at 16kHz | `docker compose logs wwd` — if scores stay < 0.5, lower `WW_THRESHOLD` to `0.3`. Verify the WAV with `python3 -c "import wave; w=wave.open('data/alexa_test.wav'); print(w.getframerate(), w.getnchannels(), w.getsampwidth())"` — expect `16000 1 2`. |
| `wwd` logs `mic ws error: ... Name or service not known` at startup | probe hasn't bound :9100 yet | the probe has a short `pip install` step; the shim retries with backoff, so this is benign. If it persists past ~30s, check `docker compose logs probe`. |
| `ai-edge-litert` / `tflite` import fails on `wwd` build | wheels missing for the base image | switch to ONNX: `wwd.environment.WW_INFERENCE_FRAMEWORK=onnx`. |
| Build downloads hundreds of MB each time | ollama-style layer caching is undone by editing earlier layers | keep `COPY requirements.txt` + `pip install` + `download_models` layers above `COPY shim.py` — editing the shim alone should reuse the heavy cached layer. |
