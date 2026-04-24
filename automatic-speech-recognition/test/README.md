# test/ — vad → asr smoke tests

Two compose stacks that exercise the speech-to-text path end to end:

| stack | audio source | use case |
|---|---|---|
| `compose.offline.yaml` | `mic-stub` plays a TTS-generated wav | self-contained smoke test, no external dep |
| `compose.online.yaml`  | live `audio-io` running on the Windows host | real-time mic input — the actual deployment shape |

Both stacks share the same `vad` and `asr` images and the same
`vad-config.toml`; the online stack just overrides `VAD_MIC_URL` to
point at audio-io instead of the in-compose mic-stub.

```text
[offline]
mic-stub ── ws://mic-stub:7010/mic ──► vad ── POST /inference (WAV) ──► asr

[online]
audio-io (Windows) ── ws://${WINDOWS_HOST}:7010/mic ──► vad ── POST /inference (WAV) ──► asr
```

VAD runs with `sink.mode = "asr-direct"` and POSTs each detected
utterance straight to whisper.cpp's `/inference` endpoint, bypassing
the not-yet-built orchestrator.

## One-time setup (both modes)

The default test runs Japanese ASR with whisper small q8_0
(~250 MB on disk, ~0.5 GB RAM at runtime, ~3–5 s inference per
8 s utterance on CPU). Bump to `ggml-large-v3-turbo-q5_0.bin` /
`ggml-large-v3-turbo-q8_0.bin` for higher accuracy if you can afford
the latency.

```bash
cd automatic-speech-recognition
./fetch-models.sh ggml-small-q8_0.bin    # ~250 MB into ./models
```

---

## Mode 1 — offline (`compose.offline.yaml`)

`mic-stub` synthesizes / replays a wav so you don't need a mic. Great
for verifying the pipeline after code changes.

### Setup

```bash
# Synthesize a Japanese test wav into test/samples (~270 KB).
# Uses gTTS in a one-shot docker container; needs internet.
./test/fetch-sample-ja.sh
```

### Run

```bash
cd automatic-speech-recognition/test
docker compose -f compose.offline.yaml up --build
```

First boot pulls `ghcr.io/ggml-org/whisper.cpp:main` and compiles VAD
in a multi-stage build (a few minutes). Subsequent runs are cached.

### Expected logs

```text
kklocalagent-test-vad      | INFO vad::sink: [event] {"name":"SpeechStarted",...}
kklocalagent-test-vad      | INFO vad::sink: [event] {"name":"SpeechEnded",...}
kklocalagent-test-vad      | INFO vad::asr:  [asr <-] transcription: "こんにちは私はクロードです..."
kklocalagent-test-mic-stub | mic-stub: completed 1 loop(s); now idling with silence ...
```

`LOOP_COUNT=1` (default): the wav plays once, then mic-stub keeps the
WS open while streaming silence. Set `LOOP_COUNT=0` in
`compose.offline.yaml` for a continuous demo.

---

## Mode 2 — online (`compose.online.yaml`)

VAD subscribes to `audio-io.exe` running natively on Windows. Speak
into the Windows microphone; transcriptions appear in the vad logs in
real time.

### Setup

1. **Run audio-io on Windows.** Configure `[server] host = "0.0.0.0"`
   so the WSL2 docker network can reach port 7010. See
   `../../audio-io/README.md` for build / run instructions.

2. **Set `WINDOWS_HOST`** so VAD knows where to find audio-io. Default
   `host.docker.internal` works on Docker Desktop and on docker-ce
   with the WSL2 backend (Docker 20.10+).

   ```bash
   cp test/.env.example test/.env
   # edit test/.env if host.docker.internal doesn't resolve for you
   # (use the explicit Windows IP from `ipconfig` instead)
   ```

### Run

```bash
cd automatic-speech-recognition/test
docker compose -f compose.online.yaml up --build
```

### Expected logs

While silent, the vad container logs `[diag] rms=0 speech_ratio=0.00`
(default `[diag] enabled = true`). Speak and you'll see:

```text
kklocalagent-test-vad-online | INFO vad::diag: [diag] rms=4321 speech_ratio=0.86 ...
kklocalagent-test-vad-online | INFO vad::sink: [event] {"name":"SpeechStarted",...}
kklocalagent-test-vad-online | INFO vad::sink: [event] {"name":"SpeechEnded",...}
kklocalagent-test-vad-online | INFO vad::asr:  [asr <-] transcription: "こんにちは..."
```

---

## Switching back to English

The original English smoke test (whisper-tiny + JFK clip) still works
in offline mode — useful as a fast sanity check.

```bash
./fetch-models.sh                 # ggml-tiny.bin (~75 MB)
./test/fetch-sample.sh            # jfk.wav (~350 KB)
```

Then edit `compose.offline.yaml`:

```yaml
mic-stub:
  environment:
    SAMPLE_PATH: /samples/jfk.wav     # was test-ja.wav
automatic-speech-recognition:
  environment:
    WHISPER_MODEL: ggml-tiny.bin                    # was small-q8_0
    WHISPER_LANGUAGE: en                            # was ja
```

---

## Tear down

```bash
docker compose -f compose.offline.yaml down
# or
docker compose -f compose.online.yaml down
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `model file not found: /models/...` in `asr` logs | model not downloaded | run `../fetch-models.sh ggml-small-q8_0.bin` |
| `mic-stub: ... expected mono` etc. (offline) | wrong sample format | re-run `./fetch-sample-ja.sh` / `./fetch-sample.sh` |
| VAD never fires SpeechEnded | input is pure silence | offline: pick a real speech sample. online: actually speak / check mic device in audio-io |
| online: `WS session error; reconnecting error=WS connect` | audio-io unreachable | check audio-io is running on Windows, listening on `0.0.0.0:7010`, and `${WINDOWS_HOST}` resolves from inside the container (`docker compose exec voice-activity-detection getent hosts $WINDOWS_HOST`) |
| `ASR call failed: error sending request` | ASR still loading when first SpeechEnded fired | uncommon with small model; for offline mode set `LOOP_COUNT=2` so the second pass lands; for online just speak again |
| Want to swap models | `WHISPER_MODEL` env in compose | `./fetch-models.sh ggml-base.bin` then set `WHISPER_MODEL: ggml-base.bin` |

## Probing ASR directly

Both compose files publish ASR on host port 7040 for ad-hoc curl:

```bash
curl http://127.0.0.1:7040/inference \
    -F file=@samples/jfk.wav \
    -F response_format=json
```
