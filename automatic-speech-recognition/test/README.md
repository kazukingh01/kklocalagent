# test/ — audio-io → vad → asr smoke test

A self-contained docker compose stack that wires three services to prove
the speech-to-text path end to end:

```text
mic-stub ── ws://mic-stub:7010/mic ──► voice-activity-detection
                                                      │
                          POST /inference (WAV) ──────┘
                                                      ▼
                              automatic-speech-recognition (whisper.cpp)
```

`audio-io` itself is Windows-native and lives outside compose, so the
`mic-stub` service replays a wav file as 20 ms PCM frames in its place.
The VAD runs with `sink.mode = "asr-direct"` and POSTs each detected
utterance straight to whisper.cpp's `/inference` endpoint, bypassing the
not-yet-built orchestrator.

## One-time setup

```bash
cd automatic-speech-recognition

# Download whisper ggml-tiny model (~75 MB) into ./models
./fetch-models.sh

# Download the sample wav (JFK clip, ~350 KB) into test/samples
./test/fetch-sample.sh
```

## Run

```bash
cd automatic-speech-recognition/test
docker compose up --build
```

First boot pulls `ghcr.io/ggml-org/whisper.cpp:server` and compiles VAD
in a multi-stage build (a few minutes). Subsequent runs are cached.

## What you should see

Within ~15 s of boot, the `vad` logs print one transcription per loop:

```text
kklocalagent-test-vad  | INFO vad::sink: [event] {"name":"SpeechStarted",...}
kklocalagent-test-vad  | INFO vad::sink: [event] {"name":"SpeechEnded",...}
kklocalagent-test-vad  | INFO vad::asr:  [asr <-] transcription: "And so my fellow Americans, ..."
```

The mic-stub loops the sample with 2 s of silence between iterations, so
the VAD hangs up and re-fires SpeechEnded each loop.

## Tear down

```bash
docker compose down
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `model file not found: /models/ggml-tiny.bin` in `asr` logs | model not downloaded | run `../fetch-models.sh` |
| `mic-stub: ... expected mono` etc. | wrong sample format | re-run `./fetch-sample.sh` (the JFK clip is already 16 kHz mono) |
| VAD never fires SpeechEnded | sample is pure silence, or `LOOP_DELAY_MS` < `hang_frames * 20 ms` | use a real speech sample; default 2000 ms gap is comfortably above the 400 ms default hang |
| `ASR call failed: error sending request` | ASR not yet ready when first event arrives | benign on first boot — events keep firing on each loop, the next one will land |
| Want to swap models | `WHISPER_MODEL` env in `compose.yaml` | `./fetch-models.sh ggml-base.bin` then set `WHISPER_MODEL: ggml-base.bin` |

## Probing ASR directly

The `asr` service publishes port 7040 on the host for ad-hoc curl
testing without involving VAD:

```bash
curl http://127.0.0.1:7040/inference \
    -F file=@samples/jfk.wav \
    -F response_format=json
```
