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

The default test runs Japanese ASR with whisper large-v3-turbo q8_0
(~830 MB on disk, ~1.5 GB RAM at runtime).

```bash
cd automatic-speech-recognition

# Download the model into ./models (~830 MB)
./fetch-models.sh ggml-large-v3-turbo-q8_0.bin

# Synthesize a Japanese test wav into test/samples (~270 KB).
# Uses gTTS in a one-shot docker container; needs internet.
./test/fetch-sample-ja.sh
```

To run the original English smoke test (whisper-tiny + JFK clip), see
[Switching back to English](#switching-back-to-english) below.

## Run

```bash
cd automatic-speech-recognition/test
docker compose up --build
```

First boot pulls `ghcr.io/ggml-org/whisper.cpp:main` and compiles VAD
in a multi-stage build (a few minutes). Subsequent runs are cached.

## What you should see

First boot loads the model (~10 s for large-v3-turbo q8_0 on CPU).
Once the wav starts playing, the `vad` logs print one transcription
per utterance (the JP test wav naturally splits into 1–3 segments
depending on pause length):

```text
kklocalagent-test-vad      | INFO vad::sink: [event] {"name":"SpeechStarted",...}
kklocalagent-test-vad      | INFO vad::sink: [event] {"name":"SpeechEnded",...}
kklocalagent-test-vad      | INFO vad::asr:  [asr <-] transcription: "こんにちは私はクロードです..."
kklocalagent-test-mic-stub | mic-stub: completed 1 loop(s); now idling with silence ...
```

By default `LOOP_COUNT=1` — the wav plays once, then mic-stub keeps the
WS open while streaming silence so the VAD doesn't reconnect-spam. Tear
down with `docker compose down` when you're satisfied.

For a continuous demo (transcriptions every ~15 s forever) set
`LOOP_COUNT=0` in `compose.yaml`.

## Switching back to English

The English smoke test (whisper-tiny + JFK clip) still works — it's
faster to boot but the model is too small to be useful for anything
beyond pipeline verification.

```bash
./fetch-models.sh                 # ggml-tiny.bin (~75 MB)
./test/fetch-sample.sh            # jfk.wav (~350 KB)
```

Then edit `test/compose.yaml`:

```yaml
mic-stub:
  environment:
    SAMPLE_PATH: /samples/jfk.wav     # was test-ja.wav
automatic-speech-recognition:
  environment:
    WHISPER_MODEL: ggml-tiny.bin                    # was large-v3-turbo q8_0
    WHISPER_LANGUAGE: en                            # was ja
```

## Tear down

```bash
docker compose down
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `model file not found: /models/...` in `asr` logs | model not downloaded | run `../fetch-models.sh ggml-large-v3-turbo-q8_0.bin` |
| `mic-stub: ... expected mono` etc. | wrong sample format | re-run `./fetch-sample-ja.sh` / `./fetch-sample.sh` (both produce 16 kHz mono s16le) |
| VAD never fires SpeechEnded | sample is pure silence, or `LOOP_DELAY_MS` < `hang_frames * 20 ms` | use a real speech sample; default 2000 ms gap is comfortably above the 400 ms default hang |
| `ASR call failed: error sending request` | ASR still loading the model when the first SpeechEnded fired | rare on tiny model (loads in ~3 s); if you see it consistently, set `LOOP_COUNT=2` so the second pass catches the missed transcription |
| Want to swap models | `WHISPER_MODEL` env in `compose.yaml` | `./fetch-models.sh ggml-base.bin` then set `WHISPER_MODEL: ggml-base.bin` |

## Probing ASR directly

The `asr` service publishes port 7040 on the host for ad-hoc curl
testing without involving VAD:

```bash
curl http://127.0.0.1:7040/inference \
    -F file=@samples/jfk.wav \
    -F response_format=json
```
