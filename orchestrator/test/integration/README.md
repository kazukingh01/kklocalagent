# test/integration/ — full-pipeline smoke

End-to-end smoke that wires every WSL2-side module (everything except
audio-io, which is Windows-native) and verifies a turn flowing from
fake mic input through wake-word detection, VAD segmentation, ASR
transcription, LLM reply, and back to the test driver.

```text
mic-stub ──WS /mic──► VAD ────────────POST /events──► orchestrator
                    ──WS /mic──► wake-word-detection ─POST /events──┘
                                                                    │
              ┌─────────────────────────────────────────────────────┤
              ▼                                                     ▼
     POST /inference                                       POST /api/chat
   automatic-speech-recognition                                   llm
   (whisper.cpp ggml-small-q8_0)                            (ollama gemma3:1b)
              │                                                     │
              └────► transcript ──► orchestrator ──► assistant ◄────┘
                                          │
                              POST /sink (TurnCompleted +     ┌───► tts-streamer ─► VOICEVOX
                                          WakeWordDetected forwards)│         │
                                          ▼                         │         ▼ /spk WS
                                       assert ◄── /stats poll ───── spk-sink (mock audio-io)
                                          │
                                       exit 0 / 1
```

## What gets verified

The `assert` service watches three signals concurrently:

1. **`WakeWordDetected`** (forwarded via `result_sink`) — fired by
   `wake-word-detection` on the "alexa" segment of the fixture stream.
   Proves `wake-word-detection` can consume audio-io-shaped PCM and
   reach the orchestrator.
2. **`TurnCompleted`** (forwarded via `result_sink`) — fired by the
   orchestrator after a SpeechEnded → ASR → LLM round trip. Proves VAD
   picked an utterance, the orchestrator wrapped + posted it, whisper
   transcribed non-empty text, and ollama generated a non-empty
   assistant reply.
3. **TTS frames on `/spk`** (polled from `spk-sink`'s `/stats`) — at
   least `SPK_MIN_BYTES` (640 B = one frame, default) of *non-silent*
   audio. Proves the orchestrator's TTS stage POSTed `/speak` to
   `tts-streamer`, which synthesised via VOICEVOX, resampled to 16 kHz
   s16le mono, and streamed paced frames over WebSocket — the same
   path that hits audio-io's `/spk` in production.

All three within `ASSERT_TIMEOUT_SEC` (300 s by default) → `assert`
exits 0. Any missing → exit 1, with partial state printed for
diagnosis.

## Models (matching each module's own test)

| Stage | Model | Where the choice comes from |
|---|---|---|
| ASR | `ggml-small-q8_0.bin` (~252 MB) | `automatic-speech-recognition/test/compose.offline.yaml` |
| LLM | `gemma3:1b` (~815 MB) | `llm/test/compose.yaml` |
| Wake | `alexa` (bundled, ~860 KB) | `wake-word-detection/test/compose.yaml` |
| TTS | VOICEVOX `cpu-latest` ONNX | `text-to-speech/Dockerfile` (default speaker = 3, ずんだもん) |

The whisper model is fetched into a named volume (`asr-models`) by an
init-only `asr-models` service on first run, so the smoke is
self-contained — no manual `fetch-models.sh` step required. Subsequent
runs reuse the cached file.

## Fixtures

| File | Source | Role |
|---|---|---|
| `alexa_test.wav` | `wake-word-detection/test/data/` (committed) | wake-word trigger (English, ~0.6 s) |
| `jfk.wav` | `whisper.cpp/samples/` (downloaded at image build) | the user utterance for ASR + LLM (~11 s) |

Both are baked into the `mic-stub` image at build time. The stream is:

```text
[silence 1.5 s] → alexa_test.wav → [silence 1.5 s] → jfk.wav → [silence 3 s]
```

VAD's `hang_frames = 20 × 20 ms = 400 ms` triggers SpeechEnded after
each utterance, so both segments produce `SpeechEnded` events the
orchestrator forwards through ASR → LLM. The assert service only needs
**one** `TurnCompleted` to pass — typically the JFK utterance produces
the more interesting transcript.

## Run

```bash
cd orchestrator/test/integration
docker compose up --build --abort-on-container-exit --exit-code-from assert
```

First run downloads:
- `ggml-small-q8_0.bin` (~252 MB) into the `asr-models` volume
- `gemma3:1b` (~815 MB) into the `ollama-data` volume
- `jfk.wav` (~352 KB) into the `mic-stub` image
- All Rust / Python build dependencies

Total: 5–10 minutes cold. Subsequent runs (volumes intact, images
cached) finish in 1–2 minutes — the bottleneck becomes whisper +
gemma3 inference time on CPU (~10–30 s combined per turn).

## Tear down

```bash
# Keep model caches — next run starts in seconds.
docker compose down

# Wipe everything including volumes.
docker compose down -v
```

## Coverage relative to the mock smoke

The sibling `orchestrator/test/compose.yaml` (mock-backed) is the
faster CI gate. **This** integration smoke complements it by exercising
real models end-to-end:

| Surface | mock smoke | integration smoke |
|---|---|---|
| `POST /events` envelope decoding | ✅ | ✅ |
| audio_base64 → WAV header | ✅ | ✅ |
| ASR multipart shape | ✅ (mock returns canned text) | ✅ (real whisper) |
| LLM JSON shape | ✅ (mock returns canned reply) | ✅ (real ollama) |
| Pipeline ordering (ASR result reaches LLM) | ✅ (assertion) | implicit (TurnCompleted has both fields) |
| **VAD → orchestrator** | ✗ | ✅ |
| **wake-word → orchestrator** | ✗ | ✅ |
| **mic-stub PCM → VAD/wake-word** | ✗ | ✅ |
| **orchestrator → TTS → /spk** | ✗ | ✅ (real VOICEVOX, mock audio-io as `spk-sink`) |
| Real model accuracy on real audio | ✗ | ✅ (smoke level — non-empty text only) |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `asr-models` step fails with curl error | upstream HuggingFace 5xx / network | re-run; fetch is idempotent (skips when file exists). |
| `assert` reports timeout with `wake_word=False, turn_completed=False` | both producers never reached the orchestrator | check `docker compose logs voice-activity-detection wake-word-detection` — most often the mic-stub WS couldn't be reached. |
| `assert` reports `wake_word=True, turn_completed=False` | ASR or LLM didn't complete in time | check `docker compose logs automatic-speech-recognition llm` — first cold gemma3 response can be slow; raise `ASSERT_TIMEOUT_SEC` if needed. |
| `assert` reports empty `user` field | whisper transcribed silence | verify the mic-stub stream (`docker compose logs mic-stub` should show "samples=[…/alexa.wav (…), …/jfk.wav (…)]"). |
| Build of `mic-stub` fails at `COPY wake-word-detection/...` | running compose from outside `integration/` | always `cd` into this directory first — `compose.yaml` builds with the *repo root* as context, but compose itself expects to be invoked from where the file lives. |
| ollama image pull is huge (~3 GB) | first ever pull of `ollama/ollama:latest` | this is unavoidable for the upstream image; subsequent runs are cached. The `ollama-data` volume separately caches `gemma3:1b`. |
