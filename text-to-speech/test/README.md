# test/ — tts smoke tests

Two compose stacks that exercise the text-to-speech path:

| stack | sink | use case |
|---|---|---|
| `compose.offline.yaml` | `out/zundamon.wav` (file) | self-contained, no Windows / audio-io needed |
| `compose.online.yaml`  | live audio-io `/spk` (Windows host) | hear ずんだもん through the speakers |

Both stacks share the same `text-to-speech` (VOICEVOX engine) image and
the same `tts-client` synth-and-send Python helper; only the
`tts-client` command and target differ.

```text
[offline]
tts-client  ── POST /audio_query, /synthesis ──►  text-to-speech
            ◄── WAV ──
            └─► writes /out/zundamon.wav  (bind-mounted from ./out)

[online]
tts-client  ── POST ... ──►  text-to-speech
            ── ffmpeg → s16le 16k mono ──
            ── WS frames ──►  audio-io (Windows) /spk  ── speakers
```

Default speaker is `3 = ずんだもん (ノーマル)`. Override with
`VOICEVOX_SPEAKER` (see `../README.md` for ids).

## Mode 1 — offline (`compose.offline.yaml`)

```bash
cd text-to-speech/test
docker compose -f compose.offline.yaml up --build
```

First boot pulls `voicevox/voicevox_engine:cpu-latest` (~1 GB) and
builds the tts-client image (python:3.12-slim + ffmpeg). The engine
takes ~10–60 s to lazy-mmap the voice models; the Dockerfile
`HEALTHCHECK` gates the synth so the first request doesn't race.

When `tts-client` exits with `wrote /out/zundamon.wav (...)` the WAV is
ready under `./out/`. Tear down with:

```bash
docker compose -f compose.offline.yaml down
```

Play it with any WAV player:

```bash
ffplay -autoexit -nodisp out/zundamon.wav
# or copy to Windows:
cp out/zundamon.wav /mnt/c/Users/$USER/Downloads/
```

Override the phrase / voice for a re-run:

```bash
TEXT='今日はとてもいい天気なのだ' VOICEVOX_SPEAKER=1 \
    docker compose -f compose.offline.yaml up --build
```

## Mode 2 — online (`compose.online.yaml`)

Plays the synthesized utterance through audio-io running natively on
Windows. **Prerequisite**: `audio-io.exe` running on Windows with
`[server] host = "0.0.0.0"` so the WSL2 docker network can reach
`ws://<windows>:7010/spk`. See `../../audio-io/README.md`.

```bash
cd text-to-speech/test
cp .env.example .env   # only needed if host.docker.internal doesn't resolve
docker compose -f compose.online.yaml up --build
```

You should hear ずんだもん speak through the Windows default output
device within ~5 s of the engine reporting healthy.

## Probing the engine directly

Both compose files publish the engine on host port 7060:

```bash
# Generate query JSON for "ぼくはずんだもんなのだ" (speaker 3 = Zundamon).
curl -s -X POST 'http://127.0.0.1:7060/audio_query?speaker=3' \
    --get --data-urlencode 'text=ぼくはずんだもんなのだ' \
    > /tmp/q.json

# Synthesize the WAV.
curl -s -X POST 'http://127.0.0.1:7060/synthesis?speaker=3' \
    -H 'Content-Type: application/json' -d @/tmp/q.json \
    > /tmp/zundamon.wav

# List all available speakers / styles.
curl -s http://127.0.0.1:7060/speakers | jq '.[] | {name, styles}'
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| compose hangs at `text-to-speech ... is starting` | engine still mmap'ing voice models on first boot (cold cache, slow disk) | wait — `start_period=60s, retries=24` allows ~2 min total; check `docker compose logs text-to-speech` for `Application startup complete` |
| online: `WS connect` error | audio-io unreachable | confirm audio-io.exe is up, `[server] host = "0.0.0.0"`, and `${WINDOWS_HOST}` resolves from inside the container (`docker compose exec tts-client getent hosts $WINDOWS_HOST`) |
| online: no sound, but logs show `done` | audio-io was started but `/start` wasn't called | check `docker compose logs tts-client` — it POSTs `/start` before streaming. If that warn-line printed an error, audio-io may need to be restarted with valid output device config |
| `ffmpeg: pipe:0: Invalid data` in tts-client | engine returned an error body instead of WAV (e.g. unknown speaker id) | check `docker compose logs text-to-speech`; verify `VOICEVOX_SPEAKER` against `/speakers` output |
