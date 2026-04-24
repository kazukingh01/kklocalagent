# text-to-speech

VOICEVOX engine packaged for kklocalagent. Wraps the upstream
`voicevox/voicevox_engine:${VOICEVOX_VARIANT}` image (default
`cpu-latest`; use `nvidia-latest` for CUDA) and exposes the HTTP API on
port 50021. The default speaker is **3 = ずんだもん (ノーマル)**.

## Build

```bash
cd text-to-speech
docker build -t kklocalagent/text-to-speech .

# GPU variant
docker build --build-arg VOICEVOX_VARIANT=nvidia-latest \
    -t kklocalagent/text-to-speech:nvidia .
```

## Run

```bash
docker run --rm -p 7060:50021 kklocalagent/text-to-speech
```

The engine takes ~10–60 s on first boot to mmap the ONNX voice models;
the Dockerfile `HEALTHCHECK` polls `/version` so dependent compose
services can `depends_on: condition: service_healthy`.

## API

VOICEVOX synthesis is a two-step flow:

```bash
# 1) Generate the audio_query JSON for `text`
curl -s -X POST 'http://127.0.0.1:7060/audio_query?speaker=3' \
    --get --data-urlencode 'text=ぼくはずんだもんなのだ。' \
    > query.json

# 2) Synthesize WAV from that query
curl -s -X POST 'http://127.0.0.1:7060/synthesis?speaker=3' \
    -H 'Content-Type: application/json' -d @query.json \
    > out.wav
```

Output is **WAV s16le, 24 kHz, mono** by default. Resample to audio-io's
wire format (16 kHz mono s16le) when streaming back through `/spk`.

## Speakers

```bash
curl -s http://127.0.0.1:7060/speakers | jq '.[] | {name, styles}'
```

`speaker` is the per-style id, not the per-character one. Common ids:

| id | character / style |
|---:|---|
| 2 | 四国めたん (ノーマル) |
| 3 | ずんだもん (ノーマル) — default |
| 8 | 春日部つむぎ (ノーマル) |
| 14 | 冥鳴ひまり (ノーマル) |

VOICEVOX licensing: generated voice content may be used commercially or
non-commercially provided you credit the speaker, e.g. `VOICEVOX:ずんだもん`.

## Smoke tests

See `test/README.md` — two compose stacks:

- `compose.offline.yaml` — synthesize a fixed phrase to `test/out/zundamon.wav`
- `compose.online.yaml`  — synthesize and stream to audio-io `/spk` for live playback
