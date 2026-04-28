# tts-streamer

HTTP shim between the orchestrator and the audio-io playback channel.
Receives `POST /speak {"text": "..."}` from the orchestrator, calls
VOICEVOX `/audio_query` + `/synthesis`, resamples the result to
audio-io's wire format (16 kHz s16le mono, 640 B / 20 ms), and streams
it as paced frames over WebSocket to audio-io's `/spk` endpoint.

The engine itself is a separate service (`text-to-speech/`); this
module is the glue that the orchestrator uses to actually make the box
speak. Keeping the stream loop here (rather than in the orchestrator
binary) avoids pulling ffmpeg + a websocket client into the Rust
service, and matches the existing `text-to-speech/test/tts-client/`
pattern by promoting that smoke client into a long-running server.

## API

| Method | Path     | Description |
|---|---|---|
| GET    | /health  | liveness probe (200 once boot finishes)         |
| POST   | /speak   | synthesize the body's `text` and stream to /spk |

`POST /speak` body:

```json
{ "text": "今日はいい天気だね。" }
```

Response on success:

```json
{
  "ok": true,
  "wav_bytes": 142876,
  "pcm_bytes":  47104,
  "sent_bytes": 47104,
  "duration_s":   1.472
}
```

Concurrent requests are serialised behind an asyncio lock — frames from
two overlapping turns must not interleave on the shared `/spk`
channel.

## Configuration

| Env             | Default                              | Notes                                       |
|---|---|---|
| `VOICEVOX_URL`     | `http://text-to-speech:50021`      | the engine                                  |
| `VOICEVOX_SPEAKER` | `3` (ずんだもん ノーマル)         | per-style id; `curl /speakers` to enumerate |
| `SPK_URL`          | _(unset, /speak returns 502)_      | WS URL — production: `ws://${WINDOWS_HOST}:7010/spk` |
| `AUDIO_IO_BASE`    | _(unset, /start is skipped)_       | HTTP base for audio-io `/start`             |
| `HOST` / `PORT`    | `0.0.0.0:7070`                     | bind                                        |

## Build

```bash
cd tts-streamer
docker build -t kklocalagent/tts-streamer .
```

## Run (standalone smoke)

```bash
docker run --rm -p 7070:7070 \
    -e VOICEVOX_URL=http://host.docker.internal:7060 \
    -e SPK_URL=ws://host.docker.internal:7010/spk \
    -e AUDIO_IO_BASE=http://host.docker.internal:7010 \
    kklocalagent/tts-streamer

# in another shell:
curl -sS -X POST http://127.0.0.1:7070/speak \
    -H 'Content-Type: application/json' \
    -d '{"text":"テスト発話なのだ。"}'
```
