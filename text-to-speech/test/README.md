# test/ — tts smoke tests

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

```bash
cd ./test
sudo docker compose -f compose.offline.yaml up --build
```

```bash
TEXT='今日はとてもいい天気なのだ' VOICEVOX_SPEAKER=1 \
    docker compose -f compose.offline.yaml up --build
```

```bash
echo "WINDOWS_HOST=$(ip route show | awk '/default/ {print $3; exit}')" > .env
sudo docker compose -f compose.online.yaml up --build
```
