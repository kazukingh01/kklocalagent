"""
Tiny WebSocket server that mimics audio-io's `/mic` endpoint by streaming
a wav file as 20 ms PCM frames. Used by the audio-io→vad→asr smoke test
because audio-io itself is Windows-native and lives outside compose.

Wire format (must match audio-io): s16le, 16 kHz, mono, 640 bytes/frame.

Behaviour:
- Accepts WS clients on any path (VAD connects to /mic).
- On connect, replays SAMPLE_PATH in real time, then injects
  LOOP_DELAY_MS of digital silence so the VAD's hang_frames trigger a
  SpeechEnded event between loops, then repeats.
- Disconnects don't crash the server; it waits for the next client.
"""

import asyncio
import os
import sys
import wave

import websockets

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "7010"))
SAMPLE_PATH = os.environ.get("SAMPLE_PATH", "/samples/jfk.wav")
LOOP_DELAY_MS = int(os.environ.get("LOOP_DELAY_MS", "2000"))

SAMPLE_RATE = 16000
FRAME_MS = 20
BYTES_PER_FRAME = SAMPLE_RATE // 1000 * FRAME_MS * 2  # 640


def load_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1:
            sys.exit(f"{path}: expected mono, got {w.getnchannels()} channels")
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: expected 16-bit, got {w.getsampwidth() * 8}-bit")
        if w.getframerate() != SAMPLE_RATE:
            sys.exit(
                f"{path}: expected {SAMPLE_RATE} Hz, got {w.getframerate()} Hz"
            )
        return w.readframes(w.getnframes())


async def stream(ws, pcm: bytes) -> None:
    silent_frame = b"\x00" * BYTES_PER_FRAME
    silence_frames = max(1, LOOP_DELAY_MS // FRAME_MS)
    frame_period = FRAME_MS / 1000.0
    while True:
        for i in range(0, len(pcm) - BYTES_PER_FRAME + 1, BYTES_PER_FRAME):
            await ws.send(pcm[i : i + BYTES_PER_FRAME])
            await asyncio.sleep(frame_period)
        for _ in range(silence_frames):
            await ws.send(silent_frame)
            await asyncio.sleep(frame_period)


async def handler(ws) -> None:
    print("mic-stub: client connected", flush=True)
    try:
        await stream(ws, PCM)
    except websockets.ConnectionClosed:
        print("mic-stub: client disconnected", flush=True)


async def main() -> None:
    async with websockets.serve(handler, HOST, PORT):
        print(
            f"mic-stub: listening on ws://{HOST}:{PORT}/mic "
            f"(sample={SAMPLE_PATH}, {len(PCM)} bytes loaded)",
            flush=True,
        )
        await asyncio.Future()


if __name__ == "__main__":
    PCM = load_pcm(SAMPLE_PATH)
    asyncio.run(main())
