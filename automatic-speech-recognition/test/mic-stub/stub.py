import asyncio
import os
import sys
import wave
from typing import List

import websockets

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "7010"))
SAMPLE_PATH = os.environ.get("SAMPLE_PATH", "/samples/jfk.wav")
SAMPLE_PATHS_RAW = os.environ.get("SAMPLE_PATHS", "")
LOOP_DELAY_MS = int(os.environ.get("LOOP_DELAY_MS", "2000"))
LOOP_COUNT = int(os.environ.get("LOOP_COUNT", "1"))
INTER_SAMPLE_SILENCE_MS = int(os.environ.get("INTER_SAMPLE_SILENCE_MS", str(LOOP_DELAY_MS)))

SAMPLE_RATE = 16000
FRAME_MS = 20
BYTES_PER_FRAME = SAMPLE_RATE // 1000 * FRAME_MS * 2


def resolve_sample_paths() -> List[str]:
    if SAMPLE_PATHS_RAW.strip():
        return [p.strip() for p in SAMPLE_PATHS_RAW.split(",") if p.strip()]
    return [SAMPLE_PATH]


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


async def stream_pcm(ws, pcm: bytes, frame_period: float) -> None:
    for i in range(0, len(pcm) - BYTES_PER_FRAME + 1, BYTES_PER_FRAME):
        await ws.send(pcm[i : i + BYTES_PER_FRAME])
        await asyncio.sleep(frame_period)


async def stream_silence(ws, frames: int, frame_period: float) -> None:
    silent_frame = b"\x00" * BYTES_PER_FRAME
    for _ in range(frames):
        await ws.send(silent_frame)
        await asyncio.sleep(frame_period)


async def stream(ws, pcms: List[bytes]) -> None:
    inter_sample_silence_frames = max(1, INTER_SAMPLE_SILENCE_MS // FRAME_MS)
    loop_delay_frames = max(1, LOOP_DELAY_MS // FRAME_MS)
    frame_period = FRAME_MS / 1000.0

    loops_done = 0
    while LOOP_COUNT == 0 or loops_done < LOOP_COUNT:
        for idx, pcm in enumerate(pcms):
            await stream_pcm(ws, pcm, frame_period)
            if idx < len(pcms) - 1:
                await stream_silence(ws, inter_sample_silence_frames, frame_period)
        loops_done += 1
        await stream_silence(ws, loop_delay_frames, frame_period)

    print(
        f"mic-stub: completed {loops_done} loop(s); now idling with silence "
        "(stop with `docker compose down`)",
        flush=True,
    )
    silent_frame = b"\x00" * BYTES_PER_FRAME
    while True:
        await ws.send(silent_frame)
        await asyncio.sleep(frame_period)


async def handler(ws) -> None:
    print("mic-stub: client connected", flush=True)
    try:
        await stream(ws, PCMS)
    except websockets.ConnectionClosed:
        print("mic-stub: client disconnected", flush=True)


async def main() -> None:
    async with websockets.serve(handler, HOST, PORT):
        sources = ", ".join(
            f"{p} ({len(pcm)}B)" for p, pcm in zip(SAMPLE_FILES, PCMS)
        )
        print(
            f"mic-stub: listening on ws://{HOST}:{PORT}/mic "
            f"(samples=[{sources}], inter={INTER_SAMPLE_SILENCE_MS}ms, "
            f"loop_delay={LOOP_DELAY_MS}ms, loops={LOOP_COUNT})",
            flush=True,
        )
        await asyncio.Future()


if __name__ == "__main__":
    SAMPLE_FILES = resolve_sample_paths()
    PCMS = [load_pcm(p) for p in SAMPLE_FILES]
    asyncio.run(main())
