"""
Smoke-test client for the VOICEVOX engine.

Two modes:

  --mode save    : synthesize TEXT, write the engine's WAV to --out.
                   Used by compose.offline.yaml — no audio-io needed.

  --mode stream  : synthesize TEXT, resample to 16 kHz s16le mono via
                   ffmpeg, then stream 640-byte (20 ms) frames over
                   ws://...:7010/spk for real-time playback through
                   audio-io. Used by compose.online.yaml.

The synth pipeline is the standard two-step VOICEVOX flow:
    POST /audio_query?text=...&speaker=N     → query JSON
    POST /synthesis?speaker=N    body=query  → WAV bytes
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import httpx
import websockets

VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://text-to-speech:50021")

# audio-io wire format (must match audio-io README): s16le, 16 kHz, mono.
SAMPLE_RATE = 16000
FRAME_MS = 20
BYTES_PER_FRAME = SAMPLE_RATE // 1000 * FRAME_MS * 2  # 640


def synthesize(text: str, speaker: int) -> bytes:
    with httpx.Client(base_url=VOICEVOX_URL, timeout=60.0) as c:
        q = c.post("/audio_query", params={"text": text, "speaker": speaker})
        q.raise_for_status()
        r = c.post(
            "/synthesis",
            params={"speaker": speaker},
            content=q.content,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return r.content


def to_pcm_16k_mono(wav_bytes: bytes) -> bytes:
    """Decode arbitrary WAV → raw s16le 16 kHz mono via ffmpeg pipes."""
    try:
        p = subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-i", "pipe:0",
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
                "-f", "s16le", "pipe:1",
            ],
            input=wav_bytes,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # Surface ffmpeg's own diagnostics — the common failure here is
        # the engine returning an error JSON body (e.g. unknown speaker
        # id) instead of WAV, which shows up as "Invalid data found
        # when processing input" on stderr.
        sys.stderr.write(
            f"ffmpeg failed (rc={e.returncode}):\n{e.stderr.decode(errors='replace')}"
        )
        raise
    return p.stdout


def save(out_path: Path, text: str, speaker: int) -> None:
    print(f"synthesizing speaker={speaker} text={text!r}", flush=True)
    wav = synthesize(text, speaker)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(wav)
    print(f"wrote {out_path} ({len(wav)} bytes)", flush=True)


async def stream(
    spk_url: str, audio_io_base: str | None, text: str, speaker: int
) -> None:
    print(f"synthesizing speaker={speaker} text={text!r}", flush=True)
    wav = synthesize(text, speaker)
    pcm = to_pcm_16k_mono(wav)
    print(
        f"resampled wav={len(wav)}B → pcm={len(pcm)}B "
        f"({len(pcm) / (SAMPLE_RATE * 2):.2f}s)",
        flush=True,
    )

    # Idempotent /start on audio-io: the playback pipeline must be
    # running before /spk frames are accepted. Skip silently if audio-io
    # doesn't expose it (older builds).
    if audio_io_base:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                await c.post(f"{audio_io_base}/start")
            print(f"POST {audio_io_base}/start ok", flush=True)
        except Exception as e:
            print(f"warn: POST /start failed ({e}); continuing", flush=True)

    print(f"connecting {spk_url}", flush=True)
    async with websockets.connect(spk_url) as ws:
        # Pace at frame_ms so audio-io's playback buffer drains naturally
        # rather than queuing the whole utterance up front.
        frame_period = FRAME_MS / 1000.0
        sent = 0
        for i in range(0, len(pcm) - BYTES_PER_FRAME + 1, BYTES_PER_FRAME):
            await ws.send(pcm[i : i + BYTES_PER_FRAME])
            sent += BYTES_PER_FRAME
            await asyncio.sleep(frame_period)
        # Pad the trailing partial frame with zeros so the tail isn't
        # truncated — audio-io's frame parser drops short writes.
        rem = len(pcm) - sent
        if rem:
            tail = pcm[sent:] + b"\x00" * (BYTES_PER_FRAME - rem)
            await ws.send(tail)
            await asyncio.sleep(frame_period)
        # Brief silence so the last samples actually flush through the
        # device buffer before the WS close.
        for _ in range(10):
            await ws.send(b"\x00" * BYTES_PER_FRAME)
            await asyncio.sleep(frame_period)
    print("done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["save", "stream"], required=True)
    ap.add_argument("--out", default="/out/zundamon.wav")
    ap.add_argument("--spk-url", default=os.environ.get("SPK_URL", ""))
    ap.add_argument(
        "--audio-io-base",
        default=os.environ.get("AUDIO_IO_BASE", ""),
        help="HTTP base for audio-io (used to POST /start before streaming).",
    )
    args = ap.parse_args()

    # Parse env-driven inputs here (not at import) so a bad
    # VOICEVOX_SPEAKER surfaces as a friendly exit, not an import-time
    # stack trace.
    raw_speaker = os.environ.get("VOICEVOX_SPEAKER", "3")
    try:
        speaker = int(raw_speaker)
    except ValueError:
        sys.exit(f"invalid VOICEVOX_SPEAKER={raw_speaker!r} (must be int)")
    text = os.environ.get("TEXT", "ぼくはずんだもんなのだ。")

    if args.mode == "save":
        save(Path(args.out), text, speaker)
    else:
        if not args.spk_url:
            sys.exit("--spk-url or SPK_URL is required for stream mode")
        asyncio.run(
            stream(args.spk_url, args.audio_io_base or None, text, speaker)
        )


if __name__ == "__main__":
    main()
