"""Evaluation harness for a trained wake-word ONNX.

Walks a directory of recordings, runs each through the same shape the
Rust runtime uses (16 kHz mono i16, 2 s window, 80 ms hop), and reports
recall / FPPH / score distribution at a sweep of thresholds.

This is the quality gate before publishing a model. Per the design doc
(issue #12, milestone M4), a Japanese wake word is only adopted if it
clears recall >= 0.85 and FPPH <= 1.0 against the operator's own voice.
For English the bar is the same, but it's typically met easily because
the upstream embedding model is English-trained.

Recording layout
----------------
Each WAV file is named ``<label>_<NN>.wav``:

* ``positive_01.wav`` ... ``positive_30.wav`` — operator saying the
  target phrase, room-natural, distance and prosody varied.
* ``negative_01.wav`` ... — silence / TV / typing / unrelated speech;
  used to estimate FPPH (false positives per hour). Concatenate ~20
  minutes of negatives for stable numbers.

WAVs must be 16 kHz mono s16le. Convert with sox if needed:
    sox in.wav -r 16000 -c 1 -b 16 -e signed-integer out.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
import soundfile as sf

WINDOW_MS = 2000
HOP_MS = 80
SAMPLE_RATE = 16_000

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_pcm(path: Path) -> np.ndarray:
    pcm, sr = sf.read(path, dtype="int16")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {sr} != {SAMPLE_RATE}")
    if pcm.ndim != 1:
        raise ValueError(f"{path}: expected mono, got shape {pcm.shape}")
    return pcm


def stride_windows(pcm: np.ndarray) -> Iterable[np.ndarray]:
    """Yield 2 s windows hopping every 80 ms — the same shape the
    Rust detector feeds to predict()."""
    win = WINDOW_MS * SAMPLE_RATE // 1000
    hop = HOP_MS * SAMPLE_RATE // 1000
    if len(pcm) < win:
        return
    for start in range(0, len(pcm) - win + 1, hop):
        yield pcm[start:start + win]


def max_score(session: ort.InferenceSession, pcm: np.ndarray) -> float:
    """Run all 80-ms-strided 2 s windows through the ONNX classifier
    and return the peak score across the recording.

    NOTE: this currently passes raw i16 PCM to the ONNX. The actual
    livekit-wakeword runtime feeds the bundled mel + embedding pipeline
    *before* the classifier. For a faithful eval we would either
    replicate that pipeline in Python or call into the Rust runtime
    binary; this stub uses raw PCM as a placeholder until M3 lands the
    pipeline replication or a CLI surface on the runtime."""
    peak = 0.0
    inp_name = session.get_inputs()[0].name
    for window in stride_windows(pcm):
        arr = window.astype(np.int16).reshape(1, -1)
        out = session.run(None, {inp_name: arr})
        score = float(np.max(out[0]))
        peak = max(peak, score)
    return peak


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--recordings", required=True, type=Path,
                   help="directory of <label>_NN.wav files")
    p.add_argument(
        "--allow-stub",
        action="store_true",
        help="bypass the M3 guard. Required because max_score() currently "
             "passes raw i16 PCM directly to the classifier, skipping the "
             "mel + embedding pipeline the runtime applies. Numbers from "
             "this stub are NOT comparable to the runtime's scores and "
             "should never be used to decide whether to publish a model. "
             "Use this only to smoke-test the eval harness wiring itself.",
    )
    args = p.parse_args()

    if not args.allow_stub:
        print(
            "eval.py: refusing to run — current implementation is a "
            "pre-M3 stub that feeds raw PCM to the classifier, bypassing "
            "the mel + embedding pipeline. Outputs do NOT reflect the "
            "runtime's behaviour. See the docstring on max_score(); "
            "pass --allow-stub to override for harness testing.",
            file=sys.stderr,
        )
        return 2

    if not args.model.exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1
    if not args.recordings.is_dir():
        print(f"recordings dir not found: {args.recordings}", file=sys.stderr)
        return 1

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    pos_scores: list[float] = []
    neg_total_hours = 0.0
    neg_fires_by_threshold = {t: 0 for t in THRESHOLDS}

    for wav in sorted(args.recordings.glob("*.wav")):
        pcm = load_pcm(wav)
        peak = max_score(session, pcm)
        # Strict label match: anything that isn't exactly "positive" or
        # "negative" is rejected so a typo (e.g. "postive_03.wav") fails
        # loud rather than silently being counted as a negative — the
        # original startswith("pos") check would have skewed FPPH by
        # treating mislabelled positives as negatives.
        label = wav.stem.split("_", 1)[0].lower()
        if label not in {"positive", "negative"}:
            print(
                f"{wav.name}: unrecognised label prefix {label!r} "
                f"(expected 'positive' or 'negative')",
                file=sys.stderr,
            )
            return 1
        if label == "positive":
            pos_scores.append(peak)
            print(f"[positive] {wav.name}: peak={peak:.3f}")
        else:
            duration_h = len(pcm) / SAMPLE_RATE / 3600.0
            neg_total_hours += duration_h
            for t in THRESHOLDS:
                if peak >= t:
                    neg_fires_by_threshold[t] += 1
            print(f"[negative] {wav.name}: peak={peak:.3f} dur={duration_h*3600:.1f}s")

    print()
    print(f"positives: {len(pos_scores)} recordings")
    print(f"negatives: {neg_total_hours*60:.1f} minutes total")
    print()
    print(f"{'threshold':>10} {'recall':>8} {'FPPH':>8}")
    for t in THRESHOLDS:
        recall = sum(1 for s in pos_scores if s >= t) / max(len(pos_scores), 1)
        fpph = neg_fires_by_threshold[t] / max(neg_total_hours, 1e-9)
        print(f"{t:>10.2f} {recall:>8.3f} {fpph:>8.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
