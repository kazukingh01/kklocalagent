# livekit-wakeword training

Custom wake-word training driven by the upstream
[`livekit-wakeword`](https://github.com/livekit/livekit-wakeword)
**Python toolkit** (separate from the Rust crate at `../runtime/`,
which is hosted in the `livekit/rust-sdks` monorepo). The toolkit is
a single CLI:

```sh
livekit-wakeword run configs/<phrase>.yaml
```

It generates synthetic positives via VoxCPM2, mixes negatives + RIRs +
augmentation, trains a small Conv-Attention classifier head over the
crate's bundled embedding model, and exports an ONNX file the runtime
can consume directly (`WW_MODEL_PATHS=...`).

## Why this lives outside the runtime image

* The training stack pulls TTS, audio augmentation, and PyTorch — many
  GB of deps that have no business inside the inference container.
* Training is rare (per phrase, run once); inference is always-on. The
  two have different release cadences and dep graphs.
* GPU is required for a reasonable wall-clock (CPU training is hours
  → days).

So this directory only holds **configs + thin wrappers + an evaluation
harness**. Run training on a separate GPU host; the resulting ONNX is
the only artefact that comes back into this repo's release flow.

## Quickstart

```sh
# 1. Set up Python env (3.10+) on a GPU host
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Edit a config — start from configs/example.yaml
cp configs/example.yaml configs/my_phrase.yaml
$EDITOR configs/my_phrase.yaml

# 3. Train (initial run downloads ACAV100M / AudioSet caches, ~30 min,
#    ~30-80 GB on disk; subsequent runs are ~20 min)
bash train.sh configs/my_phrase.yaml

# 4. Evaluate against your own voice (~30 short recordings of the
#    target phrase + a few minutes of silence/random speech)
python eval.py --model out/my_phrase.onnx --recordings ./my_recordings/

# 5. Publish — upload my_phrase.onnx as a GitHub release asset, then
#    point runtime/Dockerfile's wget at the release URL.
```

## Configs

* [`configs/example.yaml`](configs/example.yaml) — English phrase
  template (recommended for first PoC; English is the strongest
  ensemble in the upstream embedding model).
* [`configs/ja_example.yaml`](configs/ja_example.yaml) — Japanese
  phrase template. Multilingual support exists via VoxCPM2 but
  accuracy trails English; expect to validate carefully against real
  recordings before adopting in production.

## Evaluation harness

[`eval.py`](eval.py) is the quality gate before publishing a model.
It walks a directory of `<label>_NN.wav` files, runs the trained ONNX
through the same pipeline shape the Rust runtime uses (16 kHz mono
i16, 2 s window, 80 ms hop, threshold sweep), and reports recall /
false-positive rate / per-recording score distribution.

The Rust runtime has no equivalent because evaluation is one-shot and
needs the Python ecosystem (numpy / soundfile / matplotlib) anyway.

## Reference

* Upstream training toolkit: <https://github.com/livekit/livekit-wakeword>
* Launch blog: <https://livekit.com/blog/livekit-wakeword>
* Rust crate (consumed by `../runtime/`): <https://crates.io/crates/livekit-wakeword>
