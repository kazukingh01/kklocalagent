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

This directory uses [`uv`](https://docs.astral.sh/uv/) to manage Python
and the dependency graph. The `.venv/`, `data/`, and `output/`
directories are all gitignored; install once on a GPU host, run
training, and only the resulting ONNX leaves the host.

```sh
# 0. (one-time, per host) install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Materialise the locked env (Python 3.12.8 + ~180 packages,
#    pinned by uv.lock). First run pulls torch + nvidia-cuda wheels
#    (~5 GB). Subsequent runs are seconds.
uv sync

# 2. Download external data (VoxCPM2 weights, ACAV100M features
#    ~16 GB, RIRs, backgrounds). One-time per host.
uv run livekit-wakeword setup --config configs/my_phrase.yaml

# 3. Edit a config — start from configs/example.yaml
cp configs/example.yaml configs/my_phrase.yaml
$EDITOR configs/my_phrase.yaml

# 4. Train. train.sh wraps `uv run livekit-wakeword run`.
bash train.sh configs/my_phrase.yaml

# 5. Evaluate against your own voice (~30 short recordings of the
#    target phrase + a few minutes of silence/random speech)
uv run python eval.py --model output/my_phrase/my_phrase.onnx \
    --recordings ./my_recordings/

# 6. Publish — upload my_phrase.onnx as a GitHub release asset, then
#    point runtime/Dockerfile's wget at the release URL.
```

### Re-pinning dependencies

`pyproject.toml` declares high-level deps with `==` patch pins;
`uv.lock` freezes the full transitive tree. To bump:

```sh
uv lock --upgrade            # refresh uv.lock against latest matching versions
uv sync                      # apply the new lock to .venv
```

To bump a single package: edit its `==X.Y.Z` in `pyproject.toml`,
then `uv lock && uv sync`.

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
