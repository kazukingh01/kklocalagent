//! ONNX-backed wake-word detector. Ported from `livekit-wakeword`
//! 0.1.3 (`src/{lib,wakeword,melspectrogram,embedding}.rs`) with two
//! material changes:
//!
//!   1. The crate runs ONNX through `ort-tract` (pure-Rust). Tract's
//!      matmul/conv kernels are 5–10× slower than onnxruntime's MLAS
//!      on x86_64 — predict took 200–450 ms per 80 ms hop, so
//!      `audio_lag_ms` grew unboundedly. This module uses real
//!      onnxruntime via `ort` with the `load-dynamic` feature, which
//!      drops predict to ~20 ms (measured against the same ONNX with
//!      Python onnxruntime).
//!   2. Mel and embedding ONNX are loaded *from disk paths supplied
//!      by the caller* (typically the train-side uv venv resources)
//!      instead of `include_bytes!`. The classifier was trained
//!      against a specific upstream `livekit-wakeword` Python release;
//!      reading the same files at runtime is the only way to
//!      guarantee feature parity, since Rust crate 0.1.3 and Python
//!      pkg 0.2.0 (the version `train/pyproject.toml` pins) ship
//!      different binaries.
//!
//! The on-disk resampler from upstream is dropped — `audio-io` always
//! emits 16 kHz, so we never need to resample.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ndarray::{Array, Array1, Array2, Axis};
use ort::session::Session;
use ort::value::Tensor;

const SAMPLE_RATE: usize = 16_000;
const MEL_BINS: usize = 32; // openWakeWord melspectrogram output bins
const EMBEDDING_WINDOW: usize = 76; // mel frames per embedding
const EMBEDDING_STRIDE: usize = 8; // mel frames between embeddings
const EMBEDDING_DIM: usize = 96;
const MIN_EMBEDDINGS: usize = 16; // classifier input length

/// 1 / 32768 — i16 → [-1.0, 1.0] f32 normalisation. The training
/// pipeline does the same thing, so this constant is load-bearing for
/// score parity (not just convenience).
const I16_TO_F32: f32 = 1.0 / 32768.0;

const _: () = {
    // Compile-time anchor for the 16 kHz contract: callers feed i16
    // PCM at this rate and the embedding stride below assumes it.
    assert!(SAMPLE_RATE == 16_000);
};

/// Mel spectrogram extractor.
///
/// Input:  f32 PCM, shape `(1, num_samples)`, normalised to [-1, 1].
/// Output: f32 mel features, shape `(time_frames, MEL_BINS)`, after
/// the `x/10 + 2` post-processing that openWakeWord's
/// `melspec_transform` applies.
struct MelspectrogramModel {
    session: Session,
}

impl MelspectrogramModel {
    fn from_path(path: &Path) -> Result<Self> {
        let session = build_session_from_file(path)
            .with_context(|| format!("load mel ONNX: {}", path.display()))?;
        Ok(Self { session })
    }

    fn detect(&mut self, samples: Vec<f32>) -> Result<Array2<f32>> {
        // Take ownership of the f32 buffer the caller already allocated;
        // ndarray + ort can use it directly without a per-predict copy.
        let audio_2d = Array1::from_vec(samples).insert_axis(Axis(0));
        let audio_tensor = Tensor::from_array(audio_2d)?;

        let outputs = self.session.run(ort::inputs![audio_tensor])?;
        let raw = outputs["output"].try_extract_array::<f32>()?;
        // Upstream returns (1, 1, time_frames, mel_bins). Drop the two
        // leading singletons by reshaping; .into_owned() materialises
        // the buffer so into_shape_with_order can rearrange freely.
        let rows = raw.shape()[2];
        let cols = raw.shape()[3];
        let mut output = raw.into_owned().into_shape_with_order((rows, cols))?;
        output.mapv_inplace(|x| x / 10.0 + 2.0);
        Ok(output)
    }
}

/// Embedding model: 76-frame mel window → 96-dim embedding.
///
/// Input:  f32, shape `(1, 76, MEL_BINS, 1)`, row-major flat slice.
/// Output: f32, shape `(1, 1, 1, 96)`. Output tensor name is
/// `conv2d_19` in this specific upstream ONNX export — if a future
/// upstream rebuild renames it, this lookup is what will break first.
struct EmbeddingModel {
    session: Session,
}

impl EmbeddingModel {
    fn from_path(path: &Path) -> Result<Self> {
        let session = build_session_from_file(path)
            .with_context(|| format!("load embedding ONNX: {}", path.display()))?;
        Ok(Self { session })
    }

    fn detect(&mut self, mel_features: Vec<f32>) -> Result<Array1<f32>> {
        let input = Array::from_shape_vec((1, EMBEDDING_WINDOW, MEL_BINS, 1), mel_features)?;
        let tensor = Tensor::from_array(input)?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let raw = outputs["conv2d_19"].try_extract_array::<f32>()?;
        let embedding = raw.into_owned().into_shape_with_order(EMBEDDING_DIM)?;
        Ok(embedding)
    }
}

/// Wake-word inference pipeline: PCM → mel → embeddings → classifier.
///
/// Mel + embedding ONNX are ported from train-side resources;
/// classifier ONNX paths are caller-supplied (typically the trained
/// `tanuki.onnx`).
pub struct WakeWordModel {
    mel_model: MelspectrogramModel,
    emb_model: EmbeddingModel,
    classifiers: BTreeMap<String, Session>,
}

impl WakeWordModel {
    pub fn new(
        mel_onnx_path: &Path,
        embedding_onnx_path: &Path,
        classifier_paths: &[impl AsRef<Path>],
    ) -> Result<Self> {
        let mut model = Self {
            mel_model: MelspectrogramModel::from_path(mel_onnx_path)?,
            emb_model: EmbeddingModel::from_path(embedding_onnx_path)?,
            classifiers: BTreeMap::new(),
        };
        for path in classifier_paths {
            model.load_classifier(path.as_ref())?;
        }
        if model.classifiers.is_empty() {
            return Err(anyhow!(
                "WakeWordModel: no classifier paths supplied"
            ));
        }
        Ok(model)
    }

    fn load_classifier(&mut self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(anyhow!(
                "wake word classifier not found: {}",
                path.display()
            ));
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let session = build_session_from_file(path)
            .with_context(|| format!("load classifier ONNX: {}", path.display()))?;
        self.classifiers.insert(name, session);
        Ok(())
    }

    /// Run inference on ~2 s of i16 PCM at 16 kHz. Windows shorter
    /// than `MIN_EMBEDDINGS * EMBEDDING_STRIDE + EMBEDDING_WINDOW` mel
    /// frames return zeros (warm-up).
    pub fn predict(&mut self, audio_chunk: &[i16]) -> Result<BTreeMap<String, f32>> {
        if self.classifiers.is_empty() {
            return Ok(BTreeMap::new());
        }

        // Build the f32 PCM Vec once and pass it by value to detect()
        // so the model can move it into the input tensor without an
        // internal to_vec() copy. The conversion still allocates once
        // per predict but the second copy inside detect() is gone.
        let samples_f32: Vec<f32> = audio_chunk
            .iter()
            .map(|&x| x as f32 * I16_TO_F32)
            .collect();
        let mel = self.mel_model.detect(samples_f32)?;
        let num_frames = mel.shape()[0];

        if num_frames < EMBEDDING_WINDOW {
            return Ok(self.zero_scores());
        }

        let mut embeddings = Vec::new();
        let mut start = 0;
        while start + EMBEDDING_WINDOW <= num_frames {
            let window = mel.slice(ndarray::s![start..start + EMBEDDING_WINDOW, ..]);
            let window_slice = window.as_standard_layout();
            // to_vec() materialises the EMBEDDING_WINDOW × MEL_BINS slice
            // into a fresh Vec we can hand to detect() by value (which
            // moves it into the tensor — no further copy inside).
            let owned: Vec<f32> = window_slice.as_slice().unwrap().to_vec();
            let emb = self.emb_model.detect(owned)?;
            embeddings.push(emb);
            start += EMBEDDING_STRIDE;
        }

        if embeddings.len() < MIN_EMBEDDINGS {
            return Ok(self.zero_scores());
        }

        let last = &embeddings[embeddings.len() - MIN_EMBEDDINGS..];
        let views: Vec<_> = last.iter().map(|e| e.view()).collect();
        let emb_sequence = ndarray::stack(Axis(0), &views)?;
        let emb_input = emb_sequence.insert_axis(Axis(0));

        // BTreeMap has no with_capacity; iteration order is by sorted key
        // (classifier name) so the "best score on tie" decision below is
        // reproducible across runs.
        let mut predictions: BTreeMap<String, f32> = BTreeMap::new();
        let n_classifiers = self.classifiers.len();
        let mut emb_input = Some(emb_input);
        for (idx, (name, session)) in (&mut self.classifiers).into_iter().enumerate() {
            // Single classifier (the common case) → move the array in
            // without cloning. Multi-classifier path still has to clone
            // for all but the last entry.
            let tensor_in = if idx + 1 == n_classifiers {
                emb_input.take().unwrap()
            } else {
                emb_input.as_ref().unwrap().clone()
            };
            let tensor = Tensor::from_array(tensor_in)?;
            let outputs = session.run(ort::inputs!["embeddings" => tensor])?;
            let raw = outputs["score"].try_extract_array::<f32>()?;
            // Upstream livekit-wakeword classifiers (and our M3 Japanese
            // classifiers trained off the same template) emit a single
            // sigmoid score named "score" with shape (1,) or (1, 1).
            // A 2-class softmax export (shape (1, 2)) would silently
            // pick negative-class as "score" without this assert,
            // inverting the threshold check — fail loud instead.
            let total: usize = raw.shape().iter().product();
            if total != 1 {
                return Err(anyhow!(
                    "classifier {name:?} output \"score\" must be a single \
                     scalar (sigmoid wake probability); got shape {:?} \
                     (total {total} elements). 2-class softmax exports \
                     are not supported — re-export with sigmoid head.",
                    raw.shape()
                ));
            }
            let score = raw.iter().copied().next().unwrap_or(0.0);
            predictions.insert(name.clone(), score);
        }

        Ok(predictions)
    }

    fn zero_scores(&self) -> BTreeMap<String, f32> {
        self.classifiers.keys().map(|k| (k.clone(), 0.0)).collect()
    }
}

fn build_session_from_file(path: &Path) -> Result<Session> {
    // commit_from_file lets onnxruntime mmap the model rather than
    // double-buffering it through a Vec<u8> + commit_from_memory. For
    // the upstream mel/embedding models (~3 MB each) the saving is
    // small; for larger custom classifiers it matters.
    let session = Session::builder()?
        .commit_from_file(path)
        .with_context(|| format!("load ONNX file: {}", path.display()))?;
    Ok(session)
}
