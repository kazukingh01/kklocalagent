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

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ndarray::{Array, Array1, Array2, Axis};
use ort::session::Session;
use ort::value::Tensor;

const SAMPLE_RATE: usize = 16_000;
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

    fn detect(&mut self, samples: &[f32]) -> Result<Array2<f32>> {
        let audio_2d = Array1::from_vec(samples.to_vec()).insert_axis(Axis(0));
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

    fn detect(&mut self, mel_features: &[f32]) -> Result<Array1<f32>> {
        let input = Array::from_shape_vec((1, EMBEDDING_WINDOW, 32, 1), mel_features.to_vec())?;
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
    classifiers: HashMap<String, Session>,
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
            classifiers: HashMap::new(),
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
    pub fn predict(&mut self, audio_chunk: &[i16]) -> Result<HashMap<String, f32>> {
        if self.classifiers.is_empty() {
            return Ok(HashMap::new());
        }

        let samples_f32: Vec<f32> = audio_chunk.iter().map(|&x| x as f32 * I16_TO_F32).collect();

        let mel = self.mel_model.detect(&samples_f32)?;
        let num_frames = mel.shape()[0];

        if num_frames < EMBEDDING_WINDOW {
            return Ok(self.zero_scores());
        }

        let mut embeddings = Vec::new();
        let mut start = 0;
        while start + EMBEDDING_WINDOW <= num_frames {
            let window = mel.slice(ndarray::s![start..start + EMBEDDING_WINDOW, ..]);
            let window_slice = window.as_standard_layout();
            let emb = self.emb_model.detect(window_slice.as_slice().unwrap())?;
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

        let mut predictions = HashMap::with_capacity(self.classifiers.len());
        for (name, session) in &mut self.classifiers {
            let tensor = Tensor::from_array(emb_input.clone())?;
            let outputs = session.run(ort::inputs!["embeddings" => tensor])?;
            let raw = outputs["score"].try_extract_array::<f32>()?;
            let score = raw.iter().copied().next().unwrap_or(0.0);
            predictions.insert(name.clone(), score);
        }

        Ok(predictions)
    }

    fn zero_scores(&self) -> HashMap<String, f32> {
        self.classifiers.keys().map(|k| (k.clone(), 0.0)).collect()
    }
}

fn build_session_from_file(path: &Path) -> Result<Session> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read ONNX file: {}", path.display()))?;
    let session = Session::builder()?.commit_from_memory(&bytes)?;
    Ok(session)
}
