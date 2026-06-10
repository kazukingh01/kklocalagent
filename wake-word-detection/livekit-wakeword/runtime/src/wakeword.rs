//! Ported from `livekit-wakeword` 0.1.3 with two material changes:
//! (1) real onnxruntime via `ort` `load-dynamic` instead of the crate's
//! `ort-tract` — tract was 5–10× slower (predict 200–450 ms per 80 ms hop,
//! so `audio_lag_ms` grew unboundedly; onnxruntime is ~20 ms);
//! (2) mel/embedding ONNX loaded from caller-supplied disk paths, not
//! `include_bytes!` — Rust crate 0.1.3 and the Python pkg 0.2.0 the
//! classifier was trained against ship different binaries, and reading the
//! train-side files is the only way to guarantee feature parity.
//! Upstream's resampler is dropped — audio-io always emits 16 kHz.

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

/// i16 → [-1.0, 1.0] normalisation; the training pipeline does the same,
/// so this constant is load-bearing for score parity.
const I16_TO_F32: f32 = 1.0 / 32768.0;

const _: () = {
    // Compile-time anchor for the 16 kHz contract the embedding stride assumes.
    assert!(SAMPLE_RATE == 16_000);
};

/// Input: f32 PCM `(1, num_samples)` in [-1, 1]; output: mel features
/// `(time_frames, MEL_BINS)` after the `x/10 + 2` post-processing that
/// openWakeWord's `melspec_transform` applies.
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
        let audio_2d = Array1::from_vec(samples).insert_axis(Axis(0));
        let audio_tensor = Tensor::from_array(audio_2d)?;

        let outputs = self.session.run(ort::inputs![audio_tensor])?;
        let raw = outputs["output"].try_extract_array::<f32>()?;
        // Upstream returns (1, 1, time_frames, mel_bins); drop the two
        // leading singletons.
        let rows = raw.shape()[2];
        let cols = raw.shape()[3];
        let mut output = raw.into_owned().into_shape_with_order((rows, cols))?;
        output.mapv_inplace(|x| x / 10.0 + 2.0);
        Ok(output)
    }
}

/// 76-frame mel window `(1, 76, MEL_BINS, 1)` → 96-dim embedding
/// `(1, 1, 1, 96)`. Output tensor name `conv2d_19` is specific to this
/// upstream ONNX export — a future rebuild renaming it breaks here first.
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

    /// Run inference on ~2 s of i16 PCM at 16 kHz. Windows shorter than
    /// `MIN_EMBEDDINGS * EMBEDDING_STRIDE + EMBEDDING_WINDOW` mel frames
    /// return zeros (warm-up).
    pub fn predict(&mut self, audio_chunk: &[i16]) -> Result<BTreeMap<String, f32>> {
        if self.classifiers.is_empty() {
            return Ok(BTreeMap::new());
        }

        // Pass the f32 Vec by value so detect() moves it into the input
        // tensor without a second copy.
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

        // BTreeMap: sorted-key iteration keeps the "best score on tie"
        // decision reproducible across runs.
        let mut predictions: BTreeMap<String, f32> = BTreeMap::new();
        let n_classifiers = self.classifiers.len();
        let mut emb_input = Some(emb_input);
        for (idx, (name, session)) in (&mut self.classifiers).into_iter().enumerate() {
            // Move the array into the last (usually only) classifier; clone
            // for the rest.
            let tensor_in = if idx + 1 == n_classifiers {
                emb_input.take().unwrap()
            } else {
                emb_input.as_ref().unwrap().clone()
            };
            let tensor = Tensor::from_array(tensor_in)?;
            let outputs = session.run(ort::inputs!["embeddings" => tensor])?;
            let raw = outputs["score"].try_extract_array::<f32>()?;
            // Classifiers must emit a single sigmoid "score" of shape (1,) or
            // (1, 1). A 2-class softmax export (1, 2) would silently surface
            // the negative class and invert the threshold check — fail loud.
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
    // commit_from_file lets onnxruntime mmap the model instead of
    // double-buffering through a Vec<u8>.
    let session = Session::builder()?
        .commit_from_file(path)
        .with_context(|| format!("load ONNX file: {}", path.display()))?;
    Ok(session)
}
