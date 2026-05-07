//! Runtime configuration sourced from environment variables. The names
//! mirror the openwakeword Python shim where the semantics are
//! identical (`WW_MIC_URL`, `WW_MODELS`, `WW_THRESHOLD`, etc.) so
//! flipping `compose.yaml`'s `build.context` between the two
//! implementations requires no env-var renames in the common case.
//!
//! Differences from the Python shim:
//!   * `WW_MODELS` is comma-separated **filenames** (resolved against
//!     `WW_MODELS_DIR`), not bare names from an internal registry —
//!     the runtime loads plain ONNX files.
//!   * `WW_MODELS_DIR` (default `/opt/models`) is new — directory
//!     containing the classifier ONNX(s) plus the upstream
//!     `melspectrogram.onnx` + `embedding_model.onnx`. Using the train
//!     artefacts directly (rather than freezing a copy in the binary)
//!     guarantees the runtime extracts features identically to how the
//!     classifier was trained — silent drift across upstream version
//!     bumps was the failure mode this avoids.
//!   * `WW_INFERENCE_FRAMEWORK` is gone — onnxruntime is the only
//!     backend (loaded via `load-dynamic` from ldconfig paths).
//!   * `WW_PREDICT_WINDOW_MS` is new — bounds the ring buffer fed to
//!     `WakeWordModel::predict`. The model returns all-zero scores
//!     for windows shorter than ~2 s, so the default is 2000.

use anyhow::{anyhow, Context, Result};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub enum SinkMode {
    Orchestrator,
    DryRun,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub mic_url: String,
    pub orchestrator_url: String,
    pub model_paths: Vec<PathBuf>,
    pub mel_onnx_path: PathBuf,
    pub embedding_onnx_path: PathBuf,
    pub threshold: f32,
    pub cooldown: Duration,
    pub predict_window_ms: u32,
    pub predict_interval_ms: u32,
    pub listen_addr: SocketAddr,
    pub sink_mode: SinkMode,
    pub peak_log_interval: Option<Duration>,
    pub peak_log_floor: f32,
}

/// Filenames inside `WW_MODELS_DIR`. These match what the upstream
/// Python `livekit-wakeword` package ships under
/// `livekit/wakeword/resources/`, so a bind-mount of that directory
/// (or a COPY of those two files) into `WW_MODELS_DIR` works without
/// renaming.
const MEL_ONNX_FILENAME: &str = "melspectrogram.onnx";
const EMBEDDING_ONNX_FILENAME: &str = "embedding_model.onnx";
const DEFAULT_MODELS_DIR: &str = "/opt/models";
const DEFAULT_CLASSIFIER_FILENAME: &str = "hey_livekit.onnx";

impl Config {
    pub fn from_env() -> Result<Self> {
        let mic_url = std::env::var("WW_MIC_URL")
            .unwrap_or_else(|_| "ws://audio-io:7010/mic?ts=1".to_string());
        let orchestrator_url = std::env::var("WW_ORCHESTRATOR_URL")
            .unwrap_or_else(|_| "http://orchestrator:7000/events".to_string());

        let models_dir = PathBuf::from(
            std::env::var("WW_MODELS_DIR").unwrap_or_else(|_| DEFAULT_MODELS_DIR.to_string()),
        );

        let raw_models = std::env::var("WW_MODELS")
            .unwrap_or_else(|_| DEFAULT_CLASSIFIER_FILENAME.to_string());
        let model_paths: Vec<PathBuf> = raw_models
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|name| models_dir.join(name))
            .collect();
        if model_paths.is_empty() {
            return Err(anyhow!(
                "WW_MODELS resolved to no filenames (raw: {raw_models:?})"
            ));
        }
        for p in &model_paths {
            if !p.exists() {
                return Err(anyhow!(
                    "classifier ONNX not found at {} (set WW_MODELS_DIR or mount the file)",
                    p.display()
                ));
            }
        }

        let mel_onnx_path = models_dir.join(MEL_ONNX_FILENAME);
        let embedding_onnx_path = models_dir.join(EMBEDDING_ONNX_FILENAME);
        for (label, p) in [("mel", &mel_onnx_path), ("embedding", &embedding_onnx_path)] {
            if !p.exists() {
                return Err(anyhow!(
                    "{label} ONNX not found at {} (set WW_MODELS_DIR to a directory \
                     containing {MEL_ONNX_FILENAME} + {EMBEDDING_ONNX_FILENAME}, \
                     e.g. the train uv venv resources dir)",
                    p.display()
                ));
            }
        }

        let threshold = parse_env_f32("WW_THRESHOLD", 0.5)?;
        let cooldown = Duration::from_secs_f32(parse_env_f32("WW_COOLDOWN_SEC", 2.0)?);
        let predict_window_ms = parse_env_u32("WW_PREDICT_WINDOW_MS", 2000)?;
        let predict_interval_ms = parse_env_u32("WW_PREDICT_INTERVAL_MS", 100)?;
        if predict_interval_ms == 0 {
            return Err(anyhow!("WW_PREDICT_INTERVAL_MS must be > 0"));
        }

        let listen_str =
            std::env::var("WW_LISTEN").unwrap_or_else(|_| "0.0.0.0:7030".to_string());
        let listen_addr: SocketAddr = listen_str
            .parse()
            .with_context(|| format!("WW_LISTEN parse: {listen_str}"))?;

        let sink_raw = std::env::var("WW_SINK_MODE")
            .unwrap_or_else(|_| "orchestrator".to_string())
            .to_lowercase();
        let sink_mode = match sink_raw.as_str() {
            "orchestrator" => SinkMode::Orchestrator,
            "dry-run" => SinkMode::DryRun,
            other => {
                return Err(anyhow!(
                    "WW_SINK_MODE must be one of orchestrator|dry-run, got {other:?}"
                ))
            }
        };

        let peak_log_interval_secs = parse_env_f32("WW_PEAK_LOG_INTERVAL_SEC", 0.0)?;
        let peak_log_interval = (peak_log_interval_secs > 0.0)
            .then(|| Duration::from_secs_f32(peak_log_interval_secs));
        let peak_log_floor = parse_env_f32("WW_PEAK_LOG_FLOOR", 0.05)?;

        Ok(Config {
            mic_url,
            orchestrator_url,
            model_paths,
            mel_onnx_path,
            embedding_onnx_path,
            threshold,
            cooldown,
            predict_window_ms,
            predict_interval_ms,
            listen_addr,
            sink_mode,
            peak_log_interval,
            peak_log_floor,
        })
    }
}

fn parse_env_f32(name: &str, default: f32) -> Result<f32> {
    match std::env::var(name) {
        Ok(v) => v.parse::<f32>().with_context(|| format!("{name}={v} not f32")),
        Err(_) => Ok(default),
    }
}

fn parse_env_u32(name: &str, default: u32) -> Result<u32> {
    match std::env::var(name) {
        Ok(v) => v.parse::<u32>().with_context(|| format!("{name}={v} not u32")),
        Err(_) => Ok(default),
    }
}
