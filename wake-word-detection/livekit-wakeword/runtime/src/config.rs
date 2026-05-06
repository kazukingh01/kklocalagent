//! Runtime configuration sourced from environment variables. The names
//! mirror the openwakeword Python shim where the semantics are
//! identical (`WW_MIC_URL`, `WW_THRESHOLD`, etc.) so flipping
//! `compose.yaml`'s `build.context` between the two implementations
//! requires no env-var renames in the common case.
//!
//! Differences from the Python shim:
//!   * `WW_MODEL_PATHS` (file paths, comma-separated) replaces
//!     `WW_MODELS` (bare model names resolved against an internal
//!     registry). The runtime consumes plain ONNX files.
//!   * `WW_INFERENCE_FRAMEWORK` is gone — onnxruntime is the only
//!     backend (loaded via `load-dynamic` from ldconfig paths).
//!   * `WW_PREDICT_WINDOW_MS` is new — bounds the ring buffer fed to
//!     `WakeWordModel::predict`. The model returns all-zero scores
//!     for windows shorter than ~2 s, so the default is 2000.
//!   * `WW_FEATURE_ONNX_DIR` / `WW_MEL_ONNX_PATH` /
//!     `WW_EMBEDDING_ONNX_PATH` are new — they point at the mel and
//!     embedding ONNX files. At least one must be set; the Dockerfile
//!     supplies `WW_FEATURE_ONNX_DIR=/opt/models` as the default mount
//!     point so feature ONNX sits alongside the classifier ONNX
//!     referenced by `WW_MODEL_PATHS`. Using the train artefacts
//!     directly (rather than a frozen copy bundled into the binary)
//!     is what guarantees the runtime extracts features identically
//!     to how the classifier was trained — silent drift across
//!     upstream version bumps was the failure mode this avoids.

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
    pub listen_addr: SocketAddr,
    pub sink_mode: SinkMode,
    pub peak_log_interval: Option<Duration>,
    pub peak_log_floor: f32,
}

/// Filenames inside `WW_FEATURE_ONNX_DIR`. These match what the
/// upstream Python `livekit-wakeword` package ships under
/// `livekit/wakeword/resources/`, so a bind-mount of that directory
/// (or a COPY of those two files) into `WW_FEATURE_ONNX_DIR` works
/// without renaming.
const MEL_ONNX_FILENAME: &str = "melspectrogram.onnx";
const EMBEDDING_ONNX_FILENAME: &str = "embedding_model.onnx";

impl Config {
    pub fn from_env() -> Result<Self> {
        let mic_url = std::env::var("WW_MIC_URL")
            .unwrap_or_else(|_| "ws://audio-io:7010/mic".to_string());
        let orchestrator_url = std::env::var("WW_ORCHESTRATOR_URL")
            .unwrap_or_else(|_| "http://orchestrator:7000/events".to_string());

        let raw_paths = std::env::var("WW_MODEL_PATHS")
            .unwrap_or_else(|_| "/opt/models/hey_livekit.onnx".to_string());
        let model_paths: Vec<PathBuf> = raw_paths
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
        if model_paths.is_empty() {
            return Err(anyhow!(
                "WW_MODEL_PATHS resolved to no paths (raw: {raw_paths:?})"
            ));
        }

        let (mel_onnx_path, embedding_onnx_path) = resolve_feature_onnx_paths()?;

        let threshold = parse_env_f32("WW_THRESHOLD", 0.5)?;
        let cooldown = Duration::from_secs_f32(parse_env_f32("WW_COOLDOWN_SEC", 2.0)?);
        let predict_window_ms = parse_env_u32("WW_PREDICT_WINDOW_MS", 2000)?;

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
            listen_addr,
            sink_mode,
            peak_log_interval,
            peak_log_floor,
        })
    }
}

/// Resolve mel + embedding ONNX paths from env vars. Priority:
///
/// 1. `WW_MEL_ONNX_PATH` / `WW_EMBEDDING_ONNX_PATH` (individual file
///    paths). Each override falls back to the directory below if not
///    set, so partial overrides are fine.
/// 2. `WW_FEATURE_ONNX_DIR` joined with the upstream filenames
///    (`melspectrogram.onnx`, `embedding_model.onnx`).
///
/// At least one of these must produce a path for both files. The
/// Dockerfile sets `WW_FEATURE_ONNX_DIR=/opt/models` so the feature
/// ONNX sits alongside the classifier ONNX (`WW_MODEL_PATHS`); local
/// non-Docker runs need to export one of the env vars themselves —
/// most commonly pointing at the train uv venv resources directory.
fn resolve_feature_onnx_paths() -> Result<(PathBuf, PathBuf)> {
    let mel_override = std::env::var("WW_MEL_ONNX_PATH").ok();
    let emb_override = std::env::var("WW_EMBEDDING_ONNX_PATH").ok();
    let dir_env = std::env::var("WW_FEATURE_ONNX_DIR").ok();

    let mel = match (mel_override.as_ref(), dir_env.as_ref()) {
        (Some(p), _) => PathBuf::from(p),
        (None, Some(d)) => PathBuf::from(d).join(MEL_ONNX_FILENAME),
        (None, None) => {
            return Err(anyhow!(
                "mel ONNX path unset. Set WW_MEL_ONNX_PATH or \
                 WW_FEATURE_ONNX_DIR (e.g. /opt/models for the docker \
                 image, or the train uv venv resources directory \
                 locally)."
            ));
        }
    };
    let emb = match (emb_override.as_ref(), dir_env.as_ref()) {
        (Some(p), _) => PathBuf::from(p),
        (None, Some(d)) => PathBuf::from(d).join(EMBEDDING_ONNX_FILENAME),
        (None, None) => {
            return Err(anyhow!(
                "embedding ONNX path unset. Set WW_EMBEDDING_ONNX_PATH \
                 or WW_FEATURE_ONNX_DIR."
            ));
        }
    };

    if !mel.exists() {
        return Err(anyhow!(
            "mel ONNX not found at {}. Check WW_MEL_ONNX_PATH / \
             WW_FEATURE_ONNX_DIR, or that the file is mounted into \
             the container.",
            mel.display()
        ));
    }
    if !emb.exists() {
        return Err(anyhow!(
            "embedding ONNX not found at {}. Check WW_EMBEDDING_ONNX_PATH \
             / WW_FEATURE_ONNX_DIR, or that the file is mounted into \
             the container.",
            emb.display()
        ));
    }
    Ok((mel, emb))
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
