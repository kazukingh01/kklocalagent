//! Env-var config. Names mirror the openwakeword Python shim so swapping
//! compose's `build.context` needs no env renames; unlike the shim,
//! `WW_MODELS` is comma-separated filenames resolved under `WW_MODELS_DIR`,
//! which must also hold the upstream `melspectrogram.onnx` +
//! `embedding_model.onnx` (same artefacts as training, so feature
//! extraction can't silently drift across upstream version bumps).

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
    /// Threshold crossings (cooldown-deduplicated) required within
    /// `confirm_window` before a Detection is forwarded; 1 = immediate.
    pub confirm_count: u32,
    pub confirm_window: Duration,
    pub predict_window_ms: u32,
    pub predict_interval_ms: u32,
    pub listen_addr: SocketAddr,
    pub sink_mode: SinkMode,
    pub peak_log_interval: Option<Duration>,
    pub peak_log_floor: f32,
}

/// Filenames match what upstream `livekit-wakeword` ships under
/// `livekit/wakeword/resources/`, so a bind-mount works without renaming.
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
        // Reject path separators / `..` so WW_MODELS entries can't escape
        // the bind-mounted models dir.
        let names: Vec<&str> = raw_models
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        for name in &names {
            let p = std::path::Path::new(name);
            let escapes = p.components().any(|c| {
                matches!(
                    c,
                    std::path::Component::ParentDir
                        | std::path::Component::RootDir
                        | std::path::Component::Prefix(_)
                )
            });
            if escapes || name.contains('/') || name.contains('\\') {
                return Err(anyhow!(
                    "WW_MODELS entry {name:?} must be a plain filename \
                     under WW_MODELS_DIR (no '/', '\\\\', or '..')"
                ));
            }
        }
        let model_paths: Vec<PathBuf> = names.iter().map(|n| models_dir.join(n)).collect();
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
        // NOTE: a single utterance scores high for up to ~1.5 s and only
        // `cooldown` stops it counting twice, so confirm_window MUST be >
        // cooldown for confirm_count>=2 to be reachable by separate utterances.
        let confirm_count = parse_env_u32("WW_CONFIRM_COUNT", 1)?;
        if confirm_count == 0 {
            return Err(anyhow!("WW_CONFIRM_COUNT must be >= 1"));
        }
        let confirm_window =
            Duration::from_millis(parse_env_u32("WW_CONFIRM_WINDOW_MS", 3000)? as u64);
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
            confirm_count,
            confirm_window,
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
