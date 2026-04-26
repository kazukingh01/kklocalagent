use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use orchestrator::config::Config;

#[derive(Debug, Parser)]
#[command(about = "orchestrator: receives VAD / wake-word events, drives ASR → LLM pipeline")]
struct Args {
    /// Path to a TOML config file.
    #[arg(long, env = "ORCH_CONFIG")]
    config: Option<PathBuf>,

    /// Override `server.listen` (e.g. `0.0.0.0:7000`).
    #[arg(long, env = "ORCH_LISTEN")]
    listen: Option<String>,

    /// Override `asr.url`.
    #[arg(long, env = "ORCH_ASR_URL")]
    asr_url: Option<String>,

    /// Override `llm.url`.
    #[arg(long, env = "ORCH_LLM_URL")]
    llm_url: Option<String>,

    /// Override `llm.model`.
    #[arg(long, env = "ORCH_LLM_MODEL")]
    llm_model: Option<String>,

    /// Override `tts.url`. Empty disables the TTS stage.
    #[arg(long, env = "ORCH_TTS_URL")]
    tts_url: Option<String>,

    /// Override `tts.stop_url`. Empty disables barge-in TTS cancel.
    #[arg(long, env = "ORCH_TTS_STOP_URL")]
    tts_stop_url: Option<String>,

    /// Override `wake.required` (true / false).
    #[arg(long, env = "ORCH_WAKE_REQUIRED")]
    wake_required: Option<bool>,

    /// Override `wake.arm_window_ms`.
    #[arg(long, env = "ORCH_WAKE_ARM_WINDOW_MS")]
    wake_arm_window_ms: Option<u64>,

    /// Override `wake.barge_in` (true / false).
    #[arg(long, env = "ORCH_WAKE_BARGE_IN")]
    wake_barge_in: Option<bool>,

    /// Override `result_sink.url`. Empty disables forwarding.
    #[arg(long, env = "ORCH_RESULT_SINK_URL")]
    result_sink_url: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let mut config = match &args.config {
        Some(p) => Config::from_file(p)?,
        None => Config::default(),
    };
    if let Some(v) = args.listen {
        config.server.listen = v;
    }
    if let Some(v) = args.asr_url {
        config.asr.url = v;
    }
    if let Some(v) = args.llm_url {
        config.llm.url = v;
    }
    if let Some(v) = args.llm_model {
        config.llm.model = v;
    }
    if let Some(v) = args.tts_url {
        config.tts.url = v;
    }
    if let Some(v) = args.tts_stop_url {
        config.tts.stop_url = v;
    }
    if let Some(v) = args.wake_required {
        config.wake.required = v;
    }
    if let Some(v) = args.wake_arm_window_ms {
        config.wake.arm_window_ms = v;
    }
    if let Some(v) = args.wake_barge_in {
        config.wake.barge_in = v;
    }
    if let Some(v) = args.result_sink_url {
        config.result_sink.url = v;
    }

    orchestrator::run(config).await
}
