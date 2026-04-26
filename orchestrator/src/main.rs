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
    if let Some(v) = args.result_sink_url {
        config.result_sink.url = v;
    }

    orchestrator::run(config).await
}
