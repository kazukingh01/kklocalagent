use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use voice_activity_detection::config::{Config, SinkMode};

#[derive(Debug, Parser)]
#[command(
    about = "voice_activity_detection: subscribe to audio-io /mic and emit speech events"
)]
struct Args {
    /// Path to a TOML config file.
    #[arg(long, env = "VAD_CONFIG")]
    config: Option<PathBuf>,

    /// Override the audio-io /mic WebSocket URL from config.
    #[arg(long, env = "VAD_MIC_URL")]
    mic_url: Option<String>,

    /// Override the sink mode from config.
    /// Values: dry-run | asr-direct | orchestrator (orchestrator is TODO).
    #[arg(long, value_enum, env = "VAD_SINK_MODE")]
    sink_mode: Option<SinkMode>,

    /// Override the ASR /inference URL — used in `asr-direct` mode.
    #[arg(long, env = "VAD_ASR_URL")]
    asr_url: Option<String>,

    /// Shortcut for `--sink-mode asr-direct`. Kept for back-compat with
    /// the early dry-run/live split.
    #[arg(long, conflicts_with = "sink_mode")]
    live: bool,
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
    if let Some(url) = args.mic_url {
        config.source.mic_url = url;
    }
    if let Some(url) = args.asr_url {
        config.sink.asr_url = url;
    }
    if let Some(mode) = args.sink_mode {
        config.sink.mode = mode;
    } else if args.live {
        // --live predates the multi-mode sink and meant "anything but
        // dry-run". The only live mode wired up today is asr-direct.
        config.sink.mode = SinkMode::AsrDirect;
    }

    voice_activity_detection::run(config).await
}
