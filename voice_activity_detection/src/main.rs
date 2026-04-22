use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use voice_activity_detection::config::Config;

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

    /// Enable live mode (POST events to orchestrator). Default is the test /
    /// dry-run mode that logs events instead. Live mode is not yet
    /// implemented — passing --live currently drops events with a warning.
    #[arg(long)]
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
    // CLI --live takes precedence over config file; default is dry-run so
    // `voice-activity-detection` with no flags is safe to run.
    if args.live {
        config.sink.dry_run = false;
    }

    voice_activity_detection::run(config).await
}
