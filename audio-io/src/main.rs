use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use audio_io::config::Config;

#[derive(Debug, Parser)]
#[command(about = "audio-io: mic capture & speaker playback for kklocalagent")]
struct Args {
    #[arg(long, env = "AUDIO_IO_CONFIG")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let config = match args.config {
        Some(p) => Config::from_file(&p)?,
        None => Config::default(),
    };

    audio_io::run(config).await
}
