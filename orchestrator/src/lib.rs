pub mod config;
pub mod events;
pub mod pipeline;
pub mod service;

use anyhow::Result;

pub use config::Config;

pub async fn run(config: Config) -> Result<()> {
    config.validate()?;
    service::run(config).await
}
