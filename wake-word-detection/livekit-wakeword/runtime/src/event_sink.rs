use std::time::Duration;

use anyhow::Result;
use reqwest::Client;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::Detection;
use crate::config::{Config, SinkMode};

#[derive(serde::Serialize)]
struct Envelope<'a> {
    name: &'static str,
    model: &'a str,
    score: f32,
    ts: f64,
}

pub async fn run(cfg: Config, mut rx: mpsc::Receiver<Detection>) -> Result<()> {
    let client = Client::builder().timeout(Duration::from_secs(5)).build()?;

    while let Some(det) = rx.recv().await {
        let env = Envelope {
            name: "WakeWordDetected",
            model: &det.model,
            score: det.score,
            ts: det.ts,
        };
        match cfg.sink_mode {
            SinkMode::DryRun => {
                let body = serde_json::to_string(&env).unwrap_or_default();
                info!(payload = %body, "[dry-run] would POST WakeWordDetected");
            }
            SinkMode::Orchestrator => match client.post(&cfg.orchestrator_url).json(&env).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        info!(model = %det.model, "fired event");
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        let trim: String = body.chars().take(200).collect();
                        warn!(%status, body = %trim, "POST /events non-2xx");
                    }
                }
                Err(e) => warn!(error = %e, "POST /events failed"),
            },
        }
    }
    Ok(())
}
