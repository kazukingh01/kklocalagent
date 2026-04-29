//! Detection sink. Two modes (mirroring the openwakeword shim):
//!   * `orchestrator` — POST `WakeWordDetected` to `WW_ORCHESTRATOR_URL`.
//!   * `dry-run`      — log the would-be envelope and skip the POST.
//!
//! Detection events are best-effort — we log and drop POST failures
//! rather than retrying. The orchestrator is typically reachable
//! (compose `service_healthy` chain), and a missed wake event is
//! recoverable on the next utterance.

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
                        info!(model = %det.model, score = det.score, "fired event");
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        let trim = &body[..body.len().min(200)];
                        warn!(%status, body = %trim, "POST /events non-2xx");
                    }
                }
                Err(e) => warn!(error = %e, "POST /events failed"),
            },
        }
    }
    Ok(())
}
