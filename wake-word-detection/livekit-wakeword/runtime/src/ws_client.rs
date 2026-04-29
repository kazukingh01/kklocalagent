//! Subscribe to audio-io's `/mic` WebSocket. The server emits one
//! binary frame per ~20 ms of capture (s16le mono 16 kHz; 320 samples
//! = 640 bytes). We decode each frame into `Vec<i16>` and forward to
//! the detector.
//!
//! Reconnects with exponential backoff on disconnect or read error.
//! `connected` is the flag the /health probe reads — true between
//! handshake and disconnect, false otherwise.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn};

pub async fn run(
    url: String,
    tx: mpsc::Sender<Vec<i16>>,
    connected: Arc<AtomicBool>,
) -> Result<()> {
    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(30);

    loop {
        info!(url = %url, "connecting to mic ws");
        match connect_async(&url).await {
            Ok((ws, _)) => {
                connected.store(true, Ordering::Relaxed);
                backoff = Duration::from_secs(1);
                info!("mic ws connected");
                let (_write, mut read) = ws.split();
                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Binary(bytes)) => {
                            // s16le → Vec<i16>. audio-io rejects
                            // odd-length frames upstream, so the
                            // exact-chunks split drops nothing.
                            let samples: Vec<i16> = bytes
                                .chunks_exact(2)
                                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                                .collect();
                            if tx.send(samples).await.is_err() {
                                // detector dropped — runtime is shutting down.
                                return Ok(());
                            }
                        }
                        Ok(Message::Close(_)) => {
                            warn!("mic ws closed by peer");
                            break;
                        }
                        // Audio-io's server (axum) does not currently
                        // send pings, and we do not write to the WS, so
                        // ping/pong/text frames are dropped silently.
                        // Re-introduce explicit pong handling here if a
                        // future audio-io revision starts pinging.
                        Ok(_) => {}
                        Err(e) => {
                            warn!("mic ws read error: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                warn!("mic ws connect failed: {e}");
            }
        }
        connected.store(false, Ordering::Relaxed);
        warn!(?backoff, "reconnecting after backoff");
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(max_backoff);
    }
}
