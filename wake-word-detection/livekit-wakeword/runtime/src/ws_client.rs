//! Subscribe to audio-io's `/mic?ts=1` WebSocket. The server emits one
//! binary frame per ~20 ms of capture: an 8-byte little-endian u64
//! header carrying epoch-ns of the frame's *last* sample, followed by
//! s16le mono 16 kHz PCM (320 samples = 640 bytes). We split the
//! header off, decode the body into `Vec<i16>`, and forward both to
//! the detector via `MicFrame`.
//!
//! The `?ts=1` query is appended automatically if the configured URL
//! doesn't already include it — older audio-io revisions that don't
//! understand the parameter ignore unknown query params, but they
//! also won't prepend the header, so this client only works against
//! a header-aware audio-io.
//!
//! Reconnects with exponential backoff on disconnect or read error.
//! `connected` is the flag the /health probe reads — true between
//! handshake and disconnect, false otherwise.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn};

use crate::MicFrame;

const HEADER_LEN: usize = 8;

/// Only reset the reconnect backoff once a connection has lived this
/// long. Without this gate, a 100 ms flap would reset to 1 s on every
/// handshake and the loop would hammer the server at 1 Hz indefinitely.
const STABLE_RESET_AFTER: Duration = Duration::from_secs(10);

pub async fn run(
    url: String,
    tx: mpsc::Sender<MicFrame>,
    connected: Arc<AtomicBool>,
) -> Result<()> {
    let url = ensure_ts_query(&url);
    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(30);

    loop {
        info!(url = %url, "connecting to mic ws");
        let mut connect_inst: Option<Instant> = None;
        match connect_async(&url).await {
            Ok((ws, _)) => {
                connected.store(true, Ordering::Relaxed);
                connect_inst = Some(Instant::now());
                info!("mic ws connected");
                let (_write, mut read) = ws.split();
                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Binary(bytes)) => {
                            if bytes.len() < HEADER_LEN {
                                warn!(
                                    len = bytes.len(),
                                    "mic ws: dropping short frame (no ts header)"
                                );
                                continue;
                            }
                            let mut hdr = [0u8; HEADER_LEN];
                            hdr.copy_from_slice(&bytes[..HEADER_LEN]);
                            let end_epoch_ns = u64::from_le_bytes(hdr);
                            // s16le → Vec<i16>. audio-io rejects
                            // odd-length frames upstream, so the
                            // exact-chunks split drops nothing.
                            let samples: Vec<i16> = bytes[HEADER_LEN..]
                                .chunks_exact(2)
                                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                                .collect();
                            let frame = MicFrame {
                                end_epoch_ns,
                                samples,
                            };
                            if tx.send(frame).await.is_err() {
                                // detector dropped — runtime is shutting down.
                                // Clear connected so /health stops claiming
                                // the WS is up while the runtime tears down.
                                connected.store(false, Ordering::Relaxed);
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
        // Only reset backoff if the connection lived long enough to
        // count as "stable". A 100 ms flap that resets every time keeps
        // hammering the peer at 1 Hz; gating on connect lifetime makes
        // the backoff actually accumulate for unhealthy peers.
        let stable = connect_inst
            .map(|t| t.elapsed() >= STABLE_RESET_AFTER)
            .unwrap_or(false);
        if stable {
            backoff = Duration::from_secs(1);
        }
        warn!(?backoff, stable_reset = stable, "reconnecting after backoff");
        tokio::time::sleep(backoff).await;
        if !stable {
            backoff = (backoff * 2).min(max_backoff);
        }
    }
}

fn ensure_ts_query(url: &str) -> String {
    // Look for ts=1 as an actual query parameter, not just a substring.
    // The naive `url.contains("ts=1")` matches `?footsie=1` etc., which
    // would silently skip the auto-append for an oddly-named foreign
    // parameter and we'd end up without the per-frame epoch-ns header.
    if let Some((_, query)) = url.split_once('?') {
        for pair in query.split('&') {
            if pair == "ts=1" || pair.starts_with("ts=1#") {
                return url.to_string();
            }
        }
    }
    let sep = if url.contains('?') { '&' } else { '?' };
    format!("{url}{sep}ts=1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_ts_appends_when_no_query() {
        assert_eq!(
            ensure_ts_query("ws://audio-io:7010/mic"),
            "ws://audio-io:7010/mic?ts=1"
        );
    }

    #[test]
    fn ensure_ts_appends_with_amp_when_query_present() {
        assert_eq!(
            ensure_ts_query("ws://x/mic?foo=bar"),
            "ws://x/mic?foo=bar&ts=1"
        );
    }

    #[test]
    fn ensure_ts_idempotent() {
        let u = "ws://x/mic?ts=1";
        assert_eq!(ensure_ts_query(u), u);
    }

    #[test]
    fn ensure_ts_substring_false_positive_rejected() {
        // `ts=1` appears as a substring of `footsie=1` but is not the
        // actual `ts` parameter — we still need to append our own.
        assert_eq!(
            ensure_ts_query("ws://x/mic?footsie=1"),
            "ws://x/mic?footsie=1&ts=1"
        );
    }

    #[test]
    fn ensure_ts_among_other_params() {
        let u = "ws://x/mic?foo=bar&ts=1&baz=qux";
        assert_eq!(ensure_ts_query(u), u);
    }
}
