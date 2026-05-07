//! Two-task split: an *ingester* keeps a shared ring buffer of the
//! latest `WW_PREDICT_WINDOW_MS` of audio always up to date, and a
//! *predictor* fires on a wallclock timer (`WW_PREDICT_INTERVAL_MS`,
//! default 100 ms), snapshots the ring, and runs inference. Both
//! tasks are independent, so a slow predict no longer stalls the WS
//! drain — frames keep flowing into the ring while predict is busy.
//!
//! Skip semantics:
//!   * Predict still running when the next tick fires → tick is
//!     coalesced via `MissedTickBehavior::Skip` (the predictor is
//!     a single task, so two predicts can never overlap by
//!     construction).
//!   * Ring not yet full (warm-up under `predict_window_ms`) → the
//!     predictor returns to its tick loop without invoking the model.
//!
//! `WakeWordModel::predict` requires `&mut self` and is sync, so the
//! call still goes through `tokio::task::spawn_blocking`. The model
//! mutex is single-locker (only the predictor task takes it) so it
//! exists purely to satisfy `Send`/`'static` for the spawn.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, info, trace, warn};

use crate::config::Config;
use crate::wakeword::WakeWordModel;
use crate::{Detection, MicFrame};

/// audio-io always emits at this rate; the model is configured to match.
const SAMPLE_RATE_HZ: u32 = 16_000;

/// Shared rolling buffer. `latest_end_epoch_ns` mirrors the most
/// recently appended frame's stamp so the predictor can compute lag
/// off a snapshot without holding the lock.
struct Ring {
    samples: VecDeque<i16>,
    capacity: usize,
    latest_end_epoch_ns: u64,
}

pub async fn run(
    cfg: Config,
    rx: mpsc::Receiver<MicFrame>,
    tx: mpsc::Sender<Detection>,
    model_loaded: Arc<AtomicBool>,
) -> Result<()> {
    // Build the model on a blocking thread — ONNX init parses several
    // MB of mel/embedding/classifier bytes, taking hundreds of ms. The
    // runtime worker shouldn't block on it.
    let classifier_paths = cfg.model_paths.clone();
    let mel_path = cfg.mel_onnx_path.clone();
    let emb_path = cfg.embedding_onnx_path.clone();
    let model = tokio::task::spawn_blocking(move || {
        WakeWordModel::new(&mel_path, &emb_path, &classifier_paths)
    })
    .await?
    .map_err(|e| anyhow!("WakeWordModel::new failed: {e:?}"))?;
    let model = Arc::new(std::sync::Mutex::new(model));
    model_loaded.store(true, Ordering::Relaxed);
    info!(
        model_paths = ?cfg.model_paths,
        threshold = cfg.threshold,
        cooldown_ms = cfg.cooldown.as_millis() as u64,
        window_ms = cfg.predict_window_ms,
        interval_ms = cfg.predict_interval_ms,
        "model loaded"
    );

    let capacity = (cfg.predict_window_ms as usize) * (SAMPLE_RATE_HZ as usize) / 1000;
    let ring = Arc::new(std::sync::Mutex::new(Ring {
        samples: VecDeque::with_capacity(capacity),
        capacity,
        latest_end_epoch_ns: 0,
    }));

    let h_ingest = tokio::spawn(ingest(rx, Arc::clone(&ring)));
    let h_predict = tokio::spawn(predict_loop(
        cfg.clone(),
        Arc::clone(&ring),
        Arc::clone(&model),
        tx,
    ));

    // Either task ending means we're done. The ingester only exits on
    // ws_client dropping its sender (shutdown), and the predictor only
    // exits on event_sink dropping its receiver — both are shutdown
    // signals, so propagating the first one out is correct.
    tokio::select! {
        r = h_ingest => match r {
            Ok(()) => Ok(()),
            Err(e) => Err(anyhow!("ingest task panicked: {e}")),
        },
        r = h_predict => match r {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(anyhow!("predict task panicked: {e}")),
        },
    }
}

/// Drain the ws_client mpsc into the shared ring. Drops oldest
/// samples to keep the ring at exactly `capacity`. Holds the mutex
/// only for the push/pop — never across `.await`.
async fn ingest(mut rx: mpsc::Receiver<MicFrame>, ring: Arc<std::sync::Mutex<Ring>>) {
    while let Some(frame) = rx.recv().await {
        let n = frame.samples.len();
        let frame_end = frame.end_epoch_ns;
        let ring_len = {
            let mut r = ring.lock().expect("ring poisoned");
            let overflow = r.samples.len() + n;
            if overflow > r.capacity {
                for _ in 0..(overflow - r.capacity) {
                    r.samples.pop_front();
                }
            }
            r.samples.extend(frame.samples);
            r.latest_end_epoch_ns = frame_end;
            r.samples.len()
        };
        let audio_lag_ms = ns_diff_ms(epoch_ns_now(), frame_end);
        trace!(
            chunk_samples = n,
            ring_len = ring_len,
            frame_end_epoch_ns = frame_end,
            audio_lag_ms = audio_lag_ms,
            "chunk consumed"
        );
    }
}

/// Wallclock-driven predict loop. Snapshots the ring, runs predict
/// off-thread, then handles the cooldown / threshold / peak-log
/// bookkeeping. A slow predict only delays *its own* next tick (via
/// `MissedTickBehavior::Skip`); the ingester keeps draining the WS
/// the whole time.
async fn predict_loop(
    cfg: Config,
    ring: Arc<std::sync::Mutex<Ring>>,
    model: Arc<std::sync::Mutex<WakeWordModel>>,
    tx: mpsc::Sender<Detection>,
) -> Result<()> {
    let mut tick = interval(Duration::from_millis(cfg.predict_interval_ms as u64));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut last_fire: Option<Instant> = None;
    let mut peak_score: f32 = 0.0;
    let mut peak_model: String = String::new();
    let mut last_peak_log = Instant::now();

    loop {
        tick.tick().await;

        // Snapshot under brief lock. Skip if the ring hasn't reached
        // the configured window yet — the model returns all-zero
        // scores below ~2 s, so calling predict() during warm-up is
        // pure overhead.
        let (snapshot, window_end_epoch_ns) = {
            let r = ring.lock().expect("ring poisoned");
            if r.samples.len() < r.capacity {
                continue;
            }
            (
                r.samples.iter().copied().collect::<Vec<i16>>(),
                r.latest_end_epoch_ns,
            )
        };

        let audio_lag_ms = ns_diff_ms(epoch_ns_now(), window_end_epoch_ns);
        debug!(
            window_samples = snapshot.len(),
            window_end_epoch_ns = window_end_epoch_ns,
            audio_lag_ms = audio_lag_ms,
            "predict START"
        );
        let predict_started = Instant::now();

        let model_clone = Arc::clone(&model);
        let scores: HashMap<String, f32> =
            match tokio::task::spawn_blocking(move || {
                let mut m = model_clone.lock().expect("model mutex poisoned");
                m.predict(&snapshot)
            })
            .await
            {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => {
                    warn!("predict failed: {e:?}");
                    continue;
                }
                Err(e) => {
                    warn!("predict join error: {e}");
                    continue;
                }
            };

        let predict_dur_ms = predict_started.elapsed().as_millis() as u64;
        let e2e_lag_ms = ns_diff_ms(epoch_ns_now(), window_end_epoch_ns);
        debug!(
            duration_ms = predict_dur_ms,
            e2e_lag_ms = e2e_lag_ms,
            scores = ?scores,
            "predict DONE"
        );

        let now = Instant::now();
        let in_cooldown = last_fire
            .map(|t| now.duration_since(t) < cfg.cooldown)
            .unwrap_or(false);

        let best = scores.iter().max_by(|a, b| {
            a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some((name, &score)) = best {
            if !in_cooldown && score >= cfg.threshold {
                last_fire = Some(now);
                let det = Detection {
                    model: name.clone(),
                    score,
                    ts: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs_f64())
                        .unwrap_or(0.0),
                };
                if tx.send(det).await.is_err() {
                    return Ok(());
                }
            }

            if let Some(interval_dur) = cfg.peak_log_interval {
                update_peak(&mut peak_score, &mut peak_model, name, score);
                if now.duration_since(last_peak_log) >= interval_dur {
                    if peak_score >= cfg.peak_log_floor {
                        info!(
                            interval_sec = interval_dur.as_secs_f32(),
                            model = %peak_model,
                            score = peak_score,
                            threshold = cfg.threshold,
                            "peak score in window"
                        );
                    }
                    peak_score = 0.0;
                    peak_model.clear();
                    last_peak_log = now;
                }
            }
        }
    }
}

fn epoch_ns_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Saturating signed difference in milliseconds: `a - b`. Negative
/// when the frame timestamp is in the future relative to local clock
/// (clock skew between hosts, expected to be small in compose).
fn ns_diff_ms(a: u64, b: u64) -> i64 {
    if a >= b {
        ((a - b) / 1_000_000) as i64
    } else {
        -(((b - a) / 1_000_000) as i64)
    }
}

fn update_peak(peak_score: &mut f32, peak_model: &mut String, name: &str, score: f32) {
    if score > *peak_score {
        *peak_score = score;
        peak_model.clear();
        peak_model.push_str(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_peak_replaces_on_higher_score() {
        let mut score = 0.1;
        let mut model = String::from("a");
        update_peak(&mut score, &mut model, "b", 0.3);
        assert_eq!(score, 0.3);
        assert_eq!(model, "b");
    }

    #[test]
    fn update_peak_keeps_higher_existing() {
        let mut score = 0.5;
        let mut model = String::from("a");
        update_peak(&mut score, &mut model, "b", 0.2);
        assert_eq!(score, 0.5);
        assert_eq!(model, "a");
    }

    #[test]
    fn cooldown_predicate() {
        let now = Instant::now();
        let last = now - Duration::from_millis(500);
        assert!(now.duration_since(last) < Duration::from_secs(2));
        assert!(!(now.duration_since(last) < Duration::from_millis(100)));
    }
}
