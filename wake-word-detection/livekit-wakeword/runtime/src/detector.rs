//! Ingester/predictor task split: the ingester keeps a shared ring of the
//! latest audio window so a slow predict never stalls the WS drain; the
//! predictor ticks on wallclock (ticks coalesce via MissedTickBehavior::Skip)
//! and runs the sync model via spawn_blocking. The model mutex is
//! single-locker, existing only to satisfy `Send`/`'static` for the spawn.

use std::collections::{BTreeMap, VecDeque};
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

/// `latest_end_epoch_ns` mirrors the newest frame's stamp so the predictor
/// can compute lag off a snapshot without holding the lock.
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
    // ONNX init parses several MB and takes hundreds of ms — keep it off
    // the runtime worker.
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
        confirm_count = cfg.confirm_count,
        confirm_window_ms = cfg.confirm_window.as_millis() as u64,
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

    let mut h_ingest = tokio::spawn(ingest(rx, Arc::clone(&ring)));
    let mut h_predict = tokio::spawn(predict_loop(
        cfg.clone(),
        Arc::clone(&ring),
        Arc::clone(&model),
        tx,
    ));

    // Either task ending is a shutdown signal. Dropping a JoinHandle does
    // NOT cancel the tokio task, so after select! we explicitly abort +
    // await the loser — otherwise it would orphan (ingester holding the mpsc
    // receiver, or a predictor pinning a spawn_blocking worker thread).
    tokio::select! {
        r = &mut h_ingest => {
            h_predict.abort();
            let _ = (&mut h_predict).await;
            match r {
                Ok(()) => Ok(()),
                Err(e) => Err(anyhow!("ingest task panicked: {e}")),
            }
        },
        r = &mut h_predict => {
            h_ingest.abort();
            let _ = (&mut h_ingest).await;
            match r {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(e),
                Err(e) => Err(anyhow!("predict task panicked: {e}")),
            }
        },
    }
}

/// Drain the ws_client mpsc into the ring; mutex is never held across
/// `.await`. On `PoisonError` we recover via `into_inner()` — panicking here
/// would tear down `run()`'s select! including the ws_client reconnect loop.
async fn ingest(mut rx: mpsc::Receiver<MicFrame>, ring: Arc<std::sync::Mutex<Ring>>) {
    while let Some(frame) = rx.recv().await {
        let n = frame.samples.len();
        let frame_end = frame.end_epoch_ns;
        let ring_len = {
            let mut r = match ring.lock() {
                Ok(g) => g,
                Err(poisoned) => {
                    warn!("ring mutex poisoned; recovering inner state");
                    poisoned.into_inner()
                }
            };
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

/// Anti-false-fire gate: confirms once `count` cooldown-deduplicated
/// detections land within `window`; confirming clears the history so the
/// next wake counts from zero. `count=1` confirms immediately.
struct ConfirmGate {
    count: u32,
    window: Duration,
    times: VecDeque<Instant>,
}

impl ConfirmGate {
    fn new(count: u32, window: Duration) -> Self {
        Self {
            count,
            window,
            times: VecDeque::new(),
        }
    }

    /// Returns `(confirmed, have)`; `have` counts in-window detections
    /// INCLUDING this one, taken before the on-confirm history clear.
    fn record(&mut self, now: Instant) -> (bool, usize) {
        self.times.push_back(now);
        while self
            .times
            .front()
            .map(|t| now.duration_since(*t) > self.window)
            .unwrap_or(false)
        {
            self.times.pop_front();
        }
        let have = self.times.len();
        let confirmed = have as u32 >= self.count;
        if confirmed {
            self.times.clear();
        }
        (confirmed, have)
    }
}

/// A slow predict only delays its own next tick (`MissedTickBehavior::Skip`);
/// the ingester keeps draining the WS the whole time.
async fn predict_loop(
    cfg: Config,
    ring: Arc<std::sync::Mutex<Ring>>,
    model: Arc<std::sync::Mutex<WakeWordModel>>,
    tx: mpsc::Sender<Detection>,
) -> Result<()> {
    let mut tick = interval(Duration::from_millis(cfg.predict_interval_ms as u64));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut last_fire: Option<Instant> = None;
    let mut confirm = ConfirmGate::new(cfg.confirm_count, cfg.confirm_window);
    let mut peak_score: f32 = 0.0;
    let mut peak_model: String = String::new();
    let mut last_peak_log = Instant::now();

    // Snapshot buffer round-trips through spawn_blocking each tick so the
    // allocation is reused (~64 KB at 10 Hz → avoids ~640 KB/s of churn).
    let cap = (cfg.predict_window_ms as usize) * (SAMPLE_RATE_HZ as usize) / 1000;
    let mut snapshot_buf: Vec<i16> = Vec::with_capacity(cap);

    loop {
        tick.tick().await;

        // Skip until the ring is full — the model returns all-zero scores
        // for windows under ~2 s, so predicting during warm-up is pure overhead.
        let window_end_epoch_ns = {
            let r = match ring.lock() {
                Ok(g) => g,
                Err(poisoned) => {
                    warn!("ring mutex poisoned; recovering inner state");
                    poisoned.into_inner()
                }
            };
            if r.samples.len() < r.capacity {
                continue;
            }
            snapshot_buf.clear();
            snapshot_buf.extend(r.samples.iter().copied());
            r.latest_end_epoch_ns
        };
        let snapshot = std::mem::replace(&mut snapshot_buf, Vec::with_capacity(cap));

        let audio_lag_ms = ns_diff_ms(epoch_ns_now(), window_end_epoch_ns);
        debug!(
            window_samples = snapshot.len(),
            window_end_epoch_ns = window_end_epoch_ns,
            audio_lag_ms = audio_lag_ms,
            "predict START"
        );
        let predict_started = Instant::now();

        let model_clone = Arc::clone(&model);
        let join = tokio::task::spawn_blocking(move || {
            let result = {
                // Recover from PoisonError — the mutex is single-locker, so
                // poison can only come from a prior predict() panic; retrying
                // is safe and recurring failures hit the warn! path below.
                let mut m = match model_clone.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
                m.predict(&snapshot)
            };
            (result, snapshot)
        })
        .await;
        let scores: BTreeMap<String, f32> = match join {
            Ok((Ok(s), buf)) => {
                snapshot_buf = buf;
                s
            }
            Ok((Err(e), buf)) => {
                snapshot_buf = buf;
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
                // A false positive rarely repeats within a few seconds, so
                // e.g. 2-within-3s suppresses spurious fires.
                let (confirmed, have) = confirm.record(now);
                info!(
                    model = %name,
                    score,
                    have,
                    need = cfg.confirm_count,
                    confirmed,
                    "wake detected"
                );
                if confirmed {
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

/// Signed `a - b` in ms; negative means the frame timestamp is ahead of the
/// local clock (cross-host skew, expected small in compose).
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

    #[test]
    fn confirm_count_one_forwards_immediately() {
        let mut gate = ConfirmGate::new(1, Duration::from_millis(3000));
        let t0 = Instant::now();
        assert_eq!(gate.record(t0), (true, 1));
        assert_eq!(gate.record(t0 + Duration::from_millis(10_000)), (true, 1));
    }

    #[test]
    fn confirm_two_within_window_then_resets() {
        let mut gate = ConfirmGate::new(2, Duration::from_millis(3000));
        let t0 = Instant::now();
        assert_eq!(gate.record(t0), (false, 1));
        assert_eq!(gate.record(t0 + Duration::from_millis(2000)), (true, 2));
        // Confirm cleared history → lone detection does not re-confirm,
        // and a follow-up outside the window still leaves only one in-window.
        assert_eq!(gate.record(t0 + Duration::from_millis(5000)), (false, 1));
        assert_eq!(gate.record(t0 + Duration::from_millis(9000)), (false, 1));
    }
}
