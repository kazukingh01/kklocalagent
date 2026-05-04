//! Ring buffer + predict loop. The crate's predict() needs ~2 s of
//! audio (returns all-zero scores below that), so we hold the last
//! `WW_PREDICT_WINDOW_MS` worth in a `VecDeque` and call predict every
//! 80 ms once warm. 80 ms is the model's own embedding stride —
//! calling more often is wasted CPU.
//!
//! `WakeWordModel::predict` requires `&mut self` and is sync, so the
//! call goes inside `tokio::task::spawn_blocking`. The mutex is
//! single-writer (this task is the only locker) so contention is zero.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use livekit_wakeword::WakeWordModel;
use tokio::sync::mpsc;
use tracing::{debug, info, trace, warn};

use crate::Detection;
use crate::config::Config;

/// audio-io always emits at this rate; the model is configured to match.
const SAMPLE_RATE_HZ: u32 = 16_000;

/// 80 ms hop. livekit-wakeword's embedding stride is also 80 ms, so
/// calling predict() more often than this is wasted work — the model
/// has nothing new to ingest.
const HOP_SAMPLES: usize = 1280;

pub async fn run(
    cfg: Config,
    mut rx: mpsc::Receiver<Vec<i16>>,
    tx: mpsc::Sender<Detection>,
    model_loaded: Arc<AtomicBool>,
) -> Result<()> {
    // Build the model on a blocking thread — ONNX init parses a few MB
    // of bundled mel/embedding bytes plus the classifier ONNX, taking
    // hundreds of ms. The runtime worker shouldn't block on it.
    let paths = cfg.model_paths.clone();
    let model = tokio::task::spawn_blocking(move || {
        WakeWordModel::new(&paths, SAMPLE_RATE_HZ)
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
        "model loaded"
    );

    let window_samples =
        (cfg.predict_window_ms as usize) * (SAMPLE_RATE_HZ as usize) / 1000;
    let mut ring: VecDeque<i16> = VecDeque::with_capacity(window_samples);
    let mut samples_since_predict: usize = 0;
    let mut last_fire: Option<Instant> = None;
    let mut peak_score: f32 = 0.0;
    let mut peak_model: String = String::new();
    let mut last_peak_log = Instant::now();

    // Diagnostic counters. `total_samples` is the cumulative count of
    // i16 samples received from the WS — it doubles as an "audio
    // timeline" (audio_t = total_samples / SAMPLE_RATE_HZ). Comparing
    // audio_t against wall-clock since first chunk reveals dropped
    // frames upstream (audio_t falls behind wall time).
    let mut total_samples: u64 = 0;
    let mut first_chunk_at: Option<Instant> = None;
    let mut predict_n: u64 = 0;

    while let Some(chunk) = rx.recv().await {
        let chunk_arrived = Instant::now();
        let n = chunk.len();
        if first_chunk_at.is_none() {
            first_chunk_at = Some(chunk_arrived);
        }
        total_samples += n as u64;

        let overflow = ring.len() + n;
        if overflow > window_samples {
            for _ in 0..(overflow - window_samples) {
                ring.pop_front();
            }
        }
        ring.extend(chunk);
        samples_since_predict += n;

        let audio_t = total_samples as f64 / SAMPLE_RATE_HZ as f64;
        let wall_t = first_chunk_at
            .map(|t| chunk_arrived.duration_since(t).as_secs_f64())
            .unwrap_or(0.0);
        trace!(
            chunk_samples = n,
            ring_len = ring.len(),
            total_samples = total_samples,
            samples_since_predict = samples_since_predict,
            audio_t_sec = audio_t,
            wall_t_sec = wall_t,
            audio_lag_ms = ((wall_t - audio_t) * 1000.0) as i64,
            "chunk consumed"
        );

        // Warm-up: predict() returns all zeros for windows < ~2 s, so
        // there is no point calling it before the ring is full.
        if ring.len() < window_samples {
            continue;
        }
        if samples_since_predict < HOP_SAMPLES {
            continue;
        }
        samples_since_predict = 0;

        let window_end_t = total_samples as f64 / SAMPLE_RATE_HZ as f64;
        let window_start_t = window_end_t - (ring.len() as f64) / (SAMPLE_RATE_HZ as f64);
        debug!(
            predict_n = predict_n,
            window_samples = ring.len(),
            audio_t_start_sec = window_start_t,
            audio_t_end_sec = window_end_t,
            wall_t_sec = wall_t,
            audio_lag_ms = ((wall_t - window_end_t) * 1000.0) as i64,
            "predict START"
        );
        let predict_started = Instant::now();

        let window_vec: Vec<i16> = ring.iter().copied().collect();
        let model_clone = Arc::clone(&model);
        let scores: HashMap<String, f32> =
            match tokio::task::spawn_blocking(move || {
                let mut m = model_clone.lock().expect("predict mutex poisoned");
                m.predict(&window_vec)
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
        debug!(
            predict_n = predict_n,
            duration_ms = predict_dur_ms,
            scores = ?scores,
            "predict DONE"
        );
        predict_n += 1;

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

            if let Some(interval) = cfg.peak_log_interval {
                update_peak(&mut peak_score, &mut peak_model, name, score);
                if now.duration_since(last_peak_log) >= interval {
                    if peak_score >= cfg.peak_log_floor {
                        info!(
                            interval_sec = interval.as_secs_f32(),
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
    Ok(())
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
    use std::time::Duration;

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
