use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use crate::config::AudioConfig;
use crate::framer::PlaybackFramer;
use crate::state::FlushSignals;

/// One unit of work for the playback producer task.
///
/// `Frame` is the existing path: a 20 ms s16le PCM payload to push at
/// the cpal output ring. `Eos` is the drain-handshake added so the
/// upstream (tts-streamer over the /spk WS) can know exactly when the
/// last sample has actually been consumed by the device — instead of
/// guessing with a tail timeout. The producer task keeps draining the
/// ring after Eos arrives and then fires `drain_done` so the WS
/// handler can echo `{"type":"drained"}` back at the client. Frames
/// queued *after* Eos (e.g. a barge-in starting a new utterance
/// immediately) are processed normally — Eos is per-message, not a
/// permanent terminator.
pub enum PlaybackMessage {
    Frame(Bytes),
    Eos { drain_done: oneshot::Sender<()> },
}

pub struct PlaybackHandle {
    thread_shutdown: std_mpsc::SyncSender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
    task: Option<tokio::task::JoinHandle<()>>,
    logger_task: Option<tokio::task::JoinHandle<()>>,
    spk_tx: mpsc::Sender<PlaybackMessage>,
}

impl PlaybackHandle {
    pub fn sender(&self) -> mpsc::Sender<PlaybackMessage> {
        self.spk_tx.clone()
    }
}

impl Drop for PlaybackHandle {
    fn drop(&mut self) {
        // Cancel producer task first so it stops pushing into the ring buffer.
        if let Some(t) = self.task.take() {
            t.abort();
        }
        if let Some(t) = self.logger_task.take() {
            t.abort();
        }
        let _ = self.thread_shutdown.try_send(());
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

/// Counters incremented by the cpal output callback every time the
/// playback ring is empty and the `unwrap_or(0.0)` silence fallback
/// fires. Read by a periodic logger task — direct logging from the
/// audio thread risks blocking on tracing's lock and missing the
/// callback's deadline (= further underruns).
struct UnderrunStats {
    /// Total number of zero-sample emissions across all callbacks.
    samples: AtomicU64,
    /// Number of callbacks that hit the silence path at least once.
    callbacks: AtomicU64,
    /// Set true by the producer task on every Frame received from the
    /// /spk WS. The logger task swap-clears this each tick and only
    /// emits a warn when the flag was true — i.e., when a sender was
    /// actively pushing audio in the just-elapsed window. Suppresses
    /// the steady stream of "ring empty" warns that would otherwise
    /// fire continuously while no client is connected (cpal keeps
    /// running and pulling 0.0 silence from the empty ring).
    audio_seen: AtomicBool,
}

impl UnderrunStats {
    fn new() -> Self {
        Self {
            samples: AtomicU64::new(0),
            callbacks: AtomicU64::new(0),
            audio_seen: AtomicBool::new(false),
        }
    }
}

struct PlaybackReady {
    producer: HeapProd<f32>,
    native_rate: u32,
    native_channels: u16,
}

pub fn start_playback(
    device_name: &str,
    audio: AudioConfig,
    buffer_ms: u32,
    flush: Arc<FlushSignals>,
) -> Result<PlaybackHandle> {
    let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<PlaybackReady>>(1);
    let (shutdown_tx, shutdown_rx) = std_mpsc::sync_channel::<()>(1);
    let device_name = device_name.to_string();
    let flush_cb = flush.clone();
    let stats = Arc::new(UnderrunStats::new());
    let stats_cb = stats.clone();

    let thread = std::thread::Builder::new()
        .name("audio-playback".into())
        .spawn(move || {
            let ready_tx_clone = ready_tx.clone();
            if let Err(e) = run_playback(
                &device_name,
                buffer_ms,
                flush_cb,
                stats_cb,
                ready_tx,
                shutdown_rx,
            ) {
                error!("playback thread error: {e:?}");
                let _ = ready_tx_clone.send(Err(e));
            }
        })?;

    let ready = match ready_rx.recv() {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            let _ = thread.join();
            return Err(e);
        }
        Err(_) => {
            let _ = thread.join();
            return Err(anyhow!("playback thread exited before ready"));
        }
    };

    // 32 frames ≈ 640 ms at 20 ms/frame — a small multiple of the
    // playback ring so backpressure hits the WebSocket well before a
    // multi-second backlog can accumulate (important for barge-in).
    let (spk_tx, spk_rx) = mpsc::channel::<PlaybackMessage>(32);
    let task = tokio::spawn(playback_producer_task(
        spk_rx,
        ready.producer,
        audio.sample_rate,
        ready.native_rate,
        ready.native_channels,
        flush,
        stats.clone(),
    ));

    // Periodic underrun reporter. Polls the atomics every 500 ms and
    // emits a warn line whenever the sample count grew. Lives on the
    // tokio runtime alongside the producer task so it gets aborted
    // automatically when PlaybackHandle is dropped (= /stop or process
    // exit), and does not slow down the cpal audio thread (which only
    // pays a single fetch_add per affected callback). Sized at 500 ms
    // so an operator's stack-trace mental model of "TTS spoke ~5 s
    // ago, did the ring underrun?" can be answered against ~10 log
    // lines worth of detail rather than a blow-by-blow.
    let stats_log = stats.clone();
    let logger_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(500));
        interval.tick().await; // immediate first tick, skip
        let mut last_samples: u64 = 0;
        let mut last_callbacks: u64 = 0;
        loop {
            interval.tick().await;
            // Swap-clear the activity flag: only emit a warn when a
            // sender actually pushed audio in the just-elapsed 500 ms.
            // Without this gate, cpal's normal "ring empty" behavior
            // during idle (no /spk client) fires the warn continuously.
            let active = stats_log.audio_seen.swap(false, Ordering::Relaxed);
            let s = stats_log.samples.load(Ordering::Relaxed);
            let c = stats_log.callbacks.load(Ordering::Relaxed);
            if active && s > last_samples {
                let delta_samples = s - last_samples;
                let delta_callbacks = c - last_callbacks;
                warn!(
                    delta_samples,
                    delta_callbacks,
                    total_samples = s,
                    total_callbacks = c,
                    "playback underrun: cpal got 0.0 fallback (ring drained — sender too slow OR audio-io behind)"
                );
            }
            // Always advance the high-water marks so that an idle
            // window doesn't make the next active window's delta
            // include the silently-skipped idle underruns.
            last_samples = s;
            last_callbacks = c;
        }
    });

    Ok(PlaybackHandle {
        thread_shutdown: shutdown_tx,
        thread: Some(thread),
        task: Some(task),
        logger_task: Some(logger_task),
        spk_tx,
    })
}

fn find_output_device(name: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();
    if name == "default" {
        return host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device"));
    }
    for dev in host.output_devices().context("listing output devices")? {
        if dev.name().unwrap_or_default() == name {
            return Ok(dev);
        }
    }
    Err(anyhow!("output device '{name}' not found"))
}

fn run_playback(
    device_name: &str,
    buffer_ms: u32,
    flush: Arc<FlushSignals>,
    stats: Arc<UnderrunStats>,
    ready_tx: std_mpsc::SyncSender<Result<PlaybackReady>>,
    shutdown_rx: std_mpsc::Receiver<()>,
) -> Result<()> {
    let device = find_output_device(device_name)?;
    let default_config = device
        .default_output_config()
        .context("default_output_config")?;
    let sample_format = default_config.sample_format();
    let native_rate = default_config.sample_rate().0;
    let native_channels = default_config.channels();
    let stream_config: StreamConfig = default_config.into();

    info!(
        device = ?device.name().ok(),
        native_rate,
        native_channels,
        ?sample_format,
        "opening playback stream"
    );

    let rb_capacity = (native_rate as usize
        * native_channels as usize
        * buffer_ms as usize
        / 1000)
        .max(1024);
    let rb = HeapRb::<f32>::new(rb_capacity);
    let (producer, consumer) = rb.split();

    ready_tx
        .send(Ok(PlaybackReady {
            producer,
            native_rate,
            native_channels,
        }))
        .map_err(|_| anyhow!("failed to signal playback ready"))?;

    let err_fn = |e| error!("cpal output stream error: {e}");
    let flush_cb = flush.clone();

    let stream = match sample_format {
        SampleFormat::F32 => {
            let mut consumer: HeapCons<f32> = consumer;
            let flush = flush_cb;
            let stats = stats.clone();
            device.build_output_stream(
                &stream_config,
                move |data: &mut [f32], _| {
                    if flush.consumer.swap(false, Ordering::Relaxed) {
                        while consumer.try_pop().is_some() {}
                    }
                    let mut underrun: u64 = 0;
                    for sample in data.iter_mut() {
                        match consumer.try_pop() {
                            Some(s) => *sample = s,
                            None => {
                                *sample = 0.0;
                                underrun += 1;
                            }
                        }
                    }
                    if underrun > 0 {
                        stats.samples.fetch_add(underrun, Ordering::Relaxed);
                        stats.callbacks.fetch_add(1, Ordering::Relaxed);
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let mut consumer: HeapCons<f32> = consumer;
            let flush = flush_cb;
            let stats = stats.clone();
            device.build_output_stream(
                &stream_config,
                move |data: &mut [i16], _| {
                    if flush.consumer.swap(false, Ordering::Relaxed) {
                        while consumer.try_pop().is_some() {}
                    }
                    let mut underrun: u64 = 0;
                    for sample in data.iter_mut() {
                        let v = match consumer.try_pop() {
                            Some(s) => s,
                            None => {
                                underrun += 1;
                                0.0
                            }
                        };
                        *sample = (v.clamp(-1.0, 1.0) * 32767.0) as i16;
                    }
                    if underrun > 0 {
                        stats.samples.fetch_add(underrun, Ordering::Relaxed);
                        stats.callbacks.fetch_add(1, Ordering::Relaxed);
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let mut consumer: HeapCons<f32> = consumer;
            let flush = flush_cb;
            let stats = stats.clone();
            device.build_output_stream(
                &stream_config,
                move |data: &mut [u16], _| {
                    if flush.consumer.swap(false, Ordering::Relaxed) {
                        while consumer.try_pop().is_some() {}
                    }
                    let mut underrun: u64 = 0;
                    for sample in data.iter_mut() {
                        let v = match consumer.try_pop() {
                            Some(s) => s,
                            None => {
                                underrun += 1;
                                0.0
                            }
                        };
                        let scaled = (v.clamp(-1.0, 1.0) * 32767.0) as i32 + 32768;
                        *sample = scaled.clamp(0, u16::MAX as i32) as u16;
                    }
                    if underrun > 0 {
                        stats.samples.fetch_add(underrun, Ordering::Relaxed);
                        stats.callbacks.fetch_add(1, Ordering::Relaxed);
                    }
                },
                err_fn,
                None,
            )?
        }
        other => anyhow::bail!("unsupported output sample format: {other:?}"),
    };

    stream.play()?;
    let _ = shutdown_rx.recv();
    drop(stream);
    info!("playback stopped");
    Ok(())
}

async fn playback_producer_task(
    mut spk_rx: mpsc::Receiver<PlaybackMessage>,
    mut producer: HeapProd<f32>,
    source_rate: u32,
    native_rate: u32,
    native_channels: u16,
    flush: Arc<FlushSignals>,
    stats: Arc<UnderrunStats>,
) {
    let mut framer = match PlaybackFramer::new(source_rate, native_rate, native_channels) {
        Ok(f) => f,
        Err(e) => {
            error!("failed to create PlaybackFramer: {e:?}");
            return;
        }
    };
    let ring_capacity = producer.capacity().get();
    let mut total_dropped: u64 = 0;
    let mut drops_since_log: u32 = 0;
    while let Some(msg) = spk_rx.recv().await {
        if flush.producer.swap(false, Ordering::Relaxed) {
            // Cancellation (barge-in) — drain everything queued and
            // reset the framer. Any in-flight Eos requests get their
            // drain_done fired immediately because the cancel itself
            // is the "no more audio is coming" signal that the
            // upstream is waiting for.
            while let Ok(pending) = spk_rx.try_recv() {
                if let PlaybackMessage::Eos { drain_done } = pending {
                    let _ = drain_done.send(());
                }
            }
            framer.flush();
            if let PlaybackMessage::Eos { drain_done } = msg {
                let _ = drain_done.send(());
            }
            continue;
        }
        match msg {
            PlaybackMessage::Frame(bytes) => {
                // Mark this tick as "audio flowing" so the underrun
                // logger emits warns gated on actual sender activity
                // instead of spamming while idle.
                stats.audio_seen.store(true, Ordering::Relaxed);
                let samples = framer.push_s16le(&bytes);
                let mut overflow = false;
                let mut dropped_this_batch: usize = 0;
                for s in samples {
                    if overflow {
                        dropped_this_batch += 1;
                        continue;
                    }
                    if producer.try_push(s).is_err() {
                        overflow = true;
                        dropped_this_batch += 1;
                    }
                }
                if dropped_this_batch > 0 {
                    total_dropped = total_dropped.saturating_add(dropped_this_batch as u64);
                    // Per-batch debug line so an operator running with
                    // `RUST_LOG=audio_io::playback=debug` (or just =debug)
                    // can confirm whether their WS sender is overpacing
                    // the cpal consumer — even a single dropped sample
                    // shows up here, which the rate-limited warn below
                    // hides until 50 batches have piled up.
                    debug!(
                        dropped_this_batch,
                        total_dropped,
                        "playback ring full; dropping samples (WS arriving faster than cpal consumes)"
                    );
                    drops_since_log += 1;
                    // Rate-limit: roughly once per ~1s at 20ms/frame.
                    if drops_since_log >= 50 {
                        warn!(
                            total_dropped,
                            "playback ring buffer full; dropping samples (consumer slower than producer)"
                        );
                        drops_since_log = 0;
                    }
                }
            }
            PlaybackMessage::Eos { drain_done } => {
                // Wait for the cpal output thread to consume every
                // sample we've pushed. Polling vacant_len from the
                // producer side reads how much room is available; when
                // it equals capacity, the ring is empty and the cpal
                // callback has just emitted (or is about to emit) its
                // last non-zero sample. 5 ms granularity is well below
                // a 20 ms frame so the upstream sees the drained
                // signal within ~one cpal callback of true silence.
                // A flush mid-wait is treated as "drained now" — the
                // cancel path drained the ring on our behalf.
                let drain_start = Instant::now();
                while producer.vacant_len() < ring_capacity {
                    if flush.producer.load(Ordering::Relaxed) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                let drain_ms = drain_start.elapsed().as_millis();
                info!(drain_ms, "playback ring drained, signaling client");
                let _ = drain_done.send(());
            }
        }
    }
    info!("playback producer task exiting");
}
