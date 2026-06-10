use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SizedSample, StreamConfig};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use std::time::Instant;
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{debug, error, info, warn};

use crate::config::AudioConfig;
use crate::error::AudioError;
use crate::framer::{CaptureFramer, PlaybackFramer};
use crate::state::FlushSignals;

/// `Eos` is the drain handshake: `drain_done` fires once the cpal ring is
/// actually empty, so the upstream knows the last sample was consumed instead
/// of guessing with a tail timeout. Eos is per-message, not a permanent
/// terminator — frames queued after it are processed normally.
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
    stats: Arc<UnderrunStats>,
    native_rate: u32,
    native_channels: u16,
}

impl PlaybackHandle {
    pub fn sender(&self) -> mpsc::Sender<PlaybackMessage> {
        self.spk_tx.clone()
    }

    pub fn stats(&self) -> Arc<UnderrunStats> {
        self.stats.clone()
    }

    pub fn native_rate(&self) -> u32 {
        self.native_rate
    }

    pub fn native_channels(&self) -> u16 {
        self.native_channels
    }
}

impl Drop for PlaybackHandle {
    fn drop(&mut self) {
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

pub struct UnderrunStats {
    samples: AtomicU64,
    callbacks: AtomicU64,
    /// Set by the producer on every Frame; the logger swap-clears it each tick
    /// and warns only when a sender was actively pushing. Without this gate,
    /// cpal pulling silence from the empty ring while no client is connected
    /// would warn continuously.
    audio_seen: AtomicBool,
    /// Total samples consumed (incl. silence fallback), hardware-clock paced.
    consumed: AtomicU64,
    callbacks_total: AtomicU64,
}

impl UnderrunStats {
    fn new() -> Self {
        Self {
            samples: AtomicU64::new(0),
            callbacks: AtomicU64::new(0),
            audio_seen: AtomicBool::new(false),
            consumed: AtomicU64::new(0),
            callbacks_total: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.callbacks_total.load(Ordering::Relaxed),
            self.consumed.load(Ordering::Relaxed),
        )
    }
}

struct PlaybackReady {
    producer: HeapProd<f32>,
    native_rate: u32,
    native_channels: u16,
}

pub fn start_playback(
    track_id: usize,
    device_name: &str,
    audio: AudioConfig,
    buffer_ms: u32,
    flush: Arc<FlushSignals>,
    ref_tap: Option<broadcast::Sender<(usize, Bytes)>>,
) -> Result<PlaybackHandle> {
    let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<PlaybackReady>>(1);
    let (shutdown_tx, shutdown_rx) = std_mpsc::sync_channel::<()>(1);
    let device_name = device_name.to_string();
    let flush_cb = flush.clone();
    let stats = Arc::new(UnderrunStats::new());
    let stats_cb = stats.clone();
    let audio_cb = audio.clone();

    let thread = std::thread::Builder::new()
        .name(format!("audio-playback-{track_id}"))
        .spawn(move || {
            let ready_tx_clone = ready_tx.clone();
            if let Err(e) = run_playback(
                track_id,
                &device_name,
                audio_cb,
                buffer_ms,
                flush_cb,
                stats_cb,
                ref_tap,
                ready_tx,
                shutdown_rx,
            ) {
                error!(track_id, "playback thread error: {e:?}");
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

    let native_rate = ready.native_rate;
    let native_channels = ready.native_channels;

    // 32 frames ≈ 640 ms at 20 ms/frame: backpressure hits the WebSocket well
    // before a multi-second backlog can accumulate (important for barge-in).
    let (spk_tx, spk_rx) = mpsc::channel::<PlaybackMessage>(32);
    let task = tokio::spawn(playback_producer_task(
        spk_rx,
        ready.producer,
        audio.sample_rate,
        native_rate,
        native_channels,
        flush,
        stats.clone(),
    ));

    // Periodic underrun reporter: polling atomics here keeps the cpal audio
    // thread down to a single fetch_add per affected callback.
    let stats_log = stats.clone();
    let logger_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(500));
        interval.tick().await; // immediate first tick, skip
        let mut last_samples: u64 = 0;
        let mut last_callbacks: u64 = 0;
        loop {
            interval.tick().await;
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
            // Always advance the marks so an idle window's skipped underruns
            // don't leak into the next active window's delta.
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
        stats,
        native_rate,
        native_channels,
    })
}

fn find_output_device(name: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();
    if name == "default" {
        return host
            .default_output_device()
            .ok_or_else(|| AudioError::NoDefaultDevice("output").into());
    }
    for dev in host.output_devices().context("listing output devices")? {
        if dev.name().unwrap_or_default() == name {
            return Ok(dev);
        }
    }
    Err(AudioError::DeviceNotFound(name.into()).into())
}

fn emit_ref(tx: &broadcast::Sender<(usize, Bytes)>, track_id: usize, frames: Vec<Vec<u8>>) {
    for frame in frames {
        let _ = tx.send((track_id, Bytes::from(frame)));
    }
}

fn build_ref_framer(
    ref_tap: &Option<broadcast::Sender<(usize, Bytes)>>,
    native_rate: u32,
    native_channels: u16,
    audio: &AudioConfig,
) -> Result<Option<CaptureFramer>> {
    if ref_tap.is_some() {
        Ok(Some(CaptureFramer::new(
            native_rate,
            native_channels,
            audio.sample_rate,
            audio.samples_per_frame(),
        )?))
    } else {
        Ok(None)
    }
}

/// Keeps the *exact* historical per-format scaling so device bytes stay
/// bit-for-bit identical. cpal's `FromSample` would differ subtly: ×32768 +
/// rounding vs the historical ×32767 + truncation, and it clamps the f32 path
/// that previously passed through unclamped.
trait PlaybackSample: SizedSample + Send + 'static {
    fn from_playback_f32(v: f32) -> Self;
}
impl PlaybackSample for f32 {
    #[inline]
    fn from_playback_f32(v: f32) -> f32 {
        v
    }
}
impl PlaybackSample for i16 {
    #[inline]
    fn from_playback_f32(v: f32) -> i16 {
        (v.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}
impl PlaybackSample for u16 {
    #[inline]
    fn from_playback_f32(v: f32) -> u16 {
        let scaled = (v.clamp(-1.0, 1.0) * 32767.0) as i32 + 32768;
        scaled.clamp(0, u16::MAX as i32) as u16
    }
}

#[allow(clippy::too_many_arguments)]
fn build_output_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    mut consumer: HeapCons<f32>,
    flush: Arc<FlushSignals>,
    stats: Arc<UnderrunStats>,
    mut ref_framer: Option<CaptureFramer>,
    ref_tap: Option<broadcast::Sender<(usize, Bytes)>>,
    track_id: usize,
) -> Result<cpal::Stream>
where
    T: PlaybackSample,
{
    let err_fn = |e| error!("cpal output stream error: {e}");
    // Reused across callbacks (grows once) — no per-callback allocation on the
    // audio thread.
    let mut tap_scratch: Vec<f32> = Vec::new();
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            if flush.consumer.swap(false, Ordering::Relaxed) {
                while consumer.try_pop().is_some() {}
            }
            let tapping = ref_framer.is_some();
            if tapping {
                tap_scratch.clear();
            }
            let mut underrun: u64 = 0;
            for slot in data.iter_mut() {
                let v = match consumer.try_pop() {
                    Some(s) => s,
                    None => {
                        underrun += 1;
                        0.0
                    }
                };
                *slot = T::from_playback_f32(v);
                if tapping {
                    tap_scratch.push(v);
                }
            }
            if underrun > 0 {
                stats.samples.fetch_add(underrun, Ordering::Relaxed);
                stats.callbacks.fetch_add(1, Ordering::Relaxed);
            }
            stats.consumed.fetch_add(data.len() as u64, Ordering::Relaxed);
            stats.callbacks_total.fetch_add(1, Ordering::Relaxed);
            // Tap the exact PCM handed to the device as the AEC far-end.
            if let (Some(framer), Some(tx)) = (ref_framer.as_mut(), ref_tap.as_ref()) {
                emit_ref(tx, track_id, framer.push_f32(&tap_scratch));
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

#[allow(clippy::too_many_arguments)]
fn run_playback(
    track_id: usize,
    device_name: &str,
    audio: AudioConfig,
    buffer_ms: u32,
    flush: Arc<FlushSignals>,
    stats: Arc<UnderrunStats>,
    ref_tap: Option<broadcast::Sender<(usize, Bytes)>>,
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
        track_id,
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

    let stream = match sample_format {
        SampleFormat::F32 => build_output_stream::<f32>(
            &device,
            &stream_config,
            consumer,
            flush.clone(),
            stats.clone(),
            build_ref_framer(&ref_tap, native_rate, native_channels, &audio)?,
            ref_tap.clone(),
            track_id,
        )?,
        SampleFormat::I16 => build_output_stream::<i16>(
            &device,
            &stream_config,
            consumer,
            flush.clone(),
            stats.clone(),
            build_ref_framer(&ref_tap, native_rate, native_channels, &audio)?,
            ref_tap.clone(),
            track_id,
        )?,
        SampleFormat::U16 => build_output_stream::<u16>(
            &device,
            &stream_config,
            consumer,
            flush.clone(),
            stats.clone(),
            build_ref_framer(&ref_tap, native_rate, native_channels, &audio)?,
            ref_tap.clone(),
            track_id,
        )?,
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
    // Idle silence keep-alive: top up the ring with 0.0 so the OS audio
    // pipeline (WASAPI prefetch / ALSA period buffer) stays warm — without it,
    // the first sentence of second-and-later turns was head-clipped after the
    // pipeline went cold during inter-turn idle. 50 ms absorbs the ~10 ms cpal
    // callback jitter without delaying real audio.
    let keep_alive_threshold = (native_rate as usize) * (native_channels as usize) * 50 / 1000;
    let mut total_dropped: u64 = 0;
    let mut drops_since_log: u32 = 0;
    loop {
        tokio::select! {
            // `biased`: real audio strictly preferred over the keep-alive's silence.
            biased;
            recv = spk_rx.recv() => {
                let Some(msg) = recv else { break; };
                if flush.producer.swap(false, Ordering::Relaxed) {
                    // Barge-in: drain everything queued. In-flight Eos requests
                    // get drain_done fired immediately — the cancel itself is
                    // the "no more audio is coming" signal upstream waits for.
                    // When this fires at the start of a new utterance (a
                    // turn-start /spk/stop racing the new burst), the drained
                    // frames ARE the clipped head; approx_ms is how much.
                    let mut drained_frames: u32 = 0;
                    let mut drained_bytes: usize = 0;
                    if let PlaybackMessage::Frame(ref b) = msg {
                        drained_frames += 1;
                        drained_bytes += b.len();
                    }
                    while let Ok(pending) = spk_rx.try_recv() {
                        match pending {
                            PlaybackMessage::Frame(b) => {
                                drained_frames += 1;
                                drained_bytes += b.len();
                            }
                            PlaybackMessage::Eos { drain_done } => {
                                let _ = drain_done.send(());
                            }
                        }
                    }
                    if drained_frames > 0 {
                        let bytes_per_ms = (source_rate as usize * 2 / 1000).max(1);
                        debug!(
                            drained_frames,
                            drained_bytes,
                            approx_ms = drained_bytes / bytes_per_ms,
                            "playback flush discarded queued frames (/spk/stop); at an utterance start this is the clipped head"
                        );
                    }
                    framer.flush();
                    if let PlaybackMessage::Eos { drain_done } = msg {
                        let _ = drain_done.send(());
                    }
                    continue;
                }
                match msg {
                    PlaybackMessage::Frame(bytes) => {
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
                            total_dropped =
                                total_dropped.saturating_add(dropped_this_batch as u64);
                            debug!(
                                dropped_this_batch,
                                total_dropped,
                                "playback ring full; dropping samples (WS arriving faster than cpal consumes)"
                            );
                            drops_since_log += 1;
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
                        // Once tokio picks this select branch it polls *only*
                        // this future until Ready — the inner sleep is a yield
                        // within the same future, not a re-entry into the
                        // select, so the idle keep-alive arm cannot race
                        // silence top-ups into the ring being drained. A flush
                        // mid-wait counts as drained.
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
                        // The keep-alive will refill ~50 ms of silence next
                        // iteration; harmless filler — the next turn's
                        // /spk/stop flush erases it before real audio, so no
                        // head-clip risk.
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                let depth = ring_capacity - producer.vacant_len();
                if depth < keep_alive_threshold {
                    let need = keep_alive_threshold - depth;
                    for _ in 0..need {
                        if producer.try_push(0.0).is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }
    info!("playback producer task exiting");
}
