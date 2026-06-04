use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use std::time::Instant;
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{debug, error, info, warn};

use crate::config::AudioConfig;
use crate::error::AudioError;
use crate::framer::{CaptureFramer, PlaybackFramer};
use crate::pcm::epoch_ns;
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

/// Counters incremented by the cpal output callback. The silence-path
/// counters (`samples`, `callbacks`) feed the periodic underrun logger;
/// `consumed` and `callbacks_total` track every callback regardless of
/// underrun and let an external observer (e.g. the /spk WS handler)
/// measure hardware-vs-system clock drift over a session window.
pub struct UnderrunStats {
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
    /// Total interleaved samples consumed by the cpal output callback,
    /// including silence-fallback samples. Driven by the hardware audio
    /// clock; comparing against system-clock elapsed time exposes
    /// drift between the DAC and the OS wall clock.
    consumed: AtomicU64,
    /// Total cpal output callbacks invoked (hardware-clock paced).
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

    /// Returns `(callbacks_total, consumed_samples)` snapshot. Both grow
    /// monotonically from playback start; subtract two snapshots to get
    /// session-scoped deltas.
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
    ref_tap: Option<broadcast::Sender<(usize, u64, Bytes)>>,
) -> Result<PlaybackHandle> {
    let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<PlaybackReady>>(1);
    let (shutdown_tx, shutdown_rx) = std_mpsc::sync_channel::<()>(1);
    let device_name = device_name.to_string();
    let flush_cb = flush.clone();
    let stats = Arc::new(UnderrunStats::new());
    let stats_cb = stats.clone();
    // The producer task (below) keeps `audio`; the cpal thread gets its own
    // clone for the consumption-side reference tap.
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

    // 32 frames ≈ 640 ms at 20 ms/frame — a small multiple of the
    // playback ring so backpressure hits the WebSocket well before a
    // multi-second backlog can accumulate (important for barge-in).
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

/// Emit the playback tap's downsampled reference frames to the AEC mixer.
/// `frames` are 16 kHz mono s16le (one AEC frame each); `frame_ns` back-dates
/// frames produced together in a single callback so each carries its own play
/// timestamp (mirrors the capture-side `dispatch`). Send failures (AEC mixer
/// not running) are ignored.
fn emit_ref(
    tx: &broadcast::Sender<(usize, u64, Bytes)>,
    track_id: usize,
    frames: Vec<Vec<u8>>,
    frame_ns: u64,
) {
    if frames.is_empty() {
        return;
    }
    let now_ns = epoch_ns();
    let n = frames.len();
    for (i, frame) in frames.into_iter().enumerate() {
        let end_ns = now_ns.saturating_sub(((n - 1 - i) as u64) * frame_ns);
        let _ = tx.send((track_id, end_ns, Bytes::from(frame)));
    }
}

/// Build the consumption-side reference resampler (native rate/channels →
/// 16 kHz mono) — but only when AEC is enabled (`ref_tap` present). When
/// disabled this is `None` and the output callback skips the tap entirely, so
/// playback is byte-for-byte identical to the no-AEC path.
fn build_ref_framer(
    ref_tap: &Option<broadcast::Sender<(usize, u64, Bytes)>>,
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

/// Build a cpal output stream for device sample type `T`. One generic body for
/// f32/i16/u16: pop f32 from the playback ring, convert to `T` via cpal, and —
/// when AEC is on — tap the pre-conversion f32 as the far-end reference. This
/// replaces three near-identical per-format callbacks (and the hand-written i16
/// / u16 scaling they each carried).
#[allow(clippy::too_many_arguments)]
fn build_output_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    mut consumer: HeapCons<f32>,
    flush: Arc<FlushSignals>,
    stats: Arc<UnderrunStats>,
    mut ref_framer: Option<CaptureFramer>,
    ref_tap: Option<broadcast::Sender<(usize, u64, Bytes)>>,
    track_id: usize,
    ref_frame_ns: u64,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32> + Send + 'static,
{
    let err_fn = |e| error!("cpal output stream error: {e}");
    // Reused across callbacks (grows once): the f32 samples for the AEC tap.
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
                *slot = T::from_sample(v.clamp(-1.0, 1.0));
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
                emit_ref(tx, track_id, framer.push_f32(&tap_scratch), ref_frame_ns);
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
    ref_tap: Option<broadcast::Sender<(usize, u64, Bytes)>>,
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

    // Far-end reference tap (issue #20). When AEC is enabled, the output
    // callback converts the native-rate, native-channel PCM the device consumes
    // down to the AEC's 16 kHz mono and hands it to `emit_ref`, timestamped at
    // consumption. `ref_frame_ns` is the duration of one emitted 16 kHz frame,
    // used to back-date batched frames.
    let ref_frame_ns: u64 =
        (audio.samples_per_frame() as u64 * 1_000_000_000) / audio.sample_rate.max(1) as u64;

    // One generic callback covers every device sample format (cpal converts f32
    // → T). `consumer`/`flush` are moved into whichever arm runs; the others are
    // dead branches, so moving the same value in each arm is fine.
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
            ref_frame_ns,
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
            ref_frame_ns,
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
            ref_frame_ns,
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
    // Idle silence keep-alive threshold. When no Frame messages are
    // arriving, top up the ring with 0.0 samples so the OS-level
    // audio pipeline (WASAPI prefetch / ALSA period buffer) stays warm
    // — without this, the first sentence of the second-and-later turns
    // was head-clipped because the pipeline had gone cold during the
    // inter-turn idle period. 50ms is enough margin to absorb the
    // ~10ms cpal callback jitter without delaying real audio (the
    // top-up only fires while ring_depth < threshold; real audio
    // pushes the depth far above this).
    let keep_alive_threshold = (native_rate as usize) * (native_channels as usize) * 50 / 1000;
    let mut total_dropped: u64 = 0;
    let mut drops_since_log: u32 = 0;
    loop {
        tokio::select! {
            // Frame / Eos / shutdown gets strictly preferred over the
            // idle keep-alive: if a Frame is ready, we'd rather push
            // real audio than synthetic silence. `biased` means tokio
            // polls the `recv` arm first every iteration before even
            // looking at the keep-alive sleep.
            biased;
            recv = spk_rx.recv() => {
                let Some(msg) = recv else { break; };
                if flush.producer.swap(false, Ordering::Relaxed) {
                    // Cancellation (barge-in) — drain everything queued
                    // and reset the framer. Any in-flight Eos requests
                    // get their drain_done fired immediately because
                    // the cancel itself is the "no more audio is
                    // coming" signal that the upstream is waiting for.
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
                        // Mark this tick as "audio flowing" so the
                        // underrun logger emits warns gated on actual
                        // sender activity instead of spamming while
                        // idle.
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
                            // Per-batch debug line so an operator
                            // running with `RUST_LOG=audio_io::playback=debug`
                            // (or just =debug) can confirm whether
                            // their WS sender is overpacing the cpal
                            // consumer — even a single dropped sample
                            // shows up here, which the rate-limited
                            // warn below hides until 50 batches have
                            // piled up.
                            debug!(
                                dropped_this_batch,
                                total_dropped,
                                "playback ring full; dropping samples (WS arriving faster than cpal consumes)"
                            );
                            drops_since_log += 1;
                            // Rate-limit: roughly once per ~1s at
                            // 20ms/frame.
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
                        // Wait for the cpal output thread to consume
                        // every real sample we've pushed. The Eos arm
                        // runs inside the select; once tokio picks
                        // this branch it polls *only* this future
                        // until it returns Poll::Ready — the inner
                        // `sleep(5ms).await` is a yield point within
                        // the same future, not a re-entry into the
                        // select, so the idle keep-alive arm is NOT
                        // polled and cannot race silence top-ups into
                        // the ring we're trying to drain to empty.
                        //
                        // A flush mid-wait is treated as "drained now"
                        // — the cancel path drained the ring on our
                        // behalf.
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
                        // Note: on the next loop iteration the
                        // keep-alive arm will see an empty ring and
                        // refill up to keep_alive_threshold (50 ms of
                        // silence). The drain handshake has already
                        // fired so the upstream is unblocked; that 50
                        // ms tail is harmless filler. The subsequent
                        // /speak path POSTs /spk/stop, which sets the
                        // flush flag and erases this filler before the
                        // next real Frame is pushed — so no head-clip
                        // risk for the next turn.
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                // Idle keep-alive: when the ring drops below
                // keep_alive_threshold (50 ms), top it up to that
                // threshold with 0.0 silence. Keeps the OS audio
                // pipeline pre-fetched so the next real audio doesn't
                // pay a wake-up latency on the speaker. Cheap no-op
                // while real audio is flowing (ring depth >> 50 ms).
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
