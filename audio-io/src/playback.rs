use std::sync::atomic::Ordering;
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::config::AudioConfig;
use crate::framer::PlaybackFramer;
use crate::state::FlushSignals;

pub struct PlaybackHandle {
    thread_shutdown: std_mpsc::SyncSender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
    task: Option<tokio::task::JoinHandle<()>>,
    spk_tx: mpsc::Sender<Bytes>,
}

impl PlaybackHandle {
    pub fn sender(&self) -> mpsc::Sender<Bytes> {
        self.spk_tx.clone()
    }
}

impl Drop for PlaybackHandle {
    fn drop(&mut self) {
        // Cancel producer task first so it stops pushing into the ring buffer.
        if let Some(t) = self.task.take() {
            t.abort();
        }
        let _ = self.thread_shutdown.try_send(());
        if let Some(h) = self.thread.take() {
            let _ = h.join();
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

    let thread = std::thread::Builder::new()
        .name("audio-playback".into())
        .spawn(move || {
            let ready_tx_clone = ready_tx.clone();
            if let Err(e) = run_playback(&device_name, buffer_ms, flush_cb, ready_tx, shutdown_rx) {
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
    let (spk_tx, spk_rx) = mpsc::channel::<Bytes>(32);
    let task = tokio::spawn(playback_producer_task(
        spk_rx,
        ready.producer,
        audio.sample_rate,
        ready.native_rate,
        ready.native_channels,
        flush,
    ));

    Ok(PlaybackHandle {
        thread_shutdown: shutdown_tx,
        thread: Some(thread),
        task: Some(task),
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
            device.build_output_stream(
                &stream_config,
                move |data: &mut [f32], _| {
                    if flush.consumer.swap(false, Ordering::Relaxed) {
                        while consumer.try_pop().is_some() {}
                    }
                    for sample in data.iter_mut() {
                        *sample = consumer.try_pop().unwrap_or(0.0);
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let mut consumer: HeapCons<f32> = consumer;
            let flush = flush_cb;
            device.build_output_stream(
                &stream_config,
                move |data: &mut [i16], _| {
                    if flush.consumer.swap(false, Ordering::Relaxed) {
                        while consumer.try_pop().is_some() {}
                    }
                    for sample in data.iter_mut() {
                        let v = consumer.try_pop().unwrap_or(0.0);
                        *sample = (v.clamp(-1.0, 1.0) * 32767.0) as i16;
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let mut consumer: HeapCons<f32> = consumer;
            let flush = flush_cb;
            device.build_output_stream(
                &stream_config,
                move |data: &mut [u16], _| {
                    if flush.consumer.swap(false, Ordering::Relaxed) {
                        while consumer.try_pop().is_some() {}
                    }
                    for sample in data.iter_mut() {
                        let v = consumer.try_pop().unwrap_or(0.0);
                        let scaled = (v.clamp(-1.0, 1.0) * 32767.0) as i32 + 32768;
                        *sample = scaled.clamp(0, u16::MAX as i32) as u16;
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
    mut spk_rx: mpsc::Receiver<Bytes>,
    mut producer: HeapProd<f32>,
    source_rate: u32,
    native_rate: u32,
    native_channels: u16,
    flush: Arc<FlushSignals>,
) {
    let mut framer = match PlaybackFramer::new(source_rate, native_rate, native_channels) {
        Ok(f) => f,
        Err(e) => {
            error!("failed to create PlaybackFramer: {e:?}");
            return;
        }
    };
    let mut total_dropped: u64 = 0;
    let mut drops_since_log: u32 = 0;
    while let Some(bytes) = spk_rx.recv().await {
        if flush.producer.swap(false, Ordering::Relaxed) {
            while spk_rx.try_recv().is_ok() {}
            framer.flush();
            continue;
        }
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
    info!("playback producer task exiting");
}
