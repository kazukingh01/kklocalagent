use std::sync::mpsc as std_mpsc;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};
use tokio::sync::broadcast;
use tracing::{error, info};

use crate::config::AudioConfig;
use crate::error::AudioError;
use crate::framer::CaptureFramer;
use crate::pcm::epoch_ns;

pub struct CaptureHandle {
    shutdown: std_mpsc::SyncSender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        let _ = self.shutdown.try_send(());
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

pub fn start_capture(
    device_name: &str,
    audio: AudioConfig,
    mic_tx: broadcast::Sender<(u64, Bytes)>,
) -> Result<CaptureHandle> {
    let (shutdown_tx, shutdown_rx) = std_mpsc::sync_channel::<()>(1);
    let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<()>>(1);
    let device_name = device_name.to_string();
    let thread = std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || {
            if let Err(e) = run_capture(&device_name, &audio, mic_tx, shutdown_rx, ready_tx.clone())
            {
                error!("capture thread error: {e:?}");
                let _ = ready_tx.send(Err(e));
            }
        })?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(CaptureHandle {
            shutdown: shutdown_tx,
            thread: Some(thread),
        }),
        Ok(Err(e)) => {
            let _ = thread.join();
            Err(e)
        }
        Err(_) => {
            let _ = thread.join();
            Err(anyhow!("capture thread exited before ready"))
        }
    }
}

fn find_input_device(name: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();
    if name == "default" {
        return host
            .default_input_device()
            .ok_or_else(|| AudioError::NoDefaultDevice("input").into());
    }
    for dev in host.input_devices().context("listing input devices")? {
        if dev.name().unwrap_or_default() == name {
            return Ok(dev);
        }
    }
    Err(AudioError::DeviceNotFound(name.into()).into())
}

fn run_capture(
    device_name: &str,
    audio: &AudioConfig,
    mic_tx: broadcast::Sender<(u64, Bytes)>,
    shutdown_rx: std_mpsc::Receiver<()>,
    ready_tx: std_mpsc::SyncSender<Result<()>>,
) -> Result<()> {
    let device = find_input_device(device_name)?;
    let default_config = device
        .default_input_config()
        .context("default_input_config")?;
    let sample_format = default_config.sample_format();
    let native_rate = default_config.sample_rate().0;
    let native_channels = default_config.channels();
    let stream_config: StreamConfig = default_config.into();

    info!(
        device = ?device.name().ok(),
        native_rate,
        native_channels,
        ?sample_format,
        "opening capture stream"
    );

    let framer = CaptureFramer::new(
        native_rate,
        native_channels,
        audio.sample_rate,
        audio.samples_per_frame(),
    )?;

    let frame_ns: u64 =
        (audio.samples_per_frame() as u64 * 1_000_000_000) / audio.sample_rate as u64;

    let stream = match sample_format {
        SampleFormat::F32 => {
            build_input_stream::<f32>(&device, &stream_config, framer, mic_tx.clone(), frame_ns)?
        }
        SampleFormat::I16 => {
            build_input_stream::<i16>(&device, &stream_config, framer, mic_tx.clone(), frame_ns)?
        }
        SampleFormat::U16 => {
            build_input_stream::<u16>(&device, &stream_config, framer, mic_tx.clone(), frame_ns)?
        }
        other => anyhow::bail!("unsupported input sample format: {other:?}"),
    };

    stream.play()?;
    let _ = ready_tx.send(Ok(()));
    let _ = shutdown_rx.recv();
    drop(stream);
    info!("capture stopped");
    Ok(())
}

fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    mut framer: CaptureFramer,
    tx: broadcast::Sender<(u64, Bytes)>,
    frame_ns: u64,
) -> Result<cpal::Stream>
where
    T: SizedSample + Send + 'static,
    f32: FromSample<T>,
{
    let err_fn = |e| error!("cpal input stream error: {e}");
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _| dispatch(framer.push(data), &tx, frame_ns),
        err_fn,
        None,
    )?;
    Ok(stream)
}

fn dispatch(frames: Vec<Vec<u8>>, tx: &broadcast::Sender<(u64, Bytes)>, frame_ns: u64) {
    if frames.is_empty() {
        return;
    }
    let now_ns = epoch_ns();
    let n = frames.len();
    for (i, frame) in frames.into_iter().enumerate() {
        let end_ns = now_ns.saturating_sub(((n - 1 - i) as u64) * frame_ns);
        let _ = tx.send((end_ns, Bytes::from(frame)));
    }
}
