use std::sync::mpsc as std_mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use tokio::sync::broadcast;
use tracing::{error, info};

use crate::config::AudioConfig;
use crate::framer::CaptureFramer;

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
            .ok_or_else(|| anyhow!("no default input device"));
    }
    for dev in host.input_devices().context("listing input devices")? {
        if dev.name().unwrap_or_default() == name {
            return Ok(dev);
        }
    }
    Err(anyhow!("input device '{name}' not found"))
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

    // Duration of one emitted frame in nanoseconds. Used to back-date
    // earlier frames when a single cpal callback yields more than one
    // frame (rare in steady state, possible during catch-up bursts).
    let frame_ns: u64 =
        (audio.samples_per_frame() as u64 * 1_000_000_000) / audio.sample_rate as u64;

    let err_fn = |e| error!("cpal input stream error: {e}");

    let stream = match sample_format {
        SampleFormat::F32 => {
            let mut framer = framer;
            let tx = mic_tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| dispatch(framer.push_f32(data), &tx, frame_ns),
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let mut framer = framer;
            let tx = mic_tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| dispatch(framer.push_i16(data), &tx, frame_ns),
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let mut framer = framer;
            let tx = mic_tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _| dispatch(framer.push_u16(data), &tx, frame_ns),
                err_fn,
                None,
            )?
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

fn dispatch(frames: Vec<Vec<u8>>, tx: &broadcast::Sender<(u64, Bytes)>, frame_ns: u64) {
    // `now` is captured once after the cpal callback's framer.push_*
    // returns — i.e. immediately after the *last* emitted frame's tail
    // sample arrived. Earlier frames in this batch ended `frame_ns`
    // earlier, so we subtract a per-position offset.
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = frames.len();
    for (i, frame) in frames.into_iter().enumerate() {
        let end_ns = now_ns.saturating_sub(((n - 1 - i) as u64) * frame_ns);
        let _ = tx.send((end_ns, Bytes::from(frame)));
    }
}
