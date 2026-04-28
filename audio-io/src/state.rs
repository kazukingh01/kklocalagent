use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::{broadcast, mpsc, Mutex};

use crate::capture::CaptureHandle;
use crate::config::Config;
use crate::playback::{PlaybackHandle, PlaybackMessage};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub mic_tx: broadcast::Sender<Bytes>,
    pub spk_tx: Arc<Mutex<Option<mpsc::Sender<PlaybackMessage>>>>,
    pub flush: Arc<FlushSignals>,
    pub handles: Arc<Mutex<ServiceHandles>>,
}

pub struct FlushSignals {
    pub producer: AtomicBool,
    pub consumer: AtomicBool,
}

impl FlushSignals {
    pub fn new() -> Self {
        Self {
            producer: AtomicBool::new(false),
            consumer: AtomicBool::new(false),
        }
    }

    pub fn trigger(&self) {
        self.producer.store(true, Ordering::Relaxed);
        self.consumer.store(true, Ordering::Relaxed);
    }
}

impl Default for FlushSignals {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct ServiceHandles {
    pub capture: Option<CaptureHandle>,
    pub playback: Option<PlaybackHandle>,
}
