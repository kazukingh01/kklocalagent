//! Typed errors for the service layer so the HTTP handlers can map failures to
//! status codes by matching variants, instead of grepping error message text.

/// An error from starting/stopping the audio services.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// A concurrent start/stop is holding the services lock.
    #[error("start/stop already in progress")]
    Busy,
    /// The configured device name matched no available device.
    #[error("audio device '{0}' not found")]
    DeviceNotFound(String),
    /// No default device of this kind exists (`"input"` / `"output"`).
    #[error("no default {0} audio device")]
    NoDefaultDevice(&'static str),
    /// `aec.backend` requested a backend this binary was not built with.
    #[error("aec.backend '{0}' is not available (rebuild with `--features speex` for Speex)")]
    UnsupportedBackend(String),
    /// Anything else (device enumeration, cpal stream build, config, ...).
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AudioError {
    /// Classify an `anyhow` error bubbling up from the audio threads, recovering
    /// a device error that was wrapped (via `?`/`.context`) anywhere in the
    /// cause chain; everything else becomes [`AudioError::Other`].
    pub fn from_chain(e: anyhow::Error) -> Self {
        for cause in e.chain() {
            if let Some(a) = cause.downcast_ref::<AudioError>() {
                match a {
                    AudioError::DeviceNotFound(n) => return AudioError::DeviceNotFound(n.clone()),
                    AudioError::NoDefaultDevice(k) => return AudioError::NoDefaultDevice(k),
                    _ => {}
                }
            }
        }
        AudioError::Other(e)
    }
}
