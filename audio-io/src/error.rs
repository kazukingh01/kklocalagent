#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("start/stop already in progress")]
    Busy,
    #[error("audio device '{0}' not found")]
    DeviceNotFound(String),
    #[error("no default {0} audio device")]
    NoDefaultDevice(&'static str),
    #[error("aec.backend '{0}' is not available (rebuild with `--features speex` for Speex)")]
    UnsupportedBackend(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AudioError {
    /// Recover a typed device error wrapped anywhere in the `anyhow` cause chain.
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
