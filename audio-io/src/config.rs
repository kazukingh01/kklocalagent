use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub audio: AudioConfig,
    pub input: DeviceConfig,
    pub output: DeviceConfig,
    pub runtime: RuntimeConfig,
    pub aec: AecConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_ms: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DeviceConfig {
    pub device: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    pub autostart: bool,
    pub playback_buffer_ms: u32,
    pub mic_broadcast_frames: u32,
    /// Number of parallel playback tracks. Each track owns its own cpal
    /// output stream, ring buffer, producer task, and `FlushSignals`,
    /// and is identified externally by integer id `0..playback_tracks`.
    /// WASAPI shared mode (the Windows default) mixes the per-stream
    /// outputs at the OS layer, so independent senders never interfere.
    /// `/spk` (no query) defaults to track 0 for backwards compatibility
    /// with the existing TTS-streamer client; new callers (e.g. an agent
    /// playing pre-rendered audio files) pass `?track=1`.
    pub playback_tracks: u32,
}

/// Acoustic echo cancellation. Runs *inside* audio-io because near-end
/// (mic capture) and far-end (the mixed `/spk` track PCM) live in the same
/// process on the same clock — so the echo-cancelled mic can be served to
/// every consumer (VAD, wake-word-detection) from a single computation via
/// `/mic?aec=1`, with `/mic` (raw) left byte-identical. Disabled by default;
/// when off, no mixer/AEC task is spawned and `/mic?aec=1` is rejected.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AecConfig {
    pub enabled: bool,
    /// Adaptive-filter backend. Only `"nlms"` (a pure-Rust normalized LMS
    /// filter — no native dependency, cross-compiles cleanly to the mingw
    /// Windows target) is implemented today; the field exists so a
    /// `"speex"` / `"webrtc"` backend can be added later without a config
    /// break.
    pub backend: String,
    /// Adaptive filter length in milliseconds — covers the residual echo
    /// tail (reverberation) *after* `initial_delay_ms` is removed. Longer
    /// captures more reverb at higher CPU cost (taps = ms * sample_rate).
    pub filter_length_ms: u32,
    /// Bulk far-end delay hint in milliseconds: the playback ring residency
    /// (`runtime.playback_buffer_ms`) + output + acoustic + capture latency.
    /// The far-end reference is delayed by this much before entering the
    /// adaptive filter so the filter only has to model the room tail.
    pub initial_delay_ms: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            audio: AudioConfig::default(),
            input: DeviceConfig::default(),
            output: DeviceConfig::default(),
            runtime: RuntimeConfig::default(),
            aec: AecConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 7010,
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            frame_ms: 20,
        }
    }
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device: "default".into(),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            autostart: true,
            playback_buffer_ms: 200,
            mic_broadcast_frames: 64,
            playback_tracks: 2,
        }
    }
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: "nlms".into(),
            filter_length_ms: 150,
            initial_delay_ms: 120,
        }
    }
}

impl AudioConfig {
    pub fn samples_per_frame(&self) -> usize {
        (self.sample_rate as usize * self.frame_ms as usize) / 1000
    }

    pub fn bytes_per_frame(&self) -> usize {
        self.samples_per_frame() * self.channels as usize * 2
    }
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        let cfg: Self = toml::from_str(&text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.audio.channels == 0 {
            anyhow::bail!("audio.channels must be >= 1");
        }
        if self.audio.sample_rate == 0 {
            anyhow::bail!("audio.sample_rate must be > 0");
        }
        if self.audio.frame_ms == 0 {
            anyhow::bail!("audio.frame_ms must be > 0");
        }
        if (self.audio.sample_rate as usize * self.audio.frame_ms as usize) % 1000 != 0 {
            anyhow::bail!("sample_rate * frame_ms must be divisible by 1000");
        }
        if self.runtime.mic_broadcast_frames == 0 {
            anyhow::bail!("runtime.mic_broadcast_frames must be >= 1");
        }
        if self.runtime.playback_tracks == 0 {
            anyhow::bail!("runtime.playback_tracks must be >= 1");
        }
        if self.aec.enabled {
            if self.aec.backend != "nlms" {
                anyhow::bail!(
                    "aec.backend '{}' is not supported (only 'nlms')",
                    self.aec.backend
                );
            }
            if self.aec.filter_length_ms == 0 {
                anyhow::bail!("aec.filter_length_ms must be > 0 when aec.enabled");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn frame_sizes() {
        let a = AudioConfig::default();
        assert_eq!(a.samples_per_frame(), 320);
        assert_eq!(a.bytes_per_frame(), 640);
    }

    #[test]
    fn parses_example() {
        let text = include_str!("../config.example.toml");
        let cfg: Config = toml::from_str(text).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.server.port, 7010);
    }
}
