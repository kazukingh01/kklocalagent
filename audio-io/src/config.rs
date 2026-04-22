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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            audio: AudioConfig::default(),
            input: DeviceConfig::default(),
            output: DeviceConfig::default(),
            runtime: RuntimeConfig::default(),
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
