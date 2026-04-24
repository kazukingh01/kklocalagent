use std::path::Path;

use clap::ValueEnum;
use serde::Deserialize;

/// Where SpeechStarted/SpeechEnded events go.
///
/// - `DryRun` (default, safe to run without any peer): log each event as
///   JSON.
/// - `AsrDirect`: still log the event, and on `SpeechEnded` POST the
///   utterance audio (wrapped as a WAV) to whisper.cpp's `/inference`
///   endpoint at `sink.asr_url`. Used for the audio-io→vad→asr smoke
///   test before the orchestrator exists.
/// - `Orchestrator`: forward events to `sink.orchestrator_url`. Not yet
///   implemented — events are dropped with a warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
#[clap(rename_all = "kebab-case")]
pub enum SinkMode {
    DryRun,
    AsrDirect,
    Orchestrator,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub source: SourceConfig,
    pub detector: DetectorConfig,
    pub sink: SinkConfig,
    pub diag: DiagConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DiagConfig {
    /// When true, emit per-window RMS + speech_ratio stats under the
    /// `vad::diag` target. Useful for debugging false triggers.
    pub enabled: bool,
    /// Window size in frames. Default 50 = 1 second at 20 ms/frame.
    pub window_frames: u32,
}

impl Default for DiagConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_frames: 50,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SourceConfig {
    /// audio-io /mic WebSocket URL.
    pub mic_url: String,
    /// Delay before reconnecting after a disconnect, in ms.
    pub reconnect_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DetectorConfig {
    /// webrtc-vad aggressiveness 0..=3
    /// (0=Quality, 1=LowBitrate, 2=Aggressive, 3=VeryAggressive).
    pub aggressiveness: u8,
    /// Must match audio-io wire format; webrtc-vad accepts 8/16/32/48 kHz only.
    pub sample_rate: u32,
    /// Must match audio-io wire format; webrtc-vad accepts 10/20/30 ms only.
    pub frame_ms: u32,
    /// Number of consecutive voiced frames before SpeechStarted fires. Any
    /// silent frame resets the run to 0, so single-frame blips can't trigger
    /// spurious utterances.
    pub start_frames: u32,
    /// Number of consecutive silent frames before SpeechEnded fires.
    /// Covers natural breath pauses mid-utterance.
    pub hang_frames: u32,
    /// Hard cap on utterance length. Forces an end event if speech runs longer.
    pub max_utterance_frames: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SinkConfig {
    /// See [`SinkMode`].
    pub mode: SinkMode,
    /// Orchestrator events endpoint — used only when `mode = "orchestrator"`.
    pub orchestrator_url: String,
    /// whisper.cpp `/inference` endpoint — used only when `mode = "asr-direct"`.
    pub asr_url: String,
    /// HTTP timeout for `asr-direct` POSTs, in milliseconds. Larger whisper
    /// models take longer per utterance — `large-v3-turbo` on a 30 s
    /// utterance can approach the default 30 s ceiling.
    pub asr_timeout_ms: u64,
    /// Maximum simultaneous in-flight `asr-direct` POSTs. Excess utterances
    /// are dropped with a warning rather than queued, so backpressure is
    /// observable instead of presenting as silent timeouts. 1 = strictly
    /// serial (matches whisper-server's own single-request behaviour).
    pub asr_max_inflight: u32,
    /// Include base64-encoded utterance audio in the *log* JSON for each
    /// SpeechEnded event. Independent of asr-direct, which always uploads
    /// the audio regardless of this flag.
    pub log_audio_in_event: bool,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            mic_url: "ws://127.0.0.1:7010/mic".into(),
            reconnect_ms: 1000,
        }
    }
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            aggressiveness: 2,
            sample_rate: 16000,
            frame_ms: 20,
            start_frames: 3,          // ~60 ms of speech
            hang_frames: 20,          // ~400 ms of silence
            max_utterance_frames: 1500, // 30 s
        }
    }
}

impl Default for SinkConfig {
    fn default() -> Self {
        Self {
            mode: SinkMode::DryRun,
            orchestrator_url: "http://127.0.0.1:7000/events".into(),
            asr_url: "http://127.0.0.1:7040/inference".into(),
            asr_timeout_ms: 30_000,
            asr_max_inflight: 1,
            log_audio_in_event: false,
        }
    }
}

impl DetectorConfig {
    pub fn samples_per_frame(&self) -> usize {
        (self.sample_rate as usize * self.frame_ms as usize) / 1000
    }
    pub fn bytes_per_frame(&self) -> usize {
        self.samples_per_frame() * 2
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
        if self.detector.aggressiveness > 3 {
            anyhow::bail!("detector.aggressiveness must be 0..=3");
        }
        if !matches!(self.detector.sample_rate, 8000 | 16000 | 32000 | 48000) {
            anyhow::bail!("detector.sample_rate must be 8000, 16000, 32000 or 48000");
        }
        if !matches!(self.detector.frame_ms, 10 | 20 | 30) {
            anyhow::bail!("detector.frame_ms must be 10, 20 or 30");
        }
        if self.detector.start_frames == 0 {
            anyhow::bail!("detector.start_frames must be >= 1");
        }
        if self.detector.hang_frames == 0 {
            anyhow::bail!("detector.hang_frames must be >= 1");
        }
        if self.detector.max_utterance_frames == 0 {
            anyhow::bail!("detector.max_utterance_frames must be >= 1");
        }
        if self.diag.window_frames == 0 {
            anyhow::bail!("diag.window_frames must be >= 1");
        }
        if self.sink.asr_timeout_ms == 0 {
            anyhow::bail!("sink.asr_timeout_ms must be >= 1");
        }
        if self.sink.asr_max_inflight == 0 {
            anyhow::bail!("sink.asr_max_inflight must be >= 1");
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
        let d = DetectorConfig::default();
        assert_eq!(d.samples_per_frame(), 320);
        assert_eq!(d.bytes_per_frame(), 640);
    }

    #[test]
    fn parses_example() {
        let text = include_str!("../config.example.toml");
        let cfg: Config = toml::from_str(text).unwrap();
        cfg.validate().unwrap();
    }

    #[test]
    fn rejects_unsupported_rate() {
        let mut cfg = Config::default();
        cfg.detector.sample_rate = 44100;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_unsupported_frame_ms() {
        let mut cfg = Config::default();
        cfg.detector.frame_ms = 25;
        assert!(cfg.validate().is_err());
    }
}
