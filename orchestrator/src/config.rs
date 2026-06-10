use std::path::Path;

use serde::Deserialize;
use tracing::warn;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub asr: AsrConfig,
    pub llm: LlmConfig,
    pub tts: TtsConfig,
    pub wake: WakeConfig,
    pub result_sink: ResultSinkConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ResultSinkConfig {
    pub url: String,
    pub timeout_ms: u64,
}

impl Default for ResultSinkConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            timeout_ms: 5_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub listen: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AsrConfig {
    pub url: String,
    pub timeout_ms: u64,
    pub max_inflight: u32,
    /// Substring blacklist for known Whisper hallucinations on near-silence
    /// (Whisper fills ambiguous quiet input with stock YouTube end-of-video
    /// phrases); a match drops the turn like an empty transcription.
    pub hallucination_blacklist: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub url: String,
    pub model: String,
    pub system_prompt: String,
    pub timeout_ms: u64,
    pub max_inflight: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    pub url: String,
    /// `tts-streamer` `/append` URL, used for the **second and later**
    /// sentences. Unlike `/speak` it doesn't cancel or re-burst — this
    /// prevents the audio-io ring overflow from issue #16 (re-bursting
    /// 5 s every sentence caused dropouts). Required when `url` is set:
    /// routing every sentence through `/speak` is no longer equivalent
    /// because `/speak` FIFO-aborts the previous task, cutting sentence
    /// N mid-stream when sentence N+1 arrives.
    pub append_url: String,
    pub stop_url: String,
    /// `tts-streamer` `/finalize` URL. tts-streamer sends EOS and awaits
    /// audio-io's drained reply, so the HTTP response is the precise
    /// moment the speaker fell silent. Empty skips the call, but then
    /// `tail_quiet_ms` must cover audio-io's playback ring drain time
    /// *and* VAD's hangover, not just the latter.
    pub finalize_url: String,
    pub timeout_ms: u64,
    pub max_inflight: u32,
    /// After each turn's TTS completes, drop all VAD events for this many
    /// extra ms. Even with the finalize drain handshake, VAD's silence
    /// hangover (default 200 ms via VAD_HANG_FRAMES=10) fires SE that
    /// long after audio stopped, so this window must cover hangover +
    /// margin. 400 ms covers hang_frames up to ~15; 0 disables (only
    /// safe with upstream AEC).
    pub tail_quiet_ms: u64,
    pub wake_ack_text: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WakeConfig {
    pub required: bool,
    pub wake_window_ms: u64,
    pub turn_followup_window_ms: u64,
    pub barge_in: bool,
    /// SpeechEnded events within this many ms of the most recent wake are
    /// dropped. Guards against VAD firing SE for the wake word's own audio
    /// (otherwise that SE dispatches a turn whose ASR text is just the wake
    /// word and the LLM answers nothing the user asked). 800 ms swallows
    /// the wake-word-alone SE but lets a continuous "Hey Jarvis, what's
    /// the weather?" through, since VAD's hangover pushes that SE to
    /// ~1.5–2 s after the wake. 0 disables; only honoured when
    /// `required = true`.
    pub post_wake_se_dropout_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:7000".into(),
        }
    }
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            url: "http://automatic-speech-recognition:8080/inference".into(),
            timeout_ms: 60_000,
            max_inflight: 1,
            hallucination_blacklist: vec![
                "ご視聴ありがとう".into(),
                "(拍手)".into(),
                "(笑)".into(),
                "Thanks for watching".into(),
                "Subscribe to my channel".into(),
            ],
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            url: "http://llm:11434/api/chat".into(),
            model: "gemma3:4b".into(),
            system_prompt: String::new(),
            timeout_ms: 120_000,
            max_inflight: 1,
        }
    }
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            append_url: String::new(),
            stop_url: String::new(),
            finalize_url: String::new(),
            timeout_ms: 60_000,
            max_inflight: 1,
            tail_quiet_ms: 400,
            wake_ack_text: String::new(),
        }
    }
}

impl Default for WakeConfig {
    fn default() -> Self {
        Self {
            required: true,
            wake_window_ms: 5_000,
            turn_followup_window_ms: 10_000,
            barge_in: true,
            post_wake_se_dropout_ms: 800,
        }
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
        if self.server.listen.is_empty() {
            anyhow::bail!("server.listen must not be empty");
        }
        if self.asr.url.is_empty() {
            anyhow::bail!("asr.url must not be empty");
        }
        if self.asr.timeout_ms == 0 {
            anyhow::bail!("asr.timeout_ms must be >= 1");
        }
        if self.asr.max_inflight == 0 {
            anyhow::bail!("asr.max_inflight must be >= 1");
        }
        if self.llm.url.is_empty() {
            anyhow::bail!("llm.url must not be empty");
        }
        if self.llm.model.is_empty() {
            anyhow::bail!("llm.model must not be empty");
        }
        if self.llm.timeout_ms == 0 {
            anyhow::bail!("llm.timeout_ms must be >= 1");
        }
        if self.llm.max_inflight == 0 {
            anyhow::bail!("llm.max_inflight must be >= 1");
        }
        if !self.tts.url.is_empty() {
            if self.tts.timeout_ms == 0 {
                anyhow::bail!("tts.timeout_ms must be >= 1 when url is set");
            }
            if self.tts.max_inflight == 0 {
                anyhow::bail!("tts.max_inflight must be >= 1 when url is set");
            }
            // issue #16: 空フォールバックを許すと全文が /speak に流れ、
            // streamer 側の FIFO abort が文 N を文 N+1 で中断してしまう。
            if self.tts.append_url.is_empty() {
                anyhow::bail!(
                    "tts.append_url must be set when tts.url is set \
                     (issue #16: /append routes continuation sentences so /speak's \
                     FIFO-abort doesn't cut mid-utterance)"
                );
            }
        }
        if self.wake.required {
            if self.wake.wake_window_ms == 0 {
                anyhow::bail!("wake.wake_window_ms must be >= 1 when wake.required is true");
            }
            if self.wake.turn_followup_window_ms == 0 {
                anyhow::bail!("wake.turn_followup_window_ms must be >= 1 when wake.required is true");
            }
        }
        if !self.result_sink.url.is_empty() && self.result_sink.timeout_ms == 0 {
            anyhow::bail!("result_sink.timeout_ms must be >= 1 when url is set");
        }
        if self.wake.barge_in && !self.tts.url.is_empty() && self.tts.stop_url.is_empty() {
            anyhow::bail!(
                "wake.barge_in=true requires tts.stop_url when tts.url is set \
                 (without it, mid-TTS barge-in blocks until the /speak HTTP timeout)"
            );
        }
        if !self.tts.url.is_empty() && self.tts.finalize_url.is_empty() {
            warn!(
                target: "orch::config",
                "tts.finalize_url empty: tts.tail_quiet_ms must cover audio-io playback drain + VAD hangover (not just the latter)"
            );
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
    fn parses_example() {
        let text = include_str!("../config.example.toml");
        let cfg: Config = toml::from_str(text).unwrap();
        cfg.validate().unwrap();
    }

    #[test]
    fn rejects_empty_model() {
        let mut cfg = Config::default();
        cfg.llm.model.clear();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn barge_in_without_stop_url_is_rejected() {
        let mut cfg = Config::default();
        cfg.tts.url = "http://tts:7070/speak".into();
        cfg.tts.append_url = "http://tts:7070/append".into();
        cfg.tts.stop_url = String::new();
        cfg.wake.barge_in = true;
        let err = cfg.validate().expect_err("should reject barge_in without stop_url");
        let msg = format!("{err:#}");
        assert!(msg.contains("stop_url"), "unexpected error: {msg}");
    }

    #[test]
    fn barge_in_without_stop_url_is_ok_when_tts_disabled() {
        let mut cfg = Config::default();
        cfg.tts.url = String::new();
        cfg.tts.stop_url = String::new();
        cfg.wake.barge_in = true;
        cfg.validate().expect("tts disabled => barge_in flag is moot");
    }

    #[test]
    fn barge_in_with_stop_url_is_accepted() {
        let mut cfg = Config::default();
        cfg.tts.url = "http://tts:7070/speak".into();
        cfg.tts.stop_url = "http://tts:7070/stop".into();
        cfg.wake.barge_in = true;
        cfg.tts.append_url = "http://tts:7070/append".into();
        cfg.validate().expect("barge_in + stop_url is valid");
    }

    #[test]
    fn tts_url_without_append_url_is_rejected() {
        let mut cfg = Config::default();
        cfg.tts.url = "http://tts:7070/speak".into();
        cfg.tts.append_url = String::new();
        let err = cfg.validate().expect_err("should reject url without append_url");
        let msg = format!("{err:#}");
        assert!(msg.contains("append_url"), "unexpected error: {msg}");
    }
}
