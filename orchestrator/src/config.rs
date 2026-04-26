use std::path::Path;

use serde::Deserialize;

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

/// Optional outbound forwarder. When `url` is set, the orchestrator
/// POSTs:
/// * `WakeWordDetected` events as-is, **before** their normal handling,
/// * `TurnCompleted` synthetic events on successful pipeline completion.
///
/// When `url` is empty (the default), the forwarder is disabled — the
/// orchestrator runs in pure log-only mode for the v0.1 smoke. This
/// is the hook §10 anticipates for "Orchestrator が中央で記録".
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ResultSinkConfig {
    pub url: String,
    pub timeout_ms: u64,
}

impl Default for ResultSinkConfig {
    fn default() -> Self {
        Self {
            // Empty URL = forwarder disabled. The non-empty branch of
            // Config::validate() requires timeout_ms >= 1, so a sane
            // default protects the env-override path (Config::default()
            // bypasses serde defaults — only derived `Default` runs).
            url: String::new(),
            timeout_ms: 5_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// `host:port` to bind for the HTTP server.
    pub listen: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AsrConfig {
    /// whisper.cpp `/inference` URL.
    pub url: String,
    /// HTTP timeout per transcription POST, in milliseconds.
    pub timeout_ms: u64,
    /// Maximum simultaneous in-flight transcription requests. 1 mirrors
    /// whisper-server's own single-request behaviour; raise only if the
    /// backend was compiled with request batching.
    pub max_inflight: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// ollama `/api/chat` URL.
    pub url: String,
    /// Model name passed through to ollama (must exist in the LLM
    /// container's cache — see the `llm/` module).
    pub model: String,
    /// HTTP timeout per chat POST, in milliseconds.
    pub timeout_ms: u64,
    /// Maximum simultaneous in-flight LLM chat requests.
    pub max_inflight: u32,
}

/// Optional outbound TTS speak channel. When `url` is empty (the
/// default) the orchestrator skips TTS entirely — the assistant reply
/// is logged but not voiced. Useful for dev environments where the
/// `tts-streamer` service isn't running, or for headless CI runs.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    /// `tts-streamer` `/speak` URL. Empty disables the stage.
    pub url: String,
    /// `tts-streamer` `/stop` URL. Used for barge-in (`wake.barge_in`)
    /// — orchestrator POSTs here when a new WakeWordDetected arrives
    /// while a previous turn's TTS is still streaming. Empty disables
    /// the cancel side-effect (any new turn still queues normally
    /// behind the streamer's serial speak).
    pub stop_url: String,
    /// HTTP timeout per speak POST, in milliseconds. Generous because
    /// the upstream call covers VOICEVOX synthesis + WS streaming the
    /// frames at real time — a 5-second utterance physically takes 5 s
    /// to push through audio-io.
    pub timeout_ms: u64,
    /// Maximum simultaneous in-flight TTS requests. 1 mirrors the
    /// streamer's own single-flight lock; raise only if the streamer
    /// is replaced with a multi-channel speaker.
    pub max_inflight: u32,
}

/// Wake-word gating policy. Controls whether SpeechEnded events
/// trigger the ASR→LLM→TTS pipeline based on prior WakeWordDetected
/// events. v1.0 default; set `required=false` to fall back to v0.1
/// always-listening mode (every SpeechEnded triggers).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WakeConfig {
    /// When true, only SpeechEnded events that arrive within
    /// `arm_window_ms` of a WakeWordDetected event run the pipeline.
    /// Other SpeechEnded events are dropped with a log line. When
    /// false the orchestrator runs in always-listening mode (v0.1
    /// behaviour) — useful for headless tests and noisy debug
    /// sessions where saying the wake word is impractical.
    pub required: bool,
    /// How long after a WakeWordDetected to accept SpeechEnded, in
    /// milliseconds. Refreshed on every WakeWordDetected. The window
    /// is consumed (single-use) the first time SpeechEnded fires
    /// inside it — subsequent SpeechEnded events require another
    /// wake.
    pub arm_window_ms: u64,
    /// When true, a WakeWordDetected event arriving while the
    /// pipeline is `Processing` (ASR / LLM / TTS in flight) cancels
    /// the in-flight TTS via `tts.stop_url`, transitions back to
    /// `Armed`, and accepts the next utterance — the operator
    /// interrupting the assistant. When false, mid-turn wake events
    /// still arm the next window but don't interrupt the current
    /// reply.
    pub barge_in: bool,
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
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            url: "http://llm:11434/api/chat".into(),
            model: "gemma3:4b".into(),
            timeout_ms: 120_000,
            max_inflight: 1,
        }
    }
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            // Empty = TTS stage disabled. Production compose populates
            // this; dev / CI runs without `tts-streamer` leave it empty
            // so the pipeline still completes without trying to speak.
            url: String::new(),
            stop_url: String::new(),
            timeout_ms: 60_000,
            max_inflight: 1,
        }
    }
}

impl Default for WakeConfig {
    fn default() -> Self {
        // v1.0 defaults: gated on wake, 8 s window, barge-in on.
        // arm_window_ms = 8 s gives a natural pause for the user to
        // collect their thoughts after saying the wake word and still
        // tightly bounds the "false-positive whisper hallucination on
        // background noise" failure mode that v0.1 exhibits.
        Self {
            required: true,
            arm_window_ms: 8_000,
            barge_in: true,
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
        }
        if self.wake.required && self.wake.arm_window_ms == 0 {
            anyhow::bail!("wake.arm_window_ms must be >= 1 when wake.required is true");
        }
        if !self.result_sink.url.is_empty() && self.result_sink.timeout_ms == 0 {
            anyhow::bail!("result_sink.timeout_ms must be >= 1 when url is set");
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
}
