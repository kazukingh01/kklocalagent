use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub asr: AsrConfig,
    pub llm: LlmConfig,
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
