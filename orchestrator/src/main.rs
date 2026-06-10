use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use orchestrator::config::Config;

#[derive(Debug, Parser)]
#[command(about = "orchestrator: receives VAD / wake-word events, drives ASR → LLM pipeline")]
struct Args {
    /// Path to a TOML config file.
    #[arg(long, env = "ORCH_CONFIG")]
    config: Option<PathBuf>,

    /// Override `server.listen` (e.g. `0.0.0.0:7000`).
    #[arg(long, env = "ORCH_LISTEN")]
    listen: Option<String>,

    /// Override `asr.url`.
    #[arg(long, env = "ORCH_ASR_URL")]
    asr_url: Option<String>,

    /// Override `asr.hallucination_blacklist` (pipe-separated substrings;
    /// empty value disables the filter).
    #[arg(long, env = "ORCH_ASR_HALLUCINATION_BLACKLIST", value_delimiter = '|')]
    asr_hallucination_blacklist: Option<Vec<String>>,

    /// Override `llm.url`.
    #[arg(long, env = "ORCH_LLM_URL")]
    llm_url: Option<String>,

    /// Override `llm.model`.
    #[arg(long, env = "ORCH_LLM_MODEL")]
    llm_model: Option<String>,

    /// Override `llm.system_prompt`. Empty disables the system message.
    #[arg(long, env = "ORCH_LLM_SYSTEM_PROMPT")]
    llm_system_prompt: Option<String>,

    /// Override `tts.url`. Empty disables the TTS stage.
    #[arg(long, env = "ORCH_TTS_URL")]
    tts_url: Option<String>,

    /// Override `tts.append_url`, used for continuation sentences (issue
    /// #16 — re-bursting via /speak every sentence overflowed audio-io's ring).
    #[arg(long, env = "ORCH_TTS_APPEND_URL")]
    tts_append_url: Option<String>,

    /// Override `tts.stop_url`. Empty disables barge-in TTS cancel.
    #[arg(long, env = "ORCH_TTS_STOP_URL")]
    tts_stop_url: Option<String>,

    /// Override `tts.finalize_url`. Empty falls back to a pure timeout
    /// (tail_quiet_ms must compensate).
    #[arg(long, env = "ORCH_TTS_FINALIZE_URL")]
    tts_finalize_url: Option<String>,

    /// Override `tts.tail_quiet_ms` (post-TTS VAD quiet window; 0 disables).
    #[arg(long, env = "ORCH_TTS_TAIL_QUIET_MS")]
    tts_tail_quiet_ms: Option<u64>,

    /// Override `tts.wake_ack_text` (spoken wake ack; empty disables).
    #[arg(long, env = "ORCH_WAKE_ACK_TEXT")]
    wake_ack_text: Option<String>,

    /// Override `wake.required` (true / false).
    #[arg(long, env = "ORCH_WAKE_REQUIRED")]
    wake_required: Option<bool>,

    /// Override `wake.wake_window_ms`.
    #[arg(long, env = "ORCH_WAKE_WINDOW_MS")]
    wake_window_ms: Option<u64>,

    /// Override `wake.turn_followup_window_ms`.
    #[arg(long, env = "ORCH_TURN_FOLLOWUP_WINDOW_MS")]
    turn_followup_window_ms: Option<u64>,

    /// Override `wake.barge_in` (true / false).
    #[arg(long, env = "ORCH_WAKE_BARGE_IN")]
    wake_barge_in: Option<bool>,

    /// Override `wake.post_wake_se_dropout_ms` (0 disables).
    #[arg(long, env = "ORCH_POST_WAKE_SE_DROPOUT_MS")]
    post_wake_se_dropout_ms: Option<u64>,

    /// Override `result_sink.url`. Empty disables forwarding.
    #[arg(long, env = "ORCH_RESULT_SINK_URL")]
    result_sink_url: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let mut config = match &args.config {
        Some(p) => Config::from_file(p)?,
        None => Config::default(),
    };
    if let Some(v) = args.listen {
        config.server.listen = v;
    }
    if let Some(v) = args.asr_url {
        config.asr.url = v;
    }
    if let Some(v) = args.asr_hallucination_blacklist {
        // `ORCH_ASR_HALLUCINATION_BLACKLIST=` (deliberately empty) splits to
        // one empty entry, which would match every ASR output via contains("").
        config.asr.hallucination_blacklist = v.into_iter().filter(|s| !s.is_empty()).collect();
    }
    if let Some(v) = args.llm_url {
        config.llm.url = v;
    }
    if let Some(v) = args.llm_model {
        config.llm.model = v;
    }
    if let Some(v) = args.llm_system_prompt {
        config.llm.system_prompt = v;
    }
    if let Some(v) = args.tts_url {
        config.tts.url = v;
    }
    if let Some(v) = args.tts_append_url {
        config.tts.append_url = v;
    }
    if let Some(v) = args.tts_stop_url {
        config.tts.stop_url = v;
    }
    if let Some(v) = args.tts_finalize_url {
        config.tts.finalize_url = v;
    }
    if let Some(v) = args.tts_tail_quiet_ms {
        config.tts.tail_quiet_ms = v;
    }
    if let Some(v) = args.wake_ack_text {
        config.tts.wake_ack_text = v;
    }
    if let Some(v) = args.wake_required {
        config.wake.required = v;
    }
    if let Some(v) = args.wake_window_ms {
        config.wake.wake_window_ms = v;
    }
    if let Some(v) = args.turn_followup_window_ms {
        config.wake.turn_followup_window_ms = v;
    }
    if let Some(v) = args.wake_barge_in {
        config.wake.barge_in = v;
    }
    if let Some(v) = args.post_wake_se_dropout_ms {
        config.wake.post_wake_se_dropout_ms = v;
    }
    if let Some(v) = args.result_sink_url {
        config.result_sink.url = v;
    }

    orchestrator::run(config).await
}
