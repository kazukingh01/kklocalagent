use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use voice_activity_detection::config::{Config, SinkMode};

#[derive(Debug, Parser)]
#[command(
    about = "voice_activity_detection: subscribe to audio-io /mic and emit speech events"
)]
struct Args {
    /// Path to a TOML config file.
    #[arg(long, env = "VAD_CONFIG")]
    config: Option<PathBuf>,

    /// Override the audio-io /mic WebSocket URL from config.
    #[arg(long, env = "VAD_MIC_URL")]
    mic_url: Option<String>,

    /// Override the sink mode from config.
    /// Values: dry-run | asr-direct | orchestrator.
    #[arg(long, value_enum, env = "VAD_SINK_MODE")]
    sink_mode: Option<SinkMode>,

    /// Override the ASR /inference URL — used in `asr-direct` mode.
    #[arg(long, env = "VAD_ASR_URL")]
    asr_url: Option<String>,

    /// Override the orchestrator /events URL — used in `orchestrator` mode.
    #[arg(long, env = "VAD_ORCHESTRATOR_URL")]
    orchestrator_url: Option<String>,

    /// Include base64-encoded utterance audio in event envelopes.
    /// Required when `--sink-mode orchestrator` is set so the
    /// orchestrator can run ASR; off by default for asr-direct/dry-run.
    #[arg(long, env = "VAD_LOG_AUDIO_IN_EVENT")]
    log_audio_in_event: Option<bool>,

    /// Override `detector.hang_frames` (consecutive silent 20 ms
    /// frames before SpeechEnded fires). Each frame is `frame_ms`
    /// (default 20 ms), so 10 ≈ 200 ms of silence, 20 ≈ 400 ms
    /// (the legacy default). Lower values shave end-of-utterance
    /// latency at the cost of cutting off natural mid-utterance
    /// pauses.
    #[arg(long, env = "VAD_HANG_FRAMES")]
    hang_frames: Option<u32>,

    /// Override `detector.start_frames` (consecutive voiced 20 ms
    /// frames before SpeechStarted fires). Default 3 ≈ 60 ms.
    /// Bump to 5–7 (100–140 ms) to filter out short impulse noise
    /// (single claps, taps, mouse clicks) that webrtc-vad would
    /// otherwise mis-classify as speech onset. Trade-off: very
    /// fast utterance starts ("はい") get their first frames eaten.
    #[arg(long, env = "VAD_START_FRAMES")]
    start_frames: Option<u32>,

    /// Override `detector.aggressiveness` (0..=3). webrtc-vad
    /// internal strictness: 0 = Quality (most permissive),
    /// 3 = VeryAggressive (strictest, most false-negatives on
    /// quiet speech but cuts impulse / borderline noise hardest).
    /// Default 2.
    #[arg(long, env = "VAD_AGGRESSIVENESS")]
    aggressiveness: Option<u8>,

    /// Override `detector.denoise`. When true, every incoming
    /// 20 ms frame is run through nnnoiseless (RNNoise) before VAD
    /// classification *and* before being buffered for ASR. Cleans
    /// steady-state background noise (fans, AC, low hum); reduces
    /// VAD false positives and Whisper's silence-hallucination
    /// rate ("ご視聴ありがとうございました" etc.).
    #[arg(long, env = "VAD_DENOISE")]
    denoise: Option<bool>,

    /// Override `detector.min_utterance_rms_dbfs`. SpeechEnded
    /// events whose buffered audio has RMS below this dBFS value
    /// are dropped before reaching the sink. 0 (or any non-
    /// negative value) disables the gate. Recommended starting
    /// point for hallucination-suppression: -45.
    #[arg(long, env = "VAD_MIN_UTTERANCE_RMS_DBFS")]
    min_utterance_rms_dbfs: Option<f32>,

    /// Shortcut for `--sink-mode asr-direct`. Kept for back-compat with
    /// the early dry-run/live split.
    #[arg(long, conflicts_with = "sink_mode")]
    live: bool,
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
    if let Some(url) = args.mic_url {
        config.source.mic_url = url;
    }
    if let Some(url) = args.asr_url {
        config.sink.asr_url = url;
    }
    if let Some(url) = args.orchestrator_url {
        config.sink.orchestrator_url = url;
    }
    if let Some(v) = args.log_audio_in_event {
        config.sink.log_audio_in_event = v;
    }
    if let Some(v) = args.hang_frames {
        config.detector.hang_frames = v;
    }
    if let Some(v) = args.start_frames {
        config.detector.start_frames = v;
    }
    if let Some(v) = args.aggressiveness {
        config.detector.aggressiveness = v;
    }
    if let Some(v) = args.denoise {
        config.detector.denoise = v;
    }
    if let Some(v) = args.min_utterance_rms_dbfs {
        config.detector.min_utterance_rms_dbfs = v;
    }
    if let Some(mode) = args.sink_mode {
        config.sink.mode = mode;
    } else if args.live {
        // --live predates the multi-mode sink and meant "anything but
        // dry-run". The only live mode wired up today is asr-direct.
        config.sink.mode = SinkMode::AsrDirect;
    }

    voice_activity_detection::run(config).await
}
