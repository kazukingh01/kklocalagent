//! tts-streamer: HTTP shim that turns a text request into VOICEVOX
//! synthesis + audio-io playback streaming.
//!
//! Endpoints:
//!     POST /speak    body: {"text": "..."}   start of turn / barge-in
//!     POST /append   body: {"text": "..."}   continuation within the turn
//!     POST /finalize                         drain handshake
//!     POST /stop                             cancel + drop ring
//!     GET  /health                           200 once boot finishes
//!
//! Two-endpoint design (issue #16):
//!   * `/speak` is api①: forcibly cancels any in-flight task, POSTs
//!     `/spk/stop` to drop audio-io's playback ring, **resets** the
//!     burst budget, then synthesises and pushes the new utterance.
//!   * `/append` is api②: does NOT cancel or reset, but pulls from the
//!     remaining burst budget so a turn's later sentences land in the
//!     ring without re-bursting the full prebuffer (the old single-
//!     `/speak` behaviour overflowed the ring on every continuation,
//!     causing audible glitches).
//!
//! Burst budget (BurstBudget): a per-process token-bucket-style estimate
//! of how many seconds of audio audio-io still has queued. capacity = 5 s
//! (matches the prebuffer that the old code unconditionally bursted).
//! `/speak` resets it; `/append` consumes from whatever is left after
//! realtime drain. See `BurstBudget` below.
//!
//! Concurrency: at most one synthesise→push pipeline runs at a time,
//! enforced by `speak_permit` (single-permit Semaphore). `/speak`
//! additionally aborts the in-flight task before queueing its own (so a
//! barge-in cuts mid-utterance); `/append` just queues. A cancelled
//! task surfaces as 499 Client Closed Request so the caller can
//! distinguish "interrupted" from synth/network errors.

use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Semaphore};
use tokio::task::AbortHandle;
use tokio::time::{sleep, timeout, Instant};
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

/// 499 Client Closed Request — non-standard nginx code we use to
/// distinguish a barge-in cancel from a network/synth error. Wrapped
/// here so we don't repeat the `from_u16` fallible call at each site.
const STATUS_CLIENT_CLOSED: u16 = 499;

// audio-io wire format — must match audio-io README:
//   s16le, 16 kHz, mono, 20 ms / frame = 640 bytes / frame.
const SAMPLE_RATE: u32 = 16_000;
const FRAME_MS: u64 = 20;
const BYTES_PER_FRAME: usize = (SAMPLE_RATE as usize / 1000) * FRAME_MS as usize * 2;

#[derive(Clone)]
struct Config {
    voicevox_url: String,
    voicevox_speaker: u32,
    voicevox_speed_scale: f32,
    spk_url: Option<String>,
    audio_io_base: Option<String>,
    /// Wall-clock interval (ms) between consecutive WS sends after the
    /// initial prebuffer burst. Default = 500 ms = exactly the per-batch
    /// audio length (= realtime). Set lower than 500 to overrate the
    /// wire and compensate for measured environment drift; e.g. on a
    /// WSL2/Docker host where audio-io's cpal hardware clock outpaces
    /// the streamer's wall-clock by ~10 %, set WS_PACING_MS=450 to send
    /// ~11 % faster than realtime and keep audio-io's ring topped up.
    /// Setting higher than 500 underrates → audio-io ring drains and
    /// underruns; only useful for debugging.
    ws_pacing_ms: u64,
}

impl Config {
    fn from_env() -> Result<Self> {
        Ok(Self {
            voicevox_url: env::var("VOICEVOX_URL")
                .unwrap_or_else(|_| "http://text-to-speech:50021".into()),
            voicevox_speaker: env::var("VOICEVOX_SPEAKER")
                .unwrap_or_else(|_| "3".into())
                .parse()
                .context("invalid VOICEVOX_SPEAKER")?,
            voicevox_speed_scale: env::var("VOICEVOX_SPEED_SCALE")
                .unwrap_or_else(|_| "1.0".into())
                .parse()
                .context("invalid VOICEVOX_SPEED_SCALE")?,
            spk_url: env::var("SPK_URL").ok().filter(|s| !s.is_empty()),
            audio_io_base: env::var("AUDIO_IO_BASE").ok().filter(|s| !s.is_empty()),
            ws_pacing_ms: env::var("WS_PACING_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(500),
        })
    }
}

#[derive(Clone)]
struct AppState {
    cfg: Config,
    http: reqwest::Client,
    // AbortHandle is Clone+Send and decouples cancellation from the
    // JoinHandle that the request handler awaits locally — same
    // pattern as the Python `CURRENT_TASK` global, except here only
    // the abort capability is shared while the result future stays
    // owned by the awaiting handler. This is what lets the cancelled
    // request return 499 cleanly while the new request awaits its
    // own task.
    current_task: Arc<Mutex<Option<AbortHandle>>>,
    // Single-permit semaphore around speak_one. A burst of /speak
    // requests aborts the previous task via current_task.abort(), but
    // the abort signal only takes effect at the next .await point —
    // a request mid-`reqwest::send().await` keeps running until the
    // network resolves. The semaphore makes the new request wait for
    // the previous one's permit to drop, preventing two
    // synthesize→ws-push pipelines from overlapping on VOICEVOX and
    // /spk. `/append` shares this semaphore, so a continuation queues
    // behind an in-flight `/speak` rather than racing onto the WS.
    speak_permit: Arc<Semaphore>,
    // Shared burst budget (issue #16). Tracks how much room is left in
    // audio-io's ring for a "send-as-fast-as-possible" burst before we
    // must pace at realtime. Reset by `/speak`; consumed by both
    // `/speak` and `/append`. See `BurstBudget` below.
    burst_budget: Arc<Mutex<BurstBudget>>,
}

/// Token-bucket-style accounting for audio-io's playback ring depth.
///
/// audio-io has a 10 s playback buffer (`playback_buffer_ms` default).
/// We keep the **first** 5 s of an utterance arriving as a burst (so
/// audio starts immediately without WS round-trip jitter), then pace
/// the rest at realtime. The same 5 s budget is shared across `/speak`
/// + `/append` within a turn: once it's spent, the next call can only
/// pace (no burst), preventing the buffer overflow that caused the
/// audible glitches before issue #16.
///
/// Model: `depth_s` is the ring depth at `at`. Without further sends,
/// audio-io drains at realtime so the depth decays by `now - at`
/// seconds. A burst of S seconds at time t becomes `depth_at_t + S`
/// queued, drained from t onwards. A paced span maintains `depth_s`
/// (feed = consume rate), so it only advances `at` without changing
/// `depth_s`.
struct BurstBudget {
    /// Seconds of audio in audio-io's ring as of `at`.
    depth_s: f32,
    /// Reference instant for `depth_s`.
    at: Instant,
    /// Maximum ring depth we'll push to — burst stops here. 5 s matches
    /// the prebuffer the old single-`/speak` path unconditionally sent.
    capacity_s: f32,
}

impl BurstBudget {
    fn new(capacity_s: f32) -> Self {
        Self {
            depth_s: 0.0,
            at: Instant::now(),
            capacity_s,
        }
    }

    /// Expected ring depth right now, accounting for realtime drain
    /// since `at`. Saturates at 0 (audio-io can't queue negative audio).
    fn current_depth_s(&self) -> f32 {
        let elapsed = self.at.elapsed().as_secs_f32();
        (self.depth_s - elapsed).max(0.0)
    }

    /// How many more seconds can be burst-pushed before the ring is at
    /// capacity. The next call (`/speak` or `/append`) reads this once
    /// and bursts up to that many seconds, then paces the rest.
    fn available_burst_s(&self) -> f32 {
        (self.capacity_s - self.current_depth_s()).max(0.0)
    }

    /// Record a burst send of `secs` seconds. The send is treated as
    /// instantaneous on the wire, so the entire `secs` lands in the
    /// ring immediately and starts draining from `at = now`.
    fn record_burst(&mut self, secs: f32) {
        self.depth_s = self.current_depth_s() + secs;
        self.at = Instant::now();
    }

    /// Record a paced span of `_secs` seconds at realtime cadence. The
    /// ring depth is unchanged (feed rate ≈ drain rate during pacing),
    /// but `at` jumps forward so subsequent `current_depth_s()` calls
    /// don't double-count the wall-clock spent pacing as additional
    /// drain time.
    fn record_paced(&mut self, _secs: f32) {
        self.at = Instant::now();
    }

    /// Forced reset (called by `/speak` after a `/spk/stop`). audio-io's
    /// ring has been dropped, so the budget restarts at empty.
    fn reset(&mut self) {
        self.depth_s = 0.0;
        self.at = Instant::now();
    }
}

/// Burst capacity in seconds. Must match the audio length the old code
/// unconditionally bursted (PREBUFFER_BATCHES × BATCH_MS = 10 × 500 ms
/// = 5 s). If you tune this, audit `push_to_spk`'s pacing math too.
const BURST_CAPACITY_S: f32 = 5.0;

#[derive(Deserialize)]
struct SpeakBody {
    text: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cfg = Config::from_env()?;
    let bind: SocketAddr = format!(
        "{}:{}",
        env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into()),
        env::var("PORT").unwrap_or_else(|_| "7070".into()),
    )
    .parse()
    .context("invalid HOST/PORT")?;

    info!(
        voicevox = %cfg.voicevox_url,
        spk = cfg.spk_url.as_deref().unwrap_or("(unset)"),
        speaker = cfg.voicevox_speaker,
        ws_pacing_ms = cfg.ws_pacing_ms,
        "tts-streamer starting"
    );

    let state = AppState {
        cfg,
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("build reqwest client")?,
        current_task: Arc::new(Mutex::new(None)),
        speak_permit: Arc::new(Semaphore::new(1)),
        burst_budget: Arc::new(Mutex::new(BurstBudget::new(BURST_CAPACITY_S))),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/speak", post(speak))
        .route("/append", post(append))
        .route("/finalize", post(finalize))
        .route("/stop", post(stop))
        .with_state(state);

    let listener = TcpListener::bind(bind).await.context("bind listener")?;
    info!("listening on {}", bind);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("server")?;
    Ok(())
}

// --- handlers ------------------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({"ok": true}))
}

/// api① — start of turn or barge-in.
///
/// Forcibly cancels any in-flight task, POSTs `/spk/stop` to drop
/// audio-io's playback ring, and resets the burst budget. Then
/// synthesises the new utterance and pushes from a fresh 5 s budget.
async fn speak(State(state): State<AppState>, Json(body): Json<SpeakBody>) -> Response {
    enter_speak(state, body.text, ApiMode::Speak).await
}

/// api② — continuation within the same turn.
///
/// Does NOT cancel the in-flight task or reset the budget — instead,
/// queues behind `speak_permit` and consumes whatever burst headroom
/// is left after realtime drain. Lets the orchestrator hand off
/// per-sentence utterances back-to-back without re-prebuffering 5 s
/// each time (which used to overflow audio-io's ring; see issue #16).
async fn append(State(state): State<AppState>, Json(body): Json<SpeakBody>) -> Response {
    enter_speak(state, body.text, ApiMode::Append).await
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ApiMode {
    Speak,  // api① — reset budget + cancel previous
    Append, // api② — keep current budget + no cancel
}

async fn enter_speak(state: AppState, text: String, mode: ApiMode) -> Response {
    let text = text.trim().to_string();
    if text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"ok": false, "error": "text must be a non-empty string"})),
        )
            .into_response();
    }
    if state.cfg.spk_url.is_none() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"ok": false, "error": "SPK_URL is not configured"})),
        )
            .into_response();
    }

    if mode == ApiMode::Speak {
        // Barge-in: tear down the in-flight task and the audio already
        // queued in audio-io's ring, then reset the burst budget so
        // the new utterance starts from a full 5 s headroom.
        let prev_abort = {
            let mut guard = state.current_task.lock().await;
            guard.take()
        };
        if let Some(prev) = prev_abort {
            prev.abort();
            // Drop already-buffered playback. The abort above stops
            // further frames from being WS-pushed, but audio-io still
            // has up to playback_buffer_ms of ring on the speaker —
            // without /spk/stop the user keeps hearing the cancelled
            // utterance for up to 10 s (fixes PR #15 review #27).
            if let Some(base) = state.cfg.audio_io_base.as_deref() {
                match state
                    .http
                    .post(format!("{}/spk/stop", base))
                    .timeout(Duration::from_secs(2))
                    .send()
                    .await
                {
                    Ok(_) => {}
                    Err(e) => warn!("POST /spk/stop after barge-in failed: {}", e),
                }
            }
        }
        state.burst_budget.lock().await.reset();
    }
    // Append doesn't cancel or reset — the previous task (if any) will
    // complete naturally, and speak_permit serialises us behind it.

    let task = tokio::spawn(speak_one(state.clone(), text));
    let abort = task.abort_handle();
    {
        // Store our abort handle so a subsequent /speak can cancel us.
        // For ApiMode::Speak we already take()d above so this just
        // installs ours; for ApiMode::Append the slot may hold a stale
        // handle of a task that's already running (and which we must
        // NOT abort — that's why Append doesn't call abort here).
        // Replace either way: the stored handle's purpose is "what the
        // NEXT /speak should abort", and that's us now.
        let mut guard = state.current_task.lock().await;
        guard.replace(abort);
    }

    match task.await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(e)) => {
            error!("speak failed: {:#}", e);
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({"ok": false, "error": e.to_string()})),
            )
                .into_response()
        }
        Err(je) if je.is_cancelled() => {
            // 499 Client Closed Request — surfacing the cancel as a
            // non-success status lets the orchestrator log it as
            // "cancelled" without conflating it with synth/network
            // errors.
            info!("speak cancelled (barge-in)");
            let status = StatusCode::from_u16(STATUS_CLIENT_CLOSED)
                .expect("499 is a valid HTTP status code");
            (
                status,
                Json(json!({"ok": false, "cancelled": true})),
            )
                .into_response()
        }
        Err(je) => {
            error!("speak task join error: {}", je);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"ok": false, "error": je.to_string()})),
            )
                .into_response()
        }
    }
}

async fn finalize(State(state): State<AppState>) -> Response {
    let Some(spk_url) = state.cfg.spk_url.as_deref() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"ok": false, "error": "SPK_URL not configured"})),
        )
            .into_response();
    };
    let start = std::time::Instant::now();
    match drain_handshake(spk_url).await {
        Ok(()) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            info!("finalize: drained after {:.0} ms", elapsed_ms);
            Json(json!({"ok": true, "drained_ms": (elapsed_ms * 10.0).round() / 10.0}))
                .into_response()
        }
        Err(e) => {
            error!("finalize failed: {:#}", e);
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({"ok": false, "error": e.to_string()})),
            )
                .into_response()
        }
    }
}

async fn stop(State(state): State<AppState>) -> Json<Value> {
    let cancelled = {
        let mut guard = state.current_task.lock().await;
        if let Some(prev) = guard.take() {
            prev.abort();
            true
        } else {
            false
        }
    };
    // Drop already-buffered playback. The abort above stops further
    // frames from being WS-pushed, but audio-io still has a few
    // hundred ms in its ring — /spk/stop drains it for a clean cut.
    if let Some(base) = state.cfg.audio_io_base.as_deref() {
        match state
            .http
            .post(format!("{}/spk/stop", base))
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(_) => {}
            Err(e) => warn!("POST /spk/stop failed: {}", e),
        }
    }
    // Ring is now empty (or being emptied). Resetting the budget here
    // means the next /speak or /append starts from a full 5 s headroom
    // — without it, /append-after-/stop would think the ring still
    // holds the cancelled audio and refuse to burst.
    state.burst_budget.lock().await.reset();
    Json(json!({"ok": true, "cancelled": cancelled}))
}

// --- core flow -----------------------------------------------------

async fn speak_one(state: AppState, text: String) -> Result<Value> {
    // Serialise speak_one regardless of which caller spawned us. The
    // /speak handler aborts the previous task before spawning a new
    // one, but abort only takes effect at .await boundaries — a task
    // blocked in `reqwest::send().await` finishes the in-flight HTTP
    // call before observing the cancel. Holding a permit for the
    // whole synthesise→push pipeline guarantees the old task has
    // fully released VOICEVOX and the /spk WS before the new one
    // starts, even under barge-in races.
    let _permit = state
        .speak_permit
        .clone()
        .acquire_owned()
        .await
        .context("acquire speak permit")?;
    let speaker = state.cfg.voicevox_speaker;
    info!(
        speaker,
        "speak: text={:?}",
        text.chars().take(40).collect::<String>()
    );

    let wav = synthesize(&state, &text, speaker).await?;
    let pcm = parse_wav_pcm(&wav)?;
    let duration_s = pcm.len() as f64 / (SAMPLE_RATE as f64 * 2.0);
    info!(
        wav_bytes = wav.len(),
        pcm_bytes = pcm.len(),
        duration_s,
        "synthesized"
    );

    // Idempotent /start so the playback pipeline is up before the WS
    // write. Failures here are non-fatal — older audio-io builds
    // without /start still accept frames.
    if let Some(base) = state.cfg.audio_io_base.as_deref() {
        let t = Instant::now();
        match state
            .http
            .post(format!("{}/start", base))
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(_) => {}
            Err(e) => warn!("POST /start failed ({}); continuing", e),
        }
        info!(elapsed_ms = t.elapsed().as_millis() as u64, "POST /start done");
    }

    // The /speak handler bails out before spawn if spk_url is None, so
    // this should always be Some here. Surface a clean error rather
    // than panicking if the invariant ever changes (e.g. a future
    // refactor calls speak_one from another path).
    let spk_url = state
        .cfg
        .spk_url
        .as_deref()
        .ok_or_else(|| anyhow!("SPK_URL not configured (speak_one invariant violated)"))?;
    let push_t = Instant::now();
    let sent = push_to_spk(
        spk_url,
        &pcm,
        state.cfg.ws_pacing_ms,
        state.burst_budget.clone(),
    )
    .await?;
    info!(
        elapsed_ms = push_t.elapsed().as_millis() as u64,
        "push_to_spk returned"
    );
    info!(sent, target = spk_url, "streamed");

    Ok(json!({
        "ok": true,
        "wav_bytes": wav.len(),
        "pcm_bytes": pcm.len(),
        "sent_bytes": sent,
        "duration_s": (duration_s * 1000.0).round() / 1000.0,
    }))
}

async fn synthesize(state: &AppState, text: &str, speaker: u32) -> Result<Vec<u8>> {
    // /audio_query's response is the canonical input to /synthesis.
    // Mutate three fields and leave every other key untouched so a
    // future VOICEVOX schema change doesn't trip us up:
    //   - speedScale         : env-controlled "brisker / slower" voice agent.
    //   - outputSamplingRate : ask VOICEVOX to render at 16 kHz so we
    //                          don't have to resample on the wire.
    //   - outputStereoToMono : explicit beats implicit; downstream WS is mono.
    // Together these eliminate the prior ffmpeg pipe stage — synthesis
    // output is already exactly the format /spk wants and we just
    // strip the WAV header.
    let q = state
        .http
        .post(format!("{}/audio_query", state.cfg.voicevox_url))
        .query(&[("text", text), ("speaker", &speaker.to_string())])
        .send()
        .await
        .context("POST /audio_query")?
        .error_for_status()
        .context("/audio_query non-2xx")?;
    let mut params: Value = q.json().await.context("parse /audio_query json")?;
    {
        let obj = params
            .as_object_mut()
            .ok_or_else(|| anyhow!("/audio_query response was not a JSON object"))?;
        obj.insert("outputSamplingRate".into(), json!(SAMPLE_RATE));
        obj.insert("outputStereoToMono".into(), json!(true));
        if (state.cfg.voicevox_speed_scale - 1.0).abs() > f32::EPSILON {
            obj.insert("speedScale".into(), json!(state.cfg.voicevox_speed_scale));
        }
    }

    let r = state
        .http
        .post(format!("{}/synthesis", state.cfg.voicevox_url))
        .query(&[("speaker", &speaker.to_string())])
        .json(&params)
        .send()
        .await
        .context("POST /synthesis")?
        .error_for_status()
        .context("/synthesis non-2xx")?;
    Ok(r.bytes().await.context("read /synthesis body")?.to_vec())
}

/// Validate the WAV header against (16 kHz, mono, s16le) and return
/// the data chunk's raw PCM bytes. Walks RIFF chunks rather than
/// assuming the canonical 44-byte header so a VOICEVOX upgrade that
/// adds an INFO chunk doesn't trip us up.
fn parse_wav_pcm(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        let head = String::from_utf8_lossy(&bytes[..bytes.len().min(200)]);
        bail!(
            "failed to parse VOICEVOX response as WAV; head={:?}. \
             VOICEVOX may have returned a JSON error body with content-type audio/wav",
            head
        );
    }
    let mut i = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<Vec<u8>> = None;
    while i + 8 <= bytes.len() {
        let chunk_id = &bytes[i..i + 4];
        let chunk_size = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        let body_start = i + 8;
        let body_end = body_start
            .checked_add(chunk_size)
            .ok_or_else(|| anyhow!("WAV chunk size overflow"))?;
        if body_end > bytes.len() {
            bail!("truncated WAV chunk");
        }
        match chunk_id {
            b"fmt " => {
                let body = &bytes[body_start..body_end];
                if body.len() < 16 {
                    bail!("fmt chunk too small");
                }
                let format = u16::from_le_bytes(body[0..2].try_into().unwrap());
                let channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                let sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
                fmt = Some((format, channels, sample_rate, bits));
            }
            b"data" => {
                data = Some(bytes[body_start..body_end].to_vec());
            }
            _ => {}
        }
        // RIFF chunks are word-aligned: skip a pad byte if size is odd.
        i = body_end + (chunk_size & 1);
    }
    let (format, channels, sample_rate, bits) =
        fmt.ok_or_else(|| anyhow!("WAV has no fmt chunk"))?;
    if format != 1 || channels != 1 || sample_rate != SAMPLE_RATE || bits != 16 {
        bail!(
            "VOICEVOX returned unexpected WAV: format={} channels={} sample_rate={} bits={} \
             (expected 1/1/{}/16). The engine may have ignored outputSamplingRate / \
             outputStereoToMono in the AudioQuery — check the VOICEVOX engine version.",
            format,
            channels,
            sample_rate,
            bits,
            SAMPLE_RATE
        );
    }
    data.ok_or_else(|| anyhow!("WAV has no data chunk"))
}

/// Stream `pcm` to audio-io's /spk WS, with pacing and WS send running
/// as concurrent halves of this task.
///
/// Two concerns drive the structure:
///
///   1. **Batching to 500 ms / message** (FRAMES_PER_BATCH = 25 ×
///      20 ms FRAME_MS). Amortises per-send overhead and gives the
///      pacing loop a 500 ms budget per cycle instead of 20 ms —
///      small TCP/WS hiccups no longer eat the entire window.
///   2. **Decoupling pacing from network send** via an in-task mpsc.
///      A serial `sleep_until → ws.send().await → repeat` loop
///      (everything in one future) is *sequential*: any 150 ms
///      WSL2/Docker TCP spike inside `ws.send()` blocks the next
///      `sleep_until` from even being entered, so the spike's full
///      duration is added to every later batch's arrival time at
///      audio-io. The cpal ring drains permanently and the
///      `unwrap_or(0.0)` silence fallback bleeds zero-samples →
///      audible buzz / non-smooth voice on long utterances. With a
///      channel in between, a stall just backs up 1–2 batches in the
///      queue; the pacing future keeps hitting its absolute deadlines
///      and the writer future catches up the moment the network
///      releases.
///
/// `tokio::join!` runs both futures inside the *same* task so they
/// share its cancellation (barge-in via speak_one's abort tears down
/// the WS write immediately — a detached `tokio::spawn` would leak
/// queued batches past /spk/stop).
///
/// **Burst budget (issue #16):** the first N batches are queued back-
/// to-back (the "burst" phase, lands in audio-io's ring all at once);
/// the rest are paced at realtime. N is determined dynamically by
/// `burst_budget.available_burst_s()` — `/speak` resets the budget
/// so the first call within a turn bursts the full 5 s prebuffer,
/// while `/append` continues with whatever drain has freed up. Without
/// this, every continuation re-bursted the full 5 s, overflowing
/// audio-io's 10 s playback ring and producing the audible drop-outs
/// the issue describes.
///
/// Returns when the last batch has been sent — does NOT wait for
/// audio-io's ring to drain (that's /finalize's job).
async fn push_to_spk(
    spk_url: &str,
    pcm: &[u8],
    cadence_ms: u64,
    burst_budget: Arc<Mutex<BurstBudget>>,
) -> Result<usize> {
    const FRAMES_PER_BATCH: usize = 25;
    const BYTES_PER_BATCH: usize = FRAMES_PER_BATCH * BYTES_PER_FRAME;
    /// Audio length per batch in milliseconds, derived from
    /// FRAMES_PER_BATCH × FRAME_MS = 25 × 20 = 500. Burst budget
    /// converts seconds ↔ batches using this constant.
    const BATCH_MS: u64 = (FRAMES_PER_BATCH as u64) * FRAME_MS;
    // 32 batches × 500 ms = 16 s of in-flight queue between pacing
    // and writer. Larger than any expected single-utterance wire
    // backlog so `tx.send().await` is effectively non-blocking even
    // through a multi-second WSL2 stall.
    const CHANNEL_DEPTH: usize = 32;

    let connect_t = Instant::now();
    let (ws, _resp) = tokio_tungstenite::connect_async(spk_url)
        .await
        .with_context(|| format!("connect WS {}", spk_url))?;
    info!(
        elapsed_ms = connect_t.elapsed().as_millis() as u64,
        "ws connect_async returned"
    );

    let mut iter = pcm.chunks_exact(BYTES_PER_BATCH);
    let mut batches: Vec<Vec<u8>> = iter.by_ref().map(|c| c.to_vec()).collect();
    // Pad the trailing partial batch with zeros so audio-io's
    // even-length parser accepts it and the tail isn't truncated. At
    // most ~99 ms of trailing silence; /finalize's drain handshake
    // makes the orchestrator's timing independent of this padding.
    let rem = iter.remainder();
    if !rem.is_empty() {
        let mut tail = Vec::with_capacity(BYTES_PER_BATCH);
        tail.extend_from_slice(rem);
        tail.resize(BYTES_PER_BATCH, 0);
        batches.push(tail);
    }

    // Compute how many batches go in the burst phase based on whatever
    // headroom audio-io's ring has left (issue #16). At /speak time
    // budget was just reset, so this is full capacity (= 5 s ÷ BATCH_MS
    // = 10 batches). At /append time, drain since the previous burst
    // has freed some of that back. We cap at total_batches so a short
    // utterance bursts entirely with no paced tail.
    let total_batches = batches.len();
    let available_s = burst_budget.lock().await.available_burst_s();
    let burst_batches = ((available_s * 1000.0 / BATCH_MS as f32) as usize).min(total_batches);
    let paced_batches = total_batches - burst_batches;
    let burst_secs = burst_batches as f32 * BATCH_MS as f32 / 1000.0;
    let paced_secs = paced_batches as f32 * BATCH_MS as f32 / 1000.0;
    info!(
        total_batches,
        burst_batches,
        paced_batches,
        available_s = (available_s * 100.0).round() / 100.0,
        "burst plan"
    );

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(CHANNEL_DEPTH);

    let bb_for_pacing = burst_budget.clone();
    let pacing = async move {
        let mut batches_iter = batches.into_iter();

        // Burst phase: queue the first `burst_batches` back-to-back
        // without waiting. The writer drains as fast as the WS allows
        // and these land in audio-io's ring nearly simultaneously
        // (CHANNEL_DEPTH provides slack so tx.send rarely blocks).
        for i in 0..burst_batches {
            let Some(batch) = batches_iter.next() else {
                break;
            };
            let send_t = Instant::now();
            if tx.send(batch).await.is_err() {
                warn!(batch_idx = i, "burst tx closed early — writer dead");
                return;
            }
            let blocked = send_t.elapsed();
            if blocked > Duration::from_millis(5) {
                warn!(
                    batch_idx = i,
                    blocked_ms = blocked.as_millis() as u64,
                    "burst tx.send blocked — writer behind"
                );
            }
        }
        // Account for the burst now so a /append queued behind us
        // observes the updated budget the moment our pacing yields.
        // Doing this BEFORE pacing (rather than at end of function)
        // also means /speak's `cancel_prev` reset path doesn't race
        // with our final `record_paced` after abort.
        if burst_batches > 0 {
            bb_for_pacing.lock().await.record_burst(burst_secs);
            info!(
                burst_batches,
                burst_secs = (burst_secs * 100.0).round() / 100.0,
                "burst phase done"
            );
        }

        // Pace phase: queue remaining batches at cadence_ms intervals.
        //
        // Wall-clock-anchored pacing. tokio::time::sleep_until uses
        // CLOCK_MONOTONIC (= Instant), which on WSL2 docker can lag
        // SystemTime by ~10–15% — Hyper-V VM scheduling / pause-resume
        // mechanics keep monotonic time pegged to VM-active wallclock,
        // not the host's real wallclock. Audio hardware (cpal/WASAPI
        // on Windows) runs on its own crystal at the device's reported
        // rate, so a monotonic-paced sender feeds 10–15% slower than
        // realtime and audio-io's playback ring drains continuously →
        // continuous cpal underrun (= the warn lines we get from
        // playback.rs) → the audible "ノイズ + なめらかでない音声"
        // that started after the prebuffer ran out. Polling SystemTime
        // every ≤5 ms gates each batch on the host's wallclock and
        // eliminates the drift; tokio::sleep is still monotonic for
        // the short naps but we re-check wall-clock on every iteration
        // so any monotonic-side lag gets corrected within one tick.
        //
        // Risk of SystemTime as primary: a forward NTP step (or VM
        // resume after a long pause) makes `target.duration_since(now)`
        // return Err for every queued batch, flushing the rest of the
        // utterance to the WS as fast as the writer can drain it. We
        // bound that burst on a monotonic floor below (see
        // `min_inter_send`) — even if wall-clock says "send now", we
        // never fire batches closer than ~half a cadence apart in
        // monotonic time.
        //
        // Pacing model: after the burst phase, audio-io's ring is at
        // (burst_batches * BATCH_MS) of buffered audio; the consumer
        // drains at realtime, so by t=cadence_ms the ring is one
        // cadence shorter and we top it up by one BATCH_MS. The +1 in
        // `offset_ms = (j + 1) * cadence_ms` keeps the first paced
        // send anchored one cadence after the burst end (rather than
        // at t=0), preserving that ring depth instead of letting it
        // deplete first.
        //
        // cadence_ms < BATCH_MS = overrate (each batch delivers
        // BATCH_MS of audio in cadence_ms wall-clock), which steadily
        // fills audio-io's ring above the steady state and tolerates a
        // faster-than-reported cpal hardware clock. Overrate cannot
        // "recover" the burst's head start in any short time — it only
        // catches up at (BATCH_MS - cadence_ms) per tick, by design.
        let mut start_wall = std::time::SystemTime::now();
        let start_inst = Instant::now();
        let min_inter_send = Duration::from_millis(cadence_ms / 2);
        let mut last_send_inst = start_inst;
        info!(paced_batches, "pacing start");
        let mut last_late_ms: u64 = 0;
        let mut max_late_ms: u64 = 0;
        let mut late_ticks: u32 = 0;
        let mut wall_jumps: u32 = 0;
        for (j, batch) in batches_iter.enumerate() {
            let offset_ms = (j + 1) as u64 * cadence_ms;
            let target = start_wall + Duration::from_millis(offset_ms);
            loop {
                let now_wall = std::time::SystemTime::now();
                let remaining_wall = target
                    .duration_since(now_wall)
                    .unwrap_or(Duration::ZERO);
                // Monotonic floor: protect against an unbounded burst
                // if SystemTime stepped forward (NTP / VM resume).
                let monotonic_remaining =
                    min_inter_send.saturating_sub(last_send_inst.elapsed());
                let remaining = remaining_wall.max(monotonic_remaining);
                if remaining.is_zero() {
                    break;
                }
                let nap = remaining.min(Duration::from_millis(5));
                sleep(nap).await;
            }
            if let Ok(late) = std::time::SystemTime::now().duration_since(target) {
                let late_ms = late.as_millis() as u64;
                last_late_ms = late_ms;
                if late_ms > max_late_ms {
                    max_late_ms = late_ms;
                }
                if late_ms >= 5 {
                    late_ticks += 1;
                }
                if late_ms > cadence_ms * 2 {
                    wall_jumps += 1;
                    warn!(
                        late_ms,
                        batch_idx = j,
                        "wall-clock jumped forward; re-anchoring pacing"
                    );
                    start_wall = std::time::SystemTime::now()
                        - Duration::from_millis(offset_ms);
                }
            }
            let send_t = Instant::now();
            if tx.send(batch).await.is_err() {
                break;
            }
            last_send_inst = Instant::now();
            let send_elapsed = send_t.elapsed();
            if send_elapsed > Duration::from_millis(5) {
                warn!(
                    batch_idx = j,
                    blocked_ms = send_elapsed.as_millis() as u64,
                    "pacing tx.send blocked — channel saturated, writer behind"
                );
            }
        }
        // Account for the paced portion. record_paced just snaps `at`
        // to now without changing depth_s — paced spans feed at the
        // same rate audio-io drains, so the ring depth right after
        // pacing matches the depth right after the burst.
        if paced_batches > 0 {
            bb_for_pacing.lock().await.record_paced(paced_secs);
        }
        // Both clocks reported so the operator can confirm the WSL2
        // wall-vs-monotonic skew: a healthy run has both within a few
        // ms; total_inst_ms << total_wall_ms is the smoking gun for
        // the underrun pattern this whole construction is fixing.
        let total_inst = start_inst.elapsed();
        let total_wall = start_wall.elapsed().unwrap_or_default();
        info!(
            total_inst_ms = total_inst.as_millis() as u64,
            total_wall_ms = total_wall.as_millis() as u64,
            max_late_ms,
            last_late_ms,
            late_ticks_ge_5ms = late_ticks,
            wall_jumps,
            "pacing summary"
        );
        // tx dropped on scope exit → rx.recv() returns None → writer
        // drains remaining queued batches and closes the WS cleanly.
    };

    let writer = async move {
        let mut ws = ws;
        let mut sent = 0usize;
        // Track per-batch ws.send() wall-clock and aggregate. Sustained
        // averages near or over BATCH_MS (100 ms) mean the writer is
        // the bottleneck — pacing keeps queueing into the channel
        // faster than the wire can drain. If the channel ever
        // saturates, pacing's `tx.send().await` blocks and the
        // decoupling effectively unwinds; the summary log below makes
        // that diagnosable post-hoc.
        let mut batch_idx = 0usize;
        let mut total_send: Duration = Duration::ZERO;
        let mut max_send: Duration = Duration::ZERO;
        let mut slow_sends: u32 = 0;
        while let Some(batch) = rx.recv().await {
            let n = batch.len();
            let send_start = Instant::now();
            if let Err(e) = ws.send(Message::Binary(batch)).await {
                error!("WS send batch: {}", e);
                break;
            }
            let elapsed = send_start.elapsed();
            total_send += elapsed;
            if elapsed > max_send {
                max_send = elapsed;
            }
            if elapsed > Duration::from_millis(50) {
                slow_sends += 1;
                warn!(
                    batch_idx,
                    elapsed_ms = elapsed.as_millis() as u64,
                    "slow WS send (≥ 50 ms) — likely cause of audio-io ring drain"
                );
            }
            sent += n;
            batch_idx += 1;
        }
        if batch_idx > 0 {
            let avg_ms = total_send.as_secs_f64() * 1000.0 / batch_idx as f64;
            info!(
                batches = batch_idx,
                avg_send_ms = (avg_ms * 100.0).round() / 100.0,
                max_send_ms = max_send.as_millis() as u64,
                slow_sends,
                "ws writer summary"
            );
        }
        let close_t = Instant::now();
        let _ = ws.close(None).await;
        let close_ms = close_t.elapsed().as_millis() as u64;
        info!(close_ms, "ws close completed");
        sent
    };

    let (_, sent) = tokio::join!(pacing, writer);
    Ok(sent)
}

/// Open a fresh WS to /spk, send `{"type":"eos"}`, and await the
/// matching `{"type":"drained"}`. That reply fires only when audio-io's
/// cpal output ring has been fully consumed by the device, so this
/// function's return is the precise moment the speaker fell silent.
async fn drain_handshake(spk_url: &str) -> Result<()> {
    // Bound the whole handshake on top of the per-recv timeout so a
    // misbehaving peer can't keep us alive indefinitely with a steady
    // stream of non-drained text frames. ~12 s is comfortably above
    // the worst-case real drain (10 s ring at 1× playback rate + WS
    // overhead) and short enough to surface a stuck peer.
    const DRAIN_DEADLINE: Duration = Duration::from_secs(12);
    timeout(DRAIN_DEADLINE, drain_handshake_inner(spk_url))
        .await
        .context("drain handshake overall deadline exceeded")?
}

async fn drain_handshake_inner(spk_url: &str) -> Result<()> {
    let (mut ws, _resp) = tokio_tungstenite::connect_async(spk_url)
        .await
        .with_context(|| format!("connect WS {}", spk_url))?;
    ws.send(Message::Text(json!({"type": "eos"}).to_string()))
        .await
        .context("WS send eos")?;
    loop {
        let msg = timeout(Duration::from_secs(2), ws.next())
            .await
            .context("drain recv timeout")?
            .ok_or_else(|| anyhow!("WS closed before drained"))?
            .context("WS recv")?;
        match msg {
            Message::Text(s) => {
                if let Ok(v) = serde_json::from_str::<Value>(&s) {
                    if v.get("type").and_then(|x| x.as_str()) == Some("drained") {
                        let _ = ws.close(None).await;
                        return Ok(());
                    }
                }
            }
            Message::Binary(_) | Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
            Message::Close(_) => bail!("WS closed before drained"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #16 worked example, time-compressed.
    ///
    /// Capacity 5 s, four 2 s utterances back-to-back with ~0 gap.
    /// Expected burst plan per call:
    ///   1. api①  budget reset → burst 2 s (≥ all 2 s of audio).
    ///   2. api②  available 3 s → burst 2 s.
    ///   3. api②  available 1 s → burst 1 s, pace 1 s.
    ///   4. api②  available 0 s → burst 0 s, pace 2 s.
    ///
    /// We can't sleep through realtime in a unit test, so this
    /// exercises only the budget arithmetic (`record_burst` /
    /// `available_burst_s`). The paced-span side effect is the time-
    /// advancing `record_paced` — covered separately below.
    #[test]
    fn issue_16_worked_example_burst_amounts() {
        let mut b = BurstBudget::new(5.0);
        b.reset();

        // Call 1 (api①): audio 2 s, all fits in fresh 5 s budget.
        assert!((b.available_burst_s() - 5.0).abs() < 1e-3);
        let burst1 = 2.0f32.min(b.available_burst_s());
        assert!((burst1 - 2.0).abs() < 1e-3);
        b.record_burst(burst1);

        // Call 2 (api②): immediate. depth = 2 → 3 s headroom.
        let avail2 = b.available_burst_s();
        assert!(avail2 > 2.9 && avail2 < 3.05, "avail2={avail2}");
        let burst2 = 2.0f32.min(avail2);
        assert!((burst2 - 2.0).abs() < 1e-3);
        b.record_burst(burst2);

        // Call 3 (api②): immediate. depth = 4 → 1 s headroom, paces 1 s.
        let avail3 = b.available_burst_s();
        assert!(avail3 > 0.9 && avail3 < 1.05, "avail3={avail3}");
        let burst3 = 2.0f32.min(avail3);
        assert!((burst3 - 1.0).abs() < 0.05);
        b.record_burst(burst3);
        b.record_paced(2.0 - burst3);

        // Call 4 (api②): immediate after pacing kept depth at 5 →
        // 0 headroom, full 2 s is paced.
        let avail4 = b.available_burst_s();
        assert!(avail4 < 0.05, "avail4={avail4}");
    }

    #[test]
    fn budget_decays_with_realtime_drain() {
        // Manually fast-forward `at` to simulate elapsed wall-clock.
        let mut b = BurstBudget::new(5.0);
        b.record_burst(5.0);
        assert!(b.available_burst_s() < 0.05);
        // 3 s later, audio-io has drained 3 s of the burst → 3 s headroom.
        b.at = Instant::now() - Duration::from_secs(3);
        let avail = b.available_burst_s();
        assert!(avail > 2.9 && avail < 3.05, "avail={avail}");
    }

    #[test]
    fn reset_zeros_the_depth() {
        let mut b = BurstBudget::new(5.0);
        b.record_burst(4.0);
        assert!(b.available_burst_s() < 1.05);
        b.reset();
        assert!((b.available_burst_s() - 5.0).abs() < 1e-3);
    }
}
