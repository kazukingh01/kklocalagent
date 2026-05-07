//! tts-streamer: HTTP shim that turns a text request into VOICEVOX
//! synthesis + audio-io playback streaming.
//!
//! Endpoints (compatible 1:1 with the prior Python implementation):
//!     POST /speak    body: {"text": "..."}     → streams to SPK_URL
//!     POST /finalize                            → drain handshake
//!     POST /stop                                → cancel + drop
//!     GET  /health                              → 200 once boot finishes
//!
//! Concurrency: "newest wins" — at most one /speak runs at a time, and
//! a fresh /speak aborts the in-flight one before starting. The
//! cancelled task surfaces as 499 Client Closed Request so the caller
//! can distinguish "interrupted" from synth/network errors. The
//! orchestrator's per-turn TTS permit normally prevents overlapping
//! /speak in the no-barge-in path, so this cancel-on-overlap mostly
//! fires during /stop or barge-in.

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
use tokio::sync::Mutex;
use tokio::task::AbortHandle;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

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
}

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
        "tts-streamer starting"
    );

    let state = AppState {
        cfg,
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("build reqwest client")?,
        current_task: Arc::new(Mutex::new(None)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/speak", post(speak))
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

async fn speak(State(state): State<AppState>, Json(body): Json<SpeakBody>) -> Response {
    let text = body.text.trim().to_string();
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

    // Cancel any in-flight task before starting the new one. This is
    // the barge-in case (orchestrator preferring the new utterance
    // over the old reply); aborting the previous task makes the
    // previous handler's `await task` raise JoinError::is_cancelled
    // and return 499 to its caller.
    let task = tokio::spawn(speak_one(state.clone(), text));
    let abort = task.abort_handle();
    {
        let mut guard = state.current_task.lock().await;
        if let Some(prev) = guard.replace(abort) {
            prev.abort();
        }
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
            (
                StatusCode::from_u16(499).unwrap(),
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
    Json(json!({"ok": true, "cancelled": cancelled}))
}

// --- core flow -----------------------------------------------------

async fn speak_one(state: AppState, text: String) -> Result<Value> {
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
    }

    let spk_url = state.cfg.spk_url.as_deref().unwrap();
    let sent = push_to_spk(spk_url, &pcm).await?;
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

/// Stream `pcm` to audio-io's /spk WS in 20 ms frames, paced.
///
/// Returns when the last frame has been sent — does NOT wait for
/// audio-io's playback ring to drain. The drain handshake is factored
/// into /finalize so a mid-turn /speak doesn't block the next one:
/// audio-io's ring keeps feeding the cpal callback while the next
/// sentence's PCM is being pushed in, so the speaker hears continuous
/// audio with no inter-sentence silence gaps.
async fn push_to_spk(spk_url: &str, pcm: &[u8]) -> Result<usize> {
    let (mut ws, _resp) = tokio_tungstenite::connect_async(spk_url)
        .await
        .with_context(|| format!("connect WS {}", spk_url))?;

    let mut sent = 0usize;
    let mut iter = pcm.chunks_exact(BYTES_PER_FRAME);
    for chunk in iter.by_ref() {
        ws.send(Message::Binary(chunk.to_vec()))
            .await
            .context("WS send frame")?;
        sent += BYTES_PER_FRAME;
        sleep(Duration::from_millis(FRAME_MS)).await;
    }
    // Pad the trailing partial frame with zeros so the tail isn't
    // truncated — audio-io's frame parser drops short writes.
    let rem = iter.remainder();
    if !rem.is_empty() {
        let mut tail = Vec::with_capacity(BYTES_PER_FRAME);
        tail.extend_from_slice(rem);
        tail.resize(BYTES_PER_FRAME, 0);
        ws.send(Message::Binary(tail)).await.context("WS send tail")?;
        sent += BYTES_PER_FRAME;
        sleep(Duration::from_millis(FRAME_MS)).await;
    }
    let _ = ws.close(None).await;
    Ok(sent)
}

/// Open a fresh WS to /spk, send `{"type":"eos"}`, and await the
/// matching `{"type":"drained"}`. That reply fires only when audio-io's
/// cpal output ring has been fully consumed by the device, so this
/// function's return is the precise moment the speaker fell silent.
async fn drain_handshake(spk_url: &str) -> Result<()> {
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
