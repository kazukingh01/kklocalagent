//! Offline AEC harness for human A/B listening (issue #20).
//!
//! Runs the production [`audio_io::aec::Aec`] over WAV files and writes
//! before/after WAVs so a person can *listen* to whether the echo is gone
//! and the near-end voice survives — the verification that automated ERLE
//! asserts can't fully capture.
//!
//! Two ways to provide the near-end (mic) signal:
//!   * synthetic (default): the harness fabricates the echo as a delayed,
//!     attenuated copy of the far-end mix — no hardware needed, but the echo
//!     is a linear model, not a real room.
//!   * recorded (`--near rec.wav`): feed a real mic recording captured while
//!     audio-io played the far-end (e.g. `websocat /mic` during playback) —
//!     real acoustic echo, still offline.
//!
//! Usage:
//!   cargo run --example aec_offline -- --far tts.wav [--far2 file.wav] \
//!       [--near-voice voice.wav] [--near recorded.wav] \
//!       [--delay-ms 120] [--gain 0.5] \
//!       [--filter-ms 150] [--aec-delay-ms 120] [--out-dir /tmp/aec]
//!
//! Inputs must be 16 kHz, 16-bit PCM WAV. Mono is used as-is; stereo is
//! downmixed. Convert anything else first, e.g.
//!   ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le tts.wav
//!
//! Outputs (in --out-dir, default ./aec_out):
//!   far_mix.wav   the far-end (what the speaker played)
//!   near.wav      the AEC input (echo [+ voice]) — listen: assistant bleeds in
//!   residual.wav  the AEC output (= what /mic serves) — listen: echo gone
//!
//! The program prints ERLE (dB) and, when --near-voice is given, how much of
//! the voice survived. To *see* it, render spectrograms (printed at the end).

use std::path::Path;

use audio_io::aec::Aec;

const RATE: u32 = 16000;
const FRAME: usize = 320; // 20 ms @ 16 kHz

// --- minimal WAV I/O (std only; keeps audio-io's deps untouched) ----------

/// Read a 16-bit PCM WAV, returning mono i16 samples at its sample rate.
/// Stereo is downmixed by averaging. Errors out on non-PCM / non-16-bit.
fn read_wav_mono_i16(path: &str) -> Result<Vec<i16>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{path}: not a RIFF/WAVE file"));
    }
    // Walk chunks to find "fmt " and "data" (don't assume a 44-byte header —
    // real-world WAVs carry LIST/fact chunks before data).
    let mut pos = 12;
    let (mut channels, mut rate, mut bits) = (0u16, 0u32, 0u16);
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if body + sz > bytes.len() {
            break;
        }
        match id {
            b"fmt " => {
                let fmt = u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap());
                channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap());
                rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap());
                bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap());
                if fmt != 1 {
                    return Err(format!("{path}: not PCM (fmt={fmt})"));
                }
            }
            b"data" => data = Some(&bytes[body..body + sz]),
            _ => {}
        }
        pos = body + sz + (sz & 1); // chunks are word-aligned
    }
    if bits != 16 {
        return Err(format!("{path}: need 16-bit PCM, got {bits}-bit"));
    }
    if rate != RATE {
        return Err(format!(
            "{path}: need {RATE} Hz, got {rate} Hz — resample with `ffmpeg -ar {RATE}`"
        ));
    }
    let data = data.ok_or_else(|| format!("{path}: no data chunk"))?;
    let ch = channels.max(1) as usize;
    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|p| i16::from_le_bytes([p[0], p[1]]))
        .collect();
    if ch == 1 {
        Ok(samples)
    } else {
        // Downmix interleaved → mono by averaging the channels.
        Ok(samples
            .chunks(ch)
            .map(|f| (f.iter().map(|&s| s as i32).sum::<i32>() / ch as i32) as i16)
            .collect())
    }
}

fn write_wav_mono_i16(path: &Path, samples: &[i16]) -> Result<(), String> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + samples.len() * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&RATE.to_le_bytes());
    wav.extend_from_slice(&(RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        wav.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, wav).map_err(|e| format!("{}: {e}", path.display()))
}

// --- helpers ---------------------------------------------------------------

fn rms(s: &[i16]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let sum: f64 = s.iter().map(|&v| (v as f64).powi(2)).sum();
    (sum / s.len() as f64).sqrt()
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn arg_or<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    arg(args, key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") || arg(&args, "--far").is_none() {
        eprintln!(
            "aec_offline — run the AEC over WAVs for human A/B listening\n\n\
             required:\n  \
               --far FILE          far-end / track0 WAV (16kHz 16-bit PCM)\n\n\
             optional:\n  \
               --far2 FILE         second far-end track to mix in (track1)\n  \
               --near FILE         real mic recording (overrides synthetic echo)\n  \
               --near-voice FILE   near-end voice to mix into the synthetic echo\n  \
               --delay-ms N        synthetic echo delay (default 120)\n  \
               --gain F            synthetic echo attenuation (default 0.5)\n  \
               --filter-ms N       AEC adaptive filter length (default 150)\n  \
               --aec-delay-ms N    AEC bulk delay hint (default = --delay-ms)\n  \
               --out-dir DIR       output dir (default ./aec_out)\n\n\
             example:\n  \
               cargo run --example aec_offline -- --far tts.wav --near-voice me.wav"
        );
        return Ok(());
    }

    let delay_ms: u32 = arg_or(&args, "--delay-ms", 120);
    let gain: f32 = arg_or(&args, "--gain", 0.5);
    let filter_ms: u32 = arg_or(&args, "--filter-ms", 150);
    let aec_delay_ms: u32 = arg_or(&args, "--aec-delay-ms", delay_ms);
    let out_dir = arg(&args, "--out-dir").unwrap_or_else(|| "aec_out".into());

    // Far-end = mix of track0 (+ optional track1), the audio audio-io plays.
    let far0 = read_wav_mono_i16(&arg(&args, "--far").unwrap())?;
    let far2 = arg(&args, "--far2")
        .map(|p| read_wav_mono_i16(&p))
        .transpose()?;
    let n = far2
        .as_ref()
        .map_or(far0.len(), |f| far0.len().max(f.len()));
    let far_mix: Vec<i16> = (0..n)
        .map(|i| {
            let a = *far0.get(i).unwrap_or(&0) as i32;
            let b = far2.as_ref().and_then(|f| f.get(i)).copied().unwrap_or(0) as i32;
            (a + b).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        })
        .collect();

    // Near-end (mic): real recording if given, else synthesize the echo as a
    // delayed + attenuated copy of the far mix, plus an optional near voice.
    let voice = arg(&args, "--near-voice")
        .map(|p| read_wav_mono_i16(&p))
        .transpose()?;
    let near: Vec<i16> = if let Some(p) = arg(&args, "--near") {
        read_wav_mono_i16(&p)?
    } else {
        let delay = (RATE as usize * delay_ms as usize) / 1000;
        (0..n)
            .map(|i| {
                let echo = if i >= delay {
                    (far_mix[i - delay] as f32 * gain) as i32
                } else {
                    0
                };
                let v = voice.as_ref().and_then(|x| x.get(i)).copied().unwrap_or(0) as i32;
                (echo + v).clamp(i16::MIN as i32, i16::MAX as i32) as i16
            })
            .collect()
    };

    // Run the production AEC, frame by frame (20 ms), as the live path does.
    let mut aec = Aec::new(RATE, filter_ms, aec_delay_ms);
    let mut residual = Vec::with_capacity(near.len());
    let mut i = 0;
    while i < near.len() {
        let end = (i + FRAME).min(near.len());
        let near_f = &near[i..end];
        // Far reference aligned index-for-index with near (undelayed; the
        // AEC's internal delay line models the transport delay).
        let far_f: Vec<i16> = (i..end)
            .map(|j| far_mix.get(j).copied().unwrap_or(0))
            .collect();
        residual.extend(aec.process_frame(near_f, &far_f));
        i = end;
    }

    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{out_dir}: {e}"))?;
    let dir = Path::new(&out_dir);
    write_wav_mono_i16(&dir.join("far_mix.wav"), &far_mix)?;
    write_wav_mono_i16(&dir.join("near.wav"), &near)?;
    write_wav_mono_i16(&dir.join("residual.wav"), &residual)?;

    // ERLE only means "echo reduction" over stretches with no near voice
    // (during double-talk the residual *should* keep the voice). When we know
    // where the voice is (synthetic path with --near-voice), measure ERLE on
    // the echo-only samples; otherwise use the whole signal.
    let echo_only: Vec<usize> = match &voice {
        Some(v) => (0..near.len())
            .filter(|&i| v.get(i).copied().unwrap_or(0).abs() < 100)
            .collect(),
        None => (0..near.len()).collect(),
    };
    let pick = |s: &[i16]| -> Vec<i16> { echo_only.iter().map(|&i| s[i]).collect() };
    let near_e = rms(&pick(&near));
    let resid_e = rms(&pick(&residual));
    let erle = if resid_e > 0.0 {
        20.0 * (near_e / resid_e).log10()
    } else {
        f64::INFINITY
    };
    println!(
        "AEC offline result ({} taps, aec_delay={aec_delay_ms}ms)",
        aec.num_taps()
    );
    println!("  echo-only near RMS  : {near_e:8.1}");
    println!("  echo-only resid RMS : {resid_e:8.1}");
    println!("  ERLE (echo region)  : {erle:8.1} dB   (higher = more echo removed)");
    if let Some(v) = &voice {
        // During the voiced region, the residual should retain most of the
        // voice energy (ratio near 1.0 = voice preserved, ~0 = over-suppressed).
        let voiced: Vec<usize> = (0..near.len())
            .filter(|&i| v.get(i).copied().unwrap_or(0).abs() >= 100)
            .collect();
        if !voiced.is_empty() {
            let resid_v = rms(&voiced.iter().map(|&i| residual[i]).collect::<Vec<_>>());
            let voice_v = rms(&voiced.iter().map(|&i| v[i]).collect::<Vec<_>>());
            println!(
                "  voice retention     : {:6.2}   (resid/voice in voiced region; ~1 = kept)",
                resid_v / voice_v.max(1.0)
            );
        }
    }
    println!("\nWrote {out_dir}/{{far_mix,near,residual}}.wav");
    println!("Listen (on a machine with audio):");
    println!("  ffplay {out_dir}/near.wav       # echo (+voice) going in");
    println!("  ffplay {out_dir}/residual.wav   # echo removed, voice kept");
    println!("See (spectrogram PNGs):");
    println!("  ffmpeg -i {out_dir}/near.wav     -lavfi showspectrumpic=s=1024x512 {out_dir}/near_spec.png");
    println!("  ffmpeg -i {out_dir}/residual.wav -lavfi showspectrumpic=s=1024x512 {out_dir}/resid_spec.png");
    Ok(())
}
