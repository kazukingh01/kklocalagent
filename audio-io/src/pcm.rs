//! Small shared PCM / sample helpers used across the audio pipeline:
//! s16le byte↔sample conversion, i16↔normalized-f32 scaling, and a wall-clock
//! epoch-ns reading. Centralized so the AEC, framer, capture and playback paths
//! share one definition instead of each re-deriving them.

use std::time::{SystemTime, UNIX_EPOCH};

/// Wall-clock nanoseconds since the Unix epoch (0 if the clock predates epoch).
pub fn epoch_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Parse little-endian s16 PCM bytes into samples (a trailing odd byte, which
/// the s16le framing never produces, is ignored).
pub fn bytes_to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|p| i16::from_le_bytes([p[0], p[1]]))
        .collect()
}

/// Serialize i16 samples to little-endian s16 PCM bytes.
pub fn i16_to_bytes(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// i16 sample → normalized f32 in roughly [-1, 1].
#[inline]
pub fn i16_to_f32(s: i16) -> f32 {
    s as f32 / 32768.0
}

/// Normalized f32 → i16 sample, clamped to the valid range.
#[inline]
pub fn f32_to_i16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0) as i16
}
