//! Shared WAV helpers used by both the orchestrator and the
//! voice-activity-detection crate.
//!
//! Lives as a standalone crate — not a Cargo workspace member — so each
//! consumer keeps its own `Cargo.lock` and Docker build cache. The
//! Dockerfiles for both consumers extend their build context to the
//! repo root and `COPY wav-utils ./wav-utils` so the path dependency
//! resolves at build time.

/// Wrap a raw little-endian 16-bit mono PCM buffer in a 44-byte WAV
/// header. whisper.cpp's `/inference` endpoint accepts WAV uploads via
/// multipart upload, and the orchestrator + VAD both need to convert
/// their internal s16le buffers into a properly-headed WAV before
/// posting.
pub fn wav_from_pcm_s16le_mono(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let chunk_size = 36 + data_len;
    let byte_rate = sample_rate * 2; // mono * 2 bytes
    let block_align: u16 = 2;
    let bits_per_sample: u16 = 16;

    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&1u16.to_le_bytes()); // channels
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_layout_is_canonical() {
        let pcm = vec![0u8; 6400]; // 200 ms @ 16 kHz s16 mono
        let wav = wav_from_pcm_s16le_mono(&pcm, 16000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u32::from_le_bytes(wav[16..20].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 32_000);
        assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(
            u32::from_le_bytes(wav[40..44].try_into().unwrap()),
            pcm.len() as u32
        );
        assert_eq!(wav.len(), 44 + pcm.len());
    }

    #[test]
    fn empty_pcm_produces_44_byte_header() {
        let wav = wav_from_pcm_s16le_mono(&[], 16000);
        assert_eq!(wav.len(), 44);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 0);
    }
}
