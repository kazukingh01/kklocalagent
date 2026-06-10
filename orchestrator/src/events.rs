//! Event envelope schema for `POST /events`. Modelled as a loose record
//! (not a sealed enum) so new upstream event types don't break parsing
//! before the orchestrator learns about them; dispatch is on `name`.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct EventEnvelope {
    pub name: String,

    /// Producer-side wall-clock timestamp (seconds since epoch).
    #[serde(default)]
    pub ts: Option<f64>,

    #[serde(default)]
    pub sample_rate: Option<u32>,

    /// Base64-encoded PCM s16le mono bytes for the completed utterance.
    /// Present on `SpeechEnded` when VAD is configured to include it.
    #[serde(default)]
    pub audio_base64: Option<String>,

    // SpeechStarted
    #[serde(default)]
    pub frame_index: Option<u64>,

    // SpeechEnded
    #[serde(default)]
    pub end_frame_index: Option<u64>,
    #[serde(default)]
    pub duration_frames: Option<u64>,
    #[serde(default)]
    pub utterance_bytes: Option<u64>,

    // WakeWordDetected
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
}

impl EventEnvelope {
    /// True when the envelope carries a completed utterance with decodable audio.
    pub fn has_utterance_audio(&self) -> bool {
        self.name == "SpeechEnded"
            && self.audio_base64.is_some()
            && self.sample_rate.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_speech_ended_with_audio() {
        let json = r#"{
            "name": "SpeechEnded",
            "end_frame_index": 123,
            "duration_frames": 50,
            "utterance_bytes": 32000,
            "ts": 1744284000.5,
            "sample_rate": 16000,
            "audio_base64": "AAAA"
        }"#;
        let ev: EventEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(ev.name, "SpeechEnded");
        assert_eq!(ev.sample_rate, Some(16000));
        assert!(ev.has_utterance_audio());
    }

    #[test]
    fn parses_wake_word_detected() {
        let json = r#"{
            "name": "WakeWordDetected",
            "model": "alexa",
            "score": 0.994,
            "ts": 1744284000.5
        }"#;
        let ev: EventEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(ev.name, "WakeWordDetected");
        assert_eq!(ev.model.as_deref(), Some("alexa"));
        assert!(!ev.has_utterance_audio());
    }

    #[test]
    fn parses_unknown_event_gracefully() {
        // Forward-compat: new event names from upstream shouldn't 400 —
        // the handler logs them and moves on.
        let json = r#"{"name":"SomethingNew","extra_field":42}"#;
        let ev: EventEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(ev.name, "SomethingNew");
    }
}
