//! Machine candidate conversion onto the integer 20 ms grid

use serde_json::{Value, json};

use crate::{
    canonical_json::sha256_json,
    domain::{
        Activity, ActivityInterval, FRAME_SECONDS, Speaker, defects::parse_finite_number,
        frames_intersecting_seconds,
    },
    error::{ReviewError, ReviewResult},
    record::Record,
    text::Sha256Hex,
};

/// Conversion policy recorded in every proposal
pub const CONVERSION_POLICY: &str = "speakrs-open-yap-grid-proposal-v1";

const PROPOSAL_SCHEMA: &str = "speakrs-open-yap-grid-proposal";

/// Normalized machine activity bound to its candidate file
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridProposal {
    pub activity: Activity,
    pub content_sha256: Sha256Hex,
}

impl GridProposal {
    /// Convert raw candidate intervals into frame occupancy and bind the content hash
    pub fn from_candidate(
        candidate: &Value,
        window_id: &str,
        candidate_sha256: &Sha256Hex,
    ) -> ReviewResult<Self> {
        let record = Record::object(candidate, "candidate annotation")?;
        if record.get("window_id").as_str() != Some(window_id) {
            return Err(ReviewError::contract(
                "candidate annotation window_id does not match",
            ));
        }
        let Value::Array(items) = record.get("speaker_activity") else {
            return Err(ReviewError::contract(
                "candidate annotation speaker_activity must be an array",
            ));
        };

        let mut intervals = Vec::new();
        for item in items {
            let item = Record::object(item, "candidate speaker activity")?;
            // python picks speaker_role with `or`, so any falsy role falls back to speaker
            let role = Some(item.get("speaker_role"))
                .filter(|value| is_truthy(value))
                .unwrap_or(item.get("speaker"));
            let speaker = Speaker::parse(role, "speaker_role")?;
            let Value::Array(raw_intervals) = item.get("intervals") else {
                return Err(ReviewError::contract(
                    "candidate intervals must be an array",
                ));
            };
            for raw in raw_intervals {
                let raw = Record::object(raw, "candidate interval")?;
                let start = parse_finite_number(raw.get("start_seconds"), "start_seconds")?;
                let end = parse_finite_number(raw.get("end_seconds"), "end_seconds")?;
                if let Some((start_frame, end_frame)) = frames_intersecting_seconds(start, end) {
                    intervals.push(ActivityInterval::from_frames(
                        speaker,
                        start_frame,
                        end_frame,
                    )?);
                }
            }
        }

        let activity = Activity::normalize(intervals);
        let payload = json!({
            "schema": PROPOSAL_SCHEMA,
            "schema_version": 1,
            "window_id": window_id,
            "candidate_sha256": candidate_sha256.as_str(),
            "conversion_policy": CONVERSION_POLICY,
            "frame_seconds": FRAME_SECONDS,
            "intervals": activity.to_json(),
        });
        let content_sha256 = Sha256Hex::from_digest(sha256_json(&payload));
        Ok(Self {
            activity,
            content_sha256,
        })
    }
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|float| float != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::GridProposal;
    use crate::{
        domain::{ActivityInterval, Speaker},
        text::Sha256Hex,
    };

    #[test]
    fn proposal_conversion_is_content_bound_and_speaker_independent() {
        let candidate = json!({
            "window_id": "win-1",
            "speaker_activity": [
                {"speaker_role": "speaker_a", "intervals": [
                    {"start_seconds": 0.0, "end_seconds": 0.05},
                    {"start_seconds": 0.05, "end_seconds": 0.08},
                ]},
                {"speaker_role": "", "speaker": "speaker_b", "intervals": [{"start_seconds": 0.04, "end_seconds": 0.07}]},
            ],
        });
        let digest = Sha256Hex::parse_str(&"a".repeat(64), "candidate").unwrap();

        let first = GridProposal::from_candidate(&candidate, "win-1", &digest).unwrap();
        let second = GridProposal::from_candidate(&candidate, "win-1", &digest).unwrap();

        assert_eq!(first, second);
        assert_eq!(
            first.activity.intervals(),
            &[
                ActivityInterval::from_frames(Speaker::A, 0, 4).unwrap(),
                ActivityInterval::from_frames(Speaker::B, 2, 4).unwrap(),
            ]
        );
        assert!(GridProposal::from_candidate(&candidate, "win-2", &digest).is_err());
    }

    #[test]
    fn content_hash_matches_python_conversion() {
        // digest from recipes.speakrs.large.review_models.proposal_from_candidate on this input
        let candidate = json!({
            "window_id": "win-1",
            "speaker_activity": [
                {"speaker_role": "speaker_a", "intervals": [{"start_seconds": 0.0, "end_seconds": 0.05}]},
                {"speaker_role": "speaker_b", "intervals": [{"start_seconds": 29.99, "end_seconds": 31.0}]},
            ],
        });
        let digest = Sha256Hex::parse_str(&"a".repeat(64), "candidate").unwrap();

        let proposal = GridProposal::from_candidate(&candidate, "win-1", &digest).unwrap();

        assert_eq!(proposal.content_sha256.as_str(), PYTHON_CONTENT_SHA256);
    }

    const PYTHON_CONTENT_SHA256: &str =
        "2dfe19e1463b69a2d6f5a1026937140c770f8269e180c5e7f8956c0407274338";
}
