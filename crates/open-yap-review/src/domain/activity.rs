//! Speaker activity intervals and their normalized sets

use serde_json::{Value, json};

use crate::{
    domain::frames::{EndFrame, FrameIndex},
    error::{ReviewError, ReviewResult},
    record::Record,
};

/// One of the two frozen source speaker tracks
///
/// The derived order matches the Python sort on the `speaker_a` / `speaker_b` strings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Speaker {
    A,
    B,
}

impl Speaker {
    /// Parse `speaker_a` or `speaker_b`
    pub fn parse(value: &Value, label: &str) -> ReviewResult<Self> {
        match value.as_str() {
            Some("speaker_a") => Ok(Self::A),
            Some("speaker_b") => Ok(Self::B),
            _ => Err(ReviewError::contract_with(
                format!("{label} must be speaker_a or speaker_b"),
                json!({ "value": value }),
            )),
        }
    }

    /// Return the wire name
    pub fn as_str(self) -> &'static str {
        match self {
            Self::A => "speaker_a",
            Self::B => "speaker_b",
        }
    }
}

/// One speaker-active half-open frame range with end greater than start
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivityInterval {
    speaker: Speaker,
    start: FrameIndex,
    end: EndFrame,
}

impl ActivityInterval {
    /// Build an interval and require a positive length
    pub fn new(speaker: Speaker, start: FrameIndex, end: EndFrame) -> ReviewResult<Self> {
        if end.get() <= start.get() {
            return Err(ReviewError::contract(
                "activity interval end must be greater than start",
            ));
        }
        Ok(Self {
            speaker,
            start,
            end,
        })
    }

    /// Build an interval from raw frame numbers
    pub fn from_frames(speaker: Speaker, start: u16, end: u16) -> ReviewResult<Self> {
        Self::new(speaker, FrameIndex::new(start)?, EndFrame::new(end)?)
    }

    /// Parse one `{speaker, start_frame, end_frame}` object
    pub fn parse(value: &Value) -> ReviewResult<Self> {
        const FIELDS: &[&str] = &["speaker", "start_frame", "end_frame"];
        let record = Record::object(value, "activity interval")?.exact(FIELDS, FIELDS)?;
        let speaker = Speaker::parse(record.get("speaker"), "speaker")?;
        let start = FrameIndex::parse(record.get("start_frame"), "start_frame")?;
        let end = EndFrame::parse(record.get("end_frame"), "end_frame")?;
        Self::new(speaker, start, end)
    }

    /// The active speaker
    pub fn speaker(self) -> Speaker {
        self.speaker
    }

    /// The first active frame
    pub fn start_frame(self) -> u16 {
        self.start.get()
    }

    /// The exclusive end frame
    pub fn end_frame(self) -> u16 {
        self.end.get()
    }

    /// Serialize the interval
    pub fn to_json(self) -> Value {
        json!({
            "speaker": self.speaker.as_str(),
            "start_frame": self.start.get(),
            "end_frame": self.end.get(),
        })
    }
}

/// A normalized activity set: per-speaker merged ranges ordered by start then speaker
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Activity(Vec<ActivityInterval>);

impl Activity {
    /// Merge contiguous or overlapping ranges for the same speaker
    pub fn normalize(intervals: impl IntoIterator<Item = ActivityInterval>) -> Self {
        let mut sorted: Vec<ActivityInterval> = intervals.into_iter().collect();
        sorted.sort_by_key(|item| (item.speaker, item.start, item.end));

        let mut merged: Vec<ActivityInterval> = Vec::with_capacity(sorted.len());
        for item in sorted {
            if let Some(current) = merged.last_mut()
                && current.speaker == item.speaker
                && item.start.get() <= current.end.get()
            {
                current.end = current.end.max(item.end);
                continue;
            }
            merged.push(item);
        }
        // stable sort keeps the per-speaker order for equal (start, speaker) keys
        merged.sort_by_key(|item| (item.start, item.speaker));
        Self(merged)
    }

    /// Parse and normalize a JSON array of intervals
    pub fn parse(value: &Value, label: &str) -> ReviewResult<Self> {
        let Value::Array(items) = value else {
            return Err(ReviewError::contract(format!("{label} must be an array")));
        };
        let intervals = items
            .iter()
            .map(ActivityInterval::parse)
            .collect::<ReviewResult<Vec<_>>>()?;
        Ok(Self::normalize(intervals))
    }

    /// The normalized intervals
    pub fn intervals(&self) -> &[ActivityInterval] {
        &self.0
    }

    /// Whether no speaker is active
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Serialize the intervals as a JSON array
    pub fn to_json(&self) -> Value {
        Value::Array(self.0.iter().map(|interval| interval.to_json()).collect())
    }
}

/// A normalized activity set with at least one interval
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonEmptyActivity(Activity);

impl NonEmptyActivity {
    /// Require at least one interval, failing with the given message
    pub fn new(activity: Activity, empty_message: &str) -> ReviewResult<Self> {
        if activity.is_empty() {
            return Err(ReviewError::contract(empty_message));
        }
        Ok(Self(activity))
    }

    /// Borrow the activity
    pub fn activity(&self) -> &Activity {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Activity, ActivityInterval, Speaker};

    fn interval(speaker: Speaker, start: u16, end: u16) -> ActivityInterval {
        ActivityInterval::from_frames(speaker, start, end).unwrap()
    }

    #[test]
    fn interval_rejects_non_positive_and_out_of_window_ranges() {
        let equal = ActivityInterval::from_frames(Speaker::A, 10, 10).unwrap_err();
        assert!(equal.to_string().contains("greater than start"));
        let outside = ActivityInterval::parse(
            &json!({"speaker": "speaker_a", "start_frame": -1, "end_frame": 4}),
        )
        .unwrap_err();
        assert!(outside.to_string().contains("outside"));
        let past_end = ActivityInterval::parse(
            &json!({"speaker": "speaker_a", "start_frame": 0, "end_frame": 1501}),
        )
        .unwrap_err();
        assert!(past_end.to_string().contains("outside"));
        let fractional = ActivityInterval::parse(
            &json!({"speaker": "speaker_a", "start_frame": 0.5, "end_frame": 4}),
        )
        .unwrap_err();
        assert!(fractional.to_string().contains("integer"));
        let unknown = ActivityInterval::parse(
            &json!({"speaker": "speaker_a", "start_frame": 0, "end_frame": 4, "extra": 1}),
        )
        .unwrap_err();
        assert!(unknown.to_string().contains("unknown fields"));
    }

    #[test]
    fn normalize_merges_adjacent_same_speaker_and_keeps_overlap() {
        let merged = Activity::normalize([
            interval(Speaker::A, 0, 2),
            interval(Speaker::A, 2, 5),
            interval(Speaker::A, 8, 9),
            interval(Speaker::B, 1, 4),
            interval(Speaker::B, 3, 6),
        ]);

        assert_eq!(
            merged.intervals(),
            &[
                interval(Speaker::A, 0, 5),
                interval(Speaker::B, 1, 6),
                interval(Speaker::A, 8, 9)
            ]
        );
    }

    #[test]
    fn short_turn_and_gap_stay_distinct() {
        let intervals =
            Activity::normalize([interval(Speaker::A, 10, 11), interval(Speaker::A, 12, 14)]);

        assert_eq!(
            intervals.intervals(),
            &[interval(Speaker::A, 10, 11), interval(Speaker::A, 12, 14)]
        );
    }

    #[test]
    fn normalize_orders_ties_by_speaker_name() {
        let intervals = Activity::normalize([
            interval(Speaker::B, 0, 3),
            interval(Speaker::A, 1, 2),
            interval(Speaker::A, 0, 1),
        ]);

        assert_eq!(
            intervals.intervals(),
            &[interval(Speaker::A, 0, 2), interval(Speaker::B, 0, 3)]
        );
    }
}
