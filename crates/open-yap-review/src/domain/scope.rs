//! Whole-window or bounded-range scope for follow-up and uncertainty

use serde_json::{Value, json};

use crate::{
    domain::frames::{EndFrame, FrameIndex},
    error::{ReviewError, ReviewResult},
    record::Record,
};

/// A half-open frame range with end greater than start
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRange {
    start: FrameIndex,
    end: EndFrame,
}

impl FrameRange {
    fn parse(value: &Value) -> ReviewResult<Self> {
        const FIELDS: &[&str] = &["start_frame", "end_frame"];
        let record = Record::object(value, "bounded range")?.exact(FIELDS, FIELDS)?;
        let start = FrameIndex::parse(record.get("start_frame"), "range.start_frame")?;
        let end = EndFrame::parse(record.get("end_frame"), "range.end_frame")?;
        if end.get() <= start.get() {
            return Err(ReviewError::contract(
                "bounded range end must be greater than start",
            ));
        }
        Ok(Self { start, end })
    }

    fn to_json(self) -> Value {
        json!({ "start_frame": self.start.get(), "end_frame": self.end.get() })
    }
}

/// The part of a window a follow-up or uncertainty applies to
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewScope {
    /// The whole window
    WholeWindow,
    /// Explicit frame ranges, never empty
    BoundedRanges(Vec<FrameRange>),
}

impl ReviewScope {
    /// Parse a tagged scope object
    pub fn parse(value: &Value) -> ReviewResult<Self> {
        let record = Record::object(value, "review scope")?;
        match record.kind() {
            Some("whole_window") => {
                Record::object(value, "whole-window scope")?.exact(&["kind"], &[])?;
                Ok(Self::WholeWindow)
            }
            Some("bounded_ranges") => {
                const FIELDS: &[&str] = &["kind", "ranges"];
                let record = Record::object(value, "bounded-range scope")?.exact(FIELDS, FIELDS)?;
                let Value::Array(items) = record.get("ranges") else {
                    return Err(ReviewError::contract("bounded ranges must be an array"));
                };
                let ranges = items
                    .iter()
                    .map(FrameRange::parse)
                    .collect::<ReviewResult<Vec<_>>>()?;
                if ranges.is_empty() {
                    return Err(ReviewError::contract(
                        "bounded range scope needs at least one range",
                    ));
                }
                Ok(Self::BoundedRanges(ranges))
            }
            _ => Err(record.unknown_kind("review scope")),
        }
    }

    /// Serialize the scope
    pub fn to_json(&self) -> Value {
        match self {
            Self::WholeWindow => json!({ "kind": "whole_window" }),
            Self::BoundedRanges(ranges) => json!({
                "kind": "bounded_ranges",
                "ranges": ranges.iter().map(|range| range.to_json()).collect::<Vec<_>>(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ReviewScope;

    #[test]
    fn bounded_scope_needs_valid_ranges() {
        let scope = ReviewScope::parse(
            &json!({"kind": "bounded_ranges", "ranges": [{"start_frame": 10, "end_frame": 20}]}),
        )
        .unwrap();
        assert_eq!(scope.to_json()["ranges"][0]["end_frame"], 20);
        assert!(ReviewScope::parse(&json!({"kind": "bounded_ranges", "ranges": []})).is_err());
        assert!(
            ReviewScope::parse(
                &json!({"kind": "bounded_ranges", "ranges": [{"start_frame": 5, "end_frame": 5}]})
            )
            .is_err()
        );
        assert!(ReviewScope::parse(&json!({"kind": "whole_window", "x": 1})).is_err());
        assert!(ReviewScope::parse(&json!({"kind": "everything"})).is_err());
    }
}
