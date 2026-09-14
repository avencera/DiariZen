//! Identity, clock, synchronization, and redaction assessments

use serde_json::{Value, json};

use crate::{
    error::{ReviewError, ReviewResult},
    record::Record,
    text::NonEmptyText,
};

/// A reviewer assessment of one channel defect
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DefectAssessment {
    #[default]
    NotReviewed,
    Clear,
    Unresolved {
        reason: NonEmptyText,
    },
}

impl DefectAssessment {
    fn parse(value: &Value, label: &str) -> ReviewResult<Self> {
        let record = Record::object(value, label)?;
        match record.kind() {
            Some("not_reviewed") => {
                record.exact(&["kind"], &[])?;
                Ok(Self::NotReviewed)
            }
            Some("clear") => {
                record.exact(&["kind"], &[])?;
                Ok(Self::Clear)
            }
            Some("unresolved") => {
                let record = record.exact(&["kind", "reason"], &["kind", "reason"])?;
                Ok(Self::Unresolved {
                    reason: NonEmptyText::parse(record.get("reason"), "reason")?,
                })
            }
            _ => Err(record.unknown_kind(label)),
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Self::NotReviewed => json!({ "kind": "not_reviewed" }),
            Self::Clear => json!({ "kind": "clear" }),
            Self::Unresolved { reason } => {
                json!({ "kind": "unresolved", "reason": reason.as_str() })
            }
        }
    }
}

/// Clock offset and drift, present only when the reviewer entered them
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ClockMeasurement {
    #[default]
    Absent,
    /// Entered values are stored as floats, so an integer `0` is written as `0.0`
    Entered {
        offset_seconds: f64,
        drift_seconds_per_second: f64,
    },
}

impl ClockMeasurement {
    fn parse(value: &Value) -> ReviewResult<Self> {
        let record = Record::object(value, "clock measurement")?;
        match record.kind() {
            Some("absent") => {
                record.exact(&["kind"], &[])?;
                Ok(Self::Absent)
            }
            Some("entered") => {
                const FIELDS: &[&str] = &["kind", "offset_seconds", "drift_seconds_per_second"];
                let record = record.exact(FIELDS, FIELDS)?;
                Ok(Self::Entered {
                    offset_seconds: parse_finite_number(
                        record.get("offset_seconds"),
                        "offset_seconds",
                    )?,
                    drift_seconds_per_second: parse_finite_number(
                        record.get("drift_seconds_per_second"),
                        "drift_seconds_per_second",
                    )?,
                })
            }
            _ => Err(record.unknown_kind("clock measurement")),
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Self::Absent => json!({ "kind": "absent" }),
            Self::Entered {
                offset_seconds,
                drift_seconds_per_second,
            } => json!({
                "kind": "entered",
                "offset_seconds": offset_seconds,
                "drift_seconds_per_second": drift_seconds_per_second,
            }),
        }
    }
}

// f64 fields are always finite because parsing rejects NaN and infinities
impl Eq for ClockMeasurement {}

/// Parse a finite JSON number as a float, rejecting bools
pub fn parse_finite_number(value: &Value, label: &str) -> ReviewResult<f64> {
    let Value::Number(number) = value else {
        return Err(ReviewError::contract(format!("{label} must be numeric")));
    };
    match number.as_f64() {
        Some(float) if float.is_finite() => Ok(float),
        _ => Err(ReviewError::contract(format!("{label} must be finite"))),
    }
}

/// A reviewer assessment of clock alignment
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ClockAssessment {
    #[default]
    NotReviewed,
    Clear {
        measurement: ClockMeasurement,
    },
    Unresolved {
        reason: NonEmptyText,
        measurement: ClockMeasurement,
    },
}

impl ClockAssessment {
    fn parse(value: &Value) -> ReviewResult<Self> {
        let record = Record::object(value, "clock assessment")?;
        match record.kind() {
            Some("not_reviewed") => {
                record.exact(&["kind"], &[])?;
                Ok(Self::NotReviewed)
            }
            Some("clear") => {
                let record = record.exact(&["kind", "measurement"], &["kind", "measurement"])?;
                Ok(Self::Clear {
                    measurement: ClockMeasurement::parse(record.get("measurement"))?,
                })
            }
            Some("unresolved") => {
                const FIELDS: &[&str] = &["kind", "reason", "measurement"];
                let record = record.exact(FIELDS, FIELDS)?;
                Ok(Self::Unresolved {
                    reason: NonEmptyText::parse(record.get("reason"), "reason")?,
                    measurement: ClockMeasurement::parse(record.get("measurement"))?,
                })
            }
            _ => Err(record.unknown_kind("clock assessment")),
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Self::NotReviewed => json!({ "kind": "not_reviewed" }),
            Self::Clear { measurement } => {
                json!({ "kind": "clear", "measurement": measurement.to_json() })
            }
            Self::Unresolved {
                reason,
                measurement,
            } => json!({
                "kind": "unresolved",
                "reason": reason.as_str(),
                "measurement": measurement.to_json(),
            }),
        }
    }
}

/// The four channel assessments recorded per window
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DefectSet {
    pub identity: DefectAssessment,
    pub clock: ClockAssessment,
    pub synchronization: DefectAssessment,
    pub redaction: DefectAssessment,
}

impl DefectSet {
    /// Parse the `{identity, clock, synchronization, redaction}` object
    pub fn parse(value: &Value) -> ReviewResult<Self> {
        const FIELDS: &[&str] = &["identity", "clock", "synchronization", "redaction"];
        let record = Record::object(value, "defects")?.exact(FIELDS, FIELDS)?;
        Ok(Self {
            identity: DefectAssessment::parse(record.get("identity"), "identity")?,
            clock: ClockAssessment::parse(record.get("clock"))?,
            synchronization: DefectAssessment::parse(
                record.get("synchronization"),
                "synchronization",
            )?,
            redaction: DefectAssessment::parse(record.get("redaction"), "redaction")?,
        })
    }

    /// Whether any assessment is unresolved or not reviewed
    pub fn has_unresolved(&self) -> bool {
        let channels = [&self.identity, &self.synchronization, &self.redaction];
        channels
            .iter()
            .any(|item| **item != DefectAssessment::Clear)
            || !matches!(self.clock, ClockAssessment::Clear { .. })
    }

    /// Serialize the four assessments
    pub fn to_json(&self) -> Value {
        json!({
            "identity": self.identity.to_json(),
            "clock": self.clock.to_json(),
            "synchronization": self.synchronization.to_json(),
            "redaction": self.redaction.to_json(),
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ClockAssessment, ClockMeasurement, DefectAssessment, DefectSet};

    #[test]
    fn clock_measurement_is_present_only_when_entered() {
        let payload = json!({
            "identity": {"kind": "clear"},
            "clock": {"kind": "clear", "measurement": {"kind": "entered", "offset_seconds": 0.02, "drift_seconds_per_second": 0}},
            "synchronization": {"kind": "clear"},
            "redaction": {"kind": "unresolved", "reason": " possible cut "},
        });
        let defects = DefectSet::parse(&payload).unwrap();

        assert_eq!(
            defects.clock,
            ClockAssessment::Clear {
                measurement: ClockMeasurement::Entered {
                    offset_seconds: 0.02,
                    drift_seconds_per_second: 0.0
                }
            }
        );
        let DefectAssessment::Unresolved { reason } = &defects.redaction else {
            panic!("redaction")
        };
        assert_eq!(reason.as_str(), "possible cut");
        assert!(defects.has_unresolved());
        let mut extra = payload.clone();
        extra["extra"] = json!(1);
        assert!(
            DefectSet::parse(&extra)
                .unwrap_err()
                .to_string()
                .contains("unknown fields")
        );
    }

    #[test]
    fn default_defects_are_unresolved_and_all_clear_is_not() {
        assert!(DefectSet::default().has_unresolved());
        let clear = DefectSet {
            identity: DefectAssessment::Clear,
            clock: ClockAssessment::Clear {
                measurement: ClockMeasurement::Absent,
            },
            synchronization: DefectAssessment::Clear,
            redaction: DefectAssessment::Clear,
        };
        assert!(!clear.has_unresolved());
    }

    #[test]
    fn reason_must_be_a_string() {
        let payload = json!({
            "identity": {"kind": "unresolved", "reason": null},
            "clock": {"kind": "not_reviewed"},
            "synchronization": {"kind": "not_reviewed"},
            "redaction": {"kind": "not_reviewed"},
        });

        assert!(DefectSet::parse(&payload).is_err());
    }
}
