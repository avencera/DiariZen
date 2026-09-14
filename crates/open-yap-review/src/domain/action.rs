//! Review actions carried by events

use serde_json::{Value, json};

use crate::{
    domain::{
        activity::{Activity, NonEmptyActivity},
        defects::DefectSet,
        scope::ReviewScope,
    },
    error::ReviewResult,
    record::Record,
    text::{NonEmptyText, Sha256Hex},
};

/// An action that one event records
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewAction {
    /// A state transition applied by [`crate::state::WindowReviewState::apply`]
    Transition(TransitionAction),
    /// Restore the state before the latest event of the same window
    Undo { reverted_event_hash: Sha256Hex },
}

/// A review action that moves a window state forward
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionAction {
    Confirm,
    Correct(NonEmptyActivity),
    NoSpeech,
    NeedsFollowUp {
        reason: NonEmptyText,
        scope: ReviewScope,
    },
    Uncertain {
        reason: NonEmptyText,
        scope: ReviewScope,
    },
    SetDefects(DefectSet),
    SignOffAccept,
    SignOffReturn {
        reason: NonEmptyText,
    },
}

impl ReviewAction {
    /// Parse a tagged action object
    pub fn parse(value: &Value) -> ReviewResult<Self> {
        let record = Record::object(value, "action")?;
        let transition = match record.kind() {
            Some("confirm") => {
                exact(value, "confirm action", &["kind"], &[])?;
                TransitionAction::Confirm
            }
            Some("correct") => {
                let record = exact(
                    value,
                    "correct action",
                    &["kind", "activity"],
                    &["kind", "activity"],
                )?;
                let activity = Activity::parse(record.get("activity"), "activity")?;
                TransitionAction::Correct(NonEmptyActivity::new(
                    activity,
                    "correction must contain at least one interval",
                )?)
            }
            Some("no_speech") => {
                exact(value, "no-speech action", &["kind"], &[])?;
                TransitionAction::NoSpeech
            }
            Some("needs_follow_up") => {
                let (reason, scope) = reason_and_scope(value, "needs-follow-up action")?;
                TransitionAction::NeedsFollowUp { reason, scope }
            }
            Some("uncertain") => {
                let (reason, scope) = reason_and_scope(value, "uncertain action")?;
                TransitionAction::Uncertain { reason, scope }
            }
            Some("set_defects") => {
                let record = exact(
                    value,
                    "set-defects action",
                    &["kind", "defects"],
                    &["kind", "defects"],
                )?;
                TransitionAction::SetDefects(DefectSet::parse(record.get("defects"))?)
            }
            Some("undo") => {
                const FIELDS: &[&str] = &["kind", "reverted_event_hash"];
                let record = exact(value, "undo action", FIELDS, FIELDS)?;
                let reverted_event_hash =
                    Sha256Hex::parse(record.get("reverted_event_hash"), "reverted_event_hash")?;
                return Ok(Self::Undo {
                    reverted_event_hash,
                });
            }
            Some("sign_off_accept") => {
                exact(value, "sign-off accept action", &["kind"], &[])?;
                TransitionAction::SignOffAccept
            }
            Some("sign_off_return") => {
                let record = exact(
                    value,
                    "sign-off return action",
                    &["kind", "reason"],
                    &["kind", "reason"],
                )?;
                TransitionAction::SignOffReturn {
                    reason: NonEmptyText::parse(record.get("reason"), "reason")?,
                }
            }
            _ => return Err(record.unknown_kind("action")),
        };
        Ok(Self::Transition(transition))
    }

    /// Return the wire `kind` tag
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Undo { .. } => "undo",
            Self::Transition(TransitionAction::Confirm) => "confirm",
            Self::Transition(TransitionAction::Correct(_)) => "correct",
            Self::Transition(TransitionAction::NoSpeech) => "no_speech",
            Self::Transition(TransitionAction::NeedsFollowUp { .. }) => "needs_follow_up",
            Self::Transition(TransitionAction::Uncertain { .. }) => "uncertain",
            Self::Transition(TransitionAction::SetDefects(_)) => "set_defects",
            Self::Transition(TransitionAction::SignOffAccept) => "sign_off_accept",
            Self::Transition(TransitionAction::SignOffReturn { .. }) => "sign_off_return",
        }
    }

    /// Serialize the action as stored in the event body
    pub fn to_json(&self) -> Value {
        let kind = self.kind();
        match self {
            Self::Undo {
                reverted_event_hash,
            } => {
                json!({ "kind": kind, "reverted_event_hash": reverted_event_hash.as_str() })
            }
            Self::Transition(TransitionAction::Correct(activity)) => {
                json!({ "kind": kind, "activity": activity.activity().to_json() })
            }
            Self::Transition(
                TransitionAction::NeedsFollowUp { reason, scope }
                | TransitionAction::Uncertain { reason, scope },
            ) => json!({ "kind": kind, "reason": reason.as_str(), "scope": scope.to_json() }),
            Self::Transition(TransitionAction::SetDefects(defects)) => {
                json!({ "kind": kind, "defects": defects.to_json() })
            }
            Self::Transition(TransitionAction::SignOffReturn { reason }) => {
                json!({ "kind": kind, "reason": reason.as_str() })
            }
            Self::Transition(
                TransitionAction::Confirm
                | TransitionAction::NoSpeech
                | TransitionAction::SignOffAccept,
            ) => json!({ "kind": kind }),
        }
    }
}

fn exact<'a>(
    value: &'a Value,
    label: &'a str,
    allowed: &[&str],
    required: &[&str],
) -> ReviewResult<Record<'a>> {
    Record::object(value, label)?.exact(allowed, required)
}

fn reason_and_scope(value: &Value, label: &str) -> ReviewResult<(NonEmptyText, ReviewScope)> {
    const FIELDS: &[&str] = &["kind", "reason", "scope"];
    let record = exact(value, label, FIELDS, FIELDS)?;
    let reason = NonEmptyText::parse(record.get("reason"), "reason")?;
    let scope = ReviewScope::parse(record.get("scope"))?;
    Ok((reason, scope))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ReviewAction;
    use crate::canonical_json::canonical_json;

    #[test]
    fn actions_round_trip_with_stripped_reasons_and_float_measurements() {
        let action = ReviewAction::parse(&json!({
            "kind": "set_defects",
            "defects": {
                "clock": {"kind": "unresolved", "reason": " drift ", "measurement": {"kind": "entered", "offset_seconds": 0, "drift_seconds_per_second": 1e-05}},
                "identity": {"kind": "clear"},
                "redaction": {"kind": "unresolved", "reason": "cut"},
                "synchronization": {"kind": "not_reviewed"},
            }
        }))
        .unwrap();

        assert_eq!(
            canonical_json(&action.to_json()),
            "{\"defects\":{\"clock\":{\"kind\":\"unresolved\",\"measurement\":{\"drift_seconds_per_second\":1e-05,\"kind\":\"entered\",\"offset_seconds\":0.0},\"reason\":\"drift\"},\"identity\":{\"kind\":\"clear\"},\"redaction\":{\"kind\":\"unresolved\",\"reason\":\"cut\"},\"synchronization\":{\"kind\":\"not_reviewed\"}},\"kind\":\"set_defects\"}\n"
        );
    }

    #[test]
    fn correct_normalizes_and_requires_intervals() {
        let action = ReviewAction::parse(&json!({
            "kind": "correct",
            "activity": [
                {"speaker": "speaker_b", "start_frame": 3, "end_frame": 6},
                {"speaker": "speaker_b", "start_frame": 1, "end_frame": 4},
            ]
        }))
        .unwrap();
        assert_eq!(
            action.to_json()["activity"],
            json!([{"speaker": "speaker_b", "start_frame": 1, "end_frame": 6}])
        );

        let empty = ReviewAction::parse(&json!({"kind": "correct", "activity": []})).unwrap_err();
        assert!(empty.to_string().contains("at least one interval"));
    }

    #[test]
    fn unknown_kinds_and_fields_are_rejected() {
        assert!(ReviewAction::parse(&json!({"kind": "approve"})).is_err());
        assert!(ReviewAction::parse(&json!({"kind": "confirm", "accepted": true})).is_err());
        assert!(ReviewAction::parse(&json!({"kind": "undo"})).is_err());
        assert!(ReviewAction::parse(&json!({"kind": "sign_off_return", "reason": 5})).is_err());
        assert!(
            ReviewAction::parse(&json!({"kind": "undo", "reverted_event_hash": "A".repeat(64)}))
                .is_err()
        );
    }
}
