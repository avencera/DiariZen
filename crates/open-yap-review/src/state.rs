//! Derived window review state, transition rules, and session progress

use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    domain::{Decision, DefectSet, TransitionAction},
    error::{ReviewError, ReviewResult},
    text::{NonEmptyText, Sha256Hex},
};

/// The latest event that changed a window, and the reviewer of the latest decision
///
/// A defect update before any decision sets the event without a reviewer, while
/// every reviewer decision sets both, so a reviewer never exists without an event
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LastReview {
    pub event_hash: Sha256Hex,
    pub reviewer: Option<NonEmptyText>,
}

/// Derived review state for one window
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WindowReviewState {
    pub decision: Decision,
    pub defects: DefectSet,
    pub last_review: Option<LastReview>,
    pub revision: u64,
}

impl WindowReviewState {
    /// Return the next state for a transition recorded by `actor` in `event_hash`
    pub fn apply(
        &self,
        action: &TransitionAction,
        actor: &NonEmptyText,
        event_hash: &Sha256Hex,
    ) -> ReviewResult<Self> {
        let revision = self.revision + 1;
        match action {
            TransitionAction::SetDefects(defects) => {
                self.require_editable()?;
                Ok(Self {
                    decision: self.decision.clone(),
                    defects: defects.clone(),
                    last_review: Some(LastReview {
                        event_hash: event_hash.clone(),
                        reviewer: self.reviewer().cloned(),
                    }),
                    revision,
                })
            }
            TransitionAction::SignOffAccept => {
                if !self.decision.is_signoff_eligible() {
                    return Err(ReviewError::contract(
                        "sign-off accept requires confirmed, corrected, or no-speech activity",
                    ));
                }
                let (review_event_hash, reviewer) =
                    self.independent_review(actor, "sign-off accept")?;
                Ok(self.signed(
                    Decision::SignedOff {
                        review_event_hash,
                        signer: actor.clone(),
                    },
                    reviewer,
                    event_hash,
                ))
            }
            TransitionAction::SignOffReturn { reason } => {
                if !self.decision.is_unsigned_review() {
                    return Err(ReviewError::contract(
                        "sign-off return requires an unsigned review decision",
                    ));
                }
                let (review_event_hash, reviewer) =
                    self.independent_review(actor, "sign-off return")?;
                Ok(self.signed(
                    Decision::Returned {
                        review_event_hash,
                        signer: actor.clone(),
                        reason: reason.clone(),
                    },
                    reviewer,
                    event_hash,
                ))
            }
            TransitionAction::Confirm => {
                self.reviewed(Decision::ConfirmedProposal, actor, event_hash)
            }
            TransitionAction::Correct(activity) => {
                self.reviewed(Decision::Corrected(activity.clone()), actor, event_hash)
            }
            TransitionAction::NoSpeech => self.reviewed(Decision::NoSpeech, actor, event_hash),
            TransitionAction::NeedsFollowUp { reason, scope } => self.reviewed(
                Decision::NeedsFollowUp {
                    reason: reason.clone(),
                    scope: scope.clone(),
                },
                actor,
                event_hash,
            ),
            TransitionAction::Uncertain { reason, scope } => self.reviewed(
                Decision::Uncertain {
                    reason: reason.clone(),
                    scope: scope.clone(),
                },
                actor,
                event_hash,
            ),
        }
    }

    /// Return the same state content under a new revision, used by undo
    pub fn restored_at(&self, revision: u64) -> Self {
        Self {
            revision,
            ..self.clone()
        }
    }

    /// The reviewer of the latest decision
    pub fn reviewer(&self) -> Option<&NonEmptyText> {
        self.last_review
            .as_ref()
            .and_then(|last| last.reviewer.as_ref())
    }

    /// Serialize the state
    pub fn to_json(&self) -> Value {
        json!({
            "decision": self.decision.to_json(),
            "defects": self.defects.to_json(),
            "last_review_actor": self.reviewer().map(NonEmptyText::as_str),
            "last_review_event_hash": self.last_review.as_ref().map(|last| last.event_hash.as_str()),
            "revision": self.revision,
        })
    }

    fn require_editable(&self) -> ReviewResult<()> {
        if matches!(self.decision, Decision::SignedOff { .. }) {
            return Err(ReviewError::contract("signed-off windows cannot be edited"));
        }
        Ok(())
    }

    fn reviewed(
        &self,
        decision: Decision,
        actor: &NonEmptyText,
        event_hash: &Sha256Hex,
    ) -> ReviewResult<Self> {
        self.require_editable()?;
        Ok(Self {
            decision,
            defects: self.defects.clone(),
            last_review: Some(LastReview {
                event_hash: event_hash.clone(),
                reviewer: Some(actor.clone()),
            }),
            revision: self.revision + 1,
        })
    }

    fn independent_review(
        &self,
        signer: &NonEmptyText,
        label: &str,
    ) -> ReviewResult<(Sha256Hex, NonEmptyText)> {
        let Some(LastReview {
            event_hash,
            reviewer: Some(reviewer),
        }) = &self.last_review
        else {
            return Err(ReviewError::contract(format!(
                "{label} requires a prior review event"
            )));
        };
        if reviewer == signer {
            return Err(ReviewError::contract(
                "independent sign-off requires a different actor",
            ));
        }
        Ok((event_hash.clone(), reviewer.clone()))
    }

    fn signed(&self, decision: Decision, reviewer: NonEmptyText, event_hash: &Sha256Hex) -> Self {
        Self {
            decision,
            defects: self.defects.clone(),
            last_review: Some(LastReview {
                event_hash: event_hash.clone(),
                reviewer: Some(reviewer),
            }),
            revision: self.revision + 1,
        }
    }
}

/// Reviewed and signed counts for one selection group
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GroupProgress {
    pub total: usize,
    pub reviewed: usize,
    pub signed: usize,
}

/// Session-wide progress and the next window to open
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReviewProgress {
    pub window_count: usize,
    pub reviewed_count: usize,
    pub signed_count: usize,
    pub uniform: GroupProgress,
    pub targeted: GroupProgress,
    pub next_window_id: Option<String>,
}

/// One window as seen by progress accounting
#[derive(Debug, Clone, Copy)]
pub struct ProgressEntry<'a> {
    pub window_id: &'a str,
    pub uniform: bool,
    pub decision: &'a Decision,
}

impl ReviewProgress {
    /// Count windows in overlay order and choose the next window
    ///
    /// The next window is the first pending one, else the first returned one,
    /// else the first one that is not signed off
    pub fn from_entries(entries: &[ProgressEntry<'_>]) -> Self {
        let mut progress = Self {
            window_count: entries.len(),
            reviewed_count: 0,
            signed_count: 0,
            uniform: GroupProgress::default(),
            targeted: GroupProgress::default(),
            next_window_id: None,
        };
        for entry in entries {
            let reviewed = !matches!(entry.decision, Decision::Pending);
            let signed = matches!(entry.decision, Decision::SignedOff { .. });
            let group = if entry.uniform {
                &mut progress.uniform
            } else {
                &mut progress.targeted
            };
            group.total += 1;
            group.reviewed += usize::from(reviewed);
            group.signed += usize::from(signed);
            progress.reviewed_count += usize::from(reviewed);
            progress.signed_count += usize::from(signed);
        }

        let first = |predicate: fn(&Decision) -> bool| {
            entries
                .iter()
                .find(|entry| predicate(entry.decision))
                .map(|entry| entry.window_id.to_owned())
        };
        progress.next_window_id = first(|decision| matches!(decision, Decision::Pending))
            .or_else(|| first(|decision| matches!(decision, Decision::Returned { .. })))
            .or_else(|| first(|decision| !matches!(decision, Decision::SignedOff { .. })));
        progress
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ProgressEntry, ReviewProgress, WindowReviewState};
    use crate::{
        domain::{
            ActivityInterval, Decision, DefectAssessment, NonEmptyActivity, ReviewAction, Speaker,
            TransitionAction, activity::Activity,
        },
        text::{NonEmptyText, Sha256Hex},
    };

    fn actor(name: &str) -> NonEmptyText {
        NonEmptyText::parse_str(name, "actor").unwrap()
    }

    fn hash(character: char) -> Sha256Hex {
        Sha256Hex::parse_str(&character.to_string().repeat(64), "hash").unwrap()
    }

    fn transition(value: serde_json::Value) -> TransitionAction {
        match ReviewAction::parse(&value).unwrap() {
            ReviewAction::Transition(action) => action,
            ReviewAction::Undo { .. } => panic!("expected a transition"),
        }
    }

    #[test]
    fn confirm_does_not_mark_defects_clear() {
        let state = WindowReviewState::default()
            .apply(&TransitionAction::Confirm, &actor("reviewer-1"), &hash('d'))
            .unwrap();

        assert_eq!(state.decision, Decision::ConfirmedProposal);
        assert_eq!(state.defects.identity, DefectAssessment::NotReviewed);
        assert!(state.defects.has_unresolved());
        assert_eq!(state.revision, 1);
    }

    #[test]
    fn legal_and_illegal_review_transitions() {
        let pending = WindowReviewState::default();
        let rev = actor("rev");
        let signer = actor("signer");
        let activity = NonEmptyActivity::new(
            Activity::normalize([ActivityInterval::from_frames(Speaker::A, 0, 3).unwrap()]),
            "empty",
        )
        .unwrap();

        let confirmed = pending
            .apply(&TransitionAction::Confirm, &rev, &hash('1'))
            .unwrap();
        let corrected = pending
            .apply(&TransitionAction::Correct(activity), &rev, &hash('2'))
            .unwrap();
        let silent = pending
            .apply(&TransitionAction::NoSpeech, &rev, &hash('3'))
            .unwrap();
        let follow = pending
            .apply(
                &transition(json!({"kind": "needs_follow_up", "reason": "need check", "scope": {"kind": "whole_window"}})),
                &rev,
                &hash('4'),
            )
            .unwrap();
        let uncertain = pending
            .apply(
                &transition(json!({"kind": "uncertain", "reason": "hard", "scope": {"kind": "bounded_ranges", "ranges": [{"start_frame": 10, "end_frame": 20}]}})),
                &rev,
                &hash('5'),
            )
            .unwrap();
        let signed = confirmed
            .apply(&TransitionAction::SignOffAccept, &signer, &hash('6'))
            .unwrap();
        let returned = corrected
            .apply(
                &transition(json!({"kind": "sign_off_return", "reason": "fix overlap"})),
                &signer,
                &hash('7'),
            )
            .unwrap();

        assert_eq!(silent.decision, Decision::NoSpeech);
        assert_eq!(follow.decision.kind(), "needs_follow_up");
        assert_eq!(uncertain.decision.kind(), "uncertain");
        assert_eq!(
            signed.decision,
            Decision::SignedOff {
                review_event_hash: hash('1'),
                signer: signer.clone()
            }
        );
        assert_eq!(signed.reviewer(), Some(&rev));
        assert_eq!(returned.decision.kind(), "returned");

        let same_actor = confirmed
            .apply(&TransitionAction::SignOffAccept, &rev, &hash('8'))
            .unwrap_err();
        assert!(same_actor.to_string().contains("different actor"));
        let ineligible = uncertain
            .apply(&TransitionAction::SignOffAccept, &signer, &hash('9'))
            .unwrap_err();
        assert!(
            ineligible
                .to_string()
                .contains("confirmed, corrected, or no-speech")
        );
        let edit_signed = signed
            .apply(&TransitionAction::Confirm, &rev, &hash('a'))
            .unwrap_err();
        assert!(edit_signed.to_string().contains("cannot be edited"));
        let defects_signed = signed
            .apply(
                &TransitionAction::SetDefects(Default::default()),
                &rev,
                &hash('b'),
            )
            .unwrap_err();
        assert!(defects_signed.to_string().contains("cannot be edited"));
        let return_signed = signed
            .apply(
                &transition(json!({"kind": "sign_off_return", "reason": "again"})),
                &actor("other"),
                &hash('c'),
            )
            .unwrap_err();
        assert!(
            return_signed
                .to_string()
                .contains("unsigned review decision")
        );
    }

    #[test]
    fn set_defects_keeps_reviewer_and_moves_event_hash() {
        let rev = actor("rev");
        let defects_first = WindowReviewState::default()
            .apply(
                &TransitionAction::SetDefects(Default::default()),
                &rev,
                &hash('1'),
            )
            .unwrap();
        assert_eq!(defects_first.reviewer(), None);
        let signoff = defects_first.apply(
            &TransitionAction::SignOffAccept,
            &actor("signer"),
            &hash('2'),
        );
        assert!(signoff.is_err());

        let confirmed = defects_first
            .apply(&TransitionAction::Confirm, &rev, &hash('3'))
            .unwrap();
        let updated = confirmed
            .apply(
                &TransitionAction::SetDefects(Default::default()),
                &actor("someone"),
                &hash('4'),
            )
            .unwrap();
        assert_eq!(updated.reviewer(), Some(&rev));
        assert_eq!(
            updated.to_json()["last_review_event_hash"],
            hash('4').as_str()
        );
        let signed = updated
            .apply(
                &TransitionAction::SignOffAccept,
                &actor("signer"),
                &hash('5'),
            )
            .unwrap();
        assert_eq!(
            signed.decision,
            Decision::SignedOff {
                review_event_hash: hash('4'),
                signer: actor("signer")
            }
        );
    }

    #[test]
    fn progress_prefers_pending_then_returned_then_unsigned() {
        let signed = Decision::SignedOff {
            review_event_hash: hash('1'),
            signer: actor("s"),
        };
        let returned = Decision::Returned {
            review_event_hash: hash('1'),
            signer: actor("s"),
            reason: actor("r"),
        };
        let entries = [
            ProgressEntry {
                window_id: "w1",
                uniform: true,
                decision: &signed,
            },
            ProgressEntry {
                window_id: "w2",
                uniform: false,
                decision: &Decision::ConfirmedProposal,
            },
            ProgressEntry {
                window_id: "w3",
                uniform: false,
                decision: &returned,
            },
        ];

        let progress = ReviewProgress::from_entries(&entries);
        assert_eq!(progress.next_window_id.as_deref(), Some("w3"));
        assert_eq!(progress.reviewed_count, 3);
        assert_eq!(progress.signed_count, 1);
        assert_eq!(progress.uniform.signed, 1);
        assert_eq!(progress.targeted.total, 2);

        let all_signed = [ProgressEntry {
            window_id: "w1",
            uniform: true,
            decision: &signed,
        }];
        assert_eq!(
            ReviewProgress::from_entries(&all_signed).next_window_id,
            None
        );
        let pending = [
            ProgressEntry {
                window_id: "w1",
                uniform: true,
                decision: &returned,
            },
            ProgressEntry {
                window_id: "w2",
                uniform: true,
                decision: &Decision::Pending,
            },
        ];
        assert_eq!(
            ReviewProgress::from_entries(&pending)
                .next_window_id
                .as_deref(),
            Some("w2")
        );
    }
}
