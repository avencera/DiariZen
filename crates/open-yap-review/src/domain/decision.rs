//! The activity decision recorded for one window

use serde_json::{Value, json};

use crate::{
    domain::{
        activity::{Activity, NonEmptyActivity},
        scope::ReviewScope,
    },
    text::{NonEmptyText, Sha256Hex},
};

/// The current activity decision of a window
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Decision {
    /// No accepted human activity decision exists yet
    #[default]
    Pending,
    /// The reviewer accepted the grid proposal after listening
    ConfirmedProposal,
    /// The reviewer replaced the proposal with edited activity
    Corrected(NonEmptyActivity),
    /// The reviewer marked the window as containing no speech
    NoSpeech,
    /// The reviewer recorded a reasoned follow-up
    NeedsFollowUp {
        reason: NonEmptyText,
        scope: ReviewScope,
    },
    /// The reviewer could not decide the activity
    Uncertain {
        reason: NonEmptyText,
        scope: ReviewScope,
    },
    /// An independent signer accepted the reviewed activity
    SignedOff {
        review_event_hash: Sha256Hex,
        signer: NonEmptyText,
    },
    /// An independent signer returned the window for correction
    Returned {
        review_event_hash: Sha256Hex,
        signer: NonEmptyText,
        reason: NonEmptyText,
    },
}

impl Decision {
    /// Return the wire `kind` tag
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::ConfirmedProposal => "confirmed_proposal",
            Self::Corrected(_) => "corrected",
            Self::NoSpeech => "no_speech",
            Self::NeedsFollowUp { .. } => "needs_follow_up",
            Self::Uncertain { .. } => "uncertain",
            Self::SignedOff { .. } => "signed_off",
            Self::Returned { .. } => "returned",
        }
    }

    /// Whether a reviewer decision awaits independent sign-off
    pub fn is_unsigned_review(&self) -> bool {
        matches!(
            self,
            Self::ConfirmedProposal
                | Self::Corrected(_)
                | Self::NoSpeech
                | Self::NeedsFollowUp { .. }
                | Self::Uncertain { .. }
        )
    }

    /// Whether the decision states concrete activity a signer may accept
    pub fn is_signoff_eligible(&self) -> bool {
        matches!(
            self,
            Self::ConfirmedProposal | Self::Corrected(_) | Self::NoSpeech
        )
    }

    /// Return the activity this decision states, given the window proposal
    ///
    /// Sign-off does not embed activity, so callers resolve signed windows
    /// through the state that was signed
    pub fn stated_activity(&self, proposal: &Activity) -> Option<Activity> {
        match self {
            Self::ConfirmedProposal => Some(proposal.clone()),
            Self::Corrected(activity) => Some(activity.activity().clone()),
            Self::NoSpeech => Some(Activity::default()),
            _ => None,
        }
    }

    /// Serialize the decision
    pub fn to_json(&self) -> Value {
        match self {
            Self::Pending => json!({ "kind": "pending" }),
            Self::ConfirmedProposal => json!({ "kind": "confirmed_proposal" }),
            Self::Corrected(activity) => {
                json!({ "kind": "corrected", "activity": activity.activity().to_json() })
            }
            Self::NoSpeech => json!({ "kind": "no_speech" }),
            Self::NeedsFollowUp { reason, scope } => json!({
                "kind": "needs_follow_up",
                "reason": reason.as_str(),
                "scope": scope.to_json(),
            }),
            Self::Uncertain { reason, scope } => json!({
                "kind": "uncertain",
                "reason": reason.as_str(),
                "scope": scope.to_json(),
            }),
            Self::SignedOff {
                review_event_hash,
                signer,
            } => json!({
                "kind": "signed_off",
                "review_event_hash": review_event_hash.as_str(),
                "signer": signer.as_str(),
            }),
            Self::Returned {
                review_event_hash,
                signer,
                reason,
            } => json!({
                "kind": "returned",
                "review_event_hash": review_event_hash.as_str(),
                "signer": signer.as_str(),
                "reason": reason.as_str(),
            }),
        }
    }
}
