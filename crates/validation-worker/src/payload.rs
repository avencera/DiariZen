//! Strict unit payload for one unpublished or published validation slot

pub use cloudeck_core::ArtifactCompression;
use cloudeck_core::{ArtifactLocation, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;
use thiserror::Error;

/// Schema identity carried by every validation work payload
pub const WORK_PAYLOAD_SCHEMA: &str = "diarizen-cloudeck-validation-work-payload-v1";

/// The exact snapshot point carried by a validation work payload
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SnapshotPoint {
    /// The snapshot before the first optimizer update
    EpochZero {},
    /// The snapshot after one complete epoch
    CompleteEpoch {
        /// One-based completed epoch number
        epoch: NonZeroU64,
    },
    /// The target snapshot after a nonempty partial epoch
    FinalPartial {
        /// Number of completed epochs before the partial epoch
        completed_epochs: u64,
        /// Number of updates in the partial epoch
        partial_updates: NonZeroU64,
    },
}

/// Closed state marker for a future model artifact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FutureModelState {
    /// The model has not been committed yet
    Uncommitted,
}

/// Typed immutable payload for one validation snapshot slot
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationWorkPayload {
    /// Payload schema identity
    pub schema: String,
    /// Campaign identity
    pub campaign_id: Sha256Digest,
    /// Slot identity
    pub slot_id: Sha256Digest,
    /// Launch identity that froze the campaign
    pub training_launch_id: String,
    /// Optimizer update bound to this slot
    pub updates: u64,
    /// Frozen trainer configuration digest
    pub trainer_configuration_digest: Sha256Digest,
    /// Evaluator image identity
    pub evaluator_image_identity: String,
    /// Evaluator implementation digest
    pub evaluator_implementation_digest: Sha256Digest,
    /// Frozen development-bundle digest
    pub dev_bundle_digest: Sha256Digest,
    /// Zero-based slot ordinal
    pub slot_ordinal: u64,
    /// Slot point document
    pub point: SnapshotPoint,
    /// Future manifest location
    pub manifest_location: ArtifactLocation,
    /// Future model location
    pub model_location: ArtifactLocation,
    /// Explicit uncommitted model marker
    pub future_model: FutureModelArtifact,
    /// Complete frozen development-bundle reference
    pub frozen_dev_bundle: FrozenBundleRef,
    /// Frozen trainer-configuration reference
    pub trainer_configuration: FrozenBundleRef,
}

/// Uncommitted future model location
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FutureModelArtifact {
    /// Must be `uncommitted`
    pub state: FutureModelState,
    /// Exact model object location
    pub location: ArtifactLocation,
}

/// Complete content-addressed frozen bundle
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenBundleRef {
    /// SHA-256 of the stored bytes
    pub content_digest: Sha256Digest,
    /// Stored byte length
    pub byte_length: u64,
    /// Media type
    pub media_type: String,
    /// Opaque storage location
    pub location: ArtifactLocation,
    /// Optional compression
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression: Option<ArtifactCompression>,
}

/// Payload parse failure
#[derive(Debug, Error)]
pub enum PayloadError {
    /// JSON did not match the typed payload
    #[error("validation work payload is invalid")]
    Invalid,
    /// Schema identity was not the v1 validation payload
    #[error("validation work payload schema is not supported")]
    UnsupportedSchema,
    /// Future model marker was not uncommitted or did not match the model location
    #[error("future model artifact is not the uncommitted slot model")]
    FutureModel,
    /// Frozen bundle digest did not match the campaign identity
    #[error("frozen development bundle digest does not match campaign identity")]
    BundleMismatch,
    /// Trainer-configuration digest did not match the campaign identity
    #[error("trainer configuration digest does not match campaign identity")]
    TrainerMismatch,
}

impl ValidationWorkPayload {
    /// Parses one strict payload document
    pub fn from_value(value: serde_json::Value) -> Result<Self, PayloadError> {
        let payload: Self = serde_json::from_value(value).map_err(|_| PayloadError::Invalid)?;
        payload.validate()?;
        Ok(payload)
    }

    fn validate(&self) -> Result<(), PayloadError> {
        if self.schema != WORK_PAYLOAD_SCHEMA {
            return Err(PayloadError::UnsupportedSchema);
        }
        if self.future_model.state != FutureModelState::Uncommitted
            || self.future_model.location != self.model_location
        {
            return Err(PayloadError::FutureModel);
        }
        if matches!(self.point, SnapshotPoint::EpochZero {}) && self.updates != 0 {
            return Err(PayloadError::Invalid);
        }
        if self.frozen_dev_bundle.content_digest != self.dev_bundle_digest {
            return Err(PayloadError::BundleMismatch);
        }
        if self.trainer_configuration.content_digest != self.trainer_configuration_digest {
            return Err(PayloadError::TrainerMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn snapshot_points_round_trip_with_wire_names() {
        let cases = [
            (SnapshotPoint::EpochZero {}, json!({"kind": "epoch_zero"})),
            (
                SnapshotPoint::CompleteEpoch {
                    epoch: NonZeroU64::new(4).unwrap(),
                },
                json!({"kind": "complete_epoch", "epoch": 4}),
            ),
            (
                SnapshotPoint::FinalPartial {
                    completed_epochs: 4,
                    partial_updates: NonZeroU64::new(7).unwrap(),
                },
                json!({
                    "kind": "final_partial",
                    "completed_epochs": 4,
                    "partial_updates": 7,
                }),
            ),
        ];

        for (point, expected) in cases {
            assert_eq!(serde_json::to_value(point).unwrap(), expected);
            assert_eq!(
                serde_json::from_value::<SnapshotPoint>(expected).unwrap(),
                point
            );
        }
    }

    #[test]
    fn snapshot_points_reject_invalid_or_unknown_wire_documents() {
        let invalid = [
            json!({"kind": "epoch_zero", "epoch": 0}),
            json!({"kind": "complete_epoch", "epoch": 0}),
            json!({"kind": "complete_epoch", "epoch": 1, "extra": true}),
            json!({
                "kind": "final_partial",
                "completed_epochs": 0,
                "partial_updates": 0,
            }),
            json!({"kind": "final_partial", "completed_epochs": -1, "partial_updates": 1}),
            json!({"kind": "unknown"}),
        ];

        for value in invalid {
            assert!(serde_json::from_value::<SnapshotPoint>(value).is_err());
        }
    }

    #[test]
    fn future_model_and_compression_use_closed_wire_enums() {
        assert_eq!(
            serde_json::to_value(FutureModelState::Uncommitted).unwrap(),
            json!("uncommitted")
        );
        assert!(serde_json::from_value::<FutureModelState>(json!("committed")).is_err());

        let reference = FrozenBundleRef {
            content_digest: Sha256Digest::from_bytes([0xab; 32]),
            byte_length: 123,
            media_type: "application/zstd".to_owned(),
            location: ArtifactLocation::new("s3://validation/bundle.tar.zst").unwrap(),
            compression: Some(ArtifactCompression::Zstd),
        };
        let encoded = serde_json::to_value(&reference).unwrap();
        assert_eq!(encoded["compression"], json!("zstd"));
        assert_eq!(
            serde_json::from_value::<FrozenBundleRef>(encoded).unwrap(),
            reference
        );

        let mut unsupported = serde_json::to_value(reference).unwrap();
        unsupported["compression"] = json!("zip");
        assert!(serde_json::from_value::<FrozenBundleRef>(unsupported).is_err());
    }
}
