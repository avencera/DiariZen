//! Typed standalone evaluator request and result identity checks

use std::path::{Path, PathBuf};

use chrono::{DateTime, SecondsFormat, Timelike, Utc};
use cloudeck_core::Sha256Digest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::payload::ValidationWorkPayload;

/// Failure while building a request or binding a result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestError {
    /// Snapshot or result JSON did not match the typed document
    Invalid,
}

/// Schema identity of one standalone evaluator request
pub const EVALUATOR_REQUEST_SCHEMA: &str = "diarizen-standalone-evaluator-request-v1";
/// Schema identity of one published snapshot document
pub const PUBLISHED_SNAPSHOT_SCHEMA: &str = "diarizen-published-snapshot-v1";
/// Schema identity of one compact validation result
pub const VALIDATION_RESULT_SCHEMA: &str = "diarizen-validation-result-v1";

/// Immutable published snapshot written as the slot manifest
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedSnapshot {
    /// Snapshot schema identity
    pub schema: String,
    /// Content identity of this snapshot
    pub snapshot_id: Sha256Digest,
    /// Campaign identity
    pub campaign_id: Sha256Digest,
    /// Launch identity
    pub training_launch_id: String,
    /// Slot identity
    pub slot_id: Sha256Digest,
    /// Optimizer update bound to this snapshot
    pub updates: u64,
    /// SHA-256 of the published model bytes
    pub model_digest: Sha256Digest,
    /// Published model byte length
    pub model_length: u64,
    /// Frozen trainer-configuration digest
    pub trainer_configuration_digest: Sha256Digest,
    /// Frozen development-bundle identity
    pub dev_bundle_digest: Sha256Digest,
    /// Evaluator image identity
    pub evaluator_image_identity: String,
    /// Evaluator implementation digest
    pub evaluator_implementation_digest: Sha256Digest,
    /// Recovery generation identity
    pub recovery_generation_id: Sha256Digest,
    /// One-based generation sequence
    pub generation_sequence: u64,
    /// Trainer progress digest
    pub progress_digest: Sha256Digest,
    /// UTC publication time
    pub publication_time: String,
}

impl PublishedSnapshot {
    /// Parses one published snapshot document
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RequestError> {
        let snapshot: Self = serde_json::from_slice(bytes).map_err(|_| RequestError::Invalid)?;
        snapshot.validated()
    }

    fn validated(&self) -> Result<Self, RequestError> {
        if self.schema != PUBLISHED_SNAPSHOT_SCHEMA {
            return Err(RequestError::Invalid);
        }
        if self.model_length == 0
            || self.generation_sequence == 0
            || !valid_contract_string(&self.training_launch_id, 256)
            || !valid_contract_string(&self.evaluator_image_identity, 512)
        {
            return Err(RequestError::Invalid);
        }

        let mut snapshot = self.clone();
        let (publication_time, _) = parse_utc_datetime(&self.publication_time)?;
        snapshot.publication_time = publication_time;
        if snapshot.snapshot_id != snapshot.identity_digest()? {
            return Err(RequestError::Invalid);
        }

        Ok(snapshot)
    }

    /// Rejects a snapshot that does not belong to this unit payload
    pub fn bind_payload(&self, payload: &ValidationWorkPayload) -> Result<(), RequestError> {
        if self.campaign_id != payload.campaign_id
            || self.slot_id != payload.slot_id
            || self.training_launch_id != payload.training_launch_id
            || self.updates != payload.updates
            || self.trainer_configuration_digest != payload.trainer_configuration_digest
            || self.dev_bundle_digest != payload.dev_bundle_digest
            || self.evaluator_image_identity != payload.evaluator_image_identity
            || self.evaluator_implementation_digest != payload.evaluator_implementation_digest
        {
            return Err(RequestError::Invalid);
        }
        Ok(())
    }

    fn identity_digest(&self) -> Result<Sha256Digest, RequestError> {
        let identity = SnapshotIdentity::from(self);
        let canonical = serde_json::to_vec(&identity).map_err(|_| RequestError::Invalid)?;
        Ok(Sha256Digest::from_bytes(Sha256::digest(canonical).into()))
    }
}

#[derive(Serialize)]
struct SnapshotIdentity<'a> {
    campaign_id: &'a Sha256Digest,
    dev_bundle_digest: &'a Sha256Digest,
    evaluator_image_identity: &'a str,
    evaluator_implementation_digest: &'a Sha256Digest,
    generation_sequence: u64,
    model_digest: &'a Sha256Digest,
    model_length: u64,
    progress_digest: &'a Sha256Digest,
    publication_time: &'a str,
    recovery_generation_id: &'a Sha256Digest,
    slot_id: &'a Sha256Digest,
    trainer_configuration_digest: &'a Sha256Digest,
    training_launch_id: &'a str,
    updates: u64,
}

impl<'a> From<&'a PublishedSnapshot> for SnapshotIdentity<'a> {
    fn from(snapshot: &'a PublishedSnapshot) -> Self {
        Self {
            campaign_id: &snapshot.campaign_id,
            dev_bundle_digest: &snapshot.dev_bundle_digest,
            evaluator_image_identity: &snapshot.evaluator_image_identity,
            evaluator_implementation_digest: &snapshot.evaluator_implementation_digest,
            generation_sequence: snapshot.generation_sequence,
            model_digest: &snapshot.model_digest,
            model_length: snapshot.model_length,
            progress_digest: &snapshot.progress_digest,
            publication_time: &snapshot.publication_time,
            recovery_generation_id: &snapshot.recovery_generation_id,
            slot_id: &snapshot.slot_id,
            trainer_configuration_digest: &snapshot.trainer_configuration_digest,
            training_launch_id: &snapshot.training_launch_id,
            updates: snapshot.updates,
        }
    }
}

/// Local files admitted for one evaluator invocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedArtifacts {
    /// Parsed published snapshot
    pub snapshot: PublishedSnapshot,
    /// Local model file
    pub model_path: PathBuf,
    /// Extracted development-bundle root
    pub bundle_root: PathBuf,
    /// `bundle.json` inside the development-bundle root
    pub bundle_manifest_path: PathBuf,
    /// Local trainer-configuration file
    pub trainer_configuration_path: PathBuf,
}

/// Exact standalone evaluator request document
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorRequest {
    schema: String,
    snapshot: PublishedSnapshot,
    model: LocalModelArtifact,
    dev_bundle: LocalDevBundle,
    trainer_configuration: LocalTrainerConfiguration,
    evaluator_image_identity: String,
    evaluator_implementation_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalModelArtifact {
    path: String,
    digest: Sha256Digest,
    length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalDevBundle {
    manifest_path: String,
    root: String,
    digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalTrainerConfiguration {
    path: String,
    digest: Sha256Digest,
}

/// Compact evaluator result bound to a published snapshot
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationResult {
    schema: String,
    snapshot_id: Sha256Digest,
    campaign_id: Sha256Digest,
    training_launch_id: String,
    slot_id: Sha256Digest,
    updates: u64,
    model_digest: Sha256Digest,
    model_length: u64,
    trainer_configuration_digest: Sha256Digest,
    dev_bundle_digest: Sha256Digest,
    evaluator_image_identity: String,
    evaluator_implementation_digest: Sha256Digest,
    recovery_generation_id: Sha256Digest,
    generation_sequence: u64,
    progress_digest: Sha256Digest,
    publication_time: String,
    loss: f64,
    der: f64,
    false_alarm: f64,
    miss: f64,
    confusion: f64,
    started_at: String,
    completed_at: String,
}

/// Builds the exact request document accepted by `EvaluatorRequest.from_dict`
pub fn build_evaluator_request(
    payload: &ValidationWorkPayload,
    artifacts: &ResolvedArtifacts,
) -> Result<EvaluatorRequest, RequestError> {
    let snapshot = artifacts.snapshot.validated()?;
    snapshot.bind_payload(payload)?;
    if snapshot.trainer_configuration_digest != payload.trainer_configuration.content_digest {
        return Err(RequestError::Invalid);
    }
    if !artifacts
        .bundle_manifest_path
        .starts_with(&artifacts.bundle_root)
    {
        return Err(RequestError::Invalid);
    }
    Ok(EvaluatorRequest {
        schema: EVALUATOR_REQUEST_SCHEMA.to_owned(),
        snapshot: snapshot.clone(),
        model: LocalModelArtifact {
            path: path_string(&artifacts.model_path)?,
            digest: snapshot.model_digest.clone(),
            length: snapshot.model_length,
        },
        dev_bundle: LocalDevBundle {
            manifest_path: path_string(&artifacts.bundle_manifest_path)?,
            root: path_string(&artifacts.bundle_root)?,
            digest: snapshot.dev_bundle_digest.clone(),
        },
        trainer_configuration: LocalTrainerConfiguration {
            path: path_string(&artifacts.trainer_configuration_path)?,
            digest: snapshot.trainer_configuration_digest.clone(),
        },
        evaluator_image_identity: snapshot.evaluator_image_identity.clone(),
        evaluator_implementation_digest: snapshot.evaluator_implementation_digest.clone(),
    })
}

/// Parses evaluator stdout and binds it to the assigned snapshot
pub fn validate_result(
    result: &[u8],
    payload: &ValidationWorkPayload,
    snapshot: &PublishedSnapshot,
) -> Result<ValidationResult, RequestError> {
    let mut parsed: ValidationResult =
        serde_json::from_slice(result).map_err(|_| RequestError::Invalid)?;
    if parsed.schema != VALIDATION_RESULT_SCHEMA {
        return Err(RequestError::Invalid);
    }

    if !valid_contract_string(&parsed.training_launch_id, 256)
        || !valid_contract_string(&parsed.evaluator_image_identity, 512)
        || parsed.model_length == 0
        || parsed.generation_sequence == 0
        || [
            parsed.loss,
            parsed.der,
            parsed.false_alarm,
            parsed.miss,
            parsed.confusion,
        ]
        .iter()
        .any(|metric| !metric.is_finite() || *metric < 0.0)
    {
        return Err(RequestError::Invalid);
    }

    let (publication_time, _) = parse_utc_datetime(&parsed.publication_time)?;
    let (started_at, started) = parse_utc_datetime(&parsed.started_at)?;
    let (completed_at, completed) = parse_utc_datetime(&parsed.completed_at)?;
    if completed < started {
        return Err(RequestError::Invalid);
    }
    parsed.publication_time = publication_time;
    parsed.started_at = started_at;
    parsed.completed_at = completed_at;

    let result_identity = PublishedSnapshot {
        schema: PUBLISHED_SNAPSHOT_SCHEMA.to_owned(),
        snapshot_id: parsed.snapshot_id.clone(),
        campaign_id: parsed.campaign_id.clone(),
        training_launch_id: parsed.training_launch_id.clone(),
        slot_id: parsed.slot_id.clone(),
        updates: parsed.updates,
        model_digest: parsed.model_digest.clone(),
        model_length: parsed.model_length,
        trainer_configuration_digest: parsed.trainer_configuration_digest.clone(),
        dev_bundle_digest: parsed.dev_bundle_digest.clone(),
        evaluator_image_identity: parsed.evaluator_image_identity.clone(),
        evaluator_implementation_digest: parsed.evaluator_implementation_digest.clone(),
        recovery_generation_id: parsed.recovery_generation_id.clone(),
        generation_sequence: parsed.generation_sequence,
        progress_digest: parsed.progress_digest.clone(),
        publication_time: parsed.publication_time.clone(),
    }
    .validated()?;
    let snapshot = snapshot.validated()?;
    snapshot.bind_payload(payload)?;
    if result_identity != snapshot {
        return Err(RequestError::Invalid);
    }

    Ok(parsed)
}

fn valid_contract_string(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum
        && value.chars().all(|character| !character.is_control())
}

fn parse_utc_datetime(value: &str) -> Result<(String, DateTime<Utc>), RequestError> {
    let parsed = DateTime::parse_from_rfc3339(value).map_err(|_| RequestError::Invalid)?;
    if parsed.offset().local_minus_utc() != 0 || parsed.nanosecond() >= 1_000_000_000 {
        return Err(RequestError::Invalid);
    }

    let nanoseconds = parsed.nanosecond() / 1_000 * 1_000;
    let parsed = parsed
        .with_nanosecond(nanoseconds)
        .ok_or(RequestError::Invalid)?
        .with_timezone(&Utc);
    let seconds_format = if nanoseconds == 0 {
        SecondsFormat::Secs
    } else {
        SecondsFormat::Micros
    };
    Ok((parsed.to_rfc3339_opts(seconds_format, true), parsed))
}

fn path_string(path: &Path) -> Result<String, RequestError> {
    let value = path.to_str().ok_or(RequestError::Invalid)?;
    if value.is_empty() || value.contains('\0') {
        return Err(RequestError::Invalid);
    }
    Ok(value.to_owned())
}
