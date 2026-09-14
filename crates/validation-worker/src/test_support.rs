//! Test-only exact CloudDeck profile, pool, queue, and unit fixtures

use std::{collections::BTreeSet, num::NonZeroU8};

use cloudeck_core::{
    ArtifactAccessOperation, ArtifactAccessSpec, ArtifactIssuerId, ArtifactLocation,
    CapabilityDigest, ContainerImage, CpuMillis, CudaVersion, LeaseDurationSeconds, ManagedPoolId,
    ManagedPoolRequest, ManagedProvider, ManagedReusePolicy, MaximumActiveLeases, MemoryMib,
    NetworkPolicy, PayloadSchemaId, PayloadSizeLimit, PoolCapacity, QueueProtocol, QueueRequest,
    RequiredCapability, ResultPolicy, ResultSizeLimit, RetryPolicy, Sha256Digest,
    VastAiProviderProfile, WorkQueueId, WorkUnitId, WorkUnitSpec, WorkerProfile, WorkerProfileId,
    WorkerTimeoutSeconds, WorkloadRevision, capability_digest,
};
use serde::Serialize;
use uuid::Uuid;

use crate::{CLOUDECK_REVISION, default_worker_capabilities};

pub(crate) const PUBLISHED_IMAGE_DIGEST: &str =
    "db656b2639753bbc6f574d4c40a9a70677713e9566d41f325887354172e1dc3b";
const ARTIFACT_ISSUER: &str = "diarizen-validation-artifacts";

pub(crate) fn default_worker_profile(image_digest: Sha256Digest) -> WorkerProfile {
    let capabilities = default_worker_capabilities();
    WorkerProfile {
        profile_id: WorkerProfileId::new("diarizen-validation-rtx-5060-ti").expect("profile id"),
        provider: ManagedProvider::vast_ai(VastAiProviderProfile {
            gpu_name: "RTX_5060_Ti".to_owned(),
            num_gpus: NonZeroU8::new(1).expect("one GPU"),
            min_reliability_percent: 90,
            max_hourly_price_cents: std::num::NonZeroU32::new(400).expect("price"),
            disk_gb: std::num::NonZeroU16::new(64).expect("disk"),
            min_cuda: Some("12.8".parse::<CudaVersion>().expect("cuda 12.8")),
            geolocations: BTreeSet::new(),
            verified_only: true,
        }),
        image: ContainerImage::new(
            "docker.io/praveenperera/diarizen-validation-worker",
            image_digest,
        )
        .expect("image"),
        command: vec!["claim".to_owned()],
        cpu_millis: CpuMillis::new(4_000).expect("cpu"),
        memory_mib: MemoryMib::new(32_768).expect("memory"),
        timeout_seconds: WorkerTimeoutSeconds::new(4 * 60 * 60).expect("timeout"),
        network: NetworkPolicy::Egress,
        registry_auth_secret: None,
        allowed_secrets: BTreeSet::new(),
        capacity_limit: PoolCapacity::new(1).expect("capacity"),
        capability_digest: capability_digest(&capabilities),
    }
}

pub(crate) fn published_image_digest() -> Sha256Digest {
    PUBLISHED_IMAGE_DIGEST.parse().expect("published digest")
}

pub(crate) fn default_pool_request(
    queue_id: WorkQueueId,
    profile: &WorkerProfile,
) -> Result<ManagedPoolRequest, String> {
    let profile_digest = profile.validate().map_err(|error| error.to_string())?;
    Ok(ManagedPoolRequest {
        pool_id: ManagedPoolId::new(),
        queue_id,
        profile_id: profile.profile_id.clone(),
        profile_digest,
        capacity: PoolCapacity::new(1).expect("capacity one"),
        reuse: ManagedReusePolicy::QueueLifetime,
        service: None,
    })
}

#[derive(Serialize)]
struct FixtureCampaign {
    schema: &'static str,
    campaign: &'static str,
}

pub(crate) fn default_queue_request(units: &[WorkUnitSpec]) -> Result<QueueRequest, String> {
    let capabilities = default_worker_capabilities();
    let identity = capabilities
        .negotiate(3)
        .map_err(|error| error.to_string())?;
    let digest: CapabilityDigest = capability_digest(&capabilities);
    let workload_payload = serde_json::to_value(FixtureCampaign {
        schema: "diarizen-cloudeck-validation-work-payload-v1",
        campaign: "validation",
    })
    .map_err(|error| error.to_string())?;
    let issuer = ArtifactIssuerId::new(ARTIFACT_ISSUER).map_err(|error| error.to_string())?;
    Ok(QueueRequest {
        queue_id: WorkQueueId::from_uuid(Uuid::now_v7()).map_err(|error| error.to_string())?,
        workload_revision: WorkloadRevision::digest(CLOUDECK_REVISION.as_bytes()),
        manifest_digest: QueueRequest::manifest_digest(units).map_err(|error| error.to_string())?,
        payload_schema: PayloadSchemaId::new("diarizen.validation.v1")
            .map_err(|error| error.to_string())?,
        workload_payload_digest: WorkUnitSpec::digest_payload(&workload_payload)
            .map_err(|error| error.to_string())?,
        workload_payload,
        max_payload_bytes: PayloadSizeLimit::default(),
        lease_duration_seconds: LeaseDurationSeconds::new(3_600)
            .map_err(|error| error.to_string())?,
        retry_policy: RetryPolicy::new(3, true, false).map_err(|error| error.to_string())?,
        progress_unit: None,
        result_policy: ResultPolicy::RetainJson {
            max_bytes: ResultSizeLimit::default(),
        },
        deadline_unix_seconds: 2_000_000_000,
        max_active_leases: MaximumActiveLeases::new(1).map_err(|error| error.to_string())?,
        required_capability: Some(RequiredCapability {
            capability_digest: digest,
            identity,
        }),
        protocol: QueueProtocol::v3_issuer(issuer),
    })
}

pub(crate) fn validation_unit(
    ordinal: u64,
    payload: serde_json::Value,
    not_after: i64,
) -> Result<WorkUnitSpec, String> {
    let unit_id = WorkUnitId::new(format!("validation-slot-{ordinal:02}"))
        .map_err(|error| error.to_string())?;
    let manifest =
        ArtifactLocation::new(format!("r2://validation/slot-{ordinal:02}/manifest.json"))
            .map_err(|error| error.to_string())?;
    let model = ArtifactLocation::new(format!("r2://validation/slot-{ordinal:02}/model.bin"))
        .map_err(|error| error.to_string())?;
    let bundle = ArtifactLocation::new("r2://validation/frozen-dev/bundle.tar")
        .map_err(|error| error.to_string())?;
    let trainer = ArtifactLocation::new("r2://validation/frozen-dev/trainer-config.toml")
        .map_err(|error| error.to_string())?;
    let specs = vec![
        ArtifactAccessSpec::new(manifest, ArtifactAccessOperation::Read, not_after)
            .map_err(|error| error.to_string())?,
        ArtifactAccessSpec::new(model, ArtifactAccessOperation::Read, not_after)
            .map_err(|error| error.to_string())?,
        ArtifactAccessSpec::new(bundle, ArtifactAccessOperation::Read, not_after)
            .map_err(|error| error.to_string())?,
        ArtifactAccessSpec::new(trainer, ArtifactAccessOperation::Read, not_after)
            .map_err(|error| error.to_string())?,
    ];
    WorkUnitSpec::new_with_artifact_access(unit_id, ordinal, payload, None, specs)
        .map_err(|error| error.to_string())
}
