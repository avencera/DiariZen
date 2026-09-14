//! Exact CloudDeck v3 worker capability document

use cloudeck_core::{
    CancellationSupport, ProtocolVersions, ResumeSupport, SchemaVersion, SchemaVersions,
    WorkloadCapabilities, WorkloadFamily,
};

pub(crate) const MAX_RESULT_BYTES: u32 = 65_536;

/// Pushed CloudDeck v3 git revision pinned by this worker
pub const CLOUDECK_REVISION: &str = "cfcf5028606ef436b3288494b07062d5ab7760a7";
/// Workload family advertised by the evaluator worker
pub const WORKLOAD_FAMILY: &str = "diarizen-validation";
/// Default evaluator invocation inside the image
pub const DEFAULT_EVALUATOR_MODULE: &str = "recipes.speakrs.large.validation.evaluator";

/// Default evaluator command vector
#[must_use]
pub fn default_evaluator_command() -> Vec<String> {
    vec![
        "python3.10".to_owned(),
        "-m".to_owned(),
        DEFAULT_EVALUATOR_MODULE.to_owned(),
    ]
}

/// Capability document advertised by this worker image
pub fn default_worker_capabilities() -> WorkloadCapabilities {
    WorkloadCapabilities {
        protocol_versions: ProtocolVersions::new(vec![3]).expect("protocol 3 is valid"),
        workload_family: WorkloadFamily::new(WORKLOAD_FAMILY).expect("family is a valid label"),
        request_schemas: SchemaVersions::new(vec![SchemaVersion::new(1).expect("schema 1")])
            .expect("request schemas"),
        result_schemas: SchemaVersions::new(vec![SchemaVersion::new(1).expect("schema 1")])
            .expect("result schemas"),
        proof_schemas: SchemaVersions::new(vec![SchemaVersion::new(1).expect("schema 1")])
            .expect("proof schemas"),
        max_payload_bytes: 1_048_576.try_into().expect("payload capacity"),
        max_result_bytes: MAX_RESULT_BYTES.try_into().expect("result capacity"),
        progress: None,
        cancellation: CancellationSupport::Cooperative,
        resume: ResumeSupport::UnitBoundary,
    }
}
