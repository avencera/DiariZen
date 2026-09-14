//! Typed CloudDeck v3 worker for DiariZen snapshot evaluation
//!
//! The binary owns artifact download, cache verification, Python process-group
//! lifecycle, cancellation, and result identity checks. Python owns model
//! construction and metric evaluation only.

#![deny(missing_docs)]

mod bundle;
mod cache;
mod documents;
mod evaluator_request;
mod handler;
mod invocation;
mod payload;
mod process;
mod self_check;
#[cfg(test)]
mod test_support;

pub use cache::{ArtifactCache, CacheError};
pub use documents::{
    CLOUDECK_REVISION, WORKLOAD_FAMILY, default_evaluator_command, default_worker_capabilities,
};
pub use evaluator_request::{
    EvaluatorRequest, PUBLISHED_SNAPSHOT_SCHEMA, PublishedSnapshot, ResolvedArtifacts,
    VALIDATION_RESULT_SCHEMA, build_evaluator_request, validate_result,
};
pub use handler::{DeclaredReads, ValidationWorker, WorkerError, declared_reads};
pub use invocation::{InvocationError, WorkerInvocation, worker_invocation};
pub use payload::{
    ArtifactCompression, FrozenBundleRef, FutureModelArtifact, FutureModelState, SnapshotPoint,
    ValidationWorkPayload, WORK_PAYLOAD_SCHEMA,
};
pub use process::{
    EvaluatorCommand, EvaluatorOutput, ProcessError, check_evaluator_prerequisite, run_evaluator,
};
pub use self_check::{SelfCheckReport, run_self_check};

#[cfg(test)]
mod tests;
