//! Blocking CloudDeck workload handler for one validation unit

use std::{
    fs,
    future::Future,
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use cloudeck_core::{
    ArtifactAccessSpec, ArtifactAccessUrl, ArtifactLocation, Sha256Digest, WorkAssignment,
    WorkFailureCategory, WorkFailureReason, WorkloadCapabilities,
};
use cloudeck_worker_runtime::{
    BlockingWorkContext, BlockingWorkloadHandler, WorkHandlerFailure, WorkHandlerResult,
};
use thiserror::Error;

use crate::{
    bundle::ExtractedBundle,
    cache::ArtifactCache,
    documents::default_worker_capabilities,
    evaluator_request::{
        PublishedSnapshot, ResolvedArtifacts, build_evaluator_request, validate_result,
    },
    payload::ValidationWorkPayload,
    process::{EvaluatorCommand, check_evaluator_prerequisite, run_evaluator, write_request_file},
};

const DOWNLOAD_READ_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_RETRY_DELAY: Duration = Duration::from_millis(250);
const MANIFEST_MAX_BYTES: u64 = 65_536;
const MAX_AUTHORIZATION_REFRESHES: usize = 3;

/// Worker configuration for cache, evaluator, and downloads
#[derive(Debug, Clone)]
pub struct ValidationWorker {
    cache: ArtifactCache,
    evaluator: EvaluatorCommand,
    client: reqwest::Client,
}

/// Handler failure
#[derive(Debug, Error)]
pub enum WorkerError {
    /// Unit payload was not a validation document
    #[error("validation unit payload is invalid")]
    Payload,
    /// Assignment did not declare the required read specs
    #[error(
        "assignment does not declare manifest, model, development-bundle, and trainer-configuration reads"
    )]
    Undeclared,
    /// Artifact resolution or download failed terminally
    #[error("artifact resolution failed")]
    Artifact,
    /// Cache verification failed
    #[error("artifact cache verification failed")]
    Cache,
    /// Frozen development archive did not unpack to a bundle tree
    #[error("frozen development bundle layout is invalid")]
    Layout,
    /// Evaluator process failed
    #[error("evaluator process failed")]
    Process,
    /// Result JSON was not one compact identity-bound document
    #[error("evaluator result is invalid")]
    Result,
}

/// Lease-declared reads for one validation unit
#[derive(Debug, Clone)]
pub struct DeclaredReads {
    /// Future published-snapshot location
    pub manifest: ArtifactAccessSpec,
    /// Future model location
    pub model: ArtifactAccessSpec,
    /// Frozen development-bundle location
    pub bundle: ArtifactAccessSpec,
    /// Frozen trainer-configuration location
    pub trainer_configuration: ArtifactAccessSpec,
}

impl ValidationWorker {
    /// Builds a worker over a cache directory and evaluator command
    pub fn new(
        cache_root: impl Into<PathBuf>,
        evaluator: EvaluatorCommand,
    ) -> Result<Self, WorkerError> {
        let cache = ArtifactCache::new(cache_root).map_err(|_| WorkerError::Cache)?;
        let client = reqwest::Client::builder()
            .https_only(true)
            .connect_timeout(Duration::from_secs(5))
            .build()
            .map_err(|_| WorkerError::Artifact)?;
        Ok(Self {
            cache,
            evaluator,
            client,
        })
    }

    fn execute_unit(
        &self,
        assignment: &WorkAssignment,
        context: &BlockingWorkContext,
    ) -> Result<WorkHandlerResult, WorkerError> {
        let payload = ValidationWorkPayload::from_value(assignment.payload.clone())
            .map_err(|_| WorkerError::Payload)?;
        let reads = declared_reads(&payload, assignment.artifact_access.specs())?;
        let trainer = self.admit_artifact(
            context,
            &reads.trainer_configuration,
            &payload.trainer_configuration.content_digest,
            payload.trainer_configuration.byte_length,
        )?;
        let bundle_object = self.admit_artifact(
            context,
            &reads.bundle,
            &payload.frozen_dev_bundle.content_digest,
            payload.frozen_dev_bundle.byte_length,
        )?;
        let snapshot = self.admit_future_snapshot(context, &reads.manifest, &payload)?;
        let model = self.admit_artifact(
            context,
            &reads.model,
            &snapshot.model_digest,
            snapshot.model_length,
        )?;
        let bundle = ExtractedBundle::extract(&bundle_object, &payload.frozen_dev_bundle, || {
            context.is_cancelled()
        })?;
        let artifacts = ResolvedArtifacts {
            snapshot,
            model_path: model,
            bundle_root: bundle.root().to_owned(),
            bundle_manifest_path: bundle.manifest().to_owned(),
            trainer_configuration_path: trainer,
        };
        self.complete_unit(&payload, &artifacts, || context.is_cancelled())
    }

    /// Builds the evaluator request, runs the process, and binds the result identity
    pub fn complete_unit(
        &self,
        payload: &ValidationWorkPayload,
        artifacts: &ResolvedArtifacts,
        cancelled: impl Fn() -> bool,
    ) -> Result<WorkHandlerResult, WorkerError> {
        let request =
            build_evaluator_request(payload, artifacts).map_err(|_| WorkerError::Result)?;
        let request_file =
            StagedPath::new(self.cache.staging_path(&payload.slot_id, &unique_suffix()));
        if let Some(parent) = request_file.parent() {
            fs::create_dir_all(parent).map_err(|_| WorkerError::Process)?;
        }
        write_request_file(request_file.as_path(), &request).map_err(|_| WorkerError::Process)?;
        let output = run_evaluator(&self.evaluator, request_file.as_path(), &cancelled)
            .map_err(|_| WorkerError::Process)?;
        if cancelled() {
            return Err(WorkerError::Process);
        }
        let validated = validate_result(output.as_bytes(), payload, &artifacts.snapshot)
            .map_err(|_| WorkerError::Result)?;
        let retained = serde_json::to_value(&validated).map_err(|_| WorkerError::Result)?;
        WorkHandlerResult::retained_json(retained).map_err(|_| WorkerError::Result)
    }

    fn admit_artifact(
        &self,
        context: &BlockingWorkContext,
        spec: &ArtifactAccessSpec,
        digest: &Sha256Digest,
        byte_length: u64,
    ) -> Result<PathBuf, WorkerError> {
        if let Some(path) = self
            .cache
            .verified_path(digest, byte_length)
            .map_err(|_| WorkerError::Cache)?
        {
            return Ok(path);
        }
        let staged =
            self.download_declared(context, spec, digest, DownloadLength::Exact(byte_length))?;
        self.cache
            .admit(digest, byte_length, staged.as_path())
            .map_err(|_| WorkerError::Cache)
    }

    fn admit_future_snapshot(
        &self,
        context: &BlockingWorkContext,
        spec: &ArtifactAccessSpec,
        payload: &ValidationWorkPayload,
    ) -> Result<PublishedSnapshot, WorkerError> {
        let staged = self.download_declared(
            context,
            spec,
            &payload.slot_id,
            DownloadLength::AtMost(MANIFEST_MAX_BYTES),
        )?;
        let bytes = fs::read(staged.as_path()).map_err(|_| WorkerError::Artifact)?;
        let snapshot = PublishedSnapshot::from_bytes(&bytes).map_err(|_| WorkerError::Result)?;
        snapshot
            .bind_payload(payload)
            .map_err(|_| WorkerError::Result)?;
        let digest = sha256_of(&bytes);
        self.cache
            .admit(&digest, bytes.len() as u64, staged.as_path())
            .map_err(|_| WorkerError::Cache)?;
        Ok(snapshot)
    }

    fn download_declared(
        &self,
        context: &BlockingWorkContext,
        spec: &ArtifactAccessSpec,
        staging_identity: &Sha256Digest,
        length: DownloadLength,
    ) -> Result<StagedPath, WorkerError> {
        let mut authorization_refreshes = AuthorizationRefreshBudget::new();
        loop {
            if context.is_cancelled() {
                return Err(WorkerError::Process);
            }
            if access_expired(spec) {
                return Err(WorkerError::Artifact);
            }

            let url = context
                .resolve_artifact_access(spec.clone())
                .map_err(|_| WorkerError::Artifact)?;
            let staged =
                StagedPath::new(self.cache.staging_path(staging_identity, &unique_suffix()));
            match self.download(&url, staged.as_path(), length, || context.is_cancelled()) {
                Ok(()) => return Ok(staged),
                Err(DownloadOutcome::NotFound | DownloadOutcome::Retryable) => {
                    wait_for_retry(context, spec)?
                }
                Err(DownloadOutcome::AuthorizationRejected) => {
                    authorization_refreshes.consume()?;
                    wait_for_retry(context, spec)?;
                }
                Err(DownloadOutcome::Cancelled) => {
                    return Err(WorkerError::Process);
                }
                Err(DownloadOutcome::Failed) => {
                    return Err(WorkerError::Artifact);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DownloadOutcome {
    NotFound,
    AuthorizationRejected,
    Retryable,
    Cancelled,
    Failed,
}

#[derive(Debug)]
struct AuthorizationRefreshBudget(usize);

impl AuthorizationRefreshBudget {
    const fn new() -> Self {
        Self(MAX_AUTHORIZATION_REFRESHES)
    }

    fn consume(&mut self) -> Result<(), WorkerError> {
        if self.0 == 0 {
            return Err(WorkerError::Artifact);
        }
        self.0 -= 1;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum DownloadLength {
    Exact(u64),
    AtMost(u64),
}

struct StagedPath(PathBuf);

impl StagedPath {
    fn new(path: PathBuf) -> Self {
        Self(path)
    }

    fn as_path(&self) -> &Path {
        &self.0
    }

    fn parent(&self) -> Option<&Path> {
        self.0.parent()
    }
}

impl Drop for StagedPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

impl DownloadLength {
    const fn limit(self) -> u64 {
        match self {
            Self::Exact(length) | Self::AtMost(length) => length,
        }
    }

    const fn accepts(self, actual: u64) -> bool {
        match self {
            Self::Exact(expected) => actual == expected,
            Self::AtMost(maximum) => actual <= maximum,
        }
    }
}

impl ValidationWorker {
    fn download(
        &self,
        url: &ArtifactAccessUrl,
        dest: &Path,
        length: DownloadLength,
        cancelled: impl Fn() -> bool,
    ) -> Result<(), DownloadOutcome> {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|_| DownloadOutcome::Failed)?;
        }

        if cancelled() {
            return Err(DownloadOutcome::Cancelled);
        }

        let runtime = tokio::runtime::Handle::try_current().map_err(|_| DownloadOutcome::Failed)?;
        runtime.block_on(self.download_response(url, dest, length, &cancelled))
    }

    async fn download_response(
        &self,
        url: &ArtifactAccessUrl,
        dest: &Path,
        length: DownloadLength,
        cancelled: &impl Fn() -> bool,
    ) -> Result<(), DownloadOutcome> {
        let response = self.client.get(url.as_str()).send();
        let mut response = wait_for_download(response, None, cancelled)
            .await?
            .map_err(|error| request_error(&error, cancelled))?;
        classify_download_status(response.status())?;

        if response
            .content_length()
            .is_some_and(|actual| !length.accepts(actual))
        {
            return Err(DownloadOutcome::Failed);
        }

        let mut file = fs::File::create(dest).map_err(|_| DownloadOutcome::Failed)?;
        let mut writer = BoundedDownload::new(&mut file, length);
        loop {
            let chunk = wait_for_download(response.chunk(), Some(DOWNLOAD_READ_TIMEOUT), cancelled)
                .await?
                .map_err(|error| request_error(&error, cancelled))?;
            let Some(chunk) = chunk else {
                break;
            };
            writer.write_chunk(&chunk)?;
        }

        writer.finish()?;
        file.flush().map_err(|_| DownloadOutcome::Failed)?;
        Ok(())
    }
}

fn classify_download_status(status: reqwest::StatusCode) -> Result<(), DownloadOutcome> {
    if status.is_success() {
        return Ok(());
    }

    // object responses do not distinguish an expired grant from permanent object-store auth
    match status.as_u16() {
        401 | 403 => Err(DownloadOutcome::AuthorizationRejected),
        404 => Err(DownloadOutcome::NotFound),
        408 | 429 => Err(DownloadOutcome::Retryable),
        _ if status.is_server_error() => Err(DownloadOutcome::Retryable),
        _ => Err(DownloadOutcome::Failed),
    }
}

struct BoundedDownload<'a, W> {
    destination: &'a mut W,
    length: DownloadLength,
    transferred: u64,
}

impl<W: Write> BoundedDownload<'_, W> {
    fn new(destination: &mut W, length: DownloadLength) -> BoundedDownload<'_, W> {
        BoundedDownload {
            destination,
            length,
            transferred: 0,
        }
    }

    fn write_chunk(&mut self, chunk: &[u8]) -> Result<(), DownloadOutcome> {
        self.transferred = self
            .transferred
            .checked_add(chunk.len() as u64)
            .ok_or(DownloadOutcome::Failed)?;
        if self.transferred > self.length.limit() {
            return Err(DownloadOutcome::Failed);
        }
        self.destination
            .write_all(chunk)
            .map_err(|_| DownloadOutcome::Failed)?;
        Ok(())
    }

    fn finish(self) -> Result<(), DownloadOutcome> {
        if self.length.accepts(self.transferred) {
            Ok(())
        } else {
            Err(DownloadOutcome::Failed)
        }
    }
}

async fn wait_for_download<T, E>(
    future: impl Future<Output = Result<T, E>>,
    timeout: Option<Duration>,
    cancelled: &impl Fn() -> bool,
) -> Result<Result<T, E>, DownloadOutcome> {
    tokio::pin!(future);
    let started = std::time::Instant::now();
    loop {
        if cancelled() {
            return Err(DownloadOutcome::Cancelled);
        }

        let poll_after = timeout
            .map(|limit| limit.saturating_sub(started.elapsed()))
            .unwrap_or(Duration::from_millis(25))
            .min(Duration::from_millis(25));
        if timeout.is_some() && poll_after.is_zero() {
            return Err(DownloadOutcome::Retryable);
        }

        tokio::select! {
            result = &mut future => return Ok(result),
            () = tokio::time::sleep(poll_after) => {}
        }
    }
}

fn request_error(error: &reqwest::Error, cancelled: &impl Fn() -> bool) -> DownloadOutcome {
    if cancelled() {
        DownloadOutcome::Cancelled
    } else if error.is_timeout() || error.is_connect() {
        DownloadOutcome::Retryable
    } else {
        DownloadOutcome::Failed
    }
}

fn wait_for_retry(
    context: &BlockingWorkContext,
    spec: &ArtifactAccessSpec,
) -> Result<(), WorkerError> {
    let deadline = std::time::Instant::now() + DOWNLOAD_RETRY_DELAY;
    while std::time::Instant::now() < deadline {
        if context.is_cancelled() {
            return Err(WorkerError::Process);
        }
        if access_expired(spec) {
            return Err(WorkerError::Artifact);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    Ok(())
}

fn access_expired(spec: &ArtifactAccessSpec) -> bool {
    let Ok(elapsed) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return true;
    };
    i64::try_from(elapsed.as_secs()).map_or(true, |now| now >= spec.not_after_unix_seconds())
}

impl BlockingWorkloadHandler for ValidationWorker {
    fn capabilities(&self) -> WorkloadCapabilities {
        default_worker_capabilities()
    }

    fn self_check(&self) -> Result<(), WorkFailureReason> {
        check_evaluator_prerequisite(&self.evaluator).map_err(|_| {
            WorkFailureReason::new("evaluator prerequisite check failed")
                .expect("static failure reason")
        })
    }

    fn execute(
        &self,
        assignment: WorkAssignment,
        context: BlockingWorkContext,
    ) -> Result<WorkHandlerResult, WorkHandlerFailure> {
        self.execute_unit(&assignment, &context).map_err(|error| {
            let category = match error {
                WorkerError::Process => WorkFailureCategory::DeadlineExceeded,
                WorkerError::Artifact | WorkerError::Cache => {
                    WorkFailureCategory::WorkerUnavailable
                }
                _ => WorkFailureCategory::HandlerFailed,
            };
            WorkHandlerFailure::new(
                category,
                WorkFailureReason::new(error.to_string()).unwrap_or_else(|_| {
                    WorkFailureReason::new("validation worker failed").expect("static reason")
                }),
            )
        })
    }
}

/// Matches the assignment's exact declared reads to the unit payload locations
pub fn declared_reads(
    payload: &ValidationWorkPayload,
    specs: &[ArtifactAccessSpec],
) -> Result<DeclaredReads, WorkerError> {
    if specs.len() != 4 {
        return Err(WorkerError::Undeclared);
    }
    Ok(DeclaredReads {
        manifest: spec_for(specs, &payload.manifest_location)?.clone(),
        model: spec_for(specs, &payload.model_location)?.clone(),
        bundle: spec_for(specs, &payload.frozen_dev_bundle.location)?.clone(),
        trainer_configuration: spec_for(specs, &payload.trainer_configuration.location)?.clone(),
    })
}

fn spec_for<'a>(
    specs: &'a [ArtifactAccessSpec],
    location: &ArtifactLocation,
) -> Result<&'a ArtifactAccessSpec, WorkerError> {
    specs
        .iter()
        .find(|spec| spec.location() == location)
        .ok_or(WorkerError::Undeclared)
}

fn unique_suffix() -> String {
    uuid::Uuid::now_v7().to_string()
}

fn sha256_of(bytes: &[u8]) -> Sha256Digest {
    use sha2::{Digest, Sha256};

    Sha256Digest::from_bytes(Sha256::digest(bytes).into())
}

#[cfg(test)]
mod tests {
    use std::{
        fs, future,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    use tempfile::tempdir;

    use cloudeck_worker_runtime::BlockingWorkloadHandler;

    use super::{
        AuthorizationRefreshBudget, BoundedDownload, DownloadLength, DownloadOutcome, StagedPath,
        ValidationWorker, classify_download_status, wait_for_download,
    };
    use crate::EvaluatorCommand;

    #[test]
    fn handler_self_check_fails_for_a_missing_evaluator() {
        let root = tempdir().unwrap();
        let command = EvaluatorCommand::new("definitely-not-an-evaluator", vec![]).unwrap();
        let worker = ValidationWorker::new(root.path(), command).unwrap();

        assert!(worker.self_check().is_err());
    }

    #[test]
    fn object_authorization_rejection_requests_a_fresh_grant() {
        assert_eq!(
            classify_download_status(reqwest::StatusCode::UNAUTHORIZED),
            Err(DownloadOutcome::AuthorizationRejected)
        );
        assert_eq!(
            classify_download_status(reqwest::StatusCode::FORBIDDEN),
            Err(DownloadOutcome::AuthorizationRejected)
        );
        assert!(AuthorizationRefreshBudget::new().consume().is_ok());
    }

    #[test]
    fn persistent_object_authorization_rejection_exhausts_its_refresh_budget() {
        let mut budget = AuthorizationRefreshBudget::new();
        for _ in 0..super::MAX_AUTHORIZATION_REFRESHES {
            budget.consume().unwrap();
        }

        assert!(matches!(
            budget.consume(),
            Err(super::WorkerError::Artifact)
        ));
    }

    #[test]
    fn transient_and_permanent_download_statuses_are_distinct() {
        assert_eq!(
            classify_download_status(reqwest::StatusCode::REQUEST_TIMEOUT),
            Err(DownloadOutcome::Retryable)
        );
        assert_eq!(
            classify_download_status(reqwest::StatusCode::TOO_MANY_REQUESTS),
            Err(DownloadOutcome::Retryable)
        );
        assert_eq!(
            classify_download_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR),
            Err(DownloadOutcome::Retryable)
        );
        assert_eq!(
            classify_download_status(reqwest::StatusCode::BAD_REQUEST),
            Err(DownloadOutcome::Failed)
        );
    }

    #[test]
    fn bounded_download_rejects_a_chunk_past_the_declared_length() {
        let mut bytes = Vec::new();
        let mut download = BoundedDownload::new(&mut bytes, DownloadLength::Exact(4));
        download.write_chunk(b"abc").unwrap();
        assert_eq!(download.write_chunk(b"de"), Err(DownloadOutcome::Failed));
        assert_eq!(bytes, b"abc");
    }

    #[test]
    fn a_slow_progressing_read_is_not_treated_as_stalled() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = runtime.block_on(wait_for_download(
            async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok::<_, ()>(b"chunk")
            },
            Some(Duration::from_millis(250)),
            &|| false,
        ));
        assert_eq!(result, Ok(Ok(b"chunk")));
    }

    #[test]
    fn cancellation_interrupts_a_stalled_read_on_a_bounded_cadence() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let signal = Arc::clone(&cancelled);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            signal.store(true, Ordering::Release);
        });

        let started = Instant::now();
        let result = runtime.block_on(wait_for_download(
            future::pending::<Result<(), ()>>(),
            Some(Duration::from_secs(10)),
            &|| cancelled.load(Ordering::Acquire),
        ));
        assert_eq!(result, Err(DownloadOutcome::Cancelled));
        assert!(started.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn cancellation_discards_a_partial_staged_download() {
        let root = tempdir().unwrap();
        let path = root.path().join("artifact.part");
        let staged = StagedPath::new(path.clone());
        fs::write(staged.as_path(), b"partial").unwrap();

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = runtime.block_on(wait_for_download(
            future::pending::<Result<(), ()>>(),
            Some(Duration::from_secs(10)),
            &|| true,
        ));
        assert_eq!(result, Err(DownloadOutcome::Cancelled));
        drop(staged);
        assert!(!path.exists());
    }

    #[test]
    fn a_stalled_read_is_retryable_without_cancellation() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = runtime.block_on(wait_for_download(
            future::pending::<Result<(), ()>>(),
            Some(Duration::from_millis(50)),
            &|| false,
        ));
        assert_eq!(result, Err(DownloadOutcome::Retryable));
    }
}
