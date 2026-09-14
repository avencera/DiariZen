//! CloudDeck v3 DiariZen validation worker

use std::{env, path::PathBuf, process::ExitCode};

use cloudeck_client::{ExternalWorker, WorkClient};
use cloudeck_worker_runtime::{BlockingAdapter, WorkRuntime, WorkRuntimeConfig};
use diarizen_validation_worker::{
    EvaluatorCommand, ValidationWorker, WorkerInvocation, run_self_check, worker_invocation,
};

fn main() -> ExitCode {
    match worker_invocation(env::args()) {
        Ok(WorkerInvocation::SelfCheck) => print_self_check(),
        Ok(WorkerInvocation::Claim) => match run_claim() {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => fail(error),
        },
        Err(error) => fail(error),
    }
}

fn print_self_check() -> ExitCode {
    let report = match run_self_check() {
        Ok(report) => report,
        Err(error) => return fail(error),
    };

    match serde_json::to_string(&report) {
        Ok(encoded) => {
            println!("{encoded}");
            ExitCode::SUCCESS
        }
        Err(_) => ExitCode::FAILURE,
    }
}

fn fail(error: impl std::fmt::Display) -> ExitCode {
    eprintln!("validation worker failed: {error}");
    ExitCode::FAILURE
}

fn run_claim() -> Result<(), Box<dyn std::error::Error>> {
    let worker = ExternalWorker::from_env()?;
    let cache = env::var("DIARIZEN_ARTIFACT_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/cache/diarizen-validation"));
    let handler = ValidationWorker::new(cache, EvaluatorCommand::default_python_module())?;
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        WorkRuntime::new(
            WorkClient::new()?,
            worker.endpoint,
            worker.worker_id,
            WorkRuntimeConfig::default(),
        )
        .run_workload(BlockingAdapter::new(handler))
        .await?;
        Ok::<(), Box<dyn std::error::Error>>(())
    })
}
