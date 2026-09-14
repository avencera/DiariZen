//! Credential-free worker self-check

use std::env;

use cloudeck_client::{BOOTSTRAP_TOKEN_ENV, WORK_CREDENTIAL_ENV};
use cloudeck_core::WorkloadCapabilities;
use serde::Serialize;

use crate::{
    documents::{CLOUDECK_REVISION, default_worker_capabilities},
    process::{EvaluatorCommand, ProcessError, check_evaluator_prerequisite},
};

/// Self-check report written to stdout
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelfCheckReport {
    /// Always `ok` when the local worker image is complete
    pub status: &'static str,
    /// Evaluator command the worker will spawn
    pub evaluator_command: Vec<String>,
    /// Canonical capability document
    pub worker_schemas: WorkloadCapabilities,
    /// Exact CloudDeck git revision
    pub cloudeck_revision: &'static str,
    /// Whether a queue credential is required to self-check
    pub runtime_credentials_required: bool,
    /// Whether a queue credential is present in this process
    pub queue_credential_present: bool,
}

/// Runs the credential-free self-check
pub fn run_self_check() -> Result<SelfCheckReport, ProcessError> {
    run_self_check_for(&EvaluatorCommand::default_python_module())
}

pub(crate) fn run_self_check_for(
    command: &EvaluatorCommand,
) -> Result<SelfCheckReport, ProcessError> {
    check_evaluator_prerequisite(command)?;

    Ok(SelfCheckReport {
        status: "ok",
        evaluator_command: command.argv(),
        worker_schemas: default_worker_capabilities(),
        cloudeck_revision: CLOUDECK_REVISION,
        runtime_credentials_required: false,
        queue_credential_present: env::var(WORK_CREDENTIAL_ENV).is_ok()
            || env::var(BOOTSTRAP_TOKEN_ENV).is_ok(),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::run_self_check_for;
    use crate::{EvaluatorCommand, ProcessError};

    #[test]
    fn self_check_rejects_a_missing_interpreter() {
        let command =
            EvaluatorCommand::new("definitely-not-an-installed-interpreter", vec![]).unwrap();

        assert!(matches!(
            run_self_check_for(&command),
            Err(ProcessError::Spawn)
        ));
    }

    #[test]
    fn self_check_rejects_a_failing_evaluator_command() {
        let root = tempdir().unwrap();
        let script = root.path().join("failing.py");
        fs::write(&script, "import sys\nsys.exit(3)\n").unwrap();
        let command =
            EvaluatorCommand::new("python3", vec![script.to_string_lossy().into_owned()]).unwrap();

        assert!(matches!(
            run_self_check_for(&command),
            Err(ProcessError::Output)
        ));
    }

    #[test]
    fn self_check_accepts_a_help_capable_evaluator_command() {
        let root = tempdir().unwrap();
        let script = root.path().join("working.py");
        fs::write(
            &script,
            "import sys\nsys.exit(0 if '--help' in sys.argv else 3)\n",
        )
        .unwrap();
        let command =
            EvaluatorCommand::new("python3", vec![script.to_string_lossy().into_owned()]).unwrap();

        let report = run_self_check_for(&command).unwrap();
        assert_eq!(report.evaluator_command, command.argv());
    }
}
