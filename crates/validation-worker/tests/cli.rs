//! Drives the shipped worker binary command selection

use std::process::Command;

const MISSPELLED_CREDENTIAL_ENV: &str = "CLOUDDECK_WORK_CREDENTIAL";
const DUMMY_CREDENTIAL: &str = "0123456789abcdef0123456789abcdef";

fn worker_bin() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_diarizen-validation-worker"));
    command.current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    command
}

fn without_queue_credentials(command: &mut Command) -> &mut Command {
    command
        .env_remove(cloudeck_client::WORK_CREDENTIAL_ENV)
        .env_remove(cloudeck_client::BOOTSTRAP_TOKEN_ENV)
        .env_remove(MISSPELLED_CREDENTIAL_ENV)
}

#[test]
fn self_check_subcommand_succeeds_without_a_credential() {
    let output = without_queue_credentials(worker_bin().arg("self-check"))
        .output()
        .expect("worker binary");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "ok");
    assert_eq!(
        report["queue_credential_present"],
        serde_json::Value::Bool(false)
    );
    assert_eq!(
        report["cloudeck_revision"],
        diarizen_validation_worker::CLOUDECK_REVISION
    );
}

#[test]
fn self_check_fails_when_the_default_interpreter_is_missing() {
    let output = without_queue_credentials(worker_bin().arg("self-check"))
        .env("PATH", "/definitely-not-a-real-bin-directory")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn missing_credential_does_not_select_self_check() {
    let output = without_queue_credentials(&mut worker_bin())
        .output()
        .expect("worker binary");
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("\"status\":\"ok\""),
        "claim without a credential must not print a self-check report: {stdout}"
    );
}

#[test]
fn misspelled_credential_env_does_not_select_self_check() {
    let output = without_queue_credentials(&mut worker_bin())
        .env(MISSPELLED_CREDENTIAL_ENV, DUMMY_CREDENTIAL)
        .output()
        .expect("worker binary");
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("\"status\":\"ok\""));
}

#[test]
fn self_check_detects_the_cloudeck_client_credential_env() {
    let present = without_queue_credentials(worker_bin().arg("self-check"))
        .env(cloudeck_client::WORK_CREDENTIAL_ENV, DUMMY_CREDENTIAL)
        .output()
        .expect("worker binary");
    assert!(present.status.success());
    let report: serde_json::Value = serde_json::from_slice(&present.stdout).unwrap();
    assert_eq!(
        report["queue_credential_present"],
        serde_json::Value::Bool(true)
    );

    let misspelled = without_queue_credentials(worker_bin().arg("self-check"))
        .env(MISSPELLED_CREDENTIAL_ENV, DUMMY_CREDENTIAL)
        .output()
        .expect("worker binary");
    assert!(misspelled.status.success());
    let report: serde_json::Value = serde_json::from_slice(&misspelled.stdout).unwrap();
    assert_eq!(
        report["queue_credential_present"],
        serde_json::Value::Bool(false)
    );
}
