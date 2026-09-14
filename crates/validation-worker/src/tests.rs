//! Owner tests for cache, process output, documents, and self-check

use std::{fs, path::PathBuf, process::Command};

use cloudeck_core::{
    ManagedPoolRequest, QueueRequest, Sha256Digest, WorkUnitSpec, WorkerProfile,
    WorkloadCapabilities, capability_digest,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tempfile::tempdir;

use crate::{
    ArtifactCache, CLOUDECK_REVISION, EvaluatorCommand, InvocationError, WorkerInvocation,
    default_worker_capabilities,
    test_support::{
        default_pool_request, default_queue_request, default_worker_profile,
        published_image_digest, validation_unit,
    },
    worker_invocation,
};

#[test]
fn cache_verifies_hits_and_deletes_corrupt_entries() {
    let root = tempdir().unwrap();
    let cache = ArtifactCache::new(root.path()).unwrap();
    let bytes = b"frozen-dev-bundle";
    let digest = Sha256Digest::from_bytes(Sha256::digest(bytes).into());
    let staged = cache.staging_path(&digest, "test");
    fs::write(&staged, bytes).unwrap();
    let admitted = cache.admit(&digest, bytes.len() as u64, &staged).unwrap();
    assert_eq!(
        cache.verified_path(&digest, bytes.len() as u64).unwrap(),
        Some(admitted.clone())
    );

    fs::write(&admitted, b"corrupt").unwrap();
    assert!(
        cache
            .verified_path(&digest, bytes.len() as u64)
            .unwrap()
            .is_none()
    );
    assert!(!admitted.exists());

    let invalid = cache.staging_path(&digest, "invalid");
    fs::write(&invalid, b"same-size-corrupt").unwrap();
    assert!(
        cache
            .admit(&digest, b"same-size-corrupt".len() as u64, &invalid)
            .is_err()
    );
    assert!(!invalid.exists());
}

#[test]
fn typed_result_enforces_exactly_one_json_object() {
    let payload = sample_payload();
    let snapshot = sample_snapshot(&payload, digest_label('f'), 12);
    let encoded = serde_json::to_vec(&matching_result(&snapshot)).unwrap();
    crate::validate_result(&encoded, &payload, &snapshot).unwrap();

    let mut extra = encoded;
    extra.extend_from_slice(br#"{"extra":true}"#);
    assert!(crate::validate_result(&extra, &payload, &snapshot).is_err());
    assert!(crate::validate_result(b"", &payload, &snapshot).is_err());
    assert!(crate::validate_result(b"[]", &payload, &snapshot).is_err());
}

#[test]
fn process_group_cancel_rejects_late_output() {
    let script = tempdir().unwrap();
    let path = script.path().join("hang.py");
    fs::write(
        &path,
        "import sys,time\ntime.sleep(30)\nprint('{\"late\":true}')\n",
    )
    .unwrap();
    let command =
        EvaluatorCommand::new("python3", vec![path.to_string_lossy().into_owned()]).unwrap();
    let request = script.path().join("request.json");
    fs::write(&request, "{}").unwrap();
    let started = std::time::Instant::now();
    let result = crate::run_evaluator(&command, &request, || started.elapsed().as_millis() > 50);
    assert!(matches!(
        result,
        Err(crate::ProcessError::Cancelled | crate::ProcessError::LateResult)
    ));
    assert!(started.elapsed().as_secs() < 10);
}

#[cfg(unix)]
#[test]
fn stdout_overflow_cleans_the_entire_process_group_without_deadlock() {
    let root = tempdir().unwrap();
    let script = root.path().join("overflow.py");
    let pids = root.path().join("pids.txt");
    fs::write(
        &script,
        r#"import os, signal, subprocess, sys, time
pids = sys.argv[1]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
])
with open(pids, "w", encoding="utf-8") as handle:
    handle.write(f"{os.getpid()} {child.pid}")
    handle.flush()
    os.fsync(handle.fileno())
sys.stdout.write("x" * 70000)
sys.stdout.flush()
time.sleep(30)
"#,
    )
    .unwrap();
    let command = EvaluatorCommand::new(
        "python3",
        vec![
            script.to_string_lossy().into_owned(),
            pids.to_string_lossy().into_owned(),
        ],
    )
    .unwrap();
    let request = root.path().join("request.json");
    fs::write(&request, "{}").unwrap();

    let started = std::time::Instant::now();
    let result = crate::run_evaluator(&command, &request, || false);
    assert!(matches!(result, Err(crate::ProcessError::Output)));
    assert!(started.elapsed() < std::time::Duration::from_secs(5));

    let pids = fs::read_to_string(pids).unwrap();
    let pids: Vec<i32> = pids
        .split_whitespace()
        .map(|value| value.parse().unwrap())
        .collect();
    let wait_until = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while pids.iter().copied().any(process_exists) && std::time::Instant::now() < wait_until {
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    let survivors: Vec<_> = pids
        .iter()
        .copied()
        .filter(|pid| process_exists(*pid))
        .collect();
    for pid in &survivors {
        unsafe {
            libc::kill(*pid, libc::SIGKILL);
        }
    }
    assert!(
        survivors.is_empty(),
        "processes survived cleanup: {survivors:?}"
    );
}

#[cfg(unix)]
fn process_exists(pid: i32) -> bool {
    let result = unsafe { libc::kill(pid, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[test]
fn capabilities_and_profile_are_exact_cloudeck_documents() {
    let capabilities = default_worker_capabilities();
    let encoded = serde_json::to_value(&capabilities).unwrap();
    let parsed: WorkloadCapabilities = serde_json::from_value(encoded.clone()).unwrap();
    assert_eq!(parsed.canonical_bytes(), capabilities.canonical_bytes());
    assert_eq!(capability_digest(&parsed), capability_digest(&capabilities));
    assert_eq!(encoded["protocol_versions"], serde_json::json!([3]));
    assert_eq!(encoded["cancellation"], "cooperative");
    assert_eq!(encoded["resume"], "unit_boundary");

    let profile = default_worker_profile(published_image_digest());
    profile.validate().unwrap();
    let profile_json = serde_json::to_value(&profile).unwrap();
    let reparsed: WorkerProfile = serde_json::from_value(profile_json.clone()).unwrap();
    assert_eq!(reparsed, profile);
    assert_eq!(profile_json["provider"]["provider"], "vast_ai");
    assert_eq!(profile_json["provider"]["gpu_name"], "RTX_5060_Ti");
    assert_eq!(profile_json["network"], "egress");
    assert!(profile.allowed_secrets.is_empty());
    assert!(profile.registry_auth_secret.is_none());
    assert_eq!(profile.capacity_limit.get(), 1);
    assert_eq!(profile.command, vec!["claim".to_owned()]);
}

#[test]
fn default_queue_is_one_lease_and_one_capacity_one_pool() {
    let units: Vec<_> = (0..22)
        .map(|ordinal| {
            validation_unit(
                ordinal,
                serde_json::json!({"schema":"diarizen-cloudeck-validation-work-payload-v1","n":ordinal}),
                2_000_000_000,
            )
            .unwrap()
        })
        .collect();
    let request = default_queue_request(&units).unwrap();
    assert_eq!(request.max_active_leases.get(), 1);
    assert_eq!(request.protocol.version.as_u16(), 3);
    assert_eq!(units.len(), 22);
    let encoded = serde_json::to_value(&request).unwrap();
    let parsed: QueueRequest = serde_json::from_value(encoded).unwrap();
    assert_eq!(parsed.queue_id, request.queue_id);
    let text = serde_json::to_string(&request).unwrap();
    assert!(!text.contains("https://"));
    assert!(!text.contains("token"));
    assert!(!text.contains("password"));

    let profile = default_worker_profile(published_image_digest());
    let pool = default_pool_request(request.queue_id.clone(), &profile).unwrap();
    assert_eq!(pool.capacity.get(), 1);
    assert!(matches!(
        pool.reuse,
        cloudeck_core::ManagedReusePolicy::QueueLifetime
    ));
    let pool_json = serde_json::to_value(&pool).unwrap();
    let _: ManagedPoolRequest = serde_json::from_value(pool_json).unwrap();
}

#[test]
fn self_check_names_command_schemas_revision_and_missing_credentials() {
    let root = tempdir().unwrap();
    let script = root.path().join("evaluator.py");
    fs::write(
        &script,
        "import sys\nsys.exit(0 if '--help' in sys.argv else 3)\n",
    )
    .unwrap();
    let command =
        EvaluatorCommand::new("python3", vec![script.to_string_lossy().into_owned()]).unwrap();
    let report = crate::self_check::run_self_check_for(&command).unwrap();
    assert_eq!(report.status, "ok");
    assert_eq!(report.evaluator_command, command.argv());
    assert_eq!(report.cloudeck_revision, CLOUDECK_REVISION);
    assert!(!report.runtime_credentials_required);
    assert_eq!(
        report.worker_schemas.workload_family.as_str(),
        "diarizen-validation"
    );
}

#[test]
fn worker_invocation_is_an_explicit_subcommand() {
    assert_eq!(
        worker_invocation(["diarizen-validation-worker"]),
        Ok(WorkerInvocation::Claim)
    );
    assert_eq!(
        worker_invocation(["diarizen-validation-worker", "claim"]),
        Ok(WorkerInvocation::Claim)
    );
    assert_eq!(
        worker_invocation(["diarizen-validation-worker", "self-check"]),
        Ok(WorkerInvocation::SelfCheck)
    );
    assert_eq!(
        worker_invocation(["diarizen-validation-worker", "self-check", "extra"]),
        Err(InvocationError::UnexpectedArgument)
    );
    assert_eq!(
        worker_invocation([
            "diarizen-validation-worker",
            "/usr/local/bin/diarizen-validation-worker"
        ]),
        Err(InvocationError::UnknownCommand(
            "/usr/local/bin/diarizen-validation-worker".to_owned()
        ))
    );
}

#[test]
fn self_check_uses_the_cloudeck_client_credential_name() {
    assert_eq!(
        cloudeck_client::WORK_CREDENTIAL_ENV,
        "CLOUDECK_WORK_CREDENTIAL"
    );
    assert_ne!(
        cloudeck_client::WORK_CREDENTIAL_ENV,
        "CLOUDDECK_WORK_CREDENTIAL"
    );
}

#[test]
fn python_produced_cloudeck_documents_parse_with_exact_types() {
    let documents = python_cloudeck_documents();
    let capabilities = documents.capabilities;
    assert_eq!(
        capabilities.canonical_bytes(),
        default_worker_capabilities().canonical_bytes()
    );
    assert_eq!(
        capability_digest(&capabilities),
        capability_digest(&default_worker_capabilities())
    );

    let profile = documents.profile;
    let expected = default_worker_profile(published_image_digest());
    assert_eq!(profile, expected);
    assert_eq!(profile.validate().unwrap(), expected.validate().unwrap());

    let request = documents.queue_request;
    request.validate().unwrap();
    assert_eq!(request.max_active_leases.get(), 1);
    assert_eq!(request.protocol.version.as_u16(), 3);
    assert_eq!(
        request
            .required_capability
            .as_ref()
            .unwrap()
            .capability_digest,
        capability_digest(&capabilities)
    );

    let units = documents.units;
    assert_eq!(units.len(), 22);
    assert_eq!(units[0].ordinal, 0);
    assert_eq!(units[0].artifact_access_specs.len(), 4);
    request.validate_units(&units).unwrap();

    let pool = documents.pool;
    pool.validate_for(&profile).unwrap();
    assert_eq!(pool.capacity.get(), 1);
    assert!(matches!(
        pool.reuse,
        cloudeck_core::ManagedReusePolicy::QueueLifetime
    ));
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PythonWireDocuments {
    capabilities: WorkloadCapabilities,
    profile: WorkerProfile,
    queue_request: QueueRequest,
    units: Vec<WorkUnitSpec>,
    pool: ManagedPoolRequest,
}

fn python_cloudeck_documents() -> PythonWireDocuments {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let python = root.join(".venv-large/bin/python");
    let output = Command::new(python)
        .current_dir(&root)
        .args([
            "-c",
            "from recipes.speakrs.large.validation.queue_plan import emit_cloudeck_wire_documents; import json; print(json.dumps(emit_cloudeck_wire_documents(), separators=(',', ':')))",
        ])
        .output()
        .expect("python must emit CloudDeck documents");
    assert!(
        output.status.success(),
        "python emit failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("python emitted JSON")
}

#[test]
fn cloudeck_lock_sources_use_the_pushed_revision() {
    let manifest =
        fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml")).unwrap();
    let count = manifest.matches(CLOUDECK_REVISION).count();
    assert!(count >= 3, "workspace must pin all three CloudDeck crates");
    assert!(!manifest.contains("branch ="));
    assert!(!manifest.contains("tag ="));
    assert!(!manifest.contains("path = \"/Users"));
}

fn digest_label(label: char) -> Sha256Digest {
    label.to_string().repeat(64).parse().unwrap()
}

fn location(value: &str) -> cloudeck_core::ArtifactLocation {
    cloudeck_core::ArtifactLocation::new(value).unwrap()
}

fn read_spec(value: &str) -> cloudeck_core::ArtifactAccessSpec {
    cloudeck_core::ArtifactAccessSpec::new(
        location(value),
        cloudeck_core::ArtifactAccessOperation::Read,
        2_000_000_000,
    )
    .unwrap()
}

fn artifact_ref(digest: Sha256Digest, loc: &str) -> crate::payload::FrozenBundleRef {
    crate::payload::FrozenBundleRef {
        content_digest: digest,
        byte_length: 8,
        media_type: "application/octet-stream".to_owned(),
        location: location(loc),
        compression: None,
    }
}

fn sample_payload() -> crate::ValidationWorkPayload {
    let campaign = digest_label('a');
    let slot = digest_label('b');
    let trainer = digest_label('c');
    let bundle = digest_label('d');
    crate::ValidationWorkPayload {
        schema: crate::WORK_PAYLOAD_SCHEMA.to_owned(),
        campaign_id: campaign,
        slot_id: slot,
        training_launch_id: "launch-v1".to_owned(),
        updates: 0,
        trainer_configuration_digest: trainer.clone(),
        evaluator_image_identity: "registry.example/evaluator@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_owned(),
        evaluator_implementation_digest: digest_label('e'),
        dev_bundle_digest: bundle.clone(),
        slot_ordinal: 0,
        point: crate::SnapshotPoint::EpochZero {},
        manifest_location: location("r2://validation/slot-00/manifest.json"),
        model_location: location("r2://validation/slot-00/model.bin"),
        future_model: crate::payload::FutureModelArtifact {
            state: crate::FutureModelState::Uncommitted,
            location: location("r2://validation/slot-00/model.bin"),
        },
        frozen_dev_bundle: artifact_ref(bundle, "r2://validation/frozen-dev/bundle.tar"),
        trainer_configuration: artifact_ref(trainer, "r2://validation/frozen-dev/trainer-config.toml"),
    }
}

fn sample_snapshot(
    payload: &crate::ValidationWorkPayload,
    model_digest: Sha256Digest,
    model_length: u64,
) -> crate::PublishedSnapshot {
    let identity = serde_json::json!({
        "campaign_id": payload.campaign_id,
        "dev_bundle_digest": payload.dev_bundle_digest,
        "evaluator_image_identity": payload.evaluator_image_identity,
        "evaluator_implementation_digest": payload.evaluator_implementation_digest,
        "generation_sequence": 1,
        "model_digest": model_digest,
        "model_length": model_length,
        "progress_digest": digest_label('1'),
        "publication_time": "2026-01-01T00:00:00Z",
        "recovery_generation_id": digest_label('2'),
        "slot_id": payload.slot_id,
        "trainer_configuration_digest": payload.trainer_configuration_digest,
        "training_launch_id": payload.training_launch_id,
        "updates": payload.updates,
    });
    let snapshot_id =
        Sha256Digest::from_bytes(Sha256::digest(serde_json::to_vec(&identity).unwrap()).into());
    crate::PublishedSnapshot {
        schema: "diarizen-published-snapshot-v1".to_owned(),
        snapshot_id,
        campaign_id: payload.campaign_id.clone(),
        training_launch_id: payload.training_launch_id.clone(),
        slot_id: payload.slot_id.clone(),
        updates: payload.updates,
        model_digest,
        model_length,
        trainer_configuration_digest: payload.trainer_configuration_digest.clone(),
        dev_bundle_digest: payload.dev_bundle_digest.clone(),
        evaluator_image_identity: payload.evaluator_image_identity.clone(),
        evaluator_implementation_digest: payload.evaluator_implementation_digest.clone(),
        recovery_generation_id: digest_label('2'),
        generation_sequence: 1,
        progress_digest: digest_label('1'),
        publication_time: "2026-01-01T00:00:00Z".to_owned(),
    }
}

fn matching_result(snapshot: &crate::PublishedSnapshot) -> serde_json::Value {
    serde_json::json!({
        "schema": crate::VALIDATION_RESULT_SCHEMA,
        "snapshot_id": snapshot.snapshot_id,
        "campaign_id": snapshot.campaign_id,
        "training_launch_id": snapshot.training_launch_id,
        "slot_id": snapshot.slot_id,
        "updates": snapshot.updates,
        "model_digest": snapshot.model_digest,
        "model_length": snapshot.model_length,
        "trainer_configuration_digest": snapshot.trainer_configuration_digest,
        "dev_bundle_digest": snapshot.dev_bundle_digest,
        "evaluator_image_identity": snapshot.evaluator_image_identity,
        "evaluator_implementation_digest": snapshot.evaluator_implementation_digest,
        "recovery_generation_id": snapshot.recovery_generation_id,
        "generation_sequence": snapshot.generation_sequence,
        "progress_digest": snapshot.progress_digest,
        "publication_time": snapshot.publication_time,
        "loss": 0.1,
        "der": 0.2,
        "false_alarm": 0.0,
        "miss": 0.0,
        "confusion": 0.0,
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
    })
}

#[test]
fn published_snapshot_rejects_an_identity_not_bound_to_its_fields() {
    let payload = sample_payload();
    let snapshot = sample_snapshot(&payload, digest_label('f'), 12);
    let mut encoded = serde_json::to_value(snapshot).unwrap();
    encoded["updates"] = serde_json::json!(1);
    let bytes = serde_json::to_vec(&encoded).unwrap();
    assert!(crate::PublishedSnapshot::from_bytes(&bytes).is_err());
}

#[test]
fn python_snapshot_identity_matches_rust_for_unicode_and_canonical_time() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let python = root.join(".venv-large/bin/python");
    let output = Command::new(python)
        .current_dir(&root)
        .args([
            "-c",
            r#"from datetime import datetime, timezone
import json
from recipes.speakrs.large.validation.contracts import PublishedSnapshot, Sha256Digest
d = lambda c: Sha256Digest(c * 64)
snapshot = PublishedSnapshot(
    campaign_id=d('a'), training_launch_id='lăunch-雪', slot_id=d('b'), updates=17,
    model_digest=d('c'), model_length=123, trainer_configuration_digest=d('d'),
    dev_bundle_digest=d('e'), evaluator_image_identity='registry.example/évaluator@sha256:' + 'f' * 64,
    evaluator_implementation_digest=d('1'), recovery_generation_id=d('2'),
    generation_sequence=3, progress_digest=d('3'),
    publication_time=datetime(2026, 1, 2, 3, 4, 5, 123400, tzinfo=timezone.utc),
)
print(json.dumps(snapshot.to_dict(), ensure_ascii=False, separators=(',', ':')))"#,
        ])
        .output()
        .expect("python must emit a published snapshot");
    assert!(
        output.status.success(),
        "python snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed = crate::PublishedSnapshot::from_bytes(&output.stdout).unwrap();
    assert_eq!(parsed.training_launch_id, "lăunch-雪");
    assert_eq!(parsed.publication_time, "2026-01-02T03:04:05.123400Z");
}

#[test]
fn validation_result_rejects_every_mismatched_snapshot_field() {
    let payload = sample_payload();
    let snapshot = sample_snapshot(&payload, digest_label('f'), 12);
    let mutations = [
        ("snapshot_id", serde_json::json!(digest_label('8'))),
        ("campaign_id", serde_json::json!(digest_label('9'))),
        ("training_launch_id", serde_json::json!("other-launch")),
        ("slot_id", serde_json::json!(digest_label('8'))),
        ("updates", serde_json::json!(1)),
        ("model_digest", serde_json::json!(digest_label('7'))),
        ("model_length", serde_json::json!(13)),
        (
            "trainer_configuration_digest",
            serde_json::json!(digest_label('6')),
        ),
        ("dev_bundle_digest", serde_json::json!(digest_label('5'))),
        (
            "evaluator_image_identity",
            serde_json::json!("registry.example/other@sha256:abcd"),
        ),
        (
            "evaluator_implementation_digest",
            serde_json::json!(digest_label('4')),
        ),
        (
            "recovery_generation_id",
            serde_json::json!(digest_label('3')),
        ),
        ("generation_sequence", serde_json::json!(2)),
        ("progress_digest", serde_json::json!(digest_label('0'))),
        (
            "publication_time",
            serde_json::json!("2026-01-01T00:00:01Z"),
        ),
    ];

    for (field, value) in mutations {
        let mut result = matching_result(&snapshot);
        result[field] = value;
        assert!(
            crate::validate_result(&serde_json::to_vec(&result).unwrap(), &payload, &snapshot)
                .is_err(),
            "accepted mismatched {field}"
        );
    }
}

#[test]
fn validation_result_rejects_invalid_metrics_and_times() {
    let payload = sample_payload();
    let snapshot = sample_snapshot(&payload, digest_label('f'), 12);
    for (field, value) in [
        ("loss", serde_json::json!(-0.1)),
        ("der", serde_json::json!(-1)),
        ("false_alarm", serde_json::json!(-0.1)),
        ("miss", serde_json::json!(-0.1)),
        ("confusion", serde_json::json!(-0.1)),
        ("publication_time", serde_json::json!("not-a-time")),
        ("started_at", serde_json::json!("not-a-time")),
        ("started_at", serde_json::json!("2026-01-01T00:00:00+01:00")),
        ("completed_at", serde_json::json!("2025-12-31T23:59:59Z")),
    ] {
        let mut result = matching_result(&snapshot);
        result[field] = value;
        assert!(
            crate::validate_result(&serde_json::to_vec(&result).unwrap(), &payload, &snapshot)
                .is_err(),
            "accepted invalid {field}"
        );
    }
}

#[test]
fn declared_reads_reject_undeclared_or_incomplete_specs() {
    let payload = sample_payload();
    let specs = vec![
        read_spec("r2://validation/slot-00/manifest.json"),
        read_spec("r2://validation/slot-00/model.bin"),
        read_spec("r2://validation/frozen-dev/bundle.tar"),
        read_spec("r2://validation/frozen-dev/trainer-config.toml"),
    ];
    crate::declared_reads(&payload, &specs).unwrap();

    let missing_trainer = &specs[..3];
    assert!(matches!(
        crate::declared_reads(&payload, missing_trainer),
        Err(crate::WorkerError::Undeclared)
    ));

    let mut wrong_model = specs.clone();
    wrong_model[1] = read_spec("r2://validation/slot-00/other.bin");
    assert!(matches!(
        crate::declared_reads(&payload, &wrong_model),
        Err(crate::WorkerError::Undeclared)
    ));
}

#[test]
fn evaluator_request_is_accepted_by_python_from_dict() {
    let payload = sample_payload();
    let model_bytes = b"model-bytes";
    let model_digest = Sha256Digest::from_bytes(Sha256::digest(model_bytes).into());
    let snapshot = sample_snapshot(&payload, model_digest, model_bytes.len() as u64);
    let root = tempdir().unwrap();
    let model_path = root.path().join("model.bin");
    fs::write(&model_path, model_bytes).unwrap();
    let bundle_root = root.path().join("dev");
    fs::create_dir_all(&bundle_root).unwrap();
    let bundle_manifest = bundle_root.join("bundle.json");
    fs::write(&bundle_manifest, b"{}").unwrap();
    let trainer_path = root.path().join("trainer.toml");
    fs::write(&trainer_path, b"[model]\n").unwrap();
    let artifacts = crate::ResolvedArtifacts {
        snapshot,
        model_path,
        bundle_root,
        bundle_manifest_path: bundle_manifest,
        trainer_configuration_path: trainer_path,
    };
    let request = crate::build_evaluator_request(&payload, &artifacts).unwrap();
    let encoded = serde_json::to_value(&request).unwrap();
    assert_eq!(
        encoded["schema"],
        "diarizen-standalone-evaluator-request-v1"
    );
    assert_eq!(
        encoded["snapshot"]["schema"],
        "diarizen-published-snapshot-v1"
    );
    assert!(encoded.get("model_path").is_none());
    assert!(encoded["model"].get("path").is_some());
    assert!(encoded["dev_bundle"].get("manifest_path").is_some());
    assert!(encoded["trainer_configuration"].get("path").is_some());

    let python = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv-large/bin/python");
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut child = Command::new(python)
        .current_dir(&repo)
        .args([
            "-c",
            "import json,sys; from recipes.speakrs.large.validation.evaluator import EvaluatorRequest; EvaluatorRequest.from_dict(json.load(sys.stdin)); print('ok')",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    {
        use std::io::Write;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&serde_json::to_vec(&encoded).unwrap())
            .unwrap();
    }
    let finished = child.wait_with_output().unwrap();
    assert!(
        finished.status.success(),
        "python rejected evaluator request: {}",
        String::from_utf8_lossy(&finished.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&finished.stdout).trim(), "ok");
}

#[test]
fn complete_unit_binds_result_identities_and_rejects_a_foreign_campaign() {
    let payload = sample_payload();
    let model_bytes = b"model-bytes";
    let model_digest = Sha256Digest::from_bytes(Sha256::digest(model_bytes).into());
    let snapshot = sample_snapshot(&payload, model_digest, model_bytes.len() as u64);
    let root = tempdir().unwrap();
    let model_path = root.path().join("model.bin");
    fs::write(&model_path, model_bytes).unwrap();
    let bundle_root = root.path().join("dev");
    fs::create_dir_all(&bundle_root).unwrap();
    let bundle_manifest = bundle_root.join("bundle.json");
    fs::write(&bundle_manifest, b"{}").unwrap();
    let trainer_path = root.path().join("trainer.toml");
    fs::write(&trainer_path, b"[model]\n").unwrap();
    let artifacts = crate::ResolvedArtifacts {
        snapshot: snapshot.clone(),
        model_path,
        bundle_root,
        bundle_manifest_path: bundle_manifest,
        trainer_configuration_path: trainer_path,
    };

    let matching = root.path().join("matching.py");
    fs::write(
        &matching,
        r#"
import json, sys
path = sys.argv[sys.argv.index("--request-file") + 1]
request = json.load(open(path))
snapshot = request["snapshot"]
result = {
    "schema": "diarizen-validation-result-v1",
    "snapshot_id": snapshot["snapshot_id"],
    "campaign_id": snapshot["campaign_id"],
    "training_launch_id": snapshot["training_launch_id"],
    "slot_id": snapshot["slot_id"],
    "updates": snapshot["updates"],
    "model_digest": snapshot["model_digest"],
    "model_length": snapshot["model_length"],
    "trainer_configuration_digest": snapshot["trainer_configuration_digest"],
    "dev_bundle_digest": snapshot["dev_bundle_digest"],
    "evaluator_image_identity": snapshot["evaluator_image_identity"],
    "evaluator_implementation_digest": snapshot["evaluator_implementation_digest"],
    "recovery_generation_id": snapshot["recovery_generation_id"],
    "generation_sequence": snapshot["generation_sequence"],
    "progress_digest": snapshot["progress_digest"],
    "publication_time": snapshot["publication_time"],
    "loss": 0.1,
    "der": 0.2,
    "false_alarm": 0.0,
    "miss": 0.0,
    "confusion": 0.0,
    "started_at": "2026-01-01T00:00:00Z",
    "completed_at": "2026-01-01T00:00:01Z",
}
print(json.dumps(result))
"#,
    )
    .unwrap();
    let worker = crate::ValidationWorker::new(
        root.path().join("cache"),
        crate::EvaluatorCommand::new("python3", vec![matching.to_string_lossy().into_owned()])
            .unwrap(),
    )
    .unwrap();
    worker
        .complete_unit(&payload, &artifacts, || false)
        .unwrap();

    let foreign = serde_json::json!({
        "schema": crate::VALIDATION_RESULT_SCHEMA,
        "snapshot_id": snapshot.snapshot_id,
        "campaign_id": digest_label('9'),
        "training_launch_id": snapshot.training_launch_id,
        "slot_id": snapshot.slot_id,
        "updates": snapshot.updates,
        "model_digest": snapshot.model_digest,
        "model_length": snapshot.model_length,
        "trainer_configuration_digest": snapshot.trainer_configuration_digest,
        "dev_bundle_digest": snapshot.dev_bundle_digest,
        "evaluator_image_identity": snapshot.evaluator_image_identity,
        "evaluator_implementation_digest": snapshot.evaluator_implementation_digest,
        "recovery_generation_id": snapshot.recovery_generation_id,
        "generation_sequence": snapshot.generation_sequence,
        "progress_digest": snapshot.progress_digest,
        "publication_time": snapshot.publication_time,
        "loss": 0.1,
        "der": 0.2,
        "false_alarm": 0.0,
        "miss": 0.0,
        "confusion": 0.0,
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
    });
    assert!(
        crate::validate_result(&serde_json::to_vec(&foreign).unwrap(), &payload, &snapshot,)
            .is_err()
    );
}

#[test]
fn layout_dev_bundle_extracts_zstd_tar_tree() {
    use std::io::Write;

    let tmp = tempdir().unwrap();
    let tree = tmp.path().join("tree");
    fs::create_dir_all(tree.join("audio/AMI")).unwrap();
    fs::create_dir_all(tree.join("annotations/AMI")).unwrap();
    fs::write(
        tree.join("bundle.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema": "speakrs-frozen-dev-bundle-v2",
            "recordings": [{"logical_id": "AMI:dev-000"}]
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(tree.join("audio/AMI/000-audio"), b"flac-bytes").unwrap();
    fs::write(tree.join("annotations/AMI/000-rttm"), b"rttm-bytes").unwrap();
    fs::write(tree.join("annotations/AMI/000-uem"), b"uem-bytes").unwrap();

    let tar_path = tmp.path().join("bundle.tar");
    {
        let file = fs::File::create(&tar_path).unwrap();
        let mut builder = tar::Builder::new(file);
        builder.append_dir_all(".", &tree).unwrap();
        builder.finish().unwrap();
    }

    let zst_path = tmp.path().join("bundle.tar.zst");
    {
        let tar_bytes = fs::read(&tar_path).unwrap();
        let mut encoder = zstd::Encoder::new(fs::File::create(&zst_path).unwrap(), 0).unwrap();
        encoder.write_all(&tar_bytes).unwrap();
        encoder.finish().unwrap();
    }

    let admitted = fs::read(&zst_path).unwrap();
    let bundle = crate::FrozenBundleRef {
        content_digest: Sha256Digest::from_bytes(Sha256::digest(&admitted).into()),
        byte_length: admitted.len() as u64,
        media_type: "application/zstd".to_owned(),
        location: location("r2://validation/frozen-dev/bundle.tar.zst"),
        compression: Some(crate::ArtifactCompression::Zstd),
    };

    let extracted = crate::bundle::ExtractedBundle::extract(&zst_path, &bundle, || false).unwrap();
    let root = extracted.root();
    let manifest = extracted.manifest();
    assert_eq!(manifest, root.join("bundle.json"));
    let parsed: serde_json::Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    assert_eq!(parsed["schema"], "speakrs-frozen-dev-bundle-v2");
    assert_eq!(
        fs::read(root.join("audio/AMI/000-audio")).unwrap(),
        b"flac-bytes"
    );
    assert_eq!(
        fs::read(root.join("annotations/AMI/000-rttm")).unwrap(),
        b"rttm-bytes"
    );
    assert_eq!(
        fs::read(root.join("annotations/AMI/000-uem")).unwrap(),
        b"uem-bytes"
    );
    assert_ne!(
        fs::read(&manifest).unwrap(),
        admitted,
        "layout must unpack the archive, not copy tar.zst as bundle.json"
    );
}
