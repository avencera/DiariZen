"""Focused supervisor and backup-retention contract tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from diarizen.trainer_utils import canonical_trainer_progress_digest, seal_checkpoint_directory
from recipes.speakrs.large.jsonio import write_json
from recipes.speakrs.large.recovery import BackupReceipt, RecoveryGeneration, file_inventory
from recipes.speakrs.large.remote_backup import (
    BackupGenerationPin,
    BackupRetentionInput,
    _prune_trusted_generations,
)
from recipes.speakrs.large.training_supervisor import (
    REQUIRED_RECOVERY_FILES,
    SUPERVISOR_COMPLETED_EARLY_STOP,
    SUPERVISOR_COMPLETED_MAX_UPDATES,
    SUPERVISOR_EXITED_INCOMPLETE,
    SUPERVISOR_FAILED_VALIDATION,
    supervise_training,
)
from recipes.speakrs.large.validation.config import (
    CampaignManifestReference,
    ExternalValidationConfig,
    ImmutableObjectStoreDestination,
    PollBackoffBounds,
)
from recipes.speakrs.large.validation.contracts import (
    AcceptedEarlyStop,
    CompletedTrainingRun,
    FailedTrainingRun,
    MaxUpdatesReached,
    PublishedSnapshot,
    Sha256Digest,
    StoppingPolicy,
    TrainingFailure,
    TrainingFailureKind,
    ValidationResult,
)
from recipes.speakrs.large.validation.controller import RetentionState


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _external_config() -> ExternalValidationConfig:
    return ExternalValidationConfig(
        campaign_manifest=CampaignManifestReference(Path("campaign.json"), _digest("a")),
        trusted_selection_state_path=Path("selection-state.json"),
        publication_transaction_root=Path("publication"),
        object_store_destination=ImmutableObjectStoreDestination("r2", "models", "campaign"),
        poll_backoff=PollBackoffBounds(1, 30),
        stopping_policy=StoppingPolicy.EXTERNAL_PATIENCE_OR_MAX_UPDATES,
        patience_limit=3,
    )


def _snapshot(sequence: int = 2) -> PublishedSnapshot:
    return PublishedSnapshot(
        campaign_id=_digest("a"),
        training_launch_id="launch-validation-test",
        slot_id=_digest("b"),
        updates=20,
        model_digest=_digest("c"),
        model_length=123,
        trainer_configuration_digest=_digest("d"),
        dev_bundle_digest=_digest("e"),
        evaluator_image_identity="registry.example/evaluator@sha256:" + "f" * 64,
        evaluator_implementation_digest=_digest("1"),
        recovery_generation_id=_digest("2"),
        generation_sequence=sequence,
        progress_digest=_digest("3"),
        publication_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _result(snapshot: PublishedSnapshot) -> ValidationResult:
    return ValidationResult(
        snapshot_id=snapshot.snapshot_id,
        campaign_id=snapshot.campaign_id,
        training_launch_id=snapshot.training_launch_id,
        slot_id=snapshot.slot_id,
        updates=snapshot.updates,
        model_digest=snapshot.model_digest,
        model_length=snapshot.model_length,
        trainer_configuration_digest=snapshot.trainer_configuration_digest,
        dev_bundle_digest=snapshot.dev_bundle_digest,
        evaluator_image_identity=snapshot.evaluator_image_identity,
        evaluator_implementation_digest=snapshot.evaluator_implementation_digest,
        recovery_generation_id=snapshot.recovery_generation_id,
        generation_sequence=snapshot.generation_sequence,
        progress_digest=snapshot.progress_digest,
        publication_time=snapshot.publication_time,
        loss=0.25,
        der=0.5,
        false_alarm=0.1,
        miss=0.2,
        confusion=0.2,
        started_at=snapshot.publication_time,
        completed_at=snapshot.publication_time,
    )


def _launch(tmp_path: Path) -> dict[str, object]:
    checkpoint_root = tmp_path / "checkpoints"
    experiment_root = tmp_path / "experiment"
    training_root = tmp_path / "training"
    checkpoint_root.mkdir()
    experiment_root.mkdir()
    training_root.mkdir()
    return {
        "launch_id": "launch-validation-test",
        "attempt_id": "attempt-validation-test",
        "hard_deadline": "2099-01-01T00:00:00+00:00",
        "max_updates": 100,
        "validation_config": _external_config().to_dict(),
        "offer": {
            "instance_id": "50520000",
            "destroy_at": "2099-01-01T00:00:00+00:00",
        },
        "worker_paths": {
            "training_root": str(training_root),
            "trainer_config": str(training_root / "trainer.toml"),
            "checkpoint_root": str(checkpoint_root),
            "experiment_root": str(experiment_root),
        },
    }


def _guard_proof(launch: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "speakrs-vast-rental-guard-status-v1",
        "attempt_id": launch["attempt_id"],
        "launch_id": launch["launch_id"],
        "instance_id": launch["offer"]["instance_id"],
        "state": "armed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "deletion_outcome": None,
        "attempts": 0,
        "error_code": None,
        "overdue": False,
        "retrying": False,
    }


def _checkpoint(root: Path, launch_id: str, state: object) -> None:
    checkpoint = root / "update_00000020_generation_00000002"
    checkpoint.mkdir()
    progress = {
        "schema": "diarizen-checkpoint-progress-v1",
        "updates_trained": 20,
        "launch_id": launch_id,
        "training_run_state": state.to_dict(),
    }
    (checkpoint / "progress.json").write_text(json.dumps(progress, sort_keys=True), encoding="utf-8")
    for filename in REQUIRED_RECOVERY_FILES:
        payload = checkpoint / filename
        if not payload.exists():
            payload.write_bytes(filename.encode())
    seal_checkpoint_directory(checkpoint)


class _CompletedProcess:
    pid = 123456

    def __init__(self, return_code: int) -> None:
        self.return_code = return_code

    def poll(self) -> int:
        return self.return_code

    def wait(self) -> int:
        return self.return_code


def _run_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch: dict[str, object],
    progress: dict[str, object] | None,
    return_code: int,
    *,
    resume: bool = False,
) -> dict[str, object]:
    proof_path = tmp_path / "guard-proof.json"
    proof_path.write_text(json.dumps(_guard_proof(launch)), encoding="utf-8")
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor.verify_worker_launch",
        lambda _path: launch,
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._verify_external_guard_proof",
        lambda _launch, _path: _guard_proof(launch),
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._arm_rental_watchdog",
        lambda *_args, **_kwargs: {"pid": 1},
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._worker_environment",
        lambda _launch: {},
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._process_group_exists",
        lambda _process: False,
    )
    if progress is not None:
        monkeypatch.setattr(
            "recipes.speakrs.large.training_supervisor._latest_progress",
            lambda *_args: progress,
        )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor.subprocess.Popen",
        lambda *_args, **_kwargs: _CompletedProcess(return_code),
    )
    return supervise_training(
        tmp_path / "launch.json",
        tmp_path / "status.json",
        resume=resume,
        external_guard_proof_path=proof_path,
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (CompletedTrainingRun(MaxUpdatesReached()), SUPERVISOR_COMPLETED_MAX_UPDATES),
        (
            CompletedTrainingRun(AcceptedEarlyStop(1, _result(_snapshot()), _snapshot())),
            SUPERVISOR_COMPLETED_EARLY_STOP,
        ),
        (
            FailedTrainingRun(TrainingFailure(TrainingFailureKind.SELECTION_FAILURE, "selection failed")),
            SUPERVISOR_FAILED_VALIDATION,
        ),
    ],
)
def test_external_terminal_training_states_map_to_supervisor_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: object,
    expected: str,
) -> None:
    launch = _launch(tmp_path)
    progress = {
        "generation": "update_00000020_generation_00000002",
        "updates_trained": 20,
        "launch_id": launch["launch_id"],
        "training_run_state": state.to_dict(),
    }

    result = _run_supervisor(tmp_path, monkeypatch, launch, progress, 3 if expected == "failed_validation" else 0)

    assert result["state"] == expected
    assert result["slot_republished"] is False
    assert result["work_performed"] is True
    if expected == SUPERVISOR_FAILED_VALIDATION:
        assert result["return_code"] != 0


def test_external_zero_exit_without_valid_terminal_state_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = _launch(tmp_path)
    progress = {
        "generation": "update_00000020_generation_00000002",
        "updates_trained": 20,
        "launch_id": launch["launch_id"],
        "training_run_state": {"state": "completed", "completion": {"kind": "unknown"}},
    }

    result = _run_supervisor(tmp_path, monkeypatch, launch, progress, 0)

    assert result["state"] == SUPERVISOR_EXITED_INCOMPLETE
    assert result["training_state_valid"] is False
    assert result["return_code"] == 0


def test_terminal_resume_does_no_optimizer_work_or_slot_republish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = _launch(tmp_path)
    checkpoint_root = Path(launch["worker_paths"]["checkpoint_root"])
    state = CompletedTrainingRun(MaxUpdatesReached())
    _checkpoint(checkpoint_root, launch["launch_id"], state)
    proof_path = tmp_path / "guard-proof.json"
    proof_path.write_text(json.dumps(_guard_proof(launch)), encoding="utf-8")
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor.verify_worker_launch",
        lambda _path: launch,
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._verify_external_guard_proof",
        lambda _launch, _path: _guard_proof(launch),
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("terminal resume must not launch a trainer"),
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._arm_rental_watchdog",
        lambda *_args, **_kwargs: pytest.fail("terminal resume must not arm a watchdog"),
    )

    result = supervise_training(
        tmp_path / "launch.json",
        tmp_path / "status.json",
        resume=True,
        external_guard_proof_path=proof_path,
    )

    assert result["state"] == SUPERVISOR_COMPLETED_MAX_UPDATES
    assert result["work_performed"] is False
    assert result["slot_republished"] is False
    assert result["optimizer_updates"] == 0


def test_backup_retention_pins_published_generation_until_result_and_commit() -> None:
    snapshot = _snapshot(sequence=1)
    result = _result(snapshot)
    pending = BackupRetentionInput((snapshot,), (), RetentionState.PENDING)
    accepted_pending = BackupRetentionInput((snapshot,), (result,), RetentionState.PENDING)
    committed = BackupRetentionInput((snapshot,), (result,), RetentionState.COMMITTED)
    expected = BackupGenerationPin(RecoveryGeneration(snapshot.updates, snapshot.generation_sequence))

    assert pending.pins == (expected,)
    assert accepted_pending.pins == (expected,)
    assert committed.pins == ()


def _trusted_generation(root: Path, updates: int, sequence: int) -> Path:
    path = root / f"update_{updates:08d}_generation_{sequence:08d}"
    path.mkdir()
    (path / "progress.json").write_text(json.dumps({"updates_trained": updates}), encoding="utf-8")
    seal_checkpoint_directory(path, require_trainer_progress=True)
    receipt = BackupReceipt(
        generation_id=path.name,
        files=file_inventory(path),
        source="worker:/checkpoint",
        destination=str(path),
        trainer_progress_sha256=canonical_trainer_progress_digest(path / "progress.json"),
    )
    write_json(path.with_name(path.name + ".receipt.json"), receipt.identity())
    return path


def test_backup_pruning_orders_same_update_sequences_and_honors_typed_pin(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    first = _trusted_generation(trusted, 20, 1)
    _trusted_generation(trusted, 20, 2)
    third = _trusted_generation(trusted, 20, 3)
    snapshot = _snapshot(sequence=1)
    result = _result(snapshot)

    pending = BackupRetentionInput((snapshot,), (result,), RetentionState.PENDING)
    _prune_trusted_generations(trusted, pending.pins)

    assert first.exists()
    assert third.exists()
    committed = BackupRetentionInput((snapshot,), (result,), RetentionState.COMMITTED)
    _prune_trusted_generations(trusted, committed.pins)
    assert not first.exists()
