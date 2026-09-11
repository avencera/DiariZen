"""Focused tests for the trusted remote checkpoint controller."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import recipes.speakrs.large.remote_backup as remote_backup
from diarizen.trainer_utils import seal_checkpoint_directory
from recipes.speakrs.large.errors import RuntimeGateError
from recipes.speakrs.large.recovery import BackupReceipt, file_inventory


def _checkpoint(root: Path, updates: int, launch_id: str) -> Path:
    checkpoint = root / f"update_{updates:08d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "pytorch_model.bin").write_bytes(f"model-{updates}".encode())
    (checkpoint / "progress.json").write_text(
        json.dumps({"updates_trained": updates, "launch_id": launch_id}),
        encoding="utf-8",
    )
    seal_checkpoint_directory(checkpoint)
    return checkpoint


def _launch(
    worker_root: Path,
    trusted_root: Path,
    deadline: str = "2030-01-01T00:00:00+00:00",
    destroy_at: str | None = None,
) -> dict[str, object]:
    return {
        "launch_id": "current-launch",
        "offer": {
            "ssh_host": "worker.example",
            "ssh_port": 22,
            "ssh_user": "runner",
            "rented_at": "1970-01-01T00:00:00+00:00",
            "destroy_at": destroy_at or deadline,
            "rates": {"gpu_usd_per_hour": 1.0, "disk_usd_per_hour": 0.0},
        },
        "worker_paths": {
            "checkpoint_root": str(worker_root),
            "trusted_backup_root": str(trusted_root),
        },
        "hard_deadline": deadline,
        "max_updates": 100,
        "spend_ceiling_usd": 100.0,
    }


def _patch_launch(monkeypatch, launch: dict[str, object]) -> None:
    def read_json(path: Path):
        if Path(path).name == "launch.json":
            return launch
        return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(remote_backup, "read_json", read_json)
    monkeypatch.setattr(remote_backup, "parse_launch_lock", lambda payload: payload)


def _patch_transport(monkeypatch, launch: dict[str, object], worker_checkpoint: Path, events: list[str]) -> None:
    def remote_checkpoint(_launch, mode, generation=None):
        events.append(mode)
        if mode == "list":
            return {
                "generation": worker_checkpoint.name,
                "progress": json.loads((worker_checkpoint / "progress.json").read_text()),
            }
        if mode == "pin":
            return {"generation": generation, "pinned": True}
        if mode == "hash":
            return {"generation": generation, "files": file_inventory(worker_checkpoint)}
        return {"generation": generation, "unpinned": True}

    def run(command, **_kwargs):
        assert command[0] == "rsync"
        destination = Path(command[-1].rstrip("/"))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(worker_checkpoint, destination, dirs_exist_ok=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote_backup, "_remote_checkpoint", remote_checkpoint)
    monkeypatch.setattr(remote_backup.subprocess, "run", run)


def test_remote_script_filters_complete_generations_by_launch_id(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 10, "old-launch")
    current = _checkpoint(tmp_path, 20, "current-launch")

    result = subprocess.run(
        [sys.executable, "-", str(tmp_path), "list", "-", "current-launch"],
        input=remote_backup.REMOTE_CHECKPOINT_SCRIPT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout)["generation"] == current.name


def test_ssh_list_uses_a_nonempty_generation_sentinel(tmp_path: Path, monkeypatch) -> None:
    launch = _launch(tmp_path / "worker", tmp_path / "trusted")

    def run(command, **kwargs):
        assert command[-2:] == ["-", "current-launch"]
        assert kwargs["input"] == remote_backup.REMOTE_CHECKPOINT_SCRIPT
        return SimpleNamespace(returncode=0, stdout='{"generation": null}')

    monkeypatch.setattr(remote_backup.subprocess, "run", run)

    assert remote_backup._remote_checkpoint(launch, "list")["generation"] is None


def test_remote_script_pins_and_unpins_under_worker_lock(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 20, "current-launch")

    pinned = subprocess.run(
        [sys.executable, "-", str(tmp_path), "pin", checkpoint.name, "current-launch"],
        input=remote_backup.REMOTE_CHECKPOINT_SCRIPT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(pinned.stdout)["pinned"] is True
    assert (tmp_path / ".backup-pin").read_text(encoding="utf-8").strip() == checkpoint.name

    unpinned = subprocess.run(
        [sys.executable, "-", str(tmp_path), "unpin", checkpoint.name, "current-launch"],
        input=remote_backup.REMOTE_CHECKPOINT_SCRIPT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(unpinned.stdout)["unpinned"] is True
    assert not (tmp_path / ".backup-pin").exists()


def test_backup_fsyncs_payload_before_destination_and_receipt_publish(tmp_path: Path, monkeypatch) -> None:
    worker = tmp_path / "worker"
    trusted = tmp_path / "trusted"
    worker.mkdir()
    checkpoint = _checkpoint(worker, 20, "current-launch")
    launch = _launch(worker, trusted)
    _patch_launch(monkeypatch, launch)
    events: list[str] = []
    _patch_transport(monkeypatch, launch, checkpoint, events)
    real_fsync_tree = remote_backup._fsync_tree
    destination = trusted / checkpoint.name
    observed: list[tuple[bool, bool]] = []

    def fsync_tree(path: Path) -> None:
        observed.append((path == destination.with_name(destination.name + ".partial"), destination.exists()))
        real_fsync_tree(path)

    monkeypatch.setattr(remote_backup, "_fsync_tree", fsync_tree)

    receipt = remote_backup.backup_remote_once(tmp_path / "launch.json")

    assert receipt is not None
    assert observed and observed[0] == (True, False)
    assert (destination.with_name(destination.name + ".receipt.json")).is_file()
    assert events[-1] == "unpin"


def test_crash_created_destination_without_receipt_is_repaired(tmp_path: Path, monkeypatch) -> None:
    worker = tmp_path / "worker"
    trusted = tmp_path / "trusted"
    worker.mkdir()
    checkpoint = _checkpoint(worker, 20, "current-launch")
    launch = _launch(worker, trusted)
    _patch_launch(monkeypatch, launch)
    events: list[str] = []
    _patch_transport(monkeypatch, launch, checkpoint, events)
    destination = trusted / checkpoint.name
    destination.parent.mkdir(parents=True)
    shutil.copytree(checkpoint, destination)
    (destination.with_name(destination.name + ".receipt.json")).write_text("[]", encoding="utf-8")
    partial = trusted / f"{checkpoint.name}.partial"
    partial.mkdir()
    (partial / "crash-marker").write_text("interrupted", encoding="utf-8")
    (destination.with_name(destination.name + ".receipt.json.partial")).write_text("partial", encoding="utf-8")

    receipt = remote_backup.backup_remote_once(tmp_path / "launch.json")

    assert receipt is not None
    assert destination.is_dir()
    assert not partial.exists()
    assert (destination.with_name(destination.name + ".receipt.json")).is_file()
    assert events.count("unpin") == 1


def test_monitor_retries_and_makes_final_copy_after_cutoff(tmp_path: Path, monkeypatch) -> None:
    deadline = "1970-01-01T00:00:10+00:00"
    launch = _launch(
        tmp_path / "worker",
        tmp_path / "trusted",
        deadline,
        "1970-01-01T00:15:10+00:00",
    )
    _patch_launch(monkeypatch, launch)
    attempts = []
    destination = tmp_path / "trusted" / "update_00000020"
    destination.mkdir(parents=True)
    (destination / "progress.json").write_text(
        json.dumps(
            {
                "updates_trained": 60_000,
                "training_complete": True,
                "launch_id": "current-launch",
            }
        ),
        encoding="utf-8",
    )
    receipt = BackupReceipt(
        "update_00000020",
        {"progress.json": {"sha256": "a" * 64, "size": 1}},
        "source",
        str(destination),
    )

    def backup(_path):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeGateError("transient SSH failure")
        return receipt

    monkeypatch.setattr(remote_backup, "backup_remote_once", backup)
    monkeypatch.setattr(remote_backup.time, "time", lambda: 20.0)
    monkeypatch.setattr(remote_backup.time, "sleep", lambda _seconds: None)

    result = remote_backup.monitor_remote_backups(tmp_path / "launch.json", 10)

    assert result["latest_generation"] == receipt.generation_id
    assert len(attempts) == 2
    status = json.loads(
        (Path(launch["worker_paths"]["trusted_backup_root"]) / remote_backup.BACKUP_STATUS_FILENAME).read_text()
    )
    assert status["launch_id"] == launch["launch_id"]
    assert status["latest_generation"] == receipt.generation_id
    assert status["last_success_at"] is not None
    assert status["last_error"] is None
    assert status["consecutive_failures"] == 0


def test_backup_grace_uses_admitted_destruction_deadline() -> None:
    launch = {
        "hard_deadline": "1970-01-04T23:00:00+00:00",
        "offer": {"destroy_at": "1970-01-04T23:15:00+00:00"},
    }

    assert remote_backup._backup_grace_seconds(launch) == (
        remote_backup.MAX_BACKUP_GRACE_SECONDS - remote_backup.DESTROY_REQUEST_RESERVE_SECONDS
    )
    full_budget = {**launch, "offer": {"destroy_at": "1970-01-04T23:01:00+00:00"}}
    assert remote_backup._backup_grace_seconds(full_budget) == 0.0


def test_backup_and_unpin_failure_remains_a_runtime_gate_error(tmp_path: Path, monkeypatch) -> None:
    launch = _launch(tmp_path / "worker", tmp_path / "trusted")
    _patch_launch(monkeypatch, launch)

    def remote_checkpoint(_launch, mode, generation=None):
        if mode == "list":
            return {
                "generation": "update_00000020",
                "progress": {"updates_trained": 20, "launch_id": "current-launch"},
            }
        raise RuntimeGateError(f"{mode} failed")

    monkeypatch.setattr(remote_backup, "_remote_checkpoint", remote_checkpoint)

    with pytest.raises(RuntimeGateError, match="worker pin could not be removed"):
        remote_backup.backup_remote_once(tmp_path / "launch.json")


def test_terminal_state_comes_from_the_verified_receipt(tmp_path: Path) -> None:
    destination = tmp_path / "update_00000020"
    destination.mkdir()
    (destination / "progress.json").write_text(
        json.dumps({"updates_trained": 20, "training_complete": False, "launch_id": "current-launch"}),
        encoding="utf-8",
    )
    receipt = BackupReceipt("update_00000020", {}, "source", str(destination))
    launch = _launch(tmp_path / "worker", tmp_path / "trusted")

    progress = remote_backup._receipt_progress(receipt, "current-launch")

    assert not remote_backup._terminal_progress(progress, launch)


def test_rsync_timeout_is_capped_and_rejects_expired_copy_deadline(tmp_path: Path, monkeypatch) -> None:
    temporary = tmp_path / "update_00000020.partial"
    seen: list[float] = []

    def run(_command, **kwargs):
        seen.append(kwargs["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote_backup.subprocess, "run", run)
    monkeypatch.setattr(remote_backup.time, "time", lambda: 98.0)
    try:
        remote_backup._copy_remote_generation(
            "update_00000020", temporary, "runner@worker:/checkpoints/update_00000020", 22, 100.0
        )
    except RuntimeGateError:
        pass
    else:
        raise AssertionError("an empty fake copy must fail completeness")
    assert seen == [2.0]

    monkeypatch.setattr(remote_backup.time, "time", lambda: 99.5)
    try:
        remote_backup._copy_remote_generation(
            "update_00000020", temporary, "runner@worker:/checkpoints/update_00000020", 22, 100.0
        )
    except RuntimeGateError:
        pass
    else:
        raise AssertionError("copy should be rejected when less than one second remains")
    assert seen == [2.0]
