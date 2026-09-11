"""Tests for worker supervision and remote checkpoint inventory."""

from __future__ import annotations

import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from diarizen.trainer_utils import seal_checkpoint_directory
from recipes.speakrs.large.errors import RuntimeGateError
from recipes.speakrs.large.remote_backup import REMOTE_CHECKPOINT_SCRIPT
from recipes.speakrs.large.training_supervisor import (
    REQUIRED_RECOVERY_FILES,
    VAST_RENTAL_WATCHDOG_SCRIPT,
    _arm_rental_watchdog,
    _latest_progress,
    _signal_process_group,
    _verify_vast_watchdog_access,
    _worker_environment,
    supervise_training,
)


class _JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _checkpoint(root: Path, updates: int, launch_id: str | None = None) -> Path:
    path = root / f"update_{updates:08d}"
    path.mkdir(parents=True)
    (path / "pytorch_model.bin").write_bytes(f"model-{updates}".encode())
    (path / "progress.json").write_text(
        json.dumps(
            {
                "schema": "diarizen-checkpoint-progress-v1",
                "epochs_trained": 1,
                "steps_trained": updates * 8,
                "updates_trained": updates,
                "microbatches_in_epoch": 3,
                "launch_id": launch_id,
            }
        ),
        encoding="utf-8",
    )
    for filename in REQUIRED_RECOVERY_FILES:
        payload = path / filename
        if not payload.exists():
            payload.write_bytes(filename.encode())
    seal_checkpoint_directory(path)
    return path


def test_remote_inventory_and_supervisor_choose_newest_complete_generation(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 2_000)
    newest = _checkpoint(tmp_path, 4_000)
    incomplete = tmp_path / "update_00006000"
    incomplete.mkdir()
    (incomplete / "pytorch_model.bin").write_bytes(b"partial")

    pin = subprocess.run(
        [sys.executable, "-", str(tmp_path), "pin", newest.name],
        input=REMOTE_CHECKPOINT_SCRIPT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(pin.stdout)["pinned"] is True
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path), "hash", newest.name],
        input=REMOTE_CHECKPOINT_SCRIPT,
        text=True,
        capture_output=True,
        check=True,
    )
    inventory = json.loads(result.stdout)

    assert inventory["generation"] == newest.name
    assert inventory["progress"]["updates_trained"] == 4_000
    assert set(inventory["files"]) == set(REQUIRED_RECOVERY_FILES)
    assert _latest_progress(tmp_path)["generation"] == newest.name


def test_supervisor_rejects_checkpoint_from_another_launch(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 60_000, "old-launch")

    assert _latest_progress(tmp_path, "current-launch") is None
    assert _latest_progress(tmp_path, "old-launch")["updates_trained"] == 60_000


def test_worker_environment_binds_launch_and_scrubs_provider_tokens(monkeypatch) -> None:
    monkeypatch.setenv("VAST_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "/another/path")
    launch = {
        "launch_id": "launch-1",
        "worker_paths": {"repository_root": "/opt/diarizen"},
    }

    environment = _worker_environment(launch)

    assert "VAST_API_KEY" not in environment
    assert environment["SPEAKRS_LAUNCH_ID"] == "launch-1"
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == "/opt/diarizen"
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_supervisor_signal_reaches_accelerate_descendant(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    signaled = tmp_path / "signaled"
    child = (
        "import signal,sys,time; from pathlib import Path; "
        f"ready=Path({str(ready)!r}); signaled=Path({str(signaled)!r}); "
        "signal.signal(signal.SIGTERM, lambda *_: (signaled.write_text('yes'), sys.exit(0))); "
        "ready.write_text('yes'); time.sleep(60)"
    )
    parent = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child!r}]).wait()"
    process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)
    try:
        deadline = time.time() + 5
        while not ready.is_file() and time.time() < deadline:
            time.sleep(0.02)
        assert ready.is_file()

        _signal_process_group(process, signal.SIGTERM)
        process.wait(timeout=5)

        deadline = time.time() + 5
        while not signaled.is_file() and time.time() < deadline:
            time.sleep(0.02)
        assert signaled.is_file()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_rental_watchdog_arms_before_training_without_exposing_key(tmp_path: Path, monkeypatch) -> None:
    api_key = tmp_path / "vast-key"
    api_key.write_text("private-test-key", encoding="utf-8")
    destroy_at = datetime.now(timezone.utc) + timedelta(minutes=3)
    launch = {
        "launch_id": "launch-watchdog-test",
        "attempt_id": "attempt-watchdog-test",
        "offer": {
            "provider": "vast.ai",
            "instance_id": "50520000",
            "destroy_at": destroy_at.isoformat(),
        },
    }
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._verify_vast_watchdog_access",
        lambda *_args, **_kwargs: None,
    )

    armed = _arm_rental_watchdog(launch, tmp_path / "launch.json", api_key_path=api_key)

    try:
        status = json.loads(Path(armed["status_path"]).read_text(encoding="utf-8"))
        assert status["state"] == "armed"
        assert status["launch_id"] == launch["launch_id"]
        assert "private-test-key" not in Path(armed["status_path"]).read_text(encoding="utf-8")
    finally:
        os.kill(int(armed["pid"]), signal.SIGTERM)
        os.waitpid(int(armed["pid"]), 0)


def test_rental_watchdog_preflight_checks_instance_and_write_access(tmp_path: Path) -> None:
    api_key = tmp_path / "vast-key"
    api_key.write_text("private-test-key", encoding="utf-8")
    launch = {"offer": {"instance_id": "50520000"}}
    methods: list[str] = []

    def opener(request, **_kwargs):
        methods.append(request.get_method())
        if request.get_method() == "GET":
            return _JsonResponse({"instances": {"id": 50520000, "actual_status": "running"}})
        return _JsonResponse({"success": True})

    _verify_vast_watchdog_access(launch, api_key, opener=opener)

    assert methods == ["GET", "PUT"]


def test_rental_watchdog_retries_after_destroy_at_and_rereads_rotated_key(tmp_path: Path) -> None:
    api_key = tmp_path / "vast-key"
    api_key.write_text("first-private-test-key", encoding="utf-8")
    status_path = tmp_path / "watchdog-status.json"
    requests: list[tuple[str, str | None]] = []
    first_delete = tmp_path / "first-delete"
    key_rotated = tmp_path / "key-rotated"

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            return None

        def _reply(self, code: int, payload: dict[str, object] | None = None) -> None:
            body = b"" if payload is None else json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            requests.append(("GET", self.headers.get("Authorization")))
            self._reply(200, {"instances": {"id": "50520000", "actual_status": "running"}})

        def do_DELETE(self) -> None:
            requests.append(("DELETE", self.headers.get("Authorization")))
            if not first_delete.exists():
                first_delete.write_text("yes", encoding="utf-8")
                api_key.write_text("rotated-private-test-key", encoding="utf-8")
                key_rotated.write_text("yes", encoding="utf-8")
                self._reply(500)
                return
            self._reply(200, {"success": True})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    destroy_at = time.time() - 1
    environment = os.environ.copy()
    environment["SPEAKRS_VAST_INSTANCE_URL"] = f"http://127.0.0.1:{server.server_port}/instance/50520000/"
    command = [
        sys.executable,
        "-c",
        VAST_RENTAL_WATCHDOG_SCRIPT,
        str(api_key),
        str(status_path),
        "50520000",
        str(destroy_at),
        str(destroy_at),
        "launch-watchdog-retry",
        "attempt-watchdog-retry",
    ]
    process = subprocess.Popen(command, env=environment)
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("state") == "overdue-retrying":
                    break
            time.sleep(0.02)
        assert status_path.is_file()
        retry_status = json.loads(status_path.read_text(encoding="utf-8"))
        assert retry_status["state"] == "overdue-retrying"
        assert retry_status["overdue"] is True
        assert retry_status["error"] == "provider-http-500"
        assert retry_status["next_retry_at"] > retry_status["checked_at"]
        assert "first-private-test-key" not in status_path.read_text(encoding="utf-8")
        assert "rotated-private-test-key" not in status_path.read_text(encoding="utf-8")
        assert key_rotated.is_file()

        assert process.wait(timeout=5) == 0
        final_status = json.loads(status_path.read_text(encoding="utf-8"))
        assert final_status["state"] == "destroy-requested"
        assert final_status["attempts"] == 2
        assert requests == [
            ("GET", "Bearer first-private-test-key"),
            ("DELETE", "Bearer first-private-test-key"),
            ("GET", "Bearer rotated-private-test-key"),
            ("DELETE", "Bearer rotated-private-test-key"),
        ]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def _guard_proof(launch: dict[str, object], checked_at: datetime) -> dict[str, object]:
    offer = launch["offer"]
    assert isinstance(offer, dict)
    return {
        "schema": "speakrs-vast-rental-guard-status-v1",
        "attempt_id": launch["attempt_id"],
        "launch_id": launch["launch_id"],
        "instance_id": str(offer["instance_id"]),
        "state": "armed",
        "checked_at": checked_at.isoformat(),
        "deletion_outcome": None,
        "attempts": 0,
        "error_code": None,
        "overdue": False,
        "retrying": False,
    }


def test_supervise_training_rejects_stale_external_guard_proof(tmp_path: Path, monkeypatch) -> None:
    launch = {
        "launch_id": "launch-guard-proof",
        "attempt_id": "attempt-guard-proof",
        "offer": {"instance_id": "50520000"},
    }
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor.verify_worker_launch",
        lambda _path: launch,
    )
    proof_path = tmp_path / "external-guard.json"
    proof_path.write_text(
        json.dumps(_guard_proof(launch, datetime.now(timezone.utc) - timedelta(minutes=10))),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeGateError, match="external Vast rental guard proof is stale"):
        supervise_training(
            tmp_path / "launch.json",
            tmp_path / "supervisor-status.json",
            external_guard_proof_path=proof_path,
        )


def test_supervise_training_accepts_fresh_external_guard_before_worker_launch(tmp_path: Path, monkeypatch) -> None:
    training_root = tmp_path / "training"
    experiment_root = tmp_path / "experiment"
    checkpoint_root = tmp_path / "checkpoints"
    training_root.mkdir()
    experiment_root.mkdir()
    checkpoint_root.mkdir()
    launch = {
        "launch_id": "launch-guard-proof",
        "attempt_id": "attempt-guard-proof",
        "hard_deadline": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "max_updates": 1,
        "offer": {
            "instance_id": "50520000",
            "destroy_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
        "worker_paths": {
            "training_root": str(training_root),
            "trainer_config": str(training_root / "trainer.toml"),
            "checkpoint_root": str(checkpoint_root),
            "experiment_root": str(experiment_root),
        },
    }
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor.verify_worker_launch",
        lambda _path: launch,
    )
    proof_path = tmp_path / "external-guard.json"
    proof_path.write_text(json.dumps(_guard_proof(launch, datetime.now(timezone.utc))), encoding="utf-8")
    events: list[str] = []

    class CompletedProcess:
        pid = 123456

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._arm_rental_watchdog",
        lambda *_args, **_kwargs: events.append("watchdog") or {"pid": 1},
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor.subprocess.Popen",
        lambda *_args, **_kwargs: events.append("launch") or CompletedProcess(),
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._worker_environment",
        lambda _launch: {},
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._process_group_exists",
        lambda _process: False,
    )
    monkeypatch.setattr(
        "recipes.speakrs.large.training_supervisor._latest_progress",
        lambda *_args: None,
    )

    result = supervise_training(
        tmp_path / "launch.json",
        tmp_path / "supervisor-status.json",
        external_guard_proof_path=proof_path,
    )

    assert events == ["watchdog", "launch"]
    assert result["external_guard_proof"]["attempt_id"] == launch["attempt_id"]
