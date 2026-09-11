"""Trusted-controller copy of complete worker checkpoints over SSH."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from diarizen.trainer_utils import checkpoint_directory_is_complete, fsync_directory

from .errors import RuntimeGateError
from .jsonio import read_json, write_json
from .recovery import BackupReceipt, file_inventory, verify_receipt
from .training_admission import DESTROY_REQUEST_RESERVE_SECONDS, parse_launch_lock


SAFE_REMOTE = re.compile(r"^[A-Za-z0-9_./-]+$")
SAFE_HOST = re.compile(r"^[A-Za-z0-9.:-]+$")
SAFE_GENERATION = re.compile(r"^update_[0-9]{8}$")
SAFE_DIGEST = re.compile(r"^[0-9a-f]{64}$")

MAX_BACKUP_GRACE_SECONDS = 900.0
RSYNC_TIMEOUT_SECONDS = 7200
MIN_BOUNDED_COPY_SECONDS = 1.0
BACKUP_STATUS_FILENAME = "backup-status.json"


REMOTE_CHECKPOINT_SCRIPT = r"""
import fcntl
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath


root = Path(sys.argv[1])
mode = sys.argv[2]
requested_value = sys.argv[3] if len(sys.argv) > 3 else "-"
requested = None if requested_value == "-" else requested_value
expected_launch_id = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
lock_path = root / ".retention.lock"
pin_path = root / ".backup-pin"


def sync_directory(path):
    directory_fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def safe_payload_name(name):
    relative = PurePosixPath(name)
    return bool(relative.parts) and not relative.is_absolute() and all(part not in {"", ".", ".."} for part in relative.parts)


def payload_records(path):
    records = {}
    for item in sorted(path.rglob("*")):
        if not item.is_file() or item.is_symlink():
            continue
        relative = item.relative_to(path).as_posix()
        if relative in {".complete", ".files.json"}:
            continue
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        records[relative] = {"sha256": digest.hexdigest(), "size": item.stat().st_size}
    return records


def inspect_checkpoint(path):
    if not path.is_dir() or not (path / ".complete").is_file() or not (path / ".files.json").is_file():
        return None
    try:
        manifest = json.loads((path / ".files.json").read_text(encoding="utf-8"))
        progress = json.loads((path / "progress.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        return None
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return None
    if not isinstance(progress, dict):
        return None
    if expected_launch_id is not None and progress.get("launch_id") != expected_launch_id:
        return None
    updates = progress.get("updates_trained")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        return None
    expected_names = set()
    for name, size in files.items():
        if not isinstance(name, str) or not safe_payload_name(name):
            return None
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            return None
        candidate = path.joinpath(*PurePosixPath(name).parts)
        try:
            candidate.relative_to(path)
        except ValueError:
            return None
        if not candidate.is_file() or candidate.is_symlink() or candidate.stat().st_size != size:
            return None
        expected_names.add(name)
    actual_names = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink() and item.relative_to(path).as_posix() not in {".complete", ".files.json"}
    }
    if actual_names != expected_names or "progress.json" not in expected_names:
        return None
    return progress


def candidates():
    found = []
    for path in sorted(root.glob("update_????????")):
        if not re.fullmatch(r"update_[0-9]{8}", path.name):
            continue
        progress = inspect_checkpoint(path)
        if progress is not None:
            found.append((int(progress["updates_trained"]), path, progress))
    return found


if mode == "unpin":
    if requested is None:
        raise SystemExit(2)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            pinned = pin_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            pinned = None
        unpinned = pinned in {None, requested}
        if pinned == requested:
            pin_path.unlink(missing_ok=True)
            sync_directory(root)
    print(json.dumps({"generation": requested, "unpinned": unpinned}, sort_keys=True))
    raise SystemExit


if mode == "list":
    found = candidates()
    if requested is not None:
        found = [item for item in found if item[1].name == requested]
    if not found:
        print(json.dumps({"generation": None}, sort_keys=True))
        raise SystemExit
    _, path, progress = max(found, key=lambda item: (item[0], item[1].name))
    print(json.dumps({"generation": path.name, "progress": progress}, sort_keys=True))
    raise SystemExit


if mode == "pin":
    if requested is None:
        raise SystemExit(2)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        progress = inspect_checkpoint(root / requested)
        if progress is None:
            print(json.dumps({"generation": None}, sort_keys=True))
            raise SystemExit
        temporary = pin_path.with_name(pin_path.name + ".partial")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(requested + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(pin_path)
        sync_directory(root)
    print(json.dumps({"generation": requested, "progress": progress, "pinned": True}, sort_keys=True))
    raise SystemExit


if mode == "hash":
    if requested is None:
        raise SystemExit(2)
    try:
        pinned = pin_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        pinned = None
    progress = inspect_checkpoint(root / requested)
    if pinned != requested or progress is None:
        print(json.dumps({"generation": None}, sort_keys=True))
        raise SystemExit
    files = payload_records(root / requested)
    if not files:
        print(json.dumps({"generation": None}, sort_keys=True))
        raise SystemExit
    print(json.dumps({"generation": requested, "progress": progress, "files": files}, sort_keys=True))
    raise SystemExit


raise SystemExit(2)
"""


def _connection(launch: Mapping[str, Any]) -> tuple[str, int, str]:
    try:
        offer = launch["offer"]
        host = str(offer["ssh_host"])
        user = str(offer["ssh_user"])
        port = int(offer["ssh_port"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeGateError("launch SSH endpoint is invalid") from error
    if not SAFE_HOST.fullmatch(host) or not re.fullmatch(r"[A-Za-z0-9_-]+", user) or not 1 <= port <= 65535:
        raise RuntimeGateError("launch SSH endpoint contains unsafe characters")
    return host, port, user


def _launch_id(launch: Mapping[str, Any]) -> str:
    value = launch.get("launch_id")
    if not isinstance(value, str) or not value:
        raise RuntimeGateError("launch lock has no launch identifier")
    return value


def _matching_progress(payload: Mapping[str, Any], launch_id: str) -> Mapping[str, Any]:
    progress = payload.get("progress")
    if not isinstance(progress, Mapping) or progress.get("launch_id") != launch_id:
        raise RuntimeGateError("worker checkpoint progress belongs to a different launch")
    updates = progress.get("updates_trained")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise RuntimeGateError("worker checkpoint progress is invalid")
    return progress


def _copy_deadline(launch: Mapping[str, Any]) -> float:
    try:
        destroy_at = _datetime_from_iso(str(launch["offer"]["destroy_at"]))
    except (KeyError, TypeError) as error:
        raise RuntimeGateError("launch has no rental destruction deadline") from error
    return destroy_at - DESTROY_REQUEST_RESERVE_SECONDS


def _remote_checkpoint(launch: Mapping[str, Any], mode: str, generation: str | None = None) -> Mapping[str, Any]:
    if mode not in {"list", "pin", "hash", "unpin"}:
        raise RuntimeGateError("worker checkpoint mode is invalid", {"mode": mode})
    if generation is not None and not SAFE_GENERATION.fullmatch(generation):
        raise RuntimeGateError("worker checkpoint generation is invalid", {"generation": generation})
    host, port, user = _connection(launch)
    try:
        root = str(launch["worker_paths"]["checkpoint_root"])
    except (KeyError, TypeError) as error:
        raise RuntimeGateError("launch lock has no worker checkpoint root") from error
    if not root.startswith("/") or not SAFE_REMOTE.fullmatch(root):
        raise RuntimeGateError("worker checkpoint path contains unsafe characters")
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-p",
        str(port),
        f"{user}@{host}",
        "python3",
        "-",
        root,
        mode,
        generation or "-",
        _launch_id(launch),
    ]
    timeout = 3600 if mode == "hash" else 30
    if mode == "hash":
        deadline = _copy_deadline(launch)
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= MIN_BOUNDED_COPY_SECONDS:
                raise RuntimeGateError("backup copy deadline has expired")
            timeout = min(timeout, remaining)
    try:
        result = subprocess.run(
            command,
            input=REMOTE_CHECKPOINT_SCRIPT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeGateError("cannot inspect worker checkpoints", {"mode": mode}) from error
    if result.returncode != 0:
        raise RuntimeGateError("cannot inspect worker checkpoints", {"return_code": result.returncode, "mode": mode})
    try:
        payload = json.loads(result.stdout)
    except (TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeGateError("worker checkpoint inventory is invalid", {"mode": mode}) from error
    if not isinstance(payload, Mapping):
        raise RuntimeGateError("worker checkpoint inventory is not an object", {"mode": mode})
    return payload


def _fsync_file(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError as error:
        raise RuntimeGateError("cannot synchronize backup payload", {"path": str(path)}) from error


def _fsync_tree(root: Path) -> None:
    """Synchronize every file and directory below one copied checkpoint."""

    if not root.is_dir():
        raise RuntimeGateError("backup payload directory is missing", {"path": str(root)})
    files = [path for path in root.rglob("*") if path.is_file()]
    for path in sorted(files):
        _fsync_file(path)
    directories = {root, *(path.parent for path in files)}
    for directory in sorted(directories, key=lambda path: (len(path.parts), str(path)), reverse=True):
        fsync_directory(directory)


def _validated_inventory(value: object, label: str) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeGateError(f"{label} inventory is invalid")
    inventory: dict[str, dict[str, object]] = {}
    for name, record in value.items():
        if not isinstance(name, str) or not name or not SAFE_REMOTE.fullmatch(name):
            raise RuntimeGateError(f"{label} inventory contains an unsafe file name")
        if name.startswith("/") or any(part in {"", ".", ".."} for part in Path(name).parts):
            raise RuntimeGateError(f"{label} inventory contains an unsafe file name")
        if not isinstance(record, Mapping) or set(record) != {"sha256", "size"}:
            raise RuntimeGateError(f"{label} inventory contains an invalid file record")
        digest = record.get("sha256")
        size = record.get("size")
        if not isinstance(digest, str) or not SAFE_DIGEST.fullmatch(digest):
            raise RuntimeGateError(f"{label} inventory contains an invalid SHA-256 digest")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeGateError(f"{label} inventory contains an invalid file size")
        inventory[name] = {"sha256": digest, "size": size}
    return inventory


def _local_inventory(root: Path) -> dict[str, dict[str, object]]:
    try:
        return _validated_inventory(file_inventory(root), "trusted")
    except (OSError, RuntimeGateError) as error:
        if isinstance(error, RuntimeGateError):
            raise
        raise RuntimeGateError("trusted checkpoint inventory could not be read", {"path": str(root)}) from error


def _remove_path(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError as error:
        raise RuntimeGateError("cannot repair interrupted trusted backup", {"path": str(path)}) from error


def _discard_backup_artifacts(destination: Path) -> None:
    _remove_path(destination)
    _remove_path(destination.with_name(destination.name + ".partial"))
    _remove_path(destination.with_name(destination.name + ".receipt.json"))
    _remove_path(destination.with_name(destination.name + ".receipt.json.partial"))
    fsync_directory(destination.parent)


def _receipt_matches(receipt: BackupReceipt, destination: Path, generation: str, source: str) -> bool:
    return (
        receipt.generation_id == generation
        and receipt.source == source
        and receipt.destination == str(destination)
        and bool(receipt.files)
    )


def _trusted_receipt(destination: Path, generation: str, source: str) -> BackupReceipt | None:
    try:
        receipt = verify_receipt(destination)
    except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeGateError):
        return None
    return receipt if _receipt_matches(receipt, destination, generation, source) else None


def _prune_trusted_generations(trusted_root: Path) -> None:
    """Retain at most the two newest generations with valid receipts."""

    verified: list[tuple[int, Path]] = []
    invalid: list[Path] = []
    for candidate in sorted(trusted_root.glob("update_????????")):
        if not SAFE_GENERATION.fullmatch(candidate.name):
            continue
        try:
            receipt = verify_receipt(candidate)
            if not _receipt_matches(receipt, candidate, candidate.name, receipt.source):
                raise RuntimeGateError("trusted receipt identity does not match its generation")
        except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeGateError):
            invalid.append(candidate)
            continue
        verified.append((int(candidate.name[7:]), candidate))

    verified.sort(key=lambda item: item[0])
    for _, candidate in verified:
        _fsync_tree(candidate)
        _fsync_file(candidate.with_name(candidate.name + ".receipt.json"))
    fsync_directory(trusted_root)
    for old in invalid:
        _remove_path(old)
        _remove_path(old.with_name(old.name + ".receipt.json"))
    for _, old in verified[:-2]:
        _remove_path(old)
        _remove_path(old.with_name(old.name + ".receipt.json"))
    for receipt in trusted_root.glob("update_*.receipt.json*"):
        generation_name = receipt.name.split(".receipt.json", 1)[0]
        if not (trusted_root / generation_name).exists() or receipt.name.endswith(".partial"):
            _remove_path(receipt)
    fsync_directory(trusted_root)


def _copy_remote_generation(
    generation: str,
    temporary: Path,
    source: str,
    port: int,
    copy_deadline: float | None,
) -> None:
    timeout = float(RSYNC_TIMEOUT_SECONDS)
    if copy_deadline is not None:
        remaining = copy_deadline - time.time()
        if remaining <= MIN_BOUNDED_COPY_SECONDS:
            raise RuntimeGateError("backup copy deadline has expired", {"generation": generation})
        timeout = min(timeout, remaining)
    try:
        temporary.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--checksum",
                "-e",
                f"ssh -o BatchMode=yes -o ConnectTimeout=15 -p {port}",
                f"{source}/",
                f"{temporary}/",
            ],
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeGateError("checkpoint rsync failed", {"generation": generation}) from error
    if result.returncode != 0:
        raise RuntimeGateError("checkpoint rsync failed", {"return_code": result.returncode, "generation": generation})
    _fsync_tree(temporary)
    try:
        complete = checkpoint_directory_is_complete(temporary)
    except OSError as error:
        raise RuntimeGateError("copied checkpoint inventory could not be read", {"generation": generation}) from error
    if not complete:
        raise RuntimeGateError("copied checkpoint is incomplete", {"generation": generation})


def backup_remote_once(launch_path: Path) -> BackupReceipt | None:
    """Copy and verify the newest complete worker generation once."""

    launch = parse_launch_lock(read_json(launch_path))
    expected_launch_id = _launch_id(launch)
    latest = _remote_checkpoint(launch, "list")
    generation = latest.get("generation")
    if generation is None:
        trusted_root = Path(launch["worker_paths"]["trusted_backup_root"])
        try:
            trusted_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise RuntimeGateError("trusted backup root is not writable", {"path": str(trusted_root)}) from error
        _prune_trusted_generations(trusted_root)
        return None
    if not isinstance(generation, str) or not SAFE_GENERATION.fullmatch(generation):
        raise RuntimeGateError("worker returned an invalid checkpoint generation")
    _matching_progress(latest, expected_launch_id)
    pin_attempted = False
    pending_error: BaseException | None = None
    receipt: BackupReceipt | None = None
    try:
        pin_attempted = True
        pinned_result = _remote_checkpoint(launch, "pin", generation)
        if pinned_result.get("generation") != generation or pinned_result.get("pinned") is not True:
            raise RuntimeGateError("worker checkpoint could not be pinned", {"generation": generation})
        if "progress" in pinned_result:
            _matching_progress(pinned_result, expected_launch_id)
        trusted_root = Path(launch["worker_paths"]["trusted_backup_root"])
        try:
            trusted_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise RuntimeGateError("trusted backup root is not writable", {"path": str(trusted_root)}) from error
        destination = trusted_root / generation
        host, port, user = _connection(launch)
        worker_root = str(launch["worker_paths"]["checkpoint_root"])
        source = f"{user}@{host}:{worker_root}/{generation}"
        temporary = destination.with_name(destination.name + ".partial")
        copy_deadline = _copy_deadline(launch)
        existing = _trusted_receipt(destination, generation, source) if destination.exists() else None
        if existing is not None:
            remote = _remote_checkpoint(launch, "hash", generation)
            remote_inventory = _validated_inventory(remote.get("files"), "worker")
            local_inventory = _local_inventory(destination)
            if remote.get("generation") != generation or remote_inventory != local_inventory:
                _discard_backup_artifacts(destination)
            else:
                receipt = existing
        elif (
            destination.exists()
            or temporary.exists()
            or destination.with_name(destination.name + ".receipt.json").exists()
            or destination.with_name(destination.name + ".receipt.json.partial").exists()
        ):
            _discard_backup_artifacts(destination)

        if receipt is None:
            _copy_remote_generation(generation, temporary, source, port, copy_deadline)
            remote = _remote_checkpoint(launch, "hash", generation)
            if remote.get("generation") != generation:
                raise RuntimeGateError("worker returned an invalid checkpoint hash", {"generation": generation})
            remote_inventory = _validated_inventory(remote.get("files"), "worker")
            local_inventory = _local_inventory(temporary)
            if remote_inventory != local_inventory:
                raise RuntimeGateError("copied checkpoint hashes differ from the worker", {"generation": generation})
            _fsync_tree(temporary)
            try:
                temporary.replace(destination)
            except OSError as error:
                raise RuntimeGateError("cannot publish trusted checkpoint", {"generation": generation}) from error
            fsync_directory(destination.parent)
            receipt = BackupReceipt(
                generation_id=generation,
                files=remote_inventory,
                source=source,
                destination=str(destination),
            )
            try:
                receipt_path = destination.with_name(destination.name + ".receipt.json")
                write_json(receipt_path, receipt.identity())
                _fsync_file(receipt_path)
            except OSError as error:
                raise RuntimeGateError("cannot publish trusted backup receipt", {"generation": generation}) from error
            fsync_directory(destination.parent)
        _prune_trusted_generations(trusted_root)
        return receipt
    except BaseException as error:
        pending_error = error
        raise
    finally:
        if pin_attempted:
            cleanup_error: RuntimeGateError | None = None
            try:
                unpinned = _remote_checkpoint(launch, "unpin", generation)
                if unpinned.get("unpinned") is not True:
                    cleanup_error = RuntimeGateError(
                        "worker checkpoint pin could not be removed", {"generation": generation}
                    )
            except RuntimeGateError as error:
                cleanup_error = error
            if cleanup_error is not None:
                if pending_error is None:
                    raise cleanup_error
                raise RuntimeGateError(
                    "checkpoint backup failed and the worker pin could not be removed",
                    {"generation": generation},
                ) from pending_error


def _backup_grace_seconds(launch: Mapping[str, Any]) -> float:
    """Return the admitted backup grace before automatic rental destruction."""

    try:
        hard_deadline = _datetime_from_iso(str(launch["hard_deadline"]))
        destroy_at = _copy_deadline(launch)
    except (KeyError, TypeError, ValueError, RuntimeGateError):
        return 0.0
    return max(0.0, min(MAX_BACKUP_GRACE_SECONDS, destroy_at - hard_deadline))


def _sleep_until(deadline: float, poll_seconds: int) -> bool:
    remaining = deadline - time.time()
    if remaining <= 0:
        return False
    time.sleep(min(float(poll_seconds), remaining))
    return True


def _status_error(error: RuntimeGateError) -> str:
    """Return a token-safe status message without subprocess output."""

    return error.message


def _write_backup_status(
    trusted_root: Path,
    launch_id: str,
    *,
    state: str,
    latest_generation: str | None,
    last_success_at: str | None,
    last_error: str | None,
    consecutive_failures: int,
) -> None:
    checked_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema": "speakrs-remote-backup-status-v1",
        "state": state,
        "launch_id": launch_id,
        "latest_generation": latest_generation,
        "last_success_at": last_success_at,
        "last_error": last_error,
        "consecutive_failures": consecutive_failures,
        "checked_at": checked_at,
    }
    try:
        trusted_root.mkdir(parents=True, exist_ok=True)
        write_json(trusted_root / BACKUP_STATUS_FILENAME, payload)
    except OSError as error:
        raise RuntimeGateError("cannot publish backup monitor status", {"path": str(trusted_root)}) from error


def _receipt_progress(receipt: BackupReceipt, launch_id: str) -> Mapping[str, Any]:
    try:
        progress = read_json(Path(receipt.destination) / "progress.json")
    except (OSError, ValueError) as error:
        raise RuntimeGateError("trusted checkpoint progress is invalid") from error
    return _matching_progress({"progress": progress}, launch_id)


def monitor_remote_backups(launch_path: Path, poll_seconds: int) -> dict[str, object]:
    """Keep the newest two trusted checkpoints until completion or the cutoff."""

    if poll_seconds < 10:
        raise RuntimeGateError("backup poll interval must be at least ten seconds")
    launch = parse_launch_lock(read_json(launch_path))
    launch_id = _launch_id(launch)
    trusted_root = Path(launch["worker_paths"]["trusted_backup_root"])
    deadline = _datetime_from_iso(str(launch["hard_deadline"]))
    grace_deadline = _copy_deadline(launch)
    last_generation: str | None = None
    last_success_at: str | None = None
    last_error: str | None = None
    consecutive_failures = 0
    terminal = False
    _write_backup_status(
        trusted_root,
        launch_id,
        state="running",
        latest_generation=last_generation,
        last_success_at=last_success_at,
        last_error=last_error,
        consecutive_failures=consecutive_failures,
    )
    while time.time() < deadline:
        try:
            receipt = backup_remote_once(launch_path)
            if receipt is not None:
                last_generation = receipt.generation_id
            last_success_at = datetime.now(timezone.utc).isoformat()
            last_error = None
            consecutive_failures = 0
            terminal = (
                _terminal_progress(_receipt_progress(receipt, launch_id), launch) if receipt is not None else False
            )
            _write_backup_status(
                trusted_root,
                launch_id,
                state="running",
                latest_generation=last_generation,
                last_success_at=last_success_at,
                last_error=last_error,
                consecutive_failures=consecutive_failures,
            )
            if terminal:
                break
        except RuntimeGateError as error:
            consecutive_failures += 1
            last_error = _status_error(error)
            _write_backup_status(
                trusted_root,
                launch_id,
                state="retrying",
                latest_generation=last_generation,
                last_success_at=last_success_at,
                last_error=last_error,
                consecutive_failures=consecutive_failures,
            )
        if not _sleep_until(deadline, poll_seconds):
            break

    generation_at_cutoff = last_generation
    attempted_final_copy = False
    while True:
        if attempted_final_copy and time.time() >= grace_deadline:
            break
        attempted_final_copy = True
        try:
            receipt = backup_remote_once(launch_path)
            if receipt is not None:
                last_generation = receipt.generation_id
            last_success_at = datetime.now(timezone.utc).isoformat()
            last_error = None
            consecutive_failures = 0
            terminal = (
                _terminal_progress(_receipt_progress(receipt, launch_id), launch) if receipt is not None else False
            )
            _write_backup_status(
                trusted_root,
                launch_id,
                state="running",
                latest_generation=last_generation,
                last_success_at=last_success_at,
                last_error=last_error,
                consecutive_failures=consecutive_failures,
            )
            if terminal or last_generation != generation_at_cutoff:
                break
        except RuntimeGateError as error:
            consecutive_failures += 1
            last_error = _status_error(error)
            _write_backup_status(
                trusted_root,
                launch_id,
                state="retrying",
                latest_generation=last_generation,
                last_success_at=last_success_at,
                last_error=last_error,
                consecutive_failures=consecutive_failures,
            )
        if not _sleep_until(grace_deadline, poll_seconds):
            break
    _write_backup_status(
        trusted_root,
        launch_id,
        state="completed" if terminal else "stopped",
        latest_generation=last_generation,
        last_success_at=last_success_at,
        last_error=last_error,
        consecutive_failures=consecutive_failures,
    )
    return {"ok": True, "command": "backup", "latest_generation": last_generation}


def _terminal_progress(progress: object, launch: Mapping[str, Any]) -> bool:
    """Return whether a backed-up checkpoint durably ends the worker stage."""

    if not isinstance(progress, Mapping):
        return False
    updates = progress.get("updates_trained")
    complete = progress.get("training_complete") is True
    try:
        maximum = int(launch["max_updates"])
    except (KeyError, TypeError, ValueError):
        return complete
    return complete or (isinstance(updates, int) and not isinstance(updates, bool) and updates >= maximum)


def _datetime_from_iso(value: str) -> float:
    """Return a deadline as epoch seconds."""

    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError as error:
        raise RuntimeGateError("backup deadline is invalid") from error
