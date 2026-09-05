"""Transactional recovery generations and trusted backup receipts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import RuntimeGateError
from .hashing import sha256_file
from .jsonio import write_json


def _checkpoint_ops():
    from diarizen.trainer_utils import checkpoint_directory_is_complete, fsync_directory, seal_checkpoint_directory

    return checkpoint_directory_is_complete, fsync_directory, seal_checkpoint_directory


GENERATION_PREFIX = "update_"
WORKER_RECOVERY_KEEP = 3
TRUSTED_KEEP = 2


def generation_name(updates: int) -> str:
    """Return a discoverable update-boundary generation directory name."""

    return f"{GENERATION_PREFIX}{int(updates):08d}"


def is_generation_dir(path: Path) -> bool:
    """Return whether a directory uses the Large recovery naming."""

    name = path.name
    return name.startswith(GENERATION_PREFIX) and name[len(GENERATION_PREFIX) :].isdigit() and len(name) == 15


def complete_generations(root: Path) -> list[Path]:
    """Return complete generations in increasing update order."""

    if not root.is_dir():
        return []
    found = [path for path in root.iterdir() if path.is_dir() and is_generation_dir(path)]
    found.sort(key=lambda path: int(path.name.split("_", 1)[1]))
    checkpoint_directory_is_complete, _, _ = _checkpoint_ops()
    return [path for path in found if checkpoint_directory_is_complete(path)]


def newest_complete_generation(root: Path) -> Path | None:
    """Return the newest complete generation, ignoring incomplete copies."""

    generations = complete_generations(root)
    return generations[-1] if generations else None


def publish_generation(temporary: Path, destination: Path) -> Path:
    """Seal a temporary payload and publish it as an immutable generation."""

    if destination.exists():
        raise RuntimeGateError("generation already exists", {"path": str(destination)})
    checkpoint_directory_is_complete, fsync_directory, seal_checkpoint_directory = _checkpoint_ops()
    seal_checkpoint_directory(temporary)
    if not checkpoint_directory_is_complete(temporary):
        raise RuntimeGateError("incomplete generation cannot be published", {"path": str(temporary)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    fsync_directory(destination.parent)
    return destination


def retain_worker_generations(root: Path, keep: int = WORKER_RECOVERY_KEEP, pinned: Path | None = None) -> None:
    """Keep the newest worker recovery states. Never delete a pinned copy source."""

    generations = complete_generations(root)
    extra = generations[:-keep] if keep else generations
    for path in extra:
        if pinned is not None and path.resolve() == pinned.resolve():
            continue
        shutil.rmtree(path)


@dataclass(frozen=True)
class BackupReceipt:
    """Trusted-controller proof that a generation was copied and hashed."""

    generation_id: str
    files: dict[str, dict[str, object]]
    source: str
    destination: str

    def identity(self) -> dict[str, object]:
        """Return the sealed receipt."""

        return {
            "generation_id": self.generation_id,
            "files": self.files,
            "source": self.source,
            "destination": self.destination,
        }


def file_inventory(root: Path) -> dict[str, dict[str, object]]:
    """Return size and SHA-256 for every payload file."""

    records = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {".complete", ".files.json"}:
            continue
        records[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    if not records:
        raise RuntimeGateError("backup source has no payload files", {"path": str(root)})
    return records


def copy_generation(source: Path, destination: Path) -> BackupReceipt:
    """Copy a complete generation through a temporary directory and receipt."""

    checkpoint_directory_is_complete, fsync_directory, seal_checkpoint_directory = _checkpoint_ops()
    if not checkpoint_directory_is_complete(source):
        raise RuntimeGateError("refusing to copy an incomplete generation", {"path": str(source)})
    if destination.exists():
        raise RuntimeGateError("backup destination already exists", {"path": str(destination)})
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    expected = file_inventory(source)
    actual = file_inventory(temporary)
    if actual != expected:
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeGateError("backup copy hash mismatch", {"source": str(source)})
    seal_checkpoint_directory(temporary)
    temporary.replace(destination)
    fsync_directory(destination.parent)
    receipt = BackupReceipt(
        generation_id=source.name,
        files=expected,
        source=str(source),
        destination=str(destination),
    )
    write_json(destination.with_name(destination.name + ".receipt.json"), receipt.identity())
    return receipt


def verify_receipt(destination: Path, receipt: Mapping[str, object] | None = None) -> BackupReceipt:
    """Reject a corrupt or incomplete trusted backup."""

    checkpoint_directory_is_complete, _, _ = _checkpoint_ops()
    if not checkpoint_directory_is_complete(destination):
        raise RuntimeGateError("backup is incomplete", {"path": str(destination)})
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    payload = receipt or json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_files = payload.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise RuntimeGateError("backup receipt is missing file hashes")
    actual = file_inventory(destination)
    comparable = dict(actual)
    expected = dict(expected_files)
    if comparable != expected:
        raise RuntimeGateError("backup receipt does not match files", {"path": str(destination)})
    return BackupReceipt(
        generation_id=str(payload["generation_id"]),
        files=expected,
        source=str(payload["source"]),
        destination=str(payload["destination"]),
    )


class LocalTransport:
    """Filesystem transport used by CPU backup tests and the trusted controller."""

    def pin_and_copy(self, source: Path, destination: Path) -> BackupReceipt:
        """Copy one complete generation onto trusted storage."""

        return copy_generation(source, destination)


def restore_into(trusted_root: Path, worker_root: Path) -> Path:
    """Restore the newest valid trusted generation into a fresh worker directory."""

    generations = []
    for path in complete_generations(trusted_root):
        try:
            verify_receipt(path)
        except RuntimeGateError:
            continue
        generations.append(path)
    if not generations:
        raise RuntimeGateError("no valid trusted generation to restore")
    source = generations[-1]
    if worker_root.exists() and any(worker_root.iterdir()):
        raise RuntimeGateError("restore requires an empty destination", {"path": str(worker_root)})
    destination = worker_root / source.name
    shutil.copytree(source, destination)
    checkpoint_directory_is_complete, _, _ = _checkpoint_ops()
    if not checkpoint_directory_is_complete(destination):
        raise RuntimeGateError("restored generation is incomplete")
    return destination


def ranked_model_cannot_resume(path: Path) -> None:
    """Reject model-only ranked files as training recovery."""

    required = ("pytorch_model.bin",)
    checkpoint_directory_is_complete, _, _ = _checkpoint_ops()
    if checkpoint_directory_is_complete(path, required) and not (path / "optimizer.bin").exists():
        raise RuntimeGateError("model-only ranked files cannot resume training", {"path": str(path)})
