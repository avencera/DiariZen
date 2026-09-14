"""Transactional recovery generations and trusted backup receipts."""

from __future__ import annotations

import json
import os
import re
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
GENERATION_SEPARATOR = "_generation_"
GENERATION_WIDTH = 8
WORKER_RECOVERY_KEEP = 3
TRUSTED_KEEP = 2

_GENERATION_PATTERN = re.compile(r"^update_([0-9]{8})_generation_([0-9]{8})$")
_LEGACY_GENERATION_PATTERN = re.compile(r"^update_([0-9]{8})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_FIELDS = frozenset({"generation_id", "files", "source", "destination"})
_RECEIPT_FIELDS_WITH_PROGRESS = _RECEIPT_FIELDS | frozenset({"trainer_progress_sha256"})


@dataclass(frozen=True, order=True)
class RecoveryGeneration:
    """Canonical identity for one immutable optimizer-update generation."""

    updates: int
    sequence: int

    def __post_init__(self) -> None:
        if isinstance(self.updates, bool) or not isinstance(self.updates, int) or self.updates < 0:
            raise ValueError("optimizer updates must be a nonnegative integer")
        if self.updates >= 10**GENERATION_WIDTH:
            raise ValueError("optimizer updates exceeds the canonical name limit")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("generation sequence must be a positive integer")
        if self.sequence >= 10**GENERATION_WIDTH:
            raise ValueError("generation sequence exceeds the canonical name limit")

    @property
    def name(self) -> str:
        """Return the canonical directory name."""

        return (
            f"{GENERATION_PREFIX}{self.updates:0{GENERATION_WIDTH}d}"
            f"{GENERATION_SEPARATOR}{self.sequence:0{GENERATION_WIDTH}d}"
        )

    @classmethod
    def from_name(cls, name: str) -> "RecoveryGeneration":
        """Parse a canonical generation directory name."""

        if not isinstance(name, str):
            raise ValueError("generation name must be text")
        match = _GENERATION_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError("generation name is not canonical")
        return cls(int(match.group(1)), int(match.group(2)))


def generation_name(updates: int, sequence: int = 1) -> str:
    """Return the canonical name for one recovery generation."""

    return RecoveryGeneration(updates, sequence).name


def parse_generation_name(name: str) -> RecoveryGeneration:
    """Parse one canonical recovery generation name."""

    return RecoveryGeneration.from_name(name)


def _legacy_updates(name: str) -> int | None:
    match = _LEGACY_GENERATION_PATTERN.fullmatch(name)
    return int(match.group(1)) if match is not None else None


def _canonical_identity(path: Path) -> RecoveryGeneration | None:
    try:
        return RecoveryGeneration.from_name(path.name)
    except ValueError:
        return None


def _validate_updates(updates: int) -> int:
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("optimizer updates must be a nonnegative integer")
    if updates >= 10**GENERATION_WIDTH:
        raise ValueError("optimizer updates exceeds the canonical name limit")
    return updates


def _is_canonical_generation(path: Path) -> bool:
    return _canonical_identity(path) is not None


def is_generation_dir(path: Path) -> bool:
    """Return whether a path uses canonical or persisted legacy generation naming."""

    return _is_canonical_generation(path) or _legacy_updates(path.name) is not None


def next_generation(root: Path, updates: int) -> RecoveryGeneration:
    """Return the next sequence for an optimizer update without replacing history."""

    update = _validate_updates(updates)
    highest = 0
    if root.is_dir():
        for candidate in root.iterdir():
            if not candidate.is_dir() or not _is_canonical_generation(candidate):
                continue
            identity = _canonical_identity(candidate)
            if identity is not None and identity.updates == update:
                highest = max(highest, identity.sequence)
    return RecoveryGeneration(update, highest + 1)


def _generation_sort_key(path: Path) -> tuple[int, int, int, str]:
    identity = _canonical_identity(path)
    if identity is not None:
        return (identity.updates, identity.sequence, 0, path.name)
    legacy_updates = _legacy_updates(path.name)
    if legacy_updates is not None:
        # persisted update-only names sort before the first canonical sequence at the same update
        return (legacy_updates, 0, -1, path.name)
    return (-1, -1, -1, path.name)


def _generation_is_complete(path: Path) -> bool:
    checkpoint_directory_is_complete, _, _ = _checkpoint_ops()
    if _is_canonical_generation(path):
        return checkpoint_directory_is_complete(
            path,
            ("progress.json",),
            require_hashed_manifest=True,
        )
    # this is a read-only compatibility path for source-commit update-only directories
    return checkpoint_directory_is_complete(path)


def complete_generations(root: Path) -> list[Path]:
    """Return complete generations in optimizer-update and sequence order."""

    if not root.is_dir():
        return []
    found = [path for path in root.iterdir() if path.is_dir() and is_generation_dir(path)]
    found.sort(key=_generation_sort_key)
    return [path for path in found if _generation_is_complete(path)]


def newest_complete_generation(root: Path) -> Path | None:
    """Return the newest complete generation, ignoring incomplete copies."""

    generations = complete_generations(root)
    return generations[-1] if generations else None


def _fsync_tree(root: Path, fsync_directory) -> None:
    """Synchronize all files and directories below a staged generation."""

    files = [path for path in root.rglob("*") if path.is_file()]
    for path in sorted(files, key=lambda candidate: candidate.relative_to(root).as_posix()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directories = {path.parent for path in files} | {root}
    for directory in sorted(directories, key=lambda candidate: len(candidate.parts), reverse=True):
        fsync_directory(directory)


def _trainer_progress_digest(root: Path) -> str | None:
    progress = root / "progress.json"
    if not progress.is_file():
        return None
    from diarizen.trainer_utils import canonical_trainer_progress_digest

    try:
        return canonical_trainer_progress_digest(progress)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeGateError("trainer progress is invalid", {"path": str(progress)}) from error


def _generation_content(root: Path) -> tuple[dict[str, dict[str, object]], str | None]:
    return file_inventory(root), _trainer_progress_digest(root)


def _same_generation_content(left: Path, right: Path) -> bool:
    try:
        return _generation_content(left) == _generation_content(right)
    except (OSError, RuntimeGateError):
        return False


def _require_canonical_destination(destination: Path) -> None:
    if not _is_canonical_generation(destination):
        raise RuntimeGateError("new generation identity is not canonical", {"path": str(destination)})


def publish_generation(temporary: Path, destination: Path) -> Path:
    """Seal and atomically publish one immutable canonical recovery generation."""

    _require_canonical_destination(destination)
    if temporary == destination:
        raise RuntimeGateError("generation staging path must differ from destination")
    if not temporary.is_dir():
        raise RuntimeGateError("generation staging directory is missing", {"path": str(temporary)})

    checkpoint_directory_is_complete, fsync_directory, seal_checkpoint_directory = _checkpoint_ops()
    if (temporary / ".complete").exists():
        complete = checkpoint_directory_is_complete(
            temporary,
            ("progress.json",),
            require_hashed_manifest=True,
        )
        if not complete:
            raise RuntimeGateError("staged generation is already sealed but invalid", {"path": str(temporary)})
    else:
        seal_checkpoint_directory(temporary, require_trainer_progress=True)
        if not checkpoint_directory_is_complete(
            temporary,
            ("progress.json",),
            require_hashed_manifest=True,
        ):
            raise RuntimeGateError("incomplete generation cannot be published", {"path": str(temporary)})
    _fsync_tree(temporary, fsync_directory)

    if destination.exists() or destination.is_symlink():
        if (
            destination.is_dir()
            and _generation_is_complete(destination)
            and _same_generation_content(temporary, destination)
        ):
            shutil.rmtree(temporary)
            return destination
        raise RuntimeGateError("generation identity conflicts with published content", {"path": str(destination)})

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.replace(destination)
    except FileExistsError as error:
        # a concurrent identical publication may win between the existence check and rename
        if (
            destination.is_dir()
            and _generation_is_complete(destination)
            and _same_generation_content(temporary, destination)
        ):
            shutil.rmtree(temporary)
            return destination
        raise RuntimeGateError(
            "generation identity conflicts with published content", {"path": str(destination)}
        ) from error
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
    trainer_progress_sha256: str | None = None

    def identity(self) -> dict[str, object]:
        """Return the sealed receipt."""

        payload: dict[str, object] = {
            "generation_id": self.generation_id,
            "files": {name: dict(record) for name, record in self.files.items()},
            "source": self.source,
            "destination": self.destination,
        }
        if self.trainer_progress_sha256 is not None:
            payload["trainer_progress_sha256"] = self.trainer_progress_sha256
        return payload


def _safe_payload_name(name: object) -> bool:
    if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name:
        return False
    return all(part not in {"", ".", ".."} for part in name.split("/"))


def _validated_inventory(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeGateError("backup receipt has no file inventory")
    inventory: dict[str, dict[str, object]] = {}
    for name, record in value.items():
        if not _safe_payload_name(name):
            raise RuntimeGateError("backup receipt contains an unsafe file name")
        if not isinstance(record, Mapping) or set(record) != {"sha256", "size"}:
            raise RuntimeGateError("backup receipt contains an invalid file record")
        digest = record.get("sha256")
        size = record.get("size")
        if (
            not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise RuntimeGateError("backup receipt contains an invalid file record")
        inventory[name] = {"sha256": digest, "size": size}
    return inventory


def file_inventory(root: Path) -> dict[str, dict[str, object]]:
    """Return exact size and SHA-256 records for every payload file."""

    if not root.is_dir():
        raise RuntimeGateError("backup source is not a directory", {"path": str(root)})
    records: dict[str, dict[str, object]] = {}
    excluded = {".complete", ".complete.partial", ".files.json", ".files.json.partial"}
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise RuntimeGateError("backup payload contains a symbolic link", {"path": str(path)})
        relative = path.relative_to(root).as_posix()
        if not _safe_payload_name(relative):
            raise RuntimeGateError("backup payload contains an unsafe file name", {"path": relative})
        if relative in excluded:
            continue
        records[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    if not records:
        raise RuntimeGateError("backup source has no payload files", {"path": str(root)})
    return records


def _receipt_from_payload(payload: object) -> BackupReceipt:
    if not isinstance(payload, Mapping):
        raise RuntimeGateError("backup receipt is not an object")
    fields = _RECEIPT_FIELDS_WITH_PROGRESS if "trainer_progress_sha256" in payload else _RECEIPT_FIELDS
    if set(payload) != fields:
        raise RuntimeGateError("backup receipt fields are not exact")
    generation_id = payload.get("generation_id")
    source = payload.get("source")
    destination = payload.get("destination")
    if not isinstance(generation_id, str) or not is_generation_dir(Path(generation_id)):
        raise RuntimeGateError("backup receipt has an invalid generation identity")
    if not isinstance(source, str) or not source or not isinstance(destination, str) or not destination:
        raise RuntimeGateError("backup receipt has invalid locations")
    progress_digest = payload.get("trainer_progress_sha256")
    if progress_digest is not None and (
        not isinstance(progress_digest, str) or _SHA256_PATTERN.fullmatch(progress_digest) is None
    ):
        raise RuntimeGateError("backup receipt has an invalid trainer-progress digest")
    return BackupReceipt(
        generation_id=generation_id,
        files=_validated_inventory(payload.get("files")),
        source=source,
        destination=destination,
        trainer_progress_sha256=progress_digest if isinstance(progress_digest, str) else None,
    )


def copy_generation(source: Path, destination: Path) -> BackupReceipt:
    """Copy one complete generation through a staged immutable destination."""

    if not _generation_is_complete(source):
        raise RuntimeGateError("refusing to copy an incomplete generation", {"path": str(source)})
    if destination.name != source.name:
        raise RuntimeGateError("backup destination does not preserve generation identity")
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_dir()
            and _generation_is_complete(destination)
            and _same_generation_content(source, destination)
        ):
            receipt_path = destination.with_name(destination.name + ".receipt.json")
            if receipt_path.is_file():
                return verify_receipt(destination)
            receipt = BackupReceipt(
                generation_id=source.name,
                files=file_inventory(source),
                source=str(source),
                destination=str(destination),
                trainer_progress_sha256=_trainer_progress_digest(source),
            )
            write_json(receipt_path, receipt.identity())
            return receipt
        raise RuntimeGateError("backup destination already contains different content", {"path": str(destination)})
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    expected = file_inventory(source)
    actual = file_inventory(temporary)
    if actual != expected:
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeGateError("backup copy hash mismatch", {"source": str(source)})
    _, fsync_directory, seal_checkpoint_directory = _checkpoint_ops()
    if (temporary / ".complete").exists():
        (temporary / ".complete").unlink()
    if (temporary / ".files.json").exists():
        (temporary / ".files.json").unlink()
    seal_checkpoint_directory(temporary)
    _fsync_tree(temporary, fsync_directory)
    temporary.replace(destination)
    fsync_directory(destination.parent)
    receipt = BackupReceipt(
        generation_id=source.name,
        files=expected,
        source=str(source),
        destination=str(destination),
        trainer_progress_sha256=_trainer_progress_digest(source),
    )
    write_json(destination.with_name(destination.name + ".receipt.json"), receipt.identity())
    return receipt


def verify_receipt(destination: Path, receipt: Mapping[str, object] | None = None) -> BackupReceipt:
    """Reject a corrupt or incomplete trusted backup."""

    if not _generation_is_complete(destination):
        raise RuntimeGateError("backup is incomplete", {"path": str(destination)})
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    if receipt is None:
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeGateError("backup receipt cannot be read", {"path": str(receipt_path)}) from error
    else:
        payload = receipt
    parsed = _receipt_from_payload(payload)
    if parsed.generation_id != destination.name or parsed.destination != str(destination):
        raise RuntimeGateError("backup receipt identity does not match its destination")
    actual = file_inventory(destination)
    if actual != parsed.files:
        raise RuntimeGateError("backup receipt does not match files", {"path": str(destination)})
    expected_progress = _trainer_progress_digest(destination)
    if _is_canonical_generation(destination) and parsed.trainer_progress_sha256 is None:
        raise RuntimeGateError("backup receipt is missing trainer progress")
    if parsed.trainer_progress_sha256 is not None and parsed.trainer_progress_sha256 != expected_progress:
        raise RuntimeGateError("backup receipt does not match trainer progress", {"path": str(destination)})
    return parsed


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
    if (destination / ".complete").exists():
        (destination / ".complete").unlink()
    if (destination / ".files.json").exists():
        (destination / ".files.json").unlink()
    _, _, seal_checkpoint_directory = _checkpoint_ops()
    seal_checkpoint_directory(destination)
    if not _generation_is_complete(destination):
        raise RuntimeGateError("restored generation is incomplete")
    return destination


def ranked_model_cannot_resume(path: Path) -> None:
    """Reject model-only ranked files as training recovery."""

    required = ("pytorch_model.bin",)
    checkpoint_directory_is_complete, _, _ = _checkpoint_ops()
    if checkpoint_directory_is_complete(path, required) and not (path / "optimizer.bin").exists():
        raise RuntimeGateError("model-only ranked files cannot resume training", {"path": str(path)})
