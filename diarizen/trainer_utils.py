# Licensed under the MIT license.
# Copy from https://github.com/haoxiangsnr/spiking-fullsubnet/blob/main/audiozen/trainer_utils.py
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate.utils import set_seed


CHECKPOINT_COMPLETE_MARKER = ".complete"
CHECKPOINT_FILE_MANIFEST = ".files.json"
CHECKPOINT_FILE_MANIFEST_VERSION = 2
LEGACY_CHECKPOINT_FILE_MANIFEST_VERSION = 1
CHECKPOINT_PROGRESS_FILE = "progress.json"
_CHECKPOINT_MANIFEST_FIELDS = frozenset({"version", "files"})
_CHECKPOINT_MANIFEST_FIELDS_WITH_PROGRESS = frozenset({"version", "files", "trainer_progress_sha256"})
_SHA256_LENGTH = 64


def fsync_directory(path: Path) -> None:
    """Synchronize directory entries when the host file system supports it."""

    try:
        directory_fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return

    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys before a JSON document is trusted."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object contains duplicate fields")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON numbers from canonical documents."""

    raise ValueError(f"JSON constant is not supported: {value}")


def _canonical_json_bytes(value: object) -> bytes:
    """Encode one JSON value in the canonical representation used for digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_trainer_progress_digest(progress: Mapping[str, object] | Path) -> str:
    """Return the SHA-256 of canonical trainer-progress JSON."""

    if isinstance(progress, Path):
        try:
            parsed = json.loads(
                progress.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("trainer progress is not valid JSON") from error
        if not isinstance(parsed, Mapping):
            raise ValueError("trainer progress must be a JSON object")
        progress = parsed

    if not isinstance(progress, Mapping):
        raise TypeError("trainer progress must be a mapping or path")
    try:
        encoded = _canonical_json_bytes(dict(progress))
    except (TypeError, ValueError) as error:
        raise ValueError("trainer progress cannot be canonicalized") from error
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(checkpoint_dir: Path, path: Path) -> str:
    """Return a payload path only when it cannot escape its checkpoint."""

    if path.is_symlink():
        raise ValueError("checkpoint payload cannot be a symbolic link")
    relative = path.relative_to(checkpoint_dir).as_posix()
    parts = relative.split("/")
    if not relative or any(part in {"", ".", ".."} for part in parts) or relative.startswith("/"):
        raise ValueError("checkpoint payload path is unsafe")
    return relative


def _checkpoint_payload_files(checkpoint_dir: Path) -> tuple[Path, ...]:
    """Return checkpoint payload files in stable relative-path order."""

    excluded = {
        CHECKPOINT_COMPLETE_MARKER,
        f"{CHECKPOINT_COMPLETE_MARKER}.partial",
        CHECKPOINT_FILE_MANIFEST,
        f"{CHECKPOINT_FILE_MANIFEST}.partial",
    }
    payload_files: list[Path] = []
    for path in checkpoint_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = _safe_relative_path(checkpoint_dir, path)
        if relative not in excluded:
            payload_files.append(path)
    return tuple(sorted(payload_files, key=lambda path: path.relative_to(checkpoint_dir).as_posix()))


def seal_checkpoint_directory(checkpoint_dir: Path, *, require_trainer_progress: bool = False) -> None:
    """Synchronize payloads and publish a hash-and-length manifest and marker."""

    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    marker = checkpoint_dir / CHECKPOINT_COMPLETE_MARKER
    if marker.exists():
        raise RuntimeError(f"Checkpoint is already sealed: {checkpoint_dir}")

    payload_files = _checkpoint_payload_files(checkpoint_dir)
    if not payload_files:
        raise RuntimeError(f"Checkpoint has no payload files: {checkpoint_dir}")

    file_records: dict[str, dict[str, object]] = {}
    for path in payload_files:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        relative = path.relative_to(checkpoint_dir).as_posix()
        file_records[relative] = {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
    payload_directories = {path.parent for path in payload_files}
    for directory in sorted(payload_directories, key=lambda path: len(path.parts), reverse=True):
        fsync_directory(directory)

    manifest = {
        "version": CHECKPOINT_FILE_MANIFEST_VERSION,
        "files": file_records,
    }
    progress_path = checkpoint_dir / CHECKPOINT_PROGRESS_FILE
    progress_digest: str | None = None
    if progress_path.is_file():
        try:
            progress_digest = canonical_trainer_progress_digest(progress_path)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Checkpoint trainer progress is invalid: {checkpoint_dir}") from error
        manifest["trainer_progress_sha256"] = progress_digest
    elif require_trainer_progress:
        raise RuntimeError(f"Checkpoint trainer progress is missing: {checkpoint_dir}")

    manifest_path = checkpoint_dir / CHECKPOINT_FILE_MANIFEST
    temporary_manifest = checkpoint_dir / f"{CHECKPOINT_FILE_MANIFEST}.partial"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_manifest.replace(manifest_path)
    fsync_directory(checkpoint_dir)

    temporary_marker = checkpoint_dir / f"{CHECKPOINT_COMPLETE_MARKER}.partial"
    with temporary_marker.open("w", encoding="utf-8") as handle:
        handle.write("complete\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_marker.replace(marker)
    fsync_directory(checkpoint_dir)


def _valid_digest(value: object) -> bool:
    """Return whether a value is a lowercase SHA-256 digest."""

    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _safe_manifest_path(value: object) -> bool:
    """Return whether a manifest path is a safe relative POSIX path."""

    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        return False
    return all(part not in {"", ".", ".."} for part in value.split("/"))


def _parse_manifest(manifest: object) -> tuple[int, dict[str, object], str | None] | None:
    """Parse a checkpoint manifest, including the isolated legacy read path."""

    if not isinstance(manifest, dict):
        return None
    version = manifest.get("version")
    if version == LEGACY_CHECKPOINT_FILE_MANIFEST_VERSION:
        if set(manifest) != _CHECKPOINT_MANIFEST_FIELDS:
            return None
        records = manifest.get("files")
        if not isinstance(records, dict) or not records:
            return None
        legacy_records: dict[str, object] = {}
        for relative, size in records.items():
            if not _safe_manifest_path(relative) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
                return None
            legacy_records[relative] = size
        return version, legacy_records, None

    if version != CHECKPOINT_FILE_MANIFEST_VERSION:
        return None
    fields = (
        _CHECKPOINT_MANIFEST_FIELDS_WITH_PROGRESS
        if "trainer_progress_sha256" in manifest
        else _CHECKPOINT_MANIFEST_FIELDS
    )
    if set(manifest) != fields:
        return None
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        return None
    parsed_records: dict[str, object] = {}
    for relative, record in records.items():
        if not _safe_manifest_path(relative) or not isinstance(record, dict) or set(record) != {"sha256", "size"}:
            return None
        digest = record.get("sha256")
        size = record.get("size")
        if not _valid_digest(digest) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            return None
        parsed_records[relative] = {"sha256": digest, "size": size}
    progress_digest = manifest.get("trainer_progress_sha256")
    if "trainer_progress_sha256" in fields and not _valid_digest(progress_digest):
        return None
    return version, parsed_records, progress_digest if isinstance(progress_digest, str) else None


def checkpoint_directory_is_complete(
    checkpoint_dir: Path,
    required_files: tuple[str, ...] = (),
    *,
    require_hashed_manifest: bool = False,
) -> bool:
    """Return whether a checkpoint marker and exact file manifest are valid."""

    marker = checkpoint_dir / CHECKPOINT_COMPLETE_MARKER
    if not checkpoint_dir.is_dir() or marker.is_symlink() or not marker.is_file():
        return False
    if (checkpoint_dir / f"{CHECKPOINT_COMPLETE_MARKER}.partial").exists() or (
        checkpoint_dir / f"{CHECKPOINT_FILE_MANIFEST}.partial"
    ).exists():
        return False
    try:
        manifest = json.loads(
            (checkpoint_dir / CHECKPOINT_FILE_MANIFEST).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        return False
    parsed_manifest = _parse_manifest(manifest)
    if parsed_manifest is None:
        return False
    version, records, progress_digest = parsed_manifest
    if require_hashed_manifest and version != CHECKPOINT_FILE_MANIFEST_VERSION:
        return False

    try:
        actual_files = _checkpoint_payload_files(checkpoint_dir)
    except (OSError, ValueError):
        return False
    if version == LEGACY_CHECKPOINT_FILE_MANIFEST_VERSION:
        actual_records: dict[str, object] = {
            path.relative_to(checkpoint_dir).as_posix(): path.stat().st_size for path in actual_files
        }
    else:
        actual_records = {
            path.relative_to(checkpoint_dir).as_posix(): {
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in actual_files
        }
    if records != actual_records:
        return False
    if not all(_safe_manifest_path(filename) and filename in records for filename in required_files):
        return False
    if version == CHECKPOINT_FILE_MANIFEST_VERSION and CHECKPOINT_PROGRESS_FILE in records:
        if progress_digest is None:
            return False
        try:
            actual_progress_digest = canonical_trainer_progress_digest(checkpoint_dir / CHECKPOINT_PROGRESS_FILE)
        except (OSError, TypeError, ValueError):
            return False
        if actual_progress_digest != progress_digest:
            return False
    return True


class AutoClipGradHistory(list[float]):
    """Store the bounded gradient-norm history used by automatic clipping."""

    def __init__(self, max_size: int) -> None:
        max_size = int(max_size)
        if max_size < 1:
            raise ValueError("Automatic clipping history size must be at least one")

        super().__init__()
        self.max_size = max_size

    def append(self, value: float) -> None:
        """Append a gradient norm and discard the oldest value when full."""
        super().append(float(value))
        if len(self) > self.max_size:
            del self[: len(self) - self.max_size]

    def extend(self, values) -> None:
        """Append multiple gradient norms while preserving the size bound."""
        for value in values:
            self.append(value)

    def state_dict(self) -> dict[str, list[float]]:
        """Return the history in a format accepted by Accelerate checkpoints."""
        return {"values": list(self)}

    def load_state_dict(self, state_dict: dict[str, list[float]]) -> None:
        """Restore the history while enforcing the configured maximum size."""
        values = state_dict["values"]
        if not isinstance(values, (list, tuple)):
            raise TypeError("Automatic clipping history values must be a list")

        self.clear()
        for value in values[-self.max_size :]:
            self.append(value)


def raise_for_non_finite_loss(loss: torch.Tensor, optimizers, batch_idx: int) -> None:
    """Fail before backward when a loss is NaN or infinite and clear old gradients."""
    if torch.isfinite(loss).all().item():
        return

    for optimizer in optimizers:
        optimizer.zero_grad()

    loss_value = loss.detach().float().cpu().tolist()
    raise RuntimeError(f"Non-finite training loss at batch {batch_idx}: {loss_value}")


def scalar_to_float(value) -> float:
    """Convert a scalar metric value to a host Python float."""
    if torch.is_tensor(value):
        return value.detach().float().cpu().item()
    return float(value)


def reject_fp16_dual_optimizer(accelerator) -> None:
    """Reject FP16 for trainers whose two optimizers share one scaler lifecycle."""
    if getattr(accelerator, "mixed_precision", None) == "fp16":
        raise RuntimeError(
            "Dual-optimizer training does not support mixed_precision='fp16'; use 'bf16' or 'no' instead"
        )


def seed_worker(_):
    """Helper function to set worker seed during Dataloader initialization.

    In recent check-ins, we may have no longer needed this function because PyTorch has already set the worker seed
    for numpy and random. But there is no adverse effect to keeping this function, since the initial_seed is
    inner_seed + worker_ids.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


class TrainerState:
    """Checkpointed progress and terminal state for one training run."""

    def __init__(self, save_max_score) -> None:
        self.epochs_trained = 0
        self.steps_trained = 0
        self.training_complete = False

        self.patience = 0

        self.best_score = -np.inf if save_max_score else np.inf
        self.best_score_epoch = 0
        self.updates_trained = 0
        self.microbatches_in_epoch = 0
        self.epoch_data_rng_state = None
        self.cycles_trained = 0
        self.scoring_phase = "idle"
        self.stop_reason = None
        self.recipe_state = {}

    def load_state_dict(self, state_dict: dict) -> None:
        self.epochs_trained = state_dict["epochs_trained"]
        self.steps_trained = state_dict["steps_trained"]
        self.training_complete = bool(state_dict.get("training_complete", False))

        self.best_score = state_dict["best_score"]
        self.best_score_epoch = state_dict["best_score_epoch"]

        self.patience = state_dict["patience"]
        self.updates_trained = int(state_dict.get("updates_trained", 0))
        self.microbatches_in_epoch = int(state_dict.get("microbatches_in_epoch", 0))
        self.epoch_data_rng_state = state_dict.get("epoch_data_rng_state")
        self.cycles_trained = int(state_dict.get("cycles_trained", 0))
        self.scoring_phase = str(state_dict.get("scoring_phase", "idle"))
        self.stop_reason = state_dict.get("stop_reason")
        self.recipe_state = dict(state_dict.get("recipe_state") or {})

    def state_dict(self) -> dict:
        return {
            "epochs_trained": self.epochs_trained,
            "steps_trained": self.steps_trained,
            "training_complete": self.training_complete,
            "patience": self.patience,
            "best_score": self.best_score,
            "best_score_epoch": self.best_score_epoch,
            "updates_trained": self.updates_trained,
            "microbatches_in_epoch": self.microbatches_in_epoch,
            "epoch_data_rng_state": self.epoch_data_rng_state,
            "cycles_trained": self.cycles_trained,
            "scoring_phase": self.scoring_phase,
            "stop_reason": self.stop_reason,
            "recipe_state": dict(self.recipe_state),
        }
