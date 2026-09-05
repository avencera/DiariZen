# Licensed under the MIT license.
# Copy from https://github.com/haoxiangsnr/spiking-fullsubnet/blob/main/audiozen/trainer_utils.py
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)

import json
import os
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import set_seed


CHECKPOINT_COMPLETE_MARKER = ".complete"
CHECKPOINT_FILE_MANIFEST = ".files.json"
CHECKPOINT_FILE_MANIFEST_VERSION = 1


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


def _checkpoint_payload_files(checkpoint_dir: Path) -> tuple[Path, ...]:
    """Return checkpoint payload files in stable relative-path order."""

    excluded = {CHECKPOINT_COMPLETE_MARKER, CHECKPOINT_FILE_MANIFEST, f"{CHECKPOINT_FILE_MANIFEST}.partial"}
    return tuple(
        sorted(
            (
                path
                for path in checkpoint_dir.rglob("*")
                if path.is_file() and path.relative_to(checkpoint_dir).as_posix() not in excluded
            ),
            key=lambda path: path.relative_to(checkpoint_dir).as_posix(),
        )
    )


def seal_checkpoint_directory(checkpoint_dir: Path) -> None:
    """Synchronize checkpoint payloads and publish their size manifest and marker."""

    payload_files = _checkpoint_payload_files(checkpoint_dir)
    if not payload_files:
        raise RuntimeError(f"Checkpoint has no payload files: {checkpoint_dir}")

    file_records = {}
    for path in payload_files:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        file_records[path.relative_to(checkpoint_dir).as_posix()] = path.stat().st_size
    payload_directories = {path.parent for path in payload_files}
    for directory in sorted(payload_directories, key=lambda path: len(path.parts), reverse=True):
        fsync_directory(directory)

    manifest = {
        "version": CHECKPOINT_FILE_MANIFEST_VERSION,
        "files": file_records,
    }
    manifest_path = checkpoint_dir / CHECKPOINT_FILE_MANIFEST
    temporary_manifest = checkpoint_dir / f"{CHECKPOINT_FILE_MANIFEST}.partial"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_manifest.replace(manifest_path)
    fsync_directory(checkpoint_dir)

    marker = checkpoint_dir / CHECKPOINT_COMPLETE_MARKER
    temporary_marker = checkpoint_dir / f"{CHECKPOINT_COMPLETE_MARKER}.partial"
    with temporary_marker.open("w", encoding="utf-8") as handle:
        handle.write("complete\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_marker.replace(marker)
    fsync_directory(checkpoint_dir)


def checkpoint_directory_is_complete(checkpoint_dir: Path, required_files: tuple[str, ...] = ()) -> bool:
    """Return whether a checkpoint marker and exact payload-size manifest are valid."""

    if not checkpoint_dir.is_dir() or not (checkpoint_dir / CHECKPOINT_COMPLETE_MARKER).is_file():
        return False
    try:
        manifest = json.loads((checkpoint_dir / CHECKPOINT_FILE_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict) or manifest.get("version") != CHECKPOINT_FILE_MANIFEST_VERSION:
        return False
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        return False

    actual_files = _checkpoint_payload_files(checkpoint_dir)
    actual_records = {path.relative_to(checkpoint_dir).as_posix(): path.stat().st_size for path in actual_files}
    if records != actual_records:
        return False
    return all(filename in records for filename in required_files)


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

    def load_state_dict(self, state_dict: dict) -> None:
        self.epochs_trained = state_dict["epochs_trained"]
        self.steps_trained = state_dict["steps_trained"]
        self.training_complete = bool(state_dict.get("training_complete", False))

        self.best_score = state_dict["best_score"]
        self.best_score_epoch = state_dict["best_score_epoch"]

        self.patience = state_dict["patience"]

    def state_dict(self) -> dict:
        return {
            "epochs_trained": self.epochs_trained,
            "steps_trained": self.steps_trained,
            "training_complete": self.training_complete,
            "patience": self.patience,
            "best_score": self.best_score,
            "best_score_epoch": self.best_score_epoch,
        }
