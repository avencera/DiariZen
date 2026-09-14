"""Typed model-only warm start with an explicit task-head reset."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


WARM_START_MODE = "compatible-backbone-reset-powerset-head-v1"
RESET_PARAMETERS = frozenset({"classifier.weight", "classifier.bias"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PowersetWarmStartPolicy:
    """A checkpoint identity and the only task-head shape change allowed."""

    checkpoint: Path
    checkpoint_sha256: str
    receipt: Path
    source_powerset_classes: int
    target_powerset_classes: int

    def __post_init__(self) -> None:
        if len(self.checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.checkpoint_sha256
        ):
            raise ValueError("warm-start checkpoint SHA-256 is invalid")
        if self.source_powerset_classes <= 0 or self.target_powerset_classes <= 0:
            raise ValueError("warm-start powerset class counts must be positive")
        if self.source_powerset_classes == self.target_powerset_classes:
            raise ValueError("warm-start task-head reset requires different source and target geometries")

    @classmethod
    def from_config(cls, payload: object) -> PowersetWarmStartPolicy:
        """Parse a strict warm-start policy from a TOML mapping."""

        if not isinstance(payload, Mapping):
            raise ValueError("warm-start policy must be an object")
        expected = {
            "finetune",
            "mode",
            "ckpt_dir",
            "checkpoint_sha256",
            "receipt",
            "source_powerset_classes",
            "target_powerset_classes",
            "reinitialized_parameters",
        }
        if set(payload) != expected or payload.get("finetune") is not True or payload.get("mode") != WARM_START_MODE:
            raise ValueError("warm-start policy fields or mode are invalid")
        reset = payload.get("reinitialized_parameters")
        if not isinstance(reset, list) or set(reset) != RESET_PARAMETERS or len(reset) != len(RESET_PARAMETERS):
            raise ValueError("warm-start policy must reset only the powerset classifier")
        checkpoint = payload.get("ckpt_dir")
        digest = payload.get("checkpoint_sha256")
        receipt = payload.get("receipt")
        source_classes = payload.get("source_powerset_classes")
        target_classes = payload.get("target_powerset_classes")
        if not isinstance(checkpoint, str) or not isinstance(digest, str) or not isinstance(receipt, str):
            raise ValueError("warm-start paths and digest are invalid")
        if (
            isinstance(source_classes, bool)
            or not isinstance(source_classes, int)
            or isinstance(target_classes, bool)
            or not isinstance(target_classes, int)
        ):
            raise ValueError("warm-start powerset class counts are invalid")
        return cls(Path(checkpoint), digest, Path(receipt), source_classes, target_classes)


def _load_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("warm-start checkpoint is not a state dictionary")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in payload.items()):
        raise ValueError("warm-start checkpoint contains non-tensor state")
    return payload


def warm_start_powerset_model(model: torch.nn.Module, policy: PowersetWarmStartPolicy) -> dict[str, object]:
    """Load every compatible parameter and reset only the changed powerset head."""

    checkpoint = policy.checkpoint.expanduser().resolve(strict=True)
    if _sha256_file(checkpoint) != policy.checkpoint_sha256:
        raise ValueError("warm-start checkpoint differs from its pinned SHA-256")
    source = _load_checkpoint(checkpoint)
    target = model.state_dict()
    if set(source) != set(target):
        raise ValueError("warm-start checkpoint parameter names differ from the target model")
    mismatches = {name for name in target if tuple(source[name].shape) != tuple(target[name].shape)}
    if mismatches != RESET_PARAMETERS:
        raise ValueError(f"warm-start shape mismatches are not the exact powerset head: {sorted(mismatches)}")
    if tuple(source["classifier.weight"].shape)[:1] != (policy.source_powerset_classes,):
        raise ValueError("warm-start source classifier geometry differs from policy")
    if tuple(target["classifier.weight"].shape)[:1] != (policy.target_powerset_classes,):
        raise ValueError("warm-start target classifier geometry differs from policy")
    compatible = {name: value for name, value in source.items() if name not in RESET_PARAMETERS}
    incompatible = model.load_state_dict(compatible, strict=False)
    if set(incompatible.missing_keys) != RESET_PARAMETERS or incompatible.unexpected_keys:
        raise ValueError("warm-start load result differs from the declared task-head reset")
    for name, value in compatible.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), value.detach().cpu()):
            raise ValueError(f"warm-start compatible parameter did not load: {name}")

    receipt = {
        "schema": "speakrs-powerset-warm-start-receipt-v1",
        "mode": WARM_START_MODE,
        "checkpoint": {"path": checkpoint.as_posix(), "sha256": policy.checkpoint_sha256},
        "source_powerset_classes": policy.source_powerset_classes,
        "target_powerset_classes": policy.target_powerset_classes,
        "loaded_parameters": len(compatible),
        "loaded_parameter_names_sha256": hashlib.sha256(
            ("\n".join(sorted(compatible)) + "\n").encode("utf-8")
        ).hexdigest(),
        "reinitialized_parameters": [
            {
                "name": name,
                "source_shape": list(source[name].shape),
                "target_shape": list(target[name].shape),
                "target_initialization_sha256": _tensor_sha256(target[name]),
            }
            for name in sorted(RESET_PARAMETERS)
        ],
        "strict_resume": False,
        "optimizer_state_loaded": False,
    }
    destination = policy.receipt.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return receipt
