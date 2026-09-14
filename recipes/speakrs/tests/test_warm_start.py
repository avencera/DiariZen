"""Tests for the explicit powerset-head warm-start contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from diarizen.warm_start import PowersetWarmStartPolicy, warm_start_powerset_model


class TinyModel(torch.nn.Module):
    """Provide backbone and classifier parameters with production key names."""

    def __init__(self, classes: int, hidden: int = 3) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(hidden, hidden)
        self.classifier = torch.nn.Linear(hidden, classes)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy(checkpoint: Path, receipt: Path) -> PowersetWarmStartPolicy:
    return PowersetWarmStartPolicy(checkpoint, _sha256(checkpoint), receipt, 11, 16)


def test_warm_start_loads_backbone_and_resets_only_classifier(tmp_path) -> None:
    torch.manual_seed(1)
    source = TinyModel(11)
    checkpoint = tmp_path / "pytorch_model.bin"
    torch.save(source.state_dict(), checkpoint)
    torch.manual_seed(2)
    target = TinyModel(16)
    initial_weight = target.classifier.weight.detach().clone()
    initial_bias = target.classifier.bias.detach().clone()
    receipt_path = tmp_path / "receipt.json"

    receipt = warm_start_powerset_model(target, _policy(checkpoint, receipt_path))

    assert torch.equal(target.backbone.weight, source.backbone.weight)
    assert torch.equal(target.backbone.bias, source.backbone.bias)
    assert torch.equal(target.classifier.weight, initial_weight)
    assert torch.equal(target.classifier.bias, initial_bias)
    assert receipt["source_powerset_classes"] == 11
    assert receipt["target_powerset_classes"] == 16
    assert [row["name"] for row in receipt["reinitialized_parameters"]] == [
        "classifier.bias",
        "classifier.weight",
    ]
    assert receipt["strict_resume"] is False
    assert receipt["optimizer_state_loaded"] is False
    assert receipt_path.is_file()


def test_warm_start_rejects_a_non_head_shape_change(tmp_path) -> None:
    checkpoint = tmp_path / "pytorch_model.bin"
    torch.save(TinyModel(11, hidden=4).state_dict(), checkpoint)

    with pytest.raises(ValueError, match="not the exact powerset head"):
        warm_start_powerset_model(TinyModel(16, hidden=3), _policy(checkpoint, tmp_path / "receipt.json"))


def test_warm_start_rejects_checkpoint_hash_change(tmp_path) -> None:
    checkpoint = tmp_path / "pytorch_model.bin"
    torch.save(TinyModel(11).state_dict(), checkpoint)
    policy = PowersetWarmStartPolicy(checkpoint, "0" * 64, tmp_path / "receipt.json", 11, 16)

    with pytest.raises(ValueError, match="pinned SHA-256"):
        warm_start_powerset_model(TinyModel(16), policy)
