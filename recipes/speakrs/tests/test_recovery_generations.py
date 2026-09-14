"""Tests for immutable resumable recovery generations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from diarizen.trainer_utils import checkpoint_directory_is_complete, seal_checkpoint_directory
from recipes.speakrs.large.errors import RuntimeGateError
from recipes.speakrs.large.recovery import (
    RecoveryGeneration,
    complete_generations,
    copy_generation,
    generation_name,
    next_generation,
    publish_generation,
    verify_receipt,
)


def _stage(root: Path, updates: int, sequence: int, model: bytes = b"model") -> tuple[Path, Path]:
    destination = root / generation_name(updates, sequence)
    temporary = root / f".{destination.name}.partial"
    temporary.mkdir()
    (temporary / "pytorch_model.bin").write_bytes(model)
    (temporary / "progress.json").write_text(
        json.dumps(
            {
                "schema": "diarizen-checkpoint-progress-v1",
                "updates_trained": updates,
                "training_complete": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return temporary, destination


def test_generation_identity_orders_same_update_sequences() -> None:
    first = RecoveryGeneration(250, 1)
    second = RecoveryGeneration(250, 2)
    later_update = RecoveryGeneration(500, 1)

    assert first < second < later_update
    assert second.name == generation_name(250, 2)


def test_generation_sequence_must_be_one_based() -> None:
    with pytest.raises(ValueError, match="generation sequence"):
        RecoveryGeneration(250, 0)


def test_next_generation_uses_a_strictly_greater_sequence(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / generation_name(250, 1)).mkdir()
    (root / generation_name(250, 4)).mkdir()

    assert next_generation(root, 250) == RecoveryGeneration(250, 5)


def test_next_generation_starts_at_one_and_ignores_legacy_sequence(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "update_00000250").mkdir()

    assert next_generation(root, 250) == RecoveryGeneration(250, 1)


def test_complete_generations_sort_by_update_then_sequence(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    for updates, sequence in ((500, 1), (250, 2), (250, 1)):
        temporary, destination = _stage(root, updates, sequence)
        publish_generation(temporary, destination)

    assert [path.name for path in complete_generations(root)] == [
        generation_name(250, 1),
        generation_name(250, 2),
        generation_name(500, 1),
    ]


def test_manifest_binds_hash_length_and_trainer_progress(tmp_path: Path) -> None:
    generation = tmp_path / generation_name(250, 1)
    generation.mkdir()
    model = generation / "pytorch_model.bin"
    model.write_bytes(b"model")
    (generation / "progress.json").write_text('{"updates_trained": 250}\n', encoding="utf-8")
    seal_checkpoint_directory(generation, require_trainer_progress=True)

    manifest = json.loads((generation / ".files.json").read_text(encoding="utf-8"))
    record = manifest["files"]["pytorch_model.bin"]
    assert record == {"sha256": hashlib.sha256(b"model").hexdigest(), "size": 5}
    assert isinstance(manifest["trainer_progress_sha256"], str)
    assert checkpoint_directory_is_complete(generation, ("progress.json",), require_hashed_manifest=True)

    model.write_bytes(b"other")
    assert not checkpoint_directory_is_complete(generation, require_hashed_manifest=True)

    model.write_bytes(b"model")
    (generation / "progress.json").write_text('{"updates_trained": 251}\n', encoding="utf-8")
    assert not checkpoint_directory_is_complete(generation, require_hashed_manifest=True)


def test_update_only_new_seal_is_v2_and_old_v1_fixture_is_readable(tmp_path: Path) -> None:
    new_generation = tmp_path / "update_00000250"
    new_generation.mkdir()
    payload = new_generation / "pytorch_model.bin"
    payload.write_bytes(b"model")
    seal_checkpoint_directory(new_generation)

    manifest = json.loads((new_generation / ".files.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    assert manifest["files"][payload.name] == {
        "sha256": hashlib.sha256(b"model").hexdigest(),
        "size": len(b"model"),
    }
    assert checkpoint_directory_is_complete(new_generation, require_hashed_manifest=True)

    old_generation = tmp_path / "update_00000100"
    old_generation.mkdir()
    old_payload = old_generation / "pytorch_model.bin"
    old_payload.write_bytes(b"legacy")
    (old_generation / ".files.json").write_text(
        json.dumps({"version": 1, "files": {old_payload.name: old_payload.stat().st_size}}) + "\n",
        encoding="utf-8",
    )
    (old_generation / ".complete").write_text("complete\n", encoding="utf-8")

    assert checkpoint_directory_is_complete(old_generation, (old_payload.name,))
    assert not checkpoint_directory_is_complete(old_generation, require_hashed_manifest=True)


def test_manifest_rejects_unknown_fields_and_path_traversal(tmp_path: Path) -> None:
    generation = tmp_path / generation_name(250, 1)
    generation.mkdir()
    (generation / "payload.bin").write_bytes(b"payload")
    seal_checkpoint_directory(generation)
    manifest_path = generation / ".files.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not checkpoint_directory_is_complete(generation, require_hashed_manifest=True)

    manifest = json.loads((generation / ".files.json").read_text(encoding="utf-8"))
    manifest["files"]["../outside"] = manifest["files"].pop("payload.bin")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not checkpoint_directory_is_complete(generation, require_hashed_manifest=True)


def test_publication_replay_is_idempotent_and_conflicts_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    temporary, destination = _stage(root, 250, 1)
    publish_generation(temporary, destination)
    original_progress = (destination / "progress.json").read_bytes()

    replay, replay_destination = _stage(root, 250, 1)
    (replay / "progress.json").write_bytes(original_progress)
    assert publish_generation(replay, replay_destination) == destination
    assert not replay.exists()
    assert (destination / "progress.json").read_bytes() == original_progress

    conflict, conflict_destination = _stage(root, 250, 1, model=b"different")
    with pytest.raises(RuntimeGateError, match="conflicts"):
        publish_generation(conflict, conflict_destination)
    assert (destination / "pytorch_model.bin").read_bytes() == b"model"


def test_receipt_verification_rejects_payload_mutation_and_unsafe_paths(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    trusted = tmp_path / "trusted"
    worker.mkdir()
    temporary, source = _stage(worker, 250, 1)
    publish_generation(temporary, source)
    receipt = copy_generation(source, trusted / source.name)

    (trusted / source.name / "pytorch_model.bin").write_bytes(b"other")
    with pytest.raises(RuntimeGateError, match="incomplete"):
        verify_receipt(trusted / source.name)

    unsafe_receipt = receipt.identity()
    unsafe_receipt["files"] = {"../outside": {"sha256": "0" * 64, "size": 1}}
    with pytest.raises(RuntimeGateError, match="unsafe"):
        verify_receipt(source, unsafe_receipt)


def test_completion_marker_is_published_last_and_destination_is_atomic(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    temporary, destination = _stage(root, 250, 1)
    events: list[tuple[str, bool]] = []
    original_replace = Path.replace

    def track_replace(path: Path, target: Path) -> Path:
        if path.name in {".files.json.partial", ".complete.partial"} or target == destination:
            events.append((path.name, target.exists()))
        return original_replace(path, target)

    with patch.object(Path, "replace", track_replace):
        publish_generation(temporary, destination)

    assert [name for name, _ in events] == [".files.json.partial", ".complete.partial", temporary.name]
    assert events[-1] == (temporary.name, False)
    assert (destination / ".complete").is_file()
