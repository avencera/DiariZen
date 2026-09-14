"""Focused tests for immutable model-snapshot publication and recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from diarizen.trainer_utils import canonical_trainer_progress_digest, seal_checkpoint_directory
from recipes.speakrs.large.contracts import ObjectStoreDestination
from recipes.speakrs.large.recovery import RecoveryGeneration, generation_name, publish_generation
from recipes.speakrs.large.storage import (
    ImmutableObjectConflictError,
    ImmutableObjectLimits,
    ImmutableObjectOutcome,
    ImmutableObjectUnavailableError,
)
from recipes.speakrs.large.validation.contracts import ArtifactLocation, Sha256Digest
from recipes.speakrs.large.validation.publication import (
    PublicationConflictError,
    PublicationError,
    PublicationRecoveryError,
    PublicationUnavailableError,
    SnapshotPublication,
    derive_recovery_generation_id,
)
from recipes.speakrs.large.validation.slot_plan import build_validation_campaign


_PUBLICATION_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _InjectedCrash(RuntimeError):
    pass


@dataclass
class _Object:
    payload: bytes
    digest: str


class _FakeImmutableBackend:
    """Small conditional object store fake with no network surface."""

    def __init__(self) -> None:
        self.objects: dict[str, _Object] = {}
        self.calls: list[str] = []
        self.immutable_object_limits = ImmutableObjectLimits()
        self.fail_next = False

    def put_immutable_file_if_absent(self, upload) -> ImmutableObjectOutcome:
        self.calls.append(upload.key)
        if self.fail_next:
            self.fail_next = False
            raise ImmutableObjectUnavailableError("temporary backend failure")
        payload = upload.path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != upload.expected_size or digest != upload.expected_sha256:
            raise AssertionError("the publication owner supplied an invalid upload")
        current = self.objects.get(upload.key)
        if current is not None:
            if current.digest != digest or len(current.payload) != len(payload):
                raise ImmutableObjectConflictError("different immutable object")
            return ImmutableObjectOutcome.ALREADY_PRESENT
        self.objects[upload.key] = _Object(payload, digest)
        return ImmutableObjectOutcome.CREATED


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _campaign(prefix: str = "models"):
    return build_validation_campaign(
        training_launch_id="launch",
        max_updates=4,
        updates_per_complete_epoch=2,
        artifact_prefix=prefix,
        dev_bundle_digest=_digest("a"),
        trainer_configuration_digest=_digest("b"),
        evaluator_image_identity="image",
        evaluator_implementation_digest=_digest("c"),
    )


def _generation(tmp_path: Path, updates: int = 2, sequence: int = 1, model: bytes = b"model") -> Path:
    root = tmp_path / "checkpoints"
    root.mkdir(parents=True)
    temporary = root / f".{generation_name(updates, sequence)}.partial"
    destination = root / generation_name(updates, sequence)
    temporary.mkdir()
    (temporary / "pytorch_model.bin").write_bytes(model)
    (temporary / "optimizer.bin").write_bytes(b"optimizer")
    (temporary / "progress.json").write_text(
        json.dumps({"updates_trained": updates, "training_complete": False}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    publish_generation(temporary, destination)
    return destination


def _owner(
    tmp_path: Path,
    backend: _FakeImmutableBackend,
    *,
    phase: str | None = None,
) -> SnapshotPublication:
    destination = ObjectStoreDestination(
        provider="r2",
        endpoint="object-store",
        bucket="private",
        prefix="models",
        credential_reference="reference",
    )

    def phase_hook(current: str) -> None:
        if current == phase:
            raise _InjectedCrash(current)

    return SnapshotPublication(
        destination,
        backend,
        tmp_path / "trusted-publication",
        clock=lambda: _PUBLICATION_TIME,
        _phase_hook=phase_hook if phase is not None else None,
    )


def _publish(owner: SnapshotPublication, campaign, generation: Path, **kwargs):
    slot = campaign.slots[1]
    progress = Sha256Digest(canonical_trainer_progress_digest(generation / "progress.json"))
    return owner.publish(campaign, slot, generation, progress, **kwargs)


def test_publication_uploads_model_before_manifest_and_only_model_file(tmp_path: Path) -> None:
    backend = _FakeImmutableBackend()
    campaign = _campaign()
    generation = _generation(tmp_path)

    snapshot = _publish(_owner(tmp_path, backend), campaign, generation)

    slot = campaign.slots[1]
    assert backend.calls == [slot.model_location.value, slot.manifest_location.value]
    assert set(backend.objects) == {slot.model_location.value, slot.manifest_location.value}
    assert snapshot.recovery_generation_id == derive_recovery_generation_id(RecoveryGeneration(2, 1))


@pytest.mark.parametrize("point", ("prepared", "model", "manifest"))
def test_every_injected_crash_recovers_without_repeating_training(tmp_path: Path, point: str) -> None:
    backend = _FakeImmutableBackend()
    campaign = _campaign()
    generation = _generation(tmp_path)
    owner = _owner(tmp_path, backend, phase=point)

    with pytest.raises(_InjectedCrash):
        _publish(owner, campaign, generation)
    if point in {"prepared", "model"}:
        assert campaign.slots[1].manifest_location.value not in backend.objects

    snapshot = _publish(_owner(tmp_path, backend), campaign, generation)
    assert snapshot == _publish(_owner(tmp_path, backend), campaign, generation)
    assert (owner.transaction_root / f"{campaign.slots[1].slot_id.value}.receipt.json").is_file()
    expected_model_calls = 1 if point == "prepared" else 2
    assert backend.calls.count(campaign.slots[1].model_location.value) == expected_model_calls


def test_changed_model_after_prepare_is_a_conflict(tmp_path: Path) -> None:
    backend = _FakeImmutableBackend()
    campaign = _campaign()
    generation = _generation(tmp_path)
    owner = _owner(tmp_path, backend)

    with pytest.raises(_InjectedCrash):
        _publish(_owner(tmp_path, backend, phase="prepared"), campaign, generation)
    (generation / "pytorch_model.bin").write_bytes(b"changed")

    with pytest.raises(PublicationConflictError):
        _publish(owner, campaign, generation)
    assert backend.calls == []


def test_progress_and_update_slot_mismatches_are_rejected_before_prepare(tmp_path: Path) -> None:
    backend = _FakeImmutableBackend()
    campaign = _campaign()
    generation = _generation(tmp_path)
    owner = _owner(tmp_path, backend)
    slot = campaign.slots[1]

    with pytest.raises(PublicationConflictError):
        owner.publish(campaign, slot, generation, _digest("d"))
    assert not owner.transaction_root.exists()

    with pytest.raises(PublicationConflictError):
        owner.publish(campaign, campaign.slots[0], generation, _digest("d"))

    changed_slot = replace(campaign.slots[1], model_location=ArtifactLocation("models/changed-model.bin"))
    with pytest.raises(PublicationConflictError):
        owner.publish(campaign, changed_slot, generation, _digest("d"))

    other_generation = _generation(tmp_path / "other", updates=4)
    with pytest.raises(PublicationConflictError):
        owner.publish(campaign, slot, other_generation, _digest("d"))


def test_non_one_based_generation_is_not_publishable(tmp_path: Path) -> None:
    backend = _FakeImmutableBackend()
    campaign = _campaign()
    generation = tmp_path / "update_00000002_generation_00000000"
    generation.mkdir(parents=True)
    (generation / "pytorch_model.bin").write_bytes(b"model")
    (generation / "progress.json").write_text('{"updates_trained": 2}\n', encoding="utf-8")
    seal_checkpoint_directory(generation, require_trainer_progress=True)
    owner = _owner(tmp_path, backend)
    progress = Sha256Digest(canonical_trainer_progress_digest(generation / "progress.json"))

    with pytest.raises(PublicationRecoveryError):
        owner.publish(campaign, campaign.slots[1], generation, progress)


def test_identical_replay_is_idempotent_and_different_replay_conflicts(tmp_path: Path) -> None:
    backend = _FakeImmutableBackend()
    campaign = _campaign()
    generation = _generation(tmp_path)
    owner = _owner(tmp_path, backend)

    first = _publish(owner, campaign, generation)
    assert _publish(owner, campaign, generation) == first
    with pytest.raises(PublicationConflictError):
        owner.publish(campaign, campaign.slots[1], generation, _digest("d"))


def test_model_conflict_stops_before_manifest(tmp_path: Path) -> None:
    backend = _FakeImmutableBackend()
    campaign = _campaign()
    generation = _generation(tmp_path)
    slot = campaign.slots[1]
    backend.objects[slot.model_location.value] = _Object(b"different", hashlib.sha256(b"different").hexdigest())
    owner = _owner(tmp_path, backend)

    with pytest.raises(PublicationConflictError):
        _publish(owner, campaign, generation)
    assert slot.manifest_location.value not in backend.objects


def test_transient_backend_failure_keeps_prepared_transaction_retryable(tmp_path: Path) -> None:
    backend = _FakeImmutableBackend()
    backend.fail_next = True
    campaign = _campaign()
    generation = _generation(tmp_path)
    owner = _owner(tmp_path, backend)
    slot = campaign.slots[1]

    with pytest.raises(PublicationUnavailableError):
        _publish(owner, campaign, generation)

    transaction_path = owner.transaction_root / f"{slot.slot_id.value}.publication.json"
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    assert transaction["state"] == "prepared"
    assert slot.model_location.value not in backend.objects
    assert slot.manifest_location.value not in backend.objects

    snapshot = _publish(_owner(tmp_path, backend), campaign, generation)
    assert snapshot.slot_id == slot.slot_id
    assert slot.manifest_location.value in backend.objects


def test_manifest_conflict_leaves_transaction_prepared_and_never_replaces_it(tmp_path: Path) -> None:
    backend = _FakeImmutableBackend()
    campaign = _campaign()
    generation = _generation(tmp_path)
    slot = campaign.slots[1]
    backend.objects[slot.manifest_location.value] = _Object(b"wrong", hashlib.sha256(b"wrong").hexdigest())
    owner = _owner(tmp_path, backend)

    with pytest.raises(PublicationConflictError):
        _publish(owner, campaign, generation)
    payload = json.loads(
        (owner.transaction_root / f"{slot.slot_id.value}.publication.json").read_text(encoding="utf-8")
    )
    assert payload["state"] == "prepared"
    assert backend.objects[slot.manifest_location.value].payload == b"wrong"


def test_unknown_receipt_fields_are_rejected(tmp_path: Path) -> None:
    backend = _FakeImmutableBackend()
    campaign = _campaign()
    generation = _generation(tmp_path)
    owner = _owner(tmp_path, backend)
    _publish(owner, campaign, generation)
    path = owner.transaction_root / f"{campaign.slots[1].slot_id.value}.receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PublicationError):
        _publish(owner, campaign, generation)
