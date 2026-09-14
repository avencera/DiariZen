"""Crash-safe publication of one immutable validation model snapshot.

The publication owner is deliberately small and recovery-first.  A committed
recovery generation is the only source of model bytes.  The local transaction
is durable before the first remote write, and the manifest is the final remote
write that makes the snapshot visible to evaluators.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from diarizen.trainer_utils import (
    canonical_trainer_progress_digest,
    checkpoint_directory_is_complete,
)

from ..contracts import ObjectStoreDestination
from ..errors import ContractError, PreparationError
from ..hashing import canonical_json, sha256_bytes
from ..recovery import RecoveryGeneration
from ..storage import (
    ImmutableObjectConflictError,
    ImmutableObjectError,
    ImmutableObjectOutcome,
    ImmutableObjectPublisher,
    ImmutableObjectUnavailableError,
    ImmutableObjectUnsupportedSizeError,
    publish_immutable_file,
)
from .contracts import (
    PublicationTransaction,
    PublicationTransactionState,
    PublishedSnapshot,
    Sha256Digest,
    ValidationCampaign,
    ValidationContractError,
    ValidationSlot,
    canonical_digest,
)


PUBLICATION_RECEIPT_SCHEMA = "diarizen-publication-receipt-v1"
TRANSACTION_FILE_SUFFIX = ".publication.json"
SNAPSHOT_FILE_SUFFIX = ".snapshot.json"
RECEIPT_FILE_SUFFIX = ".receipt.json"
MODEL_FILE_NAME = "pytorch_model.bin"
PROGRESS_FILE_NAME = "progress.json"


class PublicationError(ValidationContractError):
    """A publication request or its durable evidence is invalid."""


class PublicationConflictError(PublicationError):
    """An immutable publication identity conflicts with existing evidence."""


class PublicationUnavailableError(PublicationError):
    """An object-store operation did not prove a publication outcome."""


class PublicationUnsupportedError(PublicationError):
    """An object-store adapter cannot publish the requested immutable object."""


class PublicationRecoveryError(PublicationError):
    """A recovery generation cannot authorize publication."""


@dataclass(frozen=True)
class PublicationReceipt:
    """Digest-bound local proof that one immutable snapshot manifest committed."""

    transaction: PublicationTransaction
    snapshot: PublishedSnapshot
    manifest_digest: Sha256Digest
    manifest_length: int

    def __post_init__(self) -> None:
        if self.transaction.state is not PublicationTransactionState.MANIFEST_COMMITTED:
            raise PublicationError("publication receipt requires a committed transaction")
        if self.transaction.slot_id != self.snapshot.slot_id:
            raise PublicationError("publication receipt slot does not match its transaction")
        if self.transaction.model_digest != self.snapshot.model_digest:
            raise PublicationError("publication receipt model digest does not match its transaction")
        if self.transaction.model_length != self.snapshot.model_length:
            raise PublicationError("publication receipt model length does not match its transaction")
        if self.transaction.recovery_generation_id != self.snapshot.recovery_generation_id:
            raise PublicationError("publication receipt recovery identity does not match its transaction")
        if self.transaction.generation_sequence != self.snapshot.generation_sequence:
            raise PublicationError("publication receipt generation sequence does not match its transaction")
        if self.transaction.progress_digest != self.snapshot.progress_digest:
            raise PublicationError("publication receipt progress identity does not match its transaction")
        if (
            isinstance(self.manifest_length, bool)
            or not isinstance(self.manifest_length, int)
            or self.manifest_length < 1
        ):
            raise PublicationError("publication manifest length must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "schema": PUBLICATION_RECEIPT_SCHEMA,
            "transaction": self.transaction.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "manifest_digest": self.manifest_digest.value,
            "manifest_length": self.manifest_length,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PublicationReceipt":
        """Parse one strict digest-bound publication receipt."""

        if not isinstance(value, Mapping):
            raise PublicationError("publication receipt must be an object")
        required = {"schema", "transaction", "snapshot", "manifest_digest", "manifest_length"}
        actual = set(value)
        if actual != required:
            raise PublicationError(
                f"publication receipt fields are not exact: missing={sorted(required - actual)}, "
                f"extra={sorted(actual - required)}"
            )
        if value["schema"] != PUBLICATION_RECEIPT_SCHEMA:
            raise PublicationError("publication receipt schema is not supported")
        return cls(
            transaction=PublicationTransaction.from_dict(value["transaction"]),
            snapshot=PublishedSnapshot.from_dict(value["snapshot"]),
            manifest_digest=Sha256Digest.parse(value["manifest_digest"], "manifest_digest"),
            manifest_length=_positive_integer(value["manifest_length"], "manifest_length"),
        )


@dataclass(frozen=True)
class _PreparedPublication:
    """Validated bytes and identities derived from one complete generation."""

    generation: RecoveryGeneration
    generation_id: Sha256Digest
    progress_digest: Sha256Digest
    model_path: Path
    model_digest: Sha256Digest
    model_length: int


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PublicationError(f"{field} must be a positive integer")
    return value


def _parse_digest(value: Sha256Digest | str, field: str) -> Sha256Digest:
    if isinstance(value, Sha256Digest):
        return value
    return Sha256Digest.parse(value, field)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return canonical_json(value).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        raise PublicationError("publication directory cannot be opened for synchronization") from None
    try:
        os.fsync(descriptor)
    except OSError:
        raise PublicationError("publication directory cannot be synchronized") from None
    finally:
        os.close(descriptor)


def _write_canonical(path: Path, value: Mapping[str, object]) -> None:
    """Write one canonical JSON document through fsync and replace."""

    _replace_temporary(path, _canonical_json_bytes(value))


def _read_json(path: Path, context: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError("JSON object contains duplicate fields")
            parsed[key] = value
        return parsed

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON constant is not supported: {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PublicationError(f"{context} cannot be read") from error


def _file_digest(path: Path) -> tuple[Sha256Digest, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not isinstance(chunk, bytes):
                    raise PublicationRecoveryError("model file returned a non-byte chunk")
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise PublicationRecoveryError("model file cannot be read") from error
    if size < 1:
        raise PublicationRecoveryError("model file must not be empty")
    return Sha256Digest(digest.hexdigest()), size


def derive_recovery_generation_id(generation: RecoveryGeneration) -> Sha256Digest:
    """Derive the canonical recovery identity from update and sequence."""

    if not isinstance(generation, RecoveryGeneration):
        raise PublicationRecoveryError("recovery generation must be typed")
    return Sha256Digest(canonical_digest({"sequence": generation.sequence, "updates": generation.updates}))


def _parse_generation(path: Path) -> RecoveryGeneration:
    try:
        generation = RecoveryGeneration.from_name(path.name)
    except (TypeError, ValueError) as error:
        raise PublicationRecoveryError("recovery generation directory name is not canonical") from error
    if generation.sequence < 1:
        raise PublicationRecoveryError("publication requires a one-based recovery generation sequence")
    return generation


def _prepare_generation(
    generation_directory: Path,
    slot: ValidationSlot,
    expected_progress: Sha256Digest,
) -> _PreparedPublication:
    generation_directory = Path(generation_directory)
    if generation_directory.is_symlink() or not generation_directory.is_dir():
        raise PublicationRecoveryError("recovery generation directory is not a trusted directory")
    generation = _parse_generation(generation_directory)
    if generation.updates != slot.updates:
        raise PublicationConflictError("recovery generation updates do not match the validation slot")
    if not checkpoint_directory_is_complete(
        generation_directory,
        (PROGRESS_FILE_NAME, MODEL_FILE_NAME),
        require_hashed_manifest=True,
    ):
        raise PublicationRecoveryError("recovery generation is not a complete hashed checkpoint")
    progress_path = generation_directory / PROGRESS_FILE_NAME
    try:
        actual_progress = Sha256Digest(canonical_trainer_progress_digest(progress_path))
    except (OSError, TypeError, ValueError) as error:
        raise PublicationRecoveryError("trainer progress is not canonical") from error
    if actual_progress != expected_progress:
        raise PublicationConflictError("trainer progress digest does not match the committed generation")
    model_path = generation_directory / MODEL_FILE_NAME
    model_digest, model_length = _file_digest(model_path)
    return _PreparedPublication(
        generation=generation,
        generation_id=derive_recovery_generation_id(generation),
        progress_digest=actual_progress,
        model_path=model_path,
        model_digest=model_digest,
        model_length=model_length,
    )


def _validate_campaign_slot(campaign: ValidationCampaign, slot: ValidationSlot) -> None:
    if not isinstance(campaign, ValidationCampaign) or not isinstance(slot, ValidationSlot):
        raise PublicationError("campaign and slot must use the typed validation contracts")
    matching = next((candidate for candidate in campaign.slots if candidate.slot_id == slot.slot_id), None)
    if matching is None or matching != slot:
        raise PublicationConflictError("slot is not the exact slot declared by the campaign")
    if slot.model_location == slot.manifest_location:
        raise PublicationConflictError("model and manifest locations must be distinct")


def _validate_destination_locations(destination: ObjectStoreDestination, slot: ValidationSlot) -> None:
    prefix = destination.prefix.strip("/")
    if not prefix or any(
        not location.startswith(prefix + "/") for location in (slot.model_location.value, slot.manifest_location.value)
    ):
        raise PublicationConflictError("slot locations are outside the configured object-store destination")


def _transaction_path(root: Path, slot_id: Sha256Digest) -> Path:
    return Path(root) / f"{slot_id.value}{TRANSACTION_FILE_SUFFIX}"


def _snapshot_path(root: Path, slot_id: Sha256Digest) -> Path:
    return Path(root) / f"{slot_id.value}{SNAPSHOT_FILE_SUFFIX}"


def _receipt_path(root: Path, slot_id: Sha256Digest) -> Path:
    return Path(root) / f"{slot_id.value}{RECEIPT_FILE_SUFFIX}"


def _transaction_identity_matches(actual: PublicationTransaction, expected: PublicationTransaction) -> bool:
    return actual.identity_fields() == expected.identity_fields()


def _snapshot_matches_request(
    snapshot: PublishedSnapshot,
    campaign: ValidationCampaign,
    slot: ValidationSlot,
    prepared: _PreparedPublication,
) -> None:
    expected = {
        "campaign_id": campaign.campaign_id,
        "training_launch_id": campaign.training_launch_id,
        "slot_id": slot.slot_id,
        "updates": slot.updates,
        "model_digest": prepared.model_digest,
        "model_length": prepared.model_length,
        "trainer_configuration_digest": campaign.trainer_configuration_digest,
        "dev_bundle_digest": campaign.dev_bundle_digest,
        "evaluator_image_identity": campaign.evaluator_image_identity,
        "evaluator_implementation_digest": campaign.evaluator_implementation_digest,
        "recovery_generation_id": prepared.generation_id,
        "generation_sequence": prepared.generation.sequence,
        "progress_digest": prepared.progress_digest,
    }
    for field, wanted in expected.items():
        if getattr(snapshot, field) != wanted:
            raise PublicationConflictError(f"published snapshot {field} does not match the request")


def _load_snapshot(
    path: Path,
    campaign: ValidationCampaign,
    slot: ValidationSlot,
    prepared: _PreparedPublication,
    clock: Callable[[], datetime],
) -> PublishedSnapshot:
    def normalized_time(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise PublicationError("publication time must use UTC")
        return value.astimezone(timezone.utc)

    if path.exists():
        snapshot = PublishedSnapshot.from_dict(_read_json(path, "prepared snapshot"))
        _snapshot_matches_request(snapshot, campaign, slot, prepared)
        return snapshot
    publication_time = clock()
    publication_time = normalized_time(publication_time)
    snapshot = PublishedSnapshot(
        campaign_id=campaign.campaign_id,
        training_launch_id=campaign.training_launch_id,
        slot_id=slot.slot_id,
        updates=slot.updates,
        model_digest=prepared.model_digest,
        model_length=prepared.model_length,
        trainer_configuration_digest=campaign.trainer_configuration_digest,
        dev_bundle_digest=campaign.dev_bundle_digest,
        evaluator_image_identity=campaign.evaluator_image_identity,
        evaluator_implementation_digest=campaign.evaluator_implementation_digest,
        recovery_generation_id=prepared.generation_id,
        generation_sequence=prepared.generation.sequence,
        progress_digest=prepared.progress_digest,
        publication_time=publication_time,
    )
    _write_canonical(path, snapshot.to_dict())
    return snapshot


def _load_transaction(path: Path) -> PublicationTransaction | None:
    if not path.exists():
        return None
    return PublicationTransaction.from_dict(_read_json(path, "publication transaction"))


def _load_receipt(path: Path) -> PublicationReceipt | None:
    if not path.exists():
        return None
    return PublicationReceipt.from_dict(_read_json(path, "publication receipt"))


def _manifest_bytes(snapshot: PublishedSnapshot) -> bytes:
    return _canonical_json_bytes(snapshot.to_dict())


def _write_temporary_bytes(path: Path, payload: bytes) -> Path:
    """Write one uniquely named same-directory temporary file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".partial",
            dir=path.parent,
        )
    except OSError:
        raise PublicationError("publication temporary file cannot be created") from None
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        temporary.unlink(missing_ok=True)
        raise PublicationError("publication temporary file cannot be synchronized") from None
    return temporary


def _replace_temporary(path: Path, payload: bytes) -> None:
    temporary = _write_temporary_bytes(path, payload)
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise PublicationError("publication payload cannot be atomically replaced") from None
    _fsync_directory(path.parent)


def _cleanup_temporary(path: Path) -> None:
    """Remove abandoned temporary files owned by one publication path."""

    removed = False
    for temporary in path.parent.glob(f".{path.name}.*.partial"):
        if temporary.is_symlink() or not temporary.is_file():
            continue
        try:
            temporary.unlink()
        except OSError:
            raise PublicationError("abandoned publication temporary file cannot be removed") from None
        removed = True
    if removed:
        _fsync_directory(path.parent)


def _notify_phase(hook: Callable[[str], None] | None, phase: str) -> None:
    if hook is not None:
        hook(phase)


def _publish(
    backend: ImmutableObjectPublisher,
    destination: ObjectStoreDestination,
    key: str,
    path: Path,
    digest: Sha256Digest,
    length: int,
    *,
    content_type: str = "application/octet-stream",
) -> ImmutableObjectOutcome:
    try:
        outcome = publish_immutable_file(
            backend,
            destination,
            key,
            path,
            expected_sha256=digest.value,
            expected_size=length,
            content_type=content_type,
        )
    except ImmutableObjectConflictError:
        raise PublicationConflictError("immutable object publication conflicts with existing content") from None
    except ImmutableObjectUnsupportedSizeError:
        raise PublicationUnsupportedError("immutable object publication exceeds the adapter limit") from None
    except ImmutableObjectUnavailableError:
        raise PublicationUnavailableError("immutable object publication is temporarily unavailable") from None
    except ImmutableObjectError:
        raise PublicationError("immutable object publication returned an invalid storage result") from None
    except ContractError:
        raise PublicationError("immutable object publication request is invalid") from None
    except PreparationError:
        raise PublicationUnavailableError("immutable object publication is temporarily unavailable") from None
    except PublicationError:
        raise
    if not isinstance(outcome, ImmutableObjectOutcome):
        raise PublicationError("immutable object publisher returned an invalid outcome")
    return outcome


def _receipt_matches(
    receipt: PublicationReceipt,
    transaction: PublicationTransaction,
    snapshot: PublishedSnapshot,
    manifest_digest: Sha256Digest,
    manifest_length: int,
) -> None:
    if receipt.transaction != transaction or receipt.snapshot != snapshot:
        raise PublicationConflictError("publication receipt conflicts with the immutable transaction")
    if receipt.manifest_digest != manifest_digest or receipt.manifest_length != manifest_length:
        raise PublicationConflictError("publication receipt conflicts with the snapshot manifest")


class SnapshotPublication:
    """Own one recovery-first immutable snapshot publication transaction."""

    def __init__(
        self,
        destination: ObjectStoreDestination,
        backend: ImmutableObjectPublisher,
        transaction_root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        _phase_hook: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(destination, ObjectStoreDestination):
            raise PublicationError("object-store destination must use the typed contract")
        self.destination = destination
        self.backend = backend
        self.transaction_root = Path(transaction_root)
        self.clock = clock or _utc_now
        self._phase_hook = _phase_hook

    def publish(
        self,
        campaign: ValidationCampaign,
        slot: ValidationSlot,
        generation_directory: Path,
        progress_digest: Sha256Digest | str,
    ) -> PublishedSnapshot:
        """Publish or recover one immutable model snapshot."""

        _validate_campaign_slot(campaign, slot)
        _validate_destination_locations(self.destination, slot)
        expected_progress = _parse_digest(progress_digest, "progress_digest")
        transaction_path = _transaction_path(self.transaction_root, slot.slot_id)
        try:
            prepared = _prepare_generation(generation_directory, slot, expected_progress)
        except PublicationRecoveryError as error:
            if transaction_path.exists():
                raise PublicationConflictError("recovery evidence conflicts with the prepared publication") from error
            raise
        expected_transaction = PublicationTransaction(
            state=PublicationTransactionState.PREPARED,
            slot_id=slot.slot_id,
            model_location=slot.model_location,
            model_digest=prepared.model_digest,
            model_length=prepared.model_length,
            recovery_generation_id=prepared.generation_id,
            generation_sequence=prepared.generation.sequence,
            progress_digest=prepared.progress_digest,
        )
        self.transaction_root.mkdir(parents=True, exist_ok=True)
        if self.transaction_root.is_symlink():
            raise PublicationError("trusted publication root must not be a symbolic link")
        snapshot_path = _snapshot_path(self.transaction_root, slot.slot_id)
        receipt_path = _receipt_path(self.transaction_root, slot.slot_id)
        manifest_payload_path = self.transaction_root / f"{slot.slot_id.value}.manifest.payload"
        for path in (transaction_path, snapshot_path, receipt_path, manifest_payload_path):
            _cleanup_temporary(path)
        transaction = _load_transaction(transaction_path)
        if transaction is not None:
            if not _transaction_identity_matches(transaction, expected_transaction):
                raise PublicationConflictError("publication transaction identity conflicts with the request")
            if transaction.state not in {
                PublicationTransactionState.PREPARED,
                PublicationTransactionState.MANIFEST_COMMITTED,
            }:
                raise PublicationError("publication transaction state is not supported")
        else:
            transaction = expected_transaction
            _write_canonical(transaction_path, transaction.to_dict())

        snapshot = _load_snapshot(snapshot_path, campaign, slot, prepared, self.clock)
        payload = _manifest_bytes(snapshot)
        manifest_digest = Sha256Digest(sha256_bytes(payload))
        manifest_length = len(payload)
        _replace_temporary(manifest_payload_path, payload)
        _notify_phase(self._phase_hook, "prepared")

        committed_transaction = replace(transaction, state=PublicationTransactionState.MANIFEST_COMMITTED)
        receipt = _load_receipt(receipt_path)
        if transaction.state is PublicationTransactionState.MANIFEST_COMMITTED:
            if receipt is not None:
                _receipt_matches(receipt, committed_transaction, snapshot, manifest_digest, manifest_length)
                return snapshot
            # a committed local transaction may have lost only its receipt
            manifest_outcome = _publish(
                self.backend,
                self.destination,
                slot.manifest_location.value,
                manifest_payload_path,
                manifest_digest,
                manifest_length,
                content_type="application/json",
            )
            if manifest_outcome not in {ImmutableObjectOutcome.CREATED, ImmutableObjectOutcome.ALREADY_PRESENT}:
                raise PublicationError("snapshot manifest publication returned an unsupported outcome")
            _write_canonical(transaction_path, committed_transaction.to_dict())
            receipt = PublicationReceipt(committed_transaction, snapshot, manifest_digest, manifest_length)
            _write_canonical(receipt_path, receipt.to_dict())
            return snapshot

        _publish(
            self.backend,
            self.destination,
            slot.model_location.value,
            prepared.model_path,
            prepared.model_digest,
            prepared.model_length,
        )
        _notify_phase(self._phase_hook, "model")

        _publish(
            self.backend,
            self.destination,
            slot.manifest_location.value,
            manifest_payload_path,
            manifest_digest,
            manifest_length,
            content_type="application/json",
        )
        _notify_phase(self._phase_hook, "manifest")
        _write_canonical(transaction_path, committed_transaction.to_dict())
        receipt = PublicationReceipt(committed_transaction, snapshot, manifest_digest, manifest_length)
        _write_canonical(receipt_path, receipt.to_dict())
        return snapshot


__all__ = [
    "PublicationError",
    "PublicationConflictError",
    "PublicationUnavailableError",
    "PublicationUnsupportedError",
    "PublicationRecoveryError",
    "PublicationReceipt",
    "SnapshotPublication",
    "TRANSACTION_FILE_SUFFIX",
    "SNAPSHOT_FILE_SUFFIX",
    "RECEIPT_FILE_SUFFIX",
    "derive_recovery_generation_id",
]
