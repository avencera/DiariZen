"""Strict data contracts for external validation campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Mapping, TypeAlias


CAMPAIGN_SCHEMA = "diarizen-validation-campaign-v1"
SLOT_SCHEMA = "diarizen-validation-slot-v1"
PUBLICATION_TRANSACTION_SCHEMA = "diarizen-publication-transaction-v1"
PUBLISHED_SNAPSHOT_SCHEMA = "diarizen-published-snapshot-v1"
VALIDATION_RESULT_SCHEMA = "diarizen-validation-result-v1"
SELECTION_STATE_SCHEMA = "diarizen-selection-state-v1"
TRAINING_RUN_STATE_SCHEMA = "diarizen-training-run-state-v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ValidationContractError(ValueError):
    """A validation campaign document violates its schema or invariants."""


def _exact_fields(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    actual = frozenset(value)
    missing = required - actual
    extra = actual - required - optional
    if missing or extra:
        raise ValidationContractError(
            f"{context} fields are not exact: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _string(value: object, field: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ValidationContractError(f"{field} must be a non-empty string of at most {maximum} UTF-8 bytes")
    if any(not character.isprintable() for character in value):
        raise ValidationContractError(f"{field} cannot contain control characters")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationContractError(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    """Parse one finite nonnegative metric."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationContractError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValidationContractError(f"{field} must be a finite nonnegative number")
    return number


def _utc_datetime(value: object, field: str) -> datetime:
    """Parse one timezone-aware UTC datetime."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValidationContractError(f"{field} must be an ISO-8601 datetime") from error
    else:
        raise ValidationContractError(f"{field} must be an ISO-8601 datetime")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValidationContractError(f"{field} must use UTC")
    return parsed.astimezone(timezone.utc)


def _datetime_string(value: datetime) -> str:
    """Return the deterministic UTC representation of one datetime."""

    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _enum(value: object, enum_type: type[Enum], field: str) -> Enum:
    """Parse one closed enum value."""

    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ValidationContractError(f"{field} is not supported") from error


def _digest(value: object, field: str) -> Sha256Digest:
    """Parse one digest field after the digest type has been declared."""

    return Sha256Digest.parse(value, field)


def canonical_digest(value: Mapping[str, object]) -> str:
    """Return the SHA-256 of one canonical compact JSON object."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, order=True)
class Sha256Digest:
    """One canonical lowercase SHA-256 digest."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or _DIGEST.fullmatch(self.value) is None:
            raise ValidationContractError("SHA-256 digest must contain 64 lowercase hexadecimal characters")

    @classmethod
    def parse(cls, value: object, field: str) -> Sha256Digest:
        """Parse a digest field."""

        if not isinstance(value, str):
            raise ValidationContractError(f"{field} must be a SHA-256 digest")
        return cls(value)


@dataclass(frozen=True, order=True)
class ArtifactLocation:
    """One bounded opaque immutable object location."""

    value: str

    def __post_init__(self) -> None:
        _string(self.value, "artifact location", maximum=2_048)

    @classmethod
    def parse(cls, value: object, field: str) -> ArtifactLocation:
        """Parse an artifact location field."""

        return cls(_string(value, field, maximum=2_048))


@dataclass(frozen=True)
class EpochZero:
    """The snapshot before the first optimizer update."""

    def to_dict(self) -> dict[str, object]:
        """Return the tagged JSON representation."""

        return {"kind": "epoch_zero"}


@dataclass(frozen=True)
class CompleteEpoch:
    """The snapshot after one complete epoch."""

    epoch: int

    def __post_init__(self) -> None:
        _integer(self.epoch, "complete epoch", minimum=1)

    def to_dict(self) -> dict[str, object]:
        """Return the tagged JSON representation."""

        return {"kind": "complete_epoch", "epoch": self.epoch}


@dataclass(frozen=True)
class FinalPartial:
    """The target snapshot after a nonempty partial epoch."""

    completed_epochs: int
    partial_updates: int

    def __post_init__(self) -> None:
        _integer(self.completed_epochs, "completed epochs")
        _integer(self.partial_updates, "partial updates", minimum=1)

    def to_dict(self) -> dict[str, object]:
        """Return the tagged JSON representation."""

        return {
            "kind": "final_partial",
            "completed_epochs": self.completed_epochs,
            "partial_updates": self.partial_updates,
        }


SnapshotPoint: TypeAlias = EpochZero | CompleteEpoch | FinalPartial


def snapshot_point_from_dict(value: object) -> SnapshotPoint:
    """Parse one strict tagged snapshot point."""

    if not isinstance(value, Mapping):
        raise ValidationContractError("snapshot point must be an object")
    kind = value.get("kind")
    if kind == "epoch_zero":
        _exact_fields(value, required=frozenset({"kind"}), context="epoch-zero point")
        return EpochZero()
    if kind == "complete_epoch":
        _exact_fields(value, required=frozenset({"kind", "epoch"}), context="complete-epoch point")
        return CompleteEpoch(_integer(value["epoch"], "complete epoch", minimum=1))
    if kind == "final_partial":
        _exact_fields(
            value,
            required=frozenset({"kind", "completed_epochs", "partial_updates"}),
            context="final-partial point",
        )
        return FinalPartial(
            completed_epochs=_integer(value["completed_epochs"], "completed epochs"),
            partial_updates=_integer(value["partial_updates"], "partial updates", minimum=1),
        )
    raise ValidationContractError("snapshot point kind is not supported")


class StoppingPolicy(str, Enum):
    """The owner and terminal bounds for external validation."""

    EXTERNAL_PATIENCE_OR_MAX_UPDATES = "external_patience_or_max_updates"


@dataclass(frozen=True)
class ValidationSlot:
    """One deterministic immutable validation snapshot point."""

    slot_id: Sha256Digest
    ordinal: int
    point: SnapshotPoint
    updates: int
    manifest_location: ArtifactLocation
    model_location: ArtifactLocation

    def __post_init__(self) -> None:
        _integer(self.ordinal, "slot ordinal")
        _integer(self.updates, "slot updates")
        if isinstance(self.point, EpochZero) and self.updates != 0:
            raise ValidationContractError("epoch-zero slot must use update zero")

    def identity_fields(self) -> dict[str, object]:
        """Return every field bound by the deterministic slot identity."""

        return {
            "ordinal": self.ordinal,
            "point": self.point.to_dict(),
            "updates": self.updates,
            "manifest_location": self.manifest_location.value,
            "model_location": self.model_location.value,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "schema": SLOT_SCHEMA,
            "slot_id": self.slot_id.value,
            **self.identity_fields(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ValidationSlot:
        """Parse and validate one strict slot document."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("validation slot must be an object")
        _exact_fields(
            value,
            required=frozenset(
                {
                    "schema",
                    "slot_id",
                    "ordinal",
                    "point",
                    "updates",
                    "manifest_location",
                    "model_location",
                }
            ),
            context="validation slot",
        )
        if value["schema"] != SLOT_SCHEMA:
            raise ValidationContractError("validation slot schema is not supported")
        return cls(
            slot_id=Sha256Digest.parse(value["slot_id"], "slot_id"),
            ordinal=_integer(value["ordinal"], "slot ordinal"),
            point=snapshot_point_from_dict(value["point"]),
            updates=_integer(value["updates"], "slot updates"),
            manifest_location=ArtifactLocation.parse(value["manifest_location"], "manifest_location"),
            model_location=ArtifactLocation.parse(value["model_location"], "model_location"),
        )


@dataclass(frozen=True)
class ValidationCampaign:
    """The complete immutable contract for one ordered validation campaign."""

    campaign_id: Sha256Digest
    training_launch_id: str
    max_updates: int
    updates_per_complete_epoch: int
    slots: tuple[ValidationSlot, ...]
    dev_bundle_digest: Sha256Digest
    trainer_configuration_digest: Sha256Digest
    evaluator_image_identity: str
    evaluator_implementation_digest: Sha256Digest
    maximum_validation_lag: int
    stopping_policy: StoppingPolicy

    def __post_init__(self) -> None:
        _string(self.training_launch_id, "training launch identity")
        _integer(self.max_updates, "maximum updates", minimum=1)
        _integer(self.updates_per_complete_epoch, "updates per complete epoch", minimum=1)
        _string(self.evaluator_image_identity, "evaluator image identity", maximum=512)
        _integer(self.maximum_validation_lag, "maximum validation lag")
        if not self.slots:
            raise ValidationContractError("validation campaign must contain slots")
        if tuple(slot.ordinal for slot in self.slots) != tuple(range(len(self.slots))):
            raise ValidationContractError("validation slot ordinals must be contiguous from zero")
        if not isinstance(self.slots[0].point, EpochZero):
            raise ValidationContractError("validation campaign must start with epoch zero")
        if len({slot.slot_id for slot in self.slots}) != len(self.slots):
            raise ValidationContractError("validation slot identities must be unique")
        self._validate_progress()
        self._validate_identities()

    def _validate_progress(self) -> None:
        complete_epochs, remainder = divmod(self.max_updates, self.updates_per_complete_epoch)
        expected_updates = [0, *(epoch * self.updates_per_complete_epoch for epoch in range(1, complete_epochs + 1))]
        if remainder:
            expected_updates.append(self.max_updates)
        if [slot.updates for slot in self.slots] != expected_updates:
            raise ValidationContractError("validation slots do not match the campaign update plan")
        expected_points: list[SnapshotPoint] = [EpochZero()]
        expected_points.extend(CompleteEpoch(epoch) for epoch in range(1, complete_epochs + 1))
        if remainder:
            expected_points.append(FinalPartial(complete_epochs, remainder))
        if [slot.point for slot in self.slots] != expected_points:
            raise ValidationContractError("validation slots do not match the campaign epoch plan")

    def campaign_fields(self) -> dict[str, object]:
        """Return immutable campaign fields used to derive its identity."""

        return {
            "training_launch_id": self.training_launch_id,
            "max_updates": self.max_updates,
            "updates_per_complete_epoch": self.updates_per_complete_epoch,
            "dev_bundle_digest": self.dev_bundle_digest.value,
            "trainer_configuration_digest": self.trainer_configuration_digest.value,
            "evaluator_image_identity": self.evaluator_image_identity,
            "evaluator_implementation_digest": self.evaluator_implementation_digest.value,
            "maximum_validation_lag": self.maximum_validation_lag,
            "stopping_policy": self.stopping_policy.value,
        }

    def _validate_identities(self) -> None:
        expected_campaign = Sha256Digest(canonical_digest(self.campaign_fields()))
        if self.campaign_id != expected_campaign:
            raise ValidationContractError("campaign identity does not match its immutable contract")
        for slot in self.slots:
            expected_slot = Sha256Digest(
                canonical_digest({"campaign_id": self.campaign_id.value, **slot.identity_fields()})
            )
            if slot.slot_id != expected_slot:
                raise ValidationContractError("slot identity does not match its campaign and snapshot contract")

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_id": self.campaign_id.value,
            **self.campaign_fields(),
            "slots": [slot.to_dict() for slot in self.slots],
        }

    @classmethod
    def from_dict(cls, value: object) -> ValidationCampaign:
        """Parse and validate one strict campaign document."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("validation campaign must be an object")
        fields = frozenset(
            {
                "schema",
                "campaign_id",
                "training_launch_id",
                "max_updates",
                "updates_per_complete_epoch",
                "slots",
                "dev_bundle_digest",
                "trainer_configuration_digest",
                "evaluator_image_identity",
                "evaluator_implementation_digest",
                "maximum_validation_lag",
                "stopping_policy",
            }
        )
        _exact_fields(value, required=fields, context="validation campaign")
        if value["schema"] != CAMPAIGN_SCHEMA:
            raise ValidationContractError("validation campaign schema is not supported")
        slots = value["slots"]
        if not isinstance(slots, list):
            raise ValidationContractError("validation campaign slots must be an array")
        try:
            stopping_policy = StoppingPolicy(value["stopping_policy"])
        except (TypeError, ValueError) as error:
            raise ValidationContractError("validation stopping policy is not supported") from error
        return cls(
            campaign_id=Sha256Digest.parse(value["campaign_id"], "campaign_id"),
            training_launch_id=_string(value["training_launch_id"], "training_launch_id"),
            max_updates=_integer(value["max_updates"], "max_updates", minimum=1),
            updates_per_complete_epoch=_integer(
                value["updates_per_complete_epoch"], "updates_per_complete_epoch", minimum=1
            ),
            slots=tuple(ValidationSlot.from_dict(slot) for slot in slots),
            dev_bundle_digest=Sha256Digest.parse(value["dev_bundle_digest"], "dev_bundle_digest"),
            trainer_configuration_digest=Sha256Digest.parse(
                value["trainer_configuration_digest"], "trainer_configuration_digest"
            ),
            evaluator_image_identity=_string(
                value["evaluator_image_identity"], "evaluator_image_identity", maximum=512
            ),
            evaluator_implementation_digest=Sha256Digest.parse(
                value["evaluator_implementation_digest"], "evaluator_implementation_digest"
            ),
            maximum_validation_lag=_integer(value["maximum_validation_lag"], "maximum_validation_lag"),
            stopping_policy=stopping_policy,
        )


class ValidationSlotState(str, Enum):
    """The closed lifecycle of one validation slot."""

    PLANNED = "planned"
    PUBLISHED = "published"
    VALIDATED = "validated"
    UNUSED = "unused"


class ValidationSlotTerminalReason(str, Enum):
    """The typed reason a planned slot is left unused."""

    ACCEPTED_EARLY_STOP = "accepted_early_stop"


class PublicationTransactionState(str, Enum):
    """The two durable phases of one publication transaction."""

    PREPARED = "prepared"
    MANIFEST_COMMITTED = "manifest_committed"


class SelectionLifecycle(str, Enum):
    """The lifecycle owned by the trusted selection controller."""

    READY = "ready"
    WAITING = "waiting"
    STOP_REQUESTED = "stop_requested"
    COMPLETED = "completed"
    FAILED = "failed"


class BestScoreKind(str, Enum):
    """The closed states of the selection best-score record."""

    NO_BEST_RESULT = "no_best_result"
    BEST_RESULT = "best_result"


class TrainingRunState(str, Enum):
    """The closed lifecycle of one resumable training run."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingCompletionKind(str, Enum):
    """The two valid terminal completion outcomes."""

    MAX_UPDATES_REACHED = "max_updates_reached"
    ACCEPTED_EARLY_STOP = "accepted_early_stop"


class ResumableInterruptionKind(str, Enum):
    """Typed reasons that allow a run to resume from its durable progress."""

    EXTERNAL_PAUSE = "external_pause"
    RETRYABLE_FAILURE = "retryable_failure"
    RESOURCE_LIMIT = "resource_limit"
    LEASE_LOST = "lease_lost"


class TrainingFailureKind(str, Enum):
    """Typed terminal training failures."""

    PUBLICATION_CONFLICT = "publication_conflict"
    RECOVERY_CORRUPTION = "recovery_corruption"
    SELECTION_FAILURE = "selection_failure"
    TRAINER_FAILURE = "trainer_failure"


class SelectionFailureKind(str, Enum):
    """Typed terminal selection failures."""

    RESULT_IDENTITY_CONFLICT = "result_identity_conflict"
    RESULT_CONFLICT = "result_conflict"
    CONTROLLER_FAILURE = "controller_failure"


class ContiguousCursorKind(str, Enum):
    """The closed contiguous result-prefix cursor states."""

    EPOCH_ZERO_PENDING = "epoch_zero_pending"
    EPOCH_ZERO_VALIDATED = "epoch_zero_validated"
    COMPLETE_EPOCH_VALIDATED = "complete_epoch_validated"


@dataclass(frozen=True)
class PublicationTransaction:
    """One immutable, digest-bound publication transaction."""

    state: PublicationTransactionState
    slot_id: Sha256Digest
    model_location: ArtifactLocation
    model_digest: Sha256Digest
    model_length: int
    recovery_generation_id: Sha256Digest
    generation_sequence: int
    progress_digest: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.state, PublicationTransactionState):
            raise ValidationContractError("publication transaction state is not supported")
        _integer(self.model_length, "model length", minimum=1)
        _integer(self.generation_sequence, "generation sequence", minimum=1)

    def identity_fields(self) -> dict[str, object]:
        """Return the immutable fields bound to this transaction."""

        return {
            "slot_id": self.slot_id.value,
            "model_location": self.model_location.value,
            "model_digest": self.model_digest.value,
            "model_length": self.model_length,
            "recovery_generation_id": self.recovery_generation_id.value,
            "generation_sequence": self.generation_sequence,
            "progress_digest": self.progress_digest.value,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "schema": PUBLICATION_TRANSACTION_SCHEMA,
            "state": self.state.value,
            **self.identity_fields(),
        }

    @classmethod
    def from_dict(cls, value: object) -> PublicationTransaction:
        """Parse one strict publication transaction document."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("publication transaction must be an object")
        _exact_fields(
            value,
            required=frozenset({"schema", "state", *PublicationTransaction._IDENTITY_FIELDS}),
            context="publication transaction",
        )
        if value["schema"] != PUBLICATION_TRANSACTION_SCHEMA:
            raise ValidationContractError("publication transaction schema is not supported")
        return cls(
            state=_enum(value["state"], PublicationTransactionState, "publication transaction state"),
            slot_id=_digest(value["slot_id"], "slot_id"),
            model_location=ArtifactLocation.parse(value["model_location"], "model_location"),
            model_digest=_digest(value["model_digest"], "model_digest"),
            model_length=_integer(value["model_length"], "model_length", minimum=1),
            recovery_generation_id=_digest(value["recovery_generation_id"], "recovery_generation_id"),
            generation_sequence=_integer(value["generation_sequence"], "generation_sequence", minimum=1),
            progress_digest=_digest(value["progress_digest"], "progress_digest"),
        )

    _IDENTITY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "slot_id",
            "model_location",
            "model_digest",
            "model_length",
            "recovery_generation_id",
            "generation_sequence",
            "progress_digest",
        }
    )


@dataclass(frozen=True)
class PublishedSnapshot:
    """One immutable model snapshot published for a validation slot."""

    campaign_id: Sha256Digest
    training_launch_id: str
    slot_id: Sha256Digest
    updates: int
    model_digest: Sha256Digest
    model_length: int
    trainer_configuration_digest: Sha256Digest
    dev_bundle_digest: Sha256Digest
    evaluator_image_identity: str
    evaluator_implementation_digest: Sha256Digest
    recovery_generation_id: Sha256Digest
    generation_sequence: int
    progress_digest: Sha256Digest
    publication_time: datetime
    snapshot_id: Sha256Digest | None = None

    def __post_init__(self) -> None:
        _string(self.training_launch_id, "training launch identity")
        _integer(self.updates, "snapshot updates")
        _integer(self.model_length, "model length", minimum=1)
        _string(self.evaluator_image_identity, "evaluator image identity", maximum=512)
        _integer(self.generation_sequence, "generation sequence", minimum=1)
        publication_time = _utc_datetime(self.publication_time, "publication time")
        object.__setattr__(self, "publication_time", publication_time)
        expected = Sha256Digest(canonical_digest(self.identity_fields()))
        if self.snapshot_id is None:
            object.__setattr__(self, "snapshot_id", expected)
        elif self.snapshot_id != expected:
            raise ValidationContractError("snapshot identity does not match its immutable contract")

    @property
    def published_at(self) -> datetime:
        """Return the UTC publication time."""

        return self.publication_time

    @property
    def launch_id(self) -> str:
        """Return the training launch identity."""

        return self.training_launch_id

    @property
    def update_count(self) -> int:
        """Return the optimizer update count represented by this snapshot."""

        return self.updates

    def identity_fields(self) -> dict[str, object]:
        """Return every identity bound to this published snapshot."""

        return {
            "campaign_id": self.campaign_id.value,
            "training_launch_id": self.training_launch_id,
            "slot_id": self.slot_id.value,
            "updates": self.updates,
            "model_digest": self.model_digest.value,
            "model_length": self.model_length,
            "trainer_configuration_digest": self.trainer_configuration_digest.value,
            "dev_bundle_digest": self.dev_bundle_digest.value,
            "evaluator_image_identity": self.evaluator_image_identity,
            "evaluator_implementation_digest": self.evaluator_implementation_digest.value,
            "recovery_generation_id": self.recovery_generation_id.value,
            "generation_sequence": self.generation_sequence,
            "progress_digest": self.progress_digest.value,
            "publication_time": _datetime_string(self.publication_time),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "schema": PUBLISHED_SNAPSHOT_SCHEMA,
            "snapshot_id": self.snapshot_id.value,
            **self.identity_fields(),
        }

    @classmethod
    def from_dict(cls, value: object) -> PublishedSnapshot:
        """Parse one strict published snapshot document."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("published snapshot must be an object")
        fields = frozenset({"schema", "snapshot_id", *PublishedSnapshot._IDENTITY_FIELDS})
        _exact_fields(value, required=fields, context="published snapshot")
        if value["schema"] != PUBLISHED_SNAPSHOT_SCHEMA:
            raise ValidationContractError("published snapshot schema is not supported")
        return cls(
            campaign_id=_digest(value["campaign_id"], "campaign_id"),
            training_launch_id=_string(value["training_launch_id"], "training_launch_id"),
            slot_id=_digest(value["slot_id"], "slot_id"),
            updates=_integer(value["updates"], "updates"),
            model_digest=_digest(value["model_digest"], "model_digest"),
            model_length=_integer(value["model_length"], "model_length", minimum=1),
            trainer_configuration_digest=_digest(
                value["trainer_configuration_digest"], "trainer_configuration_digest"
            ),
            dev_bundle_digest=_digest(value["dev_bundle_digest"], "dev_bundle_digest"),
            evaluator_image_identity=_string(
                value["evaluator_image_identity"], "evaluator_image_identity", maximum=512
            ),
            evaluator_implementation_digest=_digest(
                value["evaluator_implementation_digest"], "evaluator_implementation_digest"
            ),
            recovery_generation_id=_digest(value["recovery_generation_id"], "recovery_generation_id"),
            generation_sequence=_integer(value["generation_sequence"], "generation_sequence", minimum=1),
            progress_digest=_digest(value["progress_digest"], "progress_digest"),
            publication_time=_utc_datetime(value["publication_time"], "publication_time"),
            snapshot_id=_digest(value["snapshot_id"], "snapshot_id"),
        )

    _IDENTITY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "campaign_id",
            "training_launch_id",
            "slot_id",
            "updates",
            "model_digest",
            "model_length",
            "trainer_configuration_digest",
            "dev_bundle_digest",
            "evaluator_image_identity",
            "evaluator_implementation_digest",
            "recovery_generation_id",
            "generation_sequence",
            "progress_digest",
            "publication_time",
        }
    )


@dataclass(frozen=True)
class ValidationResult:
    """One evaluator result bound to every published-snapshot identity."""

    snapshot_id: Sha256Digest
    campaign_id: Sha256Digest
    training_launch_id: str
    slot_id: Sha256Digest
    updates: int
    model_digest: Sha256Digest
    model_length: int
    trainer_configuration_digest: Sha256Digest
    dev_bundle_digest: Sha256Digest
    evaluator_image_identity: str
    evaluator_implementation_digest: Sha256Digest
    recovery_generation_id: Sha256Digest
    generation_sequence: int
    progress_digest: Sha256Digest
    publication_time: datetime
    loss: float
    der: float
    false_alarm: float
    miss: float
    confusion: float
    started_at: datetime
    completed_at: datetime

    def __post_init__(self) -> None:
        _string(self.training_launch_id, "training launch identity")
        _integer(self.updates, "result updates")
        _integer(self.model_length, "model length", minimum=1)
        _string(self.evaluator_image_identity, "evaluator image identity", maximum=512)
        _integer(self.generation_sequence, "generation sequence", minimum=1)
        publication_time = _utc_datetime(self.publication_time, "publication time")
        started_at = _utc_datetime(self.started_at, "evaluation start time")
        completed_at = _utc_datetime(self.completed_at, "evaluation completion time")
        if completed_at < started_at:
            raise ValidationContractError("evaluation completion time must not precede its start time")
        object.__setattr__(self, "publication_time", publication_time)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "completed_at", completed_at)
        for field in ("loss", "der", "false_alarm", "miss", "confusion"):
            _finite_nonnegative(getattr(self, field), field)
        if self.snapshot_id != Sha256Digest(canonical_digest(self.snapshot_identity_fields())):
            raise ValidationContractError("validation result snapshot identity does not match its fields")

    @property
    def false_alarm_rate(self) -> float:
        """Return the false-alarm metric."""

        return self.false_alarm

    @property
    def fa(self) -> float:
        """Return the false-alarm metric."""

        return self.false_alarm

    @property
    def miss_rate(self) -> float:
        """Return the miss metric."""

        return self.miss

    @property
    def confusion_rate(self) -> float:
        """Return the confusion metric."""

        return self.confusion

    @property
    def evaluation_started_at(self) -> datetime:
        """Return the evaluator start time."""

        return self.started_at

    @property
    def evaluation_completed_at(self) -> datetime:
        """Return the evaluator completion time."""

        return self.completed_at

    def snapshot_identity_fields(self) -> dict[str, object]:
        """Return the exact published-snapshot identity copied into this result."""

        return {
            "campaign_id": self.campaign_id.value,
            "training_launch_id": self.training_launch_id,
            "slot_id": self.slot_id.value,
            "updates": self.updates,
            "model_digest": self.model_digest.value,
            "model_length": self.model_length,
            "trainer_configuration_digest": self.trainer_configuration_digest.value,
            "dev_bundle_digest": self.dev_bundle_digest.value,
            "evaluator_image_identity": self.evaluator_image_identity,
            "evaluator_implementation_digest": self.evaluator_implementation_digest.value,
            "recovery_generation_id": self.recovery_generation_id.value,
            "generation_sequence": self.generation_sequence,
            "progress_digest": self.progress_digest.value,
            "publication_time": _datetime_string(self.publication_time),
        }

    def matches_snapshot(self, snapshot: PublishedSnapshot) -> bool:
        """Return whether this result is bound to the supplied snapshot."""

        return (
            self.snapshot_id == snapshot.snapshot_id and self.snapshot_identity_fields() == snapshot.identity_fields()
        )

    @property
    def snapshot(self) -> PublishedSnapshot:
        """Return the published snapshot represented by this result."""

        return PublishedSnapshot(
            campaign_id=self.campaign_id,
            training_launch_id=self.training_launch_id,
            slot_id=self.slot_id,
            updates=self.updates,
            model_digest=self.model_digest,
            model_length=self.model_length,
            trainer_configuration_digest=self.trainer_configuration_digest,
            dev_bundle_digest=self.dev_bundle_digest,
            evaluator_image_identity=self.evaluator_image_identity,
            evaluator_implementation_digest=self.evaluator_implementation_digest,
            recovery_generation_id=self.recovery_generation_id,
            generation_sequence=self.generation_sequence,
            progress_digest=self.progress_digest,
            publication_time=self.publication_time,
            snapshot_id=self.snapshot_id,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "schema": VALIDATION_RESULT_SCHEMA,
            "snapshot_id": self.snapshot_id.value,
            **self.snapshot_identity_fields(),
            "loss": self.loss,
            "der": self.der,
            "false_alarm": self.false_alarm,
            "miss": self.miss,
            "confusion": self.confusion,
            "started_at": _datetime_string(self.started_at),
            "completed_at": _datetime_string(self.completed_at),
        }

    @classmethod
    def from_dict(cls, value: object) -> ValidationResult:
        """Parse one strict validation result document."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("validation result must be an object")
        fields = frozenset(
            {
                "schema",
                "snapshot_id",
                *ValidationResult._SNAPSHOT_FIELDS,
                "loss",
                "der",
                "false_alarm",
                "miss",
                "confusion",
                "started_at",
                "completed_at",
            }
        )
        _exact_fields(value, required=fields, context="validation result")
        if value["schema"] != VALIDATION_RESULT_SCHEMA:
            raise ValidationContractError("validation result schema is not supported")
        return cls(
            snapshot_id=_digest(value["snapshot_id"], "snapshot_id"),
            campaign_id=_digest(value["campaign_id"], "campaign_id"),
            training_launch_id=_string(value["training_launch_id"], "training_launch_id"),
            slot_id=_digest(value["slot_id"], "slot_id"),
            updates=_integer(value["updates"], "updates"),
            model_digest=_digest(value["model_digest"], "model_digest"),
            model_length=_integer(value["model_length"], "model_length", minimum=1),
            trainer_configuration_digest=_digest(
                value["trainer_configuration_digest"], "trainer_configuration_digest"
            ),
            dev_bundle_digest=_digest(value["dev_bundle_digest"], "dev_bundle_digest"),
            evaluator_image_identity=_string(
                value["evaluator_image_identity"], "evaluator_image_identity", maximum=512
            ),
            evaluator_implementation_digest=_digest(
                value["evaluator_implementation_digest"], "evaluator_implementation_digest"
            ),
            recovery_generation_id=_digest(value["recovery_generation_id"], "recovery_generation_id"),
            generation_sequence=_integer(value["generation_sequence"], "generation_sequence", minimum=1),
            progress_digest=_digest(value["progress_digest"], "progress_digest"),
            publication_time=_utc_datetime(value["publication_time"], "publication_time"),
            loss=_finite_nonnegative(value["loss"], "loss"),
            der=_finite_nonnegative(value["der"], "der"),
            false_alarm=_finite_nonnegative(value["false_alarm"], "false_alarm"),
            miss=_finite_nonnegative(value["miss"], "miss"),
            confusion=_finite_nonnegative(value["confusion"], "confusion"),
            started_at=_utc_datetime(value["started_at"], "started_at"),
            completed_at=_utc_datetime(value["completed_at"], "completed_at"),
        )

    _SNAPSHOT_FIELDS: ClassVar[frozenset[str]] = frozenset(PublishedSnapshot._IDENTITY_FIELDS)


@dataclass(frozen=True)
class PlannedSlotState:
    """A slot that has not been published."""

    state: ClassVar[ValidationSlotState] = ValidationSlotState.PLANNED

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"state": self.state.value}


@dataclass(frozen=True)
class PublishedSlotState:
    """A slot with one immutable published snapshot."""

    snapshot: PublishedSnapshot
    state: ClassVar[ValidationSlotState] = ValidationSlotState.PUBLISHED

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"state": self.state.value, "snapshot": self.snapshot.to_dict()}


@dataclass(frozen=True)
class ValidatedSlotState:
    """A slot with one published snapshot and its matching result."""

    snapshot: PublishedSnapshot
    result: ValidationResult
    state: ClassVar[ValidationSlotState] = ValidationSlotState.VALIDATED

    def __post_init__(self) -> None:
        if not self.result.matches_snapshot(self.snapshot):
            raise ValidationContractError("validated slot result does not match its published snapshot")

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"state": self.state.value, "snapshot": self.snapshot.to_dict(), "result": self.result.to_dict()}


@dataclass(frozen=True)
class UnusedSlotState:
    """A planned slot left unused for one typed terminal reason."""

    reason: ValidationSlotTerminalReason
    state: ClassVar[ValidationSlotState] = ValidationSlotState.UNUSED

    def __post_init__(self) -> None:
        if not isinstance(self.reason, ValidationSlotTerminalReason):
            raise ValidationContractError("unused slot terminal reason is not supported")

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"state": self.state.value, "reason": self.reason.value}


ValidationSlotStateValue: TypeAlias = PlannedSlotState | PublishedSlotState | ValidatedSlotState | UnusedSlotState


def validation_slot_state_from_dict(value: object) -> ValidationSlotStateValue:
    """Parse one strict validation-slot state variant."""

    if not isinstance(value, Mapping):
        raise ValidationContractError("validation slot state must be an object")
    state = _enum(value.get("state"), ValidationSlotState, "validation slot state")
    if state is ValidationSlotState.PLANNED:
        _exact_fields(value, required=frozenset({"state"}), context="planned validation slot state")
        return PlannedSlotState()
    if state is ValidationSlotState.PUBLISHED:
        _exact_fields(value, required=frozenset({"state", "snapshot"}), context="published validation slot state")
        return PublishedSlotState(PublishedSnapshot.from_dict(value["snapshot"]))
    if state is ValidationSlotState.VALIDATED:
        _exact_fields(
            value,
            required=frozenset({"state", "snapshot", "result"}),
            context="validated validation slot state",
        )
        return ValidatedSlotState(
            snapshot=PublishedSnapshot.from_dict(value["snapshot"]),
            result=ValidationResult.from_dict(value["result"]),
        )
    _exact_fields(value, required=frozenset({"state", "reason"}), context="unused validation slot state")
    return UnusedSlotState(_enum(value["reason"], ValidationSlotTerminalReason, "unused slot reason"))


@dataclass(frozen=True)
class EpochZeroPendingCursor:
    """No result has been accepted for epoch zero."""

    kind: ClassVar[ContiguousCursorKind] = ContiguousCursorKind.EPOCH_ZERO_PENDING

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"kind": self.kind.value}


@dataclass(frozen=True)
class EpochZeroValidatedCursor:
    """Epoch zero is the contiguous validated prefix."""

    kind: ClassVar[ContiguousCursorKind] = ContiguousCursorKind.EPOCH_ZERO_VALIDATED

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"kind": self.kind.value}


@dataclass(frozen=True)
class CompleteEpochValidatedCursor:
    """A complete epoch is the latest contiguous validated prefix."""

    epoch: int
    kind: ClassVar[ContiguousCursorKind] = ContiguousCursorKind.COMPLETE_EPOCH_VALIDATED

    def __post_init__(self) -> None:
        _integer(self.epoch, "validated complete epoch", minimum=1)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"kind": self.kind.value, "epoch": self.epoch}


ContiguousValidationCursor: TypeAlias = (
    EpochZeroPendingCursor | EpochZeroValidatedCursor | CompleteEpochValidatedCursor
)


def contiguous_cursor_from_dict(value: object) -> ContiguousValidationCursor:
    """Parse one strict contiguous validation cursor."""

    if not isinstance(value, Mapping):
        raise ValidationContractError("selection cursor must be an object")
    kind = _enum(value.get("kind"), ContiguousCursorKind, "selection cursor")
    if kind is ContiguousCursorKind.EPOCH_ZERO_PENDING:
        _exact_fields(value, required=frozenset({"kind"}), context="epoch-zero-pending cursor")
        return EpochZeroPendingCursor()
    if kind is ContiguousCursorKind.EPOCH_ZERO_VALIDATED:
        _exact_fields(value, required=frozenset({"kind"}), context="epoch-zero-validated cursor")
        return EpochZeroValidatedCursor()
    _exact_fields(value, required=frozenset({"kind", "epoch"}), context="complete-epoch-validated cursor")
    return CompleteEpochValidatedCursor(_integer(value["epoch"], "validated complete epoch", minimum=1))


@dataclass(frozen=True)
class SelectionFailure:
    """A typed terminal selection failure."""

    kind: SelectionFailureKind
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SelectionFailureKind):
            raise ValidationContractError("selection failure kind is not supported")
        _string(self.detail, "selection failure detail", maximum=2_048)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"kind": self.kind.value, "detail": self.detail}

    @classmethod
    def from_dict(cls, value: object) -> SelectionFailure:
        """Parse one strict selection failure."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("selection failure must be an object")
        _exact_fields(value, required=frozenset({"kind", "detail"}), context="selection failure")
        return cls(
            kind=_enum(value["kind"], SelectionFailureKind, "selection failure kind"),
            detail=_string(value["detail"], "selection failure detail", maximum=2_048),
        )


@dataclass(frozen=True)
class NoBestResult:
    """The initial selection state before any result establishes a best score."""

    kind: ClassVar[BestScoreKind] = BestScoreKind.NO_BEST_RESULT

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"kind": self.kind.value}


@dataclass(frozen=True)
class BestResult:
    """A finite best score bound to its exact selected snapshot."""

    score: float
    snapshot_id: Sha256Digest
    kind: ClassVar[BestScoreKind] = BestScoreKind.BEST_RESULT

    def __post_init__(self) -> None:
        _finite_nonnegative(self.score, "best score")

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"kind": self.kind.value, "score": self.score, "snapshot_id": self.snapshot_id.value}


BestScore: TypeAlias = NoBestResult | BestResult


def best_score_from_dict(value: object) -> BestScore:
    """Parse one strict tagged best-score state."""

    if not isinstance(value, Mapping):
        raise ValidationContractError("selection best score must be an object")
    kind = _enum(value.get("kind"), BestScoreKind, "selection best score")
    if kind is BestScoreKind.NO_BEST_RESULT:
        _exact_fields(value, required=frozenset({"kind"}), context="selection no-best result")
        return NoBestResult()
    _exact_fields(value, required=frozenset({"kind", "score", "snapshot_id"}), context="selection best result")
    return BestResult(
        score=_finite_nonnegative(value["score"], "best score"),
        snapshot_id=_digest(value["snapshot_id"], "best snapshot_id"),
    )


@dataclass(frozen=True)
class SelectionState:
    """The durable, ordered state owned by the trusted selection controller."""

    results_received_by_slot: tuple[tuple[Sha256Digest, ValidationResult], ...]
    cursor: ContiguousValidationCursor
    best_score: BestScore
    patience: int
    top_five_snapshot_ids: tuple[Sha256Digest, ...]
    revision: int
    lifecycle: SelectionLifecycle
    failure: SelectionFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.cursor, (EpochZeroPendingCursor, EpochZeroValidatedCursor, CompleteEpochValidatedCursor)
        ):
            raise ValidationContractError("selection cursor is not supported")
        if not isinstance(self.lifecycle, SelectionLifecycle):
            raise ValidationContractError("selection lifecycle is not supported")
        if not isinstance(self.best_score, (NoBestResult, BestResult)):
            raise ValidationContractError("selection best score is not supported")
        _integer(self.patience, "selection patience")
        _integer(self.revision, "selection revision")
        if len(self.top_five_snapshot_ids) > 5:
            raise ValidationContractError("selection top-five list cannot contain more than five snapshots")
        if len(set(self.top_five_snapshot_ids)) != len(self.top_five_snapshot_ids):
            raise ValidationContractError("selection top-five snapshot identities must be unique")
        if isinstance(self.best_score, NoBestResult) and self.top_five_snapshot_ids:
            raise ValidationContractError("selection without a best result must have an empty top-five list")
        if isinstance(self.best_score, BestResult) and (
            not self.top_five_snapshot_ids or self.top_five_snapshot_ids[0] != self.best_score.snapshot_id
        ):
            raise ValidationContractError("selection best result must be the first top-five snapshot")
        if self.lifecycle is SelectionLifecycle.FAILED and self.failure is None:
            raise ValidationContractError("failed selection state requires a typed failure")
        if self.lifecycle is not SelectionLifecycle.FAILED and self.failure is not None:
            raise ValidationContractError("only failed selection state may carry a failure")
        raw_results = (
            self.results_received_by_slot.items()
            if isinstance(self.results_received_by_slot, Mapping)
            else self.results_received_by_slot
        )
        normalized: list[tuple[Sha256Digest, ValidationResult]] = []
        seen: set[Sha256Digest] = set()
        for slot_id, result in raw_results:
            if not isinstance(slot_id, Sha256Digest) or not isinstance(result, ValidationResult):
                raise ValidationContractError("selection results must be typed slot/result pairs")
            if slot_id != result.slot_id:
                raise ValidationContractError("selection result slot identity does not match its key")
            if slot_id in seen:
                raise ValidationContractError("selection cannot receive two results for one slot")
            seen.add(slot_id)
            normalized.append((slot_id, result))
        normalized.sort(key=lambda item: item[0].value)
        object.__setattr__(self, "results_received_by_slot", tuple(normalized))

    @property
    def received_results(self) -> tuple[ValidationResult, ...]:
        """Return received results in deterministic slot-identity order."""

        return tuple(result for _, result in self.results_received_by_slot)

    @property
    def results_by_slot(self) -> tuple[tuple[Sha256Digest, ValidationResult], ...]:
        """Return result identities paired with their slot identities."""

        return self.results_received_by_slot

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "schema": SELECTION_STATE_SCHEMA,
            "results_received_by_slot": [
                {"slot_id": slot_id.value, "result": result.to_dict()}
                for slot_id, result in self.results_received_by_slot
            ],
            "cursor": self.cursor.to_dict(),
            "best_score": self.best_score.to_dict(),
            "patience": self.patience,
            "top_five_snapshot_ids": [snapshot_id.value for snapshot_id in self.top_five_snapshot_ids],
            "revision": self.revision,
            "lifecycle": self.lifecycle.value,
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }

    @classmethod
    def from_dict(cls, value: object) -> SelectionState:
        """Parse one strict selection-state document."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("selection state must be an object")
        _exact_fields(
            value,
            required=frozenset(
                {
                    "schema",
                    "results_received_by_slot",
                    "cursor",
                    "best_score",
                    "patience",
                    "top_five_snapshot_ids",
                    "revision",
                    "lifecycle",
                    "failure",
                }
            ),
            context="selection state",
        )
        if value["schema"] != SELECTION_STATE_SCHEMA:
            raise ValidationContractError("selection state schema is not supported")
        raw_results = value["results_received_by_slot"]
        if not isinstance(raw_results, list):
            raise ValidationContractError("selection results received by slot must be an array")
        received: list[tuple[Sha256Digest, ValidationResult]] = []
        for raw_result in raw_results:
            if not isinstance(raw_result, Mapping):
                raise ValidationContractError("selection received result must be an object")
            _exact_fields(raw_result, required=frozenset({"slot_id", "result"}), context="selection received result")
            received.append(
                (
                    _digest(raw_result["slot_id"], "selection result slot_id"),
                    ValidationResult.from_dict(raw_result["result"]),
                )
            )
        top_five = value["top_five_snapshot_ids"]
        if not isinstance(top_five, list):
            raise ValidationContractError("selection top-five snapshot IDs must be an array")
        if value["failure"] is None:
            failure = None
        else:
            failure = SelectionFailure.from_dict(value["failure"])
        return cls(
            results_received_by_slot=tuple(received),
            cursor=contiguous_cursor_from_dict(value["cursor"]),
            best_score=best_score_from_dict(value["best_score"]),
            patience=_integer(value["patience"], "patience"),
            top_five_snapshot_ids=tuple(_digest(item, "top_five_snapshot_id") for item in top_five),
            revision=_integer(value["revision"], "revision"),
            lifecycle=_enum(value["lifecycle"], SelectionLifecycle, "selection lifecycle"),
            failure=failure,
        )


@dataclass(frozen=True)
class ResumableTrainingProgress:
    """The typed progress needed to resume a training run."""

    updates: int
    selection_revision: int
    recovery_generation_id: Sha256Digest
    generation_sequence: int
    progress_digest: Sha256Digest

    def __post_init__(self) -> None:
        _integer(self.updates, "training updates")
        _integer(self.selection_revision, "training selection revision")
        _integer(self.generation_sequence, "training generation sequence", minimum=1)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "updates": self.updates,
            "selection_revision": self.selection_revision,
            "recovery_generation_id": self.recovery_generation_id.value,
            "generation_sequence": self.generation_sequence,
            "progress_digest": self.progress_digest.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> ResumableTrainingProgress:
        """Parse resumable training progress."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("resumable training progress must be an object")
        _exact_fields(
            value,
            required=frozenset(
                {"updates", "selection_revision", "recovery_generation_id", "generation_sequence", "progress_digest"}
            ),
            context="resumable training progress",
        )
        return cls(
            updates=_integer(value["updates"], "training updates"),
            selection_revision=_integer(value["selection_revision"], "training selection revision"),
            recovery_generation_id=_digest(value["recovery_generation_id"], "recovery_generation_id"),
            generation_sequence=_integer(value["generation_sequence"], "training generation sequence", minimum=1),
            progress_digest=_digest(value["progress_digest"], "progress_digest"),
        )


@dataclass(frozen=True)
class ResumableInterruption:
    """One typed interruption from which training may resume."""

    kind: ResumableInterruptionKind
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ResumableInterruptionKind):
            raise ValidationContractError("resumable interruption kind is not supported")
        _string(self.detail, "resumable interruption detail", maximum=2_048)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"kind": self.kind.value, "detail": self.detail}

    @classmethod
    def from_dict(cls, value: object) -> ResumableInterruption:
        """Parse one strict resumable interruption."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("resumable interruption must be an object")
        _exact_fields(value, required=frozenset({"kind", "detail"}), context="resumable interruption")
        return cls(
            kind=_enum(value["kind"], ResumableInterruptionKind, "resumable interruption kind"),
            detail=_string(value["detail"], "resumable interruption detail", maximum=2_048),
        )


@dataclass(frozen=True)
class MaxUpdatesReached:
    """A completed run that reached its configured update bound."""

    kind: ClassVar[TrainingCompletionKind] = TrainingCompletionKind.MAX_UPDATES_REACHED

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"kind": self.kind.value}


@dataclass(frozen=True)
class AcceptedEarlyStop:
    """A completed run accepted by one exact selection revision and result."""

    selection_revision: int
    triggering_result: ValidationResult
    best_snapshot: PublishedSnapshot
    kind: ClassVar[TrainingCompletionKind] = TrainingCompletionKind.ACCEPTED_EARLY_STOP

    def __post_init__(self) -> None:
        _integer(self.selection_revision, "accepted early-stop selection revision", minimum=1)
        if not self.triggering_result.matches_snapshot(self.best_snapshot) and (
            self.triggering_result.campaign_id != self.best_snapshot.campaign_id
            or self.triggering_result.training_launch_id != self.best_snapshot.training_launch_id
        ):
            raise ValidationContractError("early-stop result and best snapshot do not belong to one campaign")

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "kind": self.kind.value,
            "selection_revision": self.selection_revision,
            "triggering_result": self.triggering_result.to_dict(),
            "best_snapshot": self.best_snapshot.to_dict(),
        }


TrainingCompletion: TypeAlias = MaxUpdatesReached | AcceptedEarlyStop


def training_completion_from_dict(value: object) -> TrainingCompletion:
    """Parse one strict training completion outcome."""

    if not isinstance(value, Mapping):
        raise ValidationContractError("training completion must be an object")
    kind = _enum(value.get("kind"), TrainingCompletionKind, "training completion kind")
    if kind is TrainingCompletionKind.MAX_UPDATES_REACHED:
        _exact_fields(value, required=frozenset({"kind"}), context="max-updates completion")
        return MaxUpdatesReached()
    _exact_fields(
        value,
        required=frozenset({"kind", "selection_revision", "triggering_result", "best_snapshot"}),
        context="accepted early-stop completion",
    )
    return AcceptedEarlyStop(
        selection_revision=_integer(value["selection_revision"], "accepted early-stop selection revision", minimum=1),
        triggering_result=ValidationResult.from_dict(value["triggering_result"]),
        best_snapshot=PublishedSnapshot.from_dict(value["best_snapshot"]),
    )


@dataclass(frozen=True)
class TrainingFailure:
    """A typed terminal training failure."""

    kind: TrainingFailureKind
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TrainingFailureKind):
            raise ValidationContractError("training failure kind is not supported")
        _string(self.detail, "training failure detail", maximum=2_048)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"kind": self.kind.value, "detail": self.detail}

    @classmethod
    def from_dict(cls, value: object) -> TrainingFailure:
        """Parse one strict training failure."""

        if not isinstance(value, Mapping):
            raise ValidationContractError("training failure must be an object")
        _exact_fields(value, required=frozenset({"kind", "detail"}), context="training failure")
        return cls(
            kind=_enum(value["kind"], TrainingFailureKind, "training failure kind"),
            detail=_string(value["detail"], "training failure detail", maximum=2_048),
        )


@dataclass(frozen=True)
class ActiveTrainingRun:
    """An active run with durable progress and an optional prepared publication."""

    progress: ResumableTrainingProgress
    prepared_publication: PublicationTransaction | None = None
    state: ClassVar[TrainingRunState] = TrainingRunState.ACTIVE

    def __post_init__(self) -> None:
        if self.prepared_publication is not None:
            if self.prepared_publication.state is not PublicationTransactionState.PREPARED:
                raise ValidationContractError("active training state may carry only a prepared publication")
            if (
                self.prepared_publication.recovery_generation_id != self.progress.recovery_generation_id
                or self.prepared_publication.generation_sequence != self.progress.generation_sequence
                or self.prepared_publication.progress_digest != self.progress.progress_digest
            ):
                raise ValidationContractError("prepared publication does not match active training progress")

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "state": self.state.value,
            "progress": self.progress.to_dict(),
            "prepared_publication": self.prepared_publication.to_dict() if self.prepared_publication else None,
        }


@dataclass(frozen=True)
class PausedTrainingRun:
    """A paused run with typed resumable progress and interruption."""

    progress: ResumableTrainingProgress
    interruption: ResumableInterruption
    state: ClassVar[TrainingRunState] = TrainingRunState.PAUSED

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "state": self.state.value,
            "progress": self.progress.to_dict(),
            "interruption": self.interruption.to_dict(),
        }


@dataclass(frozen=True)
class CompletedTrainingRun:
    """A terminal run with one valid completion outcome."""

    completion: TrainingCompletion
    state: ClassVar[TrainingRunState] = TrainingRunState.COMPLETED

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"state": self.state.value, "completion": self.completion.to_dict()}


@dataclass(frozen=True)
class FailedTrainingRun:
    """A terminal run with one typed failure."""

    failure: TrainingFailure
    state: ClassVar[TrainingRunState] = TrainingRunState.FAILED

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"state": self.state.value, "failure": self.failure.to_dict()}


TrainingRunStateValue: TypeAlias = ActiveTrainingRun | PausedTrainingRun | CompletedTrainingRun | FailedTrainingRun


def training_run_state_from_dict(value: object) -> TrainingRunStateValue:
    """Parse one strict training-run state variant."""

    if not isinstance(value, Mapping):
        raise ValidationContractError("training run state must be an object")
    state = _enum(value.get("state"), TrainingRunState, "training run state")
    if state is TrainingRunState.ACTIVE:
        _exact_fields(
            value,
            required=frozenset({"state", "progress", "prepared_publication"}),
            context="active training run state",
        )
        prepared = (
            None
            if value["prepared_publication"] is None
            else PublicationTransaction.from_dict(value["prepared_publication"])
        )
        return ActiveTrainingRun(ResumableTrainingProgress.from_dict(value["progress"]), prepared)
    if state is TrainingRunState.PAUSED:
        _exact_fields(
            value,
            required=frozenset({"state", "progress", "interruption"}),
            context="paused training run state",
        )
        return PausedTrainingRun(
            progress=ResumableTrainingProgress.from_dict(value["progress"]),
            interruption=ResumableInterruption.from_dict(value["interruption"]),
        )
    if state is TrainingRunState.COMPLETED:
        _exact_fields(value, required=frozenset({"state", "completion"}), context="completed training run state")
        return CompletedTrainingRun(training_completion_from_dict(value["completion"]))
    _exact_fields(value, required=frozenset({"state", "failure"}), context="failed training run state")
    return FailedTrainingRun(TrainingFailure.from_dict(value["failure"]))
