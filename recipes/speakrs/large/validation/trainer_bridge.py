"""Typed coordination between the trainer and trusted external validation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeAlias

from diarizen.trainer_utils import canonical_trainer_progress_digest

from ..recovery import RecoveryGeneration
from .config import ExternalValidationConfig
from .contracts import (
    AcceptedEarlyStop,
    EpochZeroValidatedCursor,
    FailedTrainingRun,
    MaxUpdatesReached,
    PublishedSnapshot,
    SelectionLifecycle,
    SelectionState,
    Sha256Digest,
    TrainingCompletion,
    TrainingFailure,
    TrainingFailureKind,
    TrainingRunStateValue,
    ValidationCampaign,
    ValidationContractError,
    ValidationSlot,
    training_run_state_from_dict,
)
from .controller import (
    LagPermit,
    LagWait,
    LagWaitReason,
    SelectionPolicy,
    TrustedStateError,
    TrustedStateStore,
    ValidationLagGate,
)


class TrainerBridgeError(RuntimeError):
    """A trainer bridge operation cannot satisfy the external contract."""


class TrainerBridgeSelectionFailure(TrainerBridgeError):
    """The trusted selection controller reported a terminal failure."""

    def __init__(self, failure: TrainingFailure) -> None:
        if failure.kind is not TrainingFailureKind.SELECTION_FAILURE:
            raise TypeError("trainer bridge selection failure must use selection-failure kind")
        self.failure = failure
        super().__init__(failure.detail)


class TrainerBridgePublicationFailure(TrainerBridgeError):
    """A snapshot or recovery publication could not be reconciled."""


class SnapshotPublisher(Protocol):
    """Publish one immutable typed validation snapshot."""

    def publish(
        self,
        campaign: ValidationCampaign,
        slot: ValidationSlot,
        generation_directory: Path,
        progress_digest: Sha256Digest,
    ) -> PublishedSnapshot:
        """Publish or reconcile one slot from one complete recovery generation."""


class BridgeWaiter(Protocol):
    """Wait until the caller's trusted-state source may have changed."""

    def __call__(self) -> None:
        """Wait once without changing trainer-owned state."""


class BridgeStateReader(Protocol):
    """Read one digest-bound typed selection state."""

    def __call__(self) -> SelectionState:
        """Read and validate the latest trusted selection state."""


class BridgeDecisionKind(str, Enum):
    """The closed decisions returned to the trainer at safe boundaries."""

    CONTINUE = "continue"
    WAIT = "wait"
    ACCEPT_EARLY_STOP = "accept_early_stop"
    MAX_UPDATES_REACHED = "max_updates_reached"
    FAILED = "failed"


@dataclass(frozen=True)
class ContinueTraining:
    """The trainer may continue with its next optimizer phase."""

    selection: SelectionState
    kind: BridgeDecisionKind = BridgeDecisionKind.CONTINUE


@dataclass(frozen=True)
class WaitForValidation:
    """The trainer must wait before starting another epoch."""

    selection: SelectionState
    pending_complete_epochs: int
    next_epoch: int
    reason: LagWaitReason
    kind: BridgeDecisionKind = BridgeDecisionKind.WAIT


@dataclass(frozen=True)
class AcceptEarlyStop:
    """The trusted controller accepted an early stop at this safe boundary."""

    completion: AcceptedEarlyStop
    selection: SelectionState
    kind: BridgeDecisionKind = BridgeDecisionKind.ACCEPT_EARLY_STOP


@dataclass(frozen=True)
class ReachMaximumUpdates:
    """The trainer reached its exact campaign update target."""

    completion: MaxUpdatesReached
    selection: SelectionState
    kind: BridgeDecisionKind = BridgeDecisionKind.MAX_UPDATES_REACHED


@dataclass(frozen=True)
class SelectionFailed:
    """The trusted controller failed and training must exit nonzero."""

    failure: TrainingFailure
    selection: SelectionState
    kind: BridgeDecisionKind = BridgeDecisionKind.FAILED


BridgeDecision: TypeAlias = (
    ContinueTraining | WaitForValidation | AcceptEarlyStop | ReachMaximumUpdates | SelectionFailed
)


@dataclass(frozen=True)
class _PublishedSlot:
    """One campaign slot and its immutable published snapshot."""

    slot: ValidationSlot
    snapshot: PublishedSnapshot


def _read_json_document(path: Path) -> object:
    """Read one strict JSON boundary document."""

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
        raise TrainerBridgeError(f"cannot read trusted validation document {path}") from error


def _campaign_from_reference(config: ExternalValidationConfig) -> ValidationCampaign:
    """Load and digest-check the campaign manifest named by typed config."""

    path = config.campaign_manifest.path
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise TrainerBridgeError("external validation campaign manifest cannot be read") from error
    digest = Sha256Digest(hashlib.sha256(payload).hexdigest())
    if digest != config.campaign_manifest.sha256:
        raise TrainerBridgeError("external validation campaign manifest digest does not match its config")
    try:
        return ValidationCampaign.from_dict(_read_json_document(path))
    except (TypeError, ValueError, ValidationContractError) as error:
        raise TrainerBridgeError("external validation campaign manifest is invalid") from error


def _slot_for_updates(campaign: ValidationCampaign, updates: int) -> ValidationSlot:
    """Find the one slot bound to an exact optimizer-update count."""

    matches = tuple(slot for slot in campaign.slots if slot.updates == updates)
    if len(matches) != 1:
        raise TrainerBridgeError("campaign does not contain one exact validation slot for the update boundary")
    return matches[0]


def _selection_failure(state: SelectionState) -> SelectionFailed:
    """Convert a failed trusted selection state to a typed trainer decision."""

    failure = state.failure
    if failure is None:
        raise TrainerBridgeError("trusted selection failed without its typed failure")
    training_failure = TrainingFailure(TrainingFailureKind.SELECTION_FAILURE, failure.detail)
    return SelectionFailed(training_failure, state)


class TrainerBridge:
    """Own typed publication, trusted selection reads, and lag decisions.

    The bridge never computes a score, patience count, or ranking.  Those values
    come only from the digest-bound trusted selection state.
    """

    def __init__(
        self,
        campaign: ValidationCampaign,
        selection_store: TrustedStateStore,
        publisher: SnapshotPublisher,
        *,
        state_reader: BridgeStateReader | None = None,
        wait_for_state: BridgeWaiter | None = None,
        wait_seconds: float = 1.0,
    ) -> None:
        if not isinstance(campaign, ValidationCampaign):
            raise TypeError("trainer bridge campaign must use the typed campaign contract")
        if not isinstance(selection_store, TrustedStateStore):
            raise TypeError("trainer bridge selection store must use TrustedStateStore")
        if selection_store.campaign != campaign:
            raise TrainerBridgeError("trainer bridge selection store belongs to another campaign")
        if not callable(getattr(publisher, "publish", None)):
            raise TypeError("trainer bridge publisher must implement publish")
        if wait_seconds <= 0:
            raise ValueError("trainer bridge wait_seconds must be positive")
        self.campaign = campaign
        self.selection_store = selection_store
        self.publisher = publisher
        self._state_reader = state_reader or selection_store.load
        self._wait_for_state = wait_for_state or (lambda: time.sleep(wait_seconds))
        try:
            self._selection = selection_store.load()
        except TrustedStateError as error:
            raise TrainerBridgeSelectionFailure(
                TrainingFailure(TrainingFailureKind.SELECTION_FAILURE, "trusted selection state is invalid")
            ) from error
        self._selection_revision = self._selection.revision
        self._published: dict[Sha256Digest, _PublishedSlot] = {}
        self._load_published_snapshots()

    @classmethod
    def from_config(
        cls,
        config: ExternalValidationConfig,
        publisher: SnapshotPublisher,
        *,
        state_reader: BridgeStateReader | None = None,
        wait_for_state: BridgeWaiter | None = None,
        wait_seconds: float | None = None,
        selection_store: TrustedStateStore | None = None,
    ) -> TrainerBridge:
        """Build a bridge from the canonical external TOML boundary."""

        if not isinstance(config, ExternalValidationConfig):
            raise TypeError("external trainer bridge requires ExternalValidationConfig")
        campaign = _campaign_from_reference(config)
        store = selection_store or TrustedStateStore(
            config.trusted_selection_state_path,
            campaign,
            SelectionPolicy(config.patience_limit),
        )
        return cls(
            campaign,
            store,
            publisher,
            state_reader=state_reader,
            wait_for_state=wait_for_state,
            wait_seconds=config.poll_backoff.initial_seconds if wait_seconds is None else wait_seconds,
        )

    @property
    def selection(self) -> SelectionState:
        """Return the latest trusted selection accepted by this trainer."""

        return self._selection

    @property
    def selection_revision(self) -> int:
        """Return the latest accepted trusted selection revision."""

        return self._selection_revision

    @property
    def published_snapshots(self) -> tuple[PublishedSnapshot, ...]:
        """Return published snapshots in campaign slot order."""

        return tuple(
            self._published[slot.slot_id].snapshot for slot in self.campaign.slots if slot.slot_id in self._published
        )

    def selection_payload(self) -> dict[str, object]:
        """Return the typed selection payload for a trainer recovery document."""

        return self._selection.to_dict()

    def published_snapshot_payload(self) -> list[dict[str, object]]:
        """Return typed published snapshots for a trainer recovery document."""

        return [snapshot.to_dict() for snapshot in self.published_snapshots]

    def has_published_boundary(self, updates: int) -> bool:
        """Return whether the exact update boundary already has a snapshot."""

        slot = _slot_for_updates(self.campaign, updates)
        return slot.slot_id in self._published

    def refresh_published_snapshots(self) -> None:
        """Reload committed snapshot manifests after another process publishes them."""

        self._load_published_snapshots()

    def load_training_run_state(self, generation_directory: Path) -> TrainingRunStateValue:
        """Read the typed training-run state from one complete generation."""

        path = Path(generation_directory) / "progress.json"
        document = _read_json_document(path)
        if not isinstance(document, Mapping):
            raise TrainerBridgeError("trainer progress is not an object")
        raw_state = document.get("training_run_state")
        try:
            return training_run_state_from_dict(raw_state)
        except (TypeError, ValueError, ValidationContractError) as error:
            raise TrainerBridgeError("trainer progress does not contain a typed training-run state") from error

    def _load_published_snapshots(self) -> None:
        root = getattr(self.publisher, "transaction_root", None)
        if root is None:
            return
        root = Path(root)
        if not root.is_dir():
            return
        for path in sorted(root.glob("*.snapshot.json")):
            try:
                snapshot = PublishedSnapshot.from_dict(_read_json_document(path))
            except (TypeError, ValueError, ValidationContractError) as error:
                raise TrainerBridgePublicationFailure("trusted snapshot manifest is invalid") from error
            slot = next(
                (candidate for candidate in self.campaign.slots if candidate.slot_id == snapshot.slot_id), None
            )
            if slot is None or snapshot.campaign_id != self.campaign.campaign_id:
                raise TrainerBridgePublicationFailure("trusted snapshot belongs to another campaign")
            if (
                snapshot.training_launch_id != self.campaign.training_launch_id
                or snapshot.updates != slot.updates
                or snapshot.dev_bundle_digest != self.campaign.dev_bundle_digest
                or snapshot.trainer_configuration_digest != self.campaign.trainer_configuration_digest
                or snapshot.evaluator_image_identity != self.campaign.evaluator_image_identity
                or snapshot.evaluator_implementation_digest != self.campaign.evaluator_implementation_digest
            ):
                raise TrainerBridgePublicationFailure("trusted snapshot identity is invalid")
            self._published[snapshot.slot_id] = _PublishedSlot(slot, snapshot)

    def read_newer_selection(self) -> SelectionState | None:
        """Accept one newer digest-bound selection revision, if available."""

        try:
            candidate = self._state_reader()
        except (OSError, TrustedStateError, TypeError, ValueError, ValidationContractError) as error:
            raise TrainerBridgeSelectionFailure(
                TrainingFailure(TrainingFailureKind.SELECTION_FAILURE, "trusted selection state is invalid")
            ) from error
        if not isinstance(candidate, SelectionState):
            raise TrainerBridgeSelectionFailure(
                TrainingFailure(TrainingFailureKind.SELECTION_FAILURE, "trusted selection state is not typed")
            )
        self._validate_campaign_selection(candidate)
        if candidate.revision <= self._selection_revision:
            return None
        self._selection = candidate
        self._selection_revision = candidate.revision
        return candidate

    def _validate_campaign_selection(self, state: SelectionState) -> None:
        """Check campaign ownership before accepting a state from an injected reader."""

        for _, result in state.results_received_by_slot:
            slot = next((candidate for candidate in self.campaign.slots if candidate.slot_id == result.slot_id), None)
            if slot is None or result.updates != slot.updates:
                raise TrainerBridgeSelectionFailure(
                    TrainingFailure(
                        TrainingFailureKind.SELECTION_FAILURE, "trusted selection contains an unknown slot"
                    )
                )
            if (
                result.campaign_id != self.campaign.campaign_id
                or result.training_launch_id != self.campaign.training_launch_id
            ):
                raise TrainerBridgeSelectionFailure(
                    TrainingFailure(
                        TrainingFailureKind.SELECTION_FAILURE, "trusted selection belongs to another campaign"
                    )
                )
            if (
                result.dev_bundle_digest != self.campaign.dev_bundle_digest
                or result.trainer_configuration_digest != self.campaign.trainer_configuration_digest
                or result.evaluator_image_identity != self.campaign.evaluator_image_identity
                or result.evaluator_implementation_digest != self.campaign.evaluator_implementation_digest
            ):
                raise TrainerBridgeSelectionFailure(
                    TrainingFailure(
                        TrainingFailureKind.SELECTION_FAILURE, "trusted selection evaluator identity differs"
                    )
                )

    def observe_selection(self) -> BridgeDecision:
        """Read a newer trusted state and convert terminal state to a decision."""

        try:
            self.read_newer_selection()
            if self._selection.lifecycle is SelectionLifecycle.FAILED:
                return _selection_failure(self._selection)
            if self._selection.lifecycle is SelectionLifecycle.STOP_REQUESTED:
                return self._early_stop_decision()
            if self._selection.lifecycle is SelectionLifecycle.COMPLETED:
                return SelectionFailed(
                    TrainingFailure(
                        TrainingFailureKind.SELECTION_FAILURE,
                        "trusted selection completed before trainer completion",
                    ),
                    self._selection,
                )
            return ContinueTraining(self._selection)
        except TrainerBridgeSelectionFailure as error:
            return SelectionFailed(error.failure, self._selection)

    def publish_boundary(self, generation_directory: Path) -> PublishedSnapshot:
        """Publish or reconcile the snapshot for one complete generation."""

        generation_directory = Path(generation_directory)
        try:
            generation = RecoveryGeneration.from_name(generation_directory.name)
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise TrainerBridgePublicationFailure("recovery generation name is not canonical") from error
        slot = _slot_for_updates(self.campaign, generation.updates)
        try:
            progress_digest = Sha256Digest(canonical_trainer_progress_digest(generation_directory / "progress.json"))
            snapshot = self.publisher.publish(self.campaign, slot, generation_directory, progress_digest)
        except TrainerBridgeError:
            raise
        except Exception as error:
            raise TrainerBridgePublicationFailure("validation snapshot publication failed") from error
        if not isinstance(snapshot, PublishedSnapshot):
            raise TrainerBridgePublicationFailure("snapshot publisher returned an untyped snapshot")
        if snapshot.slot_id != slot.slot_id or snapshot.campaign_id != self.campaign.campaign_id:
            raise TrainerBridgePublicationFailure("snapshot publisher returned the wrong campaign slot")
        previous = self._published.get(slot.slot_id)
        if previous is not None and previous.snapshot != snapshot:
            raise TrainerBridgePublicationFailure("snapshot publication changed an immutable slot")
        self._published[slot.slot_id] = _PublishedSlot(slot, snapshot)
        return snapshot

    def _slot_result_received(self, slot: ValidationSlot, state: SelectionState) -> bool:
        return any(slot_id == slot.slot_id for slot_id, _ in state.results_received_by_slot)

    def _early_stop_decision(self) -> AcceptEarlyStop:
        try:
            stop_request = self.selection_store.stop_request
        except TrustedStateError as error:
            raise TrainerBridgeSelectionFailure(
                TrainingFailure(TrainingFailureKind.SELECTION_FAILURE, "trusted stop request is invalid")
            ) from error
        if stop_request is None:
            raise TrainerBridgeSelectionFailure(
                TrainingFailure(TrainingFailureKind.SELECTION_FAILURE, "trusted stop request is missing")
            )
        received = dict(self._selection.results_received_by_slot)
        if received.get(stop_request.triggering_result.slot_id) != stop_request.triggering_result or not any(
            result.snapshot_id == stop_request.best_snapshot.snapshot_id
            and result.snapshot == stop_request.best_snapshot
            for result in received.values()
        ):
            raise TrainerBridgeSelectionFailure(
                TrainingFailure(TrainingFailureKind.SELECTION_FAILURE, "trusted stop request is stale")
            )
        completion = AcceptedEarlyStop(
            stop_request.selection_revision,
            stop_request.triggering_result,
            stop_request.best_snapshot,
        )
        return AcceptEarlyStop(completion, self._selection)

    def await_slot_result(self, updates: int) -> BridgeDecision:
        """Wait until the trusted controller accepts one exact slot result."""

        slot = _slot_for_updates(self.campaign, updates)
        while True:
            decision = self.observe_selection()
            if isinstance(decision, SelectionFailed | AcceptEarlyStop):
                return decision
            if self._slot_result_received(slot, self._selection):
                return ContinueTraining(self._selection)
            self._wait_for_state()

    def await_epoch_zero(self) -> BridgeDecision:
        """Wait for the epoch-zero result before any epoch-one optimizer update."""

        decision = self.await_slot_result(0)
        if not isinstance(decision, ContinueTraining):
            return decision
        if not isinstance(self._selection.cursor, EpochZeroValidatedCursor):
            return SelectionFailed(
                TrainingFailure(TrainingFailureKind.SELECTION_FAILURE, "epoch-zero result is not contiguous"),
                self._selection,
            )
        return decision

    def before_epoch(self, epoch: int) -> BridgeDecision:
        """Apply the exact validation-lag gate before starting an epoch."""

        if isinstance(epoch, bool) or epoch < 1:
            raise ValueError("epoch must be a positive integer")
        decision = self.observe_selection()
        if not isinstance(decision, ContinueTraining):
            return decision
        lag = ValidationLagGate(self.campaign).decide(self._selection, self.published_snapshots, epoch)
        if isinstance(lag, LagPermit):
            return ContinueTraining(self._selection)
        if not isinstance(lag, LagWait):
            raise TrainerBridgeError("validation lag gate returned an unsupported decision")
        return WaitForValidation(
            self._selection,
            lag.pending_complete_epochs,
            epoch,
            lag.reason,
        )

    def await_epoch_start(self, epoch: int) -> BridgeDecision:
        """Wait until the lag gate permits the requested epoch."""

        while True:
            decision = self.before_epoch(epoch)
            if not isinstance(decision, WaitForValidation):
                return decision
            self._wait_for_state()

    def await_final_result(self, updates: int) -> BridgeDecision:
        """Wait for the final full or partial slot after reaching max updates."""

        decision = self.await_slot_result(updates)
        if isinstance(decision, ContinueTraining):
            return self.completion_for_max_updates()
        return decision

    def completion_for_max_updates(self) -> ReachMaximumUpdates:
        """Return the typed max-update completion after the final result arrives."""

        return ReachMaximumUpdates(MaxUpdatesReached(), self._selection)

    def failure_from_decision(self, decision: SelectionFailed) -> FailedTrainingRun:
        """Return the typed failed run payload for a terminal generation."""

        if not isinstance(decision, SelectionFailed):
            raise TypeError("failure_from_decision requires SelectionFailed")
        return FailedTrainingRun(decision.failure)

    def completion_from_decision(self, decision: AcceptEarlyStop | ReachMaximumUpdates) -> TrainingCompletion:
        """Return the typed completion payload for a terminal generation."""

        if isinstance(decision, AcceptEarlyStop):
            return decision.completion
        if isinstance(decision, ReachMaximumUpdates):
            return decision.completion
        raise TypeError("completion_from_decision requires a terminal completion decision")


__all__ = [
    "AcceptEarlyStop",
    "BridgeDecision",
    "BridgeDecisionKind",
    "ContinueTraining",
    "ReachMaximumUpdates",
    "SelectionFailed",
    "SnapshotPublisher",
    "TrainerBridge",
    "TrainerBridgeError",
    "TrainerBridgePublicationFailure",
    "TrainerBridgeSelectionFailure",
    "WaitForValidation",
]
