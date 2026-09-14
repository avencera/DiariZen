"""Trusted ordered validation-result and campaign-completion controllers.

The controller is deliberately pure with respect to validation policy.  It only
uses the immutable contracts in :mod:`contracts` and keeps persistence in a
small digest-bound state store at the edge.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import ClassVar, TypeAlias

from .contracts import (
    AcceptedEarlyStop,
    BestResult,
    BestScore,
    CompleteEpoch,
    CompleteEpochValidatedCursor,
    ContiguousValidationCursor,
    EpochZero,
    EpochZeroPendingCursor,
    EpochZeroValidatedCursor,
    FinalPartial,
    MaxUpdatesReached,
    NoBestResult,
    PlannedSlotState,
    PublishedSlotState,
    PublishedSnapshot,
    SelectionFailure,
    SelectionFailureKind,
    SelectionLifecycle,
    SelectionState,
    Sha256Digest,
    TrainingCompletion,
    UnusedSlotState,
    ValidatedSlotState,
    ValidationCampaign,
    ValidationContractError,
    ValidationResult,
    ValidationSlot,
    ValidationSlotStateValue,
    ValidationSlotTerminalReason,
    training_completion_from_dict,
    validation_slot_state_from_dict,
)


TRUSTED_STATE_SCHEMA = "diarizen-trusted-selection-state-v1"
CAMPAIGN_COMPLETION_STATE_SCHEMA = "diarizen-campaign-completion-state-v1"


class ControllerError(ValidationContractError):
    """Base error for trusted controller operations."""


class ResultImportFailureKind(str, Enum):
    """Closed reasons why an external result cannot be imported."""

    INVALID_DOCUMENT = "invalid_document"
    UNKNOWN_SLOT = "unknown_slot"
    SNAPSHOT_MISMATCH = "snapshot_mismatch"
    CAMPAIGN_MISMATCH = "campaign_mismatch"
    RESULT_CONFLICT = "result_conflict"
    CONTROLLER_TERMINAL = "controller_terminal"


class ResultImportError(ControllerError):
    """A compact result was rejected by the trusted controller."""

    def __init__(
        self,
        kind: ResultImportFailureKind,
        detail: str,
        *,
        slot_id: Sha256Digest | None = None,
    ) -> None:
        self.kind = kind
        self.slot_id = slot_id
        super().__init__(detail)


class ResultConflictError(ResultImportError):
    """A slot already has a different result or snapshot identity."""

    def __init__(self, detail: str, *, slot_id: Sha256Digest | None = None) -> None:
        super().__init__(ResultImportFailureKind.RESULT_CONFLICT, detail, slot_id=slot_id)


class LagGateError(ControllerError):
    """A lag-gate input is not a valid campaign state."""


class CampaignCompletionError(ControllerError):
    """A campaign cannot make the requested lifecycle transition."""


class TrustedStateError(ControllerError):
    """A trusted selection-state document is corrupt or not owned by a campaign."""


class RevisionConflictError(TrustedStateError):
    """A trusted state revision is not newer than the stored revision."""


@dataclass(frozen=True)
class SelectionPolicy:
    """Pure early-stop policy for one validation campaign."""

    patience_limit: int

    def __post_init__(self) -> None:
        if isinstance(self.patience_limit, bool) or not isinstance(self.patience_limit, int):
            raise ControllerError("patience_limit must be an integer")
        if self.patience_limit < 1:
            raise ControllerError("patience_limit must be at least one")

    def improves(self, candidate: float, current: BestScore) -> bool:
        """Return whether one candidate strictly improves the current score."""

        if isinstance(current, NoBestResult):
            return True
        return candidate < current.score


@dataclass(frozen=True)
class StopRequest:
    """One durable stop request emitted when patience expires."""

    selection_revision: int
    triggering_result: ValidationResult
    best_snapshot: PublishedSnapshot

    def __post_init__(self) -> None:
        if (
            isinstance(self.selection_revision, bool)
            or not isinstance(self.selection_revision, int)
            or self.selection_revision < 1
        ):
            raise ControllerError("stop request revision must be at least one")
        if not isinstance(self.triggering_result, ValidationResult) or not isinstance(
            self.best_snapshot, PublishedSnapshot
        ):
            raise ControllerError("stop request must use typed result and snapshot contracts")
        if (
            self.triggering_result.campaign_id != self.best_snapshot.campaign_id
            or self.triggering_result.training_launch_id != self.best_snapshot.training_launch_id
        ):
            raise ControllerError("stop request identities do not belong to one campaign")

    def to_dict(self) -> dict[str, object]:
        """Return a strict representation of the stop request."""

        return {
            "selection_revision": self.selection_revision,
            "triggering_result": self.triggering_result.to_dict(),
            "best_snapshot": self.best_snapshot.to_dict(),
        }


@dataclass(frozen=True)
class SelectionUpdate:
    """The result of one import operation."""

    state: SelectionState
    applied_results: tuple[ValidationResult, ...]
    stop_request: StopRequest | None
    idempotent: bool


def initial_selection_state() -> SelectionState:
    """Build the revision-zero selection state for a new campaign."""

    return SelectionState(
        results_received_by_slot=(),
        cursor=EpochZeroPendingCursor(),
        best_score=NoBestResult(),
        patience=0,
        top_five_snapshot_ids=(),
        revision=0,
        lifecycle=SelectionLifecycle.READY,
    )


@dataclass(frozen=True)
class _SelectionProjection:
    """One pure projection of received results and ordered policy state."""

    cursor: ContiguousValidationCursor
    best_score: BestScore
    patience: int
    ranking: tuple[ValidationResult, ...]
    top_five_snapshot_ids: tuple[Sha256Digest, ...]
    contiguous_results: tuple[ValidationResult, ...]
    newly_applied_results: tuple[ValidationResult, ...]
    first_patience_trigger: ValidationResult | None
    best_at_first_trigger: BestResult | None


def _selection_projection(
    campaign: ValidationCampaign,
    policy: SelectionPolicy,
    received: Mapping[Sha256Digest, ValidationResult],
    previous_received: Mapping[Sha256Digest, ValidationResult] | None = None,
) -> _SelectionProjection:
    """Derive every ordered selection field from one contiguous result prefix."""

    best: BestScore = NoBestResult()
    patience = 0
    first_trigger: ValidationResult | None = None
    best_at_first_trigger: BestResult | None = None
    contiguous: list[tuple[ValidationResult, int]] = []
    latest_complete_epoch = 0
    last_point: EpochZero | CompleteEpoch | FinalPartial | None = None
    for ordinal, slot in enumerate(campaign.slots):
        result = received.get(slot.slot_id)
        if result is None:
            break
        last_point = slot.point
        if isinstance(slot.point, CompleteEpoch):
            latest_complete_epoch = slot.point.epoch
        contiguous.append((result, ordinal))
        if policy.improves(result.der, best):
            best = BestResult(result.der, result.snapshot_id)
            patience = 0
        else:
            patience += 1
        if first_trigger is None and patience >= policy.patience_limit:
            first_trigger = result
            if isinstance(best, BestResult):
                best_at_first_trigger = best

    if not contiguous:
        cursor: ContiguousValidationCursor = EpochZeroPendingCursor()
    elif isinstance(last_point, EpochZero):
        cursor = EpochZeroValidatedCursor()
    elif isinstance(last_point, CompleteEpoch):
        cursor = CompleteEpochValidatedCursor(last_point.epoch)
    elif isinstance(last_point, FinalPartial):
        cursor = (
            CompleteEpochValidatedCursor(latest_complete_epoch)
            if latest_complete_epoch
            else EpochZeroValidatedCursor()
        )
    else:
        raise ControllerError("campaign slot point is not supported")

    ranked = sorted(contiguous, key=lambda item: (item[0].der, item[1], item[0].snapshot_id))
    ranking = tuple(result for result, _ in ranked)
    top_five = tuple(result.snapshot_id for result in ranking[:5])
    if previous_received is None:
        newly_applied = tuple(result for result, _ in contiguous)
    else:
        first_new = next(
            (index for index, (result, _) in enumerate(contiguous) if result.slot_id not in previous_received),
            len(contiguous),
        )
        newly_applied = tuple(result for result, _ in contiguous[first_new:])
    return _SelectionProjection(
        cursor=cursor,
        best_score=best,
        patience=patience,
        ranking=ranking,
        top_five_snapshot_ids=top_five,
        contiguous_results=tuple(result for result, _ in contiguous),
        newly_applied_results=newly_applied,
        first_patience_trigger=first_trigger,
        best_at_first_trigger=best_at_first_trigger,
    )


def _stop_request_from_projection(revision: int, projection: _SelectionProjection) -> StopRequest:
    """Build the one durable stop request owned by the selection projection."""

    if projection.first_patience_trigger is None or projection.best_at_first_trigger is None:
        raise ControllerError("stop-requested state has no deterministic trigger")
    best_result = next(
        result
        for result in projection.contiguous_results
        if result.snapshot_id == projection.best_at_first_trigger.snapshot_id
    )
    return StopRequest(revision, projection.first_patience_trigger, best_result.snapshot)


def _derive_stop_request(
    campaign: ValidationCampaign,
    policy: SelectionPolicy,
    state: SelectionState,
    projection: _SelectionProjection | None = None,
) -> StopRequest:
    """Derive the one stop request from the shared selection projection."""

    received = dict(state.results_received_by_slot)
    projection = projection or _selection_projection(campaign, policy, received)
    return _stop_request_from_projection(state.revision, projection)


def _parse_snapshot(value: object) -> PublishedSnapshot:
    if isinstance(value, PublishedSnapshot):
        return value
    if isinstance(value, PublishedSlotState):
        return value.snapshot
    if isinstance(value, ValidatedSlotState):
        return value.snapshot
    if isinstance(value, Mapping):
        return PublishedSnapshot.from_dict(value)
    raise ControllerError("published snapshot must be a typed snapshot or strict document")


def _parse_slot_id(value: object, field: str = "slot_id") -> Sha256Digest:
    if isinstance(value, Sha256Digest):
        return value
    return Sha256Digest.parse(value, field)


def _snapshots_from_input(
    value: Mapping[object, object] | Iterable[object] | PublishedSnapshot | None,
) -> dict[Sha256Digest, PublishedSnapshot]:
    """Normalize snapshots while preserving their exact slot identities."""

    if value is None:
        return {}
    if isinstance(value, (PublishedSnapshot, PublishedSlotState, ValidatedSlotState)):
        snapshots = [_parse_snapshot(value)]
    elif isinstance(value, Mapping):
        if "schema" in value:
            snapshots = [_parse_snapshot(value)]
        else:
            snapshots = []
            for key, raw_snapshot in value.items():
                snapshot = _parse_snapshot(raw_snapshot)
                key_digest = _parse_slot_id(key, "published snapshot map slot_id")
                if key_digest != snapshot.slot_id:
                    raise ControllerError("published snapshot map key does not match its snapshot")
                snapshots.append(snapshot)
    else:
        snapshots = [_parse_snapshot(item) for item in value]

    normalized: dict[Sha256Digest, PublishedSnapshot] = {}
    for snapshot in snapshots:
        previous = normalized.get(snapshot.slot_id)
        if previous is not None and previous != snapshot:
            raise ResultConflictError("one slot has more than one published snapshot", slot_id=snapshot.slot_id)
        normalized[snapshot.slot_id] = snapshot
    return normalized


class SelectionController:
    """Apply trusted validation results in contiguous slot order."""

    @classmethod
    def initial(
        cls,
        campaign: ValidationCampaign,
        policy: SelectionPolicy,
        published_snapshots: Mapping[object, object] | Iterable[object] | PublishedSnapshot = (),
    ) -> SelectionController:
        """Build a controller with the revision-zero selection state."""

        return cls(campaign, policy, published_snapshots, initial_selection_state())

    def __init__(
        self,
        campaign: ValidationCampaign,
        policy: SelectionPolicy,
        published_snapshots: Mapping[object, object] | Iterable[object] | PublishedSnapshot = (),
        state: SelectionState | None = None,
        *,
        stop_request: StopRequest | None = None,
    ) -> None:
        if not isinstance(campaign, ValidationCampaign):
            raise ControllerError("campaign must use its typed contract")
        if not isinstance(policy, SelectionPolicy):
            raise ControllerError("selection policy must use its typed contract")
        if stop_request is not None and not isinstance(stop_request, StopRequest):
            raise ControllerError("stop request must use its typed contract")
        self.campaign = campaign
        self.policy = policy
        self._snapshots: dict[Sha256Digest, PublishedSnapshot] = {}
        self._initializing = True
        self._state = state if state is not None else initial_selection_state()
        self._last_stop_request = stop_request
        if self._state.lifecycle is SelectionLifecycle.STOP_REQUESTED and stop_request is None:
            raise ControllerError("stop-requested state requires its durable stop request")
        if (
            self._state.lifecycle not in (SelectionLifecycle.STOP_REQUESTED, SelectionLifecycle.COMPLETED)
            and stop_request is not None
        ):
            raise ControllerError("a stop request requires a stop-requested state")
        self._last_update = SelectionUpdate(self._state, (), stop_request, True)
        self._validate_state(self._state)
        for snapshot in _snapshots_from_input(published_snapshots).values():
            self.register_snapshot(snapshot)
        self._initializing = False
        self._validate_state(self._state)
        if self._last_stop_request is not None:
            self._validate_stop_request(self._state, self._last_stop_request)

    @property
    def state(self) -> SelectionState:
        """Return the current immutable selection state."""

        return self._state

    @property
    def last_update(self) -> SelectionUpdate:
        """Return the most recent import outcome."""

        return self._last_update

    @property
    def stop_request(self) -> StopRequest | None:
        """Return the one stop request emitted by this controller, if any."""

        return self._last_stop_request

    @property
    def published_snapshots(self) -> tuple[PublishedSnapshot, ...]:
        """Return registered snapshots in campaign slot order."""

        return tuple(self._snapshots[slot.slot_id] for slot in self.campaign.slots if slot.slot_id in self._snapshots)

    def register_snapshot(self, snapshot: PublishedSnapshot | Mapping[str, object]) -> PublishedSnapshot:
        """Register one immutable snapshot after checking campaign identities."""

        parsed = _parse_snapshot(snapshot)
        if not self._initializing and self._state.lifecycle in (
            SelectionLifecycle.STOP_REQUESTED,
            SelectionLifecycle.COMPLETED,
            SelectionLifecycle.FAILED,
        ):
            previous = self._snapshots.get(parsed.slot_id)
            if previous == parsed:
                return parsed
            raise ControllerError("terminal selection cannot change published snapshots")
        self._validate_snapshot(parsed)
        previous = self._snapshots.get(parsed.slot_id)
        if previous is not None and previous != parsed:
            raise ResultConflictError("published snapshot identity changed for one slot", slot_id=parsed.slot_id)
        self._snapshots[parsed.slot_id] = parsed
        return parsed

    def import_result(
        self,
        compact_result: Mapping[str, object],
        published_snapshot: PublishedSnapshot | Mapping[str, object] | None = None,
    ) -> SelectionState:
        """Import one strict Cloudeck result and return the new trusted state.

        The compact document is parsed exactly once by ``ValidationResult.from_dict``.
        Results may arrive out of order; only the contiguous prefix changes score,
        patience, cursor, or ranking.
        """

        previous_state = self._state
        try:
            if isinstance(compact_result, ValidationResult):
                raise ResultImportError(
                    ResultImportFailureKind.INVALID_DOCUMENT,
                    "compact validation results must cross the strict JSON boundary",
                )
            result = ValidationResult.from_dict(compact_result)
            snapshot = self._matching_snapshot(result, published_snapshot)
            self._validate_result(result, snapshot)
        except ResultImportError as error:
            if self._state.lifecycle not in (
                SelectionLifecycle.COMPLETED,
                SelectionLifecycle.FAILED,
            ):
                self._fail_import(error.kind, str(error), slot_id=error.slot_id)
            raise
        except (ValidationContractError, TypeError, ValueError) as error:
            if previous_state.lifecycle not in (
                SelectionLifecycle.COMPLETED,
                SelectionLifecycle.FAILED,
            ):
                self._fail_import(
                    ResultImportFailureKind.INVALID_DOCUMENT,
                    str(error),
                    slot_id=None,
                )
            raise ResultImportError(ResultImportFailureKind.INVALID_DOCUMENT, str(error)) from error

        existing = dict(previous_state.results_received_by_slot).get(result.slot_id)
        if existing is not None:
            if existing == result:
                self._last_update = SelectionUpdate(previous_state, (), None, True)
                return previous_state
            self._fail_import(
                ResultImportFailureKind.RESULT_CONFLICT,
                "a validation slot received different result content",
                slot_id=result.slot_id,
            )
            raise ResultConflictError("a validation slot received different result content", slot_id=result.slot_id)

        if previous_state.lifecycle in (
            SelectionLifecycle.COMPLETED,
            SelectionLifecycle.FAILED,
        ):
            raise ResultImportError(
                ResultImportFailureKind.CONTROLLER_TERMINAL,
                "selection controller cannot accept a new result after its terminal lifecycle",
                slot_id=result.slot_id,
            )

        received = dict(previous_state.results_received_by_slot)
        received[result.slot_id] = result
        projection = _selection_projection(
            self.campaign,
            self.policy,
            received,
            previous_received=dict(previous_state.results_received_by_slot),
        )

        lifecycle = (
            SelectionLifecycle.STOP_REQUESTED
            if previous_state.lifecycle is SelectionLifecycle.STOP_REQUESTED
            else SelectionLifecycle.WAITING
        )
        if projection.first_patience_trigger is not None and lifecycle is not SelectionLifecycle.STOP_REQUESTED:
            lifecycle = SelectionLifecycle.STOP_REQUESTED

        new_state = SelectionState(
            results_received_by_slot=tuple(received.items()),
            cursor=projection.cursor,
            best_score=projection.best_score,
            patience=projection.patience,
            top_five_snapshot_ids=projection.top_five_snapshot_ids,
            revision=previous_state.revision + 1,
            lifecycle=lifecycle,
        )
        self._state = new_state
        stop_request: StopRequest | None = None
        if (
            lifecycle is SelectionLifecycle.STOP_REQUESTED
            and previous_state.lifecycle is not SelectionLifecycle.STOP_REQUESTED
        ):
            stop_request = _derive_stop_request(self.campaign, self.policy, new_state, projection)
            self._last_stop_request = stop_request
        elif lifecycle is SelectionLifecycle.STOP_REQUESTED:
            stop_request = self._last_stop_request
        if stop_request is not None:
            self._validate_stop_request(new_state, stop_request)
        self._last_update = SelectionUpdate(new_state, projection.newly_applied_results, stop_request, False)
        return new_state

    def persist(self, store: TrustedStateStore) -> SelectionState:
        """Persist the latest state and its stop request in a trusted store."""

        if not isinstance(store, TrustedStateStore):
            raise ControllerError("selection state store must use its typed contract")
        return store.write(self.state, self.last_update.stop_request or self.stop_request or store.stop_request)

    def mark_completed(self) -> SelectionState:
        """Mark selection complete after the caller commits final retention."""

        if self._state.lifecycle is SelectionLifecycle.FAILED:
            raise CampaignCompletionError("failed selection cannot become completed")
        if self._state.lifecycle is SelectionLifecycle.COMPLETED:
            return self._state
        self._state = replace(self._state, lifecycle=SelectionLifecycle.COMPLETED)
        self._last_update = SelectionUpdate(self._state, (), None, False)
        return self._state

    def mark_failed(self, failure: SelectionFailure) -> SelectionState:
        """Record one typed terminal controller failure."""

        if not isinstance(failure, SelectionFailure):
            raise ControllerError("selection failure must use its typed contract")
        if self._state.lifecycle is SelectionLifecycle.COMPLETED:
            raise CampaignCompletionError("completed selection cannot become failed")
        if self._state.lifecycle is SelectionLifecycle.FAILED:
            if self._state.failure == failure:
                return self._state
            raise CampaignCompletionError("failed selection cannot change its failure")
        self._state = replace(
            self._state,
            revision=self._state.revision + 1,
            lifecycle=SelectionLifecycle.FAILED,
            failure=failure,
        )
        self._last_stop_request = None
        self._last_update = SelectionUpdate(self._state, (), None, False)
        return self._state

    def _matching_snapshot(
        self,
        result: ValidationResult,
        published_snapshot: PublishedSnapshot | Mapping[str, object] | None,
    ) -> PublishedSnapshot:
        if published_snapshot is not None:
            snapshot = _parse_snapshot(published_snapshot)
            known = self._snapshots.get(snapshot.slot_id)
            if known is not None and known != snapshot:
                raise ResultConflictError("published snapshot identity changed for one slot", slot_id=snapshot.slot_id)
            self._validate_snapshot(snapshot)
            self._snapshots[snapshot.slot_id] = snapshot
        else:
            snapshot = self._snapshots.get(result.slot_id)
            if snapshot is None:
                raise ResultImportError(
                    ResultImportFailureKind.UNKNOWN_SLOT,
                    "validation result has no matching published snapshot",
                    slot_id=result.slot_id,
                )
        return snapshot

    def _validate_snapshot(self, snapshot: PublishedSnapshot) -> ValidationSlot:
        if (
            snapshot.campaign_id != self.campaign.campaign_id
            or snapshot.training_launch_id != self.campaign.training_launch_id
        ):
            raise ResultImportError(
                ResultImportFailureKind.CAMPAIGN_MISMATCH,
                "published snapshot campaign identity does not match the campaign",
                slot_id=snapshot.slot_id,
            )
        slot = next((candidate for candidate in self.campaign.slots if candidate.slot_id == snapshot.slot_id), None)
        if slot is None:
            raise ResultImportError(
                ResultImportFailureKind.UNKNOWN_SLOT,
                "published snapshot slot is not in the campaign",
                slot_id=snapshot.slot_id,
            )
        if snapshot.updates != slot.updates:
            raise ResultImportError(
                ResultImportFailureKind.SNAPSHOT_MISMATCH,
                "published snapshot update identity does not match its slot",
                slot_id=snapshot.slot_id,
            )
        if snapshot.dev_bundle_digest != self.campaign.dev_bundle_digest:
            raise ResultImportError(
                ResultImportFailureKind.SNAPSHOT_MISMATCH,
                "published snapshot dev-bundle identity does not match the campaign",
                slot_id=snapshot.slot_id,
            )
        if snapshot.trainer_configuration_digest != self.campaign.trainer_configuration_digest:
            raise ResultImportError(
                ResultImportFailureKind.SNAPSHOT_MISMATCH,
                "published snapshot trainer configuration does not match the campaign",
                slot_id=snapshot.slot_id,
            )
        if snapshot.evaluator_image_identity != self.campaign.evaluator_image_identity:
            raise ResultImportError(
                ResultImportFailureKind.SNAPSHOT_MISMATCH,
                "published snapshot evaluator image does not match the campaign",
                slot_id=snapshot.slot_id,
            )
        if snapshot.evaluator_implementation_digest != self.campaign.evaluator_implementation_digest:
            raise ResultImportError(
                ResultImportFailureKind.SNAPSHOT_MISMATCH,
                "published snapshot evaluator implementation does not match the campaign",
                slot_id=snapshot.slot_id,
            )
        return slot

    def _validate_result(self, result: ValidationResult, snapshot: PublishedSnapshot) -> None:
        if (
            result.campaign_id != self.campaign.campaign_id
            or result.training_launch_id != self.campaign.training_launch_id
        ):
            raise ResultImportError(
                ResultImportFailureKind.CAMPAIGN_MISMATCH,
                "validation result campaign identity does not match the campaign",
                slot_id=result.slot_id,
            )
        slot = self._validate_snapshot(snapshot)
        if result.slot_id != slot.slot_id or result.updates != slot.updates:
            raise ResultImportError(
                ResultImportFailureKind.SNAPSHOT_MISMATCH,
                "validation result slot or update identity does not match its campaign slot",
                slot_id=result.slot_id,
            )
        if not result.matches_snapshot(snapshot):
            raise ResultImportError(
                ResultImportFailureKind.SNAPSHOT_MISMATCH,
                "validation result does not exactly match its published snapshot",
                slot_id=result.slot_id,
            )

    def _validate_state(self, state: SelectionState) -> None:
        if not isinstance(state, SelectionState):
            raise ControllerError("selection state must use its typed contract")
        if state.revision < 0:
            raise ControllerError("selection revision cannot be negative")
        results = dict(state.results_received_by_slot)
        for slot_id, result in results.items():
            if slot_id != result.slot_id:
                raise ControllerError("selection result key does not match its slot")
            slot = next((candidate for candidate in self.campaign.slots if candidate.slot_id == slot_id), None)
            if slot is None or slot.updates != result.updates:
                raise ControllerError("selection result slot does not belong to this campaign")
            if (
                result.campaign_id != self.campaign.campaign_id
                or result.training_launch_id != self.campaign.training_launch_id
                or result.dev_bundle_digest != self.campaign.dev_bundle_digest
                or result.trainer_configuration_digest != self.campaign.trainer_configuration_digest
                or result.evaluator_image_identity != self.campaign.evaluator_image_identity
                or result.evaluator_implementation_digest != self.campaign.evaluator_implementation_digest
            ):
                raise ControllerError("selection result identity does not belong to this campaign")
            snapshot = self._snapshots.get(slot_id)
            if snapshot is not None:
                self._validate_result(result, snapshot)
        projection = _selection_projection(self.campaign, self.policy, results)
        if (
            state.cursor != projection.cursor
            or state.best_score != projection.best_score
            or state.patience != projection.patience
        ):
            raise ControllerError("selection state projection does not match its received results")
        if state.top_five_snapshot_ids != projection.top_five_snapshot_ids:
            raise ControllerError("selection ranking does not match its received results")
        if state.lifecycle is SelectionLifecycle.READY and (state.revision != 0 or results):
            raise ControllerError("ready selection state cannot contain received results")
        if projection.first_patience_trigger is not None and state.lifecycle not in (
            SelectionLifecycle.STOP_REQUESTED,
            SelectionLifecycle.COMPLETED,
            SelectionLifecycle.FAILED,
        ):
            raise ControllerError("selection state must publish its patience stop request")

    def _validate_stop_request(self, state: SelectionState, stop_request: StopRequest) -> None:
        if stop_request.selection_revision > state.revision:
            raise ControllerError("stop request revision is newer than the selection state")
        if (
            stop_request.best_snapshot.campaign_id != self.campaign.campaign_id
            or stop_request.best_snapshot.training_launch_id != self.campaign.training_launch_id
        ):
            raise ControllerError("stop request best snapshot belongs to another campaign")
        received = dict(state.results_received_by_slot)
        if received.get(stop_request.triggering_result.slot_id) != stop_request.triggering_result:
            raise ControllerError("stop request trigger is not a trusted received result")
        if (
            stop_request.triggering_result.campaign_id != self.campaign.campaign_id
            or stop_request.triggering_result.training_launch_id != self.campaign.training_launch_id
        ):
            raise ControllerError("stop request trigger belongs to another campaign")
        best_result = next(
            (result for result in received.values() if result.snapshot_id == stop_request.best_snapshot.snapshot_id),
            None,
        )
        if best_result is None or best_result.snapshot != stop_request.best_snapshot:
            raise ControllerError("stop request best snapshot is not the trusted snapshot")
        expected = _derive_stop_request(self.campaign, self.policy, state)
        if (
            stop_request.triggering_result != expected.triggering_result
            or stop_request.best_snapshot != expected.best_snapshot
        ):
            raise ControllerError("stop request is not bound to the first patience expiration")

    def _fail_import(
        self,
        kind: ResultImportFailureKind,
        detail: str,
        *,
        slot_id: Sha256Digest | None,
    ) -> None:
        if self._state.lifecycle in (
            SelectionLifecycle.COMPLETED,
            SelectionLifecycle.FAILED,
        ):
            return
        if kind is ResultImportFailureKind.RESULT_CONFLICT:
            failure_kind = SelectionFailureKind.RESULT_CONFLICT
        else:
            failure_kind = SelectionFailureKind.RESULT_IDENTITY_CONFLICT
        failure = SelectionFailure(failure_kind, detail)
        self._state = replace(
            self._state,
            revision=self._state.revision + 1,
            lifecycle=SelectionLifecycle.FAILED,
            failure=failure,
        )
        self._last_stop_request = None
        self._last_update = SelectionUpdate(self._state, (), None, False)


class LagWaitReason(str, Enum):
    """Closed reasons why the trainer must wait before the next epoch."""

    EPOCH_ZERO_PENDING = "epoch_zero_pending"
    VALIDATION_LAG_EXCEEDED = "validation_lag_exceeded"


@dataclass(frozen=True)
class LagPermit:
    """A typed permission to start one exact next epoch."""

    next_epoch: int
    pending_complete_epochs: int

    def __post_init__(self) -> None:
        if isinstance(self.next_epoch, bool) or not isinstance(self.next_epoch, int) or self.next_epoch < 1:
            raise LagGateError("next epoch must be at least one")
        if isinstance(self.pending_complete_epochs, bool) or self.pending_complete_epochs < 0:
            raise LagGateError("pending complete epochs must be nonnegative")


@dataclass(frozen=True)
class LagWait:
    """A typed decision that blocks one exact next epoch."""

    next_epoch: int
    pending_complete_epochs: int
    reason: LagWaitReason

    def __post_init__(self) -> None:
        if isinstance(self.next_epoch, bool) or not isinstance(self.next_epoch, int) or self.next_epoch < 1:
            raise LagGateError("next epoch must be at least one")
        if isinstance(self.pending_complete_epochs, bool) or self.pending_complete_epochs < 0:
            raise LagGateError("pending complete epochs must be nonnegative")
        if not isinstance(self.reason, LagWaitReason):
            raise LagGateError("validation lag wait reason is not supported")


LagDecision: TypeAlias = LagPermit | LagWait


class ValidationLagGate:
    """Apply the exact one-epoch validation lag rule."""

    def __init__(self, campaign: ValidationCampaign) -> None:
        self.campaign = campaign
        self._slot_by_id = {slot.slot_id: slot for slot in campaign.slots}

    def largest_published_complete_epoch(
        self,
        published_snapshots: Mapping[object, object] | Iterable[object],
    ) -> int:
        """Return the greatest complete epoch represented by published snapshots."""

        snapshots = _snapshots_from_input(published_snapshots)
        largest = 0
        for snapshot in snapshots.values():
            slot = self._slot_by_id.get(snapshot.slot_id)
            if slot is None or snapshot.campaign_id != self.campaign.campaign_id:
                raise LagGateError("published snapshot is not part of this campaign")
            if (
                snapshot.training_launch_id != self.campaign.training_launch_id
                or snapshot.updates != slot.updates
                or snapshot.dev_bundle_digest != self.campaign.dev_bundle_digest
                or snapshot.trainer_configuration_digest != self.campaign.trainer_configuration_digest
                or snapshot.evaluator_image_identity != self.campaign.evaluator_image_identity
                or snapshot.evaluator_implementation_digest != self.campaign.evaluator_implementation_digest
            ):
                raise LagGateError("published snapshot identity does not match the campaign")
            if isinstance(slot.point, CompleteEpoch):
                largest = max(largest, slot.point.epoch)
        return largest

    @staticmethod
    def largest_contiguous_validated_complete_epoch(cursor: ContiguousValidationCursor) -> int:
        """Return the complete-epoch number represented by a cursor."""

        if isinstance(cursor, CompleteEpochValidatedCursor):
            return cursor.epoch
        if isinstance(cursor, (EpochZeroPendingCursor, EpochZeroValidatedCursor)):
            return 0
        raise LagGateError("unsupported contiguous validation cursor")

    def decide(
        self,
        selection: SelectionState | ContiguousValidationCursor | Mapping[object, object] | Iterable[object],
        published_snapshots: Mapping[object, object] | Iterable[object],
        next_epoch: int | None = None,
    ) -> LagDecision:
        """Return the permit/wait decision immediately before a new epoch."""

        if isinstance(selection, SelectionState):
            cursor = selection.cursor
        elif isinstance(selection, (EpochZeroPendingCursor, EpochZeroValidatedCursor, CompleteEpochValidatedCursor)):
            cursor = selection
        else:
            raise LagGateError("selection must use its typed cursor contract")
        if not isinstance(published_snapshots, (Mapping, Sequence)) and not isinstance(published_snapshots, Iterable):
            raise LagGateError("published snapshots must be a collection")
        published_epoch = self.largest_published_complete_epoch(published_snapshots)
        validated_epoch = self.largest_contiguous_validated_complete_epoch(cursor)
        if validated_epoch > published_epoch:
            raise LagGateError("validated complete epoch is ahead of published snapshots")
        pending = published_epoch - validated_epoch
        candidate_epoch = published_epoch + 1 if next_epoch is None else next_epoch
        if isinstance(candidate_epoch, bool) or not isinstance(candidate_epoch, int) or candidate_epoch < 1:
            raise LagGateError("next epoch must be at least one")
        if candidate_epoch != published_epoch + 1:
            raise LagGateError("next epoch does not follow the largest published complete epoch")
        if isinstance(cursor, EpochZeroPendingCursor):
            return LagWait(candidate_epoch, pending, LagWaitReason.EPOCH_ZERO_PENDING)
        if pending > self.campaign.maximum_validation_lag:
            return LagWait(candidate_epoch, pending, LagWaitReason.VALIDATION_LAG_EXCEEDED)
        return LagPermit(candidate_epoch, pending)


class CampaignLifecycle(str, Enum):
    """Closed lifecycle of the trusted validation campaign."""

    RUNNING = "running"
    WAITING_FOR_RESULTS = "waiting_for_results"
    TRAINER_TERMINAL = "trainer_terminal"
    DRAINING = "draining"
    COMPLETED = "completed"
    FAILED = "failed"


class QueueDrainState(str, Enum):
    """Queue and managed-pool drain state."""

    OPEN = "open"
    CANCEL_REQUESTED = "cancel_requested"
    DRAINED = "drained"


class RetentionState(str, Enum):
    """Final ranking and retention decision state."""

    PENDING = "pending"
    COMMITTED = "committed"


class BatchTrigger(str, Enum):
    """Reasons for constructing a queue batch."""

    INITIAL = "initial"
    HARD_LIFETIME = "hard_lifetime"
    RECOVERABLE_QUEUE_FAILURE = "recoverable_queue_failure"


class CampaignFailureKind(str, Enum):
    """Typed terminal campaign failure reasons."""

    PUBLICATION_CONFLICT = "publication_conflict"
    VALIDATION_FAILURE = "validation_failure"
    RESULT_CONFLICT = "result_conflict"
    INCOMPLETE = "incomplete"
    TRAINER_FAILURE = "trainer_failure"


@dataclass(frozen=True)
class CampaignFailure:
    """One durable terminal campaign failure."""

    kind: CampaignFailureKind
    detail: str
    slot_id: Sha256Digest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CampaignFailureKind):
            raise CampaignCompletionError("campaign failure kind is not supported")
        if not isinstance(self.detail, str) or not self.detail:
            raise CampaignCompletionError("campaign failure detail is required")
        if self.slot_id is not None and not isinstance(self.slot_id, Sha256Digest):
            raise CampaignCompletionError("campaign failure slot identity must be a typed digest")

    def to_dict(self) -> dict[str, object]:
        """Return the strict local failure representation."""

        value: dict[str, object] = {"kind": self.kind.value, "detail": self.detail}
        if self.slot_id is not None:
            value["slot_id"] = self.slot_id.value
        return value


@dataclass(frozen=True)
class ValidationSlotFailureState:
    """A published slot whose evaluator result failed terminally."""

    snapshot: PublishedSnapshot
    detail: str
    state: ClassVar[str] = "validation_failed"

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, PublishedSnapshot):
            raise CampaignCompletionError("validation failure must use a typed snapshot")
        if not isinstance(self.detail, str) or not self.detail:
            raise CampaignCompletionError("validation failure detail is required")

    def to_dict(self) -> dict[str, object]:
        """Return the strict representation of the local failure state."""

        return {"state": self.state, "snapshot": self.snapshot.to_dict(), "detail": self.detail}


CampaignSlotState: TypeAlias = ValidationSlotStateValue | ValidationSlotFailureState


@dataclass(frozen=True)
class ValidationBatch:
    """One immutable queue batch admission record."""

    ordinal: int
    slot_ids: tuple[Sha256Digest, ...]
    trigger: BatchTrigger
    worker_admissions: int
    cold_cache_admissions: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or not self.slot_ids
        ):
            raise CampaignCompletionError("validation batch must contain slots")
        if not isinstance(self.trigger, BatchTrigger):
            raise CampaignCompletionError("validation batch trigger is not supported")
        if (
            isinstance(self.worker_admissions, bool)
            or not isinstance(self.worker_admissions, int)
            or isinstance(self.cold_cache_admissions, bool)
            or not isinstance(self.cold_cache_admissions, int)
            or self.worker_admissions != 1
            or self.cold_cache_admissions != 1
        ):
            raise CampaignCompletionError("one batch must record one worker and one cold-cache admission")
        if not isinstance(self.slot_ids, tuple):
            raise CampaignCompletionError("validation batch slots must be an immutable tuple")
        if any(not isinstance(slot_id, Sha256Digest) for slot_id in self.slot_ids):
            raise CampaignCompletionError("validation batch slot identities must be typed digests")
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise CampaignCompletionError("validation batch slot identities must be unique")

    def to_dict(self) -> dict[str, object]:
        """Return the local queue-batch representation."""

        return {
            "ordinal": self.ordinal,
            "slot_ids": [slot_id.value for slot_id in self.slot_ids],
            "trigger": self.trigger.value,
            "worker_admissions": self.worker_admissions,
            "cold_cache_admissions": self.cold_cache_admissions,
        }


@dataclass(frozen=True)
class RankingRetentionCommit:
    """One retention receipt bound to one trusted ranking revision."""

    selection_revision: int
    snapshot_ids: tuple[Sha256Digest, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.selection_revision, bool)
            or not isinstance(self.selection_revision, int)
            or self.selection_revision < 0
        ):
            raise CampaignCompletionError("retention selection revision must be nonnegative")
        if not isinstance(self.snapshot_ids, tuple) or len(self.snapshot_ids) > 5:
            raise CampaignCompletionError("retention ranking must contain at most five snapshots")
        if any(not isinstance(snapshot_id, Sha256Digest) for snapshot_id in self.snapshot_ids):
            raise CampaignCompletionError("retention snapshot identities must be typed digests")
        if len(set(self.snapshot_ids)) != len(self.snapshot_ids):
            raise CampaignCompletionError("retention snapshot identities must be unique")

    def to_dict(self) -> dict[str, object]:
        """Return the strict local retention receipt."""

        return {
            "selection_revision": self.selection_revision,
            "snapshot_ids": [snapshot_id.value for snapshot_id in self.snapshot_ids],
        }


@dataclass(frozen=True)
class CampaignCompletionState:
    """Immutable state required before a campaign can be completed."""

    campaign_id: Sha256Digest
    slot_states: tuple[tuple[Sha256Digest, CampaignSlotState], ...]
    training_completion: TrainingCompletion | None
    queue_state: QueueDrainState
    retention_state: RetentionState
    retention_commit: RankingRetentionCommit | None
    batches: tuple[ValidationBatch, ...]
    lifecycle: CampaignLifecycle
    failure: CampaignFailure | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, Sha256Digest):
            raise CampaignCompletionError("campaign completion identity must be a typed digest")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise CampaignCompletionError("campaign completion revision must be nonnegative")
        if not isinstance(self.slot_states, tuple) or not self.slot_states:
            raise CampaignCompletionError("campaign completion must contain slot states")
        slot_ids: set[Sha256Digest] = set()
        for entry in self.slot_states:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise CampaignCompletionError("campaign slot state entries must be typed pairs")
            slot_id, slot_state = entry
            if not isinstance(slot_id, Sha256Digest):
                raise CampaignCompletionError("campaign slot identity must be a typed digest")
            if slot_id in slot_ids:
                raise CampaignCompletionError("campaign slot identities must be unique")
            if not isinstance(
                slot_state,
                (
                    PlannedSlotState,
                    PublishedSlotState,
                    ValidatedSlotState,
                    UnusedSlotState,
                    ValidationSlotFailureState,
                ),
            ):
                raise CampaignCompletionError("campaign slot state is not supported")
            slot_ids.add(slot_id)
        if not isinstance(self.queue_state, QueueDrainState):
            raise CampaignCompletionError("queue drain state is not supported")
        if not isinstance(self.retention_state, RetentionState):
            raise CampaignCompletionError("retention state is not supported")
        if self.retention_commit is not None and not isinstance(self.retention_commit, RankingRetentionCommit):
            raise CampaignCompletionError("retention commit must use its typed contract")
        if (self.retention_state is RetentionState.COMMITTED) != (self.retention_commit is not None):
            raise CampaignCompletionError("retention state must match its commit")
        if not isinstance(self.lifecycle, CampaignLifecycle):
            raise CampaignCompletionError("campaign lifecycle is not supported")
        if self.training_completion is not None and not isinstance(
            self.training_completion, (MaxUpdatesReached, AcceptedEarlyStop)
        ):
            raise CampaignCompletionError("training completion must use its typed contract")
        if self.failure is not None and not isinstance(self.failure, CampaignFailure):
            raise CampaignCompletionError("campaign failure must use its typed contract")
        if (self.lifecycle is CampaignLifecycle.FAILED) != (self.failure is not None):
            raise CampaignCompletionError("campaign failure must match the failed lifecycle")
        if self.training_completion is None:
            if self.queue_state is not QueueDrainState.OPEN:
                raise CampaignCompletionError("queue cannot drain before trainer terminal progress")
            if self.retention_state is not RetentionState.PENDING:
                raise CampaignCompletionError("retention cannot commit before trainer terminal progress")
            if self.lifecycle in (
                CampaignLifecycle.TRAINER_TERMINAL,
                CampaignLifecycle.DRAINING,
                CampaignLifecycle.COMPLETED,
            ):
                raise CampaignCompletionError("terminal lifecycle requires trainer terminal progress")
        else:
            if self.lifecycle in (CampaignLifecycle.RUNNING, CampaignLifecycle.WAITING_FOR_RESULTS):
                raise CampaignCompletionError("trainer terminal progress requires a terminal lifecycle")
        if self.queue_state is QueueDrainState.OPEN and self.lifecycle is CampaignLifecycle.DRAINING:
            raise CampaignCompletionError("draining lifecycle requires queue cancellation")
        if self.queue_state is QueueDrainState.CANCEL_REQUESTED and self.lifecycle not in (
            CampaignLifecycle.DRAINING,
            CampaignLifecycle.FAILED,
        ):
            raise CampaignCompletionError("queue cancellation requires draining or failed lifecycle")
        if self.queue_state is QueueDrainState.DRAINED and self.lifecycle not in (
            CampaignLifecycle.DRAINING,
            CampaignLifecycle.COMPLETED,
            CampaignLifecycle.FAILED,
        ):
            raise CampaignCompletionError("queue drain requires draining, completed, or failed lifecycle")
        if self.lifecycle is CampaignLifecycle.COMPLETED:
            if self.queue_state is not QueueDrainState.DRAINED:
                raise CampaignCompletionError("completed campaign requires a drained queue")
            if self.retention_state is not RetentionState.COMMITTED:
                raise CampaignCompletionError("completed campaign requires committed retention")
            if any(not isinstance(state, (ValidatedSlotState, UnusedSlotState)) for _, state in self.slot_states):
                raise CampaignCompletionError("completed campaign requires terminal slot states")
        if not isinstance(self.batches, tuple) or not self.batches:
            raise CampaignCompletionError("campaign completion requires an initial batch")
        if any(not isinstance(batch, ValidationBatch) for batch in self.batches):
            raise CampaignCompletionError("campaign batches must use their typed contract")
        if tuple(batch.ordinal for batch in self.batches) != tuple(range(len(self.batches))):
            raise CampaignCompletionError("campaign batch ordinals must be contiguous")
        if self.batches[0].trigger is not BatchTrigger.INITIAL:
            raise CampaignCompletionError("campaign must start with an initial batch")
        if any(batch.trigger is BatchTrigger.INITIAL for batch in self.batches[1:]):
            raise CampaignCompletionError("only the first batch may be initial")

    @property
    def extra_worker_admissions(self) -> int:
        """Return worker admissions after the initial queue batch."""

        return max(0, sum(batch.worker_admissions for batch in self.batches) - 1)

    @property
    def extra_cold_cache_admissions(self) -> int:
        """Return cold-cache admissions after the initial queue batch."""

        return max(0, sum(batch.cold_cache_admissions for batch in self.batches) - 1)

    def to_dict(self) -> dict[str, object]:
        """Return the local campaign-completion representation."""

        return {
            "campaign_id": self.campaign_id.value,
            "slot_states": [
                {"slot_id": slot_id.value, "state": slot_state.to_dict()} for slot_id, slot_state in self.slot_states
            ],
            "training_completion": (
                self.training_completion.to_dict() if self.training_completion is not None else None
            ),
            "queue_state": self.queue_state.value,
            "retention_state": self.retention_state.value,
            "retention_commit": self.retention_commit.to_dict() if self.retention_commit is not None else None,
            "batches": [batch.to_dict() for batch in self.batches],
            "lifecycle": self.lifecycle.value,
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "revision": self.revision,
        }


def _initial_campaign_state(campaign: ValidationCampaign) -> CampaignCompletionState:
    initial_batch = ValidationBatch(
        ordinal=0,
        slot_ids=tuple(slot.slot_id for slot in campaign.slots),
        trigger=BatchTrigger.INITIAL,
        worker_admissions=1,
        cold_cache_admissions=1,
    )
    return CampaignCompletionState(
        campaign_id=campaign.campaign_id,
        slot_states=tuple((slot.slot_id, PlannedSlotState()) for slot in campaign.slots),
        training_completion=None,
        queue_state=QueueDrainState.OPEN,
        retention_state=RetentionState.PENDING,
        retention_commit=None,
        batches=(initial_batch,),
        lifecycle=CampaignLifecycle.RUNNING,
        revision=0,
    )


class CampaignCompletionController:
    """Own slot terminalization, queue drain, retention, and completion."""

    @classmethod
    def initial(
        cls,
        campaign: ValidationCampaign,
        policy: SelectionPolicy,
        published_snapshots: Mapping[object, object] | Iterable[object] = (),
    ) -> CampaignCompletionController:
        """Build a completion controller with all slots planned."""

        return cls(campaign, policy, published_snapshots)

    def __init__(
        self,
        campaign: ValidationCampaign,
        policy: SelectionPolicy,
        published_snapshots: Mapping[object, object] | Iterable[object] = (),
    ) -> None:
        if not isinstance(campaign, ValidationCampaign):
            raise CampaignCompletionError("campaign must use its typed contract")
        if not isinstance(policy, SelectionPolicy):
            raise CampaignCompletionError("selection policy must use its typed contract")
        self.campaign = campaign
        snapshots = _snapshots_from_input(published_snapshots)
        self.selection = SelectionController(campaign, policy, snapshots)
        self._state = _initial_campaign_state(campaign)

        for snapshot in snapshots.values():
            self.publish_snapshot(snapshot)

    @property
    def state(self) -> CampaignCompletionState:
        """Return current campaign-completion state."""

        return self._state

    @property
    def selection_state(self) -> SelectionState:
        """Return current trusted selection state."""

        return self.selection.state

    def publish_snapshot(self, snapshot: PublishedSnapshot | Mapping[str, object]) -> CampaignCompletionState:
        """Transition one planned slot to its immutable published snapshot."""

        parsed = _parse_snapshot(snapshot)
        current = dict(self._state.slot_states).get(parsed.slot_id)
        if current is None:
            raise CampaignCompletionError("snapshot slot is not in the campaign")
        if self._state.lifecycle is CampaignLifecycle.FAILED:
            if isinstance(current, (PublishedSlotState, ValidatedSlotState, ValidationSlotFailureState)) and (
                current.snapshot == parsed
            ):
                return self._state
            raise CampaignCompletionError("failed campaign cannot change published snapshots")
        if self._state.lifecycle is CampaignLifecycle.COMPLETED:
            if isinstance(current, (PublishedSlotState, ValidatedSlotState, ValidationSlotFailureState)) and (
                current.snapshot == parsed
            ):
                return self._state
            raise CampaignCompletionError("completed campaign cannot change published snapshots")
        self.selection.register_snapshot(parsed)
        if isinstance(current, PlannedSlotState):
            states = dict(self._state.slot_states)
            states[parsed.slot_id] = PublishedSlotState(parsed)
            self._state = replace(
                self._state,
                slot_states=tuple(states.items()),
                lifecycle=CampaignLifecycle.WAITING_FOR_RESULTS,
                revision=self._state.revision + 1,
            )
            return self._state
        if isinstance(current, PublishedSlotState) and current.snapshot == parsed:
            return self._state
        if isinstance(current, ValidatedSlotState) and current.snapshot == parsed:
            return self._state
        if isinstance(current, ValidationSlotFailureState) and current.snapshot == parsed:
            return self._state
        self._fail_campaign(
            CampaignFailureKind.PUBLICATION_CONFLICT,
            "one validation slot cannot accept a different published snapshot",
            parsed.slot_id,
        )
        raise ResultConflictError(
            "one validation slot cannot accept a different published snapshot",
            slot_id=parsed.slot_id,
        )

    def import_result(
        self,
        compact_result: Mapping[str, object],
        published_snapshot: PublishedSnapshot | Mapping[str, object] | None = None,
    ) -> CampaignCompletionState:
        """Import a result and transition its published slot to validated."""

        try:
            result = ValidationResult.from_dict(compact_result)
        except (ValidationContractError, TypeError, ValueError) as error:
            if self._state.lifecycle not in (CampaignLifecycle.FAILED, CampaignLifecycle.COMPLETED):
                self._fail_campaign(CampaignFailureKind.VALIDATION_FAILURE, str(error))
            raise ResultImportError(ResultImportFailureKind.INVALID_DOCUMENT, str(error)) from error
        current = dict(self._state.slot_states).get(result.slot_id)
        if current is None:
            error = ResultImportError(
                ResultImportFailureKind.UNKNOWN_SLOT, "result slot is not in the campaign", slot_id=result.slot_id
            )
            if self._state.lifecycle not in (CampaignLifecycle.FAILED, CampaignLifecycle.COMPLETED):
                self._fail_campaign(CampaignFailureKind.VALIDATION_FAILURE, str(error), result.slot_id)
            raise error
        if isinstance(current, PlannedSlotState):
            error = ResultImportError(
                ResultImportFailureKind.SNAPSHOT_MISMATCH,
                "a validation result requires a published slot",
                slot_id=result.slot_id,
            )
            if self._state.lifecycle not in (CampaignLifecycle.FAILED, CampaignLifecycle.COMPLETED):
                self._fail_campaign(CampaignFailureKind.VALIDATION_FAILURE, str(error), result.slot_id)
            raise error
        if isinstance(current, PublishedSlotState):
            snapshot = current.snapshot
        elif isinstance(current, ValidatedSlotState):
            snapshot = current.snapshot
        elif isinstance(current, ValidationSlotFailureState):
            raise ResultImportError(
                ResultImportFailureKind.CONTROLLER_TERMINAL,
                "a terminally failed validation slot cannot accept a result",
                slot_id=result.slot_id,
            )
        else:
            raise CampaignCompletionError("unsupported campaign slot state")
        if published_snapshot is not None and _parse_snapshot(published_snapshot) != snapshot:
            error = ResultConflictError(
                "result was delivered for a different published snapshot", slot_id=result.slot_id
            )
            if self._state.lifecycle not in (CampaignLifecycle.FAILED, CampaignLifecycle.COMPLETED):
                self._fail_campaign(CampaignFailureKind.RESULT_CONFLICT, str(error), result.slot_id)
            raise error
        try:
            self.selection.import_result(compact_result, snapshot)
        except ResultImportError as error:
            if self._state.lifecycle in (CampaignLifecycle.FAILED, CampaignLifecycle.COMPLETED) or (
                self.selection.state.lifecycle is SelectionLifecycle.STOP_REQUESTED
                and error.kind is ResultImportFailureKind.CONTROLLER_TERMINAL
            ):
                raise
            kind = (
                CampaignFailureKind.RESULT_CONFLICT
                if error.kind is ResultImportFailureKind.RESULT_CONFLICT
                else CampaignFailureKind.VALIDATION_FAILURE
            )
            self._fail_campaign(kind, str(error), result.slot_id)
            raise
        if isinstance(current, PublishedSlotState):
            states = dict(self._state.slot_states)
            states[result.slot_id] = ValidatedSlotState(snapshot, result)
            self._state = replace(
                self._state,
                slot_states=tuple(states.items()),
                retention_state=RetentionState.PENDING,
                retention_commit=None,
                revision=self._state.revision + 1,
            )
        return self._state

    def mark_validation_failure(self, slot_id: Sha256Digest | str, detail: str) -> CampaignCompletionState:
        """Record a terminal failure for one published validation slot."""

        if self._state.lifecycle is CampaignLifecycle.FAILED:
            return self._state
        if self._state.lifecycle is CampaignLifecycle.COMPLETED:
            raise CampaignCompletionError("completed campaign cannot record validation failure")
        parsed_slot_id = _parse_slot_id(slot_id)
        current = dict(self._state.slot_states).get(parsed_slot_id)
        if not isinstance(current, (PublishedSlotState, ValidatedSlotState)):
            raise CampaignCompletionError("only a published slot can fail validation")
        failed = ValidationSlotFailureState(current.snapshot, detail)
        states = dict(self._state.slot_states)
        states[parsed_slot_id] = failed
        self.selection.mark_failed(SelectionFailure(SelectionFailureKind.CONTROLLER_FAILURE, detail))
        self._state = replace(
            self._state,
            slot_states=tuple(states.items()),
            lifecycle=CampaignLifecycle.FAILED,
            failure=CampaignFailure(CampaignFailureKind.VALIDATION_FAILURE, detail, parsed_slot_id),
            retention_state=RetentionState.PENDING,
            retention_commit=None,
            revision=self._state.revision + 1,
        )
        return self._state

    def mark_trainer_terminal(
        self,
        completion: TrainingCompletion,
    ) -> CampaignCompletionState:
        """Record terminal trainer progress and make early-stop slots unused."""

        if self._state.lifecycle is CampaignLifecycle.FAILED:
            return self._state
        if self._state.lifecycle is CampaignLifecycle.COMPLETED:
            if self._state.training_completion is not None and self._state.training_completion == completion:
                return self._state
            raise CampaignCompletionError("completed campaign cannot change trainer completion")
        if isinstance(completion, MaxUpdatesReached):
            is_early_stop = False
        elif isinstance(completion, AcceptedEarlyStop):
            is_early_stop = True
            self._validate_early_stop(completion)
        else:
            raise CampaignCompletionError("unsupported trainer completion outcome")
        if self._state.training_completion is not None:
            if self._state.training_completion == completion:
                return self._state
            raise CampaignCompletionError("trainer terminal progress cannot change")
        states = dict(self._state.slot_states)
        if is_early_stop:
            for slot_id, slot_state in tuple(states.items()):
                if isinstance(slot_state, PlannedSlotState):
                    states[slot_id] = UnusedSlotState(ValidationSlotTerminalReason.ACCEPTED_EARLY_STOP)
        self._state = replace(
            self._state,
            slot_states=tuple(states.items()),
            training_completion=completion,
            lifecycle=CampaignLifecycle.TRAINER_TERMINAL,
            revision=self._state.revision + 1,
        )
        return self._state

    def cancel_queue(self) -> CampaignCompletionState:
        """Request cancellation of pending and leased unused queue units."""

        if self._state.lifecycle in (CampaignLifecycle.FAILED, CampaignLifecycle.COMPLETED):
            return self._state
        if self._state.training_completion is None:
            raise CampaignCompletionError("queue cancellation requires durable trainer terminal progress")
        if self._state.queue_state is QueueDrainState.DRAINED:
            return self._state
        self._state = replace(
            self._state,
            queue_state=QueueDrainState.CANCEL_REQUESTED,
            lifecycle=CampaignLifecycle.DRAINING,
            revision=self._state.revision + 1,
        )
        return self._state

    def mark_queue_drained(self) -> CampaignCompletionState:
        """Record durable queue and worker-pool drain."""

        if self._state.lifecycle in (CampaignLifecycle.FAILED, CampaignLifecycle.COMPLETED):
            return self._state
        if self._state.queue_state is QueueDrainState.OPEN:
            raise CampaignCompletionError("queue cancellation must be requested before drain")
        if self._state.queue_state is QueueDrainState.DRAINED:
            return self._state
        self._state = replace(
            self._state,
            queue_state=QueueDrainState.DRAINED,
            revision=self._state.revision + 1,
        )
        return self._state

    def commit_ranking_retention(
        self,
        retained_snapshot_ids: Sequence[Sha256Digest | str] | None = None,
    ) -> CampaignCompletionState:
        """Commit retention only for the controller's current deterministic ranking."""

        if self._state.lifecycle in (CampaignLifecycle.FAILED, CampaignLifecycle.COMPLETED):
            return self._state
        if self._state.training_completion is None:
            raise CampaignCompletionError("ranking retention requires durable trainer terminal progress")
        expected = self.selection.state.top_five_snapshot_ids
        if retained_snapshot_ids is not None:
            actual = tuple(_parse_slot_id(item, "retained_snapshot_id") for item in retained_snapshot_ids)
            if actual != expected:
                raise CampaignCompletionError("ranking retention does not match the trusted top-five ranking")
        commit = RankingRetentionCommit(self.selection.state.revision, expected)
        if self._state.retention_commit == commit:
            return self._state
        self._state = replace(
            self._state,
            retention_state=RetentionState.COMMITTED,
            retention_commit=commit,
            revision=self._state.revision + 1,
        )
        return self._state

    def ready_to_complete(self) -> bool:
        """Return whether all terminal campaign invariants hold."""

        if self._state.lifecycle is CampaignLifecycle.FAILED or self._state.training_completion is None:
            return False
        if self._state.queue_state is not QueueDrainState.DRAINED:
            return False
        if self._state.retention_state is not RetentionState.COMMITTED:
            return False
        if self._state.retention_commit is None:
            return False
        if (
            self._state.retention_commit.selection_revision != self.selection.state.revision
            or self._state.retention_commit.snapshot_ids != self.selection.state.top_five_snapshot_ids
        ):
            return False
        return all(isinstance(state, (ValidatedSlotState, UnusedSlotState)) for _, state in self._state.slot_states)

    def complete(self) -> CampaignCompletionState:
        """Publish completed only after results, retention, and queue drain."""

        if self._state.lifecycle in (CampaignLifecycle.FAILED, CampaignLifecycle.COMPLETED):
            return self._state
        if self.selection.state.lifecycle is SelectionLifecycle.FAILED:
            self._fail_campaign(CampaignFailureKind.VALIDATION_FAILURE, "selection controller is failed")
            return self._state
        if not self.ready_to_complete():
            raise CampaignCompletionError("campaign completion prerequisites are not durable")
        self._state = replace(
            self._state,
            lifecycle=CampaignLifecycle.COMPLETED,
            revision=self._state.revision + 1,
        )
        self.selection.mark_completed()
        return self._state

    def try_complete(self) -> bool:
        """Complete when possible and return whether completion was published."""

        if not self.ready_to_complete():
            return False
        self.complete()
        return True

    def persist(self, store: CampaignCompletionStateStore) -> CampaignCompletionState:
        """Persist the current campaign lifecycle and queue state."""

        if not isinstance(store, CampaignCompletionStateStore):
            raise CampaignCompletionError("campaign state store must use its typed contract")
        if store.campaign != self.campaign:
            raise CampaignCompletionError("campaign state store belongs to another campaign")
        return store.write(self._state)

    @classmethod
    def restore(
        cls,
        campaign: ValidationCampaign,
        policy: SelectionPolicy,
        store: CampaignCompletionStateStore,
        selection: SelectionController,
    ) -> CampaignCompletionController:
        """Restore campaign lifecycle state with a matching selection controller."""

        if not isinstance(store, CampaignCompletionStateStore):
            raise CampaignCompletionError("campaign state store must use its typed contract")
        if not isinstance(selection, SelectionController) or selection.campaign != campaign:
            raise CampaignCompletionError("selection controller belongs to another campaign")
        state = store.load()
        snapshots = selection.published_snapshots
        controller = cls(campaign, policy, snapshots)
        controller.selection = selection
        controller._state = state
        controller._validate_restored_state()
        return controller

    def create_additional_batch(
        self,
        trigger: BatchTrigger,
        unresolved_slot_ids: Sequence[Sha256Digest | str] | None = None,
    ) -> ValidationBatch:
        """Create one hard-lifetime/recoverable batch for unresolved slots."""

        if self._state.lifecycle in (CampaignLifecycle.COMPLETED, CampaignLifecycle.FAILED):
            raise CampaignCompletionError("terminal campaign cannot create another validation batch")
        if trigger is BatchTrigger.INITIAL:
            raise CampaignCompletionError("the initial queue batch already exists")
        unresolved = tuple(
            slot_id
            for slot_id, state in self._state.slot_states
            if not isinstance(state, (ValidatedSlotState, UnusedSlotState, ValidationSlotFailureState))
        )
        expected = tuple(unresolved)
        if unresolved_slot_ids is not None:
            actual = tuple(_parse_slot_id(item, "unresolved_slot_id") for item in unresolved_slot_ids)
            if actual != expected:
                raise CampaignCompletionError("additional batch must contain exactly unresolved slots")
        if not expected:
            raise CampaignCompletionError("no unresolved slots require another batch")
        batch = ValidationBatch(
            ordinal=len(self._state.batches),
            slot_ids=expected,
            trigger=trigger,
            worker_admissions=1,
            cold_cache_admissions=1,
        )
        self._state = replace(
            self._state,
            batches=(*self._state.batches, batch),
            revision=self._state.revision + 1,
        )
        return batch

    def _validate_early_stop(self, completion: AcceptedEarlyStop) -> None:
        if (
            completion.triggering_result.campaign_id != self.campaign.campaign_id
            or completion.triggering_result.training_launch_id != self.campaign.training_launch_id
        ):
            raise CampaignCompletionError("early-stop result belongs to another campaign")
        if (
            completion.best_snapshot.campaign_id != self.campaign.campaign_id
            or completion.best_snapshot.training_launch_id != self.campaign.training_launch_id
        ):
            raise CampaignCompletionError("early-stop best snapshot belongs to another campaign")
        stop_request = self.selection.stop_request
        if stop_request is None:
            raise CampaignCompletionError("early-stop completion requires a trusted stop request")
        if completion.selection_revision != stop_request.selection_revision:
            raise CampaignCompletionError("early-stop revision is not the trusted stop revision")
        if completion.triggering_result != stop_request.triggering_result:
            raise CampaignCompletionError("early-stop trigger is not the trusted stop trigger")
        if completion.best_snapshot != stop_request.best_snapshot:
            raise CampaignCompletionError("early-stop best snapshot is not the trusted stop snapshot")
        known = dict(self.selection.state.results_received_by_slot).get(completion.triggering_result.slot_id)
        if known != completion.triggering_result:
            raise CampaignCompletionError("early-stop trigger is not a trusted received result")

    def _fail_campaign(
        self,
        kind: CampaignFailureKind,
        detail: str,
        slot_id: Sha256Digest | None = None,
    ) -> CampaignCompletionState:
        self._state = replace(
            self._state,
            lifecycle=CampaignLifecycle.FAILED,
            failure=CampaignFailure(kind, detail, slot_id),
            revision=self._state.revision + 1,
        )
        return self._state

    def _validate_restored_state(self) -> None:
        if self._state.campaign_id != self.campaign.campaign_id:
            raise CampaignCompletionError("campaign state belongs to another campaign")
        states = dict(self._state.slot_states)
        if tuple(states) != tuple(slot.slot_id for slot in self.campaign.slots):
            raise CampaignCompletionError("restored campaign slots do not match the campaign")
        selection_results = dict(self.selection.state.results_received_by_slot)
        for slot_id, slot_state in self._state.slot_states:
            if isinstance(slot_state, (PublishedSlotState, ValidatedSlotState, ValidationSlotFailureState)):
                self.selection.register_snapshot(slot_state.snapshot)
            if isinstance(slot_state, ValidatedSlotState) and selection_results.get(slot_id) != slot_state.result:
                raise CampaignCompletionError("restored validated slot does not match selection state")
        if self._state.retention_commit is not None and (
            self._state.retention_commit.selection_revision != self.selection.state.revision
            or self._state.retention_commit.snapshot_ids != self.selection.state.top_five_snapshot_ids
        ):
            self._state = replace(
                self._state,
                retention_state=RetentionState.PENDING,
                retention_commit=None,
                revision=self._state.revision + 1,
            )


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class TrustedStateStore:
    """Persist selection states with a digest-bound monotonic revision."""

    def __init__(
        self,
        path: str | Path,
        campaign: ValidationCampaign,
        policy: SelectionPolicy,
        initial_state: SelectionState | None = None,
    ) -> None:
        if not isinstance(campaign, ValidationCampaign):
            raise TrustedStateError("campaign must use its typed contract")
        if not isinstance(policy, SelectionPolicy):
            raise TrustedStateError("selection policy must use its typed contract")
        if initial_state is not None and not isinstance(initial_state, SelectionState):
            raise TrustedStateError("initial selection state must use its typed contract")
        self.path = Path(path)
        self.campaign = campaign
        self.policy = policy
        self.initial_state = initial_state if initial_state is not None else initial_selection_state()
        self._stop_request: StopRequest | None = None
        self._validate_campaign_state(self.initial_state)

    @property
    def stop_request(self) -> StopRequest | None:
        """Return the durable stop request from the last verified envelope."""

        if self.path.exists() and self._stop_request is None:
            self.load()
        return self._stop_request

    def load(self) -> SelectionState:
        """Read and verify the current state, or return revision-zero state."""

        if not self.path.exists():
            self._validate_campaign_state(self.initial_state)
            self._stop_request = None
            return self.initial_state
        try:
            encoded = self.path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise TrustedStateError("trusted selection state cannot be decoded") from error
        if not isinstance(raw, Mapping):
            raise TrustedStateError("trusted selection state must be an object")
        required = frozenset({"schema", "campaign_id", "revision", "state", "stop_request", "digest"})
        if frozenset(raw) != required:
            raise TrustedStateError("trusted selection state fields are not exact")
        if raw["schema"] != TRUSTED_STATE_SCHEMA or raw["campaign_id"] != self.campaign.campaign_id.value:
            raise TrustedStateError("trusted selection state schema or campaign does not match")
        state = SelectionState.from_dict(raw["state"])
        self._validate_campaign_state(state)
        if raw["stop_request"] is None:
            stop_request = None
        else:
            if not isinstance(raw["stop_request"], Mapping):
                raise TrustedStateError("trusted stop request must be an object or null")
            if frozenset(raw["stop_request"]) != frozenset(
                {"selection_revision", "triggering_result", "best_snapshot"}
            ):
                raise TrustedStateError("trusted stop request fields are not exact")
            try:
                stop_request = StopRequest(
                    selection_revision=raw["stop_request"]["selection_revision"],
                    triggering_result=ValidationResult.from_dict(raw["stop_request"]["triggering_result"]),
                    best_snapshot=PublishedSnapshot.from_dict(raw["stop_request"]["best_snapshot"]),
                )
            except (KeyError, TypeError, ValueError, ValidationContractError) as error:
                raise TrustedStateError("trusted stop request is invalid") from error
            self._validate_stop_request(state, stop_request)
        if state.lifecycle is SelectionLifecycle.STOP_REQUESTED and stop_request is None:
            raise TrustedStateError("stop-requested state requires its durable stop request")
        if (
            state.lifecycle not in (SelectionLifecycle.STOP_REQUESTED, SelectionLifecycle.COMPLETED)
            and stop_request is not None
        ):
            raise TrustedStateError("a stop request requires a stop-requested state")
        self._stop_request = stop_request
        if raw["revision"] != state.revision:
            raise TrustedStateError("trusted selection revision does not match its state")
        body = {
            "schema": TRUSTED_STATE_SCHEMA,
            "campaign_id": self.campaign.campaign_id.value,
            "revision": state.revision,
            "state": state.to_dict(),
            "stop_request": stop_request.to_dict() if stop_request is not None else None,
        }
        expected = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        if raw["digest"] != expected:
            raise TrustedStateError("trusted selection state digest does not match its content")
        if encoded != _canonical_bytes({**body, "digest": expected}):
            raise TrustedStateError("trusted selection state is not canonical JSON")
        return state

    def write(self, state: SelectionState, stop_request: StopRequest | None = None) -> SelectionState:
        """Atomically publish a newer state and its canonical content digest."""

        if not isinstance(state, SelectionState):
            raise TrustedStateError("trusted selection state must use its typed contract")
        self._validate_campaign_state(state)
        if stop_request is None and state.lifecycle is SelectionLifecycle.STOP_REQUESTED:
            try:
                stop_request = _derive_stop_request(self.campaign, self.policy, state)
            except ControllerError as error:
                raise TrustedStateError(str(error)) from error
        elif (
            stop_request is None and state.lifecycle is SelectionLifecycle.COMPLETED and self._stop_request is not None
        ):
            stop_request = self._stop_request
        if (
            state.lifecycle not in (SelectionLifecycle.STOP_REQUESTED, SelectionLifecycle.COMPLETED)
            and stop_request is not None
        ):
            raise TrustedStateError("a stop request requires a stop-requested state")
        if stop_request is not None:
            self._validate_stop_request(state, stop_request)
        path_existed = self.path.exists()
        current = self.load()
        if state.revision < current.revision:
            raise RevisionConflictError("trusted selection revision is older than the stored revision")
        if state.revision == current.revision:
            if state == current and stop_request == self._stop_request and path_existed:
                return current
            if path_existed:
                raise RevisionConflictError("trusted selection revision reused for different content")
        body = {
            "schema": TRUSTED_STATE_SCHEMA,
            "campaign_id": self.campaign.campaign_id.value,
            "revision": state.revision,
            "state": state.to_dict(),
            "stop_request": stop_request.to_dict() if stop_request is not None else None,
        }
        envelope = {**body, "digest": hashlib.sha256(_canonical_bytes(body)).hexdigest()}
        encoded = _canonical_bytes(envelope)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as error:
            raise TrustedStateError("trusted selection state replacement failed") from error
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        self._stop_request = stop_request
        return state

    def write_update(self, update: SelectionUpdate) -> SelectionState:
        """Persist a controller update, including its one stop request."""

        return self.write(update.state, update.stop_request)

    def load_update(self) -> SelectionUpdate:
        """Load the durable state and its stop request as one update."""

        state = self.load()
        return SelectionUpdate(state, (), self._stop_request, True)

    def accept_newer_revision(self, last_revision: int) -> SelectionState:
        """Return only a valid state newer than the trainer's last revision."""

        state = self.load()
        if isinstance(last_revision, bool) or not isinstance(last_revision, int) or last_revision < 0:
            raise TrustedStateError("last trainer revision must be a nonnegative integer")
        if state.revision <= last_revision:
            raise RevisionConflictError("trusted selection state is not newer than the trainer revision")
        return state

    def _validate_campaign_state(self, state: SelectionState) -> None:
        """Check the campaign identities carried by every received result."""

        if not isinstance(state, SelectionState):
            raise TrustedStateError("trusted selection state must use its typed contract")
        if state.revision < 0:
            raise TrustedStateError("trusted selection revision cannot be negative")
        received = dict(state.results_received_by_slot)
        for _, result in state.results_received_by_slot:
            if (
                result.campaign_id != self.campaign.campaign_id
                or result.training_launch_id != self.campaign.training_launch_id
            ):
                raise TrustedStateError("trusted selection result belongs to another campaign")
            slot = next((item for item in self.campaign.slots if item.slot_id == result.slot_id), None)
            if slot is None or slot.updates != result.updates:
                raise TrustedStateError("trusted selection result slot does not belong to its campaign")
            if (
                result.dev_bundle_digest != self.campaign.dev_bundle_digest
                or result.trainer_configuration_digest != self.campaign.trainer_configuration_digest
                or result.evaluator_image_identity != self.campaign.evaluator_image_identity
                or result.evaluator_implementation_digest != self.campaign.evaluator_implementation_digest
            ):
                raise TrustedStateError("trusted selection result evaluator identity does not match its campaign")

        projection = _selection_projection(self.campaign, self.policy, received)
        if (
            state.cursor != projection.cursor
            or state.best_score != projection.best_score
            or state.patience != projection.patience
        ):
            raise TrustedStateError("trusted selection projection does not match its received results")
        if state.top_five_snapshot_ids != projection.top_five_snapshot_ids:
            raise TrustedStateError("trusted top-five ranking does not match its received results")
        if state.lifecycle is SelectionLifecycle.READY and (state.revision != 0 or received):
            raise TrustedStateError("ready trusted state cannot contain received results")
        if projection.first_patience_trigger is not None and state.lifecycle not in (
            SelectionLifecycle.STOP_REQUESTED,
            SelectionLifecycle.COMPLETED,
            SelectionLifecycle.FAILED,
        ):
            raise TrustedStateError("trusted state must publish its patience stop request")
        result_by_snapshot = {result.snapshot_id for _, result in state.results_received_by_slot}
        if any(snapshot_id not in result_by_snapshot for snapshot_id in state.top_five_snapshot_ids):
            raise TrustedStateError("trusted top-five ranking references an unknown result")
        if isinstance(state.best_score, BestResult) and state.best_score.snapshot_id not in result_by_snapshot:
            raise TrustedStateError("trusted best score references an unknown result")

    def _validate_stop_request(self, state: SelectionState, stop_request: StopRequest) -> None:
        if stop_request.selection_revision > state.revision:
            raise TrustedStateError("trusted stop request revision is newer than its selection state")
        if (
            stop_request.best_snapshot.campaign_id != self.campaign.campaign_id
            or stop_request.best_snapshot.training_launch_id != self.campaign.training_launch_id
            or stop_request.triggering_result.campaign_id != self.campaign.campaign_id
            or stop_request.triggering_result.training_launch_id != self.campaign.training_launch_id
        ):
            raise TrustedStateError("trusted stop request best snapshot belongs to another campaign")
        received = dict(state.results_received_by_slot)
        if received.get(stop_request.triggering_result.slot_id) != stop_request.triggering_result:
            raise TrustedStateError("trusted stop request trigger is not a received result")
        best_result = next(
            (result for result in received.values() if result.snapshot_id == stop_request.best_snapshot.snapshot_id),
            None,
        )
        if best_result is None or best_result.snapshot != stop_request.best_snapshot:
            raise TrustedStateError("trusted stop request best snapshot is not the selected snapshot")
        try:
            expected = _derive_stop_request(self.campaign, self.policy, state)
        except ControllerError as error:
            raise TrustedStateError(str(error)) from error
        if (
            stop_request.triggering_result != expected.triggering_result
            or stop_request.best_snapshot != expected.best_snapshot
        ):
            raise TrustedStateError("trusted stop request is not bound to the first patience expiration")


def _parse_campaign_failure(value: object) -> CampaignFailure | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TrustedStateError("campaign failure must be an object or null")
    fields = {"kind", "detail"}
    if "slot_id" in value:
        fields.add("slot_id")
    if frozenset(value) != fields:
        raise TrustedStateError("campaign failure fields are not exact")
    try:
        kind = CampaignFailureKind(value["kind"])
        detail = value["detail"]
        slot_id = _parse_slot_id(value["slot_id"], "campaign failure slot_id") if "slot_id" in value else None
        return CampaignFailure(kind, detail, slot_id)
    except (TypeError, ValueError, CampaignCompletionError) as error:
        raise TrustedStateError("campaign failure is invalid") from error


def _parse_campaign_slot_state(value: object) -> CampaignSlotState:
    if not isinstance(value, Mapping):
        raise TrustedStateError("campaign slot state must be an object")
    if value.get("state") == ValidationSlotFailureState.state:
        if frozenset(value) != frozenset({"state", "snapshot", "detail"}):
            raise TrustedStateError("validation failure slot fields are not exact")
        try:
            return ValidationSlotFailureState(
                PublishedSnapshot.from_dict(value["snapshot"]),
                value["detail"],
            )
        except (TypeError, ValueError, ValidationContractError, CampaignCompletionError) as error:
            raise TrustedStateError("validation failure slot state is invalid") from error
    try:
        return validation_slot_state_from_dict(value)
    except (TypeError, ValueError, ValidationContractError) as error:
        raise TrustedStateError("campaign slot state is invalid") from error


def _parse_retention_commit(value: object) -> RankingRetentionCommit | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or frozenset(value) != frozenset({"selection_revision", "snapshot_ids"}):
        raise TrustedStateError("retention commit fields are not exact")
    snapshot_ids = value["snapshot_ids"]
    if not isinstance(snapshot_ids, list):
        raise TrustedStateError("retention commit snapshot IDs must be an array")
    try:
        return RankingRetentionCommit(
            value["selection_revision"],
            tuple(_parse_slot_id(item, "retention snapshot_id") for item in snapshot_ids),
        )
    except (TypeError, ValueError, CampaignCompletionError) as error:
        raise TrustedStateError("retention commit is invalid") from error


def _parse_validation_batch(value: object) -> ValidationBatch:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(
        {"ordinal", "slot_ids", "trigger", "worker_admissions", "cold_cache_admissions"}
    ):
        raise TrustedStateError("validation batch fields are not exact")
    slot_ids = value["slot_ids"]
    if not isinstance(slot_ids, list):
        raise TrustedStateError("validation batch slot IDs must be an array")
    try:
        return ValidationBatch(
            ordinal=value["ordinal"],
            slot_ids=tuple(_parse_slot_id(item, "validation batch slot_id") for item in slot_ids),
            trigger=BatchTrigger(value["trigger"]),
            worker_admissions=value["worker_admissions"],
            cold_cache_admissions=value["cold_cache_admissions"],
        )
    except (TypeError, ValueError, CampaignCompletionError) as error:
        raise TrustedStateError("validation batch is invalid") from error


def _parse_campaign_completion_state(value: object) -> CampaignCompletionState:
    required = frozenset(
        {
            "campaign_id",
            "slot_states",
            "training_completion",
            "queue_state",
            "retention_state",
            "retention_commit",
            "batches",
            "lifecycle",
            "failure",
            "revision",
        }
    )
    if not isinstance(value, Mapping) or frozenset(value) != required:
        raise TrustedStateError("campaign completion state fields are not exact")
    raw_slots = value["slot_states"]
    raw_batches = value["batches"]
    if not isinstance(raw_slots, list) or not isinstance(raw_batches, list):
        raise TrustedStateError("campaign completion slots and batches must be arrays")
    slot_states: list[tuple[Sha256Digest, CampaignSlotState]] = []
    try:
        for raw_slot in raw_slots:
            if not isinstance(raw_slot, Mapping) or frozenset(raw_slot) != frozenset({"slot_id", "state"}):
                raise TrustedStateError("campaign slot entry fields are not exact")
            slot_states.append(
                (
                    _parse_slot_id(raw_slot["slot_id"], "campaign slot_id"),
                    _parse_campaign_slot_state(raw_slot["state"]),
                )
            )
        training_completion = (
            None
            if value["training_completion"] is None
            else training_completion_from_dict(value["training_completion"])
        )
        return CampaignCompletionState(
            campaign_id=_parse_slot_id(value["campaign_id"], "campaign_id"),
            slot_states=tuple(slot_states),
            training_completion=training_completion,
            queue_state=QueueDrainState(value["queue_state"]),
            retention_state=RetentionState(value["retention_state"]),
            retention_commit=_parse_retention_commit(value["retention_commit"]),
            batches=tuple(_parse_validation_batch(raw_batch) for raw_batch in raw_batches),
            lifecycle=CampaignLifecycle(value["lifecycle"]),
            failure=_parse_campaign_failure(value["failure"]),
            revision=value["revision"],
        )
    except TrustedStateError:
        raise
    except (TypeError, ValueError, ValidationContractError, CampaignCompletionError) as error:
        raise TrustedStateError("campaign completion state is invalid") from error


class CampaignCompletionStateStore:
    """Persist campaign lifecycle state with a canonical digest envelope."""

    def __init__(
        self,
        path: str | Path,
        campaign: ValidationCampaign,
        initial_state: CampaignCompletionState | None = None,
    ) -> None:
        if not isinstance(campaign, ValidationCampaign):
            raise TrustedStateError("campaign must use its typed contract")
        if initial_state is not None and not isinstance(initial_state, CampaignCompletionState):
            raise TrustedStateError("initial campaign state must use its typed contract")
        self.path = Path(path)
        self.campaign = campaign
        self.initial_state = initial_state if initial_state is not None else _initial_campaign_state(campaign)
        self._validate_campaign_state(self.initial_state)

    def load(self) -> CampaignCompletionState:
        """Read and verify the current campaign state, or return its initial state."""

        if not self.path.exists():
            self._validate_campaign_state(self.initial_state)
            return self.initial_state
        try:
            encoded = self.path.read_bytes()
            raw = json.loads(encoded)
        except (OSError, ValueError) as error:
            raise TrustedStateError("campaign completion state cannot be decoded") from error
        if not isinstance(raw, Mapping):
            raise TrustedStateError("campaign completion envelope must be an object")
        required = frozenset({"schema", "campaign_id", "revision", "state", "digest"})
        if frozenset(raw) != required:
            raise TrustedStateError("campaign completion envelope fields are not exact")
        if raw["schema"] != CAMPAIGN_COMPLETION_STATE_SCHEMA:
            raise TrustedStateError("campaign completion schema is not supported")
        if raw["campaign_id"] != self.campaign.campaign_id.value:
            raise TrustedStateError("campaign completion state belongs to another campaign")
        state = _parse_campaign_completion_state(raw["state"])
        self._validate_campaign_state(state)
        if raw["revision"] != state.revision:
            raise TrustedStateError("campaign completion revision does not match its state")
        body = {
            "schema": CAMPAIGN_COMPLETION_STATE_SCHEMA,
            "campaign_id": self.campaign.campaign_id.value,
            "revision": state.revision,
            "state": state.to_dict(),
        }
        expected = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        if raw["digest"] != expected:
            raise TrustedStateError("campaign completion digest does not match its content")
        if encoded != _canonical_bytes({**body, "digest": expected}):
            raise TrustedStateError("campaign completion state is not canonical JSON")
        return state

    def write(self, state: CampaignCompletionState) -> CampaignCompletionState:
        """Atomically publish a newer campaign state and its content digest."""

        self._validate_campaign_state(state)
        path_existed = self.path.exists()
        current = self.load()
        if state.revision < current.revision:
            raise RevisionConflictError("campaign completion revision is older than the stored revision")
        if state.revision == current.revision:
            if path_existed and state == current:
                return current
            if path_existed:
                raise RevisionConflictError("campaign completion revision reused for different content")
        body = {
            "schema": CAMPAIGN_COMPLETION_STATE_SCHEMA,
            "campaign_id": self.campaign.campaign_id.value,
            "revision": state.revision,
            "state": state.to_dict(),
        }
        envelope = {**body, "digest": hashlib.sha256(_canonical_bytes(body)).hexdigest()}
        encoded = _canonical_bytes(envelope)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as error:
            raise TrustedStateError("campaign completion state replacement failed") from error
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        return state

    def _validate_campaign_state(self, state: CampaignCompletionState) -> None:
        if not isinstance(state, CampaignCompletionState):
            raise TrustedStateError("campaign completion state must use its typed contract")
        if state.campaign_id != self.campaign.campaign_id:
            raise TrustedStateError("campaign completion state belongs to another campaign")
        expected_slot_ids = tuple(slot.slot_id for slot in self.campaign.slots)
        if tuple(slot_id for slot_id, _ in state.slot_states) != expected_slot_ids:
            raise TrustedStateError("campaign completion slots do not match the campaign")
        slot_by_id = {slot.slot_id: slot for slot in self.campaign.slots}
        snapshots: dict[Sha256Digest, PublishedSnapshot] = {}
        results: dict[Sha256Digest, ValidationResult] = {}
        for slot_id, slot_state in state.slot_states:
            slot = slot_by_id[slot_id]
            if isinstance(slot_state, PlannedSlotState):
                continue
            if isinstance(slot_state, UnusedSlotState):
                if not isinstance(state.training_completion, AcceptedEarlyStop):
                    raise TrustedStateError("unused slot requires accepted early-stop completion")
                continue
            snapshot = slot_state.snapshot
            self._validate_snapshot(snapshot, slot)
            snapshots[slot_id] = snapshot
            if isinstance(slot_state, ValidatedSlotState):
                if not slot_state.result.matches_snapshot(snapshot):
                    raise TrustedStateError("validated campaign slot result does not match its snapshot")
                results[slot_id] = slot_state.result
            elif isinstance(slot_state, ValidationSlotFailureState):
                if state.lifecycle is not CampaignLifecycle.FAILED:
                    raise TrustedStateError("validation failure slot requires a failed campaign")
            elif not isinstance(slot_state, PublishedSlotState):
                raise TrustedStateError("campaign slot state is not supported")

        for batch in state.batches:
            if any(slot_id not in slot_by_id for slot_id in batch.slot_ids):
                raise TrustedStateError("validation batch contains an unknown campaign slot")
        if state.batches[0].slot_ids != expected_slot_ids:
            raise TrustedStateError("initial validation batch does not contain all campaign slots")

        completion = state.training_completion
        if isinstance(completion, AcceptedEarlyStop):
            self._validate_snapshot(completion.best_snapshot, slot_by_id.get(completion.best_snapshot.slot_id))
            trigger_slot = slot_by_id.get(completion.triggering_result.slot_id)
            if trigger_slot is None or trigger_slot.slot_id not in results:
                raise TrustedStateError("accepted early-stop trigger is not a validated campaign result")
            if results[trigger_slot.slot_id] != completion.triggering_result:
                raise TrustedStateError("accepted early-stop trigger does not match its validated result")
            if completion.best_snapshot.slot_id not in snapshots:
                raise TrustedStateError("accepted early-stop best snapshot is not published")
        if state.failure is not None and state.failure.slot_id is not None:
            if state.failure.slot_id not in slot_by_id:
                raise TrustedStateError("campaign failure references an unknown slot")

    def _validate_snapshot(self, snapshot: PublishedSnapshot, slot: ValidationSlot | None) -> None:
        if slot is None:
            raise TrustedStateError("published snapshot slot is not in the campaign")
        if (
            snapshot.campaign_id != self.campaign.campaign_id
            or snapshot.training_launch_id != self.campaign.training_launch_id
            or snapshot.slot_id != slot.slot_id
            or snapshot.updates != slot.updates
            or snapshot.dev_bundle_digest != self.campaign.dev_bundle_digest
            or snapshot.trainer_configuration_digest != self.campaign.trainer_configuration_digest
            or snapshot.evaluator_image_identity != self.campaign.evaluator_image_identity
            or snapshot.evaluator_implementation_digest != self.campaign.evaluator_implementation_digest
        ):
            raise TrustedStateError("published snapshot identity does not match the campaign")


__all__ = [
    "BatchTrigger",
    "CampaignCompletionController",
    "CampaignCompletionError",
    "CampaignCompletionState",
    "CampaignCompletionStateStore",
    "CampaignFailure",
    "CampaignFailureKind",
    "CampaignLifecycle",
    "CampaignSlotState",
    "ControllerError",
    "LagDecision",
    "LagPermit",
    "LagGateError",
    "LagWait",
    "LagWaitReason",
    "QueueDrainState",
    "RankingRetentionCommit",
    "ResultConflictError",
    "ResultImportError",
    "ResultImportFailureKind",
    "RetentionState",
    "SelectionController",
    "SelectionPolicy",
    "SelectionUpdate",
    "StopRequest",
    "TrustedStateError",
    "TrustedStateStore",
    "ValidationBatch",
    "ValidationLagGate",
    "ValidationSlotFailureState",
    "initial_selection_state",
]
