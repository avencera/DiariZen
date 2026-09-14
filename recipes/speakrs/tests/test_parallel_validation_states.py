"""Tests for the strict parallel-validation state contracts."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recipes.speakrs.large.validation import (
    AcceptedEarlyStop,
    ActiveTrainingRun,
    ArtifactLocation,
    BestResult,
    CompletedTrainingRun,
    CompleteEpochValidatedCursor,
    EpochZeroPendingCursor,
    EpochZeroValidatedCursor,
    FailedTrainingRun,
    NoBestResult,
    PublicationTransaction,
    PublicationTransactionState,
    PublishedSnapshot,
    ResumableTrainingProgress,
    SelectionFailure,
    SelectionFailureKind,
    SelectionLifecycle,
    SelectionState,
    Sha256Digest,
    TrainingFailure,
    TrainingFailureKind,
    UnusedSlotState,
    ValidationContractError,
    ValidationResult,
    ValidationSlotState,
    ValidationSlotTerminalReason,
    contiguous_cursor_from_dict,
    training_run_state_from_dict,
    validation_slot_state_from_dict,
)


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _snapshot() -> PublishedSnapshot:
    return PublishedSnapshot(
        campaign_id=_digest("a"),
        training_launch_id="launch-v1",
        slot_id=_digest("b"),
        updates=0,
        model_digest=_digest("c"),
        model_length=123,
        trainer_configuration_digest=_digest("d"),
        dev_bundle_digest=_digest("e"),
        evaluator_image_identity="registry.example/evaluator@sha256:" + "f" * 64,
        evaluator_implementation_digest=_digest("1"),
        recovery_generation_id=_digest("2"),
        generation_sequence=1,
        progress_digest=_digest("3"),
        publication_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _result(snapshot: PublishedSnapshot | None = None) -> ValidationResult:
    snapshot = snapshot or _snapshot()
    return ValidationResult(
        snapshot_id=snapshot.snapshot_id,
        campaign_id=snapshot.campaign_id,
        training_launch_id=snapshot.training_launch_id,
        slot_id=snapshot.slot_id,
        updates=snapshot.updates,
        model_digest=snapshot.model_digest,
        model_length=snapshot.model_length,
        trainer_configuration_digest=snapshot.trainer_configuration_digest,
        dev_bundle_digest=snapshot.dev_bundle_digest,
        evaluator_image_identity=snapshot.evaluator_image_identity,
        evaluator_implementation_digest=snapshot.evaluator_implementation_digest,
        recovery_generation_id=snapshot.recovery_generation_id,
        generation_sequence=snapshot.generation_sequence,
        progress_digest=snapshot.progress_digest,
        publication_time=snapshot.publication_time,
        loss=0.25,
        der=0.5,
        false_alarm=0.1,
        miss=0.2,
        confusion=0.2,
        started_at=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
    )


def _transaction() -> PublicationTransaction:
    return PublicationTransaction(
        state=PublicationTransactionState.PREPARED,
        slot_id=_digest("b"),
        model_location=ArtifactLocation("s3://validation/model.bin"),
        model_digest=_digest("c"),
        model_length=123,
        recovery_generation_id=_digest("2"),
        generation_sequence=1,
        progress_digest=_digest("3"),
    )


def test_state_documents_reject_unknown_fields() -> None:
    snapshot = _snapshot()
    encoded = snapshot.to_dict()
    encoded["unexpected"] = True

    with pytest.raises(ValidationContractError, match="fields are not exact"):
        PublishedSnapshot.from_dict(encoded)

    result = _result(snapshot).to_dict()
    result["unexpected"] = True
    with pytest.raises(ValidationContractError, match="fields are not exact"):
        ValidationResult.from_dict(result)


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), float("-inf")])
def test_validation_result_rejects_nonfinite_metrics(metric: float) -> None:
    encoded = _result().to_dict()
    encoded["der"] = metric

    with pytest.raises(ValidationContractError, match="finite nonnegative"):
        ValidationResult.from_dict(encoded)


def test_validation_result_rejects_snapshot_identity_mismatch() -> None:
    encoded = _result().to_dict()
    encoded["slot_id"] = _digest("9").value

    with pytest.raises(ValidationContractError, match="snapshot identity"):
        ValidationResult.from_dict(encoded)


@pytest.mark.parametrize("sequence", [0, -1])
def test_publication_transaction_requires_positive_generation_sequence(sequence: int) -> None:
    with pytest.raises(ValidationContractError, match="generation_sequence"):
        PublicationTransaction.from_dict({**_transaction().to_dict(), "generation_sequence": sequence})


def test_unused_slot_state_has_no_snapshot_payload() -> None:
    state = UnusedSlotState(ValidationSlotTerminalReason.ACCEPTED_EARLY_STOP)
    assert state.to_dict() == {"state": ValidationSlotState.UNUSED.value, "reason": "accepted_early_stop"}

    encoded = state.to_dict()
    encoded["snapshot"] = _snapshot().to_dict()
    with pytest.raises(ValidationContractError, match="fields are not exact"):
        validation_slot_state_from_dict(encoded)


def test_contiguous_cursor_variants_are_closed_and_round_trip() -> None:
    cursors = (EpochZeroPendingCursor(), EpochZeroValidatedCursor(), CompleteEpochValidatedCursor(3))

    for cursor in cursors:
        assert contiguous_cursor_from_dict(cursor.to_dict()) == cursor

    with pytest.raises(ValidationContractError, match="fields are not exact"):
        contiguous_cursor_from_dict({"kind": "epoch_zero_pending", "epoch": 0})


def test_completed_early_stop_binds_revision_triggering_result_and_best_snapshot() -> None:
    snapshot = _snapshot()
    result = _result(snapshot)
    state = CompletedTrainingRun(AcceptedEarlyStop(7, result, snapshot))

    restored = training_run_state_from_dict(state.to_dict())

    assert restored == state
    assert state.to_dict()["completion"]["selection_revision"] == 7


def test_failed_training_state_requires_typed_failure() -> None:
    state = FailedTrainingRun(TrainingFailure(TrainingFailureKind.TRAINER_FAILURE, "optimizer stopped"))

    assert training_run_state_from_dict(state.to_dict()) == state
    with pytest.raises(ValidationContractError, match="fields are not exact"):
        training_run_state_from_dict({"state": "failed", "failure": state.failure.to_dict(), "progress": {}})


def test_active_and_selection_states_round_trip_with_typed_payloads() -> None:
    progress = ResumableTrainingProgress(0, 0, _digest("2"), 1, _digest("3"))
    active = ActiveTrainingRun(progress, _transaction())
    assert training_run_state_from_dict(active.to_dict()) == active

    result = _result()
    selection = SelectionState(
        results_received_by_slot=((result.slot_id, result),),
        cursor=EpochZeroValidatedCursor(),
        best_score=BestResult(result.der, result.snapshot_id),
        patience=3,
        top_five_snapshot_ids=(result.snapshot_id,),
        revision=1,
        lifecycle=SelectionLifecycle.WAITING,
    )
    assert SelectionState.from_dict(selection.to_dict()) == selection


def test_selection_initial_best_score_is_typed_and_top_five_consistent() -> None:
    initial = SelectionState(
        results_received_by_slot=(),
        cursor=EpochZeroPendingCursor(),
        best_score=NoBestResult(),
        patience=3,
        top_five_snapshot_ids=(),
        revision=0,
        lifecycle=SelectionLifecycle.READY,
    )
    assert SelectionState.from_dict(initial.to_dict()) == initial

    with pytest.raises(ValidationContractError, match="empty top-five"):
        SelectionState((), EpochZeroPendingCursor(), NoBestResult(), 3, (_digest("a"),), 0, SelectionLifecycle.READY)

    with pytest.raises(ValidationContractError, match="first top-five"):
        SelectionState(
            (),
            EpochZeroPendingCursor(),
            BestResult(0.0, _digest("a")),
            3,
            (_digest("b"),),
            0,
            SelectionLifecycle.READY,
        )

    encoded = initial.to_dict()
    encoded["best_score"]["unexpected"] = True
    with pytest.raises(ValidationContractError, match="fields are not exact"):
        SelectionState.from_dict(encoded)


def test_failed_selection_requires_failure_payload() -> None:
    with pytest.raises(ValidationContractError, match="requires a typed failure"):
        SelectionState((), EpochZeroPendingCursor(), NoBestResult(), 1, (), 0, SelectionLifecycle.FAILED)

    failure = SelectionFailure(SelectionFailureKind.CONTROLLER_FAILURE, "controller stopped")
    state = SelectionState((), EpochZeroPendingCursor(), NoBestResult(), 1, (), 0, SelectionLifecycle.FAILED, failure)
    assert SelectionState.from_dict(state.to_dict()) == state
