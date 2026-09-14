"""Tests for trusted external validation ordering and campaign completion."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from recipes.speakrs.large.validation import (
    AcceptedEarlyStop,
    CompleteEpochValidatedCursor,
    EpochZeroPendingCursor,
    EpochZeroValidatedCursor,
    MaxUpdatesReached,
    SelectionLifecycle,
    Sha256Digest,
    ValidationResult,
    build_validation_campaign,
)
from recipes.speakrs.large.validation.contracts import PublishedSnapshot
from recipes.speakrs.large.validation.controller import (
    BatchTrigger,
    CampaignCompletionController,
    CampaignCompletionError,
    CampaignCompletionStateStore,
    CampaignLifecycle,
    LagPermit,
    LagWait,
    LagWaitReason,
    QueueDrainState,
    ResultConflictError,
    ResultImportError,
    ResultImportFailureKind,
    RetentionState,
    SelectionController,
    SelectionPolicy,
    TrustedStateError,
    TrustedStateStore,
    ValidationLagGate,
    ValidationSlotFailureState,
    _selection_projection,
    initial_selection_state,
)


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _campaign(max_updates: int = 5, maximum_validation_lag: int = 1):
    return build_validation_campaign(
        training_launch_id="launch-controller-test",
        max_updates=max_updates,
        updates_per_complete_epoch=2,
        artifact_prefix="s3://validation-tests",
        dev_bundle_digest=_digest("a"),
        trainer_configuration_digest=_digest("b"),
        evaluator_image_identity="registry.example/evaluator@sha256:" + "c" * 64,
        evaluator_implementation_digest=_digest("d"),
        maximum_validation_lag=maximum_validation_lag,
    )


def _snapshots(campaign):
    hex_characters = "123456789abcdef0"
    publication_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return tuple(
        PublishedSnapshot(
            campaign_id=campaign.campaign_id,
            training_launch_id=campaign.training_launch_id,
            slot_id=slot.slot_id,
            updates=slot.updates,
            model_digest=_digest(hex_characters[index]),
            model_length=100 + index,
            trainer_configuration_digest=campaign.trainer_configuration_digest,
            dev_bundle_digest=campaign.dev_bundle_digest,
            evaluator_image_identity=campaign.evaluator_image_identity,
            evaluator_implementation_digest=campaign.evaluator_implementation_digest,
            recovery_generation_id=_digest(hex_characters[index + 1]),
            generation_sequence=1,
            progress_digest=_digest(hex_characters[index + 2]),
            publication_time=publication_time + timedelta(days=index),
        )
        for index, slot in enumerate(campaign.slots)
    )


def _result(snapshot: PublishedSnapshot, score: float, *, loss: float | None = None) -> ValidationResult:
    metric = score
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
        loss=metric if loss is None else loss,
        der=metric,
        false_alarm=0.02,
        miss=0.03,
        confusion=0.04,
        started_at=snapshot.publication_time,
        completed_at=snapshot.publication_time,
    )


def test_initial_state_and_out_of_order_results_are_contiguous_and_idempotent() -> None:
    campaign = _campaign(max_updates=6)
    snapshots = _snapshots(campaign)
    controller = SelectionController(campaign, SelectionPolicy(3), snapshots)

    initial = controller.state
    assert isinstance(initial.cursor, EpochZeroPendingCursor)
    assert initial.best_score.kind.value == "no_best_result"
    assert initial.patience == 0
    assert initial.revision == 0

    future = _result(snapshots[2], 0.4)
    first = controller.import_result(future.to_dict())
    assert first.revision == 1
    assert isinstance(first.cursor, EpochZeroPendingCursor)
    assert first.top_five_snapshot_ids == ()
    assert first.patience == 0

    replay = controller.import_result(future.to_dict())
    assert replay == first
    assert controller.last_update.idempotent

    zero = controller.import_result(_result(snapshots[0], 0.5).to_dict())
    assert isinstance(zero.cursor, EpochZeroValidatedCursor)
    assert zero.patience == 0

    drained = controller.import_result(_result(snapshots[1], 0.3).to_dict())
    assert isinstance(drained.cursor, CompleteEpochValidatedCursor)
    assert drained.cursor.epoch == 2
    assert tuple(controller.last_update.applied_results) == (
        _result(snapshots[1], 0.3),
        future,
    )
    assert drained.revision == 3
    assert drained.patience == 1

    conflict = replace(future, loss=0.9).to_dict()
    with pytest.raises(ResultConflictError) as error:
        controller.import_result(conflict)
    assert error.value.kind is ResultImportFailureKind.RESULT_CONFLICT
    assert controller.state.lifecycle is SelectionLifecycle.FAILED


def test_result_import_rejects_unknown_fields_and_snapshot_identity_mismatches() -> None:
    campaign = _campaign()
    snapshot = _snapshots(campaign)[0]
    controller = SelectionController(campaign, SelectionPolicy(2), (snapshot,))

    unknown = _result(snapshot, 0.2).to_dict()
    unknown["temporary_url"] = "https://secret.example"
    with pytest.raises(ResultImportError) as unknown_error:
        controller.import_result(unknown)
    assert unknown_error.value.kind is ResultImportFailureKind.INVALID_DOCUMENT

    campaign = _campaign()
    snapshot = _snapshots(campaign)[0]
    controller = SelectionController(campaign, SelectionPolicy(2), (snapshot,))
    mismatch = _result(snapshot, 0.2).to_dict()
    mismatch["model_digest"] = _digest("e").value
    with pytest.raises(ResultImportError) as mismatch_error:
        controller.import_result(mismatch)
    assert mismatch_error.value.kind is ResultImportFailureKind.INVALID_DOCUMENT


def test_tie_ranking_uses_der_then_slot_ordinal_then_snapshot_digest() -> None:
    campaign = _campaign(max_updates=6)
    snapshots = _snapshots(campaign)
    controller = SelectionController(campaign, SelectionPolicy(4), snapshots)

    for snapshot in snapshots[:3]:
        controller.import_result(_result(snapshot, 0.25).to_dict())

    assert controller.state.top_five_snapshot_ids == tuple(snapshot.snapshot_id for snapshot in snapshots[:3])


def test_patience_emits_one_stop_request_and_final_partial_keeps_complete_cursor() -> None:
    campaign = _campaign(max_updates=3)
    snapshots = _snapshots(campaign)
    controller = SelectionController(campaign, SelectionPolicy(2), snapshots)

    for snapshot, loss in zip(snapshots, (0.5, 0.6, 0.7), strict=True):
        controller.import_result(_result(snapshot, loss).to_dict())

    state = controller.state
    assert state.lifecycle is SelectionLifecycle.STOP_REQUESTED
    assert isinstance(state.cursor, CompleteEpochValidatedCursor)
    assert state.cursor.epoch == 1
    assert state.patience == 2
    assert state.top_five_snapshot_ids == tuple(snapshot.snapshot_id for snapshot in snapshots)
    assert controller.stop_request is not None
    assert controller.stop_request.selection_revision == state.revision
    assert controller.stop_request.triggering_result.slot_id == snapshots[2].slot_id
    assert controller.stop_request.best_snapshot.snapshot_id == snapshots[0].snapshot_id

    revision = state.revision
    assert controller.import_result(_result(snapshots[2], 0.7).to_dict()).revision == revision


def test_stop_request_is_frozen_while_registered_later_results_still_drain() -> None:
    campaign = _campaign(max_updates=5)
    snapshots = _snapshots(campaign)
    controller = SelectionController(campaign, SelectionPolicy(2), snapshots)
    controller.import_result(_result(snapshots[0], 0.5).to_dict())
    controller.import_result(_result(snapshots[1], 0.6).to_dict())
    controller.import_result(_result(snapshots[2], 0.7).to_dict())
    stop_request = controller.stop_request
    assert stop_request is not None

    state = controller.import_result(_result(snapshots[3], 0.1).to_dict())
    assert state.lifecycle is SelectionLifecycle.STOP_REQUESTED
    assert controller.stop_request == stop_request
    assert tuple(result.slot_id for result in controller.last_update.applied_results) == (snapshots[3].slot_id,)
    assert state.top_five_snapshot_ids[0] == snapshots[3].snapshot_id


def test_lag_gate_requires_epoch_zero_and_allows_exactly_one_pending_epoch() -> None:
    campaign = _campaign(max_updates=6, maximum_validation_lag=1)
    snapshots = _snapshots(campaign)
    controller = SelectionController(campaign, SelectionPolicy(4), snapshots)
    gate = ValidationLagGate(campaign)

    waiting = gate.decide(controller.state, snapshots[:1], next_epoch=1)
    assert isinstance(waiting, LagWait)
    assert waiting.reason is LagWaitReason.EPOCH_ZERO_PENDING
    assert isinstance(gate.decide(controller.state, snapshots[:1], next_epoch=1), LagWait)
    controller.import_result(_result(snapshots[0], 0.5).to_dict())
    permit = gate.decide(controller.state, snapshots[:2], next_epoch=2)
    assert isinstance(permit, LagPermit)
    assert isinstance(gate.decide(controller.state, snapshots[:2], next_epoch=2), LagPermit)

    waiting = gate.decide(controller.state, snapshots[:3], next_epoch=3)
    assert isinstance(waiting, LagWait)
    assert waiting.reason is LagWaitReason.VALIDATION_LAG_EXCEEDED
    assert isinstance(gate.decide(controller.state, snapshots[:3], next_epoch=3), LagWait)
    controller.import_result(_result(snapshots[1], 0.4).to_dict())
    assert isinstance(gate.decide(controller.state, snapshots[:3], next_epoch=3), LagPermit)


def test_campaign_completion_waits_for_published_results_retention_and_drain() -> None:
    campaign = _campaign()
    snapshots = _snapshots(campaign)
    controller = CampaignCompletionController(campaign, SelectionPolicy(10))
    for snapshot in snapshots:
        controller.publish_snapshot(snapshot)
    for snapshot in snapshots:
        controller.import_result(_result(snapshot, 0.2).to_dict())

    controller.mark_trainer_terminal(MaxUpdatesReached())
    assert isinstance(controller.state.training_completion, MaxUpdatesReached)
    controller.commit_ranking_retention()
    assert not controller.try_complete()
    controller.cancel_queue()
    assert not controller.try_complete()
    controller.mark_queue_drained()
    assert controller.try_complete()
    assert controller.state.lifecycle is CampaignLifecycle.COMPLETED


def test_early_stop_marks_only_unpublished_slots_unused_after_durable_terminal_progress() -> None:
    campaign = _campaign(max_updates=5)
    snapshots = _snapshots(campaign)
    controller = CampaignCompletionController(campaign, SelectionPolicy(2))
    for snapshot in snapshots[:3]:
        controller.publish_snapshot(snapshot)
    for snapshot, loss in zip(snapshots[:3], (0.5, 0.6, 0.7), strict=True):
        controller.import_result(_result(snapshot, loss).to_dict())
    assert controller.selection.stop_request is not None
    early_stop = AcceptedEarlyStop(
        selection_revision=controller.selection.stop_request.selection_revision,
        triggering_result=_result(snapshots[2], 0.7),
        best_snapshot=snapshots[0],
    )
    controller.mark_trainer_terminal(early_stop)
    assert all(isinstance(state, ValidationSlotFailureState) is False for _, state in controller.state.slot_states)
    assert (
        sum(state.__class__.__name__ == "UnusedSlotState" for _, state in controller.state.slot_states)
        == len(snapshots) - 3
    )

    controller.commit_ranking_retention()
    controller.cancel_queue()
    controller.mark_queue_drained()
    assert controller.try_complete()


def test_terminal_validation_failure_prevents_campaign_completion() -> None:
    campaign = _campaign()
    snapshot = _snapshots(campaign)[0]
    controller = CampaignCompletionController(campaign, SelectionPolicy(4))
    controller.publish_snapshot(snapshot)
    failed = controller.mark_validation_failure(snapshot.slot_id, "evaluator exited")
    assert failed.lifecycle is CampaignLifecycle.FAILED
    assert isinstance(dict(failed.slot_states)[snapshot.slot_id], ValidationSlotFailureState)
    assert not controller.try_complete()


def test_hard_lifetime_batch_contains_only_unresolved_slots_and_url_expiry_is_not_a_batch_reason() -> None:
    campaign = _campaign()
    snapshots = _snapshots(campaign)
    controller = CampaignCompletionController(campaign, SelectionPolicy(4))
    controller.publish_snapshot(snapshots[0])
    controller.import_result(_result(snapshots[0], 0.2).to_dict())

    batch = controller.create_additional_batch(BatchTrigger.HARD_LIFETIME)
    assert snapshots[0].slot_id not in batch.slot_ids
    assert batch.slot_ids == tuple(slot.slot_id for slot in campaign.slots[1:])
    assert controller.state.extra_worker_admissions == 1
    assert controller.state.extra_cold_cache_admissions == 1
    with pytest.raises(CampaignCompletionError):
        controller.create_additional_batch("url_expiry")


def test_trusted_state_store_is_canonical_digest_bound_and_monotonic(tmp_path) -> None:
    campaign = _campaign()
    snapshot = _snapshots(campaign)[0]
    selection = SelectionController(campaign, SelectionPolicy(3), (snapshot,))
    state = selection.import_result(_result(snapshot, 0.2).to_dict())
    path = tmp_path / "trusted-selection.json"
    store = TrustedStateStore(path, campaign, SelectionPolicy(3), initial_selection_state())

    store.write(initial_selection_state())
    assert store.load().revision == 0
    store.write(state)
    assert store.accept_newer_revision(0) == state
    with pytest.raises(TrustedStateError):
        store.accept_newer_revision(state.revision)

    encoded = path.read_text(encoding="utf-8")
    path.write_text(encoded.replace('"digest":"', '"digest":"0', 1), encoding="utf-8")
    with pytest.raises(TrustedStateError, match="digest"):
        store.load()


def test_stop_request_persists_across_reload_and_terminal_selection_does_not_mutate(tmp_path) -> None:
    campaign = _campaign(max_updates=3)
    snapshots = _snapshots(campaign)
    selection = SelectionController(campaign, SelectionPolicy(2), snapshots)
    for snapshot, loss in zip(snapshots, (0.5, 0.6, 0.7), strict=True):
        selection.import_result(_result(snapshot, loss).to_dict())
    path = tmp_path / "stop.json"
    store = TrustedStateStore(path, campaign, SelectionPolicy(2))
    store.write_update(selection.last_update)

    restored_update = store.load_update()
    restored = SelectionController(
        campaign,
        SelectionPolicy(2),
        snapshots,
        restored_update.state,
        stop_request=restored_update.stop_request,
    )
    revision = restored.state.revision
    assert restored.stop_request == selection.stop_request
    assert restored.import_result(_result(snapshots[2], 0.7).to_dict()).revision == revision
    with pytest.raises(ResultImportError) as terminal_error:
        restored.import_result(_result(snapshots[1], 0.8).to_dict())
    assert terminal_error.value.kind is ResultImportFailureKind.RESULT_CONFLICT
    assert restored.state.revision == revision + 1


def test_completed_campaign_rejects_new_result_without_changing_terminal_state() -> None:
    campaign = _campaign()
    snapshots = _snapshots(campaign)
    controller = CampaignCompletionController(campaign, SelectionPolicy(10))
    for snapshot in snapshots:
        controller.publish_snapshot(snapshot)
        controller.import_result(_result(snapshot, 0.2).to_dict())
    controller.mark_trainer_terminal(MaxUpdatesReached())
    controller.commit_ranking_retention()
    controller.cancel_queue()
    controller.mark_queue_drained()
    controller.complete()
    before = controller.state

    with pytest.raises(ResultImportError):
        controller.import_result(_result(snapshots[0], 0.3).to_dict())
    assert controller.state == before


def test_ranking_prefers_der_when_loss_would_rank_differently() -> None:
    campaign = _campaign(max_updates=6)
    snapshots = _snapshots(campaign)
    controller = SelectionController(campaign, SelectionPolicy(4), snapshots)

    controller.import_result(_result(snapshots[0], 0.40, loss=0.10).to_dict())
    controller.import_result(_result(snapshots[1], 0.20, loss=0.90).to_dict())

    assert controller.state.top_five_snapshot_ids[0] == snapshots[1].snapshot_id
    assert controller.state.best_score.score == 0.20


def test_selection_projection_owns_cursor_ranking_patience_and_stop() -> None:
    campaign = _campaign(max_updates=3)
    snapshots = _snapshots(campaign)
    policy = SelectionPolicy(2)
    controller = SelectionController(campaign, policy, snapshots)
    for snapshot, score in zip(snapshots, (0.5, 0.6, 0.7), strict=True):
        controller.import_result(_result(snapshot, score).to_dict())

    projection = _selection_projection(campaign, policy, dict(controller.state.results_received_by_slot))
    state = controller.state
    assert state.cursor == projection.cursor
    assert state.best_score == projection.best_score
    assert state.patience == projection.patience
    assert state.top_five_snapshot_ids == projection.top_five_snapshot_ids
    assert controller.stop_request is not None
    assert controller.stop_request.triggering_result == projection.first_patience_trigger
    assert projection.best_at_first_trigger is not None
    assert controller.stop_request.best_snapshot.snapshot_id == projection.best_at_first_trigger.snapshot_id


def _restore_completion(campaign, policy, snapshots, selection, completion, tmp_path, suffix: str):
    selection_store = TrustedStateStore(tmp_path / f"selection-{suffix}.json", campaign, policy)
    completion_store = CampaignCompletionStateStore(tmp_path / f"completion-{suffix}.json", campaign)
    selection.persist(selection_store)
    completion.persist(completion_store)
    restored_update = selection_store.load_update()
    restored_selection = SelectionController(
        campaign,
        policy,
        snapshots,
        restored_update.state,
        stop_request=restored_update.stop_request,
    )
    return CampaignCompletionController.restore(campaign, policy, completion_store, restored_selection)


def test_campaign_completion_restore_at_drain_retention_unused_and_failure(tmp_path) -> None:
    campaign = _campaign(max_updates=5)
    snapshots = _snapshots(campaign)
    policy = SelectionPolicy(2)

    drain = CampaignCompletionController(campaign, policy)
    for snapshot in snapshots[:3]:
        drain.publish_snapshot(snapshot)
    for snapshot, score in zip(snapshots[:3], (0.5, 0.6, 0.7), strict=True):
        drain.import_result(_result(snapshot, score).to_dict())
    drain.mark_trainer_terminal(
        AcceptedEarlyStop(
            drain.selection.stop_request.selection_revision,
            _result(snapshots[2], 0.7),
            snapshots[0],
        )
    )
    drain.cancel_queue()
    restored_drain = _restore_completion(campaign, policy, snapshots[:3], drain.selection, drain, tmp_path, "drain")
    restored_drain.mark_queue_drained()
    assert restored_drain.state.queue_state is QueueDrainState.DRAINED

    retention = CampaignCompletionController(campaign, policy)
    for snapshot in snapshots[:3]:
        retention.publish_snapshot(snapshot)
        retention.import_result(_result(snapshot, 0.2).to_dict())
    retention.mark_trainer_terminal(MaxUpdatesReached())
    retention.commit_ranking_retention()
    restored_retention = _restore_completion(
        campaign, policy, snapshots[:3], retention.selection, retention, tmp_path, "retention"
    )
    assert restored_retention.state.retention_state is RetentionState.COMMITTED

    unused = CampaignCompletionController(campaign, policy)
    for snapshot in snapshots[:3]:
        unused.publish_snapshot(snapshot)
    for snapshot, score in zip(snapshots[:3], (0.5, 0.6, 0.7), strict=True):
        unused.import_result(_result(snapshot, score).to_dict())
    unused.mark_trainer_terminal(
        AcceptedEarlyStop(
            unused.selection.stop_request.selection_revision,
            _result(snapshots[2], 0.7),
            snapshots[0],
        )
    )
    restored_unused = _restore_completion(
        campaign, policy, snapshots[:3], unused.selection, unused, tmp_path, "unused"
    )
    unused_count = sum(state.__class__.__name__ == "UnusedSlotState" for _, state in restored_unused.state.slot_states)
    assert unused_count == len(snapshots) - 3

    failed = CampaignCompletionController(campaign, policy)
    failed.publish_snapshot(snapshots[0])
    failed.mark_validation_failure(snapshots[0].slot_id, "evaluator exited")
    restored_failed = _restore_completion(
        campaign, policy, (snapshots[0],), failed.selection, failed, tmp_path, "failure"
    )
    assert restored_failed.state.lifecycle is CampaignLifecycle.FAILED
    assert not restored_failed.try_complete()


def test_speculative_result_after_stop_invalidates_stale_retention() -> None:
    campaign = _campaign(max_updates=5)
    snapshots = _snapshots(campaign)
    controller = CampaignCompletionController(campaign, SelectionPolicy(2))
    for snapshot in snapshots[:4]:
        controller.publish_snapshot(snapshot)
    for snapshot, score in zip(snapshots[:3], (0.5, 0.6, 0.7), strict=True):
        controller.import_result(_result(snapshot, score).to_dict())
    assert controller.selection.state.lifecycle is SelectionLifecycle.STOP_REQUESTED
    controller.mark_trainer_terminal(
        AcceptedEarlyStop(
            controller.selection.stop_request.selection_revision,
            _result(snapshots[2], 0.7),
            snapshots[0],
        )
    )
    controller.commit_ranking_retention()
    assert controller.state.retention_state is RetentionState.COMMITTED
    revision_before = controller.selection.state.revision

    later = controller.import_result(_result(snapshots[3], 0.1).to_dict())
    assert later.retention_state is RetentionState.PENDING
    assert later.retention_commit is None
    assert controller.selection.state.revision == revision_before + 1
    assert controller.selection.state.top_five_snapshot_ids[0] == snapshots[3].snapshot_id
    controller.cancel_queue()
    controller.mark_queue_drained()
    with pytest.raises(CampaignCompletionError):
        controller.complete()
    controller.commit_ranking_retention()
    assert controller.try_complete()
