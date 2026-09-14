"""Focused tests for the typed external trainer bridge."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from recipes.speakrs.large.validation.contracts import (
    EpochZeroPendingCursor,
    NoBestResult,
    PublishedSnapshot,
    SelectionFailure,
    SelectionFailureKind,
    SelectionLifecycle,
    SelectionState,
    Sha256Digest,
    ValidationResult,
)
from recipes.speakrs.large.validation.controller import SelectionController, SelectionPolicy, TrustedStateStore
from recipes.speakrs.large.validation.slot_plan import build_validation_campaign
from recipes.speakrs.large.validation.trainer_bridge import ContinueTraining, SelectionFailed, TrainerBridge


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _campaign():
    return build_validation_campaign(
        training_launch_id="launch",
        max_updates=2,
        updates_per_complete_epoch=1,
        artifact_prefix="models",
        dev_bundle_digest=_digest("a"),
        trainer_configuration_digest=_digest("b"),
        evaluator_image_identity="evaluator",
        evaluator_implementation_digest=_digest("c"),
    )


def _snapshot(campaign) -> PublishedSnapshot:
    slot = campaign.slots[0]
    return PublishedSnapshot(
        campaign_id=campaign.campaign_id,
        training_launch_id=campaign.training_launch_id,
        slot_id=slot.slot_id,
        updates=slot.updates,
        model_digest=_digest("d"),
        model_length=1,
        trainer_configuration_digest=campaign.trainer_configuration_digest,
        dev_bundle_digest=campaign.dev_bundle_digest,
        evaluator_image_identity=campaign.evaluator_image_identity,
        evaluator_implementation_digest=campaign.evaluator_implementation_digest,
        recovery_generation_id=_digest("e"),
        generation_sequence=1,
        progress_digest=_digest("f"),
        publication_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _result(snapshot: PublishedSnapshot) -> ValidationResult:
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
        loss=0.1,
        der=0.1,
        false_alarm=0.0,
        miss=0.1,
        confusion=0.0,
        started_at=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
    )


class _Publisher:
    def publish(self, campaign, slot, generation_directory, progress_digest):
        raise AssertionError("publication is not part of this decision test")


def test_bridge_accepts_only_a_newer_campaign_owned_revision(tmp_path: Path) -> None:
    campaign = _campaign()
    snapshot = _snapshot(campaign)
    controller = SelectionController(campaign, SelectionPolicy(2), (snapshot,))
    state = controller.import_result(_result(snapshot).to_dict())
    store = TrustedStateStore(tmp_path / "selection.json", campaign, SelectionPolicy(2))
    store.write(state, controller.stop_request)
    bridge = TrainerBridge(campaign, store, _Publisher(), wait_for_state=lambda: None)

    assert bridge.read_newer_selection() is None
    assert bridge.selection_revision == state.revision
    assert isinstance(bridge.await_epoch_zero(), ContinueTraining)


def test_bridge_converts_trusted_failure_to_a_terminal_decision(tmp_path: Path) -> None:
    campaign = _campaign()
    initial = SelectionState(
        results_received_by_slot=(),
        cursor=EpochZeroPendingCursor(),
        best_score=NoBestResult(),
        patience=0,
        top_five_snapshot_ids=(),
        revision=0,
        lifecycle=SelectionLifecycle.READY,
    )
    failure = SelectionFailure(SelectionFailureKind.CONTROLLER_FAILURE, "controller stopped")
    failed = SelectionState(
        initial.results_received_by_slot,
        initial.cursor,
        initial.best_score,
        initial.patience,
        initial.top_five_snapshot_ids,
        1,
        SelectionLifecycle.FAILED,
        failure,
    )
    store = TrustedStateStore(tmp_path / "selection.json", campaign, SelectionPolicy(2), initial)
    store.write(failed)
    bridge = TrainerBridge(campaign, store, _Publisher(), wait_for_state=lambda: None)

    decision = bridge.observe_selection()

    assert isinstance(decision, SelectionFailed)
    assert decision.failure.kind.value == "selection_failure"


def test_lag_one_sequence_waits_until_contiguous_results_arrive(tmp_path: Path) -> None:
    from recipes.speakrs.large.validation.trainer_bridge import (
        WaitForValidation,
        _PublishedSlot,
    )

    campaign = build_validation_campaign(
        training_launch_id="lag-one",
        max_updates=4,
        updates_per_complete_epoch=1,
        artifact_prefix="models",
        dev_bundle_digest=_digest("a"),
        trainer_configuration_digest=_digest("b"),
        evaluator_image_identity="evaluator",
        evaluator_implementation_digest=_digest("c"),
    )
    snapshots = []
    for index, slot in enumerate(campaign.slots):
        snapshots.append(
            PublishedSnapshot(
                campaign_id=campaign.campaign_id,
                training_launch_id=campaign.training_launch_id,
                slot_id=slot.slot_id,
                updates=slot.updates,
                model_digest=_digest("d"),
                model_length=1 + index,
                trainer_configuration_digest=campaign.trainer_configuration_digest,
                dev_bundle_digest=campaign.dev_bundle_digest,
                evaluator_image_identity=campaign.evaluator_image_identity,
                evaluator_implementation_digest=campaign.evaluator_implementation_digest,
                recovery_generation_id=_digest("e"),
                generation_sequence=1,
                progress_digest=_digest("f"),
                publication_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
    store = TrustedStateStore(tmp_path / "selection.json", campaign, SelectionPolicy(4))
    bridge = TrainerBridge(campaign, store, _Publisher(), wait_for_state=lambda: None)
    bridge._published = {
        snapshot.slot_id: _PublishedSlot(campaign.slots[index], snapshot)
        for index, snapshot in enumerate(snapshots[:1])
    }
    waiting = bridge.before_epoch(1)
    assert isinstance(waiting, WaitForValidation)

    controller = SelectionController(campaign, SelectionPolicy(4), snapshots)
    store.write(controller.import_result(_result(snapshots[0]).to_dict()), controller.stop_request)
    bridge = TrainerBridge(campaign, store, _Publisher(), wait_for_state=lambda: None)
    bridge._published = {
        snapshot.slot_id: _PublishedSlot(campaign.slots[index], snapshot)
        for index, snapshot in enumerate(snapshots[:2])
    }
    assert isinstance(bridge.before_epoch(2), ContinueTraining)

    bridge._published = {
        snapshot.slot_id: _PublishedSlot(campaign.slots[index], snapshot)
        for index, snapshot in enumerate(snapshots[:3])
    }
    assert isinstance(bridge.before_epoch(3), WaitForValidation)

    store.write(controller.import_result(_result(snapshots[1]).to_dict()), controller.stop_request)
    bridge = TrainerBridge(campaign, store, _Publisher(), wait_for_state=lambda: None)
    bridge._published = {
        snapshot.slot_id: _PublishedSlot(campaign.slots[index], snapshot)
        for index, snapshot in enumerate(snapshots[:3])
    }
    assert isinstance(bridge.before_epoch(3), ContinueTraining)
