"""Production trusted-controller restore, import, and completion tests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from recipes.speakrs.large.validation.contracts import (
    VALIDATION_RESULT_SCHEMA,
    MaxUpdatesReached,
    PublishedSnapshot,
    SelectionLifecycle,
    Sha256Digest,
    ValidationResult,
)
from recipes.speakrs.large.validation.controller import (
    CampaignCompletionStateStore,
    CampaignLifecycle,
    RankingRetentionCommit,
    RetentionState,
    SelectionPolicy,
    TrustedStateStore,
    initial_selection_state,
)
from recipes.speakrs.large.validation.slot_plan import build_validation_campaign
from recipes.speakrs.large.validation.trusted_importer import (
    TrustedController,
    parse_exported_cloudeck_result_bytes,
)


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _campaign():
    return build_validation_campaign(
        training_launch_id="launch-trusted-importer",
        max_updates=5,
        updates_per_complete_epoch=2,
        artifact_prefix="s3://validation-importer",
        dev_bundle_digest=_digest("a"),
        trainer_configuration_digest=_digest("b"),
        evaluator_image_identity="registry.example/evaluator@sha256:" + "c" * 64,
        evaluator_implementation_digest=_digest("d"),
        maximum_validation_lag=1,
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


def _result(snapshot: PublishedSnapshot, score: float) -> ValidationResult:
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
        loss=score,
        der=score,
        false_alarm=0.02,
        miss=0.03,
        confusion=0.04,
        started_at=snapshot.publication_time,
        completed_at=snapshot.publication_time,
    )


class _Drain:
    def __init__(self) -> None:
        self.cancelled = False
        self.drained = False

    def request_cancel(self) -> None:
        self.cancelled = True

    def is_drained(self) -> bool:
        return self.drained


class _Retention:
    def __init__(self) -> None:
        self.receipts: list[RankingRetentionCommit] = []

    def commit(self, receipt: RankingRetentionCommit) -> None:
        self.receipts.append(receipt)


def _controller(tmp_path: Path, campaign, snapshots):
    policy = SelectionPolicy(10)
    selection_store = TrustedStateStore(tmp_path / "selection.json", campaign, policy, initial_selection_state())
    completion_store = CampaignCompletionStateStore(tmp_path / "completion.json", campaign)
    drain = _Drain()
    retention = _Retention()
    controller = TrustedController.restore(
        campaign,
        policy,
        selection_store,
        completion_store,
        drain=drain,
        retention=retention,
    )
    for snapshot in snapshots:
        controller.publish_snapshot(snapshot)
    return controller, drain, retention


def _exported_result_bytes(result: ValidationResult) -> bytes:
    return json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _unit_results_bytes(result: ValidationResult) -> bytes:
    body = json.loads(_exported_result_bytes(result))
    compact = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    envelope = {
        "version": 3,
        "result": "unit_results",
        "payload": {
            "queue_id": "01912345-6789-7abc-8def-0123456789ab",
            "units": [
                {
                    "unit_id": "slot-0",
                    "ordinal": 0,
                    "result_digest": hashlib.sha256(compact).hexdigest(),
                    "result": body,
                }
            ],
            "next_after_ordinal": None,
        },
    }
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def test_parse_exported_validation_result_and_cloudeck_unit_results() -> None:
    campaign = _campaign()
    snapshot = _snapshots(campaign)[0]
    result = _result(snapshot, 0.2)
    direct = parse_exported_cloudeck_result_bytes(_exported_result_bytes(result))
    assert direct[0]["schema"] == VALIDATION_RESULT_SCHEMA
    wrapped = parse_exported_cloudeck_result_bytes(_unit_results_bytes(result))
    assert wrapped[0]["snapshot_id"] == result.snapshot_id.value


def test_trusted_importer_restart_idempotent(tmp_path: Path) -> None:
    campaign = _campaign()
    snapshots = _snapshots(campaign)
    controller, drain, retention = _controller(tmp_path, campaign, snapshots)
    payload = _exported_result_bytes(_result(snapshots[0], 0.2))
    first = controller.import_exported_result(payload)
    revision = first.revision
    selection_revision = controller.selection.state.revision

    restored = TrustedController.restore(
        campaign,
        SelectionPolicy(10),
        controller.selection_store,
        controller.completion_store,
        drain=drain,
        retention=retention,
    )
    replay = restored.import_exported_result(payload)
    assert replay.revision == revision
    assert restored.selection.state.revision == selection_revision

    for snapshot in snapshots[1:]:
        restored.import_exported_result(_exported_result_bytes(_result(snapshot, 0.2)))
    restored.mark_trainer_terminal(MaxUpdatesReached())
    drain.drained = True
    restored.advance_completion()
    assert restored.completion.state.lifecycle is CampaignLifecycle.COMPLETED
    assert retention.receipts
    assert drain.cancelled


def test_importer_ranks_speculative_result_after_stop_and_refuses_stale_retention(tmp_path: Path) -> None:
    campaign = _campaign()
    snapshots = _snapshots(campaign)
    policy = SelectionPolicy(2)
    selection_store = TrustedStateStore(tmp_path / "selection.json", campaign, policy, initial_selection_state())
    completion_store = CampaignCompletionStateStore(tmp_path / "completion.json", campaign)
    drain = _Drain()
    retention = _Retention()
    controller = TrustedController.restore(
        campaign,
        policy,
        selection_store,
        completion_store,
        drain=drain,
        retention=retention,
    )
    for snapshot in snapshots[:4]:
        controller.publish_snapshot(snapshot)
    for snapshot, score in zip(snapshots[:3], (0.5, 0.6, 0.7), strict=True):
        controller.import_exported_result(_exported_result_bytes(_result(snapshot, score)))
    assert controller.selection.state.lifecycle is SelectionLifecycle.STOP_REQUESTED
    stop = controller.selection.stop_request
    assert stop is not None
    from recipes.speakrs.large.validation.contracts import AcceptedEarlyStop

    controller.mark_trainer_terminal(
        AcceptedEarlyStop(stop.selection_revision, stop.triggering_result, stop.best_snapshot)
    )
    controller.completion.commit_ranking_retention()
    controller.persist()
    assert controller.completion.state.retention_state is RetentionState.COMMITTED

    later = controller.import_exported_result(_exported_result_bytes(_result(snapshots[3], 0.1)))
    assert later.retention_state is RetentionState.PENDING
    assert controller.selection.state.top_five_snapshot_ids[0] == snapshots[3].snapshot_id
    drain.drained = True
    controller.advance_completion()
    assert controller.completion.state.lifecycle is CampaignLifecycle.COMPLETED
    assert retention.receipts[-1].snapshot_ids[0] == snapshots[3].snapshot_id
