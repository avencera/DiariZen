"""Trainer-bridge and external-trainer gate tests."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from recipes.speakrs.large.recovery import generation_name
from recipes.speakrs.large.validation.contracts import (
    MaxUpdatesReached,
    PublishedSnapshot,
    Sha256Digest,
    ValidationResult,
)
from recipes.speakrs.large.validation.controller import (
    LagWaitReason,
    SelectionController,
    SelectionPolicy,
    TrustedStateStore,
)
from recipes.speakrs.large.validation.slot_plan import build_validation_campaign
from recipes.speakrs.large.validation.trainer_bridge import (
    AcceptEarlyStop,
    ContinueTraining,
    ReachMaximumUpdates,
    TrainerBridge,
    WaitForValidation,
)


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _campaign(max_updates: int = 6, maximum_validation_lag: int = 1, updates_per_complete_epoch: int = 2):
    return build_validation_campaign(
        training_launch_id="launch-trainer-gate",
        max_updates=max_updates,
        updates_per_complete_epoch=updates_per_complete_epoch,
        artifact_prefix="s3://validation-trainer",
        dev_bundle_digest=_digest("a"),
        trainer_configuration_digest=_digest("b"),
        evaluator_image_identity="registry.example/evaluator@sha256:" + "c" * 64,
        evaluator_implementation_digest=_digest("d"),
        maximum_validation_lag=maximum_validation_lag,
    )


def _snapshot(campaign, slot, index: int) -> PublishedSnapshot:
    hex_characters = "123456789abcdef0"
    return PublishedSnapshot(
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
        publication_time=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index),
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


class _Publisher:
    def __init__(self, snapshots: dict[Sha256Digest, PublishedSnapshot]) -> None:
        self._snapshots = snapshots
        self.published_updates: list[int] = []

    def publish(self, campaign, slot, generation_directory, progress_digest):
        snapshot = self._snapshots[slot.slot_id]
        self.published_updates.append(slot.updates)
        return snapshot


def _generation_directory(root: Path, updates: int) -> Path:
    path = root / generation_name(updates, 1)
    path.mkdir(parents=True)
    (path / "progress.json").write_text(json.dumps({"updates": updates}, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _bridge(tmp_path: Path, campaign, snapshots, policy: SelectionPolicy | None = None) -> TrainerBridge:
    store = TrustedStateStore(tmp_path / "selection.json", campaign, policy or SelectionPolicy(4))
    publisher = _Publisher({snapshot.slot_id: snapshot for snapshot in snapshots})
    return TrainerBridge(campaign, store, publisher, wait_for_state=lambda: None)


def test_epoch_zero_must_be_applied_before_epoch_one(tmp_path: Path) -> None:
    campaign = _campaign()
    snapshots = tuple(_snapshot(campaign, slot, index) for index, slot in enumerate(campaign.slots))
    controller = SelectionController(campaign, SelectionPolicy(4), snapshots)
    store = TrustedStateStore(tmp_path / "selection.json", campaign, SelectionPolicy(4))
    publisher = _Publisher({snapshot.slot_id: snapshot for snapshot in snapshots})
    bridge = TrainerBridge(campaign, store, publisher, wait_for_state=lambda: None)
    bridge.publish_boundary(_generation_directory(tmp_path, 0))

    waiting = bridge.before_epoch(1)
    assert isinstance(waiting, WaitForValidation)
    assert waiting.reason is LagWaitReason.EPOCH_ZERO_PENDING

    store.write(controller.import_result(_result(snapshots[0], 0.5).to_dict()), controller.stop_request)
    permitted = bridge.before_epoch(1)
    assert isinstance(permitted, ContinueTraining)


def test_lag_one_trainer_sequence(tmp_path: Path) -> None:
    campaign = _campaign(max_updates=6, maximum_validation_lag=1)
    snapshots = tuple(_snapshot(campaign, slot, index) for index, slot in enumerate(campaign.slots))
    policy = SelectionPolicy(4)
    controller = SelectionController(campaign, policy, snapshots)
    store = TrustedStateStore(tmp_path / "selection.json", campaign, policy)
    publisher = _Publisher({snapshot.slot_id: snapshot for snapshot in snapshots})
    waits = []
    bridge = TrainerBridge(campaign, store, publisher, wait_for_state=lambda: waits.append("wait"))

    bridge.publish_boundary(_generation_directory(tmp_path, 0))
    store.write(controller.import_result(_result(snapshots[0], 0.5).to_dict()), controller.stop_request)
    assert isinstance(bridge.await_epoch_start(1), ContinueTraining)

    bridge.publish_boundary(_generation_directory(tmp_path, 2))
    assert isinstance(bridge.before_epoch(2), ContinueTraining)

    bridge.publish_boundary(_generation_directory(tmp_path, 4))
    blocked = bridge.before_epoch(3)
    assert isinstance(blocked, WaitForValidation)
    assert blocked.reason is LagWaitReason.VALIDATION_LAG_EXCEEDED
    assert waits == []

    store.write(controller.import_result(_result(snapshots[1], 0.4).to_dict()), controller.stop_request)
    assert isinstance(bridge.await_epoch_start(3), ContinueTraining)


def test_waiting_for_lag_does_not_publish_or_change_snapshots(tmp_path: Path) -> None:
    campaign = _campaign()
    snapshots = tuple(_snapshot(campaign, slot, index) for index, slot in enumerate(campaign.slots))
    bridge = _bridge(tmp_path, campaign, snapshots)
    bridge.publish_boundary(_generation_directory(tmp_path, 0))
    before = bridge.published_snapshots
    decision = bridge.before_epoch(1)
    assert isinstance(decision, WaitForValidation)
    assert bridge.published_snapshots == before
    assert bridge.publisher.published_updates == [0]


def _external_trainer(bridge: TrainerBridge, *, batches: int, max_steps: int):
    from diarizen.trainer_dual_opt import Trainer
    from diarizen.trainer_utils import TrainerState

    trainer = Trainer.__new__(Trainer)
    trainer.validation_bridge = bridge
    trainer.model = object()
    trainer.launch_id = bridge.campaign.training_launch_id
    trainer.gradient_accumulation_steps = 1
    trainer.max_steps = max_steps
    trainer.warmup_steps = 0
    trainer.use_one_cycle_lr = False
    trainer.resume = False
    trainer.freeze_wavlm = False
    trainer._stop_signal = None
    trainer.run_hooks = None
    trainer.state = TrainerState(save_max_score=False)
    trainer.accelerator = SimpleNamespace(
        is_local_main_process=True,
        wait_for_everyone=lambda: None,
        sync_gradients=True,
        optimizer_step_was_skipped=False,
        num_processes=1,
        accumulate=lambda _model: _null_context(),
    )
    trainer.set_models_to_train_mode = lambda: None
    trainer.training_step = lambda batch, batch_idx: {"Loss": 0.0}
    trainer.training_epoch_end = lambda _output: None
    trainer._prepare_epoch_dataloader = lambda loader, _steps: (loader, 0)
    trainer._install_stop_signals = lambda: None
    trainer._train_data_generator = lambda _loader: None
    trainer.published: list[int] = []
    trainer.terminals: list[object] = []

    def publish(updates: int):
        trainer.published.append(updates)
        return Path("unused")

    def save_generation(*, updates: int, completion=None, failure=None):
        trainer.terminals.append((updates, completion, failure))
        return Path("unused")

    trainer._external_publish_generation = publish
    trainer._save_external_generation = save_generation
    trainer._external_terminal_generation = lambda updates, completion=None, failure=None: trainer.terminals.append(
        (updates, completion, failure)
    )
    trainer._external_terminal_decision = lambda decision: trainer.terminals.append(decision)

    class _Loader:
        def __len__(self) -> int:
            return batches

        def __iter__(self):
            return iter([{"xs": index, "ts": index} for index in range(batches)])

    return trainer, _Loader()


@contextmanager
def _null_context():
    yield


def test_mid_epoch_early_stop_invents_no_slot(tmp_path: Path) -> None:
    campaign = _campaign(max_updates=6, updates_per_complete_epoch=2)
    snapshots = tuple(_snapshot(campaign, slot, index) for index, slot in enumerate(campaign.slots))
    policy = SelectionPolicy(2)
    controller = SelectionController(campaign, policy, snapshots)
    store = TrustedStateStore(tmp_path / "selection.json", campaign, policy)
    publisher = _Publisher({snapshot.slot_id: snapshot for snapshot in snapshots})
    for snapshot, score in zip(snapshots[:3], (0.5, 0.6, 0.7), strict=True):
        controller.import_result(_result(snapshot, score).to_dict())
    store.write(controller.state, controller.stop_request)
    bridge = TrainerBridge(campaign, store, publisher, wait_for_state=lambda: None)

    import diarizen.trainer_dual_opt as trainer_mod
    from diarizen.trainer_dual_opt import Trainer

    trainer_mod.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    trainer, loader = _external_trainer(bridge, batches=2, max_steps=6)
    seen_updates = []

    def observe_after_progress():
        if trainer.state.updates_trained >= 1:
            return bridge._early_stop_decision()
        return ContinueTraining(bridge.selection)

    bridge.has_published_boundary = lambda updates: updates == 0
    bridge.await_epoch_zero = lambda: ContinueTraining(bridge.selection)
    bridge.await_epoch_start = lambda epoch: ContinueTraining(bridge.selection)
    bridge.observe_selection = observe_after_progress
    trainer._external_publish_generation = lambda updates: seen_updates.append(updates) or Path("unused")
    Trainer._train_external(trainer, loader)
    trainer.published = seen_updates

    assert trainer.published == []
    assert trainer.terminals
    decision = trainer.terminals[0]
    assert isinstance(decision, AcceptEarlyStop)


def test_max_updates_saves_terminal_generation(tmp_path: Path) -> None:
    campaign = _campaign(max_updates=2, updates_per_complete_epoch=2)
    snapshots = tuple(_snapshot(campaign, slot, index) for index, slot in enumerate(campaign.slots))
    policy = SelectionPolicy(4)
    controller = SelectionController(campaign, policy, snapshots)
    store = TrustedStateStore(tmp_path / "selection.json", campaign, policy)
    for snapshot in snapshots:
        controller.register_snapshot(snapshot)
    store.write(controller.import_result(_result(snapshots[0], 0.5).to_dict()))
    publisher = _Publisher({snapshot.slot_id: snapshot for snapshot in snapshots})
    bridge = TrainerBridge(campaign, store, publisher, wait_for_state=lambda: None)

    from diarizen.trainer_dual_opt import Trainer

    trainer, loader = _external_trainer(bridge, batches=2, max_steps=2)
    trainer.published = []
    trainer.terminals = []
    trainer._external_publish_generation = lambda updates: trainer.published.append(updates) or Path("unused")
    trainer._external_terminal_generation = lambda updates, completion=None, failure=None: trainer.terminals.append(
        (updates, completion, failure)
    )
    trainer._external_terminal_decision = lambda decision: trainer.terminals.append(decision)
    bridge.has_published_boundary = lambda updates: updates == 0
    bridge.await_epoch_zero = lambda: ContinueTraining(bridge.selection)
    bridge.await_epoch_start = lambda epoch: ContinueTraining(bridge.selection)
    bridge.observe_selection = lambda: ContinueTraining(bridge.selection)
    bridge.await_final_result = lambda updates: ReachMaximumUpdates(MaxUpdatesReached(), bridge.selection)
    Trainer._train_external(trainer, loader)

    assert 2 in trainer.published
    assert trainer.terminals
    updates, completion, failure = trainer.terminals[0]
    assert updates == 2
    assert isinstance(completion, MaxUpdatesReached)
    assert failure is None
    assert isinstance(trainer.validation_bridge.selection, type(bridge.selection))
