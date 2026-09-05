#!/usr/bin/env python3

"""Regression tests for numerical and checkpointable trainer state."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from diarizen.trainer_utils import (
    AutoClipGradHistory,
    TrainerState,
    checkpoint_directory_is_complete,
    raise_for_non_finite_loss,
    reject_fp16_dual_optimizer,
    seal_checkpoint_directory,
)


REPO_DIR = Path(__file__).resolve().parents[3]
RECIPE_DIR = REPO_DIR / "recipes" / "diar_ssl"
TRAINER_PATHS = (
    RECIPE_DIR / "trainer_dual_opt.py",
    RECIPE_DIR / "trainer_single_opt.py",
    RECIPE_DIR.parent / "diar_ssl_mc" / "trainer_dual_opt.py",
    RECIPE_DIR.parent / "diar_ssl_pruning" / "trainer_dual_opt.py",
    RECIPE_DIR.parent / "diar_ssl_pruning" / "trainer_distill_prune.py",
)


def load_dual_optimizer_trainer():
    """Load the recipe trainer without depending on the process working directory."""
    sys.path.insert(0, str(RECIPE_DIR))
    spec = importlib.util.spec_from_file_location("diar_ssl_state_trainer", RECIPE_DIR / "trainer_dual_opt.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the dual-optimizer recipe trainer")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingOptimizer:
    """Record gradient clearing performed after an invalid loss."""

    def __init__(self):
        self.zero_grad_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1


class IdentityPowerset:
    """Provide the powerset methods used by the validation step."""

    @staticmethod
    def to_multilabel(value):
        return value

    @staticmethod
    def to_powerset(value):
        return value


class AccumulatingDERMetric:
    """Accumulate simple DER components, including false alarms on silent data."""

    def __init__(self):
        self.false_alarm = 0.0
        self.missed_detection = 0.0
        self.speech_total = 0.0
        self.update_calls = 0
        self.reset_calls = 0

    def update(self, predictions, target):
        self.update_calls += 1
        predictions = predictions >= 0.5
        target = target.bool()
        self.false_alarm += float((predictions & ~target).sum())
        self.missed_detection += float((~predictions & target).sum())
        self.speech_total += float(target.sum())

    def compute(self):
        denominator = self.speech_total + 1e-8
        der = (self.false_alarm + self.missed_detection) / denominator
        return {
            "DiarizationErrorRate": torch.tensor(der),
            "DiarizationErrorRate/FalseAlarm": torch.tensor(self.false_alarm / denominator),
            "DiarizationErrorRate/Miss": torch.tensor(self.missed_detection / denominator),
            "DiarizationErrorRate/Confusion": torch.tensor(0.0),
        }

    def reset(self):
        self.reset_calls += 1
        self.false_alarm = 0.0
        self.missed_detection = 0.0
        self.speech_total = 0.0


class TrainerStateTest(unittest.TestCase):
    """Cover numerical failure handling and state persistence."""

    def test_non_finite_loss_clears_all_optimizer_gradients(self):
        optimizers = (RecordingOptimizer(), RecordingOptimizer())

        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "Non-finite training loss"):
                raise_for_non_finite_loss(torch.tensor(value), optimizers, batch_idx=12)

        self.assertEqual([optimizer.zero_grad_calls for optimizer in optimizers], [3, 3])

    def test_auto_clip_history_roundtrip_is_bounded(self):
        history = AutoClipGradHistory(max_size=3)
        history.extend((1.0, 2.0, 3.0, 4.0))

        restored = AutoClipGradHistory(max_size=3)
        restored.load_state_dict(history.state_dict())

        self.assertEqual(history, [2.0, 3.0, 4.0])
        self.assertEqual(restored, history)

    def test_terminal_training_state_roundtrip(self):
        state = TrainerState(save_max_score=False)
        state.epochs_trained = 7
        state.training_complete = True

        restored = TrainerState(save_max_score=False)
        restored.load_state_dict(state.state_dict())

        self.assertEqual(restored.epochs_trained, 7)
        self.assertTrue(restored.training_complete)

    def test_resume_of_terminal_checkpoint_runs_no_training_batches(self):
        module = load_dual_optimizer_trainer()
        trainer = module.Trainer.__new__(module.Trainer)
        trainer.device = torch.device("cpu")
        trainer.gradient_accumulation_steps = 1
        trainer.max_steps = 0
        trainer.max_epochs = 8
        trainer.warmup_steps = 0
        trainer.use_one_cycle_lr = False
        trainer.resume = True
        trainer.validation_before_training = False
        trainer.state = TrainerState(save_max_score=False)
        trainer._load_checkpoint = lambda ckpt_path: setattr(trainer.state, "training_complete", True)
        trainer.set_models_to_train_mode = lambda: self.fail("terminal resume entered the training loop")
        trainer.accelerator = SimpleNamespace(num_processes=1)

        trainer.train([object()], validation_dataloader=None)

        self.assertEqual(trainer.state.steps_trained, 0)

    def test_checkpoint_manifest_rejects_missing_or_truncated_payload(self):
        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "epoch_0001"
            checkpoint.mkdir()
            model = checkpoint / "pytorch_model.bin"
            optimizer = checkpoint / "optimizer.bin"
            model.write_bytes(b"model-state")
            optimizer.write_bytes(b"optimizer-state")

            seal_checkpoint_directory(checkpoint)

            self.assertTrue(checkpoint_directory_is_complete(checkpoint, (model.name, optimizer.name)))
            optimizer.write_bytes(b"short")
            self.assertFalse(checkpoint_directory_is_complete(checkpoint, (model.name, optimizer.name)))

    def test_dual_optimizer_fp16_guard(self):
        with self.assertRaisesRegex(RuntimeError, "mixed_precision='fp16'"):
            reject_fp16_dual_optimizer(SimpleNamespace(mixed_precision="fp16"))

        reject_fp16_dual_optimizer(SimpleNamespace(mixed_precision="bf16"))
        reject_fp16_dual_optimizer(SimpleNamespace(mixed_precision="no"))

    def test_all_auto_clip_recipes_register_history(self):
        for path in TRAINER_PATHS:
            with self.subTest(path=path):
                source = path.read_text()
                self.assertIn("AutoClipGradHistory", source)
                self.assertIn("register_for_checkpointing(self.grad_history)", source)

    def test_validation_uses_aggregate_der_and_keeps_silent_false_alarm(self):
        module = load_dual_optimizer_trainer()
        metric = AccumulatingDERMetric()

        trainer = module.Trainer.__new__(module.Trainer)
        trainer.model = lambda value: value
        trainer.unwrap_model = SimpleNamespace(validation_metric=metric)
        trainer.unwrap_model.powerset = IdentityPowerset()
        trainer.state = SimpleNamespace(epochs_trained=1)
        trainer.accelerator = SimpleNamespace(is_local_main_process=True)
        logged = {}
        trainer.writer = SimpleNamespace(add_scalar=lambda name, value, epoch: logged.setdefault(name, value))
        trainer._write_validation_metrics = lambda values: None

        with (
            patch.object(module, "permutate", side_effect=lambda unused_prediction, expected: (expected, None)),
            patch.object(module, "nll_loss", return_value=torch.tensor(0.0)),
        ):
            outputs = [
                trainer.validation_step({"xs": torch.tensor([[[1.0, 0.0]]]), "ts": torch.tensor([[[1.0, 0.0]]])}, 0),
                trainer.validation_step({"xs": torch.tensor([[[1.0, 0.0]]]), "ts": torch.tensor([[[0.0, 0.0]]])}, 1),
            ]

        selected_loss = trainer.validation_epoch_end(
            [outputs[0] | {"Loss": torch.tensor(1.0)}, outputs[1] | {"Loss": torch.tensor(3.0)}]
        )

        self.assertAlmostEqual(selected_loss, 2.0)
        self.assertAlmostEqual(logged["Validation_Epoch/DER"], 1.0)
        self.assertAlmostEqual(logged["Validation_Epoch/FA"], 1.0)
        self.assertEqual(metric.update_calls, 2)
        self.assertEqual(metric.reset_calls, 1)
        self.assertEqual(metric.speech_total, 0.0)


if __name__ == "__main__":
    unittest.main()
