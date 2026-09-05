#!/usr/bin/env python3

"""Integration test for the active dual-optimizer checkpoint contract."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from accelerate import Accelerator

from diarizen.trainer_dual_opt import Trainer
from diarizen.trainer_utils import CHECKPOINT_COMPLETE_MARKER, AutoClipGradHistory, TrainerState


class DualOptimizerCheckpointRoundtripTest(unittest.TestCase):
    """Check publication and restoration with the pinned Accelerate runtime."""

    def assert_optimizer_state_equal(self, actual, expected):
        """Assert equality for optimizer metadata and tensor-valued moments."""

        self.assertEqual(actual["param_groups"], expected["param_groups"])
        self.assertEqual(actual["state"].keys(), expected["state"].keys())
        for parameter_id, expected_state in expected["state"].items():
            actual_state = actual["state"][parameter_id]
            self.assertEqual(actual_state.keys(), expected_state.keys())
            for name, expected_value in expected_state.items():
                actual_value = actual_state[name]
                if torch.is_tensor(expected_value):
                    torch.testing.assert_close(actual_value, expected_value)
                else:
                    self.assertEqual(actual_value, expected_value)

    def test_checkpoint_restores_model_optimizers_and_custom_state(self):
        accelerator = Accelerator(cpu=True)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        optimizer_small = torch.optim.AdamW(model[0].parameters(), lr=2e-5)
        optimizer_big = torch.optim.AdamW(model[1].parameters(), lr=1e-3)
        model, optimizer_small, optimizer_big = accelerator.prepare(model, optimizer_small, optimizer_big)

        state = TrainerState(save_max_score=False)
        state.epochs_trained = 3
        state.patience = 2
        history = AutoClipGradHistory(max_size=3)
        history.extend((1.0, 2.0, 3.0))
        accelerator.register_for_checkpointing(state)
        accelerator.register_for_checkpointing(history)

        trainer = Trainer.__new__(Trainer)
        trainer.accelerator = accelerator

        inputs = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
        loss = model(inputs).square().mean()
        accelerator.backward(loss)
        optimizer_small.step()
        optimizer_big.step()
        optimizer_small.zero_grad()
        optimizer_big.zero_grad()

        original_parameter = next(model.parameters()).detach().clone()
        original_small_state = copy.deepcopy(optimizer_small.state_dict())
        original_big_state = copy.deepcopy(optimizer_big.state_dict())
        self.assertTrue(original_small_state["state"])
        self.assertTrue(original_big_state["state"])

        with TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "epoch_0003"
            trainer._save_accelerate_checkpoint(destination)

            self.assertTrue((destination / CHECKPOINT_COMPLETE_MARKER).is_file())
            self.assertTrue((destination / ".files.json").is_file())
            self.assertTrue((destination / "pytorch_model.bin").is_file())
            self.assertTrue((destination / "optimizer.bin").is_file())
            self.assertTrue((destination / "optimizer_1.bin").is_file())
            self.assertTrue((destination / "custom_checkpoint_0.pkl").is_file())
            self.assertTrue((destination / "custom_checkpoint_1.pkl").is_file())

            with torch.no_grad():
                next(model.parameters()).add_(10)
            state.epochs_trained = 99
            state.patience = 99
            history.clear()
            optimizer_small.optimizer.state.clear()
            optimizer_big.optimizer.state.clear()
            optimizer_small.param_groups[0]["lr"] = 9.0
            optimizer_big.param_groups[0]["lr"] = 9.0

            accelerator.load_state(destination, map_location="cpu")

        torch.testing.assert_close(next(model.parameters()), original_parameter)
        self.assertEqual(state.epochs_trained, 3)
        self.assertEqual(state.patience, 2)
        self.assertEqual(history, [1.0, 2.0, 3.0])
        self.assert_optimizer_state_equal(optimizer_small.state_dict(), original_small_state)
        self.assert_optimizer_state_equal(optimizer_big.state_dict(), original_big_state)


if __name__ == "__main__":
    unittest.main()
