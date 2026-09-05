#!/usr/bin/env python3

"""Regression tests for the DiariZen dual-optimizer training step."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from accelerate import Accelerator


RECIPE_DIR = Path(__file__).resolve().parents[2] / "diar_ssl"
TRAINING_STEP_PATHS = (
    RECIPE_DIR / "trainer_dual_opt.py",
    RECIPE_DIR / "trainer_single_opt.py",
    RECIPE_DIR.parent / "diar_ssl_mc" / "trainer_dual_opt.py",
    RECIPE_DIR.parent / "diar_ssl_pruning" / "trainer_dual_opt.py",
    RECIPE_DIR.parent / "diar_ssl_pruning" / "trainer_distill_prune.py",
)


def load_recipe_trainer_module():
    """Load the recipe trainer without depending on the process working directory."""

    sys.path.insert(0, str(RECIPE_DIR))
    spec = importlib.util.spec_from_file_location("diar_ssl_trainer_dual_opt", RECIPE_DIR / "trainer_dual_opt.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the dual-optimizer recipe trainer")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINER_MODULE = load_recipe_trainer_module()


class IdentityPowerset:
    """Provide the powerset methods used by the recipe training step."""

    @staticmethod
    def to_multilabel(value):
        return value

    @staticmethod
    def to_powerset(value):
        return value


class ToyModel(torch.nn.Module):
    """Expose two parameters so both recipe optimizers take part in the test."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.offset = torch.nn.Parameter(torch.tensor(0.0))
        self.powerset = IdentityPowerset()

    def forward(self, value):
        prediction = self.scale * value + self.offset
        return torch.cat((prediction, -prediction), dim=-1)


class GradientAccumulationTest(unittest.TestCase):
    """Check that every micro-batch contributes to one optimizer update."""

    def run_training_steps(self, values, accumulation_steps):
        """Run the real recipe step and return its two updated parameters."""

        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation_steps)
        model = ToyModel()
        optimizer_small = torch.optim.SGD([model.scale], lr=0.01)
        optimizer_big = torch.optim.SGD([model.offset], lr=0.01)
        model, optimizer_small, optimizer_big = accelerator.prepare(model, optimizer_small, optimizer_big)

        trainer = TRAINER_MODULE.Trainer.__new__(TRAINER_MODULE.Trainer)
        trainer.accelerator = accelerator
        trainer.model = model
        trainer.unwrap_model = accelerator.unwrap_model(model)
        trainer.optimizer_small = optimizer_small
        trainer.optimizer_big = optimizer_big
        trainer.auto_clip_grad_norm_ = lambda unused_model: None

        optimizer_small.zero_grad()
        optimizer_big.zero_grad()
        target = torch.tensor([[[1.0, 0.0]]])

        def squared_first_logit(prediction, unused_target):
            return prediction[..., 0].square().mean()

        with (
            patch.object(
                TRAINER_MODULE,
                "permutate",
                side_effect=lambda unused_prediction, expected: (expected, None),
            ),
            patch.object(TRAINER_MODULE, "nll_loss", side_effect=squared_first_logit),
        ):
            for value in values:
                batch = {
                    "xs": torch.tensor([[[value]]]),
                    "ts": target,
                }
                with accelerator.accumulate(model):
                    trainer.training_step(batch, batch_idx=0)

        unwrapped = accelerator.unwrap_model(model)
        return unwrapped.scale.item(), unwrapped.offset.item()

    def test_training_step_preserves_accumulated_gradients(self):
        scale, offset = self.run_training_steps(values=(1.0, 2.0, 3.0, 4.0), accumulation_steps=4)

        self.assertAlmostEqual(scale, 0.85, places=6)
        self.assertAlmostEqual(offset, -0.05, places=6)

    def test_single_step_training_is_unchanged(self):
        scale, offset = self.run_training_steps(values=(2.0,), accumulation_steps=1)

        self.assertAlmostEqual(scale, 0.92, places=6)
        self.assertAlmostEqual(offset, -0.04, places=6)

    def test_all_recipe_trainers_clear_gradients_after_optimizer_steps(self):
        for path in TRAINING_STEP_PATHS:
            with self.subTest(path=path):
                module = ast.parse(path.read_text())
                training_step = next(
                    node
                    for node in ast.walk(module)
                    if isinstance(node, ast.FunctionDef) and node.name == "training_step"
                )
                calls = [
                    node
                    for node in ast.walk(training_step)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                ]
                step_lines = [node.lineno for node in calls if node.func.attr == "step"]
                zero_grad_lines = [node.lineno for node in calls if node.func.attr == "zero_grad"]

                self.assertTrue(step_lines)
                self.assertTrue(zero_grad_lines)
                self.assertGreater(min(zero_grad_lines), max(step_lines))

    def test_partial_final_accumulation_still_updates_both_optimizers(self):
        scale, offset = self.run_training_steps(values=(1.0, 2.0, 3.0), accumulation_steps=2)

        self.assertNotEqual(scale, 1.0)
        self.assertNotEqual(offset, 0.0)


if __name__ == "__main__":
    unittest.main()
