#!/usr/bin/env python3

"""Runtime, recovery, and trainer-hook tests that drive shipped owners."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from diarizen.trainer_utils import TrainerState, seal_checkpoint_directory  # noqa: E402
from recipes.speakrs.large.budget import BudgetLedger  # noqa: E402
from recipes.speakrs.large.contracts import DEFAULT_BUDGET, LicenceDecision, RecordingRow  # noqa: E402
from recipes.speakrs.large.hooks import LargeRunHooks  # noqa: E402
from recipes.speakrs.large.recovery import (  # noqa: E402
    copy_generation,
    newest_complete_generation,
    restore_into,
)
from recipes.speakrs.large.sampler import MixtureSampler  # noqa: E402


def _rows():
    rows = []
    for corpus in (
        "AMI",
        "AliMeeting",
        "AISHELL4",
        "VoxConverse",
        "NOTSOFAR_real",
        "ICSI",
        "LOTUSDIS",
        "NOTSOFAR_sim",
    ):
        rows.append(
            RecordingRow(
                recording_id=f"{corpus}-0",
                parent_id=f"{corpus}-0",
                corpus=corpus,
                split="train",
                device_view="canonical",
                label_tier="gold",
                licence=LicenceDecision.ACCEPTED_CC,
                audio_sha256="a" * 64,
                label_sha256="b" * 64,
                sample_count=16000,
                rejected=False,
                rejection_reason=None,
            )
        )
    return rows


class TrainerStateCompatTest(unittest.TestCase):
    def test_base_plus_checkpoint_loads_without_large_fields(self):
        state = TrainerState(save_max_score=False)
        state.load_state_dict(
            {
                "epochs_trained": 3,
                "steps_trained": 10,
                "training_complete": False,
                "patience": 1,
                "best_score": 0.2,
                "best_score_epoch": 2,
            }
        )
        self.assertEqual(state.updates_trained, 0)
        self.assertEqual(state.recipe_state, {})
        dumped = state.state_dict()
        self.assertIn("updates_trained", dumped)


class CoverageHookTest(unittest.TestCase):
    def test_hooks_only_commit_on_successful_update(self):
        sampler = MixtureSampler.from_rows(_rows(), seed=3407)
        hooks = LargeRunHooks(sampler=sampler, ledger=BudgetLedger(DEFAULT_BUDGET))
        example = {"stream": "AMI", "parent_id": "AMI-0", "counts_for_coverage": True, "dynamic": False}
        hooks.set_pending(example)
        self.assertFalse(
            hooks.acknowledge_update({"names": ["AMI-0"]}, optimizer_updated=False, loss_weight=1.0, valid=True)
        )
        hooks.set_pending(example)
        self.assertTrue(
            hooks.acknowledge_update({"names": ["AMI-0"]}, optimizer_updated=True, loss_weight=1.0, valid=True)
        )
        restored = LargeRunHooks.from_state_dict(hooks.state_dict(), sampler, hooks.ledger)
        self.assertIn("AMI-0", restored.sampler.coverage.seen["AMI"])


class BackupTransactionTest(unittest.TestCase):
    def test_interrupted_and_corrupt_backup_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation = root / "worker" / "update_00000250"
            generation.mkdir(parents=True)
            (generation / "optimizer_small.bin").write_bytes(b"small")
            (generation / "optimizer_big.bin").write_bytes(b"big")
            (generation / "rng.json").write_text("{}\n", encoding="utf-8")
            seal_checkpoint_directory(generation)
            trusted = root / "trusted" / generation.name
            copy_generation(generation, trusted)
            self.assertEqual(newest_complete_generation(root / "worker"), generation)
            restored = restore_into(root / "trusted", root / "fresh")
            self.assertTrue(restored.exists())
            (trusted / "optimizer_small.bin").write_bytes(b"corrupt")
            with self.assertRaises(Exception):
                copy_generation(trusted, root / "reuse")
            partial = root / "worker" / "update_00000500"
            partial.mkdir()
            (partial / "optimizer_small.bin").write_bytes(b"partial")
            self.assertEqual(newest_complete_generation(root / "worker"), generation)


class SharedTrainerDiscoveryTest(unittest.TestCase):
    def test_latest_checkpoint_discovers_update_generations(self):
        from recipes.speakrs.tests.test_checkpoint_transactions import load_trainer_module, trainer_for

        module = load_trainer_module("dual")
        with tempfile.TemporaryDirectory() as temporary:
            trainer = trainer_for(module, Path(temporary))
            epoch = trainer.checkpoints_dir / "epoch_0001"
            epoch.mkdir()
            (epoch / "pytorch_model.bin").write_bytes(b"epoch")
            (epoch / ".complete").write_text("complete\n")
            update = trainer.checkpoints_dir / "update_00000250"
            update.mkdir()
            (update / "pytorch_model.bin").write_bytes(b"update")
            (update / ".complete").write_text("complete\n")
            self.assertEqual(trainer._find_latest_ckpt_path(), update)


if __name__ == "__main__":
    unittest.main()
