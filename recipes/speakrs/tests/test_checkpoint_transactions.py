#!/usr/bin/env python3

"""Regression tests for crash-safe trainer checkpoint publication."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATHS = {
    "single": REPOSITORY_ROOT / "diarizen" / "trainer_single_opt.py",
    "dual": REPOSITORY_ROOT / "diarizen" / "trainer_dual_opt.py",
    "distill": REPOSITORY_ROOT / "diarizen" / "trainer_distill_prune.py",
}
CHECKPOINT_COMPLETE_MARKER = ".complete"


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def _module_stubs():
    """Provide import-only stubs so checkpoint tests do not need a GPU stack."""

    pandas = types.ModuleType("pandas")
    pandas.set_option = lambda *_args, **_kwargs: None

    toml = types.ModuleType("toml")
    toml.dump = lambda *_args, **_kwargs: None

    torch = types.ModuleType("torch")

    def no_grad():
        return lambda function: function

    torch.no_grad = no_grad
    torch.optim = types.ModuleType("torch.optim")
    torch.optim.lr_scheduler = types.ModuleType("torch.optim.lr_scheduler")
    torch.optim.lr_scheduler.OneCycleLR = object
    torch.optim.lr_scheduler.ReduceLROnPlateau = object
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.DataLoader = object

    accelerate = types.ModuleType("accelerate")
    accelerate.Accelerator = object
    accelerate_logging = types.ModuleType("accelerate.logging")
    accelerate_logging.get_logger = lambda *_args, **_kwargs: _Logger()

    torchinfo = types.ModuleType("torchinfo")
    torchinfo.summary = lambda *_args, **_kwargs: ""
    tqdm = types.ModuleType("tqdm")
    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = lambda values, **_kwargs: values

    diarizen = types.ModuleType("diarizen")
    diarizen.__path__ = [str(REPOSITORY_ROOT / "diarizen")]
    diarizen_logger = types.ModuleType("diarizen.logger")
    diarizen_logger.TensorboardLogger = object
    diarizen_noam = types.ModuleType("diarizen.noam_updater")
    diarizen_noam.get_rate = lambda *_args, **_kwargs: 0.0
    diarizen_optimization = types.ModuleType("diarizen.optimization")
    diarizen_optimization.get_constant_schedule_with_warmup = lambda *_args, **_kwargs: None
    diarizen_optimization.get_linear_schedule_with_warmup = lambda *_args, **_kwargs: None
    diarizen_trainer_utils = types.ModuleType("diarizen.trainer_utils")
    diarizen_trainer_utils.CHECKPOINT_COMPLETE_MARKER = ".complete"
    diarizen_trainer_utils.TrainerState = object
    diarizen_trainer_utils.checkpoint_directory_is_complete = lambda path, required=(): (
        path.is_dir() and (path / ".complete").is_file() and all((path / filename).is_file() for filename in required)
    )
    diarizen_trainer_utils.fsync_directory = lambda *_args, **_kwargs: None

    def seal_checkpoint_directory(path):
        (path / ".complete").write_text("complete\n")

    diarizen_trainer_utils.seal_checkpoint_directory = seal_checkpoint_directory
    diarizen_utils = types.ModuleType("diarizen.utils")
    diarizen_utils.prepare_empty_dir = lambda *_args, **_kwargs: None
    diarizen_utils.print_env = lambda: ""

    return {
        "pandas": pandas,
        "toml": toml,
        "torch": torch,
        "torch.optim": torch.optim,
        "torch.optim.lr_scheduler": torch.optim.lr_scheduler,
        "torch.utils": torch.utils,
        "torch.utils.data": torch.utils.data,
        "accelerate": accelerate,
        "accelerate.logging": accelerate_logging,
        "torchinfo": torchinfo,
        "tqdm": tqdm,
        "tqdm.auto": tqdm_auto,
        "diarizen": diarizen,
        "diarizen.logger": diarizen_logger,
        "diarizen.noam_updater": diarizen_noam,
        "diarizen.optimization": diarizen_optimization,
        "diarizen.trainer_utils": diarizen_trainer_utils,
        "diarizen.utils": diarizen_utils,
    }


def load_trainer_module(name):
    """Load one trainer with import-only dependency stubs."""

    path = TRAINER_PATHS[name]
    module_name = f"checkpoint_test_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer module {path}")

    module = types.ModuleType(module_name)
    with patch.dict(sys.modules, _module_stubs()):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


class FakeAccelerator:
    """Write small deterministic files while exposing Accelerate call order."""

    def __init__(self, fail_save_state=False, events=None):
        self.fail_save_state = fail_save_state
        self.events = events if events is not None else []
        self.save_state_paths = []

    def save_state(self, output_dir, safe_serialization):
        output_dir = Path(output_dir)
        self.save_state_paths.append(output_dir)
        self.events.append(("save_state", output_dir))
        (output_dir / "pytorch_model.bin").write_bytes(b"state")
        if self.fail_save_state:
            raise OSError("injected save failure")

    def get_state_dict(self, _model):
        return {"weight": 1}

    def save(self, _state_dict, output_path):
        self.events.append(("save_model", Path(output_path)))
        Path(output_path).write_bytes(b"model")


def trainer_for(module, root, accelerator=None):
    """Create a trainer shell with only checkpoint fields initialized."""

    trainer = module.Trainer.__new__(module.Trainer)
    trainer.checkpoints_dir = root / "checkpoints"
    trainer.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    trainer.ranked_checkpoints_dir = root / "ranked_checkpoints"
    trainer.accelerator = accelerator or FakeAccelerator()
    trainer.model = object()
    trainer.max_num_checkpoints = 10
    trainer.ranked_checkpoint_count = 0
    trainer.save_max_score = True
    trainer.state = SimpleNamespace(best_score=-float("inf"), best_score_epoch=0, patience=3, epochs_trained=4)
    return trainer


class CheckpointSelectionTest(unittest.TestCase):
    def test_latest_ignores_temp_and_incomplete_directories(self):
        module = load_trainer_module("single")
        with TemporaryDirectory() as temporary:
            trainer = trainer_for(module, Path(temporary))
            incomplete = trainer.checkpoints_dir / "epoch_0002"
            incomplete.mkdir()
            (incomplete / "pytorch_model.bin").write_bytes(b"partial")

            temp = trainer.checkpoints_dir / ".epoch_0003.partial"
            temp.mkdir()
            (temp / CHECKPOINT_COMPLETE_MARKER).write_text("complete\n")

            complete = trainer.checkpoints_dir / "epoch_0001"
            complete.mkdir()
            (complete / "pytorch_model.bin").write_bytes(b"complete")
            (complete / CHECKPOINT_COMPLETE_MARKER).write_text("complete\n")

            legacy = trainer.checkpoints_dir / "epoch_0004"
            legacy.mkdir()
            (legacy / "pytorch_model.bin").write_bytes(b"legacy")

            self.assertEqual(trainer._find_latest_ckpt_path(), complete)


class TransactionalCheckpointTest(unittest.TestCase):
    def test_regular_checkpoint_publishes_only_after_save_returns(self):
        module = load_trainer_module("single")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "checkpoints" / "epoch_0001"
            old.mkdir(parents=True)
            (old / CHECKPOINT_COMPLETE_MARKER).write_text("complete\n")

            failing_accelerator = FakeAccelerator(fail_save_state=True)
            trainer = trainer_for(module, root, failing_accelerator)
            with self.assertRaises(OSError):
                trainer._save_checkpoint(epoch=2, is_best_epoch=False)
            self.assertTrue(old.exists())
            self.assertFalse((root / "checkpoints" / "epoch_0002").exists())
            self.assertFalse((root / "checkpoints" / ".epoch_0002.partial" / CHECKPOINT_COMPLETE_MARKER).exists())

            successful_accelerator = FakeAccelerator()
            trainer.accelerator = successful_accelerator
            trainer._save_checkpoint(epoch=2, is_best_epoch=False)
            published = root / "checkpoints" / "epoch_0002"
            self.assertTrue((published / CHECKPOINT_COMPLETE_MARKER).is_file())
            self.assertFalse((root / "checkpoints" / ".epoch_0002.partial").exists())
            self.assertEqual(successful_accelerator.save_state_paths, [root / "checkpoints" / ".epoch_0002.partial"])

    def test_incomplete_update_generation_is_not_resumed(self):
        module = load_trainer_module("dual")
        with TemporaryDirectory() as temporary:
            trainer = trainer_for(module, Path(temporary))
            complete = trainer.checkpoints_dir / "update_00000250"
            complete.mkdir()
            (complete / "pytorch_model.bin").write_bytes(b"complete")
            (complete / CHECKPOINT_COMPLETE_MARKER).write_text("complete\n")
            incomplete = trainer.checkpoints_dir / "update_00000500"
            incomplete.mkdir()
            (incomplete / "pytorch_model.bin").write_bytes(b"partial")
            self.assertEqual(trainer._find_latest_ckpt_path(), complete)

    def test_best_replacement_keeps_previous_checkpoint_on_save_failure(self):
        module = load_trainer_module("single")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            best = root / "checkpoints" / "best"
            best.mkdir(parents=True)
            (best / CHECKPOINT_COMPLETE_MARKER).write_text("old\n")
            trainer = trainer_for(module, root, FakeAccelerator(fail_save_state=True))

            with self.assertRaises(OSError):
                trainer._save_checkpoint(epoch=2, is_best_epoch=True)

            self.assertTrue(best.exists())
            self.assertEqual((best / CHECKPOINT_COMPLETE_MARKER).read_text(), "old\n")


class RankedCheckpointTest(unittest.TestCase):
    def test_index_is_published_before_unreferenced_directory_is_deleted(self):
        module = load_trainer_module("dual")
        events = []
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            accelerator = FakeAccelerator(events=events)
            trainer = trainer_for(module, root, accelerator)
            trainer.ranked_checkpoint_count = 1
            trainer.ranked_checkpoints_dir.mkdir()

            old = trainer.ranked_checkpoints_dir / "epoch_0001"
            old.mkdir()
            (old / "pytorch_model.bin").write_bytes(b"old")
            trainer._write_checkpoint_completion_marker(old)
            index = trainer.ranked_checkpoints_dir / "index.json"
            index.write_text(json.dumps([{"epoch": 1, "score": 0.1}]) + "\n")

            original_replace = Path.replace

            def track_replace(path, target):
                if path == trainer.ranked_checkpoints_dir / "index.partial.json":
                    events.append(("index_publish", path))
                return original_replace(path, target)

            original_remove = trainer._remove_path

            def track_remove(path):
                events.append(("remove", Path(path)))
                original_remove(path)

            trainer._remove_path = track_remove
            with patch.object(Path, "replace", track_replace):
                trainer._save_ranked_checkpoint(epoch=2, score=0.2)

            index_event = next(index for index, event in enumerate(events) if event[0] == "index_publish")
            remove_event = next(
                index for index, event in enumerate(events) if event[0] == "remove" and event[1].name == "epoch_0001"
            )
            self.assertLess(index_event, remove_event)
            self.assertEqual(json.loads((trainer.ranked_checkpoints_dir / "index.json").read_text())[0]["epoch"], 2)
            self.assertTrue((trainer.ranked_checkpoints_dir / "epoch_0002" / "pytorch_model.bin").is_file())

    def test_index_failure_keeps_previous_index_and_referenced_directory(self):
        module = load_trainer_module("dual")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = trainer_for(module, root, FakeAccelerator())
            trainer.ranked_checkpoint_count = 1
            trainer.ranked_checkpoints_dir.mkdir()

            old = trainer.ranked_checkpoints_dir / "epoch_0001"
            old.mkdir()
            (old / "pytorch_model.bin").write_bytes(b"old")
            trainer._write_checkpoint_completion_marker(old)
            index = trainer.ranked_checkpoints_dir / "index.json"
            index.write_text(json.dumps([{"epoch": 1, "score": 0.1}]) + "\n")

            original_replace = Path.replace

            def fail_index_replace(path, target):
                if path == trainer.ranked_checkpoints_dir / "index.partial.json":
                    raise OSError("injected index publication failure")
                return original_replace(path, target)

            with patch.object(Path, "replace", fail_index_replace):
                with self.assertRaises(OSError):
                    trainer._save_ranked_checkpoint(epoch=2, score=0.2)

            self.assertEqual(json.loads(index.read_text())[0]["epoch"], 1)
            self.assertTrue(old.exists())
            self.assertTrue((trainer.ranked_checkpoints_dir / "epoch_0002" / "pytorch_model.bin").is_file())


class ResumeStateOrderingTest(unittest.TestCase):
    def test_patience_is_reset_before_best_checkpoint_save(self):
        for name in TRAINER_PATHS:
            with self.subTest(name=name):
                module = load_trainer_module(name)
                with TemporaryDirectory() as temporary:
                    trainer = trainer_for(module, Path(temporary))
                    saved_patience = []
                    trainer._check_improvement = lambda *_args, **_kwargs: True
                    trainer._save_checkpoint = lambda *_args, **_kwargs: saved_patience.append(trainer.state.patience)

                    trainer._run_early_stop_check(1.0)

                    self.assertEqual(saved_patience, [0])

    def test_regular_checkpoint_call_follows_validation_block(self):
        for name, path in TRAINER_PATHS.items():
            with self.subTest(name=name):
                train_function = next(
                    node
                    for node in ast.walk(ast.parse(path.read_text()))
                    if isinstance(node, ast.FunctionDef) and node.name == "train"
                )
                regular_save = next(
                    node
                    for node in ast.walk(train_function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_save_checkpoint"
                    and any(
                        keyword.arg == "is_best_epoch"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is False
                        for keyword in node.keywords
                    )
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "epoch"
                )
                validation_block = next(
                    node
                    for node in ast.walk(train_function)
                    if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.BinOp)
                    and isinstance(node.test.left.left, ast.Name)
                    and node.test.left.left.id == "epoch"
                )
                self.assertGreater(regular_save.lineno, validation_block.end_lineno)


if __name__ == "__main__":
    unittest.main()
