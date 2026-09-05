#!/usr/bin/env python3

"""Regression tests for active-speaker capacity collation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


DATASET_PATH = Path(__file__).resolve().parents[2] / "diar_ssl" / "dataset.py"


def load_dataset_module():
    """Load the dataset module without depending on the process working directory."""

    spec = importlib.util.spec_from_file_location("diar_ssl_dataset", DATASET_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the diarization dataset module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATASET_MODULE = load_dataset_module()


class SpeakerCapacityTest(unittest.TestCase):
    """Check that inactive speakers do not consume model speaker capacity."""

    def test_uint8_activity_keeps_all_nonempty_columns(self):
        x = np.zeros((221, 2), dtype=np.float32)
        y = np.zeros((221, 5), dtype=np.uint8)
        y[:50, 1] = 1
        y[:47, 2] = 1
        y[:221, 3] = 1
        y[:51, 4] = 1

        result = DATASET_MODULE._collate_fn([(x, y, "session")], max_speakers_per_chunk=4)
        retained_activity = result["ts"][0].numpy().sum(axis=0).tolist()

        self.assertEqual(result["ts"].shape, (1, 221, 4))
        self.assertEqual(sorted(retained_activity), [47, 50, 51, 221])

    def test_active_columns_are_padded_after_empty_columns_are_removed(self):
        x = np.zeros((2, 1), dtype=np.float32)
        y = np.array([[0, 25, 0]], dtype=np.uint8)

        result = DATASET_MODULE._collate_fn([(x, y, "session")], max_speakers_per_chunk=4)

        self.assertEqual(result["ts"][0].numpy().sum(axis=0).tolist(), [25, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
