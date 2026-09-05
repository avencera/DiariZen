#!/usr/bin/env python3

"""Regression tests for empty output from the core DiariZen pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from diarizen.pipelines import inference


class SilentSegmentations:
    """Provide zero-valued segmentations for a pipeline edge-case test."""

    data = np.zeros((1, 4, 2), dtype=np.uint8)


class CoreInferenceEmptyRttmTest(unittest.TestCase):
    """Check that silent audio is represented by an empty completed RTTM."""

    def test_silent_session_returns_and_publishes_empty_annotation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            pipeline = inference.DiariZenPipeline.__new__(inference.DiariZenPipeline)
            pipeline.apply_median_filtering = False
            pipeline.rttm_out_dir = output_dir.as_posix()
            pipeline.get_segmentations = lambda unused_file, soft: SilentSegmentations()
            pipeline.speaker_count = lambda *unused_args, **unused_kwargs: self.fail(
                "silent segmentation must not reach speaker counting"
            )

            waveform = inference.torch.zeros((1, 32))
            with patch.object(inference.torchaudio, "load", return_value=(waveform, 16000)):
                result = pipeline("silent.wav", sess_name="silent")

            rttm_path = output_dir / "silent.rttm"
            self.assertEqual(result.uri, "silent")
            self.assertEqual(result.to_rttm(), "")
            self.assertTrue(rttm_path.is_file())
            self.assertEqual(rttm_path.read_text(), "")
            self.assertTrue(inference.rttm_completion_marker(rttm_path).is_file())


if __name__ == "__main__":
    unittest.main()
