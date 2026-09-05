#!/usr/bin/env python3

"""Tests for resumable DiariZen inference run identity."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


RECIPE_DIR = Path(__file__).resolve().parents[2] / "diar_ssl"


def load_inference_module():
    """Load the inference script without depending on the working directory."""

    sys.path.insert(0, str(RECIPE_DIR))
    spec = importlib.util.spec_from_file_location("diar_ssl_infer_avg", RECIPE_DIR / "infer_avg.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the average-checkpoint inference script")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INFERENCE_MODULE = load_inference_module()


class InferenceRunManifestTest(unittest.TestCase):
    """Check that only matching partial runs can resume."""

    def test_matching_manifest_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            manifest = {"version": 1, "model": "first"}

            INFERENCE_MODULE.initialize_run_directory(output_dir, manifest)
            (output_dir / "session.rttm").write_text("SPEAKER session 1 0 1 <NA> <NA> spk <NA> <NA>\n")
            INFERENCE_MODULE.initialize_run_directory(output_dir, manifest)

    def test_empty_rttm_marker_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            manifest = {"version": 1, "model": "first"}
            rttm_path = output_dir / "session.rttm"

            INFERENCE_MODULE.initialize_run_directory(output_dir, manifest)
            INFERENCE_MODULE.write_rttm_atomically(rttm_path, "")
            INFERENCE_MODULE.initialize_run_directory(output_dir, manifest)

            self.assertEqual(rttm_path.read_text(), "")
            self.assertTrue(INFERENCE_MODULE.is_completed_rttm(rttm_path))
            self.assertTrue(INFERENCE_MODULE.rttm_completion_marker(rttm_path).is_file())

    def test_empty_rttm_without_manifest_is_rejected_after_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            rttm_path = output_dir / "session.rttm"
            INFERENCE_MODULE.write_rttm_atomically(rttm_path, "")

            with self.assertRaisesRegex(ValueError, "Cannot validate 1 existing RTTM"):
                INFERENCE_MODULE.initialize_run_directory(output_dir, {"version": 1, "model": "first"})

    def test_changed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            INFERENCE_MODULE.initialize_run_directory(output_dir, {"version": 1, "model": "first"})

            with self.assertRaisesRegex(ValueError, "use a new output directory"):
                INFERENCE_MODULE.initialize_run_directory(output_dir, {"version": 1, "model": "second"})

    def test_changed_engine_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            first_manifest = {"version": 4, "engine": {"dependency": "first"}}
            changed_manifest = {"version": 4, "engine": {"dependency": "changed"}}
            INFERENCE_MODULE.initialize_run_directory(output_dir, first_manifest)

            with self.assertRaisesRegex(ValueError, "use a new output directory"):
                INFERENCE_MODULE.initialize_run_directory(output_dir, changed_manifest)

    def test_legacy_rttm_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            (output_dir / "session.rttm").write_text("non-empty\n")

            with self.assertRaisesRegex(ValueError, "Cannot validate 1 existing RTTM"):
                INFERENCE_MODULE.initialize_run_directory(output_dir, {"version": 1, "model": "first"})

    def test_manifest_binds_audio_content(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio = root / "session.wav"
            audio.write_bytes(b"first audio")
            wav_scp = root / "wav.scp"
            wav_scp.write_text(f"session {audio}\n")
            config = root / "config.toml"
            config.write_text("[meta]\n")
            model = root / "model.bin"
            model.write_bytes(b"model")
            embedding = root / "embedding.bin"
            embedding.write_bytes(b"embedding")
            args = SimpleNamespace(
                embedding_model=embedding,
                in_wav_scp=wav_scp,
                seg_duration=8,
                segmentation_step=0.1,
                batch_size=16,
                apply_median_filtering=False,
                clustering_method="AgglomerativeClustering",
                min_speakers=2,
                max_speakers=8,
                ahc_criterion="centroid",
                ahc_threshold=0.7,
                min_cluster_size=30,
                Fa=0.3,
                Fb=17,
                lda_dim=128,
                max_iters=20,
            )

            with patch.object(INFERENCE_MODULE, "build_engine_identity", return_value={"fixture": "engine-v1"}):
                first_manifest = INFERENCE_MODULE.build_run_manifest(args, config, model.as_posix())
                audio.write_bytes(b"other audio")
                second_manifest = INFERENCE_MODULE.build_run_manifest(args, config, model.as_posix())

            self.assertEqual(first_manifest["version"], 4)
            self.assertNotEqual(first_manifest["input"], second_manifest["input"])

    def test_silent_session_returns_empty_annotation_without_embeddings(self):
        class SilentSegmentations:
            data = np.zeros((1, 4, 2), dtype=np.uint8)

        class SilentPipeline:
            def get_segmentations(self, unused_file, soft):
                return SilentSegmentations()

            def speaker_count(self, unused_segmentations, unused_receptive_field, warm_up):
                raise AssertionError("silent segmentation must not reach speaker counting")

            def get_embeddings(self, *unused_args, **unused_kwargs):
                raise AssertionError("silent segmentation must not extract embeddings")

        waveform = INFERENCE_MODULE.torch.zeros((1, 32))
        with patch.object(INFERENCE_MODULE.torchaudio, "load", return_value=(waveform, 16000)):
            result = INFERENCE_MODULE.diarize_session(
                sess_name="silent",
                in_wav="silent.wav",
                pipeline=SilentPipeline(),
                apply_median_filtering=False,
            )

        self.assertEqual(result.uri, "silent")
        self.assertEqual(result.to_rttm(), "")

    def test_wav_scp_rejects_duplicate_sessions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            wav_scp = Path(temporary_directory) / "wav.scp"
            wav_scp.write_text("session first.wav\nsession second.wav\n")

            with self.assertRaisesRegex(ValueError, "Duplicate session"):
                INFERENCE_MODULE.load_scp(wav_scp.as_posix())


if __name__ == "__main__":
    unittest.main()
