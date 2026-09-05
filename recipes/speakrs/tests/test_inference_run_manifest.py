#!/usr/bin/env python3

"""Tests for resumable DiariZen inference run identity."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


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

    def test_changed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            INFERENCE_MODULE.initialize_run_directory(output_dir, {"version": 1, "model": "first"})

            with self.assertRaisesRegex(ValueError, "use a new output directory"):
                INFERENCE_MODULE.initialize_run_directory(output_dir, {"version": 1, "model": "second"})

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

            first_manifest = INFERENCE_MODULE.build_run_manifest(args, config, model.as_posix())
            audio.write_bytes(b"other audio")
            second_manifest = INFERENCE_MODULE.build_run_manifest(args, config, model.as_posix())

            self.assertEqual(first_manifest["version"], 2)
            self.assertNotEqual(first_manifest["input"], second_manifest["input"])

    def test_wav_scp_rejects_duplicate_sessions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            wav_scp = Path(temporary_directory) / "wav.scp"
            wav_scp.write_text("session first.wav\nsession second.wav\n")

            with self.assertRaisesRegex(ValueError, "Duplicate session"):
                INFERENCE_MODULE.load_scp(wav_scp.as_posix())


if __name__ == "__main__":
    unittest.main()
