#!/usr/bin/env python3

"""Tests for the installed inference engine identity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from diarizen import inference_identity


class InferenceIdentityTest(unittest.TestCase):
    """Check source, dependency, and VBx asset identity inputs."""

    def test_package_snapshot_detects_changed_added_and_deleted_sources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_root = Path(temporary_directory)
            source = package_root / "module.py"
            added = package_root / "added.py"
            removed = package_root / "removed.py"
            source.write_text("first\n")
            removed.write_text("removed\n")

            with patch.object(inference_identity, "_package_roots", return_value=(package_root,)):
                first = inference_identity.package_source_identity("fake.package")
                source.write_text("changed source\n")
                added.write_text("new source\n")
                removed.unlink()
                second = inference_identity.package_source_identity("fake.package")

            self.assertNotEqual(first, second)
            self.assertEqual([record["name"] for record in second["files"]], ["added.py", "module.py"])
            self.assertEqual(second["files"][1]["size"], len("changed source\n"))
            self.assertEqual(set(second["files"][1]), {"name", "size", "sha256", "root"})

    def test_dependency_version_changes_engine_identity(self):
        source = {"files": [{"name": "module.py", "size": 1, "sha256": "source"}]}
        first = self._build_engine(source, dependency_version="1.0")
        changed = self._build_engine(source, dependency_version="2.0")
        self.assertNotEqual(first, changed)

    def test_package_source_changes_engine_identity(self):
        first_source = {"files": [{"name": "module.py", "size": 1, "sha256": "first"}]}
        changed_source = {"files": [{"name": "module.py", "size": 2, "sha256": "changed"}]}
        first = self._build_engine(first_source)
        changed = self._build_engine(changed_source)
        self.assertNotEqual(first, changed)

    def test_vbx_plda_changes_engine_identity(self):
        source = {"files": [{"name": "module.py", "size": 1, "sha256": "source"}]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            plda_dir = Path(temporary_directory) / "plda"
            plda_dir.mkdir()
            for asset in inference_identity.REQUIRED_PLDA_ASSETS:
                (plda_dir / asset).write_bytes(b"first")
            with (
                patch.object(inference_identity, "installed_dependency_versions", return_value={}),
                patch.object(inference_identity, "package_source_identity", return_value=source),
            ):
                first = inference_identity.build_engine_identity("VBxClustering", temporary_directory)
                (plda_dir / "plda.npz").write_bytes(b"changed")
                changed = inference_identity.build_engine_identity("VBxClustering", temporary_directory)
        self.assertNotEqual(first, changed)

    def test_vbx_requires_assets_loaded_by_vbx_setup(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            plda_dir = Path(temporary_directory) / "plda"
            plda_dir.mkdir()
            (plda_dir / "unexpected.bin").write_bytes(b"asset")
            with (
                patch.object(inference_identity, "installed_dependency_versions", return_value={}),
                patch.object(inference_identity, "package_source_identity", return_value={"files": []}),
            ):
                with self.assertRaisesRegex(inference_identity.InferenceIdentityError, "xvec_transform.npz"):
                    inference_identity.build_engine_identity("VBxClustering", temporary_directory)

    def test_missing_dependency_identity_fails_closed(self):
        with patch.object(
            inference_identity,
            "installed_dependency_versions",
            side_effect=inference_identity.InferenceIdentityError("missing"),
        ):
            with self.assertRaises(inference_identity.InferenceIdentityError):
                inference_identity.build_engine_identity("AgglomerativeClustering")

    def _build_engine(self, source, dependency_version="1.0"):
        dependencies = {"torch": {"distribution": "torch", "version": dependency_version}}
        with (
            patch.object(inference_identity, "installed_dependency_versions", return_value=dependencies),
            patch.object(inference_identity, "package_source_identity", return_value=source),
        ):
            return inference_identity.build_engine_identity("AgglomerativeClustering")


if __name__ == "__main__":
    unittest.main()
