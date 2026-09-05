import importlib.util
import json
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "prepare_voxconverse.py"
sys.path.insert(0, MODULE_PATH.parent.as_posix())
SPEC = importlib.util.spec_from_file_location("prepare_voxconverse", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


class PrepareVoxConverseTest(unittest.TestCase):
    def test_accepts_only_voxconverse_session_ids(self):
        self.assertEqual(prepare.validate_session_id("abjxc"), "abjxc")
        for session_id in ("../../secret", "ABJXC", "abcd", "abcdef"):
            with self.subTest(session_id=session_id), self.assertRaises(ValueError):
                prepare.validate_session_id(session_id)

    def test_validates_rttm_identity_and_activity(self):
        content = "SPEAKER abjxc 1 0.400 6.640 <NA> <NA> spk00 <NA> <NA>\n"
        annotation = prepare.parse_rttm("abjxc", content)
        self.assertEqual(annotation.session_id, "abjxc")
        self.assertEqual(annotation.rttm, content)

        wrong_identity = content.replace("abjxc", "afjiv")
        with self.assertRaises(ValueError):
            prepare.parse_rttm("abjxc", wrong_identity)

    def test_uses_only_safe_wav_members(self):
        valid = zipfile.ZipInfo("audio/abjxc.wav")
        unsafe = zipfile.ZipInfo("../../abjxc.wav")
        unsupported = zipfile.ZipInfo("audio/abjxc.flac")
        self.assertEqual(prepare.archive_member_session(valid), "abjxc")
        self.assertIsNone(prepare.archive_member_session(unsafe))
        self.assertIsNone(prepare.archive_member_session(unsupported))

    def test_detects_manifest_collisions(self):
        existing = "AMI001 /audio/AMI001.flac\n"
        addition = "abjxc /audio/abjxc.flac\n"
        self.assertFalse(prepare.manifest_ids(existing) & prepare.manifest_ids(addition))
        self.assertEqual(
            prepare.manifest_ids(addition),
            {"abjxc"},
        )

    def test_manifest_rebuild_replaces_stale_rows(self):
        with TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "wav.scp"
            destination.write_text(
                "AMI001 /audio/AMI001.flac\nabjxc /stale/abjxc.flac\n",
                encoding="utf-8",
            )

            prepare.replace_training_manifest_records(destination, "abjxc /audio/abjxc.flac\n")

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "AMI001 /audio/AMI001.flac\nabjxc /audio/abjxc.flac\n",
            )

    def test_verification_rejects_stale_marker_after_base_manifest_rebuild(self):
        sources = (
            prepare.SourceSplit(prepare.Split.DEV, "https://example.test/dev.zip", "a" * 64, 1),
            prepare.SourceSplit(prepare.Split.TEST, "https://example.test/test.zip", "b" * 64, 1),
        )
        with TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            for path in prepare.manifest_paths(output_root):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.parent == output_root / "train":
                    recording_ids = ("AMI001", "abcde")
                elif path.parent == output_root / "test" / "VoxConverse":
                    recording_ids = ("fghij",)
                else:
                    recording_ids = (path.parent.name,)
                id_column = 1 if path.name == "rttm" else 0
                lines = [
                    f"ROW {recording_id}" if id_column == 1 else f"{recording_id} value"
                    for recording_id in recording_ids
                ]
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            provenance = {
                "annotation_commit": prepare.VOXCONVERSE_COMMIT,
                "annotation_url": prepare.ANNOTATION_URL,
                "annotation_sha256": prepare.ANNOTATION_SHA256,
                "audio": {
                    source.split.value: {
                        "url": source.audio_url,
                        "sha256": source.audio_sha256,
                        "recordings": source.expected_recordings,
                    }
                    for source in sources
                },
                "manifest_provenance_version": prepare.MANIFEST_PROVENANCE_VERSION,
                "recording_ids": {"dev": ["abcde"], "test": ["fghij"]},
                "manifests": prepare.describe_manifests(output_root),
            }
            (output_root / "provenance.voxconverse.json").write_text(
                json.dumps(provenance),
                encoding="utf-8",
            )

            with (
                patch.object(prepare, "SOURCES", sources),
                patch.object(prepare, "TRAIN_RECORDINGS", 2),
                patch.object(prepare, "probe_audio", return_value=True),
            ):
                prepare.verify_materialized_data(output_root)

                for name in prepare.MANIFEST_NAMES:
                    path = output_root / "train" / name
                    row = "ROW AMI001" if name == "rttm" else "AMI001 value"
                    path.write_text(row + "\n", encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, "do not match"):
                    prepare.verify_materialized_data(output_root)


if __name__ == "__main__":
    unittest.main()
