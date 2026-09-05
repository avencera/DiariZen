import importlib.util
import sys
import unittest
import zipfile
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
