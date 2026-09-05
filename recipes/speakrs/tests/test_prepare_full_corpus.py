import importlib.util
import sys
import tarfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "prepare_full_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_full_corpus", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


class PrepareFullCorpusTest(unittest.TestCase):
    def test_classifies_all_corpora(self):
        self.assertEqual(prepare.classify_session("IS1009a"), prepare.Corpus.AMI)
        self.assertEqual(prepare.classify_session("R8002_M8002_MS802"), prepare.Corpus.ALI_MEETING)
        self.assertEqual(prepare.classify_session("20200707_L_R001S08C01"), prepare.Corpus.AISHELL4)
        self.assertEqual(prepare.classify_session("L_R003S01C02"), prepare.Corpus.AISHELL4)

    def test_rejects_unsafe_or_unknown_session_ids(self):
        for session_id in ("../../secret", "unknown", "IS100"):
            with self.subTest(session_id=session_id), self.assertRaises(ValueError):
                prepare.classify_session(session_id)

    def test_maps_aishell_train_sizes_and_test(self):
        train = prepare.Recording("20200707_L_R001S08C01", prepare.Corpus.AISHELL4, prepare.Split.TRAIN)
        test = prepare.Recording("L_R003S01C02", prepare.Corpus.AISHELL4, prepare.Split.TEST)
        self.assertEqual(prepare.aishell_archive_name(train), "train_L")
        self.assertEqual(prepare.aishell_archive_name(test), "test")

    def test_uses_only_safe_audio_archive_members(self):
        valid = tarfile.TarInfo("Train_Ali_far/audio_dir/R0003_M0046_MS002.wav")
        valid.type = tarfile.REGTYPE
        unsafe = tarfile.TarInfo("../../R0003_M0046_MS002.wav")
        unsafe.type = tarfile.REGTYPE
        directory = tarfile.TarInfo("Train_Ali_far/audio_dir/R0003_M0046_MS002.wav")
        directory.type = tarfile.DIRTYPE
        self.assertEqual(prepare.archive_member_session(valid), "R0003_M0046_MS002")
        self.assertIsNone(prepare.archive_member_session(unsafe))
        self.assertIsNone(prepare.archive_member_session(directory))


if __name__ == "__main__":
    unittest.main()
