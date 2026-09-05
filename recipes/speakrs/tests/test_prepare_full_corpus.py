import importlib.util
import io
import json
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch


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

    def test_builds_aishell_archives_with_eight_input_channels(self):
        recording = prepare.Recording(
            "20200707_L_R001S08C01",
            prepare.Corpus.AISHELL4,
            prepare.Split.TRAIN,
        )
        archive = prepare.build_archives([recording])[0]
        self.assertEqual(archive.channel_policy, prepare.ChannelPolicy.MIX)
        self.assertEqual(archive.input_channels, 8)

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

    def test_aishell_uses_equal_mean_filter_for_all_eight_channels(self):
        expected = "pan=mono|c0=" + "+".join("0.125*c" + str(channel) for channel in range(8))
        self.assertEqual(prepare.equal_mean_filter(8), expected)
        self.assertNotIn("-ac", prepare.equal_mean_filter(8))

    def test_policy_sidecar_invalidates_audio_after_policy_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "recording.flac"
            output.write_bytes(b"audio")
            with patch.object(prepare, "probe_audio", return_value=True):
                self.assertFalse(prepare.audio_matches_policy(output, prepare.ChannelPolicy.MIX, 8))

                prepare.write_audio_provenance(output, prepare.ChannelPolicy.MIX, 8)
                self.assertTrue(prepare.audio_matches_policy(output, prepare.ChannelPolicy.MIX, 8))

                stale = prepare.audio_provenance(prepare.ChannelPolicy.MIX, 8)
                stale["preparation_policy_version"] -= 1
                prepare.atomic_write(
                    prepare.audio_provenance_path(output),
                    json.dumps(stale) + "\n",
                )
                self.assertFalse(prepare.audio_matches_policy(output, prepare.ChannelPolicy.MIX, 8))

    def test_legacy_first_audio_is_adopted_but_legacy_aishell_audio_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_output = Path(temporary) / "first.flac"
            mix_output = Path(temporary) / "mix.flac"
            first_output.write_bytes(b"legacy audio")
            mix_output.write_bytes(b"legacy audio")
            with patch.object(prepare, "probe_audio", return_value=True):
                self.assertTrue(prepare.audio_ready(first_output, prepare.ChannelPolicy.FIRST))
                self.assertTrue(prepare.audio_provenance_path(first_output).is_file())
                first_metadata = json.loads(prepare.audio_provenance_path(first_output).read_text())
                self.assertEqual(first_metadata["channel_policy_version"], 1)

                self.assertFalse(prepare.audio_ready(mix_output, prepare.ChannelPolicy.MIX, 8))
                self.assertFalse(prepare.audio_provenance_path(mix_output).exists())

    def test_verify_rejects_legacy_audio_without_adopting_it(self):
        recording = prepare.Recording(
            "L_R003S01C02",
            prepare.Corpus.AISHELL4,
            prepare.Split.TEST,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            output = prepare.audio_path(output_root, recording)
            output.parent.mkdir(parents=True)
            output.write_bytes(b"legacy audio")
            prepare.atomic_write(
                output_root / "provenance.json",
                json.dumps({"preparation_policy": prepare.preparation_policy_provenance()}) + "\n",
            )
            with patch.object(prepare, "probe_audio", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "stale policy provenance"):
                    prepare.verify_prepared_corpus(
                        (prepare.ManifestSet(Path(temporary), Path(temporary), (recording,)),),
                        output_root,
                    )
            self.assertFalse(prepare.audio_provenance_path(output).exists())

    def test_verify_accepts_current_audio_and_policy_without_writing(self):
        recording = prepare.Recording(
            "R8002_M8002_MS802",
            prepare.Corpus.ALI_MEETING,
            prepare.Split.TEST,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            output = prepare.audio_path(output_root, recording)
            output.parent.mkdir(parents=True)
            output.write_bytes(b"prepared audio")
            prepare.write_audio_provenance(output, prepare.ChannelPolicy.FIRST)
            provenance_path = output_root / "provenance.json"
            prepare.atomic_write(
                provenance_path,
                json.dumps({"preparation_policy": prepare.preparation_policy_provenance()}) + "\n",
            )
            before = {
                path: path.stat().st_mtime_ns
                for path in (output, prepare.audio_provenance_path(output), provenance_path)
            }
            with (
                patch.object(prepare, "probe_audio", return_value=True),
                patch.object(
                    prepare,
                    "audio_ready",
                    side_effect=AssertionError("verify must not adopt legacy audio"),
                ),
            ):
                prepare.verify_prepared_corpus(
                    (prepare.ManifestSet(Path(temporary), Path(temporary), (recording,)),),
                    output_root,
                )
            after = {
                path: path.stat().st_mtime_ns
                for path in (output, prepare.audio_provenance_path(output), provenance_path)
            }
            self.assertEqual(before, after)

    def test_verify_rejects_stale_global_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            provenance_path = Path(temporary) / "provenance.json"
            stale = prepare.preparation_policy_provenance()
            stale["version"] -= 1
            prepare.atomic_write(
                provenance_path,
                json.dumps({"preparation_policy": stale}) + "\n",
            )
            with self.assertRaisesRegex(RuntimeError, "does not match preparation policy"):
                prepare.verify_provenance(provenance_path)

    def test_verify_mode_requires_only_ffprobe_and_does_not_prepare(self):
        arguments = Mock(
            source_root=Path("source"),
            output_root=Path("output"),
            manifest_audio_prefix=Path("audio"),
            minimum_free_gib=0,
            plan=False,
            verify=True,
        )
        with (
            patch.object(prepare, "parse_args", return_value=arguments),
            patch.object(prepare, "load_manifests", return_value=()),
            patch.object(prepare, "build_archives", return_value=()),
            patch.object(prepare, "require_executable") as require_executable,
            patch.object(prepare, "verify_prepared_corpus") as verify_prepared_corpus,
            patch.object(prepare, "download_ami") as download_ami,
            patch.object(prepare, "extract_archive") as extract_archive,
        ):
            prepare.main()

        require_executable.assert_called_once_with("ffprobe")
        verify_prepared_corpus.assert_called_once_with((), arguments.output_root.resolve())
        download_ami.assert_not_called()
        extract_archive.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe are required")
    def test_aishell_transcode_averages_synthetic_channel_impulses(self):
        channel_count = 8
        frame_count = 160
        impulse = 8000
        source = io.BytesIO()
        with wave.open(source, "wb") as writer:
            writer.setnchannels(channel_count)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            frames = bytearray()
            for frame in range(frame_count):
                for channel in range(channel_count):
                    value = impulse if frame == 20 + channel * 10 else 0
                    frames.extend(struct.pack("<h", value))
            writer.writeframes(frames)
        source.seek(0)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "aishell.flac"
            prepare.transcode(source, output, prepare.ChannelPolicy.MIX, channel_count)
            self.assertTrue(prepare.audio_matches_policy(output, prepare.ChannelPolicy.MIX, channel_count))

            decoded = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    output.as_posix(),
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            )
            samples = struct.unpack("<" + "h" * (len(decoded.stdout) // 2), decoded.stdout)
            for channel in range(channel_count):
                self.assertAlmostEqual(samples[20 + channel * 10], impulse / channel_count, delta=2)


if __name__ == "__main__":
    unittest.main()
