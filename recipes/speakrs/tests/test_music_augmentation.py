"""Regression tests for deterministic runtime music augmentation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from diarizen.music_augmentation import (
    MUSIC_AUGMENTATION_SCHEMA,
    MUSIC_MANIFEST_SCHEMA,
    MusicAugmentationError,
    MusicAugmenter,
    load_music_augmentation_config,
)
from recipes.diar_ssl.dataset import DiarizationDataset, _collate_fn


SAMPLE_RATE = 16_000
SAMPLE_COUNT = 16_000


def _pcm_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with sf.SoundFile(str(path)) as audio:
        while True:
            block = audio.read(65_536, dtype="int16", always_2d=True)
            if len(block) == 0:
                break
            digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_music_fixture(
    tmp_path: Path,
    *,
    sample_count: int = SAMPLE_COUNT,
    scale: float = 1.0,
) -> tuple[Path, Path]:
    music_path = tmp_path / "music.wav"
    time = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    music = scale * np.where(time < 0.25, 0.02, 0.18) * np.sin(2 * np.pi * 311 * time)
    sf.write(music_path, music, SAMPLE_RATE, subtype="PCM_16")
    manifest = {
        "schema": MUSIC_MANIFEST_SCHEMA,
        "sample_rate": SAMPLE_RATE,
        "tracks": [
            {
                "track_id": "toy-music",
                "path": music_path.name,
                "sha256": _file_sha256(music_path),
                "pcm_sha256": _pcm_sha256(music_path),
                "sample_count": sample_count,
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "split": "train",
                "artist": "toy-artist",
                "vocals": "N",
                "source_member": "toy/member.wav",
                "source_sha256": "a" * 64,
                "license": {"spdx": "CC0-1.0", "source": "fixture"},
            }
        ],
    }
    manifest_path = tmp_path / "music-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return music_path, manifest_path


def _write_config(
    tmp_path: Path,
    manifest_path: Path,
    *,
    clean: int = 0,
    mixed: int = 1_000_000,
    music_only: int = 0,
    enabled: bool = True,
) -> Path:
    config = {
        "schema": MUSIC_AUGMENTATION_SCHEMA,
        "enabled": enabled,
        "split": "train",
        "seed": 3407,
        "proportions": {"clean": clean, "mixed": mixed, "music_only": music_only},
        "mixed_snr_db": {"min": 10.0, "max": 20.0},
        "no_speech_rms_dbfs": {"min": -30.0, "max": -18.0},
    }
    if enabled:
        config["manifest_path"] = manifest_path.name
        config["manifest_sha256"] = _file_sha256(manifest_path)
    path = tmp_path / "music-config.json"
    path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return path


def _speech_source() -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    x = np.zeros((1, SAMPLE_COUNT), dtype=np.float64)
    x[:, 2_000:6_000] = 0.4
    y = np.zeros((12, 2), dtype=np.uint8)
    y[2:6, 0] = 1
    annotations = [{"start": 0.125, "end": 0.375}]
    return x, y, annotations


def test_mixed_uses_exact_speech_union_and_is_repeatable(tmp_path: Path) -> None:
    _, manifest_path = _write_music_fixture(tmp_path)
    augmenter = MusicAugmenter.from_config(_write_config(tmp_path, manifest_path))
    x, y, annotations = _speech_source()

    first_x, first_y, first_receipt = augmenter.apply_with_receipt(
        x,
        y,
        recording_id="recording-a",
        chunk_start_sample=0,
        chunk_end_sample=SAMPLE_COUNT,
        chunked_annotations=annotations,
    )
    second_x, second_y, second_receipt = augmenter.apply_with_receipt(
        x,
        y,
        recording_id="recording-a",
        chunk_start_sample=0,
        chunk_end_sample=SAMPLE_COUNT,
        chunked_annotations=annotations,
    )

    assert first_receipt.mode == "mixed"
    assert first_receipt.requested_snr_db is not None
    assert 10.0 <= first_receipt.requested_snr_db <= 20.0
    assert first_receipt.measured_snr_db == first_receipt.measured_snr_db
    assert abs(first_receipt.measured_snr_db - first_receipt.requested_snr_db) < 1e-9
    assert first_x.tobytes() == second_x.tobytes()
    assert first_y.tobytes() == y.tobytes() == second_y.tobytes()
    assert first_receipt == second_receipt
    assert first_receipt.speech_rms == np.sqrt(np.mean(np.square(x[:, 2_000:6_000])))
    assert first_receipt.music_track_id == "toy-music"
    assert first_receipt.music_offset_sample == 0


def test_silent_mixed_uses_safe_negative_level_and_keeps_empty_targets(tmp_path: Path) -> None:
    _, manifest_path = _write_music_fixture(tmp_path)
    augmenter = MusicAugmenter.from_config(_write_config(tmp_path, manifest_path))
    x = np.zeros((1, SAMPLE_COUNT), dtype=np.float64)
    y = np.empty((12, 0), dtype=np.uint8)

    output, output_y, receipt = augmenter.apply_with_receipt(
        x,
        y,
        recording_id="silent-recording",
        chunk_start_sample=0,
        chunk_end_sample=SAMPLE_COUNT,
        chunked_annotations=[],
    )

    assert receipt.mode == "mixed_no_speech"
    assert receipt.requested_snr_db is None
    assert receipt.requested_music_rms_dbfs is not None
    assert -30.0 <= receipt.requested_music_rms_dbfs <= -18.0
    assert receipt.measured_music_rms_dbfs is not None
    assert -30.0 <= receipt.measured_music_rms_dbfs <= -18.0
    assert output_y.tobytes() == y.tobytes()
    assert np.any(output != 0)


def test_music_only_replaces_audio_and_zeroes_targets(tmp_path: Path) -> None:
    _, manifest_path = _write_music_fixture(tmp_path)
    augmenter = MusicAugmenter.from_config(
        _write_config(tmp_path, manifest_path, clean=0, mixed=0, music_only=1_000_000)
    )
    x, y, annotations = _speech_source()

    output, output_y, receipt = augmenter.apply_with_receipt(
        x,
        y,
        recording_id="recording-a",
        chunk_start_sample=0,
        chunk_end_sample=SAMPLE_COUNT,
        chunked_annotations=annotations,
    )

    assert receipt.mode == "music_only"
    assert receipt.requested_mode == "music_only"
    assert output.shape == x.shape
    assert output_y.shape == y.shape
    assert np.all(output_y == 0)
    assert np.any(output != x)
    assert receipt.peak <= 0.99

    collated = _collate_fn([(output, output_y, "recording-a")], max_speakers_per_chunk=4)
    assert collated["ts"].shape == (1, y.shape[0], 4)
    assert int(collated["ts"].sum()) == 0


@pytest.mark.parametrize("scale", [0.1, 4.0])
def test_music_only_targets_safe_level_for_quiet_and_loud_music(tmp_path: Path, scale: float) -> None:
    _, manifest_path = _write_music_fixture(tmp_path, scale=scale)
    augmenter = MusicAugmenter.from_config(
        _write_config(tmp_path, manifest_path, clean=0, mixed=0, music_only=1_000_000)
    )
    x, y, annotations = _speech_source()

    output, output_y, receipt = augmenter.apply_with_receipt(
        x,
        y,
        recording_id="recording-a",
        chunk_start_sample=0,
        chunk_end_sample=SAMPLE_COUNT,
        chunked_annotations=annotations,
    )

    assert receipt.mode == "music_only"
    assert receipt.speech_gain == 0.0
    assert -30.0 <= receipt.requested_music_rms_dbfs <= -18.0
    assert -30.0 <= receipt.measured_music_rms_dbfs <= -18.0
    assert np.all(output_y == 0)
    assert np.isclose(np.sqrt(np.mean(np.square(output))), 10 ** (receipt.measured_music_rms_dbfs / 20), rtol=1e-12)
    assert receipt.peak <= 0.99


@pytest.mark.parametrize(
    ("clean", "mixed", "music_only", "annotations"),
    [(0, 0, 1_000_000, [{"start": 0.125, "end": 0.375}]), (0, 1_000_000, 0, [])],
)
def test_high_crest_music_rejects_infeasible_bounded_level(
    tmp_path: Path,
    clean: int,
    mixed: int,
    music_only: int,
    annotations: object,
) -> None:
    music_path, manifest_path = _write_music_fixture(tmp_path)
    impulse = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    impulse[123] = 0.5
    sf.write(music_path, impulse, SAMPLE_RATE, subtype="PCM_16")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tracks"][0]["sha256"] = _file_sha256(music_path)
    manifest["tracks"][0]["pcm_sha256"] = _pcm_sha256(music_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    augmenter = MusicAugmenter.from_config(
        _write_config(tmp_path, manifest_path, clean=clean, mixed=mixed, music_only=music_only)
    )
    x, y, _ = _speech_source()
    if mixed:
        y = np.empty((12, 0), dtype=np.uint8)

    with pytest.raises(MusicAugmentationError, match="infeasible"):
        augmenter.apply_with_receipt(
            x,
            y,
            recording_id="recording-a",
            chunk_start_sample=0,
            chunk_end_sample=SAMPLE_COUNT,
            chunked_annotations=annotations,
        )


def test_mixed_rejects_labelled_speech_with_zero_source_energy(tmp_path: Path) -> None:
    _, manifest_path = _write_music_fixture(tmp_path)
    augmenter = MusicAugmenter.from_config(_write_config(tmp_path, manifest_path))
    x = np.zeros((1, SAMPLE_COUNT), dtype=np.float64)
    y = np.ones((12, 1), dtype=np.uint8)

    with pytest.raises(MusicAugmentationError, match="zero source energy"):
        augmenter.apply_with_receipt(
            x,
            y,
            recording_id="recording-a",
            chunk_start_sample=0,
            chunk_end_sample=SAMPLE_COUNT,
            chunked_annotations=[{"start": 0.125, "end": 0.375}],
        )


@pytest.mark.parametrize("annotations", [None, [{"start": 0.125}]])
def test_mixed_rejects_positive_targets_without_a_speech_union(tmp_path: Path, annotations: object) -> None:
    _, manifest_path = _write_music_fixture(tmp_path)
    augmenter = MusicAugmenter.from_config(_write_config(tmp_path, manifest_path))
    x = np.full((1, SAMPLE_COUNT), 0.4, dtype=np.float64)
    y = np.ones((12, 1), dtype=np.uint8)

    with pytest.raises(MusicAugmentationError, match="positive targets"):
        augmenter.apply_with_receipt(
            x,
            y,
            recording_id="recording-a",
            chunk_start_sample=0,
            chunk_end_sample=SAMPLE_COUNT,
            chunked_annotations=annotations,
        )


def test_disabled_config_is_bit_exact_without_loading_a_manifest(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        tmp_path / "not-used.json",
        clean=1_000_000,
        mixed=0,
        music_only=0,
        enabled=False,
    )
    augmenter = MusicAugmenter.from_config(config_path)
    x = np.arange(SAMPLE_COUNT, dtype=np.float64).reshape(1, -1) / SAMPLE_COUNT
    y = np.arange(12, dtype=np.uint8).reshape(12, 1)

    output, output_y, receipt = augmenter.apply_with_receipt(
        x,
        y,
        recording_id="recording-a",
        chunk_start_sample=0,
        chunk_end_sample=SAMPLE_COUNT,
        chunked_annotations=[],
    )

    assert receipt.mode == "clean"
    assert output is x
    assert output_y is y
    assert output.tobytes() == x.tobytes()
    assert output_y.tobytes() == y.tobytes()


def test_dataset_wires_music_only_after_constructing_targets(tmp_path: Path) -> None:
    _, manifest_path = _write_music_fixture(tmp_path)
    config_path = _write_config(tmp_path, manifest_path, clean=0, mixed=0, music_only=1_000_000)
    source_path = tmp_path / "speech.wav"
    source = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    source[2_000:6_000] = 0.4
    sf.write(source_path, source, SAMPLE_RATE, subtype="PCM_16")
    scp_path = tmp_path / "wav.scp"
    scp_path.write_text(f"recording-a {source_path}\n", encoding="utf-8")
    rttm_path = tmp_path / "rttm"
    rttm_path.write_text("SPEAKER recording-a 1 0.125 0.250 <NA> <NA> speaker-a <NA> <NA>\n", encoding="utf-8")
    uem_path = tmp_path / "all.uem"
    uem_path.write_text("recording-a 1 0 1\n", encoding="utf-8")

    dataset = DiarizationDataset(
        str(scp_path),
        str(rttm_path),
        str(uem_path),
        model_num_frames=12,
        model_rf_duration=0.1,
        model_rf_step=0.1,
        chunk_size=0,
        sample_rate=SAMPLE_RATE,
        music_augmentation_config=str(config_path),
    )
    output, target, recording_id = dataset[0]

    assert recording_id == "recording-a"
    assert output.shape == (1, SAMPLE_COUNT)
    assert target.shape == (12, 1)
    assert np.all(target == 0)
    assert np.any(output != 0)


def test_dataset_resolves_relative_scp_audio_from_scp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_directory = tmp_path / "audio"
    audio_directory.mkdir()
    source_path = audio_directory / "speech.wav"
    source = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    source[2_000:6_000] = 0.4
    sf.write(source_path, source, SAMPLE_RATE, subtype="PCM_16")
    scp_path = tmp_path / "wav.scp"
    scp_path.write_text("recording-a audio/speech.wav\n", encoding="utf-8")
    absolute_scp_path = tmp_path / "absolute-wav.scp"
    absolute_scp_path.write_text(f"recording-a {source_path}\n", encoding="utf-8")
    rttm_path = tmp_path / "rttm"
    rttm_path.write_text("SPEAKER recording-a 1 0.125 0.250 <NA> <NA> speaker-a <NA> <NA>\n", encoding="utf-8")
    uem_path = tmp_path / "all.uem"
    uem_path.write_text("recording-a 1 0 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)

    relative_dataset = DiarizationDataset(
        str(scp_path),
        str(rttm_path),
        str(uem_path),
        model_num_frames=12,
        model_rf_duration=0.1,
        model_rf_step=0.1,
        chunk_size=0,
        sample_rate=SAMPLE_RATE,
    )
    absolute_dataset = DiarizationDataset(
        str(absolute_scp_path),
        str(rttm_path),
        str(uem_path),
        model_num_frames=12,
        model_rf_duration=0.1,
        model_rf_step=0.1,
        chunk_size=0,
        sample_rate=SAMPLE_RATE,
    )

    assert relative_dataset.rec_scp["recording-a"] == str(source_path.resolve())
    assert absolute_dataset.rec_scp["recording-a"] == str(source_path)
    assert relative_dataset[0][0].tobytes() == absolute_dataset[0][0].tobytes()


def test_short_track_is_excluded_for_a_longer_requested_chunk(tmp_path: Path) -> None:
    _, manifest_path = _write_music_fixture(tmp_path, sample_count=SAMPLE_COUNT)
    short_path = tmp_path / "short.wav"
    sf.write(short_path, np.ones(SAMPLE_COUNT // 2, dtype=np.float64) * 0.1, SAMPLE_RATE, subtype="PCM_16")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    short = dict(manifest["tracks"][0])
    short.update(
        {
            "track_id": "short-music",
            "path": short_path.name,
            "sha256": _file_sha256(short_path),
            "pcm_sha256": _pcm_sha256(short_path),
            "sample_count": SAMPLE_COUNT // 2,
        }
    )
    manifest["tracks"].insert(0, short)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    augmenter = MusicAugmenter.from_config(_write_config(tmp_path, manifest_path))
    x, y, annotations = _speech_source()

    _, _, receipt = augmenter.apply_with_receipt(
        x,
        y,
        recording_id="recording-a",
        chunk_start_sample=0,
        chunk_end_sample=SAMPLE_COUNT,
        chunked_annotations=annotations,
    )

    assert receipt.music_track_id == "toy-music"


def test_manifest_identity_and_validation_fail_closed(tmp_path: Path) -> None:
    _, manifest_path = _write_music_fixture(tmp_path)
    config_path = _write_config(tmp_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tracks"][0]["channels"] = 2
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with np.testing.assert_raises(MusicAugmentationError):
        MusicAugmenter.from_config(config_path)

    bad_config = json.loads(config_path.read_text(encoding="utf-8"))
    bad_config["split"] = "validation"
    bad_config_path = tmp_path / "bad-config.json"
    bad_config_path.write_text(json.dumps(bad_config, sort_keys=True), encoding="utf-8")
    with np.testing.assert_raises(MusicAugmentationError):
        load_music_augmentation_config(bad_config_path)


@pytest.mark.parametrize("vocals", [False, "n"])
def test_manifest_requires_exact_instrumental_vocals_marker(tmp_path: Path, vocals: object) -> None:
    _, manifest_path = _write_music_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tracks"][0]["vocals"] = vocals
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(MusicAugmentationError, match="exactly 'N'"):
        MusicAugmenter.from_config(_write_config(tmp_path, manifest_path))
