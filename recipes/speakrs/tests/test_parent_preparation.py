"""Verify bounded, measured preparation of one canonical parent."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recipes.diar_ssl.dataset import load_uem
from recipes.speakrs.large.contracts import DiskLimits
from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.prepare import _recording_row, prepare_parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _limits(root: Path, *, staging_bytes: int = 32_000_000) -> DiskLimits:
    staging = root / "staging"
    cache = root / "cache"
    staging.mkdir()
    cache.mkdir()
    return DiskLimits(staging, cache, staging_bytes, 32_000_000, 0, 1)


def test_prepare_parent_writes_measured_mono_pair_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    samples = np.zeros((32_000, 2), dtype=np.float32)
    samples[:1_000, 0] = 0.25
    samples[1_000:2_000, 0] = 1.0
    samples[:, 1] = 0.75
    sf.write(source, samples, 16_000, subtype="PCM_16")
    source_digest = _sha256(source)
    limits = _limits(tmp_path)
    destination = limits.staging_root / "parent.flac"

    result = prepare_parent(
        source,
        0,
        source_digest,
        [("parent", 0.25, 0.5, "speaker-1")],
        [(0.0, 0.5), (0.75, 1.5)],
        destination,
        limits,
        parent_id="parent",
    )

    info = sf.info(destination)
    assert (info.samplerate, info.channels, info.frames) == (16_000, 1, 32_000)
    assert _sha256(source) == source_digest
    assert result["recording"]["audio"]["sha256"] == _sha256(destination)
    assert result["recording"]["rttm"]["sha256"] == result["label_sha256"]
    assert result["recording"]["uem"]["sha256"] == result["uem_sha256"]
    assert result["audio_statistics"]["sample_count"] == 32_000
    assert result["audio_statistics"]["finite"] is True
    assert result["audio_statistics"]["clipped_samples"] >= 1_000
    assert result["audio_statistics"]["dropout_samples"] >= 30_000
    assert result["timing_mapping"]["intervals"][0]["canonical_start_sample"] == 4_000
    assert result["timing_mapping"]["intervals"][0]["canonical_end_sample"] == 8_000
    assert (limits.staging_root / "parent.rttm").read_text(encoding="utf-8").startswith("SPEAKER parent")
    assert (limits.staging_root / "parent.uem").read_text(encoding="utf-8").count("\n") == 2
    assert Path(result["receipt_path"]).is_file()

    second = prepare_parent(
        source,
        0,
        source_digest,
        [("parent", 0.25, 0.5, "speaker-1")],
        [(0.0, 0.5), (0.75, 1.5)],
        destination,
        limits,
        parent_id="parent",
    )
    assert second["idempotent"] is True
    assert second["canonical_sha256"] == result["canonical_sha256"]


def test_prepare_parent_emits_absolute_uem_endpoints_readable_by_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(32_000, dtype=np.float32), 16_000, subtype="PCM_16")
    source_digest = _sha256(source)
    limits = _limits(tmp_path)
    result = prepare_parent(
        source,
        "mono",
        source_digest,
        [],
        "uem-roundtrip 1 0.25 0.5\nuem-roundtrip 1 1.0 1.5\n",
        limits.staging_root / "uem-roundtrip.flac",
        limits,
        parent_id="uem-roundtrip",
    )

    emitted_uem = Path(result["recording"]["uem"]["path"])
    assert load_uem(str(emitted_uem)) == {"uem-roundtrip": [(0.25, 0.5), (1.0, 1.5)]}


def test_prepare_parent_records_fixed_swr_transform_timing_and_finite_waveform(tmp_path: Path) -> None:
    source = tmp_path / "source-48k.wav"
    time = np.arange(48_000, dtype=np.float32) / 48_000
    samples = np.column_stack((np.zeros_like(time), np.sin(2 * np.pi * 220 * time) * 0.2))
    sf.write(source, samples, 48_000, subtype="PCM_16")
    source_digest = _sha256(source)
    limits = _limits(tmp_path)

    first = prepare_parent(
        source,
        1,
        source_digest,
        [{"start": 0.125, "end": 0.375, "speaker": "speaker-2"}],
        [{"start": 0.0, "end": 1.0}],
        limits.staging_root / "resampled-a.flac",
        limits,
        parent_id="parent-a",
    )
    second = prepare_parent(
        source,
        1,
        source_digest,
        [{"start": 0.125, "end": 0.375, "speaker": "speaker-2"}],
        [{"start": 0.0, "end": 1.0}],
        limits.staging_root / "resampled-b.flac",
        limits,
        parent_id="parent-b",
    )

    output, sample_rate = sf.read(first["canonical_path"], dtype="float64")
    assert (sample_rate, output.shape) == (16_000, (16_000,))
    assert np.isfinite(output).all()
    expected_waveform = np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000)
    assert np.corrcoef(output, expected_waveform)[0, 1] > 0.99
    assert first["transform_receipt"]["resampler"] == {
        "name": "libswresample",
        "engine": "swr",
        "filter_type": "kaiser",
        "filter_size": 64,
        "phase_shift": 10,
        "linear_interp": False,
        "exact_rational": True,
        "cutoff": 0.97,
        "kaiser_beta": 9.0,
        "dither_method": "none",
        "target_sample_rate": 16_000,
        "output_sample_format": "s16",
        "filter_graph": (
            "pan=mono|c0=c1,aresample=resampler=swr:filter_type=kaiser:filter_size=64:"
            "phase_shift=10:linear_interp=0:exact_rational=1:cutoff=0.97:kaiser_beta=9.0:"
            "dither_method=none:osr=16000"
        ),
    }
    assert first["transform_receipt"]["decoder_executable"]["sha256"] == _sha256(
        Path(first["transform_receipt"]["decoder_executable"]["path"])
    )
    assert first["transform_receipt"]["decoder_executable"]["version"].startswith("ffmpeg version ")
    assert first["transform_receipt"] == second["transform_receipt"]
    assert first["transform_sha256"] == second["transform_sha256"]
    assert first["timing_mapping"]["sample_rate_ratio"] == {"numerator": 16_000, "denominator": 48_000}
    assert first["timing_mapping"]["intervals"][0]["canonical_start_sample"] == 2_000
    assert first["timing_mapping"]["intervals"][0]["canonical_end_sample"] == 6_000
    assert _sha256(source) == source_digest


def test_prepare_parent_requires_channel_for_multichannel_and_checks_bounds(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((16_000, 2), dtype=np.float32), 16_000, subtype="PCM_16")
    source_digest = _sha256(source)
    limits = _limits(tmp_path)

    with pytest.raises(PreparationError, match="requires an explicit selected channel"):
        prepare_parent(source, None, source_digest, [], None, limits.staging_root / "none.flac", limits)
    with pytest.raises(PreparationError, match="outside the decoded source"):
        prepare_parent(source, 2, source_digest, [], None, limits.staging_root / "bad.flac", limits)
    with pytest.raises(PreparationError, match="outside the decoded source timeline"):
        prepare_parent(
            source,
            0,
            source_digest,
            [(0.0, 2.0, "speaker")],
            None,
            limits.staging_root / "bad-label.flac",
            limits,
        )


def test_prepare_parent_rejects_storage_overcommit_before_creating_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16")
    source_digest = _sha256(source)
    limits = _limits(tmp_path, staging_bytes=1)
    destination = limits.staging_root / "over-cap.flac"

    with pytest.raises(PreparationError, match="staging cap exhausted before"):
        prepare_parent(source, "mono", source_digest, [], None, destination, limits)

    assert not destination.exists()
    assert not (limits.staging_root / "over-cap.rttm").exists()
    assert not (limits.staging_root / "over-cap.uem").exists()
    assert _sha256(source) == source_digest


def test_recording_rows_require_measured_facts() -> None:
    with pytest.raises(PreparationError, match="measured audio, label, and permission facts"):
        _recording_row("AMI", "train", "parent")
