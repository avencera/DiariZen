from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from diarizen.music_augmentation import load_music_manifest
from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.music_source import (
    GIB,
    LicenseRecord,
    PreparationConfig,
    StagedTrack,
    TrackMetadata,
    _ensure_capacity,
    _parse_md5,
    _split_tracks,
    parse_annotations,
    parse_license_records,
    prepare_stream,
    safe_archive_path,
)


def _wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16_000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _archive(files: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in files:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return buffer.getvalue()


def _config(output: Path, **overrides: object) -> PreparationConfig:
    values = {
        "output": output,
        "source_url": "https://example.test/musan.tar.gz",
        "checksum_url": "https://example.test/checksum.md5",
        "minimum_free_bytes": 0,
        "max_staging_bytes": 16 * 1024 * 1024,
        "block_bytes": 4096,
    }
    values.update(overrides)
    return PreparationConfig(**values)


def test_checksum_parser_selects_the_archive_entry() -> None:
    assert (
        _parse_md5("ace99c853c9860d64d7953df5306495a  about.html\n0c472d4fc0c5141eca47ad1ffeb2a7df *musan.tar.gz\n")
        == "0c472d4fc0c5141eca47ad1ffeb2a7df"
    )


@pytest.mark.parametrize("name", ["/musan/music/a.wav", "../a.wav", "musan/../a.wav", "musan\\music\\a.wav"])
def test_safe_archive_path_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(PreparationError):
        safe_archive_path(name)


def test_annotations_keep_artist_and_optional_composer_separate() -> None:
    records = parse_annotations(
        "music-hd-0001 westernart,romantic N Kevin_MacLoad Brahms\nmusic-hd-0002 jazz N Another_Artist\n",
        source="hd-classical",
        member="musan/music/hd-classical/ANNOTATIONS",
    )

    assert records["music-hd-0001"].artist == "Kevin_MacLoad"
    assert records["music-hd-0001"].composer == "Brahms"
    assert records["music-hd-0001"].artist_group == "kevin macleod"
    assert records["music-hd-0002"].composer is None


def test_license_blocks_do_not_poison_other_grants() -> None:
    track_ids = {"music-fma-0001", "music-fma-0002", "music-fma-0003"}
    text = """music-fma-0001
CC BY 4.0
https://example.test/by
=============================================================================
music-fma-0002
Selections from the November 2006 Concert is licensed under a Attribution-ShareAlike 3.0 International License.
https://example.test/by-sa
=============================================================================
music-fma-0003
CC BY-ND 4.0
https://example.test/by-nd
"""

    records = parse_license_records(text, source="fma", member="musan/music/fma/LICENSE", track_ids=track_ids)

    assert records["music-fma-0001"].name == "CC BY 4.0"  # type: ignore[union-attr]
    assert records["music-fma-0002"].name == "CC BY-SA 3.0"  # type: ignore[union-attr]
    assert records["music-fma-0003"] is None

    no_version = parse_license_records(
        "music-fma-0001\nPop Singles Compilation 2014 is licensed under a Attribution License.\n",
        source="fma",
        member="musan/music/fma/LICENSE",
        track_ids={"music-fma-0001"},
    )
    assert no_version["music-fma-0001"].name == "CC BY"  # type: ignore[union-attr]


def test_capacity_enforces_task_cap_and_free_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path / "music", max_staging_bytes=1024, minimum_free_bytes=900)
    config.output_root.mkdir()
    (config.output_root / "existing").write_bytes(b"x" * 800)
    monkeypatch.setattr("recipes.speakrs.large.music_source._free_bytes", lambda _: 1000)

    with pytest.raises(PreparationError, match="task-data cap"):
        _ensure_capacity(config, 225, label="candidate")
    with pytest.raises(PreparationError, match="free-space reserve"):
        _ensure_capacity(config, 150, label="candidate")


def _staged(tmp_path: Path, track_id: str, pcm_sha256: str) -> StagedTrack:
    return StagedTrack(
        source="fixture",
        track_id=track_id,
        source_member=f"musan/music/fixture/{track_id}.wav",
        source_sha256="1" * 64,
        source_bytes=100,
        audio_path=tmp_path / f"{track_id}.flac",
        sha256="2" * 64,
        pcm_sha256=pcm_sha256,
        sample_count=1600,
        sample_rate=16_000,
        channels=1,
        rms=0.1,
        peak_abs=100,
    )


def test_split_keeps_artists_and_pcm_duplicates_together(tmp_path: Path) -> None:
    licence = LicenseRecord("CC BY 4.0", "https://example.test", "attribution", "LICENSE")
    tracks = [
        (
            TrackMetadata("fixture", f"music-a-{index}", ("jazz",), "N", artist, None, "ANNOTATIONS"),
            licence,
            _staged(tmp_path, f"music-a-{index}", pcm),
        )
        for index, (artist, pcm) in enumerate(
            [
                ("Artist_One", "a" * 64),
                ("Artist_One", "b" * 64),
                ("Artist_Two", "a" * 64),
                ("Artist_Three", "c" * 64),
                ("Artist_Four", "d" * 64),
                ("Artist_Five", "e" * 64),
                ("Artist_Six", "f" * 64),
                ("Artist_Seven", "0" * 64),
                ("Artist_Eight", "3" * 64),
                ("Artist_Nine", "4" * 64),
            ]
        )
    ]

    split, record = _split_tracks(tracks, "fixture-seed")

    assert split["music-a-0"] == split["music-a-1"] == split["music-a-2"]
    assert record["validation_count"] == 1


def test_prepare_stream_emits_loader_compatible_train_manifest(tmp_path: Path) -> None:
    annotations = (
        "music-fixture-0001 jazz N Artist_One Composer_One\n"
        "music-fixture-0002 classical N Artist_Two Composer_Two\n"
        "music-fixture-0003 rock Y Vocal_Artist\n"
    ).encode()
    licenses = (
        "music-fixture-0001\nCC BY 4.0\nhttps://example.test/one\n"
        "=============================================================================\n"
        "music-fixture-0002\nCC0 1.0\nhttps://example.test/two\n"
        "=============================================================================\n"
        "music-fixture-0003\nCC BY 4.0\nhttps://example.test/three\n"
    ).encode()
    tone_one = (np.sin(np.arange(16_000) * 0.1) * 12_000).astype(np.int16)
    tone_two = (np.sin(np.arange(16_000) * 0.2) * 10_000).astype(np.int16)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "fixture_ANNOTATIONS").write_bytes(annotations)
    (metadata_dir / "fixture_LICENSE").write_bytes(licenses)
    archive = _archive(
        [
            ("musan/music/fixture/music-fixture-0001.wav", _wav_bytes(tone_one)),
            ("musan/music/fixture/music-fixture-0002.wav", _wav_bytes(tone_two)),
            ("musan/music/fixture/music-fixture-0003.wav", _wav_bytes(tone_one)),
            ("musan/music/fixture/ANNOTATIONS", annotations),
            ("musan/music/fixture/LICENSE", licenses),
        ]
    )
    output = tmp_path / "music"
    manifest = prepare_stream(
        io.BytesIO(archive),
        _config(output, metadata_dir=metadata_dir),
        hashlib.md5(archive).hexdigest(),
        expected_bytes=len(archive),
    )

    assert manifest["sample_rate"] == 16_000
    assert manifest["track_count"] == 2
    assert {track["composer"] for track in manifest["tracks"]} == {"Composer_One", "Composer_Two"}
    assert all(isinstance(track["license"], dict) for track in manifest["tracks"])
    assert not (output / "audio" / "train" / "music-fixture-0003.flac").exists()
    loaded = load_music_manifest(output / "training-manifest.json")
    assert len(loaded.tracks) == 1
    assert loaded.tracks[0].split == "train"


def test_archive_metadata_must_match_bootstrap_bytes(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "fixture_ANNOTATIONS").write_text("music-fixture-0001 jazz N Artist\n")
    (metadata_dir / "fixture_LICENSE").write_text("music-fixture-0001\nCC BY 4.0\n")
    archive = _archive(
        [
            ("musan/music/fixture/music-fixture-0001.wav", _wav_bytes(np.full(1600, 1000, dtype=np.int16))),
            ("musan/music/fixture/ANNOTATIONS", b"music-fixture-0001 jazz N Different\n"),
            ("musan/music/fixture/LICENSE", b"music-fixture-0001\nCC BY 4.0\n"),
        ]
    )
    output = tmp_path / "music"

    with pytest.raises(PreparationError, match="differs from bootstrap"):
        prepare_stream(
            io.BytesIO(archive),
            _config(output, metadata_dir=metadata_dir),
            hashlib.md5(archive).hexdigest(),
            expected_bytes=len(archive),
        )

    assert not (output / ".complete.json").exists()


def test_truncated_archive_cannot_publish_ready_manifests(tmp_path: Path) -> None:
    archive = _archive(
        [
            ("musan/music/fixture/ANNOTATIONS", b"music-fixture-0001 jazz N Artist\n"),
            (
                "musan/music/fixture/LICENSE",
                b"music-fixture-0001\nCC BY 4.0\nhttps://example.test/one\n",
            ),
            (
                "musan/music/fixture/music-fixture-0001.wav",
                _wav_bytes(np.full(16_000, 1000, dtype=np.int16)),
            ),
        ]
    )
    output = tmp_path / "music"

    with pytest.raises((PreparationError, tarfile.TarError, EOFError)):
        prepare_stream(
            io.BytesIO(archive[:-32]),
            _config(output),
            hashlib.md5(archive).hexdigest(),
            expected_bytes=len(archive),
        )

    assert not (output / ".complete.json").exists()
    assert not (output / "manifest.json").exists()
    state = json.loads((output / ".staging" / ".state.json").read_text())
    assert state["state"] == "failed"


def test_reserve_failure_after_one_track_aborts_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    annotations = b"music-fixture-0001 jazz N One\nmusic-fixture-0002 jazz N Two\n"
    licenses = (
        b"music-fixture-0001\nCC BY 4.0\n"
        b"=============================================================================\n"
        b"music-fixture-0002\nCC BY 4.0\n"
    )
    first = _wav_bytes(np.full(16_000, 1000, dtype=np.int16))
    second = _wav_bytes(np.full(16_000, 2000, dtype=np.int16))
    archive = _archive(
        [
            ("musan/music/fixture/ANNOTATIONS", annotations),
            ("musan/music/fixture/LICENSE", licenses),
            ("musan/music/fixture/music-fixture-0001.wav", first),
            ("musan/music/fixture/music-fixture-0002.wav", second),
        ]
    )
    output = tmp_path / "music"
    free_values = iter([10**9, 10**9, 10**9, 1])
    monkeypatch.setattr("recipes.speakrs.large.music_source._free_bytes", lambda _: next(free_values))

    with pytest.raises(PreparationError, match="free-space reserve"):
        prepare_stream(
            io.BytesIO(archive),
            _config(output, minimum_free_bytes=1),
            hashlib.md5(archive).hexdigest(),
            expected_bytes=len(archive),
        )

    assert (output / ".staging" / "candidates" / "music-fixture-0001.flac").is_file()
    assert not (output / ".complete.json").exists()
    assert not (output / "manifest.json").exists()


def test_configuration_rejects_more_than_eight_gib(tmp_path: Path) -> None:
    with pytest.raises(PreparationError, match="max_staging_bytes"):
        _config(tmp_path / "music", max_staging_bytes=8 * GIB + 1)
