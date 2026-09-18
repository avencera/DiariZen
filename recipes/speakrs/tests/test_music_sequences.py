"""Tests for deterministic long-gap sequence preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import recipes.speakrs.large.music_sequences as music_sequences
from recipes.speakrs.large.hashing import sha256_file
from recipes.speakrs.large.music_sequences import (
    SAMPLE_RATE,
    MusicSequenceError,
    SourceLabel,
    clip_source_rttm,
    load_parent_recordings,
    prepare_music_sequences,
)


def _digest(path: Path) -> str:
    return sha256_file(path)


def _pcm_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with sf.SoundFile(str(path)) as source:
        while True:
            block = source.read(65_536, dtype="int16", always_2d=True)
            if len(block) == 0:
                break
            digest.update(np.ascontiguousarray(block).tobytes())
    return digest.hexdigest()


def _write_audio(path: Path, seconds: int, *, frequency: float, amplitude: float) -> None:
    frames = seconds * SAMPLE_RATE
    samples = (amplitude * np.sin(2 * np.pi * frequency * np.arange(frames) / SAMPLE_RATE)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, format="FLAC", subtype="PCM_16")


def _bundle(tmp_path: Path, *, mutate_after_manifest: bool = False) -> Path:
    root = tmp_path / "bundle"
    audio = root / "audio" / "TEST" / "audio.flac"
    _write_audio(audio, 120, frequency=220, amplitude=0.2)
    recording_id = "meeting-01"
    rttm = root / "all.rttm"
    rttm.write_text(
        "\n".join(
            (
                f"SPEAKER {recording_id} 1 2 3 <NA> <NA> speaker_a <NA> <NA>",
                f"SPEAKER {recording_id} 1 47 3 <NA> <NA> speaker_b <NA> <NA>",
                f"SPEAKER {recording_id} 1 92 3 <NA> <NA> speaker_a <NA> <NA>",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    uem = root / "all.uem"
    uem.write_text(f"{recording_id} 1 0 120\n", encoding="utf-8")
    labels = root / "labels" / "TEST"
    labels.mkdir(parents=True)
    rttm_object = labels / f"{hashlib.sha256(rttm.read_bytes()).hexdigest()}.rttm"
    uem_object = labels / f"{hashlib.sha256(uem.read_bytes()).hexdigest()}.uem"
    rttm_object.write_bytes(rttm.read_bytes())
    uem_object.write_bytes(uem.read_bytes())
    wav = root / "wav.scp"
    wav.write_text(f"{recording_id} /container/audio.flac\n", encoding="utf-8")
    bundle = root / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "speakrs-training-bundle-v1",
                "release_sha256": "a" * 64,
                "wav_prefix": "/container",
                "manifests": {
                    "wav_scp": _digest(wav),
                    "rttm": _digest(rttm),
                    "uem": _digest(uem),
                },
                "recordings": [
                    {
                        "recording_id": recording_id,
                        "source": "TEST",
                        "audio_path": "audio/TEST/audio.flac",
                        "audio_sha256": _digest(audio),
                        "audio_size": audio.stat().st_size,
                        "rttm_sha256": _digest(rttm_object),
                        "uem_sha256": _digest(uem_object),
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if mutate_after_manifest:
        with audio.open("ab") as handle:
            handle.write(b"wrong")
    return root


def _pool(tmp_path: Path) -> Path:
    root = tmp_path / "pool"
    root.mkdir(parents=True)
    audio = root / "audio" / "music.flac"
    _write_audio(audio, 70, frequency=440, amplitude=0.12)
    manifest = {
        "schema": "speakrs-instrumental-music-v1",
        "sample_rate": SAMPLE_RATE,
        "tracks": [
            {
                "track_id": "music-01",
                "path": "audio/music.flac",
                "sha256": _digest(audio),
                "pcm_sha256": _pcm_digest(audio),
                "sample_count": 70 * SAMPLE_RATE,
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "source_sha256": "b" * 64,
                "split": "train",
                "artist": "fixture-artist",
                "vocals": "N",
                "source_member": "MUSAN/train/music/music.wav",
                "license": {"spdx": "CC-BY-4.0", "source": "fixture"},
            }
        ],
    }
    (root / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def test_clip_source_rttm_uses_exact_sample_offsets_and_boundaries() -> None:
    labels = (
        SourceLabel("parent", "spk", 100, 300, "0.00625", "0.01875"),
        SourceLabel("parent", "other", 400, 500, "0.025", "0.03125"),
    )
    clipped = clip_source_rttm(
        labels,
        source_start_frame=200,
        source_end_frame=450,
        output_recording_id="out",
        output_start_frame=1_000,
        parent_id="parent",
    )
    assert [(row.source_start_frame, row.source_end_frame) for row in clipped] == [(200, 300), (400, 450)]
    assert [(row.output_start_frame, row.output_end_frame) for row in clipped] == [(1_000, 1_100), (1_200, 1_250)]
    assert [row.speaker for row in clipped] == ["parent::spk", "parent::other"]


def test_prepare_renders_variants_gaps_and_full_uem(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    pool = _pool(tmp_path)
    output = tmp_path / "output"
    manifest = prepare_music_sequences(
        output=output,
        music_pool=pool,
        bundle_paths={"test": bundle},
        min_free_bytes=0,
    )
    assert manifest["schema"] == "speakrs-music-long-gaps"
    assert manifest["state"] == "ready"
    assert manifest["parent_count"] == 1
    assert len(manifest["outputs"]) == 4
    assert (output / "READY").is_file()
    assert not output.with_name(".output.partial").exists()
    assert all(not Path(row["audio_path"]).is_absolute() for row in manifest["outputs"])
    assert set(manifest["manifests"]) == {"wav_scp", "rttm", "uem"}
    assert manifest["manifests"]["wav_scp"] == _digest(output / "wav.scp")
    first_derivation = (output / "derivation.json").read_bytes()

    all_rttm = (output / "all.rttm").read_text(encoding="utf-8")
    assert "meeting-01::speaker_a" in all_rttm
    assert "meeting-01::speaker_b" in all_rttm
    wav_ids = {line.split(maxsplit=1)[0] for line in (output / "wav.scp").read_text(encoding="utf-8").splitlines()}
    uem_ids = {line.split()[0] for line in (output / "all.uem").read_text(encoding="utf-8").splitlines()}
    rttm_ids = {line.split()[1] for line in all_rttm.splitlines()}
    assert wav_ids == uem_ids == {row["recording_id"] for row in manifest["outputs"]}
    assert rttm_ids <= wav_ids
    assert {line.split()[7] for line in all_rttm.splitlines()} <= {"meeting-01::speaker_a", "meeting-01::speaker_b"}
    rttm_frames = {}
    for line in all_rttm.splitlines():
        fields = line.split()
        start = round(float(fields[3]) * SAMPLE_RATE)
        end = start + round(float(fields[4]) * SAMPLE_RATE)
        rttm_frames.setdefault(fields[1], []).append((start, end))
    for row in manifest["outputs"]:
        info = sf.info(output / row["audio_path"])
        assert info.samplerate == SAMPLE_RATE
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert info.frames == row["frames"]
        uem_line = next(
            line
            for line in (output / "all.uem").read_text(encoding="utf-8").splitlines()
            if line.startswith(row["recording_id"] + " ")
        )
        assert uem_line.split()[2:] == ["0.000000000", f"{row['frames'] / SAMPLE_RATE:.9f}"]
        for inserted in row["inserted_intervals"]:
            assert inserted["rttm_rows"] == 0
            assert all(
                max(inserted["output_start_frame"], start) >= min(inserted["output_end_frame"], end)
                for start, end in rttm_frames.get(row["recording_id"], ())
            )
            if inserted["kind"] == "silence":
                samples, _ = sf.read(
                    output / row["audio_path"],
                    start=inserted["output_start_frame"],
                    stop=inserted["output_end_frame"],
                    dtype="int16",
                )
                assert np.count_nonzero(samples) == 0
            else:
                assert inserted["track_id"] == "music-01"
                assert inserted["audio_sha256"] == manifest["music_pool"]["tracks"][0]["audio_sha256"]
                assert inserted["pcm_sha256"] == manifest["music_pool"]["tracks"][0]["pcm_sha256"]
                assert inserted["source_sha256"] == "b" * 64
                assert inserted["manifest_record"]["license"]["spdx"] == "CC-BY-4.0"
        assert all(segment["output_end_frame"] > segment["output_start_frame"] for segment in row["segments"])
        for mapping in row["source_interval_mappings"]:
            if mapping["kind"] == "speech_excerpt":
                assert mapping["output_end_frame"] - mapping["output_start_frame"] == (
                    mapping["source_end_frame"] - mapping["source_start_frame"]
                )
        assert row["parent_id"] == "meeting-01"
    by_variant = {row["recording_id"].rsplit("__", 1)[-1]: row for row in manifest["outputs"]}
    assert [segment["kind"] for segment in by_variant["leading_music"]["segments"]] == [
        "music",
        "speech",
        "silence",
        "speech",
        "speech",
    ]
    assert [segment["kind"] for segment in by_variant["silence_between"]["segments"]] == [
        "speech",
        "silence",
        "speech",
        "music",
        "speech",
    ]
    assert [segment["kind"] for segment in by_variant["music_between"]["segments"]] == [
        "speech",
        "music",
        "speech",
        "silence",
        "speech",
    ]
    assert [segment["kind"] for segment in by_variant["silence_music_between"]["segments"]] == [
        "speech",
        "silence",
        "music",
        "speech",
        "speech",
    ]
    assert {
        segment["duration_seconds"]
        for row in manifest["outputs"]
        for segment in row["segments"]
        if segment["kind"] in {"silence", "music"}
    } >= {10.0, 30.0, 60.0}

    again = prepare_music_sequences(
        output=output,
        music_pool=pool,
        bundle_paths={"test": bundle},
        min_free_bytes=0,
    )
    assert again["input_identity_sha256"] == manifest["input_identity_sha256"]
    assert (output / "derivation.json").read_bytes() == first_derivation


def test_prepare_rejects_source_hash_mismatch_before_output(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, mutate_after_manifest=True)
    pool = _pool(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(MusicSequenceError, match="SHA-256"):
        prepare_music_sequences(output=output, music_pool=pool, bundle_paths={"test": bundle}, min_free_bytes=0)
    assert not output.exists()


def test_prepare_rejects_tight_byte_bound_without_writes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    pool = _pool(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(MusicSequenceError, match="byte bound"):
        prepare_music_sequences(
            output=output,
            music_pool=pool,
            bundle_paths={"test": bundle},
            max_task_bytes=1,
            min_free_bytes=0,
        )
    assert not output.exists()


def test_failed_render_removes_only_its_staging_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    pool = _pool(tmp_path)
    output = tmp_path / "output"
    calls = 0

    def fail_after_reserve(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise MusicSequenceError("injected output write failure")

    monkeypatch.setattr(music_sequences, "_write_audio", fail_after_reserve)
    with pytest.raises(MusicSequenceError, match="injected output write failure"):
        music_sequences.prepare_music_sequences(
            output=output,
            music_pool=pool,
            bundle_paths={"test": bundle},
            min_free_bytes=0,
        )
    assert calls == 1
    assert not output.exists()
    assert not output.with_name(".output.partial").exists()


def test_load_parent_recordings_requires_declared_wav_basename(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    wav = bundle / "wav.scp"
    wav.write_text("meeting-01 /container/not-the-audio.flac\n", encoding="utf-8")
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifests"]["wav_scp"] = _digest(wav)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MusicSequenceError, match="wav.scp"):
        load_parent_recordings({"test": bundle})
