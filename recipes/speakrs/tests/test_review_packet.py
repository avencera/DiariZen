"""Protect bounded Open Yap review-packet construction."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recipes.speakrs.large.contracts import DiskLimits
from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.review_packet import (
    build_review_packet,
    load_archive_inventory,
    scan_archive_metadata,
    select_review_windows,
    validate_review_packet,
)


def _archive(tmp_path: Path, parent_count: int = 3, *, global_manifest: bool = False) -> Path:
    archive_path = tmp_path / "open-yap.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for parent_index in range(parent_count):
            parent_id = f"conv_test_{parent_index}"
            duration = 31.0
            for role_index, role in enumerate(("speaker_a", "speaker_b")):
                time = np.arange(int(duration * 48_000), dtype=np.float32) / 48_000
                samples = 0.15 * np.sin(2 * np.pi * (180 + role_index * 40) * time)
                audio = io.BytesIO()
                sf.write(audio, samples, 48_000, format="FLAC", subtype="PCM_16")
                payload = audio.getvalue()
                member = tarfile.TarInfo(f"conversations/{parent_id}/{role}.flac")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            meta = {
                "conversation_id": parent_id,
                "duration_seconds": duration,
                "speaker_a_id": f"private-speaker-a-{parent_index}",
                "speaker_b_id": f"private-speaker-b-{parent_index}",
            }
            speaker_meta = {
                "speaker_id": f"private-speaker-a-{parent_index}",
                "recording": {"sample_rate": 48_000, "channels": 1, "duration_seconds": duration},
                "audio_metrics": {"dnsmos_ovr_median": 3.0},
            }
            words_a = [
                {"start": 0.2, "end": 0.6, "word": "not-retained"},
                {"start": 10.0, "end": 10.1, "word": "not-retained"},
            ]
            words_b = [{"start": 0.4, "end": 0.8, "word": "not-retained"}]
            json_members = {
                "meta.json": meta,
                "speaker_a_meta.json": speaker_meta,
                "speaker_b_meta.json": {**speaker_meta, "speaker_id": f"private-speaker-b-{parent_index}"},
                "speaker_a_transcript.json": {"words": words_a, "text": "must-not-be-retained"},
                "speaker_b_transcript.json": {"words": words_b, "text": "must-not-be-retained"},
            }
            for basename, value in json_members.items():
                payload = json.dumps(value).encode("utf-8")
                member = tarfile.TarInfo(f"conversations/{parent_id}/{basename}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        if global_manifest:
            payload = f'{{"parents": {parent_count}, "text": "not retained"}}'.encode("utf-8")
            member = tarfile.TarInfo("conversations/manifest.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return archive_path


def _limits(tmp_path: Path) -> DiskLimits:
    staging = tmp_path / "staging"
    cache = tmp_path / "cache"
    staging.mkdir()
    cache.mkdir()
    return DiskLimits(staging, cache, 1_000_000_000, 1_000_000_000, 0, 1)


def test_scan_uses_full_parent_population_and_hashes_identity_values(tmp_path: Path) -> None:
    archive = _archive(tmp_path, parent_count=3)
    inventory = scan_archive_metadata(archive, expected_size_bytes=archive.stat().st_size)

    assert len(inventory.parents) == 3
    assert inventory.available_parents == 3
    assert inventory.population_window_count == 153
    report = json.dumps(inventory.to_dict())
    assert "private-speaker-a" not in report
    assert "must-not-be-retained" not in report
    assert all(parent.speaker_id_sha256["speaker_a"] for parent in inventory.parents)


def test_selection_is_deterministic_disjoint_and_stratified(tmp_path: Path) -> None:
    inventory = scan_archive_metadata(_archive(tmp_path, parent_count=4))

    first = select_review_windows(inventory, seed=3407, uniform_windows=4, targeted_windows=8, min_parents=3)
    second = select_review_windows(inventory, seed=3407, uniform_windows=4, targeted_windows=8, min_parents=3)

    assert first.to_dict() == second.to_dict()
    assert len(first.uniform_windows) == 4
    assert len(first.targeted_windows) == 8
    assert len({window.key() for window in first.windows}) == 12
    assert first.parent_count >= 3
    assert {window.stratum for window in first.targeted_windows} == {
        "overlap",
        "quiet-speech",
        "short-turns",
        "channel-defects",
    }
    assert all(window.start_frame % 1 == 0 and window.end_frame - window.start_frame == 1500 for window in first)


def test_build_retains_pair_references_separate_candidate_and_blank_human_fields(tmp_path: Path) -> None:
    archive = _archive(tmp_path, parent_count=2)
    inventory = scan_archive_metadata(archive)
    selection = select_review_windows(inventory, uniform_windows=1, targeted_windows=3, min_parents=2)
    limits = _limits(tmp_path)
    result = build_review_packet(
        archive,
        limits.staging_root,
        limits,
        inventory=inventory,
        selection=selection,
        maximum_packet_bytes=200_000_000,
    )

    assert result["window_count"] == 4
    assert validate_review_packet(limits.staging_root)["ok"] is True
    manifest = json.loads((limits.staging_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["upload_allowed"] is False
    assert manifest["clock_status"] == "unresolved_pending_human_review"
    assert manifest["privacy"]["transcript_text_retained"] is False
    procedure = json.loads((limits.staging_root / "source-procedure.json").read_text(encoding="utf-8"))
    format_spec = procedure["human_annotation_format"]
    assert format_spec["speaker_activity"]["speaker_allowed"] == ["speaker_a", "speaker_b"]
    assert "window-relative seconds" in format_spec["speaker_activity"]["time_fields"]
    assert "FORMAT EXAMPLE — NOT A FILLED REFERENCE:" in (limits.staging_root / "review-guide.md").read_text(
        encoding="utf-8"
    )
    record = manifest["windows"][0]
    window_dir = limits.staging_root / "windows" / record["window"]["window_id"]
    info = sf.info(window_dir / "emitted.flac")
    assert (info.samplerate, info.channels, info.frames) == (16_000, 1, 480_000)
    reference_info = sf.info(window_dir / "reference-speaker_a.flac")
    assert (reference_info.samplerate, reference_info.channels, reference_info.frames) == (48_000, 1, 1_440_000)
    candidate = json.loads((window_dir / "candidate-annotation.json").read_text(encoding="utf-8"))
    human = json.loads((window_dir / "human-annotation.json").read_text(encoding="utf-8"))
    assert candidate["human_reference"] is None
    assert "must-not-be-retained" not in json.dumps(candidate)
    assert human["reviewer_id"] is None
    assert human["independent_signoff"] is None
    assert human["speaker_activity"] is None
    assert human["window_disposition"] is None
    assert record["pairing_and_clock"]["zero_offset_assumed"] is False
    assert hashlib.sha256((window_dir / "reference-speaker_a.flac").read_bytes()).hexdigest()


def test_scan_rejects_wrong_archive_identity(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    with pytest.raises(PreparationError, match="SHA-256"):
        scan_archive_metadata(archive, expected_sha256="0" * 64)


def test_scan_records_observed_global_manifest_without_retaining_content(tmp_path: Path) -> None:
    archive = _archive(tmp_path, global_manifest=True)
    inventory = scan_archive_metadata(archive, expected_size_bytes=archive.stat().st_size)

    facts = inventory.global_metadata_members["conversations/manifest.json"]
    assert facts["size_bytes"] == len(b'{"parents": 3, "text": "not retained"}')
    assert facts["content_retained"] is False
    assert "not retained" not in json.dumps(inventory.to_dict())


def test_inventory_round_trip_reuses_sealed_archive_identity(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    inventory = scan_archive_metadata(archive)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory.to_dict()), encoding="utf-8")

    loaded = load_archive_inventory(
        inventory_path,
        archive_path=archive,
        expected_sha256=inventory.archive_sha256,
        expected_size_bytes=inventory.archive_size_bytes,
    )

    assert loaded.to_dict() == inventory.to_dict()
