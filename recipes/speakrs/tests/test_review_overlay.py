"""Protect overlay extraction, proposal conversion, and packet immutability."""

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
from recipes.speakrs.large.errors import ContractError, PreparationError
from recipes.speakrs.large.hashing import sha256_file
from recipes.speakrs.large.review_overlay import (
    _extract_selected_transcripts,
    prepare_review_overlay,
    validate_review_overlay,
)
from recipes.speakrs.large.review_packet import (
    build_review_packet,
    scan_archive_metadata,
    select_review_windows,
)


def _flac_bytes(duration: float = 31.0, role_index: int = 0) -> bytes:
    time = np.arange(int(duration * 48_000), dtype=np.float32) / 48_000
    samples = 0.15 * np.sin(2 * np.pi * (180 + role_index * 40) * time)
    audio = io.BytesIO()
    sf.write(audio, samples, 48_000, format="FLAC", subtype="PCM_16")
    return audio.getvalue()


def _transcript(parent_id: str, index: str, words: list[dict[str, object]], text: str) -> dict[str, object]:
    return {
        "conversation_id": parent_id,
        "speaker_index": index,
        "language": "en",
        "text": text,
        "words": words,
    }


def _add_json(archive: tarfile.TarFile, name: str, value: object) -> None:
    payload = json.dumps(value).encode("utf-8")
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _write_archive(path: Path, *, parent_count: int = 2, extra_parent: bool = False) -> Path:
    duration = 31.0
    with tarfile.open(path, mode="w:gz") as archive:
        parents = [f"conv_test_{index}" for index in range(parent_count)]
        if extra_parent:
            parents.append("conv_extra_not_selected")
        for parent_index, parent_id in enumerate(parents):
            for role_index, role in enumerate(("speaker_a", "speaker_b")):
                _add_bytes(archive, f"conversations/{parent_id}/{role}.flac", _flac_bytes(duration, role_index))
            words_a = [
                {"word": "hello", "start": 0.2, "end": 0.6, "type": "word", "corrections_applied": False},
                {"word": "later", "start": 10.0, "end": 10.2, "type": "word"},
                {"word": "haha", "start": 0.7, "end": 0.9, "type": "laugh"},
            ]
            words_b = [
                {"word": "yes", "start": 0.4, "end": 0.8, "type": "word"},
                {"word": "umm", "start": 1.0, "end": 1.1, "type": "filler"},
            ]
            if parent_id.endswith("not_selected"):
                words_a = [{"word": "secret", "start": 0.1, "end": 0.2, "type": "word"}]
            _add_json(
                archive,
                f"conversations/{parent_id}/meta.json",
                {
                    "conversation_id": parent_id,
                    "duration_seconds": duration,
                    "speaker_a_id": f"private-speaker-a-{parent_index}",
                    "speaker_b_id": f"private-speaker-b-{parent_index}",
                },
            )
            speaker_meta = {
                "speaker_id": f"private-speaker-a-{parent_index}",
                "recording": {"sample_rate": 48_000, "channels": 1, "duration_seconds": duration},
                "audio_metrics": {"dnsmos_ovr_median": 3.0},
            }
            _add_json(archive, f"conversations/{parent_id}/speaker_a_meta.json", speaker_meta)
            _add_json(
                archive,
                f"conversations/{parent_id}/speaker_b_meta.json",
                {**speaker_meta, "speaker_id": f"private-speaker-b-{parent_index}"},
            )
            _add_json(
                archive,
                f"conversations/{parent_id}/speaker_a_transcript.json",
                _transcript(parent_id, "a", words_a, "hello later haha"),
            )
            _add_json(
                archive,
                f"conversations/{parent_id}/speaker_b_transcript.json",
                _transcript(parent_id, "b", words_b, "yes umm"),
            )
    return path


def _limits(tmp_path: Path) -> DiskLimits:
    staging = tmp_path / "staging"
    cache = tmp_path / "cache"
    staging.mkdir()
    cache.mkdir()
    return DiskLimits(staging, cache, 1_000_000_000, 1_000_000_000, 0, 1)


def _packet_files(root: Path) -> dict[str, str]:
    digests = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[path.relative_to(root).as_posix()] = sha256_file(path)
    return digests


def _build_packet(tmp_path: Path, *, extra_parent: bool = False) -> tuple[Path, Path]:
    archive = _write_archive(tmp_path / "open-yap.tar.gz", parent_count=2, extra_parent=extra_parent)
    inventory = scan_archive_metadata(archive)
    selection = select_review_windows(inventory, uniform_windows=1, targeted_windows=3, min_parents=2)
    limits = _limits(tmp_path)
    packet = limits.staging_root / "packet"
    build_review_packet(
        archive,
        packet,
        limits,
        inventory=inventory,
        selection=selection,
        maximum_packet_bytes=200_000_000,
    )
    return archive, packet


def test_overlay_clips_words_and_leaves_packet_bytes_unchanged(tmp_path: Path) -> None:
    archive, packet = _build_packet(tmp_path)
    before = _packet_files(packet)
    session = tmp_path / "session"
    first = prepare_review_overlay(packet, archive, session)
    after = _packet_files(packet)

    assert first["ok"] is True
    assert first["window_count"] == 4
    assert before == after
    overlay_root = session / "overlay"
    validated = validate_review_overlay(overlay_root, packet_root=packet)
    assert validated["ok"] is True
    second = prepare_review_overlay(packet, archive, session)
    assert second["reused"] is True
    assert second["overlay_sha256"] == first["overlay_sha256"]
    window_dir = next((overlay_root / "windows").iterdir())
    transcript = json.loads((window_dir / "transcript.json").read_text(encoding="utf-8"))
    texts = {word["text"] for word in transcript["words"]}
    assert "secret" not in texts
    assert "hello later haha" not in json.dumps(transcript)
    assert transcript["display_only"] is True
    assert transcript["not_human_activity"] is True
    proposal = json.loads((window_dir / "proposal.json").read_text(encoding="utf-8"))
    assert proposal["conversion_policy"] == "speakrs-open-yap-grid-proposal-v1"
    assert all(item["end_frame"] > item["start_frame"] for item in proposal["intervals"])
    assert all(0 <= item["start_frame"] < item["end_frame"] <= 1500 for item in proposal["intervals"])


def test_overlay_is_deterministic_across_fresh_runs(tmp_path: Path) -> None:
    archive, packet = _build_packet(tmp_path)
    first = prepare_review_overlay(packet, archive, tmp_path / "session-a")
    second = prepare_review_overlay(packet, archive, tmp_path / "session-b")
    assert first["overlay_sha256"] == second["overlay_sha256"]
    assert first["reused"] is False
    assert second["reused"] is False


def test_partial_overlay_is_replaced_and_never_served(tmp_path: Path) -> None:
    archive, packet = _build_packet(tmp_path)
    destination = tmp_path / "session"
    partial = tmp_path / "session.partial"
    partial.mkdir()
    (partial / "junk.txt").write_text("incomplete", encoding="utf-8")
    result = prepare_review_overlay(packet, archive, destination)
    assert result["ok"] is True
    assert not partial.exists()
    assert (destination / "overlay" / "overlay-manifest.json").is_file()
    with pytest.raises(PreparationError):
        validate_review_overlay(partial)


def _member_archive(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in members.items():
            _add_bytes(archive, name, payload)
    return path


def _transcript_bytes(word: str = "hello") -> bytes:
    return json.dumps(
        _transcript(
            "conv_test_0",
            "a",
            [{"word": word, "start": 0.2, "end": 0.6, "type": "word"}],
            word,
        )
    ).encode("utf-8")


def test_unknown_transcript_field_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(_transcript_bytes())
    payload["unexpected"] = "nope"
    raw = json.dumps(payload).encode("utf-8")
    name = "conversations/conv_test_0/speaker_a_transcript.json"
    archive = _member_archive(tmp_path / "t.tar.gz", {name: raw})
    members = {
        name: {
            "parent_id": "conv_test_0",
            "role": "speaker_a",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    }
    with pytest.raises(ContractError, match="unknown fields"):
        _extract_selected_transcripts(archive, members, progress=None)


def test_missing_selected_transcript_is_rejected(tmp_path: Path) -> None:
    archive = _member_archive(tmp_path / "t.tar.gz", {})
    members = {
        "conversations/conv_test_0/speaker_a_transcript.json": {
            "parent_id": "conv_test_0",
            "role": "speaker_a",
            "sha256": "a" * 64,
            "size_bytes": 12,
        }
    }
    with pytest.raises(PreparationError, match="missing from the archive"):
        _extract_selected_transcripts(archive, members, progress=None)


def test_wrong_hash_selected_transcript_is_rejected(tmp_path: Path) -> None:
    raw = _transcript_bytes("hello")
    name = "conversations/conv_test_0/speaker_a_transcript.json"
    archive = _member_archive(tmp_path / "t.tar.gz", {name: raw})
    members = {
        name: {
            "parent_id": "conv_test_0",
            "role": "speaker_a",
            "sha256": "0" * 64,
            "size_bytes": len(raw),
        }
    }
    with pytest.raises(PreparationError, match="hash does not match"):
        _extract_selected_transcripts(archive, members, progress=None)


def test_duplicate_selected_transcript_is_rejected(tmp_path: Path) -> None:
    raw = _transcript_bytes()
    name = "conversations/conv_test_0/speaker_a_transcript.json"
    archive_path = tmp_path / "t.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        _add_bytes(archive, name, raw)
        _add_bytes(archive, name, raw)
    members = {
        name: {
            "parent_id": "conv_test_0",
            "role": "speaker_a",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    }
    with pytest.raises(PreparationError, match="duplicated"):
        _extract_selected_transcripts(archive_path, members, progress=None)


def test_oversized_transcript_member_is_rejected(tmp_path: Path) -> None:
    raw = b"{" + b"x" * (65 * 1024 * 1024)
    name = "conversations/conv_test_0/speaker_a_transcript.json"
    archive = _member_archive(tmp_path / "t.tar.gz", {name: raw})
    members = {
        name: {
            "parent_id": "conv_test_0",
            "role": "speaker_a",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    }
    with pytest.raises(PreparationError, match="bounded JSON limit"):
        _extract_selected_transcripts(archive, members, progress=None)


def test_unexpected_parent_transcript_is_not_retained(tmp_path: Path) -> None:
    archive, packet = _build_packet(tmp_path, extra_parent=True)
    session = tmp_path / "session"
    prepare_review_overlay(packet, archive, session)
    packet_parents = {
        record["window"]["parent_id"]
        for record in json.loads((packet / "manifest.json").read_text(encoding="utf-8"))["windows"]
    }
    overlay = json.loads((session / "overlay" / "overlay-manifest.json").read_text(encoding="utf-8"))
    member_parents = {facts["parent_id"] for facts in overlay["source_transcript_members"].values()}
    assert member_parents <= packet_parents
    if "conv_extra_not_selected" not in packet_parents:
        dumped = json.dumps(overlay)
        assert "conv_extra_not_selected" not in dumped
        for path in (session / "overlay").rglob("transcript.json"):
            payload = path.read_text(encoding="utf-8")
            assert "secret" not in payload
            assert "conv_extra_not_selected" not in payload


def test_original_packet_hashes_are_stable_after_failed_prepare(tmp_path: Path) -> None:
    archive, packet = _build_packet(tmp_path)
    before = _packet_files(packet)
    with tarfile.open(archive, mode="w:gz"):
        pass
    with pytest.raises(PreparationError):
        prepare_review_overlay(packet, archive, tmp_path / "session")
    assert _packet_files(packet) == before
    assert hashlib.sha256((packet / "manifest.json").read_bytes()).hexdigest() == before["manifest.json"]
