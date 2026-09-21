"""Tests for the Indic DiarBench source-bound admission rules."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.indic_diarbench import (
    INDIC_CAPACITY_PROFILES,
    DuplicateMatch,
    DuplicateReference,
    IndicDatasetType,
    IndicPreparationState,
    IndicRejectionReason,
    IndicRowError,
    RawPublisherRow,
    admit_training_rows,
    advance_state,
    audit_duplicates,
    canonical_speaker_id,
    capacity_profiles,
    validate_publisher_row,
)


def _audio_bytes(duration: float = 2.0) -> bytes:
    sample_count = int(duration * 16_000)
    samples = 0.2 * np.sin(np.linspace(0.0, 40.0, sample_count, endpoint=False))
    output = io.BytesIO()
    sf.write(output, samples, 16_000, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _row(
    audio: bytes | None = None,
    *,
    sample_id: str = "hindi_001",
    recording_id: str = "hindi_nf_001",
    dataset_type: str = "Near field",
    duration_seconds: float = 2.0,
    segments: list[dict[str, object]] | None = None,
) -> RawPublisherRow:
    audio = _audio_bytes() if audio is None else audio
    transcript = segments or [
        {"speaker_id": "spk-1", "transcript": "one", "start_time": 0.0, "end_time": 0.8},
        {"speaker_id": "spk-2", "transcript": "two", "start_time": 0.4, "end_time": 1.2},
        {"speaker_id": "spk-1", "transcript": "three", "start_time": 1.0, "end_time": 1.5},
    ]
    payload = {
        "audio": {"bytes": audio, "path": f"{sample_id}.wav"},
        "recording_id": recording_id,
        "language": "Hindi",
        "annotated_transcript": transcript,
        "dataset_type": dataset_type,
        "sample_id": sample_id,
        "num_speakers": len({item["speaker_id"] for item in transcript}),
        "num_segments": len(transcript),
        "duration_seconds": duration_seconds,
    }
    return RawPublisherRow("Hindi/test-00000-of-00001.parquet", 0, payload, audio)


def test_validation_uses_decoded_clock_and_preserves_overlap() -> None:
    row = validate_publisher_row(_row())

    assert row.dataset_type is IndicDatasetType.NEAR_FIELD
    assert row.canonical_id == "indic-diarbench::hindi_001"
    assert row.parent_identity == "hindi_nf_001"
    assert row.audio.sample_rate_hz == 16_000
    assert row.audio.channels == 1
    assert row.audio.frames == 32_000
    assert row.speaker_seconds == pytest.approx(2.1)
    assert row.speech_seconds == pytest.approx(1.5)
    assert row.overlap_seconds == pytest.approx(0.6)
    assert row.max_simultaneous_speakers == 2


def test_canonical_speaker_tokens_preserve_labels_with_whitespace() -> None:
    row = validate_publisher_row(
        _row(
            segments=[
                {"speaker_id": "Speaker 0", "transcript": "one", "start_time": 0.0, "end_time": 0.5},
                {"speaker_id": "Speaker 1", "transcript": "two", "start_time": 0.5, "end_time": 1.0},
            ]
        )
    )

    assert row.canonical_speakers == tuple(
        sorted((canonical_speaker_id("Speaker 0"), canonical_speaker_id("Speaker 1")))
    )
    assert row.speaker_id_map[canonical_speaker_id("Speaker 0")] == "Speaker 0"


def test_validation_rejects_out_of_audio_without_clipping() -> None:
    with pytest.raises(IndicRowError, match="annotation extends beyond decoded audio") as raised:
        validate_publisher_row(
            _row(segments=[{"speaker_id": "spk-1", "transcript": "bad", "start_time": 1.9, "end_time": 2.1}])
        )

    assert raised.value.reason is IndicRejectionReason.LABEL_OUT_OF_AUDIO


def test_validation_rejects_metadata_duration_mismatch() -> None:
    with pytest.raises(IndicRowError, match="metadata duration differs") as raised:
        validate_publisher_row(_row(duration_seconds=3.0))

    assert raised.value.reason is IndicRejectionReason.METADATA_DURATION_MISMATCH


def test_validation_rejects_non_mapping_publisher_rows() -> None:
    raw = RawPublisherRow("Hindi/test-00000-of-00001.parquet", 0, None, b"")

    with pytest.raises(IndicRowError, match="publisher row is not a mapping") as raised:
        validate_publisher_row(raw)

    assert raised.value.reason is IndicRejectionReason.INVALID_SCHEMA


def test_duplicate_audit_checks_decoded_audio_and_practical_identity() -> None:
    first = validate_publisher_row(_row())
    second = validate_publisher_row(
        _row(
            audio=_audio_bytes(1.9),
            sample_id="hindi_002",
            recording_id="hindi_nf_002",
            duration_seconds=1.9,
        )
    )
    reference = DuplicateReference(
        source="VoxConverse",
        recording_id="vox-1",
        parent_id="vox-parent-1",
        raw_sha256=None,
        decoded_pcm_sha256=first.audio.decoded_pcm_sha256,
        practical_fingerprint=first.audio.practical_fingerprint,
        frames=first.audio.frames,
    )

    matches = audit_duplicates((first, second), (reference,))

    assert [(item.sample_id, item.match_kind) for item in matches] == [("hindi_001", "decoded-pcm")]


def test_admission_excludes_web_rows_and_duplicate_rows_with_reasons() -> None:
    meeting = validate_publisher_row(_row())
    web = validate_publisher_row(_row(sample_id="hindi_002", recording_id="hindi_itw_001", dataset_type="In the wild"))
    duplicate = DuplicateMatch(
        sample_id=meeting.sample_id,
        against_source="AVA-AVD",
        against_recording_id="video-1",
        reason_code=IndicRejectionReason.DUPLICATE_AUDIO,
        match_kind="decoded-pcm",
    )
    inventory = SimpleNamespace(rows=(meeting, web), rejected_rows=())

    admission = admit_training_rows(inventory, (duplicate,))

    assert admission.accepted == ()
    assert [item.reason_code for item in admission.excluded] == [
        IndicRejectionReason.DUPLICATE_AUDIO,
        IndicRejectionReason.WEB_PARENT_IDENTITY_UNAVAILABLE,
    ]


def test_admission_excludes_recordings_without_an_eight_second_window() -> None:
    row = validate_publisher_row(_row())
    inventory = SimpleNamespace(rows=(row,), rejected_rows=())

    admission = admit_training_rows(inventory, ())

    assert admission.accepted == ()
    assert admission.excluded[0].reason_code is IndicRejectionReason.CAPACITY_WINDOW_UNAVAILABLE


def test_state_transitions_cannot_skip_release_evidence() -> None:
    assert advance_state(IndicPreparationState.DOWNLOADED, IndicPreparationState.INVENTORIED)
    with pytest.raises(PreparationError, match="invalid Indic"):
        advance_state(IndicPreparationState.DOWNLOADED, IndicPreparationState.ACCEPTED)


def test_capacity_contract_is_exactly_the_three_required_profiles() -> None:
    assert capacity_profiles() == INDIC_CAPACITY_PROFILES
    assert {(item["local_slots"], item["max_overlap"]) for item in capacity_profiles()} == {(4, 4), (6, 6), (8, 8)}
    assert {item["chunk_shift"] for item in capacity_profiles()} == {6}
    assert {item["model_num_frames"] for item in capacity_profiles()} == {399}
