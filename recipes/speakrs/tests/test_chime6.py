from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest
import soundfile as sf

from recipes.speakrs.large import chime6
from recipes.speakrs.large.acceptance import RttmInterval
from recipes.speakrs.large.chime6 import (
    CHIME6_CANONICAL_VIEW,
    CHIME6_SESSION_SPEAKERS,
    CHIME6_TRAIN_SESSIONS,
    Chime6ActivityKind,
    Chime6CanonicalAudio,
    Chime6Role,
    Chime6TagAudit,
    Chime6Transcript,
    chime6_canonical_audio_member,
    chime6_party_id,
    chime6_role,
    classify_chime6_words,
    parse_chime6_time,
    parse_chime6_transcript,
    prepare_chime6_train_audio,
    prepare_chime6_train_labels,
)
from recipes.speakrs.large.errors import PreparationError


def _row(words: str, *, session_id: str = "S03", speaker: str = "P12") -> dict[str, str]:
    return {
        "end_time": "00:01:00.39",
        "start_time": "00:00:57.55",
        "words": words,
        "speaker": speaker,
        "session_id": session_id,
    }


def test_publisher_roles_and_canonical_view_are_fixed() -> None:
    assert len(CHIME6_TRAIN_SESSIONS) == 16
    assert chime6_role("S03") is Chime6Role.TRAIN
    assert chime6_role("S02") is Chime6Role.DEVELOPMENT
    assert chime6_role("S01") is Chime6Role.EVALUATION
    assert chime6_party_id("S03") == chime6_party_id("S04")
    assert chime6_party_id("S03") != chime6_party_id("S05")
    assert len({chime6_party_id(session_id) for session_id in CHIME6_TRAIN_SESSIONS}) == 8
    assert CHIME6_SESSION_SPEAKERS["S03"] == frozenset({"P09", "P10", "P11", "P12"})
    assert CHIME6_CANONICAL_VIEW == "U06.CH1"
    assert chime6_canonical_audio_member("S03") == "audio/train/S03_U06.CH1.wav"

    with pytest.raises(PreparationError, match="train sessions"):
        chime6_canonical_audio_member("S02")


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ("plain speech", Chime6ActivityKind.SPEECH),
        ("[noise] plain speech [inaudible 0:00:58.96]", Chime6ActivityKind.SPEECH),
        ("[laughs]", Chime6ActivityKind.LAUGHTER),
        ("[inaudible 0:00:58.96]", Chime6ActivityKind.INAUDIBLE_SPEECH),
        ("[noise]", Chime6ActivityKind.NON_SPEECH_NOISE),
        ("", Chime6ActivityKind.UNLABELED),
    ],
)
def test_transcript_tags_have_explicit_activity_meanings(words: str, expected: Chime6ActivityKind) -> None:
    assert classify_chime6_words(words) is expected


def test_parser_excludes_noise_only_rows_and_keeps_speaker_vocal_activity() -> None:
    transcript = parse_chime6_transcript(
        [
            _row("[noise]"),
            _row(""),
            _row("[laughs]", speaker="P09"),
            _row("speech", speaker="P10"),
            _row("speech", speaker="P11"),
        ],
        expected_session_id="S03",
        expected_role=Chime6Role.TRAIN,
    )

    assert transcript.intervals == (
        RttmInterval("S03", 57.55, 60.39, "P09"),
        RttmInterval("S03", 57.55, 60.39, "P10"),
        RttmInterval("S03", 57.55, 60.39, "P11"),
    )
    assert transcript.speakers == frozenset({"P09", "P10", "P11", "P12"})
    assert transcript.tag_audit.non_speech_noise == 1
    assert transcript.tag_audit.unlabeled == 1
    assert transcript.tag_audit.laughter == 1
    assert transcript.tag_audit.speech == 2


def test_parser_rejects_role_leakage_unknown_tags_and_clock_errors() -> None:
    with pytest.raises(PreparationError, match="role"):
        parse_chime6_transcript(
            [_row("speech")],
            expected_session_id="S03",
            expected_role=Chime6Role.DEVELOPMENT,
        )
    with pytest.raises(PreparationError, match="unknown bracket tag"):
        classify_chime6_words("[redacted]")
    with pytest.raises(PreparationError, match="clock bounds"):
        parse_chime6_time("00:60:00.00")
    with pytest.raises(PreparationError, match="invalid form"):
        parse_chime6_time("0:01:00")


def test_parser_rejects_unknown_fields() -> None:
    row = _row("speech")
    row["unexpected"] = "value"

    with pytest.raises(PreparationError, match="invalid fields"):
        parse_chime6_transcript(
            [row],
            expected_session_id="S03",
            expected_role=Chime6Role.TRAIN,
        )


def test_archive_member_hash_mismatch_stops_before_conversion(tmp_path) -> None:
    archive = tmp_path / "train.tar.gz"
    payload = b"not the publisher wav"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("CHiME6/audio/train/S03_U06.CH1.wav")
        info.size = len(payload)
        output.addfile(info, io.BytesIO(payload))

    with pytest.raises(PreparationError, match="MD5"):
        prepare_chime6_train_audio(archive, tmp_path / "output", minimum_free_bytes=0)

    assert not list((tmp_path / "output").glob("*.flac"))
    assert not list((tmp_path / "output/receipts").glob("*.json"))


def test_audio_preparation_resumes_from_atomic_per_session_receipts(tmp_path, monkeypatch) -> None:
    wav = io.BytesIO()
    sf.write(wav, [0.0] * 1_600, 16_000, format="WAV", subtype="PCM_16")
    payload = wav.getvalue()
    source_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    for session_id in CHIME6_TRAIN_SESSIONS:
        monkeypatch.setitem(chime6.CHIME6_TRAIN_U06_CH1_MD5, session_id, source_md5)

    archive = tmp_path / "train.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for session_id in sorted(CHIME6_TRAIN_SESSIONS):
            info = tarfile.TarInfo(f"CHiME6/audio/train/{session_id}_U06.CH1.wav")
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))

    first = prepare_chime6_train_audio(archive, tmp_path / "output", minimum_free_bytes=0)
    assert {item.sample_count for item in first} == {1_600}
    monkeypatch.setattr(chime6, "_copy_member_with_md5", lambda *_args: pytest.fail("source recopied"))

    resumed = prepare_chime6_train_audio(archive, tmp_path / "output", minimum_free_bytes=0)

    assert resumed == first
    assert len(list((tmp_path / "output/receipts").glob("*.json"))) == 16

    corrupt = tmp_path / "output/S03.flac"
    with corrupt.open("r+b") as output:
        output.seek(-1, 2)
        final_byte = output.read(1)[0]
        output.seek(-1, 2)
        output.write(bytes([final_byte ^ 0xFF]))
    with pytest.raises(PreparationError, match="receipt does not match"):
        prepare_chime6_train_audio(archive, tmp_path / "output", minimum_free_bytes=0)


def _complete_label_inputs(tmp_path: Path):
    transcripts = []
    audio = []
    for session_id in sorted(CHIME6_TRAIN_SESSIONS):
        speakers = CHIME6_SESSION_SPEAKERS[session_id]
        intervals = [RttmInterval(session_id, 1.0, 2.0, speaker) for speaker in sorted(speakers)]
        if session_id == "S03":
            intervals.append(RttmInterval(session_id, 1.5, 3.0, "P09"))
        transcripts.append(
            Chime6Transcript(
                session_id,
                chime6_party_id(session_id),
                Chime6Role.TRAIN,
                tuple(intervals),
                speakers,
                Chime6TagAudit(speech=len(intervals)),
            )
        )
        audio.append(
            Chime6CanonicalAudio(
                session_id,
                f"CHiME6/audio/train/{session_id}_U06.CH1.wav",
                10,
                "0" * 32,
                tmp_path / f"{session_id}.flac",
                "0" * 64,
                10,
                16_000,
                1,
                64_000,
            )
        )

    return transcripts, audio


def test_train_labels_merge_same_speaker_activity_and_bind_sample_clock(tmp_path) -> None:
    transcripts, audio = _complete_label_inputs(tmp_path)

    labels = prepare_chime6_train_labels(transcripts, audio, tmp_path / "labels")

    assert len(labels) == 16
    s03 = (tmp_path / "labels/S03.rttm").read_text(encoding="utf-8")
    assert "SPEAKER S03 1 1 2 <NA> <NA> P09 <NA> <NA>\n" in s03
    assert (tmp_path / "labels/S03.uem").read_text(encoding="utf-8") == "S03 1 0 4\n"
    assert labels[0].rttm_path.parent == tmp_path / "labels"


def test_train_labels_reject_activity_beyond_the_sample_clock(tmp_path) -> None:
    transcripts, audio = _complete_label_inputs(tmp_path)
    s03 = next(item for item in transcripts if item.session_id == "S03")
    transcripts[transcripts.index(s03)] = Chime6Transcript(
        s03.session_id,
        s03.party_id,
        s03.role,
        (*s03.intervals, RttmInterval("S03", 3.0, 5.0, "P09")),
        s03.speakers,
        s03.tag_audit,
    )

    with pytest.raises(PreparationError, match="sample clock"):
        prepare_chime6_train_labels(transcripts, audio, tmp_path / "labels")

    assert not (tmp_path / "labels.partial").exists()
