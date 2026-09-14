from __future__ import annotations

from pathlib import Path

import pytest

from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.simsamu import SimsamuMetadata, parse_simsamu_rttm


def _rttm(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "call.rttm"
    path.write_text(rows, encoding="utf-8")

    return path


def test_simsamu_maps_three_local_roles_in_metadata_order(tmp_path: Path) -> None:
    metadata = SimsamuMetadata("call", "speaker_3", ("speaker_1", "speaker_4"))
    annotation = parse_simsamu_rttm(
        _rttm(
            tmp_path,
            "SPEAKER <NA> <NA> 0.00 1.00 <NA> <NA> medecin <NA> <NA>\n"
            "SPEAKER <NA> <NA> 1.00 1.00 <NA> <NA> patient_1 <NA> <NA>\n"
            "SPEAKER <NA> <NA> 2.00 1.00 <NA> <NA> patient_2 <NA> <NA>\n",
        ),
        metadata,
        3.0,
    )

    assert [interval.speaker for interval in annotation.intervals] == ["speaker_3", "speaker_1", "speaker_4"]
    assert annotation.overlap_seconds == 0


def test_simsamu_rejects_role_that_metadata_cannot_map(tmp_path: Path) -> None:
    metadata = SimsamuMetadata("call", "speaker_1", ("speaker_2",))

    with pytest.raises(PreparationError, match="role does not match metadata"):
        parse_simsamu_rttm(
            _rttm(
                tmp_path,
                "SPEAKER <NA> <NA> 0.00 1.00 <NA> <NA> medecin <NA> <NA>\n"
                "SPEAKER <NA> <NA> 1.00 1.00 <NA> <NA> patient_1 <NA> <NA>\n",
            ),
            metadata,
            2.0,
        )


def test_simsamu_rejects_overlap(tmp_path: Path) -> None:
    metadata = SimsamuMetadata("call", "speaker_1", ("speaker_2",))

    with pytest.raises(PreparationError, match="contain overlap"):
        parse_simsamu_rttm(
            _rttm(
                tmp_path,
                "SPEAKER <NA> <NA> 0.00 1.10 <NA> <NA> medecin <NA> <NA>\n"
                "SPEAKER <NA> <NA> 1.00 1.00 <NA> <NA> patient <NA> <NA>\n",
            ),
            metadata,
            2.0,
        )


def test_simsamu_rejects_interval_after_audio_clock(tmp_path: Path) -> None:
    metadata = SimsamuMetadata("call", "speaker_1", ("speaker_2",))

    with pytest.raises(PreparationError, match="outside the decoded audio clock"):
        parse_simsamu_rttm(
            _rttm(
                tmp_path,
                "SPEAKER <NA> <NA> 0.00 1.00 <NA> <NA> medecin <NA> <NA>\n"
                "SPEAKER <NA> <NA> 1.00 1.01 <NA> <NA> patient <NA> <NA>\n",
            ),
            metadata,
            2.0,
        )
