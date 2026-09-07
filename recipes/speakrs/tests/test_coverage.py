"""Tests for source-EOF annotation conversion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recipes.speakrs.large.acceptance import RttmInterval
from recipes.speakrs.large.coverage import (
    IdentityClockEvidence,
    VerifiedSourceIdentity,
    bound_rttm_to_source_eof,
)
from recipes.speakrs.large.errors import PreparationError


def _source(
    tmp_path: Path, *, mapping: str = "identity", clock_shift_observed: bool = False
) -> VerifiedSourceIdentity:
    evidence_path = tmp_path / "identity-clock.json"
    source_sha256 = hashlib.sha256(b"source bytes").hexdigest()
    evidence_path.write_text(
        json.dumps(
            {
                "recording_id": "recording",
                "source_audio": {
                    "sha256": source_sha256,
                    "frames": 100,
                    "sample_rate": 10,
                },
                "timing_facts": {
                    "source_timeline_mapping": mapping,
                    "clock_shift_observed": clock_shift_observed,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = IdentityClockEvidence.from_file(evidence_path)
    return VerifiedSourceIdentity("recording", source_sha256, 100, 10, evidence)


def test_bound_rttm_preserves_in_bounds_and_intersects_only_at_eof(tmp_path: Path) -> None:
    source = _source(tmp_path)
    intervals = (
        RttmInterval("recording", 1.0, 2.0, "speaker-a"),
        RttmInterval("recording", 9.0, 12.0, "speaker-b"),
        RttmInterval("recording", 11.0, 12.0, "speaker-c"),
    )

    result = bound_rttm_to_source_eof(intervals, source)

    assert result.bounded_intervals == (
        intervals[0],
        RttmInterval("recording", 9.0, 10.0, "speaker-b"),
    )
    assert result.bounded_intervals[0] is intervals[0]
    assert result.receipt["original_interval_count"] == 3
    assert result.receipt["bounded_interval_count"] == 2
    assert result.receipt["original_speaker_seconds"] == pytest.approx(5.0)
    assert result.receipt["retained_speaker_seconds"] == pytest.approx(2.0)
    assert result.excluded_speaker_seconds == pytest.approx(3.0)
    assert result.receipt["uem_policy"].startswith("unchanged")
    assert result.to_rttm() == (
        "SPEAKER recording 1 1 1 <NA> <NA> speaker-a <NA> <NA>\n"
        "SPEAKER recording 1 9 1 <NA> <NA> speaker-b <NA> <NA>\n"
    )


def test_bound_rttm_keeps_intervals_ending_at_eof_exactly(tmp_path: Path) -> None:
    interval = RttmInterval("recording", 9.0, 10.0, "speaker")

    result = bound_rttm_to_source_eof((interval,), _source(tmp_path))

    assert result.intervals == (interval,)
    assert result.intervals[0] is interval
    assert result.excluded_speaker_seconds == 0.0


@pytest.mark.parametrize(
    "interval",
    [
        RttmInterval("recording", -1.0, 1.0, "speaker"),
        RttmInterval("recording", float("nan"), 1.0, "speaker"),
        RttmInterval("recording", 1.0, float("nan"), "speaker"),
    ],
)
def test_bound_rttm_rejects_invalid_bounds(tmp_path: Path, interval: RttmInterval) -> None:
    with pytest.raises(PreparationError, match="bounds"):
        bound_rttm_to_source_eof((interval,), _source(tmp_path))


def test_bound_rttm_rejects_unknown_parent(tmp_path: Path) -> None:
    with pytest.raises(PreparationError, match="different parent"):
        bound_rttm_to_source_eof((RttmInterval("other", 1.0, 2.0, "speaker"),), _source(tmp_path))


def test_bound_rttm_rejects_nonidentity_clock(tmp_path: Path) -> None:
    with pytest.raises(PreparationError, match="identity timeline"):
        bound_rttm_to_source_eof(
            (RttmInterval("recording", 1.0, 2.0, "speaker"),),
            _source(tmp_path, mapping="offset"),
        )


def test_bound_rttm_rejects_missing_or_changed_evidence(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source.identity_clock_evidence.reference.unlink()
    with pytest.raises(PreparationError, match="evidence is missing"):
        bound_rttm_to_source_eof((RttmInterval("recording", 1.0, 2.0, "speaker"),), source)


def test_bound_rttm_rejects_empty_result(tmp_path: Path) -> None:
    with pytest.raises(PreparationError, match="no bounded intervals"):
        bound_rttm_to_source_eof((RttmInterval("recording", 10.0, 12.0, "speaker"),), _source(tmp_path))


def test_bound_rttm_requires_source_and_evidence_facts_to_match(tmp_path: Path) -> None:
    source = _source(tmp_path)
    mismatched = VerifiedSourceIdentity(
        "recording",
        hashlib.sha256(b"different source bytes").hexdigest(),
        source.sample_count,
        source.sample_rate,
        source.identity_clock_evidence,
    )

    with pytest.raises(PreparationError, match="source SHA-256 differs"):
        bound_rttm_to_source_eof((RttmInterval("recording", 1.0, 2.0, "speaker"),), mismatched)
