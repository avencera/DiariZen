"""Tightness statistics against the published forced-AMI reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .domain import RecordingLabels, SpeakerSegment
from .intervals import same_speaker_gaps_ns, union_spans
from .time import MID_GAP_NS, SHORT_GAP_NS, ns_to_seconds


FORCED_AMI_GAPS_LE_0_3_PER_MIN = 4.63
FORCED_AMI_GAPS_0_3_TO_1_PER_MIN = 17.00
FORCED_AMI_SMALLEST_GAP_SECONDS = 0.200


@dataclass(frozen=True, slots=True)
class TightnessStats:
    """Same-speaker gap and segment-length summary for one label set."""

    recording_count: int
    speaker_seconds: float
    median_segment_s: float | None
    p90_segment_s: float | None
    gaps_le_0_3_per_min: float
    gaps_0_3_to_1_per_min: float
    smallest_gap_s: float | None
    smallest_gap_ns: int | None
    segment_count: int
    gap_count: int

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready record."""

        return {
            "recording_count": self.recording_count,
            "speaker_seconds": self.speaker_seconds,
            "speaker_hours": self.speaker_seconds / 3600.0,
            "median_segment_s": self.median_segment_s,
            "p90_segment_s": self.p90_segment_s,
            "gaps_le_0_3_per_min": self.gaps_le_0_3_per_min,
            "gaps_0_3_to_1_per_min": self.gaps_0_3_to_1_per_min,
            "smallest_gap_s": self.smallest_gap_s,
            "smallest_gap_ns": self.smallest_gap_ns,
            "segment_count": self.segment_count,
            "gap_count": self.gap_count,
        }


FORCED_AMI_REFERENCE = TightnessStats(
    recording_count=170,
    speaker_seconds=74.0 * 3600.0,
    median_segment_s=0.97,
    p90_segment_s=3.23,
    gaps_le_0_3_per_min=FORCED_AMI_GAPS_LE_0_3_PER_MIN,
    gaps_0_3_to_1_per_min=FORCED_AMI_GAPS_0_3_TO_1_PER_MIN,
    smallest_gap_s=FORCED_AMI_SMALLEST_GAP_SECONDS,
    smallest_gap_ns=200_000_000,
    segment_count=0,
    gap_count=0,
)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def measure_tightness(
    segments: Sequence[SpeakerSegment] | Sequence[RecordingLabels],
) -> TightnessStats:
    """Measure short-gap rates after merging overlapping same-speaker segments."""

    if segments and isinstance(segments[0], RecordingLabels):
        recordings = list(segments)
        items = [item for recording in recordings for item in recording.segments]
        recording_ids = {recording.recording_id for recording in recordings}
    else:
        items = list(segments)  # type: ignore[arg-type]
        recording_ids = {item.recording_id for item in items}

    by_speaker: dict[tuple[str, str], list[SpeakerSegment]] = {}
    for item in items:
        by_speaker.setdefault((item.recording_id, item.speaker), []).append(item)

    lengths_s: list[float] = []
    gaps_ns: list[int] = []
    speaker_ns = 0
    for group in by_speaker.values():
        merged = union_spans(group)
        speaker_ns += sum(end_ns - start_ns for start_ns, end_ns in merged)
        lengths_s.extend(ns_to_seconds(end_ns - start_ns) for start_ns, end_ns in merged)
        gaps_ns.extend(same_speaker_gaps_ns(merged))

    minutes = ns_to_seconds(speaker_ns) / 60.0 if speaker_ns > 0 else 0.0
    short = sum(1 for gap_ns in gaps_ns if gap_ns <= SHORT_GAP_NS)
    mid = sum(1 for gap_ns in gaps_ns if SHORT_GAP_NS < gap_ns <= MID_GAP_NS)
    smallest_ns = min(gaps_ns) if gaps_ns else None
    return TightnessStats(
        recording_count=len(recording_ids),
        speaker_seconds=ns_to_seconds(speaker_ns),
        median_segment_s=_percentile(lengths_s, 50),
        p90_segment_s=_percentile(lengths_s, 90),
        gaps_le_0_3_per_min=(short / minutes) if minutes > 0 else 0.0,
        gaps_0_3_to_1_per_min=(mid / minutes) if minutes > 0 else 0.0,
        smallest_gap_s=None if smallest_ns is None else ns_to_seconds(smallest_ns),
        smallest_gap_ns=smallest_ns,
        segment_count=len(lengths_s),
        gap_count=len(gaps_ns),
    )
