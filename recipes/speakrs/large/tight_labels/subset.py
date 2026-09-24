"""Subset checks of tight speech against original human segments."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .domain import RecordingLabels, SpeakerSegment
from .intervals import span_is_subset, union_spans


def speaker_union(segments: Sequence[SpeakerSegment], speaker: str) -> list[tuple[int, int]]:
    """Return the merged original union for one speaker."""

    return union_spans(item for item in segments if item.speaker == speaker)


def is_subset_of_original(
    tight: Sequence[SpeakerSegment],
    original: Sequence[SpeakerSegment],
    *,
    speaker: str | None = None,
    tolerance_ns: int = 0,
) -> bool:
    """Return True when every tight interval lies inside the original union."""

    grouped: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in tight:
        if speaker is None or segment.speaker == speaker:
            grouped[segment.speaker].append(segment)
    originals: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in original:
        if speaker is None or segment.speaker == speaker:
            originals[segment.speaker].append(segment)
    for name, pieces in grouped.items():
        outer = union_spans(originals.get(name, ()))
        for piece in pieces:
            if not span_is_subset((piece.start_ns, piece.end_ns), outer, tolerance_ns=tolerance_ns):
                return False
    return True


def recording_is_subset(
    tight: RecordingLabels,
    original: RecordingLabels,
    *,
    tolerance_ns: int = 0,
) -> bool:
    """Return True when every speaker of tight is a subset of original."""

    if tight.recording_id != original.recording_id:
        raise ValueError("subset check requires the same recording_id")
    return is_subset_of_original(
        tight.segments,
        original.segments,
        tolerance_ns=tolerance_ns,
    )
