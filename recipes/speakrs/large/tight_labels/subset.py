"""Subset checks of tight speech against original human segments."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .domain import RecordingLabels, SpeakerSegment
from .intervals import merge_spans, span_is_subset, union_spans
from .time import MIN_GAP_NS


def speaker_union(segments: Sequence[SpeakerSegment], speaker: str) -> list[tuple[int, int]]:
    """Return the merged original union for one speaker."""

    return union_spans(item for item in segments if item.speaker == speaker)


def closed_speaker_union(
    segments: Sequence[SpeakerSegment],
    speaker: str,
    *,
    min_gap_ns: int = MIN_GAP_NS,
) -> list[tuple[int, int]]:
    """Return the original union after the same-speaker short-gap merge."""

    return merge_spans(
        (item for item in segments if item.speaker == speaker),
        max_gap_ns=min_gap_ns,
    )


def is_subset_of_original(
    tight: Sequence[SpeakerSegment],
    original: Sequence[SpeakerSegment],
    *,
    speaker: str | None = None,
    tolerance_ns: int = 0,
    min_gap_ns: int = MIN_GAP_NS,
) -> bool:
    """Return True when every tight interval lies inside the 200 ms-closed original.

    Original same-speaker gaps shorter than min_gap_ns are speech for this
    check, so the machine may bridge those gaps and must not add other speech.
    """

    grouped: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in tight:
        if speaker is None or segment.speaker == speaker:
            grouped[segment.speaker].append(segment)
    originals: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in original:
        if speaker is None or segment.speaker == speaker:
            originals[segment.speaker].append(segment)
    for name, pieces in grouped.items():
        outer = merge_spans(originals.get(name, ()), max_gap_ns=min_gap_ns)
        for piece in pieces:
            if not span_is_subset((piece.start_ns, piece.end_ns), outer, tolerance_ns=tolerance_ns):
                return False
    return True


def recording_is_subset(
    tight: RecordingLabels,
    original: RecordingLabels,
    *,
    tolerance_ns: int = 0,
    min_gap_ns: int = MIN_GAP_NS,
) -> bool:
    """Return True when every speaker of tight is a subset of original."""

    if tight.recording_id != original.recording_id:
        raise ValueError("subset check requires the same recording_id")
    return is_subset_of_original(
        tight.segments,
        original.segments,
        tolerance_ns=tolerance_ns,
        min_gap_ns=min_gap_ns,
    )
