"""Same-speaker short-gap merge."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .domain import RecordingLabels, SpeakerSegment
from .intervals import merge_spans, union_spans
from .time import MIN_GAP_NS, MIN_GAP_SECONDS


def merge_short_gaps(
    segments: Sequence[SpeakerSegment],
    *,
    min_gap_ns: int = MIN_GAP_NS,
    within: Sequence[SpeakerSegment] | None = None,
) -> tuple[SpeakerSegment, ...]:
    """Merge overlapping, touching, and short same-speaker gaps.

    Gaps shorter than min_gap_ns become speech. A gap of exactly min_gap_ns stays.
    When within is set, a filled gap must lie inside that speaker's original
    union so the machine does not add speech outside a human segment.
    """

    if isinstance(min_gap_ns, bool) or not isinstance(min_gap_ns, int) or min_gap_ns < 0:
        raise ValueError("min_gap_ns must be a non-negative integer")
    grouped: dict[tuple[str, str], list[SpeakerSegment]] = defaultdict(list)
    for segment in segments:
        grouped[(segment.recording_id, segment.speaker)].append(segment)
    allowed: dict[tuple[str, str], list[tuple[int, int]]] = {}
    if within is not None:
        for segment in within:
            allowed.setdefault((segment.recording_id, segment.speaker), []).append(
                (segment.start_ns, segment.end_ns)
            )
        for key, spans in list(allowed.items()):
            allowed[key] = union_spans(spans)

    merged: list[SpeakerSegment] = []
    for (recording_id, speaker), group in grouped.items():
        allowed_union = allowed.get((recording_id, speaker), []) if within is not None else None
        spans = merge_spans(group, max_gap_ns=min_gap_ns, allowed=allowed_union)
        merged.extend(SpeakerSegment(recording_id, speaker, start_ns, end_ns) for start_ns, end_ns in spans)
    merged.sort(key=lambda item: (item.start_ns, item.end_ns, item.speaker))
    return tuple(merged)


def merge_recording(
    labels: RecordingLabels,
    *,
    min_gap_ns: int = MIN_GAP_NS,
    within: Sequence[SpeakerSegment] | None = None,
) -> RecordingLabels:
    """Merge short gaps on a recording and keep uncertain regions."""

    return RecordingLabels(
        labels.recording_id,
        merge_short_gaps(labels.segments, min_gap_ns=min_gap_ns, within=within),
        labels.uncertain,
    )


__all__ = ["MIN_GAP_NS", "MIN_GAP_SECONDS", "merge_recording", "merge_short_gaps"]
