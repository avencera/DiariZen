"""Same-speaker short-gap merge."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .domain import RecordingLabels, SpeakerSegment
from .intervals import merge_spans, union_spans


MIN_GAP_SECONDS = 0.200


def merge_short_gaps(
    segments: Sequence[SpeakerSegment],
    *,
    min_gap: float = MIN_GAP_SECONDS,
    within: Sequence[SpeakerSegment] | None = None,
) -> tuple[SpeakerSegment, ...]:
    """Merge overlapping, touching, and short same-speaker gaps.

    Gaps shorter than min_gap become speech. A gap of exactly min_gap stays.
    When within is set, a filled gap must lie inside that speaker's original
    union so the machine does not add speech outside a human segment.
    """

    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")
    grouped: dict[tuple[str, str], list[SpeakerSegment]] = defaultdict(list)
    for segment in segments:
        grouped[(segment.recording_id, segment.speaker)].append(segment)
    allowed: dict[tuple[str, str], list[tuple[float, float]]] = {}
    if within is not None:
        for segment in within:
            allowed.setdefault((segment.recording_id, segment.speaker), []).append((segment.start, segment.end))
        for key, spans in list(allowed.items()):
            allowed[key] = union_spans(spans)

    merged: list[SpeakerSegment] = []
    for (recording_id, speaker), group in grouped.items():
        allowed_union = allowed.get((recording_id, speaker), []) if within is not None else None
        spans = merge_spans(group, max_gap=min_gap, allowed=allowed_union)
        merged.extend(SpeakerSegment(recording_id, speaker, start, end) for start, end in spans)
    merged.sort(key=lambda item: (item.start, item.end, item.speaker))
    return tuple(merged)


def merge_recording(
    labels: RecordingLabels,
    *,
    min_gap: float = MIN_GAP_SECONDS,
    within: Sequence[SpeakerSegment] | None = None,
) -> RecordingLabels:
    """Merge short gaps on a recording and keep uncertain regions."""

    return RecordingLabels(
        labels.recording_id,
        merge_short_gaps(labels.segments, min_gap=min_gap, within=within),
        labels.uncertain,
    )
