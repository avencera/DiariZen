"""Same-speaker short-gap merge."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .domain import RecordingLabels, SpeakerSegment, UncertainInterval
from .intervals import merge_spans, subtract_spans, union_spans
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


def canonicalize_uncertain(
    uncertain: Sequence[UncertainInterval],
    active: Sequence[SpeakerSegment],
) -> tuple[tuple[UncertainInterval, ...], int]:
    """Union UNCERTAIN per speaker, drop same-speaker speech, and return removed time."""

    grouped: dict[tuple[str, str], list[UncertainInterval]] = defaultdict(list)
    for region in uncertain:
        grouped[(region.recording_id, region.speaker)].append(region)
    active_by_speaker: dict[tuple[str, str], list[SpeakerSegment]] = defaultdict(list)
    for segment in active:
        active_by_speaker[(segment.recording_id, segment.speaker)].append(segment)
    remaining: list[UncertainInterval] = []
    removed_ns = 0
    for (recording_id, speaker), regions in grouped.items():
        unioned = union_spans(regions)
        leftover = subtract_spans(unioned, active_by_speaker.get((recording_id, speaker), ()))
        removed_ns += sum(end_ns - start_ns for start_ns, end_ns in unioned) - sum(
            end_ns - start_ns for start_ns, end_ns in leftover
        )
        remaining.extend(
            UncertainInterval(recording_id, speaker, start_ns, end_ns) for start_ns, end_ns in leftover
        )
    remaining.sort(key=lambda item: (item.start_ns, item.end_ns, item.speaker))
    return tuple(remaining), removed_ns


def subtract_uncertain_from_active(
    uncertain: Sequence[UncertainInterval],
    active: Sequence[SpeakerSegment],
) -> tuple[UncertainInterval, ...]:
    """Union overlapping and touching UNCERTAIN per speaker, then drop active speech."""

    remaining, _removed_ns = canonicalize_uncertain(uncertain, active)
    return remaining


def merge_recording(
    labels: RecordingLabels,
    *,
    min_gap_ns: int = MIN_GAP_NS,
    within: Sequence[SpeakerSegment] | None = None,
) -> RecordingLabels:
    """Merge short gaps on a recording and drop overlapping uncertain time."""

    merged = merge_short_gaps(labels.segments, min_gap_ns=min_gap_ns, within=within)
    return RecordingLabels(
        labels.recording_id,
        merged,
        labels.uncertain,
        uncertain_removed_ns=labels.uncertain_removed_ns,
    )


__all__ = [
    "MIN_GAP_NS",
    "MIN_GAP_SECONDS",
    "merge_recording",
    "merge_short_gaps",
    "subtract_uncertain_from_active",
]
