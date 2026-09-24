"""Interval geometry used by merge, tighten, mapping, and gates."""

from __future__ import annotations

from typing import Iterable, Sequence

from .domain import Interval, RecordingLabels, SpeakerSegment, UncertainInterval


NsSpan = tuple[int, int]


def as_spans(items: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan]) -> list[NsSpan]:
    """Normalise labelled or unlabelled intervals to start/end nanosecond pairs."""

    spans: list[NsSpan] = []
    for item in items:
        if isinstance(item, (Interval, SpeakerSegment, UncertainInterval)):
            start_ns, end_ns = item.start_ns, item.end_ns
        else:
            start_ns, end_ns = item
        if end_ns > start_ns:
            spans.append((start_ns, end_ns))
    return spans


def union_spans(items: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan]) -> list[NsSpan]:
    """Merge overlapping and touching spans."""

    return _merge_spans(items, touch=True)


def union_overlapping_spans(
    items: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
) -> list[NsSpan]:
    """Merge strictly overlapping spans and keep touching spans separate."""

    return _merge_spans(items, touch=False)


def _merge_spans(
    items: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
    *,
    touch: bool,
) -> list[NsSpan]:
    ordered = sorted(as_spans(items), key=lambda span: (span[0], span[1]))
    if not ordered:
        return []
    merged = [[ordered[0][0], ordered[0][1]]]
    for start_ns, end_ns in ordered[1:]:
        can_merge = start_ns <= merged[-1][1] if touch else start_ns < merged[-1][1]
        if can_merge:
            merged[-1][1] = max(merged[-1][1], end_ns)
        else:
            merged.append([start_ns, end_ns])
    return [(start_ns, end_ns) for start_ns, end_ns in merged]


def merge_spans(
    items: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
    *,
    max_gap_ns: int,
    allowed: Sequence[NsSpan] | None = None,
) -> list[NsSpan]:
    """Merge spans separated by a gap shorter than max_gap_ns.

    A gap of exactly max_gap_ns stays. When allowed is set, a positive filled
    gap must lie inside that union.
    """

    if isinstance(max_gap_ns, bool) or not isinstance(max_gap_ns, int) or max_gap_ns < 0:
        raise ValueError("max_gap_ns must be a non-negative integer")
    ordered = sorted(as_spans(items), key=lambda span: (span[0], span[1]))
    if not ordered:
        return []
    merged = [[ordered[0][0], ordered[0][1]]]
    for start_ns, end_ns in ordered[1:]:
        gap_ns = start_ns - merged[-1][1]
        can_fill = gap_ns < max_gap_ns
        if can_fill and gap_ns > 0 and allowed is not None:
            can_fill = span_is_subset((merged[-1][1], start_ns), allowed)
        if can_fill:
            merged[-1][1] = max(merged[-1][1], end_ns)
        else:
            merged.append([start_ns, end_ns])
    return [(start_ns, end_ns) for start_ns, end_ns in merged]


def intersect_spans(
    left: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
    right: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
) -> list[NsSpan]:
    """Return the pairwise intersection of two unions."""

    a = union_spans(left)
    b = union_spans(right)
    out: list[NsSpan] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        start_ns = max(a[i][0], b[j][0])
        end_ns = min(a[i][1], b[j][1])
        if end_ns > start_ns:
            out.append((start_ns, end_ns))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def duration_ns(items: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan]) -> int:
    """Return the union duration in nanoseconds."""

    return sum(end_ns - start_ns for start_ns, end_ns in union_spans(items))


def overlap_duration_ns(
    left: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
    right: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
) -> int:
    """Return the duration of the intersection of two unions."""

    return sum(end_ns - start_ns for start_ns, end_ns in intersect_spans(left, right))


def subtract_spans(
    left: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
    right: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
) -> list[NsSpan]:
    """Return left union minus right union."""

    keep = union_spans(left)
    cut = union_spans(right)
    if not keep:
        return []
    if not cut:
        return keep
    out: list[NsSpan] = []
    j = 0
    for start_ns, end_ns in keep:
        cursor = start_ns
        while j < len(cut) and cut[j][1] <= cursor:
            j += 1
        k = j
        while k < len(cut) and cut[k][0] < end_ns:
            gap_end = min(end_ns, cut[k][0])
            if gap_end > cursor:
                out.append((cursor, gap_end))
            cursor = max(cursor, cut[k][1])
            k += 1
        if end_ns > cursor:
            out.append((cursor, end_ns))
    return out


def span_is_subset(
    inner: NsSpan,
    outer: Sequence[NsSpan],
    *,
    tolerance_ns: int = 0,
) -> bool:
    """Return True when inner lies inside the outer union."""

    start_ns, end_ns = inner
    if end_ns - start_ns <= tolerance_ns:
        return True
    remaining = end_ns - start_ns
    for outer_start, outer_end in outer:
        overlap = min(end_ns, outer_end) - max(start_ns, outer_start)
        if overlap > 0:
            remaining -= overlap
            if remaining <= tolerance_ns:
                return True
    return remaining <= tolerance_ns


def same_speaker_gaps_ns(
    items: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
) -> list[int]:
    """Return positive gaps after overlapping segments of one speaker are merged."""

    merged = union_spans(items)
    return [merged[index][0] - merged[index - 1][1] for index in range(1, len(merged))]


def overlap_timeline_ns(
    by_speaker: dict[str, Sequence[SpeakerSegment | NsSpan]],
) -> int:
    """Return time when at least two speakers are active."""

    speakers = [union_spans(spans) for spans in by_speaker.values() if spans]
    if len(speakers) < 2:
        return 0
    events: list[tuple[int, int]] = []
    for spans in speakers:
        for start_ns, end_ns in spans:
            events.append((start_ns, 1))
            events.append((end_ns, -1))
    events.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    last = 0
    total = 0
    for time_ns, delta in events:
        if active >= 2:
            total += time_ns - last
        active += delta
        last = time_ns
    return total


def clip_span(start_ns: int, end_ns: int, uem_start_ns: int, uem_end_ns: int) -> NsSpan | None:
    """Clip one span to the UEM. Drop it when it starts at or after the end."""

    if start_ns >= uem_end_ns:
        return None
    clipped_start = max(start_ns, uem_start_ns)
    clipped_end = min(end_ns, uem_end_ns)
    if clipped_end <= clipped_start:
        return None
    return (clipped_start, clipped_end)


def clip_spans(
    items: Iterable[Interval | SpeakerSegment | UncertainInterval | NsSpan],
    uem_start_ns: int,
    uem_end_ns: int,
) -> list[NsSpan]:
    """Clip spans to [uem_start_ns, uem_end_ns] and drop empty results."""

    out: list[NsSpan] = []
    for start_ns, end_ns in as_spans(items):
        clipped = clip_span(start_ns, end_ns, uem_start_ns, uem_end_ns)
        if clipped is not None:
            out.append(clipped)
    return out


def clip_recording_to_uem(
    labels: RecordingLabels,
    uem_start_ns: int,
    uem_end_ns: int,
) -> RecordingLabels:
    """Clip SPEAKER and UNCERTAIN rows to the UEM before later merges."""

    segments: list[SpeakerSegment] = []
    for item in labels.segments:
        clipped = clip_span(item.start_ns, item.end_ns, uem_start_ns, uem_end_ns)
        if clipped is not None:
            segments.append(SpeakerSegment(labels.recording_id, item.speaker, clipped[0], clipped[1]))
    uncertain: list[UncertainInterval] = []
    for item in labels.uncertain:
        clipped = clip_span(item.start_ns, item.end_ns, uem_start_ns, uem_end_ns)
        if clipped is not None:
            uncertain.append(UncertainInterval(labels.recording_id, item.speaker, clipped[0], clipped[1]))
    return RecordingLabels(
        labels.recording_id,
        tuple(segments),
        tuple(uncertain),
        uncertain_removed_ns=labels.uncertain_removed_ns,
    )


def span_outside_uem(start_ns: int, end_ns: int, uem_start_ns: int, uem_end_ns: int) -> bool:
    """Return True when a row starts before the UEM or ends after it."""

    return start_ns < uem_start_ns or end_ns > uem_end_ns
