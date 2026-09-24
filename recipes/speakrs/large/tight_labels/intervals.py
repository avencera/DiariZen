"""Interval geometry used by merge, tighten, mapping, and gates."""

from __future__ import annotations

from typing import Iterable, Sequence

from .domain import Interval, SpeakerSegment


EPS = 1e-12


def as_spans(items: Iterable[Interval | SpeakerSegment | tuple[float, float]]) -> list[tuple[float, float]]:
    """Normalise labelled or unlabelled intervals to start/end pairs."""

    spans: list[tuple[float, float]] = []
    for item in items:
        if isinstance(item, (Interval, SpeakerSegment)):
            start, end = item.start, item.end
        else:
            start, end = item
        if end > start:
            spans.append((float(start), float(end)))
    return spans


def union_spans(items: Iterable[Interval | SpeakerSegment | tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping and touching spans."""

    ordered = sorted(as_spans(items), key=lambda span: (span[0], span[1]))
    if not ordered:
        return []
    merged = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def merge_spans(
    items: Iterable[Interval | SpeakerSegment | tuple[float, float]],
    *,
    max_gap: float,
    allowed: Sequence[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Merge spans separated by a gap shorter than max_gap.

    When allowed is set, a positive filled gap must lie inside that union.
    """

    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    ordered = sorted(as_spans(items), key=lambda span: (span[0], span[1]))
    if not ordered:
        return []
    merged = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        gap = start - merged[-1][1]
        can_fill = gap < max_gap
        if can_fill and gap > 0 and allowed is not None:
            can_fill = span_is_subset((merged[-1][1], start), allowed)
        if can_fill:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def intersect_spans(
    left: Iterable[Interval | SpeakerSegment | tuple[float, float]],
    right: Iterable[Interval | SpeakerSegment | tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the pairwise intersection of two unions."""

    a = union_spans(left)
    b = union_spans(right)
    out: list[tuple[float, float]] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def duration(items: Iterable[Interval | SpeakerSegment | tuple[float, float]]) -> float:
    """Return the union duration."""

    return sum(end - start for start, end in union_spans(items))


def overlap_duration(
    left: Iterable[Interval | SpeakerSegment | tuple[float, float]],
    right: Iterable[Interval | SpeakerSegment | tuple[float, float]],
) -> float:
    """Return the duration of the intersection of two unions."""

    return sum(end - start for start, end in intersect_spans(left, right))


def subtract_spans(
    left: Iterable[Interval | SpeakerSegment | tuple[float, float]],
    right: Iterable[Interval | SpeakerSegment | tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return left union minus right union."""

    keep = union_spans(left)
    cut = union_spans(right)
    if not keep:
        return []
    if not cut:
        return keep
    out: list[tuple[float, float]] = []
    j = 0
    for start, end in keep:
        cursor = start
        while j < len(cut) and cut[j][1] <= cursor:
            j += 1
        k = j
        while k < len(cut) and cut[k][0] < end:
            gap_end = min(end, cut[k][0])
            if gap_end > cursor:
                out.append((cursor, gap_end))
            cursor = max(cursor, cut[k][1])
            k += 1
        if end > cursor:
            out.append((cursor, end))
    return out


def span_is_subset(
    inner: tuple[float, float],
    outer: Sequence[tuple[float, float]],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return True when inner lies inside the outer union."""

    start, end = inner
    if end - start <= tolerance:
        return True
    remaining = end - start
    for outer_start, outer_end in outer:
        overlap = min(end, outer_end) - max(start, outer_start)
        if overlap > 0:
            remaining -= overlap
            if remaining <= tolerance:
                return True
    return remaining <= tolerance


def same_speaker_gaps(
    items: Iterable[Interval | SpeakerSegment | tuple[float, float]],
) -> list[float]:
    """Return positive gaps after overlapping segments of one speaker are merged."""

    merged = union_spans(items)
    return [merged[index][0] - merged[index - 1][1] for index in range(1, len(merged))]


def overlap_timeline(
    by_speaker: dict[str, Sequence[SpeakerSegment | tuple[float, float]]],
) -> float:
    """Return time when at least two speakers are active."""

    speakers = [union_spans(spans) for spans in by_speaker.values() if spans]
    if len(speakers) < 2:
        return 0.0
    events: list[tuple[float, int]] = []
    for spans in speakers:
        for start, end in spans:
            events.append((start, 1))
            events.append((end, -1))
    events.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    last = 0.0
    total = 0.0
    for time, delta in events:
        if active >= 2:
            total += time - last
        active += delta
        last = time
    return total
