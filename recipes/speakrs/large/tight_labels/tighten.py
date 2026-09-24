"""Tighten human segments with aligned word or phone intervals."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from .domain import Interval, RecordingLabels, SpeakerSegment, UncertainInterval
from .intervals import duration_ns, intersect_spans, union_spans
from .merge import MIN_GAP_NS, merge_recording, merge_short_gaps
from .time import COVERAGE_RATIO, coverage_meets


@dataclass(frozen=True, slots=True)
class TightenSegmentResult:
    """Active speech and optional uncertain interior for one human segment."""

    active: tuple[SpeakerSegment, ...]
    uncertain: tuple[UncertainInterval, ...]
    coverage: float
    fallback: bool


def union_human_segments(segments: Sequence[SpeakerSegment]) -> tuple[SpeakerSegment, ...]:
    """Union overlapping or touching human segments of the same speaker."""

    grouped: dict[tuple[str, str], list[SpeakerSegment]] = defaultdict(list)
    for segment in segments:
        grouped[(segment.recording_id, segment.speaker)].append(segment)
    unioned: list[SpeakerSegment] = []
    for (recording_id, speaker), group in grouped.items():
        for start_ns, end_ns in union_spans(group):
            unioned.append(SpeakerSegment(recording_id, speaker, start_ns, end_ns))
    unioned.sort(key=lambda item: (item.start_ns, item.end_ns, item.speaker))
    return tuple(unioned)


def tighten_segment(
    human: SpeakerSegment,
    aligned: Sequence[Interval | SpeakerSegment | tuple[int, int]],
    *,
    alignment_failed: bool = False,
    min_coverage: Fraction | str | float = COVERAGE_RATIO,
    min_gap_ns: int = MIN_GAP_NS,
) -> TightenSegmentResult:
    """Keep aligned speech that lies inside the human segment.

    Coverage below min_coverage, or an alignment failure, drops all active
    speech and marks the segment interior UNCERTAIN. Short-gap merge runs
    on the kept pieces of this segment only.
    """

    human_duration_ns = human.duration_ns
    inside = intersect_spans([human.bounds], aligned)
    covered_ns = duration_ns(inside)
    coverage = (covered_ns / human_duration_ns) if human_duration_ns > 0 else 0.0
    fallback = bool(alignment_failed) or not coverage_meets(covered_ns, human_duration_ns, min_coverage)
    if fallback:
        uncertain = UncertainInterval(human.recording_id, human.speaker, human.start_ns, human.end_ns)
        return TightenSegmentResult((), (uncertain,), coverage, True)
    pieces = [SpeakerSegment(human.recording_id, human.speaker, start_ns, end_ns) for start_ns, end_ns in inside]
    active = merge_short_gaps(pieces, min_gap_ns=min_gap_ns, within=(human,))
    return TightenSegmentResult(active, (), coverage, False)


def tighten_recording(
    original: RecordingLabels,
    aligned_by_speaker: Mapping[str, Sequence[Interval | SpeakerSegment | tuple[int, int]]],
    *,
    failed_speakers: frozenset[str] = frozenset(),
    min_coverage: Fraction | str | float = COVERAGE_RATIO,
    min_gap_ns: int = MIN_GAP_NS,
) -> RecordingLabels:
    """Tighten every original human segment, then merge short same-speaker gaps.

    The final merge is speaker-wide. Gaps shorter than min_gap_ns become
    speech even when they sit between adjacent human segments. A gap of
    exactly min_gap_ns stays. UNCERTAIN time that the merge overlaps is
    dropped so it never covers the same speaker's active speech.
    """

    humans = union_human_segments(original.segments)
    active: list[SpeakerSegment] = []
    uncertain: list[UncertainInterval] = []
    for segment in humans:
        result = tighten_segment(
            segment,
            aligned_by_speaker.get(segment.speaker, ()),
            alignment_failed=segment.speaker in failed_speakers,
            min_coverage=min_coverage,
            min_gap_ns=min_gap_ns,
        )
        active.extend(result.active)
        uncertain.extend(result.uncertain)
    return merge_recording(
        RecordingLabels(original.recording_id, tuple(active), tuple(uncertain)),
        min_gap_ns=min_gap_ns,
    )
