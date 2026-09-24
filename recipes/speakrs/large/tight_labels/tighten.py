"""Tighten human segments with aligned word or phone intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .domain import Interval, RecordingLabels, SpeakerSegment, UncertainInterval
from .intervals import duration, intersect_spans
from .merge import MIN_GAP_SECONDS, merge_short_gaps


COVERAGE_RATIO = 0.30


@dataclass(frozen=True, slots=True)
class TightenSegmentResult:
    """Active speech and optional uncertain interior for one human segment."""

    active: tuple[SpeakerSegment, ...]
    uncertain: tuple[UncertainInterval, ...]
    coverage: float
    fallback: bool


def tighten_segment(
    human: SpeakerSegment,
    aligned: Sequence[Interval | SpeakerSegment | tuple[float, float]],
    *,
    alignment_failed: bool = False,
    min_coverage: float = COVERAGE_RATIO,
    min_gap: float = MIN_GAP_SECONDS,
) -> TightenSegmentResult:
    """Keep aligned speech that lies inside the human segment.

    Coverage below min_coverage, or an alignment failure, drops all active
    speech and marks the segment interior UNCERTAIN. Short-gap merge runs
    on the kept pieces of this segment only.
    """

    if min_coverage < 0 or min_coverage > 1:
        raise ValueError("min_coverage must be in [0, 1]")
    human_duration = human.duration
    inside = intersect_spans([human.bounds], aligned)
    covered = duration(inside)
    coverage = covered / human_duration if human_duration > 0 else 0.0
    fallback = bool(alignment_failed) or coverage < min_coverage
    if fallback:
        uncertain = UncertainInterval(human.recording_id, human.speaker, human.start, human.end)
        return TightenSegmentResult((), (uncertain,), coverage, True)
    pieces = [SpeakerSegment(human.recording_id, human.speaker, start, end) for start, end in inside]
    active = merge_short_gaps(pieces, min_gap=min_gap, within=(human,))
    return TightenSegmentResult(active, (), coverage, False)


def tighten_recording(
    original: RecordingLabels,
    aligned_by_speaker: Mapping[str, Sequence[Interval | SpeakerSegment | tuple[float, float]]],
    *,
    failed_speakers: frozenset[str] = frozenset(),
    min_coverage: float = COVERAGE_RATIO,
    min_gap: float = MIN_GAP_SECONDS,
) -> RecordingLabels:
    """Tighten every original human segment, then merge short gaps inside them."""

    active: list[SpeakerSegment] = []
    uncertain: list[UncertainInterval] = []
    for segment in original.segments:
        result = tighten_segment(
            segment,
            aligned_by_speaker.get(segment.speaker, ()),
            alignment_failed=segment.speaker in failed_speakers,
            min_coverage=min_coverage,
            min_gap=min_gap,
        )
        active.extend(result.active)
        uncertain.extend(result.uncertain)
    merged = merge_short_gaps(active, min_gap=min_gap, within=original.segments)
    return RecordingLabels(original.recording_id, merged, tuple(uncertain))
