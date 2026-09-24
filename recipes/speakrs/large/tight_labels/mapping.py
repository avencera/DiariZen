"""Optimal speaker assignment by overlapped speech time."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .classify import (
    AlignmentFailureReason,
    ByDesignReason,
    SegmentClass,
    SegmentOutcome,
    unioned_original_segments,
)
from .domain import RecordingLabels, SpeakerSegment, UncertainInterval
from .intervals import clip_recording_to_uem, clip_span, duration_ns, overlap_duration_ns, union_spans
from .merge import MIN_GAP_NS, merge_recording
from .time import ns_to_seconds


@dataclass(frozen=True, slots=True)
class SpeakerMatch:
    """One published speaker assigned to one original speaker.

    `published_spans` records the published support used to classify original
    segments after the speaker assignment
    """

    published_speaker: str
    original_speaker: str
    overlap_ns: int
    published_ns: int
    original_ns: int
    published_spans: tuple[tuple[int, int], ...] = ()

    @property
    def overlap_seconds(self) -> float:
        """Return overlap in seconds."""

        return ns_to_seconds(self.overlap_ns)

    @property
    def published_seconds(self) -> float:
        """Return published speech time in seconds."""

        return ns_to_seconds(self.published_ns)

    @property
    def original_seconds(self) -> float:
        """Return original speech time in seconds."""

        return ns_to_seconds(self.original_ns)

    @property
    def overlap_ratio(self) -> float:
        """Return overlap divided by published speech time."""

        if self.published_ns <= 0:
            return 0.0
        return self.overlap_ns / self.published_ns


@dataclass(frozen=True, slots=True)
class SpeakerMapping:
    """Assignment of published speakers onto original identities.

    `uem` is the time window applied before published support was assigned
    """

    matches: tuple[SpeakerMatch, ...]
    unmapped_published: tuple[str, ...]
    unmapped_original: tuple[str, ...]
    unmapped_published_ns: int = 0
    uem: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, str]:
        """Return published speaker to original speaker."""

        return {item.published_speaker: item.original_speaker for item in self.matches}

    @property
    def unmapped_published_seconds(self) -> float:
        """Return dropped published speech time in seconds."""

        return ns_to_seconds(self.unmapped_published_ns)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready record of the assignment."""

        return {
            "matches": [
                {
                    "published_speaker": item.published_speaker,
                    "original_speaker": item.original_speaker,
                    "overlap_seconds": item.overlap_seconds,
                    "published_seconds": item.published_seconds,
                    "original_seconds": item.original_seconds,
                    "overlap_ratio": item.overlap_ratio,
                }
                for item in self.matches
            ],
            "unmapped_published": list(self.unmapped_published),
            "unmapped_original": list(self.unmapped_original),
            "unmapped_published_seconds": self.unmapped_published_seconds,
        }


def _speaker_spans(segments: Sequence[SpeakerSegment]) -> dict[str, list[tuple[int, int]]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for segment in segments:
        grouped.setdefault(segment.speaker, []).append((segment.start_ns, segment.end_ns))
    return {speaker: union_spans(spans) for speaker, spans in grouped.items()}


def map_speakers_by_overlap(
    published: Sequence[SpeakerSegment] | RecordingLabels,
    original: Sequence[SpeakerSegment] | RecordingLabels,
) -> SpeakerMapping:
    """Assign each published speaker to at most one original speaker.

    The assignment maximises total overlapped speech time. Pairs with zero
    overlap are left unmapped.
    """

    published_segments = published.segments if isinstance(published, RecordingLabels) else published
    original_segments = original.segments if isinstance(original, RecordingLabels) else original
    published_spans = _speaker_spans(published_segments)
    original_spans = _speaker_spans(original_segments)
    published_names = tuple(sorted(published_spans))
    original_names = tuple(sorted(original_spans))
    if not published_names or not original_names:
        unmapped_ns = sum(duration_ns(published_spans[name]) for name in published_names)
        return SpeakerMapping((), published_names, original_names, unmapped_ns)

    overlap = np.zeros((len(published_names), len(original_names)), dtype=np.int64)
    for i, published_name in enumerate(published_names):
        for j, original_name in enumerate(original_names):
            overlap[i, j] = overlap_duration_ns(published_spans[published_name], original_spans[original_name])
    rows, cols = linear_sum_assignment(-overlap)
    used_published: set[str] = set()
    used_original: set[str] = set()
    matches: list[SpeakerMatch] = []
    for row, col in zip(rows, cols, strict=True):
        seconds_ns = int(overlap[row, col])
        if seconds_ns <= 0:
            continue
        published_name = published_names[row]
        original_name = original_names[col]
        used_published.add(published_name)
        used_original.add(original_name)
        matches.append(
            SpeakerMatch(
                published_speaker=published_name,
                original_speaker=original_name,
                overlap_ns=seconds_ns,
                published_ns=duration_ns(published_spans[published_name]),
                original_ns=duration_ns(original_spans[original_name]),
                published_spans=tuple(published_spans[published_name]),
            )
        )
    matches.sort(key=lambda item: item.original_speaker)
    unmapped_published = tuple(name for name in published_names if name not in used_published)
    unmapped_ns = sum(duration_ns(published_spans[name]) for name in unmapped_published)
    return SpeakerMapping(
        tuple(matches),
        unmapped_published,
        tuple(name for name in original_names if name not in used_original),
        unmapped_ns,
    )


def apply_published_replacement(
    original: RecordingLabels,
    published: RecordingLabels,
    *,
    min_gap_ns: int = MIN_GAP_NS,
    uem: tuple[int, int] | None = None,
) -> tuple[RecordingLabels, SpeakerMapping]:
    """Map published speakers onto original identities and merge short gaps.

    When uem is set, published rows are clipped to it before mapping and merge.
    Original speakers with no published match keep their UEM-clipped speech as
    UNCERTAIN so they are not dropped from the source record.
    """

    source = published
    if uem is not None:
        source = clip_recording_to_uem(published, uem[0], uem[1])
    mapping = replace(map_speakers_by_overlap(source, original), uem=uem)
    relabeled = source.with_recording_id(original.recording_id).map_speakers(mapping.as_dict())
    mapped_original = {item.original_speaker for item in mapping.matches}
    matches_by_original = {item.original_speaker: item for item in mapping.matches}
    extra_uncertain: list[UncertainInterval] = []
    for segment in unioned_original_segments((original,)):
        match = matches_by_original.get(segment.speaker)
        if match is None:
            continue
        span = _original_span_for_mapping(segment, mapping)
        if span is None or _published_covers(match, span):
            continue
        extra_uncertain.append(
            UncertainInterval(original.recording_id, segment.speaker, span[0], span[1])
        )

    for segment in original.segments:
        if segment.speaker in mapped_original:
            continue
        start_ns, end_ns = segment.start_ns, segment.end_ns
        if uem is not None:
            clipped = clip_span(start_ns, end_ns, uem[0], uem[1])
            if clipped is None:
                continue
            start_ns, end_ns = clipped
        extra_uncertain.append(UncertainInterval(original.recording_id, segment.speaker, start_ns, end_ns))
    combined = RecordingLabels(
        original.recording_id,
        relabeled.segments,
        tuple(relabeled.uncertain) + tuple(extra_uncertain),
        uncertain_removed_ns=relabeled.uncertain_removed_ns,
    )
    return merge_recording(combined, min_gap_ns=min_gap_ns), mapping


def classify_published_replacement(
    original: Sequence[RecordingLabels],
    mappings: Mapping[str, SpeakerMapping],
) -> tuple[SegmentClass, ...]:
    """Classify published coverage failures and speakers missing by design."""

    rows: list[SegmentClass] = []
    for segment in unioned_original_segments(original):
        mapping = mappings.get(segment.recording_id)
        match = None if mapping is None else next(
            (item for item in mapping.matches if item.original_speaker == segment.speaker),
            None,
        )
        span = None if mapping is None else _original_span_for_mapping(segment, mapping)
        if match is not None and (span is None or _published_covers(match, span)):
            rows.append(SegmentClass.from_segment(segment, SegmentOutcome.ALIGNED))
            continue
        if match is not None:
            rows.append(
                SegmentClass.from_segment(
                    segment,
                    SegmentOutcome.ALIGNMENT_FAILURE,
                    failure_reason=AlignmentFailureReason.NO_PUBLISHED_COVERAGE,
                )
            )
            continue
        rows.append(
            SegmentClass.from_segment(
                segment,
                SegmentOutcome.BY_DESIGN_UNCERTAIN,
                by_design_reason=ByDesignReason.SOURCE_RULE,
            )
        )
    return tuple(rows)


def _original_span_for_mapping(
    segment: SpeakerSegment,
    mapping: SpeakerMapping,
) -> tuple[int, int] | None:
    if mapping.uem is None:
        return segment.start_ns, segment.end_ns
    return clip_span(segment.start_ns, segment.end_ns, mapping.uem[0], mapping.uem[1])


def _published_covers(match: SpeakerMatch, span: tuple[int, int]) -> bool:
    if not match.published_spans:
        return True
    return overlap_duration_ns(match.published_spans, (span,)) > 0
