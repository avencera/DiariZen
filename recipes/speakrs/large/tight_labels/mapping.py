"""Optimal speaker assignment by overlapped speech time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .domain import RecordingLabels, SpeakerSegment
from .intervals import clip_recording_to_uem, duration_ns, overlap_duration_ns, union_spans
from .merge import MIN_GAP_NS, merge_recording
from .time import ns_to_seconds


@dataclass(frozen=True, slots=True)
class SpeakerMatch:
    """One published speaker assigned to one original speaker."""

    published_speaker: str
    original_speaker: str
    overlap_ns: int
    published_ns: int
    original_ns: int

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
    """Assignment of published speakers onto original identities."""

    matches: tuple[SpeakerMatch, ...]
    unmapped_published: tuple[str, ...]
    unmapped_original: tuple[str, ...]
    unmapped_published_ns: int = 0

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
    """

    source = published
    if uem is not None:
        source = clip_recording_to_uem(published, uem[0], uem[1])
    mapping = map_speakers_by_overlap(source, original)
    relabeled = source.with_recording_id(original.recording_id).map_speakers(mapping.as_dict())
    return merge_recording(relabeled, min_gap_ns=min_gap_ns), mapping
