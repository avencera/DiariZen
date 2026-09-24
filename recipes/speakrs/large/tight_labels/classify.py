"""Typed outcomes for one original human segment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from collections import defaultdict

from .domain import RecordingLabels, SpeakerSegment, require_identity, require_ns_bounds
from .intervals import union_overlapping_spans


class SegmentOutcome(Enum):
    """How one original human segment was treated in the produced labels."""

    ALIGNED = "aligned"
    ALIGNMENT_FAILURE = "alignment_failure"
    BY_DESIGN_UNCERTAIN = "by_design_uncertain"


class AlignmentFailureReason(Enum):
    """Why an original segment was marked UNCERTAIN as an alignment failure."""

    MARKER_ONLY = "marker_only"
    EMPTY_TEXT = "empty_text"
    EMPTY_ROMANIZATION = "empty_romanization"
    NO_PUBLISHED_COVERAGE = "no_published_coverage"
    LOW_COVERAGE = "low_coverage"
    ALIGNER_ERROR = "aligner_error"


class ByDesignReason(Enum):
    """Why an original segment is UNCERTAIN by a source rule."""

    MULTI_SPEAKER = "multi_speaker"
    HELD_OUT_LANGUAGE = "held_out_language"
    SOURCE_RULE = "source_rule"


@dataclass(frozen=True, slots=True)
class SegmentClass:
    """Classification of one unioned original human segment."""

    recording_id: str
    speaker: str
    start_ns: int
    end_ns: int
    outcome: SegmentOutcome
    failure_reason: AlignmentFailureReason | None = None
    by_design_reason: ByDesignReason | None = None

    def __post_init__(self) -> None:
        require_identity(self.recording_id, "recording_id")
        require_identity(self.speaker, "speaker")
        require_ns_bounds(self.start_ns, self.end_ns)
        if not isinstance(self.outcome, SegmentOutcome):
            raise TypeError("outcome must be a SegmentOutcome")
        if self.outcome is SegmentOutcome.ALIGNMENT_FAILURE:
            if not isinstance(self.failure_reason, AlignmentFailureReason):
                raise ValueError("alignment failure requires an AlignmentFailureReason")
            if self.by_design_reason is not None:
                raise ValueError("alignment failure cannot carry a by-design reason")
            return
        if self.outcome is SegmentOutcome.BY_DESIGN_UNCERTAIN:
            if not isinstance(self.by_design_reason, ByDesignReason):
                raise ValueError("by-design UNCERTAIN requires a ByDesignReason")
            if self.failure_reason is not None:
                raise ValueError("by-design UNCERTAIN cannot carry a failure reason")
            return
        if self.failure_reason is not None or self.by_design_reason is not None:
            raise ValueError("aligned segments cannot carry a failure or by-design reason")

    @classmethod
    def from_segment(
        cls,
        segment: SpeakerSegment,
        outcome: SegmentOutcome,
        *,
        failure_reason: AlignmentFailureReason | None = None,
        by_design_reason: ByDesignReason | None = None,
    ) -> SegmentClass:
        """Build a classification from a unioned original human segment."""

        return cls(
            segment.recording_id,
            segment.speaker,
            segment.start_ns,
            segment.end_ns,
            outcome,
            failure_reason,
            by_design_reason,
        )

    @property
    def key(self) -> tuple[str, str, int, int]:
        """Return the unioned-segment identity."""

        return (self.recording_id, self.speaker, self.start_ns, self.end_ns)

    def to_json(self) -> dict[str, str | int | None]:
        """Return a JSON-ready classification record."""

        return {
            "recording_id": self.recording_id,
            "speaker": self.speaker,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "outcome": self.outcome.value,
            "failure_reason": None if self.failure_reason is None else self.failure_reason.value,
            "by_design_reason": None if self.by_design_reason is None else self.by_design_reason.value,
        }


def unioned_original_segments(original: Sequence[RecordingLabels]) -> tuple[SpeakerSegment, ...]:
    """Return original human segments with overlapping same-speaker rows collapsed.

    Touching same-speaker rows stay separate so adjacent single-speaker and
    by-design overlap intervals keep their own outcomes.
    """

    rows: list[SpeakerSegment] = []
    for labels in original:
        grouped: dict[str, list[SpeakerSegment]] = defaultdict(list)
        for segment in labels.segments:
            grouped[segment.speaker].append(segment)
        for speaker, group in grouped.items():
            for start_ns, end_ns in union_overlapping_spans(group):
                rows.append(SpeakerSegment(labels.recording_id, speaker, start_ns, end_ns))
    rows.sort(key=lambda item: (item.recording_id, item.start_ns, item.end_ns, item.speaker))
    return tuple(rows)


def aligned_classes(original: Sequence[RecordingLabels]) -> tuple[SegmentClass, ...]:
    """Mark every unioned original segment as aligned."""

    return tuple(
        SegmentClass.from_segment(segment, SegmentOutcome.ALIGNED)
        for segment in unioned_original_segments(original)
    )


def require_segment_classes(
    original: Sequence[RecordingLabels],
    classes: Sequence[SegmentClass],
) -> tuple[SegmentClass, ...]:
    """Return classes covering each unioned original segment exactly once."""

    expected = {
        (segment.recording_id, segment.speaker, segment.start_ns, segment.end_ns)
        for segment in unioned_original_segments(original)
    }
    seen: dict[tuple[str, str, int, int], SegmentClass] = {}
    for item in classes:
        if not isinstance(item, SegmentClass):
            raise TypeError("segment classification must be SegmentClass")
        if item.key in seen:
            raise ValueError(
                f"duplicate classification for {item.recording_id} {item.speaker} "
                f"{item.start_ns}-{item.end_ns}"
            )
        seen[item.key] = item
    missing = expected - set(seen)
    extra = set(seen) - expected
    if missing or extra:
        raise ValueError(
            "segment classification must cover each unioned original segment once "
            f"(missing={len(missing)} extra={len(extra)})"
        )
    ordered = unioned_original_segments(original)
    return tuple(seen[(item.recording_id, item.speaker, item.start_ns, item.end_ns)] for item in ordered)


__all__ = [
    "AlignmentFailureReason",
    "ByDesignReason",
    "SegmentClass",
    "SegmentOutcome",
    "aligned_classes",
    "require_segment_classes",
    "unioned_original_segments",
]
