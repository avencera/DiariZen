"""Typed speaker-activity intervals for tight label production."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


def require_finite_bounds(start: float, end: float, *, allow_zero_start: bool = True) -> None:
    """Reject non-finite or reversed interval bounds."""

    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError("interval bounds must be real numbers")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise ValueError("interval bounds must be real numbers")
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("interval bounds must be finite")
    if allow_zero_start and start < 0:
        raise ValueError("interval start must be non-negative")
    if end <= start:
        raise ValueError("interval end must be greater than start")


def require_identity(value: str, label: str) -> str:
    """Return a non-empty identity string."""

    if not isinstance(value, str) or not value or value in {"<NA>", "NA"}:
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class Interval:
    """A half-open time span in seconds."""

    start: float
    end: float

    def __post_init__(self) -> None:
        require_finite_bounds(self.start, self.end)

    @property
    def duration(self) -> float:
        """Return end minus start."""

        return self.end - self.start

    def intersection(self, other: Interval) -> Interval | None:
        """Return the overlapping span, if it has positive duration."""

        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if end <= start:
            return None
        return Interval(start, end)


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """One active-speech interval for a speaker on a recording."""

    recording_id: str
    speaker: str
    start: float
    end: float

    def __post_init__(self) -> None:
        require_identity(self.recording_id, "recording_id")
        require_identity(self.speaker, "speaker")
        require_finite_bounds(self.start, self.end)

    @property
    def duration(self) -> float:
        """Return end minus start."""

        return self.end - self.start

    @property
    def bounds(self) -> Interval:
        """Return the unlabelled time span."""

        return Interval(self.start, self.end)

    def with_speaker(self, speaker: str) -> SpeakerSegment:
        """Return a copy with a replacement speaker identity."""

        return SpeakerSegment(self.recording_id, speaker, self.start, self.end)

    def with_recording_id(self, recording_id: str) -> SpeakerSegment:
        """Return a copy with a replacement recording identity."""

        return SpeakerSegment(recording_id, self.speaker, self.start, self.end)


@dataclass(frozen=True, slots=True)
class UncertainInterval:
    """A region the machine cannot place as speech or silence."""

    recording_id: str
    speaker: str
    start: float
    end: float

    def __post_init__(self) -> None:
        require_identity(self.recording_id, "recording_id")
        require_identity(self.speaker, "speaker")
        require_finite_bounds(self.start, self.end)

    @property
    def duration(self) -> float:
        """Return end minus start."""

        return self.end - self.start

    @property
    def bounds(self) -> Interval:
        """Return the unlabelled time span."""

        return Interval(self.start, self.end)


def _overlap_seconds(left: Interval, right: Interval) -> float:
    start = max(left.start, right.start)
    end = min(left.end, right.end)
    return max(0.0, end - start)


@dataclass(frozen=True, slots=True)
class RecordingLabels:
    """Active speech and uncertain regions for one recording."""

    recording_id: str
    segments: tuple[SpeakerSegment, ...]
    uncertain: tuple[UncertainInterval, ...] = ()

    def __post_init__(self) -> None:
        require_identity(self.recording_id, "recording_id")
        segments = tuple(self.segments)
        uncertain = tuple(self.uncertain)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "uncertain", uncertain)
        for segment in segments:
            if not isinstance(segment, SpeakerSegment):
                raise ValueError("segments must contain SpeakerSegment values")
            if segment.recording_id != self.recording_id:
                raise ValueError("segment recording_id must match the recording")
        for region in uncertain:
            if not isinstance(region, UncertainInterval):
                raise ValueError("uncertain must contain UncertainInterval values")
            if region.recording_id != self.recording_id:
                raise ValueError("uncertain recording_id must match the recording")
        self._reject_same_speaker_overlap(segments, uncertain)

    @staticmethod
    def _reject_same_speaker_overlap(
        segments: tuple[SpeakerSegment, ...],
        uncertain: tuple[UncertainInterval, ...],
    ) -> None:
        by_speaker: dict[str, list[Interval]] = {}
        for segment in segments:
            by_speaker.setdefault(segment.speaker, []).append(segment.bounds)
        for region in uncertain:
            active = by_speaker.get(region.speaker, ())
            for interval in active:
                if _overlap_seconds(region.bounds, interval) > 1e-12:
                    raise ValueError("uncertain regions must not overlap active speech of the same speaker")

    @property
    def speakers(self) -> frozenset[str]:
        """Return every speaker with active or uncertain time."""

        names = {item.speaker for item in self.segments}
        names.update(item.speaker for item in self.uncertain)
        return frozenset(names)

    def segments_for(self, speaker: str) -> tuple[SpeakerSegment, ...]:
        """Return active segments for one speaker, in time order."""

        return tuple(item for item in self.segments if item.speaker == speaker)

    def uncertain_for(self, speaker: str) -> tuple[UncertainInterval, ...]:
        """Return uncertain intervals for one speaker, in time order."""

        return tuple(item for item in self.uncertain if item.speaker == speaker)

    def with_recording_id(self, recording_id: str) -> RecordingLabels:
        """Return a copy on a replacement recording identity."""

        return RecordingLabels(
            recording_id,
            tuple(item.with_recording_id(recording_id) for item in self.segments),
            tuple(UncertainInterval(recording_id, item.speaker, item.start, item.end) for item in self.uncertain),
        )

    def map_speakers(self, speaker_map: Mapping[str, str]) -> RecordingLabels:
        """Rewrite speakers present in the map and drop the rest."""

        segments = tuple(
            item.with_speaker(speaker_map[item.speaker]) for item in self.segments if item.speaker in speaker_map
        )
        uncertain = tuple(
            UncertainInterval(
                self.recording_id,
                speaker_map[item.speaker],
                item.start,
                item.end,
            )
            for item in self.uncertain
            if item.speaker in speaker_map
        )
        return RecordingLabels(self.recording_id, segments, uncertain)


def segments_from_pairs(
    recording_id: str,
    speaker: str,
    spans: Iterable[tuple[float, float]],
) -> tuple[SpeakerSegment, ...]:
    """Build validated segments from raw time pairs."""

    return tuple(SpeakerSegment(recording_id, speaker, start, end) for start, end in spans)
