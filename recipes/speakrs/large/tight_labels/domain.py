"""Typed speaker-activity intervals for tight label production."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .time import SecondsLike, ns_to_seconds, parse_seconds_to_ns


def require_ns_bounds(start_ns: int, end_ns: int) -> None:
    """Reject non-integer or reversed nanosecond bounds."""

    if isinstance(start_ns, bool) or isinstance(end_ns, bool):
        raise ValueError("interval bounds must be integers")
    if not isinstance(start_ns, int) or not isinstance(end_ns, int):
        raise ValueError("interval bounds must be integers")
    if start_ns < 0:
        raise ValueError("interval start must be non-negative")
    if end_ns <= start_ns:
        raise ValueError("interval end must be greater than start")


def require_identity(value: str, label: str) -> str:
    """Return a non-empty identity string."""

    if not isinstance(value, str) or not value or value in {"<NA>", "NA"}:
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class Interval:
    """A half-open time span in nanoseconds."""

    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        require_ns_bounds(self.start_ns, self.end_ns)

    @classmethod
    def from_seconds(cls, start: SecondsLike, end: SecondsLike) -> Interval:
        """Build an interval from second counts."""

        return cls(parse_seconds_to_ns(start), parse_seconds_to_ns(end))

    @property
    def start(self) -> float:
        """Return start in seconds."""

        return ns_to_seconds(self.start_ns)

    @property
    def end(self) -> float:
        """Return end in seconds."""

        return ns_to_seconds(self.end_ns)

    @property
    def duration_ns(self) -> int:
        """Return end minus start in nanoseconds."""

        return self.end_ns - self.start_ns

    @property
    def duration(self) -> float:
        """Return end minus start in seconds."""

        return ns_to_seconds(self.duration_ns)

    def intersection(self, other: Interval) -> Interval | None:
        """Return the overlapping span, if it has positive duration."""

        start_ns = max(self.start_ns, other.start_ns)
        end_ns = min(self.end_ns, other.end_ns)
        if end_ns <= start_ns:
            return None
        return Interval(start_ns, end_ns)


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """One active-speech interval for a speaker on a recording."""

    recording_id: str
    speaker: str
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        require_identity(self.recording_id, "recording_id")
        require_identity(self.speaker, "speaker")
        require_ns_bounds(self.start_ns, self.end_ns)

    @classmethod
    def from_seconds(
        cls,
        recording_id: str,
        speaker: str,
        start: SecondsLike,
        end: SecondsLike,
    ) -> SpeakerSegment:
        """Build a segment from second counts."""

        return cls(recording_id, speaker, parse_seconds_to_ns(start), parse_seconds_to_ns(end))

    @property
    def start(self) -> float:
        """Return start in seconds."""

        return ns_to_seconds(self.start_ns)

    @property
    def end(self) -> float:
        """Return end in seconds."""

        return ns_to_seconds(self.end_ns)

    @property
    def duration_ns(self) -> int:
        """Return end minus start in nanoseconds."""

        return self.end_ns - self.start_ns

    @property
    def duration(self) -> float:
        """Return end minus start in seconds."""

        return ns_to_seconds(self.duration_ns)

    @property
    def bounds(self) -> Interval:
        """Return the unlabelled time span."""

        return Interval(self.start_ns, self.end_ns)

    def with_speaker(self, speaker: str) -> SpeakerSegment:
        """Return a copy with a replacement speaker identity."""

        return SpeakerSegment(self.recording_id, speaker, self.start_ns, self.end_ns)

    def with_recording_id(self, recording_id: str) -> SpeakerSegment:
        """Return a copy with a replacement recording identity."""

        return SpeakerSegment(recording_id, self.speaker, self.start_ns, self.end_ns)


@dataclass(frozen=True, slots=True)
class UncertainInterval:
    """A region the machine cannot place as speech or silence."""

    recording_id: str
    speaker: str
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        require_identity(self.recording_id, "recording_id")
        require_identity(self.speaker, "speaker")
        require_ns_bounds(self.start_ns, self.end_ns)

    @classmethod
    def from_seconds(
        cls,
        recording_id: str,
        speaker: str,
        start: SecondsLike,
        end: SecondsLike,
    ) -> UncertainInterval:
        """Build an uncertain interval from second counts."""

        return cls(recording_id, speaker, parse_seconds_to_ns(start), parse_seconds_to_ns(end))

    @property
    def start(self) -> float:
        """Return start in seconds."""

        return ns_to_seconds(self.start_ns)

    @property
    def end(self) -> float:
        """Return end in seconds."""

        return ns_to_seconds(self.end_ns)

    @property
    def duration_ns(self) -> int:
        """Return end minus start in nanoseconds."""

        return self.end_ns - self.start_ns

    @property
    def duration(self) -> float:
        """Return end minus start in seconds."""

        return ns_to_seconds(self.duration_ns)

    @property
    def bounds(self) -> Interval:
        """Return the unlabelled time span."""

        return Interval(self.start_ns, self.end_ns)


def _overlap_ns(left: Interval, right: Interval) -> int:
    start_ns = max(left.start_ns, right.start_ns)
    end_ns = min(left.end_ns, right.end_ns)
    return max(0, end_ns - start_ns)


@dataclass(frozen=True, slots=True)
class RecordingLabels:
    """Active speech and uncertain regions for one recording."""

    recording_id: str
    segments: tuple[SpeakerSegment, ...]
    uncertain: tuple[UncertainInterval, ...] = ()
    uncertain_removed_ns: int = 0

    def __post_init__(self) -> None:
        require_identity(self.recording_id, "recording_id")
        if (
            isinstance(self.uncertain_removed_ns, bool)
            or not isinstance(self.uncertain_removed_ns, int)
            or self.uncertain_removed_ns < 0
        ):
            raise ValueError("uncertain_removed_ns must be a non-negative integer")
        segments = tuple(self.segments)
        uncertain = tuple(self.uncertain)
        object.__setattr__(self, "segments", segments)
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
        from .merge import canonicalize_uncertain

        canonical, removed_ns = canonicalize_uncertain(uncertain, segments)
        object.__setattr__(self, "uncertain", canonical)
        object.__setattr__(self, "uncertain_removed_ns", self.uncertain_removed_ns + removed_ns)
        self._reject_same_speaker_overlap(segments, canonical)

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
                if _overlap_ns(region.bounds, interval) > 0:
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
            tuple(
                UncertainInterval(recording_id, item.speaker, item.start_ns, item.end_ns) for item in self.uncertain
            ),
            uncertain_removed_ns=self.uncertain_removed_ns,
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
                item.start_ns,
                item.end_ns,
            )
            for item in self.uncertain
            if item.speaker in speaker_map
        )
        return RecordingLabels(
            self.recording_id,
            segments,
            uncertain,
            uncertain_removed_ns=self.uncertain_removed_ns,
        )


def segments_from_pairs(
    recording_id: str,
    speaker: str,
    spans: Iterable[tuple[int, int]],
) -> tuple[SpeakerSegment, ...]:
    """Build validated segments from nanosecond start/end pairs."""

    return tuple(SpeakerSegment(recording_id, speaker, start_ns, end_ns) for start_ns, end_ns in spans)
