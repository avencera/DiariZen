"""Subset checks of tight speech against original human segments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .domain import RecordingLabels, SpeakerSegment, UncertainInterval
from .intervals import duration_ns, intersect_spans, merge_spans, span_is_subset, union_spans
from .merge import merge_recording
from .time import MIN_GAP_NS, ns_to_seconds


def speaker_union(segments: Sequence[SpeakerSegment], speaker: str) -> list[tuple[int, int]]:
    """Return the merged original union for one speaker."""

    return union_spans(item for item in segments if item.speaker == speaker)


def closed_speaker_union(
    segments: Sequence[SpeakerSegment],
    speaker: str,
    *,
    min_gap_ns: int = MIN_GAP_NS,
) -> list[tuple[int, int]]:
    """Return the original union after the same-speaker short-gap merge."""

    return merge_spans(
        (item for item in segments if item.speaker == speaker),
        max_gap_ns=min_gap_ns,
    )


def is_subset_of_original(
    tight: Sequence[SpeakerSegment],
    original: Sequence[SpeakerSegment],
    *,
    speaker: str | None = None,
    tolerance_ns: int = 0,
    min_gap_ns: int = MIN_GAP_NS,
) -> bool:
    """Return True when every tight interval lies inside the 200 ms-closed original.

    Original same-speaker gaps shorter than min_gap_ns are speech for this
    check, so the machine may bridge those gaps and must not add other speech.
    """

    grouped: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in tight:
        if speaker is None or segment.speaker == speaker:
            grouped[segment.speaker].append(segment)
    originals: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in original:
        if speaker is None or segment.speaker == speaker:
            originals[segment.speaker].append(segment)
    for name, pieces in grouped.items():
        outer = merge_spans(originals.get(name, ()), max_gap_ns=min_gap_ns)
        for piece in pieces:
            if not span_is_subset((piece.start_ns, piece.end_ns), outer, tolerance_ns=tolerance_ns):
                return False
    return True


def recording_is_subset(
    tight: RecordingLabels,
    original: RecordingLabels,
    *,
    tolerance_ns: int = 0,
    min_gap_ns: int = MIN_GAP_NS,
) -> bool:
    """Return True when every speaker of tight is a subset of original."""

    if tight.recording_id != original.recording_id:
        raise ValueError("subset check requires the same recording_id")
    return is_subset_of_original(
        tight.segments,
        original.segments,
        tolerance_ns=tolerance_ns,
        min_gap_ns=min_gap_ns,
    )


def uncertain_is_subset_of_original(
    tight: RecordingLabels,
    original: RecordingLabels,
    *,
    tolerance_ns: int = 0,
) -> bool:
    """Return True when every UNCERTAIN interval lies inside original support.

    Bridges shorter than 200 ms are not original support for UNCERTAIN.
    """

    if tight.recording_id != original.recording_id:
        raise ValueError("subset check requires the same recording_id")
    originals: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in original.segments:
        originals[segment.speaker].append(segment)
    for region in tight.uncertain:
        outer = union_spans(originals.get(region.speaker, ()))
        if not span_is_subset((region.start_ns, region.end_ns), outer, tolerance_ns=tolerance_ns):
            return False
    return True


@dataclass(frozen=True, slots=True)
class ClipReport:
    """Speech and UNCERTAIN time removed because it left original support."""

    recording_id: str
    clipped_speech_ns: int
    clipped_uncertain_ns: int

    @property
    def clipped_speech_seconds(self) -> float:
        """Return clipped SPEAKER time in seconds."""

        return ns_to_seconds(self.clipped_speech_ns)

    @property
    def clipped_uncertain_seconds(self) -> float:
        """Return clipped UNCERTAIN time in seconds."""

        return ns_to_seconds(self.clipped_uncertain_ns)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready clip record."""

        return {
            "recording_id": self.recording_id,
            "clipped_speech_seconds": self.clipped_speech_seconds,
            "clipped_uncertain_seconds": self.clipped_uncertain_seconds,
        }


def clip_to_original_support(
    produced: RecordingLabels,
    original: RecordingLabels,
    *,
    min_gap_ns: int = MIN_GAP_NS,
) -> tuple[RecordingLabels, ClipReport]:
    """Clip SPEAKER to the 200 ms-closed original and UNCERTAIN to raw support.

    SPEAKER may keep permitted bridges shorter than min_gap_ns. UNCERTAIN may
    not. Time that falls outside those unions is dropped and counted.
    """

    if produced.recording_id != original.recording_id:
        raise ValueError("clip requires the same recording_id")
    segments: list[SpeakerSegment] = []
    clipped_speech_ns = 0
    speakers = {item.speaker for item in produced.segments}
    for speaker in speakers:
        allowed = closed_speaker_union(original.segments, speaker, min_gap_ns=min_gap_ns)
        before = produced.segments_for(speaker)
        kept = intersect_spans(before, allowed)
        clipped_speech_ns += duration_ns(before) - duration_ns(kept)
        segments.extend(
            SpeakerSegment(produced.recording_id, speaker, start_ns, end_ns) for start_ns, end_ns in kept
        )
    uncertain: list[UncertainInterval] = []
    clipped_uncertain_ns = 0
    uncertain_speakers = {item.speaker for item in produced.uncertain}
    for speaker in uncertain_speakers:
        allowed = speaker_union(original.segments, speaker)
        before = produced.uncertain_for(speaker)
        kept = intersect_spans(before, allowed)
        clipped_uncertain_ns += duration_ns(before) - duration_ns(kept)
        uncertain.extend(
            UncertainInterval(produced.recording_id, speaker, start_ns, end_ns) for start_ns, end_ns in kept
        )
    clipped = merge_recording(
        RecordingLabels(
            produced.recording_id,
            tuple(segments),
            tuple(uncertain),
            uncertain_removed_ns=produced.uncertain_removed_ns,
        ),
        min_gap_ns=min_gap_ns,
    )
    return clipped, ClipReport(produced.recording_id, clipped_speech_ns, clipped_uncertain_ns)
