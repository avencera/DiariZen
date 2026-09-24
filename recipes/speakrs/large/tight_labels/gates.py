"""Automatic quality gates for a tight-label source."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .domain import RecordingLabels, SpeakerSegment, UncertainInterval
from .intervals import duration_ns, overlap_duration_ns, overlap_timeline_ns, span_outside_uem, union_spans
from .mapping import SpeakerMapping
from .merge import MIN_GAP_NS
from .subset import recording_is_subset
from .tightness import (
    FORCED_AMI_GAPS_0_3_TO_1_PER_MIN,
    FORCED_AMI_GAPS_LE_0_3_PER_MIN,
    FORCED_AMI_SMALLEST_GAP_SECONDS,
    TightnessStats,
    measure_tightness,
)
from .time import ns_to_seconds


SPEAKER_RETENTION_FLOOR = 0.30
OVERLAP_RETENTION_FLOOR = 0.20
FAILURE_SHARE_LIMIT = 0.10
PUBLISHED_OUTSIDE_FLAG = 0.05
TIGHTNESS_SHORT_GAP_MIN = 2.0
TIGHTNESS_SHORT_GAP_MAX = 12.0
MIN_SPEAKER_NS_FOR_RETENTION = 1_000_000_000


@dataclass(frozen=True, slots=True)
class SpeakerRetention:
    """Share of one speaker's original speech kept in the new labels."""

    recording_id: str
    speaker: str
    original_ns: int
    kept_ns: int

    @property
    def original_seconds(self) -> float:
        """Return original speech time in seconds."""

        return ns_to_seconds(self.original_ns)

    @property
    def kept_seconds(self) -> float:
        """Return kept original speech time in seconds."""

        return ns_to_seconds(self.kept_ns)

    @property
    def ratio(self) -> float:
        """Return kept original speech divided by original speech."""

        if self.original_ns <= 0:
            return 1.0
        return self.kept_ns / self.original_ns


@dataclass(frozen=True, slots=True)
class RecordingTimeBase:
    """Speech-union overlap between original and new labels."""

    recording_id: str
    original_ns: int
    new_ns: int
    overlap_ns: int
    new_outside_share: float

    @property
    def original_seconds(self) -> float:
        """Return original union speech in seconds."""

        return ns_to_seconds(self.original_ns)

    @property
    def new_seconds(self) -> float:
        """Return new union speech in seconds."""

        return ns_to_seconds(self.new_ns)

    @property
    def overlap_seconds(self) -> float:
        """Return overlapping speech in seconds."""

        return ns_to_seconds(self.overlap_ns)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Pass/fail plus the measurements a report needs."""

    passed: bool
    tightness_ok: bool
    speaker_retention_ok: bool
    overlap_retention_ok: bool
    failure_share_ok: bool
    smallest_gap_ok: bool
    subset_ok: bool | None
    uem_ok: bool
    unmapped_speakers_ok: bool
    tightness: TightnessStats
    original_tightness: TightnessStats
    speaker_retention: tuple[SpeakerRetention, ...]
    overlap_retention: float | None
    failure_share: float
    time_bases: tuple[RecordingTimeBase, ...]
    flagged_speakers: tuple[SpeakerRetention, ...]
    flagged_recordings: tuple[str, ...]
    rows_outside_uem: tuple[str, ...]
    dropped_unmapped_seconds: float
    unmapped_published: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready gate record."""

        ratios = [item.ratio for item in self.speaker_retention if item.original_ns > 0]
        return {
            "passed": self.passed,
            "tightness_ok": self.tightness_ok,
            "speaker_retention_ok": self.speaker_retention_ok,
            "overlap_retention_ok": self.overlap_retention_ok,
            "failure_share_ok": self.failure_share_ok,
            "smallest_gap_ok": self.smallest_gap_ok,
            "subset_ok": self.subset_ok,
            "uem_ok": self.uem_ok,
            "unmapped_speakers_ok": self.unmapped_speakers_ok,
            "tightness": self.tightness.to_json(),
            "original_tightness": self.original_tightness.to_json(),
            "reference_forced_ami": {
                "gaps_le_0_3_per_min": FORCED_AMI_GAPS_LE_0_3_PER_MIN,
                "gaps_0_3_to_1_per_min": FORCED_AMI_GAPS_0_3_TO_1_PER_MIN,
                "smallest_gap_s": FORCED_AMI_SMALLEST_GAP_SECONDS,
            },
            "speaker_retention": {
                "count": len(ratios),
                "min": min(ratios) if ratios else None,
                "p10": float(np.percentile(ratios, 10)) if ratios else None,
                "median": float(np.percentile(ratios, 50)) if ratios else None,
                "p90": float(np.percentile(ratios, 90)) if ratios else None,
                "max": max(ratios) if ratios else None,
                "flagged_below_30pct": len(self.flagged_speakers),
            },
            "overlap_retention": self.overlap_retention,
            "failure_share": self.failure_share,
            "flagged_speakers": [
                {
                    "recording_id": item.recording_id,
                    "speaker": item.speaker,
                    "original_seconds": item.original_seconds,
                    "kept_seconds": item.kept_seconds,
                    "ratio": item.ratio,
                }
                for item in self.flagged_speakers
            ],
            "flagged_recordings": list(self.flagged_recordings),
            "rows_outside_uem": list(self.rows_outside_uem),
            "dropped_unmapped_seconds": self.dropped_unmapped_seconds,
            "unmapped_published": list(self.unmapped_published),
            "time_bases": [
                {
                    "recording_id": item.recording_id,
                    "original_seconds": item.original_seconds,
                    "new_seconds": item.new_seconds,
                    "overlap_seconds": item.overlap_seconds,
                    "new_outside_share": item.new_outside_share,
                }
                for item in self.time_bases
            ],
            "notes": list(self.notes),
        }


def _by_recording(items: Sequence[RecordingLabels]) -> dict[str, RecordingLabels]:
    return {item.recording_id: item for item in items}


def _speaker_retention(
    original: RecordingLabels,
    new: RecordingLabels,
) -> tuple[SpeakerRetention, ...]:
    original_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    new_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for segment in original.segments:
        original_spans[segment.speaker].append((segment.start_ns, segment.end_ns))
    for segment in new.segments:
        new_spans[segment.speaker].append((segment.start_ns, segment.end_ns))
    speakers = sorted(set(original_spans) | set(new_spans))
    rows: list[SpeakerRetention] = []
    for speaker in speakers:
        original_union = union_spans(original_spans.get(speaker, ()))
        new_union = union_spans(new_spans.get(speaker, ()))
        original_ns = duration_ns(original_union)
        kept = overlap_duration_ns(original_union, new_union)
        rows.append(SpeakerRetention(original.recording_id, speaker, original_ns, kept))
    return tuple(rows)


def _time_base(original: RecordingLabels, new: RecordingLabels) -> RecordingTimeBase:
    original_union = union_spans(original.segments)
    new_union = union_spans(new.segments)
    original_ns = duration_ns(original_union)
    new_ns = duration_ns(new_union)
    overlap = overlap_duration_ns(original_union, new_union)
    outside = 0.0 if new_ns <= 0 else max(0, new_ns - overlap) / new_ns
    return RecordingTimeBase(
        original.recording_id,
        original_ns,
        new_ns,
        overlap,
        outside,
    )


def _overlap_ns(labels: RecordingLabels) -> int:
    grouped: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for segment in labels.segments:
        grouped[segment.speaker].append(segment)
    return overlap_timeline_ns(grouped)


def _failure_share(original: Sequence[RecordingLabels], new: Sequence[RecordingLabels]) -> float:
    original_segments = sum(len(item.segments) for item in original)
    if original_segments <= 0:
        return 0.0
    failed = sum(len(item.uncertain) for item in new)
    return failed / original_segments


def _rows_outside_uem(
    labels: RecordingLabels,
    uem_start_ns: int,
    uem_end_ns: int,
) -> list[str]:
    rows: list[str] = []
    items: Sequence[SpeakerSegment | UncertainInterval] = (*labels.segments, *labels.uncertain)
    for item in items:
        if span_outside_uem(item.start_ns, item.end_ns, uem_start_ns, uem_end_ns):
            kind = "SPEAKER" if isinstance(item, SpeakerSegment) else "UNCERTAIN"
            rows.append(f"{labels.recording_id} {kind} {item.speaker} {item.start_ns}-{item.end_ns}")
    return rows


def evaluate_gates(
    original: Sequence[RecordingLabels],
    new: Sequence[RecordingLabels],
    *,
    require_subset: bool = False,
    min_gap_ns: int = MIN_GAP_NS,
    speaker_retention_floor: float = SPEAKER_RETENTION_FLOOR,
    overlap_retention_floor: float = OVERLAP_RETENTION_FLOOR,
    failure_share_limit: float = FAILURE_SHARE_LIMIT,
    published_outside_flag: float = PUBLISHED_OUTSIDE_FLAG,
    tightness_short_gap_min: float = TIGHTNESS_SHORT_GAP_MIN,
    tightness_short_gap_max: float = TIGHTNESS_SHORT_GAP_MAX,
    uem_by_recording: Mapping[str, tuple[int, int]] | None = None,
    mappings: Mapping[str, SpeakerMapping] | None = None,
) -> GateResult:
    """Score tightness, retention, subset, UEM, mapping, and failure share."""

    original_by_id = _by_recording(original)
    new_by_id = _by_recording(new)
    if set(original_by_id) != set(new_by_id):
        raise ValueError("gate evaluation requires the same recording set")

    tightness = measure_tightness(new)
    original_tightness = measure_tightness(original)
    retentions: list[SpeakerRetention] = []
    time_bases: list[RecordingTimeBase] = []
    original_overlap = 0
    new_overlap = 0
    subset_ok: bool | None = True if require_subset else None
    outside_rows: list[str] = []
    for recording_id in sorted(original_by_id):
        source = original_by_id[recording_id]
        produced = new_by_id[recording_id]
        retentions.extend(_speaker_retention(source, produced))
        time_bases.append(_time_base(source, produced))
        original_overlap += _overlap_ns(source)
        new_overlap += _overlap_ns(produced)
        if require_subset and not recording_is_subset(produced, source):
            subset_ok = False
        if uem_by_recording is not None:
            uem_start_ns, uem_end_ns = uem_by_recording[recording_id]
            outside_rows.extend(_rows_outside_uem(produced, uem_start_ns, uem_end_ns))

    flagged_speakers = tuple(
        item
        for item in retentions
        if item.original_ns >= MIN_SPEAKER_NS_FOR_RETENTION and item.ratio < speaker_retention_floor
    )
    flagged_recordings = tuple(
        item.recording_id for item in time_bases if item.new_outside_share > published_outside_flag
    )
    overlap_retention = None if original_overlap <= 0 else new_overlap / original_overlap
    failure_share = _failure_share(original, new)

    smallest_ns = tightness.smallest_gap_ns
    smallest_gap_ok = smallest_ns is None or smallest_ns >= min_gap_ns
    tightness_band_ok = tightness_short_gap_min <= tightness.gaps_le_0_3_per_min <= tightness_short_gap_max
    tightness_ok = smallest_gap_ok and tightness_band_ok
    speaker_retention_ok = not flagged_speakers
    overlap_retention_ok = overlap_retention is None or overlap_retention >= overlap_retention_floor
    failure_share_ok = failure_share <= failure_share_limit
    uem_ok = not outside_rows
    unmapped_published: list[str] = []
    dropped_unmapped_ns = 0
    if mappings:
        for recording_id, mapping in sorted(mappings.items()):
            dropped_unmapped_ns += mapping.unmapped_published_ns
            for speaker in mapping.unmapped_published:
                unmapped_published.append(f"{recording_id} {speaker}")
    unmapped_speakers_ok = not unmapped_published
    notes: list[str] = []
    if not smallest_gap_ok:
        notes.append("smallest same-speaker gap is below the 200 ms merge floor")
    if not tightness_band_ok:
        notes.append("short-gap rate is outside the forced-AMI tightness band")
    if flagged_speakers:
        notes.append(f"{len(flagged_speakers)} speakers keep less than 30% of original speech")
    if flagged_recordings:
        notes.append(
            f"{len(flagged_recordings)} recordings have more than "
            f"{published_outside_flag:.0%} new speech outside the original union"
        )
    if subset_ok is False:
        notes.append("tight speech is not a subset of the original human segments")
    if not uem_ok:
        notes.append(f"{len(outside_rows)} SPEAKER or UNCERTAIN rows fall outside the UEM")
    if not unmapped_speakers_ok:
        notes.append(
            f"{len(unmapped_published)} published speakers are unmapped; "
            f"dropped {ns_to_seconds(dropped_unmapped_ns):.3f} s of published speech"
        )

    passed = (
        tightness_ok
        and speaker_retention_ok
        and overlap_retention_ok
        and failure_share_ok
        and subset_ok is not False
        and uem_ok
        and unmapped_speakers_ok
    )
    return GateResult(
        passed=passed,
        tightness_ok=tightness_ok,
        speaker_retention_ok=speaker_retention_ok,
        overlap_retention_ok=overlap_retention_ok,
        failure_share_ok=failure_share_ok,
        smallest_gap_ok=smallest_gap_ok,
        subset_ok=subset_ok,
        uem_ok=uem_ok,
        unmapped_speakers_ok=unmapped_speakers_ok,
        tightness=tightness,
        original_tightness=original_tightness,
        speaker_retention=tuple(retentions),
        overlap_retention=overlap_retention,
        failure_share=failure_share,
        time_bases=tuple(time_bases),
        flagged_speakers=flagged_speakers,
        flagged_recordings=flagged_recordings,
        rows_outside_uem=tuple(outside_rows),
        dropped_unmapped_seconds=ns_to_seconds(dropped_unmapped_ns),
        unmapped_published=tuple(unmapped_published),
        notes=tuple(notes),
    )
