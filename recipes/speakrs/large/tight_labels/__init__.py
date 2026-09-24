"""Reusable tight speaker-activity labels with a 200 ms same-speaker gap."""

from __future__ import annotations

from .domain import Interval, RecordingLabels, SpeakerSegment, UncertainInterval
from .classify import (
    AlignmentFailureReason,
    ByDesignReason,
    SegmentClass,
    SegmentOutcome,
    aligned_classes,
    require_segment_classes,
)
from .gates import (
    GATE_POLICY_VERSION,
    AdmissionStatus,
    GateResult,
    OverlapLoss,
    UncertainShare,
    evaluate_gates,
)
from .intervals import clip_recording_to_uem
from .mapping import (
    SpeakerMapping,
    apply_published_replacement,
    classify_published_replacement,
    map_speakers_by_overlap,
)
from .merge import (
    MIN_GAP_NS,
    MIN_GAP_SECONDS,
    canonicalize_uncertain,
    merge_recording,
    merge_short_gaps,
    subtract_uncertain_from_active,
)
from .subset import ClipReport, clip_to_original_support, is_subset_of_original, recording_is_subset
from .tighten import COVERAGE_RATIO, TightenSegmentResult, tighten_recording, tighten_segment
from .tightness import FORCED_AMI_REFERENCE, TightnessStats, measure_tightness
from .time import parse_seconds_to_ns
from .writers import (
    load_written_recording_labels,
    parse_recording_rttm,
    parse_speaker_rttm,
    write_recording_labels,
    write_speaker_rttm,
    write_uncertain_rttm,
)


LIBRARY_VERSION = "4"

__all__ = [
    "COVERAGE_RATIO",
    "FORCED_AMI_REFERENCE",
    "GATE_POLICY_VERSION",
    "LIBRARY_VERSION",
    "MIN_GAP_NS",
    "MIN_GAP_SECONDS",
    "AdmissionStatus",
    "AlignmentFailureReason",
    "ByDesignReason",
    "ClipReport",
    "GateResult",
    "Interval",
    "OverlapLoss",
    "RecordingLabels",
    "SegmentClass",
    "SegmentOutcome",
    "SpeakerMapping",
    "SpeakerSegment",
    "TightenSegmentResult",
    "TightnessStats",
    "UncertainInterval",
    "UncertainShare",
    "aligned_classes",
    "apply_published_replacement",
    "canonicalize_uncertain",
    "classify_published_replacement",
    "clip_recording_to_uem",
    "clip_to_original_support",
    "evaluate_gates",
    "is_subset_of_original",
    "load_written_recording_labels",
    "map_speakers_by_overlap",
    "measure_tightness",
    "merge_recording",
    "merge_short_gaps",
    "parse_recording_rttm",
    "parse_seconds_to_ns",
    "parse_speaker_rttm",
    "recording_is_subset",
    "require_segment_classes",
    "subtract_uncertain_from_active",
    "tighten_recording",
    "tighten_segment",
    "write_recording_labels",
    "write_speaker_rttm",
    "write_uncertain_rttm",
]
