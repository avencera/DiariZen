"""Reusable tight speaker-activity labels with a 200 ms same-speaker gap."""

from __future__ import annotations

from .domain import Interval, RecordingLabels, SpeakerSegment, UncertainInterval
from .gates import GateResult, evaluate_gates
from .intervals import clip_recording_to_uem
from .mapping import SpeakerMapping, apply_published_replacement, map_speakers_by_overlap
from .merge import MIN_GAP_NS, MIN_GAP_SECONDS, merge_recording, merge_short_gaps
from .subset import is_subset_of_original, recording_is_subset
from .tighten import COVERAGE_RATIO, TightenSegmentResult, tighten_recording, tighten_segment
from .tightness import FORCED_AMI_REFERENCE, TightnessStats, measure_tightness
from .time import parse_seconds_to_ns
from .writers import (
    parse_recording_rttm,
    parse_speaker_rttm,
    write_recording_labels,
    write_speaker_rttm,
    write_uncertain_rttm,
)


LIBRARY_VERSION = "2"

__all__ = [
    "COVERAGE_RATIO",
    "FORCED_AMI_REFERENCE",
    "LIBRARY_VERSION",
    "MIN_GAP_NS",
    "MIN_GAP_SECONDS",
    "GateResult",
    "Interval",
    "RecordingLabels",
    "SpeakerMapping",
    "SpeakerSegment",
    "TightenSegmentResult",
    "TightnessStats",
    "UncertainInterval",
    "apply_published_replacement",
    "clip_recording_to_uem",
    "evaluate_gates",
    "is_subset_of_original",
    "map_speakers_by_overlap",
    "measure_tightness",
    "merge_recording",
    "merge_short_gaps",
    "parse_recording_rttm",
    "parse_seconds_to_ns",
    "parse_speaker_rttm",
    "recording_is_subset",
    "tighten_recording",
    "tighten_segment",
    "write_recording_labels",
    "write_speaker_rttm",
    "write_uncertain_rttm",
]
