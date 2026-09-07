"""Convert source annotations at a verified physical EOF boundary."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from .acceptance import RttmInterval
from .errors import PreparationError
from .hashing import sha256_file


EOF_CONVERSION_SCHEMA = "speakrs-eof-annotation-conversion"
EOF_CONVERSION_SCHEMA_VERSION = 1


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None or len(set(value.lower())) == 1:
        raise PreparationError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreparationError(f"{label} must be a positive integer")
    return value


def _require_parent(value: object, label: str = "parent_id") -> str:
    if not isinstance(value, str) or not value:
        raise PreparationError(f"{label} must be a non-empty string")
    return value


def _mapping_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparationError(f"{label} must be an object")
    return value


@dataclass(frozen=True)
class IdentityClockEvidence:
    """A hashed evidence report that proves an identity source timeline."""

    reference: Path
    sha256: str
    parent_id: str
    source_sha256: str
    sample_count: int
    sample_rate: int
    source_timeline_mapping: str
    clock_shift_observed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Path):
            raise PreparationError("identity-clock evidence reference must be a path")
        _require_digest(self.sha256, "identity-clock evidence SHA-256")
        _require_parent(self.parent_id, "identity-clock evidence parent_id")
        _require_digest(self.source_sha256, "identity-clock evidence source SHA-256")
        _require_positive_int(self.sample_count, "identity-clock evidence sample_count")
        _require_positive_int(self.sample_rate, "identity-clock evidence sample_rate")
        if not isinstance(self.source_timeline_mapping, str) or not self.source_timeline_mapping:
            raise PreparationError("identity-clock evidence must name a source timeline mapping")
        if not isinstance(self.clock_shift_observed, bool):
            raise PreparationError("identity-clock evidence clock_shift_observed must be a boolean")

    @classmethod
    def from_file(cls, reference: Path) -> IdentityClockEvidence:
        """Load source identity and clock facts from one evidence JSON file."""

        reference = Path(reference).expanduser().resolve(strict=False)
        if not reference.is_file():
            raise PreparationError("identity-clock evidence is missing", {"reference": reference.as_posix()})
        try:
            payload = _mapping_object(json.loads(reference.read_text(encoding="utf-8")), "identity-clock evidence")
            source_audio = _mapping_object(payload.get("source_audio"), "identity-clock evidence.source_audio")
            timing_facts = _mapping_object(payload.get("timing_facts"), "identity-clock evidence.timing_facts")
            parent_id = _require_parent(payload.get("recording_id"), "identity-clock evidence.recording_id")
            source_sha256 = _require_digest(source_audio.get("sha256"), "identity-clock evidence source SHA-256")
            sample_count = _require_positive_int(source_audio.get("frames"), "identity-clock evidence source frames")
            sample_rate = _require_positive_int(
                source_audio.get("sample_rate"), "identity-clock evidence source sample_rate"
            )
            source_timeline_mapping = timing_facts.get("source_timeline_mapping")
            clock_shift_observed = timing_facts.get("clock_shift_observed")
            evidence_sha256 = sha256_file(reference)
        except PreparationError:
            raise
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise PreparationError(
                "identity-clock evidence is unreadable", {"reference": reference.as_posix()}
            ) from error
        return cls(
            reference=reference,
            sha256=evidence_sha256,
            parent_id=parent_id,
            source_sha256=source_sha256,
            sample_count=sample_count,
            sample_rate=sample_rate,
            source_timeline_mapping=source_timeline_mapping,
            clock_shift_observed=clock_shift_observed,
        )


@dataclass(frozen=True)
class VerifiedSourceIdentity:
    """Original source facts bound to a required identity-clock evidence report."""

    parent_id: str
    source_sha256: str
    sample_count: int
    sample_rate: int
    identity_clock_evidence: IdentityClockEvidence

    def __post_init__(self) -> None:
        _require_parent(self.parent_id)
        _require_digest(self.source_sha256, "source_sha256")
        _require_positive_int(self.sample_count, "sample_count")
        _require_positive_int(self.sample_rate, "sample_rate")
        if not isinstance(self.identity_clock_evidence, IdentityClockEvidence):
            raise PreparationError("identity-clock evidence is required")

    @property
    def duration_seconds(self) -> float:
        """Return the physical EOF derived from verified frames and rate."""

        duration = self.sample_count / self.sample_rate
        if not math.isfinite(duration) or duration <= 0.0:
            raise PreparationError("verified source duration is not finite and positive")
        return duration


@dataclass(frozen=True)
class EofAnnotationConversion:
    """Bounded intervals and the receipt for one EOF-only conversion."""

    intervals: tuple[RttmInterval, ...]
    receipt: dict[str, object]

    @property
    def bounded_intervals(self) -> tuple[RttmInterval, ...]:
        """Return the converted intervals in original input order."""

        return self.intervals

    @property
    def excluded_speaker_seconds(self) -> float:
        """Return speaker-time removed at physical EOF."""

        return float(self.receipt["excluded_speaker_seconds"])

    def to_rttm(self) -> str:
        """Render converted intervals as deterministic RTTM speaker rows."""

        lines = []
        for interval in self.intervals:
            duration = Decimal(str(interval.end)) - Decimal(str(interval.start))
            lines.append(
                "SPEAKER "
                f"{interval.recording_id} 1 {_rttm_number(interval.start)} {_rttm_number(duration)} "
                f"<NA> <NA> {interval.speaker} <NA> <NA>\n"
            )
        return "".join(lines)


def _rttm_number(value: float | Decimal) -> str:
    """Format a finite RTTM number without binary floating-point noise."""

    number = Decimal(str(value)).quantize(Decimal("0.000000000001"))
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _interval_dict(interval: RttmInterval) -> dict[str, object]:
    return {
        "recording_id": interval.recording_id,
        "start": interval.start,
        "end": interval.end,
        "speaker": interval.speaker,
        "redacted": interval.redacted,
    }


def _validate_identity_clock_evidence(source: VerifiedSourceIdentity) -> None:
    evidence = source.identity_clock_evidence
    reference = evidence.reference
    if not reference.is_file():
        raise PreparationError("identity-clock evidence is missing", {"reference": reference.as_posix()})
    try:
        actual_sha256 = sha256_file(reference)
    except OSError as error:
        raise PreparationError(
            "identity-clock evidence cannot be hashed", {"reference": reference.as_posix()}
        ) from error
    if actual_sha256 != evidence.sha256:
        raise PreparationError("identity-clock evidence hash changed", {"reference": reference.as_posix()})
    if evidence.parent_id != source.parent_id:
        raise PreparationError("identity-clock evidence belongs to a different parent")
    if evidence.source_sha256 != source.source_sha256:
        raise PreparationError("identity-clock evidence source SHA-256 differs from verified source")
    if evidence.sample_count != source.sample_count or evidence.sample_rate != source.sample_rate:
        raise PreparationError("identity-clock evidence source duration differs from verified source")
    if evidence.source_timeline_mapping != "identity" or evidence.clock_shift_observed:
        raise PreparationError("identity-clock evidence does not prove an identity timeline")
    try:
        payload = _mapping_object(json.loads(reference.read_text(encoding="utf-8")), "identity-clock evidence")
        source_audio = _mapping_object(payload.get("source_audio"), "identity-clock evidence.source_audio")
        timing_facts = _mapping_object(payload.get("timing_facts"), "identity-clock evidence.timing_facts")
    except PreparationError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PreparationError("identity-clock evidence is unreadable", {"reference": reference.as_posix()}) from error
    if payload.get("recording_id") != source.parent_id:
        raise PreparationError("identity-clock evidence JSON belongs to a different parent")
    if source_audio.get("sha256") != source.source_sha256:
        raise PreparationError("identity-clock evidence JSON source SHA-256 differs from verified source")
    if source_audio.get("frames") != source.sample_count or source_audio.get("sample_rate") != source.sample_rate:
        raise PreparationError("identity-clock evidence JSON source facts differ from verified source")
    if (
        timing_facts.get("source_timeline_mapping") != "identity"
        or timing_facts.get("clock_shift_observed") is not False
    ):
        raise PreparationError("identity-clock evidence JSON does not prove an identity timeline")


def _validate_interval(interval: object, parent_id: str, index: int) -> RttmInterval:
    if not isinstance(interval, RttmInterval):
        raise PreparationError("EOF conversion requires typed RTTM intervals", {"index": index})
    if interval.recording_id != parent_id:
        raise PreparationError("RTTM interval belongs to a different parent", {"index": index})
    if not isinstance(interval.speaker, str) or not interval.speaker:
        raise PreparationError("RTTM interval speaker must be non-empty", {"index": index})
    if not isinstance(interval.redacted, bool):
        raise PreparationError("RTTM interval redacted must be a boolean", {"index": index})
    try:
        finite = math.isfinite(interval.start) and math.isfinite(interval.end)
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise PreparationError("RTTM interval bounds must be finite", {"index": index})
    if interval.start < 0.0 or interval.end <= interval.start:
        raise PreparationError("RTTM interval bounds must be positive and ordered", {"index": index})
    return interval


def bound_rttm_to_source_eof(
    intervals: Sequence[RttmInterval], source: VerifiedSourceIdentity
) -> EofAnnotationConversion:
    """Intersect RTTM activity with verified physical EOF and retain an audit receipt.

    In-bounds intervals are returned unchanged and in input order. Intervals
    crossing EOF are shortened only at EOF. Intervals beginning at or after EOF
    are excluded. This function does not inspect or modify UEM regions.
    """

    if not isinstance(intervals, Sequence) or isinstance(intervals, (str, bytes, bytearray)):
        raise PreparationError("EOF conversion intervals must be a sequence")
    if not isinstance(source, VerifiedSourceIdentity):
        raise PreparationError("EOF conversion requires verified source identity")
    _validate_identity_clock_evidence(source)
    eof = source.duration_seconds
    if not intervals:
        raise PreparationError("EOF conversion produced no bounded intervals")

    bounded: list[RttmInterval] = []
    audit: list[dict[str, object]] = []
    original_speaker_seconds = 0.0
    retained_speaker_seconds = 0.0
    excluded_speaker_seconds = 0.0
    for index, raw_interval in enumerate(intervals):
        interval = _validate_interval(raw_interval, source.parent_id, index)
        original_seconds = interval.end - interval.start
        original_speaker_seconds += original_seconds
        if interval.start >= eof:
            result = None
            retained_seconds = 0.0
            action = "excluded_after_eof"
        elif interval.end > eof:
            result = RttmInterval(interval.recording_id, interval.start, eof, interval.speaker, interval.redacted)
            retained_seconds = eof - interval.start
            action = "intersected_at_eof"
            bounded.append(result)
        else:
            result = interval
            retained_seconds = original_seconds
            action = "preserved"
            bounded.append(result)
        excluded_seconds = original_seconds - retained_seconds
        retained_speaker_seconds += retained_seconds
        excluded_speaker_seconds += excluded_seconds
        audit.append(
            {
                "index": index,
                "action": action,
                "original": _interval_dict(interval),
                "result": _interval_dict(result) if result is not None else None,
                "original_speaker_seconds": original_seconds,
                "retained_speaker_seconds": retained_seconds,
                "excluded_speaker_seconds": excluded_seconds,
            }
        )

    if not bounded:
        raise PreparationError("EOF conversion produced no bounded intervals")
    receipt = {
        "schema": EOF_CONVERSION_SCHEMA,
        "schema_version": EOF_CONVERSION_SCHEMA_VERSION,
        "operation": "intersect_rttm_at_verified_physical_eof",
        "parent_id": source.parent_id,
        "source": {
            "sha256": source.source_sha256,
            "sample_count": source.sample_count,
            "sample_rate": source.sample_rate,
            "duration_seconds": eof,
            "identity_clock_evidence": {
                "reference": source.identity_clock_evidence.reference.as_posix(),
                "sha256": source.identity_clock_evidence.sha256,
            },
        },
        "original_interval_count": len(intervals),
        "bounded_interval_count": len(bounded),
        "original_speaker_seconds": original_speaker_seconds,
        "retained_speaker_seconds": retained_speaker_seconds,
        "excluded_speaker_seconds": excluded_speaker_seconds,
        "intervals": audit,
        "uem_policy": "unchanged; UEM gaps are not clipped by this conversion",
    }
    return EofAnnotationConversion(tuple(bounded), receipt)
