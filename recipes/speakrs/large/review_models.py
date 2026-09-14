"""Typed Open Yap review state that cannot represent invalid combinations.

Integer 20 ms frame indices are the only authoritative activity coordinates.
Transcript words are a display aid and never become accepted human activity
without an explicit reviewer action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence, Union

from .errors import ContractError
from .hashing import canonical_json, sha256_json


WINDOW_SECONDS = 30
FRAME_SECONDS = 0.02
WINDOW_FRAME_COUNT = 1500
CONVERSION_POLICY = "speakrs-open-yap-grid-proposal-v1"
GENESIS_HASH = "0" * 64
WORD_TYPES = frozenset({"word", "filler", "laugh", "cough", "noise"})
SPEAKER_VALUES = ("speaker_a", "speaker_b")
TRANSCRIPT_TOP_FIELDS = frozenset(
    {"conversation_id", "speaker_index", "language", "text", "words", "corrections_applied"}
)
TRANSCRIPT_WORD_FIELDS = frozenset({"word", "start", "end", "type", "corrections_applied"})
REQUIRED_TRANSCRIPT_WORD_FIELDS = frozenset({"word", "start", "end", "type"})


class SpeakerRole(str, Enum):
    """One of the two frozen source speaker tracks."""

    SPEAKER_A = "speaker_a"
    SPEAKER_B = "speaker_b"


def require_mapping(value: object, label: str) -> Mapping[str, Any]:
    """Return a mapping or raise a contract error."""

    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str] | set[str],
    label: str,
) -> None:
    """Reject keys that are not part of the known schema."""

    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ContractError(f"{label} has unknown fields", {"unknown": unknown})


def require_fields(
    value: Mapping[str, Any],
    required: frozenset[str] | set[str],
    label: str,
) -> None:
    """Reject a mapping that omits required keys."""

    missing = sorted(set(required) - set(value))
    if missing:
        raise ContractError(f"{label} is missing fields", {"missing": missing})


def parse_speaker(value: object, label: str = "speaker") -> SpeakerRole:
    """Parse a speaker role."""

    if value in (SpeakerRole.SPEAKER_A, SpeakerRole.SPEAKER_B):
        return SpeakerRole(value)
    if value in SPEAKER_VALUES:
        return SpeakerRole(str(value))
    raise ContractError(f"{label} must be speaker_a or speaker_b", {"value": value})


def parse_frame_index(value: object, label: str, *, end: bool = False) -> int:
    """Parse an integer frame index on the 30-second window grid."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer frame index")
    maximum = WINDOW_FRAME_COUNT if end else WINDOW_FRAME_COUNT - 1
    if value < 0 or value > maximum:
        raise ContractError(
            f"{label} is outside the 20 ms window grid",
            {"value": value, "maximum": maximum},
        )
    if not end and value >= WINDOW_FRAME_COUNT:
        raise ContractError(f"{label} cannot start at the exclusive window end")
    return value


def parse_sha256(value: object, label: str) -> str:
    """Parse a lowercase SHA-256 hex digest."""

    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ContractError(f"{label} must be a SHA-256 hex digest") from error
    if value != value.lower():
        raise ContractError(f"{label} must be lowercase hex")
    return value


def parse_non_empty_text(value: object, label: str) -> str:
    """Parse a non-empty string."""

    if not isinstance(value, str):
        raise ContractError(f"{label} must be a string")
    text = value.strip()
    if not text:
        raise ContractError(f"{label} must be non-empty")
    return text


def parse_finite_number(value: object, label: str) -> float:
    """Parse a finite JSON number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ContractError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class ActivityInterval:
    """One speaker-active half-open frame range with end > start."""

    speaker: SpeakerRole
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        start = parse_frame_index(self.start_frame, "start_frame")
        end = parse_frame_index(self.end_frame, "end_frame", end=True)
        if end <= start:
            raise ContractError("activity interval end must be greater than start")
        object.__setattr__(self, "start_frame", start)
        object.__setattr__(self, "end_frame", end)

    def frame_set(self) -> frozenset[int]:
        """Return occupied frame indices."""

        return frozenset(range(self.start_frame, self.end_frame))

    def to_dict(self) -> dict[str, object]:
        """Serialize the interval."""

        return {
            "speaker": self.speaker.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }

    @classmethod
    def from_dict(cls, value: object) -> ActivityInterval:
        """Parse one interval object."""

        record = require_mapping(value, "activity interval")
        reject_unknown_fields(record, frozenset({"speaker", "start_frame", "end_frame"}), "activity interval")
        require_fields(record, frozenset({"speaker", "start_frame", "end_frame"}), "activity interval")
        return cls(
            speaker=parse_speaker(record["speaker"]),
            start_frame=record["start_frame"],
            end_frame=record["end_frame"],
        )


def normalize_intervals(intervals: Sequence[ActivityInterval]) -> tuple[ActivityInterval, ...]:
    """Merge contiguous or overlapping ranges for the same speaker."""

    grouped: dict[SpeakerRole, list[ActivityInterval]] = {
        SpeakerRole.SPEAKER_A: [],
        SpeakerRole.SPEAKER_B: [],
    }
    for interval in intervals:
        grouped[interval.speaker].append(interval)
    merged: list[ActivityInterval] = []
    for speaker, items in grouped.items():
        items.sort(key=lambda item: (item.start_frame, item.end_frame))
        current: ActivityInterval | None = None
        for item in items:
            if current is None:
                current = item
                continue
            if item.start_frame <= current.end_frame:
                current = ActivityInterval(speaker, current.start_frame, max(current.end_frame, item.end_frame))
                continue
            merged.append(current)
            current = item
        if current is not None:
            merged.append(current)
    merged.sort(key=lambda item: (item.start_frame, item.speaker.value))
    return tuple(merged)


def parse_intervals(value: object, label: str = "activity") -> tuple[ActivityInterval, ...]:
    """Parse and normalize a list of activity intervals."""

    if not isinstance(value, list):
        raise ContractError(f"{label} must be an array")
    return normalize_intervals([ActivityInterval.from_dict(item) for item in value])


def occupied_frames(intervals: Sequence[ActivityInterval]) -> dict[SpeakerRole, frozenset[int]]:
    """Return per-speaker occupied frame sets."""

    frames = {role: set() for role in SpeakerRole}
    for interval in intervals:
        frames[interval.speaker].update(interval.frame_set())
    return {role: frozenset(values) for role, values in frames.items()}


def seconds_from_frame(frame: int) -> float:
    """Convert a frame index to seconds by exact multiplication with 0.02."""

    parse_frame_index(frame, "frame", end=True)
    return frame * FRAME_SECONDS


def frames_intersecting_seconds(start_seconds: float, end_seconds: float) -> tuple[int, int] | None:
    """Return the half-open frame range occupied by a source interval."""

    if end_seconds <= start_seconds:
        return None
    window_end = WINDOW_FRAME_COUNT * FRAME_SECONDS
    if end_seconds <= 0.0 or start_seconds >= window_end:
        return None
    start_frame = math.floor(start_seconds / FRAME_SECONDS + 1e-9)
    end_frame = math.ceil(end_seconds / FRAME_SECONDS - 1e-9)
    start_frame = max(0, start_frame)
    end_frame = min(WINDOW_FRAME_COUNT, end_frame)
    if end_frame <= start_frame:
        return None
    return start_frame, end_frame


@dataclass(frozen=True)
class GridProposal:
    """Normalized machine activity on the integer 20 ms grid."""

    window_id: str
    candidate_sha256: str
    conversion_policy: str
    frame_seconds: float
    intervals: tuple[ActivityInterval, ...]
    content_sha256: str

    def __post_init__(self) -> None:
        if self.conversion_policy != CONVERSION_POLICY:
            raise ContractError("grid proposal conversion policy is unknown")
        if self.frame_seconds != FRAME_SECONDS:
            raise ContractError("grid proposal frame duration must be 0.02 seconds")
        object.__setattr__(self, "intervals", normalize_intervals(self.intervals))
        parse_sha256(self.candidate_sha256, "candidate_sha256")
        parse_non_empty_text(self.window_id, "window_id")

    def to_dict(self) -> dict[str, object]:
        """Serialize the proposal without its content digest."""

        return {
            "schema": "speakrs-open-yap-grid-proposal",
            "schema_version": 1,
            "window_id": self.window_id,
            "candidate_sha256": self.candidate_sha256,
            "conversion_policy": self.conversion_policy,
            "frame_seconds": self.frame_seconds,
            "intervals": [interval.to_dict() for interval in self.intervals],
        }

    @classmethod
    def from_intervals(
        cls,
        *,
        window_id: str,
        candidate_sha256: str,
        intervals: Sequence[ActivityInterval],
    ) -> GridProposal:
        """Build a proposal and bind its content identity."""

        normalized = normalize_intervals(intervals)
        payload = {
            "schema": "speakrs-open-yap-grid-proposal",
            "schema_version": 1,
            "window_id": window_id,
            "candidate_sha256": candidate_sha256,
            "conversion_policy": CONVERSION_POLICY,
            "frame_seconds": FRAME_SECONDS,
            "intervals": [interval.to_dict() for interval in normalized],
        }
        return cls(
            window_id=window_id,
            candidate_sha256=candidate_sha256,
            conversion_policy=CONVERSION_POLICY,
            frame_seconds=FRAME_SECONDS,
            intervals=normalized,
            content_sha256=sha256_json(payload),
        )

    @classmethod
    def from_dict(cls, value: object) -> GridProposal:
        """Parse a stored proposal and recompute its identity."""

        record = require_mapping(value, "grid proposal")
        reject_unknown_fields(
            record,
            frozenset(
                {
                    "schema",
                    "schema_version",
                    "window_id",
                    "candidate_sha256",
                    "conversion_policy",
                    "frame_seconds",
                    "intervals",
                    "content_sha256",
                }
            ),
            "grid proposal",
        )
        require_fields(
            record,
            frozenset(
                {
                    "schema",
                    "schema_version",
                    "window_id",
                    "candidate_sha256",
                    "conversion_policy",
                    "frame_seconds",
                    "intervals",
                }
            ),
            "grid proposal",
        )
        if record.get("schema") != "speakrs-open-yap-grid-proposal" or record.get("schema_version") != 1:
            raise ContractError("grid proposal schema is invalid")
        proposal = cls.from_intervals(
            window_id=parse_non_empty_text(record["window_id"], "window_id"),
            candidate_sha256=parse_sha256(record["candidate_sha256"], "candidate_sha256"),
            intervals=parse_intervals(record["intervals"], "intervals"),
        )
        stored_hash = record.get("content_sha256")
        if stored_hash is not None and parse_sha256(stored_hash, "content_sha256") != proposal.content_sha256:
            raise ContractError("grid proposal content hash does not match")
        return proposal


@dataclass(frozen=True)
class TranscriptWord:
    """One clipped source word retained for display only."""

    speaker: SpeakerRole
    text: str
    word_type: str
    source_start_seconds: float
    source_end_seconds: float
    window_start_seconds: float
    window_end_seconds: float
    source_transcript_sha256: str

    def __post_init__(self) -> None:
        parse_non_empty_text(self.text, "text")
        if self.word_type not in WORD_TYPES:
            raise ContractError("transcript word type is unknown", {"type": self.word_type})
        if self.source_end_seconds <= self.source_start_seconds:
            raise ContractError("transcript word source interval is not positive")
        if self.window_end_seconds <= self.window_start_seconds:
            raise ContractError("transcript word window interval is not positive")
        parse_sha256(self.source_transcript_sha256, "source_transcript_sha256")

    def to_dict(self) -> dict[str, object]:
        """Serialize the clipped word."""

        return {
            "speaker": self.speaker.value,
            "text": self.text,
            "type": self.word_type,
            "source_start_seconds": self.source_start_seconds,
            "source_end_seconds": self.source_end_seconds,
            "window_start_seconds": self.window_start_seconds,
            "window_end_seconds": self.window_end_seconds,
            "source_transcript_sha256": self.source_transcript_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> TranscriptWord:
        """Parse one retained word."""

        record = require_mapping(value, "transcript word")
        reject_unknown_fields(
            record,
            frozenset(
                {
                    "speaker",
                    "text",
                    "type",
                    "source_start_seconds",
                    "source_end_seconds",
                    "window_start_seconds",
                    "window_end_seconds",
                    "source_transcript_sha256",
                }
            ),
            "transcript word",
        )
        require_fields(
            record,
            frozenset(
                {
                    "speaker",
                    "text",
                    "type",
                    "source_start_seconds",
                    "source_end_seconds",
                    "window_start_seconds",
                    "window_end_seconds",
                    "source_transcript_sha256",
                }
            ),
            "transcript word",
        )
        return cls(
            speaker=parse_speaker(record["speaker"]),
            text=parse_non_empty_text(record["text"], "text"),
            word_type=str(record["type"]),
            source_start_seconds=parse_finite_number(record["source_start_seconds"], "source_start_seconds"),
            source_end_seconds=parse_finite_number(record["source_end_seconds"], "source_end_seconds"),
            window_start_seconds=parse_finite_number(record["window_start_seconds"], "window_start_seconds"),
            window_end_seconds=parse_finite_number(record["window_end_seconds"], "window_end_seconds"),
            source_transcript_sha256=parse_sha256(record["source_transcript_sha256"], "source_transcript_sha256"),
        )


@dataclass(frozen=True)
class WholeWindowScope:
    """A follow-up or uncertainty that applies to the whole window."""

    kind: Literal["whole_window"] = "whole_window"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "whole_window"}


@dataclass(frozen=True)
class BoundedRangeScope:
    """A follow-up or uncertainty limited to explicit frame ranges."""

    ranges: tuple[tuple[int, int], ...]
    kind: Literal["bounded_ranges"] = "bounded_ranges"

    def __post_init__(self) -> None:
        if not self.ranges:
            raise ContractError("bounded range scope needs at least one range")
        cleaned: list[tuple[int, int]] = []
        for start, end in self.ranges:
            start_frame = parse_frame_index(start, "range.start_frame")
            end_frame = parse_frame_index(end, "range.end_frame", end=True)
            if end_frame <= start_frame:
                raise ContractError("bounded range end must be greater than start")
            cleaned.append((start_frame, end_frame))
        object.__setattr__(self, "ranges", tuple(cleaned))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "bounded_ranges",
            "ranges": [{"start_frame": start, "end_frame": end} for start, end in self.ranges],
        }


ReviewScope = Union[WholeWindowScope, BoundedRangeScope]


def parse_scope(value: object) -> ReviewScope:
    """Parse a tagged whole-window or bounded-range choice."""

    record = require_mapping(value, "review scope")
    kind = record.get("kind")
    if kind == "whole_window":
        reject_unknown_fields(record, frozenset({"kind"}), "whole-window scope")
        return WholeWindowScope()
    if kind == "bounded_ranges":
        reject_unknown_fields(record, frozenset({"kind", "ranges"}), "bounded-range scope")
        require_fields(record, frozenset({"kind", "ranges"}), "bounded-range scope")
        ranges = record["ranges"]
        if not isinstance(ranges, list):
            raise ContractError("bounded ranges must be an array")
        parsed: list[tuple[int, int]] = []
        for item in ranges:
            item_record = require_mapping(item, "bounded range")
            reject_unknown_fields(item_record, frozenset({"start_frame", "end_frame"}), "bounded range")
            require_fields(item_record, frozenset({"start_frame", "end_frame"}), "bounded range")
            parsed.append((item_record["start_frame"], item_record["end_frame"]))
        return BoundedRangeScope(tuple(parsed))
    raise ContractError("review scope kind is unknown", {"kind": kind})


@dataclass(frozen=True)
class Pending:
    """No accepted human activity decision exists yet."""

    kind: Literal["pending"] = "pending"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "pending"}


@dataclass(frozen=True)
class ConfirmedProposal:
    """The reviewer accepted the grid proposal after listening."""

    kind: Literal["confirmed_proposal"] = "confirmed_proposal"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "confirmed_proposal"}


@dataclass(frozen=True)
class Corrected:
    """The reviewer replaced the proposal with edited activity."""

    activity: tuple[ActivityInterval, ...]
    kind: Literal["corrected"] = "corrected"

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity", normalize_intervals(self.activity))
        if not self.activity:
            raise ContractError("corrected activity must contain at least one interval")

    def to_dict(self) -> dict[str, object]:
        return {"kind": "corrected", "activity": [interval.to_dict() for interval in self.activity]}


@dataclass(frozen=True)
class NoSpeech:
    """The reviewer marked the window as containing no speech."""

    kind: Literal["no_speech"] = "no_speech"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "no_speech"}


@dataclass(frozen=True)
class NeedsFollowUp:
    """The reviewer denied the proposal and recorded a reasoned follow-up."""

    reason: str
    scope: ReviewScope
    kind: Literal["needs_follow_up"] = "needs_follow_up"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", parse_non_empty_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, object]:
        return {"kind": "needs_follow_up", "reason": self.reason, "scope": self.scope.to_dict()}


@dataclass(frozen=True)
class Uncertain:
    """The reviewer could not decide the activity."""

    reason: str
    scope: ReviewScope
    kind: Literal["uncertain"] = "uncertain"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", parse_non_empty_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, object]:
        return {"kind": "uncertain", "reason": self.reason, "scope": self.scope.to_dict()}


@dataclass(frozen=True)
class SignedOff:
    """An independent signer accepted the reviewed activity state."""

    review_event_hash: str
    signer: str
    kind: Literal["signed_off"] = "signed_off"

    def __post_init__(self) -> None:
        parse_sha256(self.review_event_hash, "review_event_hash")
        object.__setattr__(self, "signer", parse_non_empty_text(self.signer, "signer"))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "signed_off",
            "review_event_hash": self.review_event_hash,
            "signer": self.signer,
        }


@dataclass(frozen=True)
class Returned:
    """An independent signer returned the window for correction."""

    review_event_hash: str
    signer: str
    reason: str
    kind: Literal["returned"] = "returned"

    def __post_init__(self) -> None:
        parse_sha256(self.review_event_hash, "review_event_hash")
        object.__setattr__(self, "signer", parse_non_empty_text(self.signer, "signer"))
        object.__setattr__(self, "reason", parse_non_empty_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "returned",
            "review_event_hash": self.review_event_hash,
            "signer": self.signer,
            "reason": self.reason,
        }


DecisionState = Union[
    Pending,
    ConfirmedProposal,
    Corrected,
    NoSpeech,
    NeedsFollowUp,
    Uncertain,
    SignedOff,
    Returned,
]


def parse_decision(value: object) -> DecisionState:
    """Parse a tagged activity decision."""

    record = require_mapping(value, "decision")
    kind = record.get("kind")
    if kind == "pending":
        reject_unknown_fields(record, frozenset({"kind"}), "pending decision")
        return Pending()
    if kind == "confirmed_proposal":
        reject_unknown_fields(record, frozenset({"kind"}), "confirmed proposal")
        return ConfirmedProposal()
    if kind == "corrected":
        reject_unknown_fields(record, frozenset({"kind", "activity"}), "corrected decision")
        require_fields(record, frozenset({"kind", "activity"}), "corrected decision")
        return Corrected(parse_intervals(record["activity"]))
    if kind == "no_speech":
        reject_unknown_fields(record, frozenset({"kind"}), "no-speech decision")
        return NoSpeech()
    if kind == "needs_follow_up":
        reject_unknown_fields(record, frozenset({"kind", "reason", "scope"}), "needs-follow-up decision")
        require_fields(record, frozenset({"kind", "reason", "scope"}), "needs-follow-up decision")
        return NeedsFollowUp(str(record["reason"]), parse_scope(record["scope"]))
    if kind == "uncertain":
        reject_unknown_fields(record, frozenset({"kind", "reason", "scope"}), "uncertain decision")
        require_fields(record, frozenset({"kind", "reason", "scope"}), "uncertain decision")
        return Uncertain(str(record["reason"]), parse_scope(record["scope"]))
    if kind == "signed_off":
        reject_unknown_fields(record, frozenset({"kind", "review_event_hash", "signer"}), "signed-off decision")
        require_fields(record, frozenset({"kind", "review_event_hash", "signer"}), "signed-off decision")
        return SignedOff(str(record["review_event_hash"]), str(record["signer"]))
    if kind == "returned":
        reject_unknown_fields(
            record,
            frozenset({"kind", "review_event_hash", "signer", "reason"}),
            "returned decision",
        )
        require_fields(record, frozenset({"kind", "review_event_hash", "signer", "reason"}), "returned decision")
        return Returned(str(record["review_event_hash"]), str(record["signer"]), str(record["reason"]))
    raise ContractError("decision kind is unknown", {"kind": kind})


@dataclass(frozen=True)
class NotReviewed:
    """The reviewer has not assessed this defect."""

    kind: Literal["not_reviewed"] = "not_reviewed"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "not_reviewed"}


@dataclass(frozen=True)
class Clear:
    """The reviewer marked this defect clear."""

    kind: Literal["clear"] = "clear"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "clear"}


@dataclass(frozen=True)
class UnresolvedDefect:
    """The reviewer recorded an unresolved defect with a reason."""

    reason: str
    kind: Literal["unresolved"] = "unresolved"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", parse_non_empty_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, object]:
        return {"kind": "unresolved", "reason": self.reason}


DefectAssessment = Union[NotReviewed, Clear, UnresolvedDefect]


def parse_defect(value: object, label: str) -> DefectAssessment:
    """Parse a tagged defect assessment."""

    record = require_mapping(value, label)
    kind = record.get("kind")
    if kind == "not_reviewed":
        reject_unknown_fields(record, frozenset({"kind"}), label)
        return NotReviewed()
    if kind == "clear":
        reject_unknown_fields(record, frozenset({"kind"}), label)
        return Clear()
    if kind == "unresolved":
        reject_unknown_fields(record, frozenset({"kind", "reason"}), label)
        require_fields(record, frozenset({"kind", "reason"}), label)
        return UnresolvedDefect(str(record["reason"]))
    raise ContractError(f"{label} kind is unknown", {"kind": kind})


@dataclass(frozen=True)
class ClockMeasurementAbsent:
    """The reviewer did not enter offset or drift."""

    kind: Literal["absent"] = "absent"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "absent"}


@dataclass(frozen=True)
class ClockMeasurementEntered:
    """The reviewer entered measured offset and drift."""

    offset_seconds: float
    drift_seconds_per_second: float
    kind: Literal["entered"] = "entered"

    def __post_init__(self) -> None:
        parse_finite_number(self.offset_seconds, "offset_seconds")
        parse_finite_number(self.drift_seconds_per_second, "drift_seconds_per_second")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "entered",
            "offset_seconds": self.offset_seconds,
            "drift_seconds_per_second": self.drift_seconds_per_second,
        }


ClockMeasurement = Union[ClockMeasurementAbsent, ClockMeasurementEntered]


def parse_clock_measurement(value: object) -> ClockMeasurement:
    """Parse an optional entered clock measurement."""

    record = require_mapping(value, "clock measurement")
    kind = record.get("kind")
    if kind == "absent":
        reject_unknown_fields(record, frozenset({"kind"}), "clock measurement")
        return ClockMeasurementAbsent()
    if kind == "entered":
        reject_unknown_fields(
            record,
            frozenset({"kind", "offset_seconds", "drift_seconds_per_second"}),
            "clock measurement",
        )
        require_fields(record, frozenset({"kind", "offset_seconds", "drift_seconds_per_second"}), "clock measurement")
        return ClockMeasurementEntered(
            parse_finite_number(record["offset_seconds"], "offset_seconds"),
            parse_finite_number(record["drift_seconds_per_second"], "drift_seconds_per_second"),
        )
    raise ContractError("clock measurement kind is unknown", {"kind": kind})


@dataclass(frozen=True)
class ClockNotReviewed:
    """Clock alignment has not been assessed."""

    kind: Literal["not_reviewed"] = "not_reviewed"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "not_reviewed"}


@dataclass(frozen=True)
class ClockClear:
    """Clock alignment is clear, with measurement only if entered."""

    measurement: ClockMeasurement
    kind: Literal["clear"] = "clear"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "clear", "measurement": self.measurement.to_dict()}


@dataclass(frozen=True)
class ClockUnresolved:
    """Clock alignment is unresolved, with measurement only if entered."""

    reason: str
    measurement: ClockMeasurement
    kind: Literal["unresolved"] = "unresolved"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", parse_non_empty_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "unresolved",
            "reason": self.reason,
            "measurement": self.measurement.to_dict(),
        }


ClockAssessment = Union[ClockNotReviewed, ClockClear, ClockUnresolved]


def parse_clock(value: object) -> ClockAssessment:
    """Parse a tagged clock assessment."""

    record = require_mapping(value, "clock assessment")
    kind = record.get("kind")
    if kind == "not_reviewed":
        reject_unknown_fields(record, frozenset({"kind"}), "clock assessment")
        return ClockNotReviewed()
    if kind == "clear":
        reject_unknown_fields(record, frozenset({"kind", "measurement"}), "clock assessment")
        require_fields(record, frozenset({"kind", "measurement"}), "clock assessment")
        return ClockClear(parse_clock_measurement(record["measurement"]))
    if kind == "unresolved":
        reject_unknown_fields(record, frozenset({"kind", "reason", "measurement"}), "clock assessment")
        require_fields(record, frozenset({"kind", "reason", "measurement"}), "clock assessment")
        return ClockUnresolved(str(record["reason"]), parse_clock_measurement(record["measurement"]))
    raise ContractError("clock assessment kind is unknown", {"kind": kind})


@dataclass(frozen=True)
class DefectSet:
    """Identity, clock, synchronization, and redaction assessments."""

    identity: DefectAssessment = NotReviewed()
    clock: ClockAssessment = ClockNotReviewed()
    synchronization: DefectAssessment = NotReviewed()
    redaction: DefectAssessment = NotReviewed()

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "clock": self.clock.to_dict(),
            "synchronization": self.synchronization.to_dict(),
            "redaction": self.redaction.to_dict(),
        }

    def has_unresolved(self) -> bool:
        """Return True when any assessment is unresolved or not reviewed."""

        defects = (self.identity, self.synchronization, self.redaction)
        if any(not isinstance(item, Clear) for item in defects):
            return True
        return not isinstance(self.clock, ClockClear)

    @classmethod
    def from_dict(cls, value: object) -> DefectSet:
        record = require_mapping(value, "defects")
        reject_unknown_fields(
            record,
            frozenset({"identity", "clock", "synchronization", "redaction"}),
            "defects",
        )
        require_fields(record, frozenset({"identity", "clock", "synchronization", "redaction"}), "defects")
        return cls(
            identity=parse_defect(record["identity"], "identity"),
            clock=parse_clock(record["clock"]),
            synchronization=parse_defect(record["synchronization"], "synchronization"),
            redaction=parse_defect(record["redaction"], "redaction"),
        )


@dataclass(frozen=True)
class WindowReviewState:
    """Derived review state for one window."""

    decision: DecisionState = Pending()
    defects: DefectSet = DefectSet()
    last_review_actor: str | None = None
    last_review_event_hash: str | None = None
    revision: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.to_dict(),
            "defects": self.defects.to_dict(),
            "last_review_actor": self.last_review_actor,
            "last_review_event_hash": self.last_review_event_hash,
            "revision": self.revision,
        }

    def is_unsigned_review(self) -> bool:
        return isinstance(
            self.decision,
            (ConfirmedProposal, Corrected, NoSpeech, NeedsFollowUp, Uncertain),
        )

    def is_signoff_eligible_activity(self) -> bool:
        return isinstance(self.decision, (ConfirmedProposal, Corrected, NoSpeech))

    def is_export_eligible(self) -> bool:
        return isinstance(self.decision, SignedOff) and not self.defects.has_unresolved()

    def activity_for_export(self, proposal: GridProposal) -> tuple[ActivityInterval, ...]:
        """Return accepted activity or raise if the decision is not exportable."""

        if not isinstance(self.decision, SignedOff):
            raise ContractError("export requires a signed-off window")
        if self.defects.has_unresolved():
            raise ContractError("export excludes unresolved defect assessments")
        source = self.last_signed_activity(proposal)
        if source is None:
            raise ContractError("signed decision is not an exportable activity state")
        return source

    def last_signed_activity(self, proposal: GridProposal) -> tuple[ActivityInterval, ...] | None:
        """Return activity implied by the pre-sign-off decision stored on this state.

        SignedOff does not embed activity. Callers that need activity after
        sign-off must pass the decision that was signed. This helper is used
        only before the decision is replaced; export reconstructs from events.
        """

        if isinstance(self.decision, ConfirmedProposal):
            return proposal.intervals
        if isinstance(self.decision, Corrected):
            return self.decision.activity
        if isinstance(self.decision, NoSpeech):
            return ()
        return None


@dataclass(frozen=True)
class ConfirmAction:
    kind: Literal["confirm"] = "confirm"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "confirm"}


@dataclass(frozen=True)
class CorrectAction:
    activity: tuple[ActivityInterval, ...]
    kind: Literal["correct"] = "correct"

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity", normalize_intervals(self.activity))
        if not self.activity:
            raise ContractError("correction must contain at least one interval")

    def to_dict(self) -> dict[str, object]:
        return {"kind": "correct", "activity": [interval.to_dict() for interval in self.activity]}


@dataclass(frozen=True)
class NoSpeechAction:
    kind: Literal["no_speech"] = "no_speech"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "no_speech"}


@dataclass(frozen=True)
class NeedsFollowUpAction:
    reason: str
    scope: ReviewScope
    kind: Literal["needs_follow_up"] = "needs_follow_up"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", parse_non_empty_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, object]:
        return {"kind": "needs_follow_up", "reason": self.reason, "scope": self.scope.to_dict()}


@dataclass(frozen=True)
class UncertainAction:
    reason: str
    scope: ReviewScope
    kind: Literal["uncertain"] = "uncertain"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", parse_non_empty_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, object]:
        return {"kind": "uncertain", "reason": self.reason, "scope": self.scope.to_dict()}


@dataclass(frozen=True)
class SetDefectsAction:
    defects: DefectSet
    kind: Literal["set_defects"] = "set_defects"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "set_defects", "defects": self.defects.to_dict()}


@dataclass(frozen=True)
class UndoAction:
    reverted_event_hash: str
    kind: Literal["undo"] = "undo"

    def __post_init__(self) -> None:
        parse_sha256(self.reverted_event_hash, "reverted_event_hash")

    def to_dict(self) -> dict[str, object]:
        return {"kind": "undo", "reverted_event_hash": self.reverted_event_hash}


@dataclass(frozen=True)
class SignOffAcceptAction:
    kind: Literal["sign_off_accept"] = "sign_off_accept"

    def to_dict(self) -> dict[str, object]:
        return {"kind": "sign_off_accept"}


@dataclass(frozen=True)
class SignOffReturnAction:
    reason: str
    kind: Literal["sign_off_return"] = "sign_off_return"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", parse_non_empty_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, object]:
        return {"kind": "sign_off_return", "reason": self.reason}


ReviewAction = Union[
    ConfirmAction,
    CorrectAction,
    NoSpeechAction,
    NeedsFollowUpAction,
    UncertainAction,
    SetDefectsAction,
    UndoAction,
    SignOffAcceptAction,
    SignOffReturnAction,
]


def parse_action(value: object) -> ReviewAction:
    """Parse a tagged review action payload."""

    record = require_mapping(value, "action")
    kind = record.get("kind")
    if kind == "confirm":
        reject_unknown_fields(record, frozenset({"kind"}), "confirm action")
        return ConfirmAction()
    if kind == "correct":
        reject_unknown_fields(record, frozenset({"kind", "activity"}), "correct action")
        require_fields(record, frozenset({"kind", "activity"}), "correct action")
        return CorrectAction(parse_intervals(record["activity"]))
    if kind == "no_speech":
        reject_unknown_fields(record, frozenset({"kind"}), "no-speech action")
        return NoSpeechAction()
    if kind == "needs_follow_up":
        reject_unknown_fields(record, frozenset({"kind", "reason", "scope"}), "needs-follow-up action")
        require_fields(record, frozenset({"kind", "reason", "scope"}), "needs-follow-up action")
        return NeedsFollowUpAction(str(record["reason"]), parse_scope(record["scope"]))
    if kind == "uncertain":
        reject_unknown_fields(record, frozenset({"kind", "reason", "scope"}), "uncertain action")
        require_fields(record, frozenset({"kind", "reason", "scope"}), "uncertain action")
        return UncertainAction(str(record["reason"]), parse_scope(record["scope"]))
    if kind == "set_defects":
        reject_unknown_fields(record, frozenset({"kind", "defects"}), "set-defects action")
        require_fields(record, frozenset({"kind", "defects"}), "set-defects action")
        return SetDefectsAction(DefectSet.from_dict(record["defects"]))
    if kind == "undo":
        reject_unknown_fields(record, frozenset({"kind", "reverted_event_hash"}), "undo action")
        require_fields(record, frozenset({"kind", "reverted_event_hash"}), "undo action")
        return UndoAction(str(record["reverted_event_hash"]))
    if kind == "sign_off_accept":
        reject_unknown_fields(record, frozenset({"kind"}), "sign-off accept action")
        return SignOffAcceptAction()
    if kind == "sign_off_return":
        reject_unknown_fields(record, frozenset({"kind", "reason"}), "sign-off return action")
        require_fields(record, frozenset({"kind", "reason"}), "sign-off return action")
        return SignOffReturnAction(str(record["reason"]))
    raise ContractError("action kind is unknown", {"kind": kind})


def _editable(state: WindowReviewState) -> None:
    if isinstance(state.decision, SignedOff):
        raise ContractError("signed-off windows cannot be edited")


def apply_action(
    state: WindowReviewState,
    action: ReviewAction,
    *,
    actor: str,
    event_hash: str,
) -> WindowReviewState:
    """Return the next window state for a non-undo action."""

    actor_id = parse_non_empty_text(actor, "actor")
    parse_sha256(event_hash, "event_hash")
    if isinstance(action, UndoAction):
        raise ContractError("undo must be applied by the event store")
    if isinstance(action, SetDefectsAction):
        _editable(state)
        return WindowReviewState(
            decision=state.decision,
            defects=action.defects,
            last_review_actor=state.last_review_actor,
            last_review_event_hash=event_hash,
            revision=state.revision + 1,
        )
    if isinstance(action, SignOffAcceptAction):
        if not state.is_signoff_eligible_activity():
            raise ContractError("sign-off accept requires confirmed, corrected, or no-speech activity")
        if state.last_review_actor is None or state.last_review_event_hash is None:
            raise ContractError("sign-off accept requires a prior review event")
        if state.last_review_actor == actor_id:
            raise ContractError("independent sign-off requires a different actor")
        return WindowReviewState(
            decision=SignedOff(state.last_review_event_hash, actor_id),
            defects=state.defects,
            last_review_actor=state.last_review_actor,
            last_review_event_hash=event_hash,
            revision=state.revision + 1,
        )
    if isinstance(action, SignOffReturnAction):
        if not state.is_unsigned_review():
            raise ContractError("sign-off return requires an unsigned review decision")
        if state.last_review_actor is None or state.last_review_event_hash is None:
            raise ContractError("sign-off return requires a prior review event")
        if state.last_review_actor == actor_id:
            raise ContractError("independent sign-off requires a different actor")
        return WindowReviewState(
            decision=Returned(state.last_review_event_hash, actor_id, action.reason),
            defects=state.defects,
            last_review_actor=state.last_review_actor,
            last_review_event_hash=event_hash,
            revision=state.revision + 1,
        )
    _editable(state)
    if isinstance(state.decision, SignedOff):
        raise ContractError("signed-off windows cannot be reviewed again")
    if isinstance(action, ConfirmAction):
        decision: DecisionState = ConfirmedProposal()
    elif isinstance(action, CorrectAction):
        decision = Corrected(action.activity)
    elif isinstance(action, NoSpeechAction):
        decision = NoSpeech()
    elif isinstance(action, NeedsFollowUpAction):
        decision = NeedsFollowUp(action.reason, action.scope)
    elif isinstance(action, UncertainAction):
        decision = Uncertain(action.reason, action.scope)
    else:
        raise ContractError("action is not a review decision")
    return WindowReviewState(
        decision=decision,
        defects=state.defects,
        last_review_actor=actor_id,
        last_review_event_hash=event_hash,
        revision=state.revision + 1,
    )


def decision_activity(
    decision: DecisionState,
    proposal: GridProposal,
    *,
    signed_activity: tuple[ActivityInterval, ...] | None = None,
) -> tuple[ActivityInterval, ...] | None:
    """Return activity implied by a decision, if any."""

    if isinstance(decision, ConfirmedProposal):
        return proposal.intervals
    if isinstance(decision, Corrected):
        return decision.activity
    if isinstance(decision, NoSpeech):
        return ()
    if isinstance(decision, SignedOff):
        return signed_activity
    return None


def parse_source_transcript(value: object, *, speaker: SpeakerRole, source_sha256: str) -> list[dict[str, Any]]:
    """Parse a source transcript object and reject unknown fields.

    The full parent ``text`` field is accepted as a known schema member and is
    not returned. Callers retain only clipped window words.
    """

    record = require_mapping(value, "source transcript")
    reject_unknown_fields(record, TRANSCRIPT_TOP_FIELDS, "source transcript")
    require_fields(record, frozenset({"words"}), "source transcript")
    speaker_index = record.get("speaker_index")
    expected_index = "a" if speaker is SpeakerRole.SPEAKER_A else "b"
    if speaker_index is not None and speaker_index != expected_index:
        raise ContractError(
            "source transcript speaker_index does not match the member role",
            {"speaker_index": speaker_index, "role": speaker.value},
        )
    words = record["words"]
    if not isinstance(words, list):
        raise ContractError("source transcript words must be an array")
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(words):
        word = require_mapping(item, f"source transcript word {index}")
        reject_unknown_fields(word, TRANSCRIPT_WORD_FIELDS, f"source transcript word {index}")
        require_fields(word, REQUIRED_TRANSCRIPT_WORD_FIELDS, f"source transcript word {index}")
        text = word["word"]
        if not isinstance(text, str) or not text:
            raise ContractError("source transcript word text must be a non-empty string")
        word_type = word["type"]
        if word_type not in WORD_TYPES:
            raise ContractError("source transcript word type is unknown", {"type": word_type})
        start = word["start"]
        end = word["end"]
        if start is None or end is None:
            continue
        try:
            start_seconds = parse_finite_number(start, "word.start")
            end_seconds = parse_finite_number(end, "word.end")
        except ContractError:
            continue
        if end_seconds <= start_seconds:
            continue
        corrections = word.get("corrections_applied")
        if corrections is not None and not isinstance(corrections, bool):
            raise ContractError("corrections_applied must be a boolean")
        parsed.append(
            {
                "speaker": speaker,
                "text": text,
                "type": word_type,
                "start": start_seconds,
                "end": end_seconds,
                "source_transcript_sha256": source_sha256,
            }
        )
    return parsed


def clip_words_to_window(
    words: Sequence[Mapping[str, Any]],
    *,
    window_start_seconds: float,
    window_end_seconds: float,
) -> tuple[TranscriptWord, ...]:
    """Retain words that intersect a selected source-clock window."""

    retained: list[TranscriptWord] = []
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        if end <= window_start_seconds or start >= window_end_seconds:
            continue
        clipped_start = max(start, window_start_seconds)
        clipped_end = min(end, window_end_seconds)
        if clipped_end <= clipped_start:
            continue
        retained.append(
            TranscriptWord(
                speaker=word["speaker"],
                text=str(word["text"]),
                word_type=str(word["type"]),
                source_start_seconds=clipped_start,
                source_end_seconds=clipped_end,
                window_start_seconds=clipped_start - window_start_seconds,
                window_end_seconds=clipped_end - window_start_seconds,
                source_transcript_sha256=str(word["source_transcript_sha256"]),
            )
        )
    retained.sort(key=lambda item: (item.window_start_seconds, item.speaker.value, item.text))
    return tuple(retained)


def proposal_from_candidate(
    candidate: Mapping[str, Any],
    *,
    window_id: str,
    candidate_sha256: str,
) -> GridProposal:
    """Convert raw machine intervals into integer 20 ms frame occupancy."""

    record = require_mapping(candidate, "candidate annotation")
    if record.get("window_id") != window_id:
        raise ContractError("candidate annotation window_id does not match")
    activity = record.get("speaker_activity")
    if not isinstance(activity, list):
        raise ContractError("candidate annotation speaker_activity must be an array")
    intervals: list[ActivityInterval] = []
    for item in activity:
        item_record = require_mapping(item, "candidate speaker activity")
        speaker = parse_speaker(item_record.get("speaker_role") or item_record.get("speaker"), "speaker_role")
        raw_intervals = item_record.get("intervals")
        if not isinstance(raw_intervals, list):
            raise ContractError("candidate intervals must be an array")
        for raw in raw_intervals:
            raw_record = require_mapping(raw, "candidate interval")
            start = parse_finite_number(raw_record.get("start_seconds"), "start_seconds")
            end = parse_finite_number(raw_record.get("end_seconds"), "end_seconds")
            occupied = frames_intersecting_seconds(start, end)
            if occupied is None:
                continue
            intervals.append(ActivityInterval(speaker, occupied[0], occupied[1]))
    return GridProposal.from_intervals(
        window_id=window_id,
        candidate_sha256=candidate_sha256,
        intervals=intervals,
    )


def event_body_hash(payload: Mapping[str, Any]) -> str:
    """Hash a canonical event body that does not yet include event_hash."""

    if "event_hash" in payload:
        raise ContractError("event body must not include event_hash before hashing")
    return sha256_json(payload)


def canonical_event_bytes(payload: Mapping[str, Any]) -> str:
    """Return the canonical JSON encoding of an event."""

    return canonical_json(payload)


__all__ = [
    "ActivityInterval",
    "BoundedRangeScope",
    "Clear",
    "ClockAssessment",
    "ClockClear",
    "ClockMeasurementAbsent",
    "ClockMeasurementEntered",
    "ClockNotReviewed",
    "ClockUnresolved",
    "CONVERSION_POLICY",
    "ConfirmAction",
    "ConfirmedProposal",
    "CorrectAction",
    "Corrected",
    "DecisionState",
    "DefectSet",
    "FRAME_SECONDS",
    "GENESIS_HASH",
    "GridProposal",
    "NeedsFollowUp",
    "NeedsFollowUpAction",
    "NoSpeech",
    "NoSpeechAction",
    "NotReviewed",
    "Pending",
    "Returned",
    "ReviewAction",
    "SetDefectsAction",
    "SignOffAcceptAction",
    "SignOffReturnAction",
    "SignedOff",
    "SpeakerRole",
    "TranscriptWord",
    "Uncertain",
    "UncertainAction",
    "UndoAction",
    "UnresolvedDefect",
    "WINDOW_FRAME_COUNT",
    "WINDOW_SECONDS",
    "WholeWindowScope",
    "WindowReviewState",
    "WORD_TYPES",
    "apply_action",
    "clip_words_to_window",
    "decision_activity",
    "frames_intersecting_seconds",
    "normalize_intervals",
    "occupied_frames",
    "parse_action",
    "parse_decision",
    "parse_source_transcript",
    "parse_speaker",
    "proposal_from_candidate",
    "seconds_from_frame",
]
