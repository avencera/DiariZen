"""Strict LOTUSDIS TextGrid parsing and split admission."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping

from .acceptance import RttmInterval
from .errors import PreparationError


class LotusdisParseReason(str, Enum):
    """Typed reasons for rejecting a LOTUSDIS TextGrid boundary."""

    INVALID_TEXTGRID = "invalid-textgrid"
    MISSING_DURATION = "missing-duration"
    MISSING_INTERVAL_TIER = "missing-interval-tier"
    AMBIGUOUS_INTERVAL_TIER = "ambiguous-interval-tier"
    NON_FINITE_DURATION = "non-finite-duration"
    NEGATIVE_DURATION = "negative-duration"
    MALFORMED_INTERVAL = "malformed-interval"
    NON_FINITE_INTERVAL = "non-finite-interval"
    NEGATIVE_INTERVAL = "negative-interval"
    REVERSED_INTERVAL = "reversed-interval"
    OUT_OF_BOUNDS_INTERVAL = "out-of-bounds-interval"
    OVERLAPPING_INTERVALS = "overlapping-intervals"
    GAPPED_INTERVALS = "gapped-intervals"
    MALFORMED_MARK = "malformed-mark"
    DUPLICATE_SPEAKER = "duplicate-speaker"


class LotusdisTrainRejectionReason(str, Enum):
    """Typed reasons for excluding a publisher train parent."""

    MALFORMED_LABEL = "malformed-label"
    TOO_MANY_SPEAKERS = "too-many-speakers"
    HELDOUT_SPEAKER_LEAKAGE = "heldout-speaker-leakage"


class LotusdisTextGridError(PreparationError):
    """A sanitized, typed rejection from the LOTUSDIS TextGrid boundary."""

    def __init__(
        self,
        reason: LotusdisParseReason,
        parent_id: str,
        interval_index: int | None = None,
    ) -> None:
        self.reason = reason
        self.parent_id = parent_id
        self.interval_index = interval_index
        super().__init__(
            f"LOTUSDIS TextGrid rejected: {reason.value}",
            {"parent_id": parent_id, "interval_index": interval_index},
        )


@dataclass(frozen=True, slots=True)
class LotusdisParseResult:
    """Validated intervals, duration, and exact speakers for one parent."""

    parent_id: str
    duration: float
    intervals: tuple[RttmInterval, ...]
    speakers: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.parent_id, str) or not self.parent_id:
            raise ValueError("parent_id must be a non-empty string")
        if not math.isfinite(self.duration) or self.duration < 0:
            raise ValueError("duration must be finite and non-negative")
        intervals = tuple(self.intervals)
        seen: set[tuple[float, float, str]] = set()
        for interval in intervals:
            if not isinstance(interval, RttmInterval):
                raise ValueError("intervals must contain RttmInterval values")
            if interval.recording_id != self.parent_id:
                raise ValueError("interval recording IDs must match parent_id")
            if not isinstance(interval.speaker, str) or _SPEAKER_ID_RE.fullmatch(interval.speaker) is None:
                raise ValueError("interval speakers must match [FM][0-9]{2}")
            if (
                isinstance(interval.start, bool)
                or isinstance(interval.end, bool)
                or not isinstance(interval.start, (int, float))
                or not isinstance(interval.end, (int, float))
                or not math.isfinite(interval.start)
                or not math.isfinite(interval.end)
                or interval.start < 0
                or interval.end <= interval.start
                or interval.end > self.duration
            ):
                raise ValueError("interval bounds must be finite, ordered, and within duration")
            key = (float(interval.start), float(interval.end), interval.speaker)
            if key in seen:
                raise ValueError("duplicate speaker intervals are not allowed")
            seen.add(key)
        derived = frozenset(interval.speaker for interval in intervals)
        provided = frozenset(self.speakers)
        if provided and provided != derived:
            raise ValueError("speakers must equal the interval speaker set")
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "speakers", derived)


@dataclass(frozen=True, slots=True)
class LotusdisParentOutcome:
    """One successful or rejected parse outcome used by split admission."""

    parent_id: str
    result: LotusdisParseResult | None = None
    error: LotusdisTextGridError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parent_id, str) or not self.parent_id:
            raise ValueError("parent_id must be a non-empty string")
        if (self.result is None) == (self.error is None):
            raise ValueError("an outcome must contain exactly one result or error")
        if self.result is not None and self.result.parent_id != self.parent_id:
            raise ValueError("result parent_id must match outcome parent_id")
        if self.error is not None and self.error.parent_id != self.parent_id:
            raise ValueError("error parent_id must match outcome parent_id")

    @classmethod
    def accepted(cls, result: LotusdisParseResult) -> LotusdisParentOutcome:
        """Build a successful outcome from a validated parse result."""

        return cls(parent_id=result.parent_id, result=result)

    @classmethod
    def rejected(cls, error: LotusdisTextGridError) -> LotusdisParentOutcome:
        """Build a rejected outcome from a sanitized parser error."""

        return cls(parent_id=error.parent_id, error=error)

    @property
    def ok(self) -> bool:
        """Return whether the parent has a validated parse result."""

        return self.result is not None


@dataclass(frozen=True, slots=True)
class LotusdisParentRejection:
    """A deterministic train-parent exclusion and its typed reason."""

    parent_id: str
    reason: LotusdisTrainRejectionReason


@dataclass(frozen=True, slots=True)
class LotusdisStrictSubset:
    """A strict train subset with publisher dev/test membership preserved."""

    accepted_train_ids: tuple[str, ...]
    dev_parent_ids: tuple[str, ...]
    test_parent_ids: tuple[str, ...]
    rejected: tuple[LotusdisParentRejection, ...] = ()

    def __post_init__(self) -> None:
        accepted = tuple(self.accepted_train_ids)
        dev = tuple(self.dev_parent_ids)
        test = tuple(self.test_parent_ids)
        rejected = tuple(self.rejected)
        if len(set(accepted)) != len(accepted):
            raise ValueError("accepted train IDs must be unique")
        if len(set(dev)) != len(dev) or len(set(test)) != len(test):
            raise ValueError("heldout parent IDs must be unique")
        if set(accepted) & set(dev) or set(accepted) & set(test) or set(dev) & set(test):
            raise ValueError("train and heldout parent IDs must be disjoint")
        rejected_ids = tuple(item.parent_id for item in rejected)
        if len(set(rejected_ids)) != len(rejected_ids):
            raise ValueError("rejected parent IDs must be unique")
        if set(rejected_ids) & (set(accepted) | set(dev) | set(test)):
            raise ValueError("a parent cannot be both selected and rejected")
        object.__setattr__(self, "accepted_train_ids", accepted)
        object.__setattr__(self, "dev_parent_ids", dev)
        object.__setattr__(self, "test_parent_ids", test)
        object.__setattr__(self, "rejected", rejected)


@dataclass(frozen=True, slots=True)
class _TextGridInterval:
    """Raw interval fields retained only during boundary validation."""

    index: int
    start: float
    end: float
    mark: str


@dataclass(frozen=True, slots=True)
class _TextGridTier:
    """Parsed long-form TextGrid interval tier."""

    name: str
    intervals: tuple[_TextGridInterval, ...]


_ITEM_RE = re.compile(r"^\s*item\s*\[\s*\d+\s*\]:\s*$", re.MULTILINE)
_INTERVAL_RE = re.compile(r"^\s*intervals\s*\[\s*(\d+)\s*\]:\s*$", re.MULTILINE)
_FIELD_RE = re.compile(r"^\s*{field}\s*=\s*(.*?)\s*$", re.MULTILINE)
_SPEAKER_ID_RE = re.compile(r"^[FM][0-9]{2}$")
_SPEECH_MARK_RE = re.compile(r"^([FM]\d{2}(?:&[FM]\d{2})*),(.*)$", re.DOTALL)
_PARTITION_BOUNDARY_TOLERANCE = 1e-9
_PUBLISHER_TIER_NAMES = frozenset(
    {
        "speaker",
        "speakers",
        "speaker_id",
        "speaker ids",
        "spk",
        "diarization",
        "diarisation",
    }
)
_NON_SPEECH_MARKS = frozenset({"<n>", "<sil>", "<unk>", "<td>"})


def _error(reason: LotusdisParseReason, parent_id: str, interval_index: int | None = None) -> LotusdisTextGridError:
    return LotusdisTextGridError(reason, parent_id, interval_index)


def _source_text(source: str | Path, parent_id: str) -> str:
    if isinstance(source, Path):
        try:
            return source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise _error(LotusdisParseReason.INVALID_TEXTGRID, parent_id) from error
    if not isinstance(source, str):
        raise _error(LotusdisParseReason.INVALID_TEXTGRID, parent_id)
    try:
        candidate = Path(source)
    except (OSError, ValueError):
        return source
    if "\n" not in source and "\r" not in source and candidate.is_file():
        try:
            return candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise _error(LotusdisParseReason.INVALID_TEXTGRID, parent_id) from error
    return source


def _field_value(block: str, field: str, parent_id: str, interval_index: int | None) -> str:
    match = _FIELD_RE.pattern.format(field=re.escape(field))
    found = re.search(match, block, flags=re.MULTILINE)
    if found is None:
        raise _error(LotusdisParseReason.MALFORMED_INTERVAL, parent_id, interval_index)
    return found.group(1).strip()


def _quoted_value(value: str, parent_id: str, interval_index: int | None) -> str:
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        raise _error(LotusdisParseReason.MALFORMED_INTERVAL, parent_id, interval_index)
    if value.startswith('"""') and value.endswith('"""') and len(value) >= 6:
        body = value[3:-3]
    else:
        body = value[1:-1]
    return body.replace('""', '"')


def _number(value: str, parent_id: str, interval_index: int | None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise _error(LotusdisParseReason.MALFORMED_INTERVAL, parent_id, interval_index) from error
    if not math.isfinite(parsed):
        raise _error(LotusdisParseReason.NON_FINITE_INTERVAL, parent_id, interval_index)
    return parsed


def _parse_interval_tiers(text: str, parent_id: str) -> tuple[_TextGridTier, ...]:
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        raise _error(LotusdisParseReason.MISSING_INTERVAL_TIER, parent_id)
    tiers: list[_TextGridTier] = []
    for item_index, item_match in enumerate(matches, start=1):
        end = matches[item_index].start() if item_index < len(matches) else len(text)
        block = text[item_match.end() : end]
        class_match = re.search(r'^\s*class\s*=\s*"?([^"\r\n]+?)"?\s*$', block, flags=re.MULTILINE)
        if class_match is None or class_match.group(1).strip().casefold() != "intervaltier":
            continue
        name_match = re.search(r'^\s*name\s*=\s*"([^"]*)"\s*$', block, flags=re.MULTILINE)
        if name_match is None:
            raise _error(LotusdisParseReason.MALFORMED_INTERVAL, parent_id)
        interval_matches = list(_INTERVAL_RE.finditer(block))
        declared_match = re.search(r"^\s*intervals\s*:\s*size\s*=\s*(\d+)\s*$", block, flags=re.MULTILINE)
        if declared_match is not None and int(declared_match.group(1)) != len(interval_matches):
            raise _error(LotusdisParseReason.MALFORMED_INTERVAL, parent_id)
        intervals: list[_TextGridInterval] = []
        for position, interval_match in enumerate(interval_matches):
            interval_end = (
                interval_matches[position + 1].start() if position + 1 < len(interval_matches) else len(block)
            )
            interval_block = block[interval_match.end() : interval_end]
            index = int(interval_match.group(1)) - 1
            if index < 0:
                raise _error(LotusdisParseReason.MALFORMED_INTERVAL, parent_id, position)
            start = _number(_field_value(interval_block, "xmin", parent_id, index), parent_id, index)
            end_value = _number(_field_value(interval_block, "xmax", parent_id, index), parent_id, index)
            mark = _quoted_value(_field_value(interval_block, "text", parent_id, index), parent_id, index)
            intervals.append(_TextGridInterval(index=index, start=start, end=end_value, mark=mark))
        tiers.append(_TextGridTier(name=name_match.group(1).strip(), intervals=tuple(intervals)))
    if not tiers:
        raise _error(LotusdisParseReason.MISSING_INTERVAL_TIER, parent_id)
    return tuple(tiers)


def _looks_like_publisher_tier(tier: _TextGridTier) -> bool:
    if tier.name.casefold() in _PUBLISHER_TIER_NAMES:
        return True
    return any(
        mark.mark.strip() in _NON_SPEECH_MARKS or _SPEECH_MARK_RE.fullmatch(mark.mark.strip()) is not None
        for mark in tier.intervals
    )


def _choose_tier(tiers: tuple[_TextGridTier, ...], parent_id: str) -> _TextGridTier:
    if len(tiers) == 1:
        return tiers[0]
    named = [tier for tier in tiers if tier.name.casefold() in _PUBLISHER_TIER_NAMES]
    if len(named) == 1:
        return named[0]
    candidates = [tier for tier in tiers if _looks_like_publisher_tier(tier)]
    if len(candidates) == 1:
        return candidates[0]
    raise _error(LotusdisParseReason.AMBIGUOUS_INTERVAL_TIER, parent_id)


def _global_duration(text: str, parent_id: str) -> float:
    item_start = re.search(r"^\s*item\s*\[", text, flags=re.MULTILINE)
    header = text if item_start is None else text[: item_start.start()]
    match = re.search(r"^\s*xmax\s*=\s*(\S+)\s*$", header, flags=re.MULTILINE)
    if match is None:
        raise _error(LotusdisParseReason.MISSING_DURATION, parent_id)
    try:
        duration = float(match.group(1))
    except ValueError as error:
        raise _error(LotusdisParseReason.MISSING_DURATION, parent_id) from error
    if not math.isfinite(duration):
        raise _error(LotusdisParseReason.NON_FINITE_DURATION, parent_id)
    if duration < 0:
        raise _error(LotusdisParseReason.NEGATIVE_DURATION, parent_id)
    return duration


def _check_partition(intervals: tuple[_TextGridInterval, ...], duration: float, parent_id: str) -> None:
    if not intervals:
        if duration == 0:
            return
        raise _error(LotusdisParseReason.GAPPED_INTERVALS, parent_id, 0)
    previous_end: float | None = None
    for position, interval in enumerate(intervals):
        if not math.isfinite(interval.start) or not math.isfinite(interval.end):
            raise _error(LotusdisParseReason.NON_FINITE_INTERVAL, parent_id, position)
        if interval.start < 0 or interval.end < 0:
            raise _error(LotusdisParseReason.NEGATIVE_INTERVAL, parent_id, position)
        if interval.start > duration or interval.end > duration:
            raise _error(LotusdisParseReason.OUT_OF_BOUNDS_INTERVAL, parent_id, position)
        if interval.end <= interval.start:
            raise _error(LotusdisParseReason.REVERSED_INTERVAL, parent_id, position)
        if previous_end is None:
            if interval.start > _PARTITION_BOUNDARY_TOLERANCE:
                raise _error(LotusdisParseReason.GAPPED_INTERVALS, parent_id, position)
        elif interval.start < previous_end - _PARTITION_BOUNDARY_TOLERANCE:
            raise _error(LotusdisParseReason.OVERLAPPING_INTERVALS, parent_id, position)
        elif interval.start > previous_end + _PARTITION_BOUNDARY_TOLERANCE:
            raise _error(LotusdisParseReason.GAPPED_INTERVALS, parent_id, position)
        previous_end = interval.end
    if not math.isclose(previous_end, duration, abs_tol=_PARTITION_BOUNDARY_TOLERANCE, rel_tol=0.0):
        reason = (
            LotusdisParseReason.GAPPED_INTERVALS
            if previous_end < duration
            else LotusdisParseReason.OUT_OF_BOUNDS_INTERVAL
        )
        raise _error(reason, parent_id, len(intervals) - 1)


def _intervals_and_speakers(
    raw_intervals: tuple[_TextGridInterval, ...], parent_id: str
) -> tuple[tuple[RttmInterval, ...], frozenset[str]]:
    intervals: list[RttmInterval] = []
    speakers: set[str] = set()
    for position, raw in enumerate(raw_intervals):
        mark = raw.mark.strip()
        if not mark or mark in _NON_SPEECH_MARKS:
            continue
        match = _SPEECH_MARK_RE.fullmatch(mark)
        if match is None or not match.group(2).strip():
            raise _error(LotusdisParseReason.MALFORMED_MARK, parent_id, position)
        ids = match.group(1).split("&")
        if len(set(ids)) != len(ids):
            raise _error(LotusdisParseReason.DUPLICATE_SPEAKER, parent_id, position)
        for speaker in ids:
            speakers.add(speaker)
            intervals.append(
                RttmInterval(
                    recording_id=parent_id,
                    start=raw.start,
                    end=raw.end,
                    speaker=speaker,
                )
            )
    return tuple(intervals), frozenset(speakers)


def parse_lotusdis_textgrid(source: str | Path, parent_id: str) -> LotusdisParseResult:
    """Parse one publisher TextGrid with strict labels and full partition checks."""

    if not isinstance(parent_id, str) or not parent_id:
        raise _error(LotusdisParseReason.INVALID_TEXTGRID, str(parent_id))
    text = _source_text(source, parent_id).replace("\r\n", "\n").replace("\r", "\n")
    duration = _global_duration(text, parent_id)
    tier = _choose_tier(_parse_interval_tiers(text, parent_id), parent_id)
    _check_partition(tier.intervals, duration, parent_id)
    intervals, speakers = _intervals_and_speakers(tier.intervals, parent_id)
    return LotusdisParseResult(parent_id=parent_id, duration=duration, intervals=intervals, speakers=speakers)


def _normalise_outcome(
    parent_id: str,
    outcomes: Mapping[str, LotusdisParentOutcome | LotusdisParseResult | LotusdisTextGridError],
) -> LotusdisParentOutcome:
    value = outcomes.get(parent_id)
    if isinstance(value, LotusdisParentOutcome):
        if value.parent_id != parent_id:
            raise PreparationError("LOTUSDIS parse outcome parent identity mismatch", {"parent_id": parent_id})
        return value
    if isinstance(value, LotusdisParseResult):
        if value.parent_id != parent_id:
            raise PreparationError("LOTUSDIS parse outcome parent identity mismatch", {"parent_id": parent_id})
        return LotusdisParentOutcome.accepted(value)
    if isinstance(value, LotusdisTextGridError):
        if value.parent_id != parent_id:
            raise PreparationError("LOTUSDIS parse outcome parent identity mismatch", {"parent_id": parent_id})
        return LotusdisParentOutcome.rejected(value)
    error = LotusdisTextGridError(LotusdisParseReason.INVALID_TEXTGRID, parent_id)
    return LotusdisParentOutcome.rejected(error)


def select_lotusdis_strict_subset(
    train_parent_ids: Iterable[str],
    dev_parent_ids: Iterable[str],
    test_parent_ids: Iterable[str],
    outcomes: Mapping[str, LotusdisParentOutcome | LotusdisParseResult | LotusdisTextGridError],
    *,
    speaker_graph: Mapping[str, Iterable[str]],
) -> LotusdisStrictSubset:
    """Select a deterministic train subset without changing publisher splits.

    Held-out parents need authoritative global speaker identities, but their
    activity labels are not training inputs. Keeping that identity proof
    separate avoids requiring malformed held-out activity to become valid
    training labels.
    """

    def sorted_split(parent_ids: Iterable[str], split: str) -> tuple[str, ...]:
        values = tuple(parent_ids)
        if len(values) != len(set(values)):
            raise PreparationError("LOTUSDIS publisher split contains duplicate parents", {"split": split})
        return tuple(sorted(values))

    train = sorted_split(train_parent_ids, "train")
    dev = sorted_split(dev_parent_ids, "dev")
    test = sorted_split(test_parent_ids, "test")
    expected = set(train) | set(dev) | set(test)
    missing = sorted(set(train) - set(outcomes))
    if missing:
        raise PreparationError("LOTUSDIS train parse outcomes are incomplete", {"missing_parent_ids": missing})
    if set(train) & (set(dev) | set(test)) or set(dev) & set(test):
        raise PreparationError("LOTUSDIS publisher splits overlap")

    missing_speaker_parents = sorted(expected - set(speaker_graph))
    if missing_speaker_parents:
        raise PreparationError(
            "LOTUSDIS speaker graph is incomplete",
            {"missing_parent_ids": missing_speaker_parents},
        )
    normalized_graph: dict[str, frozenset[str]] = {}
    for parent_id in sorted(expected):
        speakers = frozenset(speaker_graph[parent_id])
        if not speakers or any(_SPEAKER_ID_RE.fullmatch(speaker) is None for speaker in speakers):
            raise PreparationError("LOTUSDIS speaker graph contains invalid identities", {"parent_id": parent_id})
        normalized_graph[parent_id] = speakers

    heldout_speakers = set().union(*(normalized_graph[parent_id] for parent_id in (*dev, *test)))

    accepted: list[str] = []
    rejected: list[LotusdisParentRejection] = []
    for parent_id in train:
        outcome = _normalise_outcome(parent_id, outcomes)
        if outcome.result is None:
            rejected.append(LotusdisParentRejection(parent_id, LotusdisTrainRejectionReason.MALFORMED_LABEL))
            continue
        if outcome.result.speakers != normalized_graph[parent_id]:
            raise PreparationError("LOTUSDIS train labels differ from the speaker graph", {"parent_id": parent_id})
        if len(normalized_graph[parent_id]) > 3:
            rejected.append(LotusdisParentRejection(parent_id, LotusdisTrainRejectionReason.TOO_MANY_SPEAKERS))
            continue
        if normalized_graph[parent_id] & heldout_speakers:
            rejected.append(LotusdisParentRejection(parent_id, LotusdisTrainRejectionReason.HELDOUT_SPEAKER_LEAKAGE))
            continue
        accepted.append(parent_id)
    return LotusdisStrictSubset(tuple(accepted), dev, test, tuple(rejected))


__all__ = [
    "LotusdisParseReason",
    "LotusdisParseResult",
    "LotusdisParentOutcome",
    "LotusdisParentRejection",
    "LotusdisStrictSubset",
    "LotusdisTextGridError",
    "LotusdisTrainRejectionReason",
    "parse_lotusdis_textgrid",
    "select_lotusdis_strict_subset",
]
