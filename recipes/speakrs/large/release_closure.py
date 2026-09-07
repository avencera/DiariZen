"""Validate and aggregate the capacity evidence used by release closure."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import CAPACITY_LOSS_LIMIT
from .errors import PreparationError


PROFILE_KEYS = ((8, 2, 4), (8, 4, 4), (16, 2, 4), (16, 4, 4))
REQUIRED_FIELDS = ("speaker_seconds", "lost_seconds")
ADDITIVE_FIELDS = (
    "speaker_seconds",
    "lost_seconds",
    "slot_lost_seconds",
    "encoded_lost_seconds",
    "overlap_limit_excess_seconds",
    "actual_lost_seconds",
    "encode_decode_lost_seconds",
    "post_slot_overlap_lost_seconds",
    "uem_speaker_seconds",
    "chunks",
    "frames",
)
INTEGER_FIELDS = {"chunks", "frames", "chunk_shift", "model_num_frames"}
POSITIVE_INTEGER_FIELDS = {"chunk_shift", "model_num_frames"}
METADATA_FIELDS = ("chunk_shift", "model_num_frames", "model_rf_duration", "model_rf_step")


def _error(message: str, label: str, **details: object) -> PreparationError:
    """Build a capacity error with a stable evidence label."""

    return PreparationError(message, {"label": label, **details})


def _number(value: Any, *, label: str, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error("capacity totals must be finite numbers", label, field=field)
    try:
        number = float(value)
    except OverflowError:
        raise _error("capacity totals must be finite numbers", label, field=field) from None
    if not math.isfinite(number):
        raise _error("capacity totals must be finite numbers", label, field=field)
    if number < minimum:
        raise _error("capacity totals cannot be negative", label, field=field)
    return number


def _integer(value: Any, *, label: str, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error("capacity profile metadata is invalid", label, field=field)
    if value < minimum:
        raise _error("capacity profile metadata is invalid", label, field=field)
    return value


def _profile_key(value: Mapping[str, Any], *, label: str) -> tuple[int, int, int]:
    values = tuple(
        _integer(value.get(name), label=label, field=name, minimum=1)
        for name in ("chunk_seconds", "max_overlap", "local_slots")
    )
    if values not in PROFILE_KEYS:
        raise _error("capacity profiles must cover the frozen 8/16 by 2/4 grid", label)
    return values


@dataclass(frozen=True)
class CapacityProfile:
    """One validated capacity profile with exact weighted totals."""

    chunk_seconds: int
    max_overlap: int
    local_slots: int
    speaker_seconds: float
    lost_seconds: float
    loss_fraction: float
    admitted: bool
    diagnostics: tuple[tuple[str, float | int], ...] = ()

    @property
    def key(self) -> tuple[int, int, int]:
        """Return the immutable profile identity."""

        return self.chunk_seconds, self.max_overlap, self.local_slots

    def value(self, field: str) -> float | int | None:
        """Return one optional diagnostic value."""

        return dict(self.diagnostics).get(field)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe profile row."""

        row: dict[str, object] = {
            "chunk_seconds": self.chunk_seconds,
            "max_overlap": self.max_overlap,
            "local_slots": self.local_slots,
            "speaker_seconds": self.speaker_seconds,
            "lost_seconds": self.lost_seconds,
            "loss_fraction": self.loss_fraction,
            "admitted": self.admitted,
        }
        row.update(dict(self.diagnostics))
        return row


@dataclass(frozen=True)
class CapacityClosure:
    """The four validated profiles for one batch or one source."""

    profiles: tuple[CapacityProfile, ...]

    def __post_init__(self) -> None:
        if tuple(profile.key for profile in self.profiles) != PROFILE_KEYS:
            raise ValueError("capacity closure profiles must use the frozen profile order")

    def as_list(self) -> list[dict[str, object]]:
        """Return all profiles, including failed profiles, in frozen order."""

        return [profile.as_dict() for profile in self.profiles]

    @property
    def admitted(self) -> tuple[CapacityProfile, ...]:
        """Return the profiles that pass the weighted threshold."""

        return tuple(profile for profile in self.profiles if profile.admitted)


def _parse_declared_profiles(value: Any, closure: CapacityClosure, *, label: str) -> None:
    if not isinstance(value, list):
        raise _error("portable manifest admitted profiles are missing", label)
    declared: list[tuple[int, int, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise _error("portable manifest admitted profile is invalid", label, index=index)
        try:
            chunk_seconds = _integer(item.get("chunk_seconds"), label=label, field="chunk_seconds", minimum=1)
            max_overlap = _integer(item.get("max_overlap"), label=label, field="max_overlap", minimum=1)
            local_slots = _integer(item.get("local_slots", 4), label=label, field="local_slots", minimum=1)
        except PreparationError:
            raise _error("portable manifest admitted profile is invalid", label, index=index) from None
        key = (chunk_seconds, max_overlap, local_slots)
        if key not in PROFILE_KEYS or key in declared:
            raise _error("portable manifest admitted profiles are inconsistent", label, index=index)
        declared.append(key)
        profile = next(profile for profile in closure.profiles if profile.key == key)
        if not profile.admitted:
            raise _error("portable manifest admitted profiles are inconsistent", label, index=index)
    expected = {profile.key for profile in closure.admitted}
    if set(declared) != expected:
        raise _error("portable manifest admitted profiles do not match capacity", label)


def parse_capacity_manifest(value: Mapping[str, Any], *, label: str) -> CapacityClosure:
    """Parse all four measured profiles and verify their admission claims."""

    capacity = value.get("capacity") if isinstance(value, Mapping) else None
    if not isinstance(capacity, list) or len(capacity) != len(PROFILE_KEYS):
        raise _error("portable manifest must contain exactly four capacity profiles", label)

    by_key: dict[tuple[int, int, int], CapacityProfile] = {}
    for index, item in enumerate(capacity):
        if not isinstance(item, Mapping):
            raise _error("capacity profile is not an object", label, index=index)
        profile_label = f"{label}[{index}]"
        key = _profile_key(item, label=profile_label)
        if key in by_key:
            raise _error("capacity profiles are missing or duplicated", label)
        normalized: dict[str, float | int] = {}
        for field in REQUIRED_FIELDS:
            if field not in item:
                raise _error("capacity totals are missing", profile_label, field=field)
            normalized[field] = _number(item[field], label=profile_label, field=field)
        for field in ADDITIVE_FIELDS:
            if field in REQUIRED_FIELDS or field not in item:
                continue
            if field in INTEGER_FIELDS:
                normalized[field] = _integer(item[field], label=profile_label, field=field, minimum=0)
            else:
                normalized[field] = _number(item[field], label=profile_label, field=field)
        for field in METADATA_FIELDS:
            if field not in item:
                continue
            if field in INTEGER_FIELDS:
                normalized[field] = _integer(
                    item[field],
                    label=profile_label,
                    field=field,
                    minimum=1 if field in POSITIVE_INTEGER_FIELDS else 0,
                )
            else:
                normalized[field] = _number(item[field], label=profile_label, field=field, minimum=0.0)

        speaker_seconds = float(normalized["speaker_seconds"])
        lost_seconds = float(normalized["lost_seconds"])
        if speaker_seconds <= 0:
            raise _error("capacity speaker-seconds must be positive", profile_label)
        expected_fraction = lost_seconds / speaker_seconds
        raw_fraction = item.get("loss_fraction")
        if (
            isinstance(raw_fraction, bool)
            or not isinstance(raw_fraction, (int, float))
            or not math.isfinite(float(raw_fraction))
            or not math.isclose(float(raw_fraction), expected_fraction, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise _error("capacity loss fraction is inconsistent with its weighted totals", profile_label)
        if expected_fraction > 1.0 + 1e-12:
            raise _error("capacity loss fraction cannot exceed one", profile_label)
        raw_admitted = item.get("admitted")
        expected_admitted = expected_fraction <= CAPACITY_LOSS_LIMIT
        if not isinstance(raw_admitted, bool) or raw_admitted is not expected_admitted:
            raise _error("capacity admission is inconsistent with its weighted totals", profile_label)
        actual_lost = normalized.get("actual_lost_seconds")
        if actual_lost is not None and not math.isclose(
            float(actual_lost), lost_seconds, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise _error("capacity actual loss is inconsistent with lost speaker-seconds", profile_label)
        diagnostics = tuple(sorted(normalized.items()))
        by_key[key] = CapacityProfile(
            chunk_seconds=key[0],
            max_overlap=key[1],
            local_slots=key[2],
            speaker_seconds=speaker_seconds,
            lost_seconds=lost_seconds,
            loss_fraction=expected_fraction,
            admitted=expected_admitted,
            diagnostics=diagnostics,
        )
    if set(by_key) != set(PROFILE_KEYS):
        raise _error("capacity profiles must cover the frozen 8/16 by 2/4 grid", label)
    closure = CapacityClosure(tuple(by_key[key] for key in PROFILE_KEYS))
    declared = value.get("admitted_profiles")
    if declared is None:
        declared = value.get("provisional_profiles")
    _parse_declared_profiles(declared, closure, label=label)
    return closure


def aggregate_capacity(
    records: Sequence[CapacityClosure | Sequence[CapacityProfile]], *, label: str
) -> CapacityClosure:
    """Sum profile totals before applying the loss threshold."""

    if not records:
        raise _error("release source has no capacity measurements", label)
    normalized_records: list[tuple[CapacityProfile, ...]] = []
    for record in records:
        profiles = record.profiles if isinstance(record, CapacityClosure) else tuple(record)
        if len(profiles) != len(PROFILE_KEYS) or tuple(profile.key for profile in profiles) != PROFILE_KEYS:
            raise _error("capacity measurements have invalid profile identities", label)
        normalized_records.append(profiles)

    aggregate: list[CapacityProfile] = []
    for profile_index, key in enumerate(PROFILE_KEYS):
        rows = [record[profile_index] for record in normalized_records]
        diagnostics: dict[str, float | int] = {}
        for field in ADDITIVE_FIELDS:
            values = [row.value(field) for row in rows]
            if field in REQUIRED_FIELDS and any(value is None for value in values):
                raise _error("capacity totals are missing", label, field=field)
            if any(value is None for value in values):
                continue
            if field in INTEGER_FIELDS:
                diagnostics[field] = sum(int(value) for value in values)
            else:
                try:
                    total = sum(float(value) for value in values)
                except OverflowError:
                    raise _error("aggregated capacity totals are invalid", label, field=field) from None
                if not math.isfinite(total):
                    raise _error("aggregated capacity totals are invalid", label, field=field)
                diagnostics[field] = total
        for field in METADATA_FIELDS:
            values = [row.value(field) for row in rows]
            if any(value is None for value in values):
                continue
            first = values[0]
            if any(
                not math.isclose(float(value), float(first), rel_tol=1e-12, abs_tol=1e-12)
                if isinstance(value, (int, float)) and isinstance(first, (int, float))
                else value != first
                for value in values[1:]
            ):
                raise _error("capacity profile metadata differs across batches", label, field=field)
            diagnostics[field] = first  # type: ignore[assignment]
        speaker_seconds = float(diagnostics["speaker_seconds"])
        lost_seconds = float(diagnostics["lost_seconds"])
        if speaker_seconds <= 0:
            raise _error("aggregated capacity totals are invalid", label)
        loss_fraction = lost_seconds / speaker_seconds
        if not math.isfinite(loss_fraction):
            raise _error("aggregated capacity totals are invalid", label)
        admitted = loss_fraction <= CAPACITY_LOSS_LIMIT
        if "actual_lost_seconds" in diagnostics:
            diagnostics["actual_lost_seconds"] = lost_seconds
        aggregate.append(
            CapacityProfile(
                chunk_seconds=key[0],
                max_overlap=key[1],
                local_slots=key[2],
                speaker_seconds=speaker_seconds,
                lost_seconds=lost_seconds,
                loss_fraction=loss_fraction,
                admitted=admitted,
                diagnostics=tuple(sorted(diagnostics.items())),
            )
        )
    return CapacityClosure(tuple(aggregate))


def common_admitted_profiles(capacity_by_source: Mapping[str, CapacityClosure]) -> list[dict[str, object]]:
    """Return only profiles admitted by every required source."""

    if not capacity_by_source:
        raise _error("release has no required source capacity", "release")
    common = set(PROFILE_KEYS)
    for closure in capacity_by_source.values():
        common.intersection_update(profile.key for profile in closure.admitted)
    first = capacity_by_source[sorted(capacity_by_source)[0]]
    return [
        {
            "chunk_seconds": profile.chunk_seconds,
            "max_overlap": profile.max_overlap,
            "local_slots": profile.local_slots,
        }
        for profile in first.profiles
        if profile.key in common
    ]


__all__ = [
    "CapacityClosure",
    "CapacityProfile",
    "aggregate_capacity",
    "common_admitted_profiles",
    "parse_capacity_manifest",
]
