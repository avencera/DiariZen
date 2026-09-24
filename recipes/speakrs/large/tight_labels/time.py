"""Exact nanosecond times parsed from decimal seconds text."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Union


NANOSECONDS_PER_SECOND = 1_000_000_000
MIN_GAP_SECONDS = 0.200
MIN_GAP_NS = 200_000_000
SHORT_GAP_NS = 300_000_000
MID_GAP_NS = 1_000_000_000
COVERAGE_RATIO = Fraction(3, 10)

SecondsLike = Union[str, int, float, Decimal]


def parse_seconds_to_ns(value: SecondsLike) -> int:
    """Parse a non-negative seconds value into an integral nanosecond count.

    Decimal text is the production path, matching the trainer. Whole ints are
    whole seconds. Floats are accepted for tests through their decimal string.
    """

    if isinstance(value, bool):
        raise ValueError("time must be a decimal number of seconds")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("time must be non-negative")
        return value * NANOSECONDS_PER_SECOND
    if isinstance(value, float):
        if value < 0 or value != value or value == float("inf"):
            raise ValueError("time must be a non-negative finite decimal")
        decimal = Decimal(str(value))
    elif isinstance(value, Decimal):
        decimal = value
    elif isinstance(value, str):
        if not value.strip() or value != value.strip():
            raise ValueError("time must be a decimal string")
        try:
            decimal = Decimal(value)
        except (InvalidOperation, ValueError) as error:
            raise ValueError("time must be a finite decimal") from error
    else:
        raise ValueError("time must be a decimal number of seconds")
    if not decimal.is_finite() or decimal < 0:
        raise ValueError("time must be a non-negative finite decimal")
    nanoseconds = decimal * Decimal(NANOSECONDS_PER_SECOND)
    if nanoseconds != nanoseconds.to_integral_value():
        raise ValueError("time has sub-nanosecond precision")
    return int(nanoseconds)


def format_ns(value: int) -> str:
    """Format nanoseconds as seconds with nine decimal places."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("time must be an integer nanosecond count")
    if value < 0:
        raise ValueError("time cannot be negative")
    seconds, nanoseconds = divmod(value, NANOSECONDS_PER_SECOND)
    return f"{seconds}.{nanoseconds:09d}"


def ns_to_seconds(value: int) -> float:
    """Convert nanoseconds to a float second count for reports."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("time must be an integer nanosecond count")
    return value / NANOSECONDS_PER_SECOND


def as_coverage_ratio(value: Fraction | str | float | Decimal) -> Fraction:
    """Return a rational coverage threshold."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    raise ValueError("coverage ratio must be a fraction")


def coverage_meets(covered_ns: int, total_ns: int, min_coverage: Fraction | str | float | Decimal) -> bool:
    """Return True when covered_ns / total_ns is at least min_coverage."""

    if total_ns <= 0:
        return False
    ratio = as_coverage_ratio(min_coverage)
    return covered_ns * ratio.denominator >= total_ns * ratio.numerator
