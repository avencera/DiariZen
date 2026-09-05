#!/usr/bin/env python3

"""Verify a dscore result against a declared DER expectation."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DerExpectation:
    """A reference DER and the largest accepted absolute difference."""

    expected: float
    tolerance: float

    def accepts(self, actual: float) -> bool:
        """Return whether an actual DER is inside this expectation."""

        difference = abs(actual - self.expected)
        return difference < self.tolerance or math.isclose(difference, self.tolerance, rel_tol=1e-12, abs_tol=1e-12)


def parse_overall_der(result_path: Path) -> float:
    """Read the aggregate DER from one dscore text result."""

    overall_rows = [line for line in result_path.read_text().splitlines() if line.startswith("*** OVERALL ***")]
    if len(overall_rows) != 1:
        raise ValueError(f"expected one OVERALL row in {result_path}, found {len(overall_rows)}")

    fields = overall_rows[0].split()
    if len(fields) < 4:
        raise ValueError(f"invalid OVERALL row in {result_path}")

    der = float(fields[3])
    if not math.isfinite(der) or der < 0:
        raise ValueError(f"invalid DER in {result_path}: {der}")

    return der


def main() -> int:
    """Run the DER expectation check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=float)
    parser.add_argument("--tolerance", required=True, type=float)
    args = parser.parse_args()

    if args.tolerance < 0:
        parser.error("--tolerance must not be negative")

    actual = parse_overall_der(args.result)
    expectation = DerExpectation(args.expected, args.tolerance)
    difference = actual - expectation.expected
    if not expectation.accepts(actual):
        print(
            f"DER check failed: actual={actual:.2f} expected={expectation.expected:.2f} "
            f"difference={difference:+.2f} tolerance={expectation.tolerance:.2f}"
        )
        return 1

    print(
        f"DER check passed: actual={actual:.2f} expected={expectation.expected:.2f} "
        f"difference={difference:+.2f} tolerance={expectation.tolerance:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
