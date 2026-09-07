#!/usr/bin/env python3

"""Regression tests for the production target-capacity measurement."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from recipes.speakrs.large.errors import PreparationError  # noqa: E402
from recipes.speakrs.large.target_capacity import measure_capacity_loss  # noqa: E402


@dataclass(frozen=True)
class Interval:
    """Small interval value used without importing the acceptance owner."""

    recording_id: str
    start: float
    end: float
    speaker: str


def full_turn(speaker: str, start: float = 1.0, end: float = 9.0) -> Interval:
    """Build one interval that fills the measured eight-second window."""

    return Interval("rec", start, end, speaker)


class TargetCapacityTest(unittest.TestCase):
    """Check windowing, slot selection, and the actual Powerset conversion."""

    def test_same_speaker_annotations_are_unioned_before_overlap_counting(self):
        intervals = [full_turn("a"), Interval("rec", 3.0, 5.0, "a"), full_turn("b")]

        report = measure_capacity_loss(
            intervals,
            duration=11.0,
            chunk_seconds=8,
            max_overlap=2,
            local_slots=4,
            chunk_shift=8,
        )

        self.assertAlmostEqual(report.uem_speaker_seconds, 16.0)
        self.assertAlmostEqual(report.speaker_seconds, 2 * 399 * 0.020)
        self.assertEqual(report.overlap_lost_seconds, 0.0)
        self.assertEqual(report.actual_lost_seconds, 0.0)

    def test_eight_and_sixteen_second_grids_have_different_slot_loss(self):
        # each eight-second window has at most four speakers; the sixteen-
        # second window sees all five and must use production top-four slots
        intervals = [
            full_turn("a"),
            full_turn("b"),
            full_turn("c"),
            full_turn("d"),
            full_turn("e", 9.0, 17.0),
        ]

        eight = measure_capacity_loss(
            intervals,
            duration=19.0,
            chunk_seconds=8,
            max_overlap=4,
            local_slots=4,
            chunk_shift=8,
        )
        sixteen = measure_capacity_loss(
            intervals,
            duration=19.0,
            chunk_seconds=16,
            max_overlap=4,
            local_slots=4,
            chunk_shift=16,
        )

        self.assertEqual(eight.frames, 399)
        self.assertEqual(sixteen.frames, 799)
        self.assertEqual(eight.chunks, 2)
        self.assertEqual(sixteen.chunks, 1)
        self.assertAlmostEqual(eight.slot_lost_seconds, 0.0)
        self.assertGreater(sixteen.slot_lost_seconds, 0.0)
        # production discretization uses inclusive end indices, so touching
        # turns share one boundary frame in the raw diagnostic
        self.assertLessEqual(sixteen.overlap_lost_seconds, 0.020)
        self.assertAlmostEqual(sixteen.encoded_lost_seconds, 0.0)
        self.assertAlmostEqual(sixteen.actual_lost_seconds, sixteen.slot_lost_seconds)

    def test_powerset_roundtrip_reports_overlap_loss(self):
        intervals = [full_turn("a"), full_turn("b"), full_turn("c")]

        report = measure_capacity_loss(
            intervals,
            duration=11.0,
            chunk_seconds=8,
            max_overlap=2,
            local_slots=4,
            chunk_shift=8,
        )

        self.assertGreater(report.overlap_lost_seconds, 0.0)
        self.assertGreater(report.encoded_lost_seconds, 0.0)
        self.assertAlmostEqual(report.encoded_lost_seconds, report.encode_decode_lost_seconds)
        self.assertAlmostEqual(report.actual_lost_seconds, report.lost_seconds)
        self.assertAlmostEqual(report.actual_lost_seconds, report.encoded_lost_seconds)

    def test_short_explicit_uem_is_rejected_by_the_production_grid(self):
        with self.assertRaisesRegex(PreparationError, "production window"):
            measure_capacity_loss(
                [Interval("rec", 0.0, 10.0, "a")],
                duration=10.0,
                chunk_seconds=8,
                max_overlap=2,
                local_slots=4,
                uem=(2.0, 4.0),
            )

    def test_default_shifts_are_the_frozen_data_only_grids(self):
        eight = measure_capacity_loss(
            [full_turn("a")],
            duration=20.0,
            chunk_seconds=8,
            max_overlap=2,
            local_slots=4,
        )
        sixteen = measure_capacity_loss(
            [full_turn("a")],
            duration=40.0,
            chunk_seconds=16,
            max_overlap=2,
            local_slots=4,
        )

        self.assertEqual(eight.chunk_shift, 6)
        self.assertEqual(sixteen.chunk_shift, 12)


if __name__ == "__main__":
    unittest.main()
