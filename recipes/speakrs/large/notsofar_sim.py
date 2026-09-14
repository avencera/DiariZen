"""Bounded NOTSOFAR simulated activity and batch-state contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .acceptance import RttmInterval
from .errors import PreparationError


class SimulatedBatchState(str, Enum):
    """One explicit state in a bounded synthetic-source batch."""

    PLANNED = "planned"
    SOURCE_BOUND = "source-bound"
    CANONICAL_READY = "canonical-ready"
    REMOTE_VERIFIED = "remote-verified"
    COLD_RESTORED = "cold-restored"
    RAW_EVICTION_ELIGIBLE = "raw-eviction-eligible"


_NEXT_BATCH_STATE = {
    SimulatedBatchState.PLANNED: SimulatedBatchState.SOURCE_BOUND,
    SimulatedBatchState.SOURCE_BOUND: SimulatedBatchState.CANONICAL_READY,
    SimulatedBatchState.CANONICAL_READY: SimulatedBatchState.REMOTE_VERIFIED,
    SimulatedBatchState.REMOTE_VERIFIED: SimulatedBatchState.COLD_RESTORED,
    SimulatedBatchState.COLD_RESTORED: SimulatedBatchState.RAW_EVICTION_ELIGIBLE,
}


@dataclass(frozen=True, slots=True)
class SimulatedBatchLimits:
    """Local bounds for one streamed NOTSOFAR simulated archive."""

    maximum_working_bytes: int = 200 * 1024**3
    minimum_free_bytes: int = 100 * 1024**3

    def __post_init__(self) -> None:
        if self.maximum_working_bytes <= 0 or self.minimum_free_bytes < 0:
            raise ValueError("synthetic batch limits must be positive")

    def require_capacity(self, *, source_bytes: int, current_working_bytes: int, free_bytes: int) -> None:
        """Reject a batch before it can cross either local storage bound."""

        if min(source_bytes, current_working_bytes, free_bytes) < 0:
            raise PreparationError("synthetic batch byte counts must be non-negative")
        if current_working_bytes + source_bytes > self.maximum_working_bytes:
            raise PreparationError("synthetic batch would cross the working-space limit")
        if free_bytes - source_bytes < self.minimum_free_bytes:
            raise PreparationError("synthetic batch would cross the free-space reserve")


def advance_simulated_batch(current: SimulatedBatchState, target: SimulatedBatchState) -> SimulatedBatchState:
    """Advance exactly one proved state and reject skipped storage proof."""

    if _NEXT_BATCH_STATE.get(current) is not target:
        raise PreparationError(
            "invalid simulated batch state transition",
            {"current": current.value, "target": target.value},
        )

    return target


def notsofar_activity_intervals(
    utterance_id: str,
    scores: np.ndarray,
    *,
    sample_rate: int = 16_000,
) -> tuple[RttmInterval, ...]:
    """Convert official sample activity scores to exact local-speaker RTTM.

    Official score ``1`` means speaking, ``0`` means borderline transition,
    and ``-1`` means not speaking. The conversion keeps both speaking and
    borderline samples as activity so it does not delete uncertain speech.
    """

    values = np.asarray(scores)
    if not utterance_id or sample_rate <= 0:
        raise PreparationError("simulated activity requires an identity and positive sample rate")
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise PreparationError("simulated activity scores must have samples and speakers")
    if values.dtype != np.int8 or not np.isin(values, (-1, 0, 1)).all():
        raise PreparationError("simulated activity scores must contain only official int8 values")

    intervals = []
    for speaker_index in range(values.shape[1]):
        active = values[:, speaker_index] >= 0
        changes = np.flatnonzero(np.diff(np.pad(active.astype(np.int8), (1, 1))))
        for start_sample, end_sample in changes.reshape(-1, 2):
            start = int(start_sample) / sample_rate
            end = int(end_sample) / sample_rate
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise PreparationError("simulated activity conversion produced invalid bounds")
            intervals.append(RttmInterval(utterance_id, start, end, f"speaker-{speaker_index}"))

    return tuple(intervals)
