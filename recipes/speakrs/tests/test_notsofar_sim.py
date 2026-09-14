from __future__ import annotations

import numpy as np
import pytest

from recipes.speakrs.large.acceptance import RttmInterval
from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.notsofar_sim import (
    SimulatedBatchLimits,
    SimulatedBatchState,
    advance_simulated_batch,
    notsofar_activity_intervals,
)


def test_activity_conversion_keeps_borderline_samples_without_joining_silence() -> None:
    scores = np.array([[-1, 1], [0, 1], [1, -1], [-1, -1], [1, 0]], dtype=np.int8)

    intervals = notsofar_activity_intervals("utterance", scores, sample_rate=1)

    assert intervals == (
        RttmInterval("utterance", 1.0, 3.0, "speaker-0"),
        RttmInterval("utterance", 4.0, 5.0, "speaker-0"),
        RttmInterval("utterance", 0.0, 2.0, "speaker-1"),
        RttmInterval("utterance", 4.0, 5.0, "speaker-1"),
    )


def test_activity_conversion_rejects_unknown_scores() -> None:
    with pytest.raises(PreparationError, match="official int8 values"):
        notsofar_activity_intervals("utterance", np.array([[2]], dtype=np.int8))


def test_batch_state_cannot_skip_remote_and_restore_proof() -> None:
    assert (
        advance_simulated_batch(SimulatedBatchState.PLANNED, SimulatedBatchState.SOURCE_BOUND)
        is SimulatedBatchState.SOURCE_BOUND
    )
    with pytest.raises(PreparationError, match="state transition"):
        advance_simulated_batch(SimulatedBatchState.CANONICAL_READY, SimulatedBatchState.RAW_EVICTION_ELIGIBLE)


def test_batch_limits_preserve_working_and_free_space() -> None:
    limits = SimulatedBatchLimits(maximum_working_bytes=200, minimum_free_bytes=100)
    limits.require_capacity(source_bytes=20, current_working_bytes=100, free_bytes=130)

    with pytest.raises(PreparationError, match="working-space"):
        limits.require_capacity(source_bytes=101, current_working_bytes=100, free_bytes=300)
    with pytest.raises(PreparationError, match="free-space"):
        limits.require_capacity(source_bytes=31, current_working_bytes=100, free_bytes=130)
