"""Typed accumulation and finalization for diarization validation metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast


class DiarizationMetric(Protocol):
    """The small metric interface shared by training and standalone evaluation."""

    def update(self, predictions: object, target: object) -> None:
        """Accumulate one prediction and target batch."""

    def compute(self) -> Mapping[str, object]:
        """Return the aggregate DER and component values."""

    def reset(self) -> None:
        """Clear all accumulated metric state."""


class _ScalarValue(Protocol):
    """The optional tensor-like scalar operations used at the metric boundary."""

    def detach(self) -> _ScalarValue:
        """Return a detached scalar."""

    def float(self) -> _ScalarValue:
        """Return the scalar in floating-point representation."""

    def cpu(self) -> _ScalarValue:
        """Return the scalar on the CPU."""

    def item(self) -> object:
        """Return the scalar's Python value."""


@dataclass(frozen=True)
class ValidationMetrics:
    """The five values emitted by one validation pass."""

    loss: float
    der: float
    false_alarm: float
    miss: float
    confusion: float

    def __post_init__(self) -> None:
        for field in ("loss", "der", "false_alarm", "miss", "confusion"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")

    def to_training_dict(self) -> dict[str, float]:
        """Return the names used by the existing trainer logs and records."""

        return {
            "Loss": self.loss,
            "DER": self.der,
            "FA": self.false_alarm,
            "Miss": self.miss,
            "Confusion": self.confusion,
        }

    def to_result_dict(self) -> dict[str, float]:
        """Return the names used by the typed external validation result."""

        return {
            "loss": self.loss,
            "der": self.der,
            "false_alarm": self.false_alarm,
            "miss": self.miss,
            "confusion": self.confusion,
        }


def _scalar_to_float(value: object, field: str) -> float:
    """Convert one scalar tensor or Python number without importing Torch."""

    detached = value
    for method_name in ("detach", "float", "cpu", "item"):
        candidate = cast(_ScalarValue, detached)
        method = getattr(candidate, method_name, None)
        if not callable(method):
            continue
        try:
            detached = method()
        except AttributeError:
            continue
    try:
        result = float(detached)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a scalar number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _metric_value(values: Mapping[str, object], key: str) -> float:
    """Read one required metric key using the pyannote metric names."""

    if key not in values:
        raise ValueError(f"validation metric output is missing {key}")
    return _scalar_to_float(values[key], key)


def update_validation_batch_metrics(
    *,
    model: object,
    powerset: object,
    metric: DiarizationMetric,
    features: object,
    target: object,
) -> object:
    """Compute one validation-batch NLL and update the shared DER metric.

    Training and standalone evaluation must call this function for every batch
    so loss, permutation, powerset conversion, and metric.update stay identical.
    """

    from diarizen.validation_step import powerset_validation_step

    loss, predictions, metric_target = powerset_validation_step(model, powerset, features, target)
    metric.update(predictions, metric_target)
    return loss


def finalize_validation_metrics(losses: Iterable[object], metric: DiarizationMetric) -> ValidationMetrics:
    """Finalize one metric collection and always reset it afterward.

    Loss is the arithmetic mean of the per-validation-step scalar losses. DER
    and its components are read from the aggregate metric after every batch has
    been updated. The metric is reset even when computation or conversion fails.
    """

    try:
        loss_values = tuple(_scalar_to_float(value, "loss") for value in losses)
        if not loss_values:
            raise ValueError("validation cannot finalize without at least one loss")
        computed = metric.compute()
        if not isinstance(computed, Mapping):
            raise ValueError("validation metric output must be a mapping")
        return ValidationMetrics(
            loss=math.fsum(loss_values) / len(loss_values),
            der=_metric_value(computed, "DiarizationErrorRate"),
            false_alarm=_metric_value(computed, "DiarizationErrorRate/FalseAlarm"),
            miss=_metric_value(computed, "DiarizationErrorRate/Miss"),
            confusion=_metric_value(computed, "DiarizationErrorRate/Confusion"),
        )
    finally:
        metric.reset()


class ValidationMetricAccumulator:
    """Collect losses and DER batches, then finalize them once."""

    def __init__(self, metric: DiarizationMetric):
        self._metric = metric
        self._losses: list[object] = []
        self._finalized = False

    def update(self, loss: object, predictions: object, target: object) -> None:
        """Accumulate one loss and one diarization prediction batch."""

        if self._finalized:
            raise RuntimeError("validation metric accumulator is already finalized")
        self._losses.append(loss)
        self._metric.update(predictions, target)

    def finalize(self) -> ValidationMetrics:
        """Return aggregate validation metrics and reset the wrapped metric."""

        if self._finalized:
            raise RuntimeError("validation metric accumulator is already finalized")
        self._finalized = True
        return finalize_validation_metrics(self._losses, self._metric)


__all__ = [
    "DiarizationMetric",
    "ValidationMetricAccumulator",
    "ValidationMetrics",
    "finalize_validation_metrics",
    "update_validation_batch_metrics",
]
