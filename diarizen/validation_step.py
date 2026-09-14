"""Shared powerset validation step used by inline training and standalone evaluation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from pyannote.audio.utils.loss import nll_loss
from pyannote.audio.utils.permutation import permutate


def powerset_validation_step(
    forward: Callable[[object], object],
    powerset: object,
    xs: object,
    target: object,
) -> tuple[object, object, object]:
    """Run one validation batch through the shared metric path.

    Returns ``(loss, metric_predictions, metric_target)`` using the same
    powerset conversion, permutation, and NLL loss as inline training.
    """

    y_pred = forward(xs)
    multilabel = powerset.to_multilabel(y_pred)
    permutated_target, _ = permutate(multilabel, target)
    powerset_target = powerset.to_powerset(permutated_target.float())
    loss = nll_loss(y_pred, torch.argmax(powerset_target, dim=-1))
    return loss, torch.transpose(multilabel, 1, 2), torch.transpose(target, 1, 2)
