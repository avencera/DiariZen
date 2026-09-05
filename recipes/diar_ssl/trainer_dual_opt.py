# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import json

import numpy as np
import torch
from accelerate.logging import get_logger
from pyannote.audio.utils.loss import nll_loss
from pyannote.audio.utils.permutation import permutate

from diarizen.trainer_dual_opt import Trainer as BaseTrainer
from diarizen.trainer_utils import (
    AutoClipGradHistory,
    raise_for_non_finite_loss,
    reject_fp16_dual_optimizer,
    scalar_to_float,
)


logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        accelerator = kwargs.get("accelerator")
        if accelerator is None and args:
            accelerator = args[0]
        reject_fp16_dual_optimizer(accelerator)

        super().__init__(*args, **kwargs)
        self.accelerator.print(self.model)

        # auto GN
        self.grad_history = AutoClipGradHistory(self.gradient_history_size)
        self.accelerator.register_for_checkpointing(self.grad_history)

    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def auto_clip_grad_norm_(self, model):
        grad_norm = self.compute_grad_norm(model)
        self.grad_history.append(grad_norm)
        clip_value = np.percentile(self.grad_history, self.gradient_percentile)
        self.accelerator.clip_grad_norm_(model.parameters(), clip_value)

    def training_step(self, batch, batch_idx):
        xs, target = batch["xs"], batch["ts"]
        y_pred = self.model(xs)
        # powerset
        multilabel = self.unwrap_model.powerset.to_multilabel(y_pred)
        permutated_target, _ = permutate(multilabel, target)
        permutated_target_powerset = self.unwrap_model.powerset.to_powerset(permutated_target.float())

        loss = nll_loss(y_pred, torch.argmax(permutated_target_powerset, dim=-1))
        raise_for_non_finite_loss(loss, (self.optimizer_small, self.optimizer_big), batch_idx)

        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            # The gradients are added across all processes in this cumulative gradient accumulation step.
            self.auto_clip_grad_norm_(self.model)

        self.optimizer_small.step()
        self.optimizer_big.step()
        # accelerate defers these calls until the synchronized optimizer step
        # clearing at the start of that step discards earlier micro-batch gradients
        self.optimizer_small.zero_grad()
        self.optimizer_big.zero_grad()

        return {"Loss": loss.detach().float()}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        xs, target = batch["xs"], batch["ts"]

        y_pred = self.model(xs)
        # powerset
        multilabel = self.unwrap_model.powerset.to_multilabel(y_pred)
        permutated_target, _ = permutate(multilabel, target)
        permutated_target_powerset = self.unwrap_model.powerset.to_powerset(permutated_target.float())

        loss = nll_loss(y_pred, torch.argmax(permutated_target_powerset, dim=-1))
        self.unwrap_model.validation_metric.update(
            torch.transpose(multilabel, 1, 2),
            torch.transpose(target, 1, 2),
        )

        return {"Loss": loss.detach().float()}

    def validation_epoch_end(self, validation_epoch_output):
        loss_items = [step_out["Loss"] for step_out in validation_epoch_output]
        metric_means = {"Loss": sum(map(scalar_to_float, loss_items)) / len(loss_items)}
        try:
            computed_metrics = self.unwrap_model.validation_metric.compute()
            metric_means.update(
                {
                    "DER": scalar_to_float(computed_metrics["DiarizationErrorRate"]),
                    "FA": scalar_to_float(computed_metrics["DiarizationErrorRate/FalseAlarm"]),
                    "Miss": scalar_to_float(computed_metrics["DiarizationErrorRate/Miss"]),
                    "Confusion": scalar_to_float(computed_metrics["DiarizationErrorRate/Confusion"]),
                }
            )
        finally:
            self.unwrap_model.validation_metric.reset()

        if self.accelerator.is_local_main_process:
            for key, metric_mean in metric_means.items():
                self.writer.add_scalar(f"Validation_Epoch/{key}", metric_mean, self.state.epochs_trained)
            logger.info(
                "Validation metrics on epoch %d: Loss=%.6f DER=%.6f FA=%.6f Miss=%.6f Confusion=%.6f",
                self.state.epochs_trained,
                metric_means["Loss"],
                metric_means["DER"],
                metric_means["FA"],
                metric_means["Miss"],
                metric_means["Confusion"],
            )
            logger.info(
                f"Validation Loss/DER on epoch {self.state.epochs_trained}: "
                f"{round(metric_means['Loss'], 3)} / {round(metric_means['DER'], 3)}"
            )
            self._write_validation_metrics(metric_means)
        # Validation loss remains the checkpoint-selection score.
        return metric_means["Loss"]

    def _write_validation_metrics(self, metric_means):
        """Write one convergent machine-readable validation record."""
        path = self.exp_dir / "validation_metrics.json"
        if path.exists():
            records = json.loads(path.read_text())
        else:
            records = []
        epoch = self.state.epochs_trained
        records = [record for record in records if record["epoch"] != epoch]
        records.append({"epoch": epoch, **metric_means})
        records.sort(key=lambda record: record["epoch"])
        temporary = path.with_suffix(".partial.json")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(path)
