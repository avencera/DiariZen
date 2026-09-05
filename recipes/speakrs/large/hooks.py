"""Typed training hooks injected into the shared dual-optimizer trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .budget import BudgetLedger
from .sampler import MixtureSampler


@dataclass
class LargeRunHooks:
    """Recipe-owned coverage, sampler, and budget state.

    Shared trainer code receives this object from the caller. It must not import
    recipe filesystem paths.
    """

    sampler: MixtureSampler
    ledger: BudgetLedger
    snapshot_every_updates: int = 250
    pending_example: dict[str, Any] | None = None
    last_snapshot_update: int = 0
    scoring_phase: str = "idle"

    def state_dict(self) -> dict[str, Any]:
        """Return checkpointable recipe state."""

        return {
            "sampler": self.sampler.state_dict(),
            "ledger": self.ledger.identity(),
            "snapshot_every_updates": self.snapshot_every_updates,
            "pending_example": self.pending_example,
            "last_snapshot_update": self.last_snapshot_update,
            "scoring_phase": self.scoring_phase,
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any], sampler: MixtureSampler, ledger: BudgetLedger
    ) -> LargeRunHooks:
        """Restore hooks after a checkpoint load."""

        restored_sampler = MixtureSampler.from_state_dict(payload["sampler"]) if "sampler" in payload else sampler
        restored_ledger = BudgetLedger.from_state_dict(payload["ledger"]) if "ledger" in payload else ledger
        hooks = cls(sampler=restored_sampler, ledger=restored_ledger)
        hooks.snapshot_every_updates = int(payload.get("snapshot_every_updates", 250))
        hooks.pending_example = payload.get("pending_example")
        hooks.last_snapshot_update = int(payload.get("last_snapshot_update", 0))
        hooks.scoring_phase = str(payload.get("scoring_phase", "idle"))
        return hooks

    def example_loss_weight(self, batch: Mapping[str, Any]) -> float:
        """Return the loss weight for coverage accounting."""

        if "loss_weight" in batch:
            return float(batch["loss_weight"])
        names = batch.get("names") or ()
        if not names:
            return 0.0
        return 1.0

    def set_pending(self, example: Mapping[str, Any]) -> None:
        """Record a proposed example. Prefetch must call this without acknowledge."""

        self.pending_example = dict(example)

    def acknowledge_update(
        self, batch: Mapping[str, Any], *, optimizer_updated: bool, loss_weight: float, valid: bool
    ) -> bool:
        """Commit coverage only at a successful optimizer boundary."""

        example = self.pending_example or {
            "stream": (batch.get("names") or ["unknown"])[0].split(":")[0] if batch.get("names") else "AMI",
            "parent_id": (batch.get("parent_ids") or batch.get("names") or ["unknown"])[0],
            "counts_for_coverage": True,
            "dynamic": False,
        }
        committed = self.sampler.acknowledge(
            example,
            optimizer_updated=optimizer_updated,
            loss_weight=loss_weight,
            valid=valid,
        )
        if optimizer_updated:
            self.pending_example = None
        return committed

    def should_snapshot(self, updates_trained: int) -> bool:
        """Return whether this update boundary should publish recovery state."""

        if updates_trained <= 0:
            return False
        if updates_trained - self.last_snapshot_update >= self.snapshot_every_updates:
            return True
        return False

    def mark_snapshot(self, updates_trained: int) -> None:
        """Record that a snapshot was published."""

        self.last_snapshot_update = updates_trained

    def can_train(self) -> bool:
        """Return whether an optimizer update is permitted."""

        return self.ledger.can_resume_training()

    def coverage_complete(self) -> bool:
        """Return whether every finite stream is covered."""

        return self.sampler.coverage.complete()
