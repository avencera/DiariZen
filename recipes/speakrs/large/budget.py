"""One cumulative budget ledger. Restarts cannot reset spend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .contracts import BOUNDARY_POLICY, DEFAULT_BUDGET, BudgetPolicy, parse_budget
from .errors import RuntimeGateError


TERMINAL_REASONS = frozenset(
    {
        "completed",
        "failed",
        "destroyed",
        "deadline",
        "safety_stop",
        "coverage_incomplete_budget",
    }
)


@dataclass
class BudgetLedger:
    """Cumulative spend across replacements, pauses, and child lineages."""

    policy: BudgetPolicy
    spent_usd: float = 0.0
    reserved_scoring_usd: float = DEFAULT_BUDGET.scoring_usd
    reserved_failure_usd: float = DEFAULT_BUDGET.reserve_usd
    state: str = "open"
    amendment_id: str | None = None
    history: list[dict[str, object]] = field(default_factory=list)

    def identity(self) -> dict[str, object]:
        """Return the sealed ledger."""

        return {
            "policy": self.policy.identity(),
            "spent_usd": self.spent_usd,
            "reserved_scoring_usd": self.reserved_scoring_usd,
            "reserved_failure_usd": self.reserved_failure_usd,
            "state": self.state,
            "amendment_id": self.amendment_id,
            "history": list(self.history),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> BudgetLedger:
        """Restore a ledger. Spend cannot be omitted."""

        policy = parse_budget(payload["policy"])
        return cls(
            policy=policy,
            spent_usd=float(payload["spent_usd"]),
            reserved_scoring_usd=float(payload["reserved_scoring_usd"]),
            reserved_failure_usd=float(payload["reserved_failure_usd"]),
            state=str(payload["state"]),
            amendment_id=payload.get("amendment_id"),
            history=list(payload.get("history") or []),
        )

    def remaining_usd(self) -> float:
        """Return unspent funds including reserves."""

        return self.policy.total_usd - self.spent_usd

    def training_funds_usd(self) -> float:
        """Return funds still available for training after reserves."""

        return self.policy.total_usd - self.spent_usd - self.reserved_scoring_usd - self.reserved_failure_usd

    def charge(self, amount_usd: float, reason: str) -> None:
        """Add a cost. Negative charges and spend resets are rejected."""

        if self.state in TERMINAL_REASONS:
            raise RuntimeGateError("terminal run cannot accept new charges", {"state": self.state})
        if amount_usd < 0:
            raise RuntimeGateError("charges cannot be negative")
        if self.spent_usd + amount_usd > self.policy.total_usd + 1e-9:
            raise RuntimeGateError(
                "charge would exceed total_usd",
                {"spent": self.spent_usd, "amount": amount_usd, "total": self.policy.total_usd},
            )
        self.spent_usd += amount_usd
        self.history.append({"amount_usd": amount_usd, "reason": reason, "spent_usd": self.spent_usd})

    def pause_for_extension(self) -> None:
        """Enter awaiting_extension. Training cannot resume until amended."""

        if self.state in TERMINAL_REASONS:
            raise RuntimeGateError("terminal run cannot pause for extension")
        self.state = "awaiting_extension"

    def amend(self, *, amendment_id: str, new_total_usd: float, authorized: bool) -> None:
        """Apply one authorized amendment to the same cumulative identity."""

        if not authorized:
            raise RuntimeGateError("budget amendment is not authorized")
        if self.state != "awaiting_extension":
            raise RuntimeGateError("amendment requires awaiting_extension", {"state": self.state})
        if new_total_usd < self.policy.total_usd:
            raise RuntimeGateError("amendment cannot reduce the cumulative ceiling")
        if not amendment_id:
            raise RuntimeGateError("amendment_id is required")
        self.policy = BudgetPolicy(
            total_usd=float(new_total_usd),
            boundary_policy=self.policy.boundary_policy,
            qualification_usd=self.policy.qualification_usd,
            training_usd=self.policy.training_usd + (new_total_usd - self.policy.total_usd),
            scoring_usd=self.policy.scoring_usd,
            reserve_usd=self.policy.reserve_usd,
        )
        self.amendment_id = amendment_id
        self.state = "open"
        self.history.append(
            {
                "amount_usd": 0.0,
                "reason": "authorized_amendment",
                "amendment_id": amendment_id,
                "total_usd": self.policy.total_usd,
                "spent_usd": self.spent_usd,
            }
        )

    def mark_terminal(self, reason: str) -> None:
        """Seal the ledger. Terminal resume cannot reset spend or train."""

        if reason not in TERMINAL_REASONS:
            raise RuntimeGateError("unknown terminal reason", {"reason": reason})
        self.state = reason

    def can_resume_training(self) -> bool:
        """Return whether a worker may execute an optimizer update."""

        if self.state == "awaiting_extension":
            return False
        if self.state in TERMINAL_REASONS:
            return False
        return self.training_funds_usd() > 0

    def reject_auto_extension(self) -> None:
        """Refuse any automatic ceiling increase."""

        raise RuntimeGateError(
            "automatic budget extension is forbidden",
            {"boundary_policy": BOUNDARY_POLICY},
        )


def replacement_ledger(previous: BudgetLedger) -> BudgetLedger:
    """Continue the same spend identity on a replacement instance."""

    if previous.state in TERMINAL_REASONS:
        raise RuntimeGateError("terminal ledger cannot start a replacement")
    clone = BudgetLedger.from_state_dict(previous.identity())
    clone.history.append({"amount_usd": 0.0, "reason": "replacement", "spent_usd": clone.spent_usd})
    return clone
