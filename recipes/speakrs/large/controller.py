"""Trusted controller: budget, fake provider, and worker token scrubbing."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping

from .budget import BudgetLedger, replacement_ledger
from .contracts import (
    QUALIFICATION_BINDING_DIGEST_FIELD,
    QUALIFICATION_LEASE_KIND,
    QualificationBinding,
    parse_kinded_lock,
    parse_qualification_binding,
    require_content_hash,
)
from .errors import RuntimeGateError
from .recovery import BackupReceipt, LocalTransport, newest_complete_generation


PROVIDER_TOKEN_NAMES = (
    "VAST_API_KEY",
    "VAST_API_TOKEN",
    "CONTAINER_API_KEY",
    "JUPYTER_TOKEN",
    "OPEN_BUTTON_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GHCR_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "DOCKER_TOKEN",
)


def scrubbed_worker_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment with provider and registry tokens removed."""

    cleaned = dict(os.environ if environment is None else environment)
    for name in PROVIDER_TOKEN_NAMES:
        cleaned.pop(name, None)
    return cleaned


def assert_worker_has_no_tokens(environment: Mapping[str, str] | None = None) -> None:
    """Fail if the worker environment still has billing or registry tokens."""

    present = [name for name in PROVIDER_TOKEN_NAMES if (environment or os.environ).get(name)]
    if present:
        raise RuntimeGateError("worker environment contains provider or registry tokens", {"names": present})


@dataclass
class FakeProvider:
    """In-process provider used for CPU tests. Never calls live Vast."""

    instances: dict[str, str] = field(default_factory=dict)
    destroyed: list[str] = field(default_factory=list)
    live_calls: list[str] = field(default_factory=list)
    fail_destroy: bool = False

    def create(self, offer_id: str) -> str:
        """Record a fake create. Tests must never treat this as live Vast."""

        instance_id = f"fake-{offer_id}"
        self.instances[instance_id] = "running"
        self.live_calls.append(f"create:{offer_id}")
        return instance_id

    def stop(self, instance_id: str) -> None:
        """Record a fake stop."""

        if instance_id not in self.instances:
            raise RuntimeGateError("unknown instance", {"instance_id": instance_id})
        self.instances[instance_id] = "stopped"
        self.live_calls.append(f"stop:{instance_id}")

    def destroy(self, instance_id: str) -> None:
        """Record a fake destroy. Optionally inject a failure."""

        if self.fail_destroy:
            raise RuntimeGateError("injected destroy failure")
        self.instances.pop(instance_id, None)
        self.destroyed.append(instance_id)
        self.live_calls.append(f"destroy:{instance_id}")


class LiveVastGuard:
    """Refuse live Vast mutation endpoints during this pre-rental goal."""

    FORBIDDEN = ("create", "start", "stop", "destroy")

    def __getattr__(self, name: str) -> Callable[..., None]:
        if name in self.FORBIDDEN:

            def blocked(*_args, **_kwargs):
                raise RuntimeGateError("live Vast mutation is forbidden in this goal", {"method": name})

            return blocked
        raise AttributeError(name)


@dataclass
class Controller:
    """One supervisor owner for qualification and main control."""

    ledger: BudgetLedger
    provider: FakeProvider | LiveVastGuard
    transport: LocalTransport
    retry_per_class: int = 2
    retry_total: int = 3
    attempts: dict[str, int] = field(default_factory=dict)
    total_attempts: int = 0
    lock_held: bool = False
    current_attempt: str | None = None

    def acquire(self) -> None:
        """Take the single supervisor lock."""

        if self.lock_held:
            raise RuntimeGateError("supervisor lock already held")
        self.lock_held = True

    def release(self) -> None:
        """Drop the supervisor lock."""

        self.lock_held = False
        self.current_attempt = None

    def start_attempt(self, kind: str) -> None:
        """Start one bounded attempt. CUDA-required failures are explicit."""

        if not self.lock_held:
            raise RuntimeGateError("attempt requires the supervisor lock")
        if self.current_attempt is not None:
            raise RuntimeGateError("previous attempt has not completed")
        used = self.attempts.get(kind, 0)
        if used >= self.retry_per_class or self.total_attempts >= self.retry_total:
            raise RuntimeGateError(
                "retry budget exhausted",
                {"kind": kind, "used": used, "total": self.total_attempts},
            )
        if not self.ledger.can_resume_training() and kind == "train":
            raise RuntimeGateError(
                "training is not permitted in the current budget state", {"state": self.ledger.state}
            )
        self.attempts[kind] = used + 1
        self.total_attempts += 1
        self.current_attempt = kind

    def complete_attempt(self, *, success: bool) -> None:
        """Close the current attempt. Failures consume retry budget already counted."""

        if self.current_attempt is None:
            raise RuntimeGateError("no current attempt")
        self.current_attempt = None
        if not success and self.total_attempts >= self.retry_total:
            self.ledger.mark_terminal("failed")

    def control_qualification(
        self,
        lease: Mapping[str, object],
        qualification_binding: QualificationBinding | Mapping[str, object] | str,
    ) -> dict[str, object]:
        """Run qualification control from a lease bound to one release identity."""

        parsed = parse_kinded_lock(lease, QUALIFICATION_LEASE_KIND)
        if parsed.get("kind") != QUALIFICATION_LEASE_KIND:
            raise RuntimeGateError("qualification control requires a qualification lease")
        expected_digest: str
        if isinstance(qualification_binding, QualificationBinding):
            expected_digest = qualification_binding.binding_sha256
        elif isinstance(qualification_binding, Mapping):
            expected_digest = parse_qualification_binding(qualification_binding).binding_sha256
        else:
            expected_digest = require_content_hash(qualification_binding, "qualification binding sha256")
        actual_digest = parsed.get(QUALIFICATION_BINDING_DIGEST_FIELD)
        if actual_digest != expected_digest:
            raise RuntimeGateError(
                "qualification lease is not bound to the supplied qualification input",
                {"expected": expected_digest, "actual": actual_digest},
            )
        actual_digest = require_content_hash(actual_digest, "qualification binding sha256")
        ceiling_raw = parsed.get("spend_ceiling_usd")
        if isinstance(ceiling_raw, bool) or not isinstance(ceiling_raw, (int, float)):
            raise RuntimeGateError("qualification lease has an invalid spend ceiling")
        ceiling = float(ceiling_raw)
        if not math.isfinite(ceiling) or ceiling <= 0 or ceiling > self.ledger.policy.qualification_usd + 1e-9:
            raise RuntimeGateError("qualification ceiling exceeds the $5 allocation")
        rates = parsed.get("rates")
        if not isinstance(rates, Mapping):
            raise RuntimeGateError("qualification lease has no hourly rates")
        gpu_rate = rates.get("gpu_usd_per_hour")
        disk_rate = rates.get("disk_usd_per_hour")
        if (
            isinstance(gpu_rate, bool)
            or not isinstance(gpu_rate, (int, float))
            or not math.isfinite(float(gpu_rate))
            or float(gpu_rate) <= 0
            or isinstance(disk_rate, bool)
            or not isinstance(disk_rate, (int, float))
            or not math.isfinite(float(disk_rate))
            or float(disk_rate) < 0
        ):
            raise RuntimeGateError("qualification lease has invalid hourly rates")
        hard_deadline = parsed.get("hard_deadline")
        if not isinstance(hard_deadline, str):
            raise RuntimeGateError("qualification lease has no hard deadline")
        try:
            deadline = datetime.fromisoformat(hard_deadline.replace("Z", "+00:00"))
        except ValueError as error:
            raise RuntimeGateError("qualification lease hard deadline is not an ISO-8601 timestamp") from error
        if deadline.tzinfo is None:
            raise RuntimeGateError("qualification lease hard deadline must include a UTC offset")
        deadline_epoch_seconds = deadline.timestamp()
        deadline_seconds = deadline_epoch_seconds - datetime.now(timezone.utc).timestamp()
        if deadline_seconds <= 0:
            raise RuntimeGateError("qualification lease hard deadline has passed")
        hourly_rate = float(gpu_rate) + float(disk_rate)
        budget_seconds = ceiling / hourly_rate * 3600.0
        max_runtime_seconds = min(deadline_seconds, budget_seconds)
        return {
            "ok": True,
            "lease_id": parsed["lease_id"],
            "spend_ceiling_usd": ceiling,
            "hourly_rate_usd": hourly_rate,
            "hard_deadline": hard_deadline,
            "deadline_epoch_seconds": deadline_epoch_seconds,
            "max_runtime_seconds": max_runtime_seconds,
            "provider": type(self.provider).__name__,
            "live_vast": False,
            QUALIFICATION_BINDING_DIGEST_FIELD: actual_digest,
        }

    def backup_newest(self, worker_root, trusted_root) -> BackupReceipt:
        """Pin the newest complete worker generation and copy it with a receipt."""

        source = newest_complete_generation(worker_root)
        if source is None:
            raise RuntimeGateError("no complete worker generation to back up")
        destination = trusted_root / source.name
        return self.transport.pin_and_copy(source, destination)

    def replace_instance(
        self, offer_id: str, semantic_config: Mapping[str, object], previous_config: Mapping[str, object]
    ) -> BudgetLedger:
        """Replace an instance only with the same semantic configuration."""

        if dict(semantic_config) != dict(previous_config):
            raise RuntimeGateError("replacement must keep the same semantic configuration")
        if isinstance(self.provider, LiveVastGuard):
            self.provider.create(offer_id)
        ledger = replacement_ledger(self.ledger)
        self.ledger = ledger
        return ledger


def lease_from_offer(
    offer: Mapping[str, object],
    preparation_digest: str,
    backup_target: str,
    qualification_binding_sha256: str,
) -> dict[str, object]:
    """Build a qualification lease from a fixture offer. Does not rent."""

    required = ("offer_id", "gpu_profile", "usd_per_hour", "disk_usd_per_hour", "hard_deadline")
    missing = [key for key in required if key not in offer]
    if missing:
        raise RuntimeGateError("offer is incomplete", {"missing": missing})
    binding_digest = require_content_hash(qualification_binding_sha256, "qualification binding sha256")
    lease = {
        "kind": QUALIFICATION_LEASE_KIND,
        "lease_id": f"lease-{offer['offer_id']}",
        "preparation_digest": preparation_digest,
        "offer": dict(offer),
        "instance": None,
        "rates": {
            "gpu_usd_per_hour": offer["usd_per_hour"],
            "disk_usd_per_hour": offer["disk_usd_per_hour"],
        },
        "hard_deadline": offer["hard_deadline"],
        "backup_target": backup_target,
        "spend_ceiling_usd": 5.0,
    }
    lease[QUALIFICATION_BINDING_DIGEST_FIELD] = binding_digest
    return lease
