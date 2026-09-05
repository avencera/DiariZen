"""Typed parsers for release, run, budget, and phase schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .errors import ContractError
from .hashing import sha256_json


SCHEMA_NAME = "speakrs-large"
SCHEMA_VERSION = 1

FINITE_STREAMS = (
    "AMI",
    "AliMeeting",
    "AISHELL4",
    "VoxConverse",
    "NOTSOFAR_real",
    "ICSI",
    "LOTUSDIS",
    "NOTSOFAR_sim",
)
DYNAMIC_STREAM = "dynamic"
STREAM_QUOTAS: dict[str, float] = {
    "AMI": 0.18,
    "AliMeeting": 0.14,
    "AISHELL4": 0.14,
    "VoxConverse": 0.12,
    "NOTSOFAR_real": 0.10,
    "ICSI": 0.08,
    "LOTUSDIS": 0.04,
    "NOTSOFAR_sim": 0.15,
    DYNAMIC_STREAM: 0.05,
}
GOLD_STREAMS = (
    "AMI",
    "AliMeeting",
    "AISHELL4",
    "VoxConverse",
    "NOTSOFAR_real",
    "ICSI",
    "LOTUSDIS",
)
BRONZE_STREAMS = ("NOTSOFAR_sim", DYNAMIC_STREAM)
LABEL_TIERS = ("gold", "silver", "bronze")
SPLITS = ("train", "dev", "test")
CYCLE_EXAMPLES = 128_000
EFFECTIVE_BATCH = 64
UPDATES_PER_CYCLE = CYCLE_EXAMPLES // EFFECTIVE_BATCH
BUDGET_TOTAL_USD = 150.0
BOUNDARY_POLICY = "pause_for_extension"
QUALIFICATION_USD = 5.0
TRAINING_USD = 120.0
SCORING_USD = 20.0
RESERVE_USD = 5.0
SEED = 3407
WAVLM_BLOCKS = 24
WAVLM_REPRESENTATIONS = 25
WAVLM_WIDTH = 1024
WAVLM_HEADS = 16
POWERSET_CLASSES = 11
CHUNK_SECONDS = 8
SAMPLE_RATE = 16_000
OUTPUT_FRAMES = 399

PREPARATION_KIND = "preparation"
QUALIFICATION_LEASE_KIND = "qualification-lease"
LAUNCH_KIND = "launch"
ALLOWED_KINDS = (PREPARATION_KIND, QUALIFICATION_LEASE_KIND, LAUNCH_KIND)

LAUNCH_ONLY_FIELDS = frozenset(
    {
        "offer",
        "instance",
        "physical_batch",
        "accumulation",
        "measured_cycle_hours",
        "affordable_cycles",
        "worker_deadline",
        "qualification_digest",
        "launch_id",
    }
)
LEASE_ONLY_FIELDS = frozenset(
    {
        "offer",
        "instance",
        "rates",
        "hard_deadline",
        "backup_target",
        "lease_id",
    }
)


class Phase(str, Enum):
    """Explicit promotion states. Unknown values fail."""

    PREPARED = "prepared"
    QUALIFICATION_PLANNED = "qualification-planned"
    LEASED = "leased"
    QUALIFIED = "qualified"
    LAUNCH_LOCKED = "launch-locked"
    RUNNING = "running"
    AWAITING_EXTENSION = "awaiting_extension"
    TERMINAL = "terminal"
    ARCHIVED = "archived"


class LicenceDecision(str, Enum):
    """Accepted-use decision recorded on every source and recording."""

    ACCEPTED_CC = "accepted_cc"
    REJECTED_NC = "rejected_nc"
    REJECTED_PAID = "rejected_paid"
    REJECTED_UNKNOWN = "rejected_unknown"
    UNRESOLVED = "unresolved"


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object", {"label": label})
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ContractError(f"{label} has unknown keys", {"unknown": unknown, "label": label})


def _require_str(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label}.{key} must be a non-empty string")
    return value


def _require_number(payload: Mapping[str, Any], key: str, label: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label}.{key} must be a number")
    return float(value)


def _optional_path(payload: Mapping[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ContractError(f"{key} must be a path string when present")
    return Path(value)


@dataclass(frozen=True)
class BudgetPolicy:
    """One cumulative Vast budget identity."""

    total_usd: float
    boundary_policy: str
    qualification_usd: float
    training_usd: float
    scoring_usd: float
    reserve_usd: float

    def identity(self) -> dict[str, object]:
        """Return the sealed budget record."""

        return {
            "total_usd": self.total_usd,
            "boundary_policy": self.boundary_policy,
            "qualification_usd": self.qualification_usd,
            "training_usd": self.training_usd,
            "scoring_usd": self.scoring_usd,
            "reserve_usd": self.reserve_usd,
        }


def parse_budget(payload: Any, label: str = "budget") -> BudgetPolicy:
    """Parse and reject any budget that is not the selected $150 pause policy."""

    data = _require_object(payload, label)
    _reject_unknown(
        data,
        {
            "total_usd",
            "boundary_policy",
            "qualification_usd",
            "training_usd",
            "scoring_usd",
            "reserve_usd",
        },
        label,
    )
    policy = BudgetPolicy(
        total_usd=_require_number(data, "total_usd", label),
        boundary_policy=_require_str(data, "boundary_policy", label),
        qualification_usd=_require_number(data, "qualification_usd", label),
        training_usd=_require_number(data, "training_usd", label),
        scoring_usd=_require_number(data, "scoring_usd", label),
        reserve_usd=_require_number(data, "reserve_usd", label),
    )
    if policy.total_usd != BUDGET_TOTAL_USD:
        raise ContractError("budget.total_usd must be 150", {"actual": policy.total_usd})
    if policy.boundary_policy != BOUNDARY_POLICY:
        raise ContractError(
            "budget.boundary_policy must be pause_for_extension",
            {"actual": policy.boundary_policy},
        )
    parts = (
        policy.qualification_usd,
        policy.training_usd,
        policy.scoring_usd,
        policy.reserve_usd,
    )
    if abs(sum(parts) - policy.total_usd) > 1e-9:
        raise ContractError("budget allocations must sum to total_usd", policy.identity())
    if min(parts) < 0:
        raise ContractError("budget allocations cannot be negative")
    return policy


DEFAULT_BUDGET = BudgetPolicy(
    total_usd=BUDGET_TOTAL_USD,
    boundary_policy=BOUNDARY_POLICY,
    qualification_usd=QUALIFICATION_USD,
    training_usd=TRAINING_USD,
    scoring_usd=SCORING_USD,
    reserve_usd=RESERVE_USD,
)


@dataclass(frozen=True)
class RelocationMap:
    """Map semantic release paths to relocatable physical roots."""

    audio_root: Path
    backup_root: Path
    source_cache: Path
    evidence_root: Path


def parse_relocation(payload: Any, label: str = "relocation") -> RelocationMap:
    """Parse relocatable roots. These are not content identity."""

    data = _require_object(payload, label)
    _reject_unknown(data, {"audio_root", "backup_root", "source_cache", "evidence_root"}, label)
    return RelocationMap(
        audio_root=Path(_require_str(data, "audio_root", label)),
        backup_root=Path(_require_str(data, "backup_root", label)),
        source_cache=Path(_require_str(data, "source_cache", label)),
        evidence_root=Path(_require_str(data, "evidence_root", label)),
    )


@dataclass(frozen=True)
class ModelIdentity:
    """Architecture identity that every Large initializer must match."""

    wavlm_blocks: int
    wavlm_representations: int
    wavlm_width: int
    wavlm_heads: int
    powerset_classes: int
    chunk_seconds: int
    sample_rate: int
    output_frames: int
    normalize_waveform: bool
    strict_load: bool

    def identity(self) -> dict[str, object]:
        """Return the sealed model identity."""

        return {
            "wavlm_blocks": self.wavlm_blocks,
            "wavlm_representations": self.wavlm_representations,
            "wavlm_width": self.wavlm_width,
            "wavlm_heads": self.wavlm_heads,
            "powerset_classes": self.powerset_classes,
            "chunk_seconds": self.chunk_seconds,
            "sample_rate": self.sample_rate,
            "output_frames": self.output_frames,
            "normalize_waveform": self.normalize_waveform,
            "strict_load": self.strict_load,
        }


def parse_model_identity(payload: Any, label: str = "model") -> ModelIdentity:
    """Parse the Large architecture identity."""

    data = _require_object(payload, label)
    allowed = {
        "wavlm_blocks",
        "wavlm_representations",
        "wavlm_width",
        "wavlm_heads",
        "powerset_classes",
        "chunk_seconds",
        "sample_rate",
        "output_frames",
        "normalize_waveform",
        "strict_load",
    }
    _reject_unknown(data, allowed, label)
    identity = ModelIdentity(
        wavlm_blocks=int(_require_number(data, "wavlm_blocks", label)),
        wavlm_representations=int(_require_number(data, "wavlm_representations", label)),
        wavlm_width=int(_require_number(data, "wavlm_width", label)),
        wavlm_heads=int(_require_number(data, "wavlm_heads", label)),
        powerset_classes=int(_require_number(data, "powerset_classes", label)),
        chunk_seconds=int(_require_number(data, "chunk_seconds", label)),
        sample_rate=int(_require_number(data, "sample_rate", label)),
        output_frames=int(_require_number(data, "output_frames", label)),
        normalize_waveform=bool(data.get("normalize_waveform")),
        strict_load=bool(data.get("strict_load")),
    )
    expected = DEFAULT_MODEL.identity()
    if identity.identity() != expected:
        raise ContractError("model identity does not match the sealed Large architecture", identity.identity())
    return identity


DEFAULT_MODEL = ModelIdentity(
    wavlm_blocks=WAVLM_BLOCKS,
    wavlm_representations=WAVLM_REPRESENTATIONS,
    wavlm_width=WAVLM_WIDTH,
    wavlm_heads=WAVLM_HEADS,
    powerset_classes=POWERSET_CLASSES,
    chunk_seconds=CHUNK_SECONDS,
    sample_rate=SAMPLE_RATE,
    output_frames=OUTPUT_FRAMES,
    normalize_waveform=True,
    strict_load=True,
)


@dataclass(frozen=True)
class MixturePolicy:
    """Deterministic 80% gold / 20% bronze quotas."""

    quotas: dict[str, float]
    cycle_examples: int
    effective_batch: int
    updates_per_cycle: int
    seed: int

    def identity(self) -> dict[str, object]:
        """Return the sealed mixture record."""

        return {
            "quotas": dict(self.quotas),
            "cycle_examples": self.cycle_examples,
            "effective_batch": self.effective_batch,
            "updates_per_cycle": self.updates_per_cycle,
            "seed": self.seed,
            "gold_mass": sum(self.quotas[name] for name in GOLD_STREAMS),
            "bronze_mass": sum(self.quotas[name] for name in BRONZE_STREAMS),
        }


def parse_mixture(payload: Any, label: str = "mixture") -> MixturePolicy:
    """Parse mixture quotas and reject silent renormalization."""

    data = _require_object(payload, label)
    _reject_unknown(
        data,
        {"quotas", "cycle_examples", "effective_batch", "updates_per_cycle", "seed"},
        label,
    )
    quotas_raw = _require_object(data.get("quotas"), f"{label}.quotas")
    quotas = {key: float(value) for key, value in quotas_raw.items()}
    if set(quotas) != set(STREAM_QUOTAS):
        raise ContractError("mixture quotas must name every selected stream", {"actual": sorted(quotas)})
    for name, expected in STREAM_QUOTAS.items():
        if abs(quotas[name] - expected) > 1e-12:
            raise ContractError(f"quota for {name} must be {expected}", {"actual": quotas[name]})
    if abs(sum(quotas.values()) - 1.0) > 1e-12:
        raise ContractError("mixture quotas must sum to 1")
    policy = MixturePolicy(
        quotas=quotas,
        cycle_examples=int(_require_number(data, "cycle_examples", label)),
        effective_batch=int(_require_number(data, "effective_batch", label)),
        updates_per_cycle=int(_require_number(data, "updates_per_cycle", label)),
        seed=int(_require_number(data, "seed", label)),
    )
    if policy.cycle_examples != CYCLE_EXAMPLES:
        raise ContractError("cycle_examples must be 128000")
    if policy.effective_batch != EFFECTIVE_BATCH:
        raise ContractError("effective_batch must be 64")
    if policy.updates_per_cycle != UPDATES_PER_CYCLE:
        raise ContractError("updates_per_cycle must be 2000")
    if policy.seed != SEED:
        raise ContractError("mixture seed must be 3407")
    gold = sum(policy.quotas[name] for name in GOLD_STREAMS)
    bronze = sum(policy.quotas[name] for name in BRONZE_STREAMS)
    if abs(gold - 0.80) > 1e-12 or abs(bronze - 0.20) > 1e-12:
        raise ContractError("mixture must be 80% gold and 20% bronze")
    return policy


DEFAULT_MIXTURE = MixturePolicy(
    quotas=dict(STREAM_QUOTAS),
    cycle_examples=CYCLE_EXAMPLES,
    effective_batch=EFFECTIVE_BATCH,
    updates_per_cycle=UPDATES_PER_CYCLE,
    seed=SEED,
)


@dataclass(frozen=True)
class RunSpec:
    """Resolved pre-rental specification. Stores no credentials."""

    schema: str
    schema_version: int
    run_id: str
    release_root: Path
    artifacts_root: Path
    ghcr_package: str
    relocation: RelocationMap
    budget: BudgetPolicy
    model: ModelIdentity
    mixture: MixturePolicy
    required_corpora: tuple[str, ...] = FINITE_STREAMS


def parse_spec(payload: Any) -> RunSpec:
    """Parse the resolved Large specification."""

    data = _require_object(payload, "spec")
    allowed = {
        "schema",
        "schema_version",
        "run_id",
        "release_root",
        "artifacts_root",
        "ghcr_package",
        "relocation",
        "budget",
        "model",
        "mixture",
        "required_corpora",
    }
    _reject_unknown(data, allowed, "spec")
    if data.get("schema") != SCHEMA_NAME:
        raise ContractError("spec.schema must be speakrs-large")
    if int(_require_number(data, "schema_version", "spec")) != SCHEMA_VERSION:
        raise ContractError("spec.schema_version must be 1")
    for secret_key in ("token", "password", "secret", "ssh_key", "hf_token", "vast_api_key"):
        if secret_key in data:
            raise ContractError("specification must not store credentials", {"key": secret_key})
    corpora = tuple(data.get("required_corpora") or FINITE_STREAMS)
    if set(corpora) != set(FINITE_STREAMS):
        raise ContractError("required_corpora cannot drop an approved stream", {"actual": list(corpora)})
    return RunSpec(
        schema=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        run_id=_require_str(data, "run_id", "spec"),
        release_root=Path(_require_str(data, "release_root", "spec")),
        artifacts_root=Path(_require_str(data, "artifacts_root", "spec")),
        ghcr_package=_require_str(data, "ghcr_package", "spec"),
        relocation=parse_relocation(data.get("relocation")),
        budget=parse_budget(data.get("budget")),
        model=parse_model_identity(data.get("model")),
        mixture=parse_mixture(data.get("mixture")),
        required_corpora=corpora,
    )


def spec_to_json(spec: RunSpec) -> dict[str, object]:
    """Serialize a parsed spec back to JSON."""

    return {
        "schema": spec.schema,
        "schema_version": spec.schema_version,
        "run_id": spec.run_id,
        "release_root": spec.release_root.as_posix(),
        "artifacts_root": spec.artifacts_root.as_posix(),
        "ghcr_package": spec.ghcr_package,
        "relocation": {
            "audio_root": spec.relocation.audio_root.as_posix(),
            "backup_root": spec.relocation.backup_root.as_posix(),
            "source_cache": spec.relocation.source_cache.as_posix(),
            "evidence_root": spec.relocation.evidence_root.as_posix(),
        },
        "budget": spec.budget.identity(),
        "model": spec.model.identity(),
        "mixture": {
            "quotas": spec.mixture.quotas,
            "cycle_examples": spec.mixture.cycle_examples,
            "effective_batch": spec.mixture.effective_batch,
            "updates_per_cycle": spec.mixture.updates_per_cycle,
            "seed": spec.mixture.seed,
        },
        "required_corpora": list(spec.required_corpora),
    }


@dataclass(frozen=True)
class RecordingRow:
    """One accepted or rejected parent recording."""

    recording_id: str
    parent_id: str
    corpus: str
    split: str
    device_view: str
    label_tier: str
    licence: LicenceDecision
    audio_sha256: str | None
    label_sha256: str | None
    sample_count: int | None
    rejected: bool
    rejection_reason: str | None

    def can_sample(self) -> bool:
        """Return whether this row may enter the sampler."""

        return (
            not self.rejected
            and self.licence is LicenceDecision.ACCEPTED_CC
            and self.split == "train"
            and self.audio_sha256 is not None
            and self.label_sha256 is not None
            and self.sample_count is not None
            and self.sample_count > 0
            and self.label_tier in LABEL_TIERS
        )


def parse_recording_row(payload: Any, label: str = "recording") -> RecordingRow:
    """Parse one recordings.jsonl object."""

    data = _require_object(payload, label)
    allowed = {
        "recording_id",
        "parent_id",
        "corpus",
        "split",
        "device_view",
        "label_tier",
        "licence",
        "audio_sha256",
        "label_sha256",
        "sample_count",
        "rejected",
        "rejection_reason",
        "language",
        "transformations",
    }
    _reject_unknown(data, allowed, label)
    try:
        licence = LicenceDecision(_require_str(data, "licence", label))
    except ValueError as error:
        raise ContractError("unknown licence decision") from error
    split = _require_str(data, "split", label)
    if split not in SPLITS:
        raise ContractError("split must be train, dev, or test", {"split": split})
    row = RecordingRow(
        recording_id=_require_str(data, "recording_id", label),
        parent_id=_require_str(data, "parent_id", label),
        corpus=_require_str(data, "corpus", label),
        split=split,
        device_view=_require_str(data, "device_view", label),
        label_tier=_require_str(data, "label_tier", label),
        licence=licence,
        audio_sha256=data.get("audio_sha256"),
        label_sha256=data.get("label_sha256"),
        sample_count=None if data.get("sample_count") is None else int(data["sample_count"]),
        rejected=bool(data.get("rejected", False)),
        rejection_reason=data.get("rejection_reason"),
    )
    if row.rejected and not row.rejection_reason:
        raise ContractError("rejected recordings require a reason", {"recording_id": row.recording_id})
    if row.licence is LicenceDecision.UNRESOLVED and not row.rejected:
        raise ContractError("unresolved licence rows cannot be accepted", {"recording_id": row.recording_id})
    return row


def parse_kinded_lock(payload: Any, expected_kind: str) -> dict[str, Any]:
    """Parse a kinded lock and reject promotion-incompatible fields."""

    data = _require_object(payload, expected_kind)
    kind = _require_str(data, "kind", expected_kind)
    if kind != expected_kind:
        raise ContractError(
            f"lock kind must be {expected_kind}",
            {"actual": kind, "expected": expected_kind},
        )
    if expected_kind == PREPARATION_KIND:
        present = sorted(LAUNCH_ONLY_FIELDS.intersection(data) | LEASE_ONLY_FIELDS.intersection(data))
        if present:
            raise ContractError("preparation lock cannot carry lease or launch fields", {"fields": present})
    if expected_kind == QUALIFICATION_LEASE_KIND:
        present = sorted((LAUNCH_ONLY_FIELDS - LEASE_ONLY_FIELDS).intersection(data))
        if present:
            raise ContractError("qualification lease cannot parse as a launch lock", {"fields": present})
        for key in ("offer", "rates", "hard_deadline", "backup_target", "lease_id"):
            if key not in data:
                raise ContractError(f"qualification lease missing {key}")
    if expected_kind == LAUNCH_KIND:
        for key in (
            "offer",
            "physical_batch",
            "accumulation",
            "affordable_cycles",
            "worker_deadline",
            "qualification_digest",
            "launch_id",
        ):
            if key not in data:
                raise ContractError(f"launch lock missing {key}")
        if data.get("gpu_qualification_status") in (None, "not_run"):
            raise ContractError("launch lock requires a real GPU qualification")
    return data


def lock_digest(payload: Mapping[str, Any]) -> str:
    """Return the content digest of a lock object."""

    return sha256_json(dict(payload))


@dataclass
class CoverageState:
    """Committed parent-recording coverage keyed by finite stream."""

    seen: dict[str, set[str]] = field(default_factory=dict)
    denominators: dict[str, int] = field(default_factory=dict)

    def state_dict(self) -> dict[str, object]:
        """Return a JSON-serializable coverage snapshot."""

        return {
            "seen": {corpus: sorted(parents) for corpus, parents in sorted(self.seen.items())},
            "denominators": dict(self.denominators),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> CoverageState:
        """Restore committed coverage."""

        seen_raw = payload.get("seen") or {}
        if not isinstance(seen_raw, Mapping):
            raise ContractError("coverage.seen must be an object")
        seen = {str(corpus): set(parents) for corpus, parents in seen_raw.items()}
        denominators_raw = payload.get("denominators") or {}
        if not isinstance(denominators_raw, Mapping):
            raise ContractError("coverage.denominators must be an object")
        denominators = {str(corpus): int(value) for corpus, value in denominators_raw.items()}
        return cls(seen=seen, denominators=denominators)

    def complete(self) -> bool:
        """Return whether every finite stream has 100% parent coverage."""

        if not self.denominators:
            return False
        for corpus, expected in self.denominators.items():
            if expected <= 0:
                return False
            if len(self.seen.get(corpus, ())) < expected:
                return False
        return True
