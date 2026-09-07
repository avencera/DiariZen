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
    ACCEPTED_CUSTOM = "accepted_custom"
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
            and self.licence in {LicenceDecision.ACCEPTED_CC, LicenceDecision.ACCEPTED_CUSTOM}
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


DATA_PREPARATION_SCHEMA = "speakrs-data-preparation"
DATA_PREPARATION_SCHEMA_VERSION = 1
DATA_FORBIDDEN_FIELDS = frozenset(
    {
        "budget",
        "offer",
        "instance",
        "gpu",
        "gpu_profile",
        "launch_id",
        "lease_id",
        "token",
        "password",
        "secret",
        "ssh_key",
        "hf_token",
        "vast_api_key",
        "signed_url",
        "access_key",
        "secret_key",
        "aws_secret_access_key",
        "aws_access_key_id",
    }
)
HEX64 = frozenset("0123456789abcdef")
REQUIRED_USES = (
    "commercial_training",
    "model_distribution",
    "derived_labels",
    "private_object_storage",
)
CAPACITY_PROFILES = (
    {"chunk_seconds": 8, "max_overlap": 2, "local_slots": 4},
    {"chunk_seconds": 8, "max_overlap": 4, "local_slots": 4},
    {"chunk_seconds": 16, "max_overlap": 2, "local_slots": 4},
    {"chunk_seconds": 16, "max_overlap": 4, "local_slots": 4},
)
CAPACITY_LOSS_LIMIT = 0.005
QA_OVERALL_LIMIT = 0.05
QA_STRATUM_LIMIT = 0.10


class UseDecision(str, Enum):
    """One required-use decision."""

    PERMITTED = "permitted"
    PROHIBITED = "prohibited"
    UNRESOLVED = "unresolved"


class SourceMembership(str, Enum):
    """Terminal membership for one matrix row."""

    ACCEPTED = "accepted"
    EXCLUDED = "excluded"
    PENDING = "pending"


class SourcePermissionState(str, Enum):
    """Permission state. Adapters cannot assign this."""

    UNRESOLVED = "unresolved"
    PERMITTED = "permitted"
    EXCLUDED = "excluded"


class SelectionState(str, Enum):
    """Batch verification state, distinct from complete-source profile admission"""

    STAGED = "staged"
    CONTENT_VERIFIED = "content-verified"
    VERIFIED = "qa-split-verified-capacity-measured"
    # retained for the immutable first-generation remote receipt reader
    ACCEPTED = "qa-split-capacity-accepted"


class ObjectState(str, Enum):
    """One remote object state."""

    PLANNED = "planned"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    READBACK_VERIFIED = "readback-verified"


class BatchState(str, Enum):
    """One remote batch state."""

    DRAFT = "draft"
    OBJECTS_VERIFIED = "accepted-selection-objects-verified"
    COMMITTED = "committed"


class RemoteReleaseState(str, Enum):
    """Complete-release state. A batch commit is not this."""

    DRAFT = "draft"
    BATCHES_ACCOUNTED = "required-batches-accounted"
    COMMITTED = "committed"


class LocalCopyState(str, Enum):
    """Local file lifecycle."""

    RETAINED = "retained"
    EVICTION_ELIGIBLE = "eviction-eligible"
    EVICTED = "evicted"


class LabelMethod(str, Enum):
    """How labels were produced. Not a permission field."""

    HUMAN_GOLD = "human-gold"
    HUMAN_REFERENCE_PILOT = "human-reference-pilot"
    CHANNEL_DERIVED = "channel-derived"
    MACHINE_UNREVIEWED = "machine-unreviewed"
    TRANSCRIPT_VAD_SELF_AGREEMENT = "transcript-vad-self-agreement"


class TermsClass(str, Enum):
    """Family of inspected terms. Custom is never accepted_cc."""

    CC_BY = "cc-by"
    CC_BY_SA = "cc-by-sa"
    CC_BY_ND = "cc-by-nd"
    CC0 = "cc0"
    CUSTOM = "custom"
    UNINSPECTED = "uninspected"


def is_placeholder_hash(value: Any) -> bool:
    """Return whether a digest is missing, repeated, or otherwise unusable."""

    if not isinstance(value, str) or len(value) != 64:
        return True
    lowered = value.lower()
    if any(char not in HEX64 for char in lowered):
        return True
    return len(set(lowered)) == 1


def require_content_hash(value: Any, label: str) -> str:
    """Return a real SHA-256 hex digest."""

    if is_placeholder_hash(value):
        raise ContractError(
            "placeholder hashes cannot seal a real release",
            {"label": label, "value": None if not isinstance(value, str) else value[:8]},
        )
    return str(value).lower()


def _require_bool(payload: Mapping[str, Any], key: str, label: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ContractError(f"{label}.{key} must be a boolean")
    return value


def _require_int(payload: Mapping[str, Any], key: str, label: str, *, minimum: int | None = None) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{label}.{key} must be >= {minimum}", {"actual": value})
    return value


@dataclass(frozen=True)
class UseRecord:
    """One required use with a clause reference."""

    decision: UseDecision
    clause_ref: str


def parse_use_record(payload: Any, label: str) -> UseRecord:
    """Parse one required-use decision."""

    data = _require_object(payload, label)
    _reject_unknown(data, {"decision", "clause_ref"}, label)
    try:
        decision = UseDecision(_require_str(data, "decision", label))
    except ValueError as error:
        raise ContractError(f"{label}.decision is unknown") from error
    return UseRecord(decision=decision, clause_ref=_require_str(data, "clause_ref", label))


@dataclass(frozen=True)
class PermissionRecord:
    """Validated permission evidence. Source adapters cannot construct this."""

    record_id: str
    source: str
    version: str
    terms_url: str
    terms_sha256: str
    terms_class: TermsClass
    recipient: str
    access_state: str
    uses: dict[str, UseRecord]
    reviewer: str
    adapter_name: str | None = None

    def permitted_for_training_storage(self) -> bool:
        """Return whether every required use is permitted."""

        return (
            bool(self.uses)
            and set(self.uses) == set(REQUIRED_USES)
            and self.access_state in {"obtained", "public", "licensed"}
            and all(record.decision is UseDecision.PERMITTED for record in self.uses.values())
        )


def parse_permission_record(payload: Any, label: str = "permission") -> PermissionRecord:
    """Parse a permission record and reject adapter self-approval."""

    data = _require_object(payload, label)
    allowed = {
        "record_id",
        "source",
        "version",
        "terms_url",
        "terms_sha256",
        "terms_class",
        "recipient",
        "access_state",
        "uses",
        "reviewer",
        "adapter_name",
    }
    _reject_unknown(data, allowed, label)
    if data.get("adapter_approved") or data.get("self_approved"):
        raise ContractError("source adapters cannot self-approve permission", {"label": label})
    try:
        terms_class = TermsClass(_require_str(data, "terms_class", label))
    except ValueError as error:
        raise ContractError(f"{label}.terms_class is unknown") from error
    uses_raw = _require_object(data.get("uses"), f"{label}.uses")
    # retain the old wire spelling only at the input boundary
    if "private_tigris" in uses_raw and "private_object_storage" not in uses_raw:
        uses_raw = {
            ("private_object_storage" if key == "private_tigris" else key): value for key, value in uses_raw.items()
        }
    if set(uses_raw) != set(REQUIRED_USES):
        raise ContractError(
            f"{label}.uses must name every required use",
            {"actual": sorted(uses_raw), "required": list(REQUIRED_USES)},
        )
    uses = {name: parse_use_record(uses_raw[name], f"{label}.uses.{name}") for name in REQUIRED_USES}
    record = PermissionRecord(
        record_id=_require_str(data, "record_id", label),
        source=_require_str(data, "source", label),
        version=_require_str(data, "version", label),
        terms_url=_require_str(data, "terms_url", label),
        terms_sha256=require_content_hash(data.get("terms_sha256"), f"{label}.terms_sha256"),
        terms_class=terms_class,
        recipient=_require_str(data, "recipient", label),
        access_state=_require_str(data, "access_state", label),
        uses=uses,
        reviewer=_require_str(data, "reviewer", label),
        adapter_name=data.get("adapter_name") if isinstance(data.get("adapter_name"), str) else None,
    )
    if record.terms_class is TermsClass.CUSTOM:
        for use_name, use in record.uses.items():
            if use.decision is UseDecision.PERMITTED and "accepted_cc" in use.clause_ref.lower():
                raise ContractError(
                    "unapproved custom agreement cannot be relabeled as accepted CC",
                    {"source": record.source, "use": use_name},
                )
    return record


def licence_from_permission(record: PermissionRecord) -> LicenceDecision:
    """Map a permission record onto the existing licence enum without fake CC labels."""

    if record.terms_class is TermsClass.CUSTOM:
        if record.permitted_for_training_storage():
            return LicenceDecision.ACCEPTED_CUSTOM
        if any(use.decision is UseDecision.PROHIBITED for use in record.uses.values()):
            return LicenceDecision.REJECTED_UNKNOWN
        return LicenceDecision.UNRESOLVED
    if record.terms_class is TermsClass.UNINSPECTED:
        return LicenceDecision.UNRESOLVED
    if not record.permitted_for_training_storage():
        if any(use.decision is UseDecision.PROHIBITED for use in record.uses.values()):
            return LicenceDecision.REJECTED_UNKNOWN
        return LicenceDecision.UNRESOLVED
    if record.terms_class in {TermsClass.CC_BY, TermsClass.CC_BY_SA, TermsClass.CC0}:
        return LicenceDecision.ACCEPTED_CC
    if record.terms_class is TermsClass.CC_BY_ND:
        return LicenceDecision.UNRESOLVED
    return LicenceDecision.UNRESOLVED


@dataclass(frozen=True)
class SourceRecord:
    """One matrix source. Permission is referenced, never self-assigned."""

    name: str
    version: str
    membership: SourceMembership
    permission_id: str
    permission_state: SourcePermissionState
    missing_action: str | None
    private_evidence_id: str
    disposition_evidence: EvidenceReference | None = None

    def __post_init__(self) -> None:
        if (self.membership is SourceMembership.EXCLUDED) != (self.disposition_evidence is not None):
            raise ContractError("excluded sources require a disposition reference; other states cannot carry one")


def parse_source_record(
    payload: Any,
    permissions: Mapping[str, PermissionRecord],
    label: str = "source",
) -> SourceRecord:
    """Parse a source membership row and bind it to a permission record."""

    data = _require_object(payload, label)
    _reject_unknown(
        data,
        {
            "name",
            "version",
            "membership",
            "permission_id",
            "permission_state",
            "missing_action",
            "private_evidence_id",
            "disposition_evidence",
            "licence",
        },
        label,
    )
    if "licence" in data:
        raise ContractError("source adapters cannot self-approve permission", {"field": "licence"})
    try:
        membership = SourceMembership(_require_str(data, "membership", label))
    except ValueError as error:
        raise ContractError(f"{label}.membership is unknown") from error
    try:
        permission_state = SourcePermissionState(_require_str(data, "permission_state", label))
    except ValueError as error:
        raise ContractError(f"{label}.permission_state is unknown") from error
    permission_id = _require_str(data, "permission_id", label)
    permission = permissions.get(permission_id)
    if permission is None:
        raise ContractError("source permission record is missing", {"permission_id": permission_id})
    if permission.source != _require_str(data, "name", label) or permission.version != _require_str(
        data, "version", label
    ):
        raise ContractError("permission record does not match source name and version")
    if permission_state is SourcePermissionState.PERMITTED and not permission.permitted_for_training_storage():
        raise ContractError("permitted state requires every required use to be permitted")
    if membership is SourceMembership.ACCEPTED and permission_state is not SourcePermissionState.PERMITTED:
        raise ContractError("accepted sources require a permitted permission state")
    if membership is SourceMembership.ACCEPTED and not permission.permitted_for_training_storage():
        raise ContractError("accepted sources require every required use to be permitted")
    if membership is SourceMembership.PENDING and not data.get("missing_action"):
        raise ContractError("pending sources must name the missing action")
    return SourceRecord(
        name=_require_str(data, "name", label),
        version=_require_str(data, "version", label),
        membership=membership,
        permission_id=permission_id,
        permission_state=permission_state,
        missing_action=data.get("missing_action") if isinstance(data.get("missing_action"), str) else None,
        private_evidence_id=_require_str(data, "private_evidence_id", label),
        disposition_evidence=EvidenceReference.parse(data["disposition_evidence"], f"{label}.disposition_evidence")
        if data.get("disposition_evidence") is not None
        else None,
    )


@dataclass(frozen=True)
class DiskLimits:
    """Measured finite staging and cache caps."""

    staging_root: Path
    cache_root: Path
    max_staging_bytes: int
    max_cache_bytes: int
    free_space_reserve_bytes: int
    concurrency: int


def parse_disk_limits(payload: Any, label: str = "disk") -> DiskLimits:
    """Parse finite disk limits. Zero or missing caps fail."""

    data = _require_object(payload, label)
    _reject_unknown(
        data,
        {
            "staging_root",
            "cache_root",
            "max_staging_bytes",
            "max_cache_bytes",
            "free_space_reserve_bytes",
            "concurrency",
        },
        label,
    )
    return DiskLimits(
        staging_root=Path(_require_str(data, "staging_root", label)),
        cache_root=Path(_require_str(data, "cache_root", label)),
        max_staging_bytes=_require_int(data, "max_staging_bytes", label, minimum=1),
        max_cache_bytes=_require_int(data, "max_cache_bytes", label, minimum=1),
        free_space_reserve_bytes=_require_int(data, "free_space_reserve_bytes", label, minimum=0),
        concurrency=_require_int(data, "concurrency", label, minimum=1),
    )


@dataclass(frozen=True)
class ObjectStoreDestination:
    """Private object-store destination. Stores a credential reference, not a secret."""

    provider: str
    endpoint: str
    bucket: str
    prefix: str
    credential_reference: str
    region: str = "auto"


def parse_r2_destination(payload: Any, label: str = "r2") -> ObjectStoreDestination:
    """Parse the private R2 destination. Secrets in-line fail."""

    data = _require_object(payload, label)
    _reject_unknown(
        data,
        {"provider", "endpoint", "bucket", "prefix", "credential_reference", "region"},
        label,
    )
    for key in ("access_key", "secret_key", "session_token", "signed_url"):
        if key in data:
            raise ContractError("r2 destination must not store credentials", {"key": key})
    prefix = _require_str(data, "prefix", label).strip("/")
    if not prefix:
        raise ContractError("r2.prefix must be a non-empty task prefix")
    provider = data.get("provider") if isinstance(data.get("provider"), str) and data.get("provider") else "r2"
    if provider != "r2":
        raise ContractError("object store provider must be r2", {"provider": provider})
    return ObjectStoreDestination(
        provider="r2",
        endpoint=_require_str(data, "endpoint", label),
        bucket=_require_str(data, "bucket", label),
        prefix=prefix,
        credential_reference=_require_str(data, "credential_reference", label),
        region=data.get("region") if isinstance(data.get("region"), str) and data.get("region") else "auto",
    )


@dataclass(frozen=True)
class ModelProfileSettings:
    """Window and slot settings used for capacity admission."""

    sample_rate: int
    local_slots: int
    chunk_seconds: tuple[int, ...]
    max_overlap: tuple[int, ...]
    output_frames_8: int
    rf_duration: float
    rf_step: float
    chunk_shifts: tuple[int, int] = (6, 12)


def parse_model_profile_settings(payload: Any, label: str = "profiles") -> ModelProfileSettings:
    """Parse capacity-profile settings without inheriting training RunSpec fields."""

    data = _require_object(payload, label)
    _reject_unknown(
        data,
        {
            "sample_rate",
            "local_slots",
            "chunk_seconds",
            "max_overlap",
            "output_frames_8",
            "rf_duration",
            "rf_step",
            "chunk_shifts",
        },
        label,
    )
    chunks = data.get("chunk_seconds")
    overlaps = data.get("max_overlap")
    if not isinstance(chunks, list) or sorted(chunks) != [8, 16]:
        raise ContractError("profiles.chunk_seconds must be [8, 16]")
    if not isinstance(overlaps, list) or sorted(overlaps) != [2, 4]:
        raise ContractError("profiles.max_overlap must be [2, 4]")
    slots = _require_int(data, "local_slots", label, minimum=1)
    if slots != 4:
        raise ContractError("profiles.local_slots must be 4", {"actual": slots})
    shifts = data.get("chunk_shifts", [6, 12])
    if shifts != [6, 12]:
        raise ContractError("profiles.chunk_shifts must be the frozen 8/6 and 16/12 grids")
    if data.get("sample_rate") != SAMPLE_RATE or data.get("output_frames_8") != OUTPUT_FRAMES:
        raise ContractError("capacity must use the current model sample rate and output frame count")
    if data.get("rf_duration") != 0.025 or data.get("rf_step") != 0.020:
        raise ContractError("capacity must use the current model receptive-field grid")
    return ModelProfileSettings(
        sample_rate=_require_int(data, "sample_rate", label, minimum=1),
        local_slots=slots,
        chunk_seconds=(8, 16),
        max_overlap=(2, 4),
        output_frames_8=_require_int(data, "output_frames_8", label, minimum=1),
        rf_duration=float(_require_number(data, "rf_duration", label)),
        rf_step=float(_require_number(data, "rf_step", label)),
        chunk_shifts=(6, 12),
    )


DEFAULT_DATA_PROFILES = ModelProfileSettings(
    sample_rate=SAMPLE_RATE,
    local_slots=4,
    chunk_seconds=(8, 16),
    max_overlap=(2, 4),
    output_frames_8=OUTPUT_FRAMES,
    rf_duration=0.025,
    rf_step=0.020,
)


@dataclass(frozen=True)
class DataPreparationSpec:
    """Versioned data-only specification. Rejects training and lease fields."""

    schema: str
    schema_version: int
    release_id: str
    sources: tuple[SourceRecord, ...]
    permissions: dict[str, PermissionRecord]
    qa_policy_path: Path
    frozen_splits: dict[str, dict[str, tuple[str, ...]]]
    profiles: ModelProfileSettings
    disk: DiskLimits
    r2: ObjectStoreDestination
    permission_records_path: Path
    private_evidence_root: Path
    local_audio_root: Path | None = None
    local_source_cache: Path | None = None


def parse_data_preparation_spec(payload: Any) -> DataPreparationSpec:
    """Parse the data-only spec and reject training, lease, and secret fields."""

    data = _require_object(payload, "data-spec")
    present_forbidden = sorted(
        set(data).intersection(DATA_FORBIDDEN_FIELDS) | set(LAUNCH_ONLY_FIELDS.intersection(data))
    )
    if present_forbidden:
        raise ContractError(
            "data-only specification cannot carry training, lease, or secret fields",
            {"fields": present_forbidden},
        )
    if "tigris" in data:
        raise ContractError("tigris destination is retired; use r2")
    allowed = {
        "schema",
        "schema_version",
        "release_id",
        "sources",
        "permissions",
        "qa_policy_path",
        "frozen_splits",
        "profiles",
        "disk",
        "r2",
        "permission_records_path",
        "private_evidence_root",
        "local_audio_root",
        "local_source_cache",
    }
    _reject_unknown(data, allowed, "data-spec")
    if data.get("schema") != DATA_PREPARATION_SCHEMA:
        raise ContractError("spec.schema must be speakrs-data-preparation")
    if int(_require_number(data, "schema_version", "data-spec")) != DATA_PREPARATION_SCHEMA_VERSION:
        raise ContractError("spec.schema_version must be 1")
    permissions_raw = data.get("permissions")
    if not isinstance(permissions_raw, list) or not permissions_raw:
        raise ContractError("data-spec.permissions must be a non-empty array")
    permissions = {}
    for index, item in enumerate(permissions_raw):
        record = parse_permission_record(item, f"permissions[{index}]")
        if record.record_id in permissions:
            raise ContractError("duplicate permission record_id", {"record_id": record.record_id})
        permissions[record.record_id] = record
    sources_raw = data.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ContractError("data-spec.sources must be a non-empty array")
    sources = tuple(
        parse_source_record(item, permissions, f"sources[{index}]") for index, item in enumerate(sources_raw)
    )
    if len({source.name for source in sources}) != len(sources):
        raise ContractError("data-spec sources must have unique names")
    frozen_raw = _require_object(data.get("frozen_splits"), "frozen_splits")
    frozen: dict[str, dict[str, tuple[str, ...]]] = {}
    for corpus, splits in frozen_raw.items():
        split_obj = _require_object(splits, f"frozen_splits.{corpus}")
        frozen[str(corpus)] = {split: tuple(ids) if isinstance(ids, list) else () for split, ids in split_obj.items()}
    return DataPreparationSpec(
        schema=DATA_PREPARATION_SCHEMA,
        schema_version=DATA_PREPARATION_SCHEMA_VERSION,
        release_id=_require_str(data, "release_id", "data-spec"),
        sources=sources,
        permissions=permissions,
        qa_policy_path=Path(_require_str(data, "qa_policy_path", "data-spec")),
        frozen_splits=frozen,
        profiles=parse_model_profile_settings(data.get("profiles")),
        disk=parse_disk_limits(data.get("disk")),
        r2=parse_r2_destination(data.get("r2")),
        permission_records_path=Path(_require_str(data, "permission_records_path", "data-spec")),
        private_evidence_root=Path(_require_str(data, "private_evidence_root", "data-spec")),
        local_audio_root=_optional_path(data, "local_audio_root"),
        local_source_cache=_optional_path(data, "local_source_cache"),
    )


def data_spec_to_json(spec: DataPreparationSpec) -> dict[str, object]:
    """Serialize a data spec without secrets."""

    return {
        "schema": spec.schema,
        "schema_version": spec.schema_version,
        "release_id": spec.release_id,
        "sources": [
            {
                "name": source.name,
                "version": source.version,
                "membership": source.membership.value,
                "permission_id": source.permission_id,
                "permission_state": source.permission_state.value,
                "missing_action": source.missing_action,
                "private_evidence_id": source.private_evidence_id,
                "disposition_evidence": {
                    "path": str(source.disposition_evidence.path),
                    "sha256": source.disposition_evidence.sha256,
                }
                if source.disposition_evidence is not None
                else None,
            }
            for source in spec.sources
        ],
        "permissions": [
            {
                "record_id": record.record_id,
                "source": record.source,
                "version": record.version,
                "terms_url": record.terms_url,
                "terms_sha256": record.terms_sha256,
                "terms_class": record.terms_class.value,
                "recipient": record.recipient,
                "access_state": record.access_state,
                "uses": {
                    name: {"decision": use.decision.value, "clause_ref": use.clause_ref}
                    for name, use in record.uses.items()
                },
                "reviewer": record.reviewer,
                "adapter_name": record.adapter_name,
            }
            for record in spec.permissions.values()
        ],
        "qa_policy_path": spec.qa_policy_path.as_posix(),
        "frozen_splits": {
            corpus: {split: list(ids) for split, ids in splits.items()}
            for corpus, splits in spec.frozen_splits.items()
        },
        "profiles": {
            "sample_rate": spec.profiles.sample_rate,
            "local_slots": spec.profiles.local_slots,
            "chunk_seconds": list(spec.profiles.chunk_seconds),
            "max_overlap": list(spec.profiles.max_overlap),
            "output_frames_8": spec.profiles.output_frames_8,
            "rf_duration": spec.profiles.rf_duration,
            "rf_step": spec.profiles.rf_step,
            "chunk_shifts": list(spec.profiles.chunk_shifts),
        },
        "disk": {
            "staging_root": spec.disk.staging_root.as_posix(),
            "cache_root": spec.disk.cache_root.as_posix(),
            "max_staging_bytes": spec.disk.max_staging_bytes,
            "max_cache_bytes": spec.disk.max_cache_bytes,
            "free_space_reserve_bytes": spec.disk.free_space_reserve_bytes,
            "concurrency": spec.disk.concurrency,
        },
        "r2": {
            "provider": spec.r2.provider,
            "endpoint": spec.r2.endpoint,
            "bucket": spec.r2.bucket,
            "prefix": spec.r2.prefix,
            "credential_reference": spec.r2.credential_reference,
            "region": spec.r2.region,
        },
        "permission_records_path": spec.permission_records_path.as_posix(),
        "private_evidence_root": spec.private_evidence_root.as_posix(),
        "local_audio_root": spec.local_audio_root.as_posix() if spec.local_audio_root else None,
        "local_source_cache": spec.local_source_cache.as_posix() if spec.local_source_cache else None,
    }


def parse_old_release_identity(payload: Any) -> dict[str, Any]:
    """Parse an old sealed identity and refuse to migrate it onto a data spec."""

    data = _require_object(payload, "old-release")
    if data.get("schema") == DATA_PREPARATION_SCHEMA:
        raise ContractError("old release identity cannot parse as a data-preparation spec")
    if data.get("kind") in {LAUNCH_KIND, QUALIFICATION_LEASE_KIND}:
        return parse_kinded_lock(data, data["kind"])
    if data.get("schema") == SCHEMA_NAME:
        return spec_to_json(parse_spec(data))
    raise ContractError("unknown old identity")


@dataclass(frozen=True)
class ObjectReceipt:
    """One content-addressed object with a verified state."""

    key: str
    sha256: str
    size: int
    purpose: str
    parent_id: str | None
    source: str
    state: ObjectState
    codec: str | None
    etag: str | None = None
    encryption: str | None = None
    public: bool = False


def parse_object_receipt(payload: Any, label: str = "object") -> ObjectReceipt:
    """Parse an object receipt. ETag is never accepted as SHA-256."""

    data = _require_object(payload, label)
    _reject_unknown(
        data,
        {
            "key",
            "sha256",
            "size",
            "purpose",
            "parent_id",
            "source",
            "state",
            "codec",
            "etag",
            "encryption",
            "public",
        },
        label,
    )
    try:
        state = ObjectState(_require_str(data, "state", label))
    except ValueError as error:
        raise ContractError(f"{label}.state is unknown") from error
    sha256 = require_content_hash(data.get("sha256"), f"{label}.sha256")
    etag = data.get("etag") if isinstance(data.get("etag"), str) else None
    receipt = ObjectReceipt(
        key=_require_str(data, "key", label),
        sha256=sha256,
        size=_require_int(data, "size", label, minimum=1),
        purpose=_require_str(data, "purpose", label),
        parent_id=data.get("parent_id") if isinstance(data.get("parent_id"), str) else None,
        source=_require_str(data, "source", label),
        state=state,
        codec=data.get("codec") if isinstance(data.get("codec"), str) else None,
        etag=etag,
        encryption=data.get("encryption") if isinstance(data.get("encryption"), str) else None,
        public=bool(data.get("public", False)),
    )
    if receipt.public:
        raise ContractError("public objects cannot enter a private release", {"key": receipt.key})
    if state is ObjectState.READBACK_VERIFIED and not receipt.encryption:
        raise ContractError("missing encryption evidence", {"key": receipt.key})
    return receipt


def assert_state_transition(current: Enum, target: Enum, allowed: Mapping[Enum, set[Enum]], label: str) -> None:
    """Reject an illegal typed-state transition."""

    permitted = allowed.get(current, set())
    if target not in permitted:
        raise ContractError(
            f"illegal {label} transition",
            {"from": current.value, "to": target.value},
        )


@dataclass(frozen=True)
class EvidenceReference:
    """Content-bound private evidence required by a selection check"""

    path: Path
    sha256: str

    @classmethod
    def parse(cls, payload: Any, label: str) -> EvidenceReference:
        """Parse a file reference without trusting its current local contents"""

        data = _require_object(payload, label)
        _reject_unknown(data, {"path", "sha256"}, label)
        return cls(Path(_require_str(data, "path", label)), require_content_hash(data.get("sha256"), label))


@dataclass(frozen=True)
class VerifiedSelectionBatch:
    """A verified storage batch whose source profile admission requires complete closure"""

    source: str
    version: str
    manifest_sha256: str
    spec_sha256: str
    policy_sha256: str
    evidence_hashes: dict[str, str]
    artifact_hashes: dict[str, str]
    provisional_profiles: tuple[tuple[int, int], ...]
    selected_parent_ids: tuple[str, ...]
    required_parent_ids: tuple[str, ...]

    @classmethod
    def parse(cls, payload: Any) -> VerifiedSelectionBatch:
        """Read verified batches and the stronger immutable first-generation receipts"""

        data = _require_object(payload, "selection-acceptance")
        legacy = data.get("schema") == "speakrs-selection-acceptance-v1"
        expected_state = SelectionState.ACCEPTED if legacy else SelectionState.VERIFIED
        if (
            data.get("schema") not in {"speakrs-selection-acceptance-v1", "speakrs-batch-verification-v2"}
            or data.get("state") != expected_state.value
        ):
            raise ContractError("a verified batch receipt is required")
        if not legacy and (data.get("training_ready") is not False or data.get("capacity_scope") != "batch"):
            raise ContractError("batch verification cannot claim complete-source training admission")
        evidence = _require_object(data.get("evidence_hashes"), "evidence_hashes")
        artifacts = _require_object(data.get("artifact_hashes"), "artifact_hashes")
        if not evidence or not artifacts:
            raise ContractError("acceptance requires evidence and artifact identities")
        profiles = data.get("admitted_profiles" if legacy else "provisional_profiles")
        if not isinstance(profiles, list) or (legacy and not profiles):
            raise ContractError("batch profile measurements are missing")
        parsed_profiles = tuple((int(item["chunk_seconds"]), int(item["max_overlap"])) for item in profiles)
        if len(set(parsed_profiles)) != len(parsed_profiles) or any(
            chunk not in {8, 16} or overlap not in {2, 4} for chunk, overlap in parsed_profiles
        ):
            raise ContractError("acceptance profiles are invalid")
        capacity = data.get("capacity")
        if not isinstance(capacity, list) or len(capacity) != 4:
            raise ContractError("batch verification requires all four measured capacity profiles")
        measured_profiles = {
            (
                _require_int(item, "chunk_seconds", "capacity", minimum=1),
                _require_int(item, "max_overlap", "capacity", minimum=1),
            )
            for item in capacity
            if isinstance(item, Mapping)
        }
        if measured_profiles != {(8, 2), (8, 4), (16, 2), (16, 4)}:
            raise ContractError("batch capacity measurements have missing or duplicate profile identities")
        selected = data.get("selected_parent_ids")
        required = data.get("required_parent_ids")
        if not isinstance(selected, list) or not selected or not isinstance(required, list) or not required:
            raise ContractError("acceptance requires declared batch and full source parent inventories")
        if (
            len(set(selected)) != len(selected)
            or len(set(required)) != len(required)
            or not set(selected) <= set(required)
        ):
            raise ContractError("accepted batch must be a unique subset of required source parents")
        return cls(
            source=_require_str(data, "source", "selection-acceptance"),
            version=_require_str(data, "version", "selection-acceptance"),
            manifest_sha256=require_content_hash(data.get("manifest_sha256"), "manifest_sha256"),
            spec_sha256=require_content_hash(data.get("spec_sha256"), "spec_sha256"),
            policy_sha256=require_content_hash(data.get("policy_sha256"), "policy_sha256"),
            evidence_hashes={key: require_content_hash(value, key) for key, value in evidence.items()},
            artifact_hashes={key: require_content_hash(value, key) for key, value in artifacts.items()},
            provisional_profiles=parsed_profiles,
            selected_parent_ids=tuple(selected),
            required_parent_ids=tuple(required),
        )
