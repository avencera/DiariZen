"""Worker-side GPU qualification for the Speakrs WavLM Large recipe."""

from __future__ import annotations

import contextlib
import copy
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import toml

from .contracts import QualificationBinding, parse_qualification_binding
from .errors import LargeError
from .hashing import sha256_file, sha256_json, sha256_text
from .jsonio import write_json


QUALIFICATION_SCHEMA = "speakrs-gpu-qualification-v1"
QUALIFICATION_SCHEMA_VERSION = 1
DEFAULT_WARMUP_UPDATES = 20
DEFAULT_MEASURED_UPDATES = 100
DEFAULT_EFFECTIVE_BATCH = 64
DEFAULT_PHYSICAL_BATCH_CANDIDATES = (2, 4, 8)
DEFAULT_UPDATES_PER_CYCLE = 2_000
DEFAULT_MAX_CYCLES = 100
DEFAULT_HEADROOM = 0.15
EXPECTED_MODEL_CLASS = "diarizen.models.eend.model_wavlm_conformer.Model"
EXPECTED_TRAINER_CLASS = "trainer_dual_opt.Trainer"
EXPECTED_DATASET_CLASS = "dataset.DiarizationDataset"
UNMEASURED_OVERHEAD = (
    "validation time",
    "DER and scoring time",
    "checkpoint and recovery I/O outside the qualification reload check",
    "data staging, worker startup, and storage transfer time",
)


class QualificationError(LargeError):
    """A worker qualification failure."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__("qualification", message, details)


@dataclass(frozen=True)
class QualificationSpec:
    """Typed inputs that control one bounded qualification run."""

    warmup_optimizer_updates: int = DEFAULT_WARMUP_UPDATES
    measured_optimizer_updates: int = DEFAULT_MEASURED_UPDATES
    effective_batch: int = DEFAULT_EFFECTIVE_BATCH
    physical_batch_candidates: tuple[int, ...] = DEFAULT_PHYSICAL_BATCH_CANDIDATES
    planned_updates_per_cycle: int = DEFAULT_UPDATES_PER_CYCLE
    planned_max_cycles: int = DEFAULT_MAX_CYCLES
    minimum_memory_headroom: float = DEFAULT_HEADROOM
    output: Path | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> QualificationSpec:
        """Parse a JSON qualification spec and reject unknown or invalid fields."""

        allowed = {
            "schema",
            "schema_version",
            "warmup_optimizer_updates",
            "warmup_updates",
            "measured_optimizer_updates",
            "measured_updates",
            "effective_batch",
            "physical_batch_candidates",
            "planned_updates_per_cycle",
            "updates_per_cycle",
            "planned_max_cycles",
            "max_cycles",
            "minimum_memory_headroom",
            "memory_headroom_fraction",
            "output",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise QualificationError("qualification spec has unknown keys", {"unknown": unknown})
        schema = value.get("schema")
        if schema is not None and schema != QUALIFICATION_SCHEMA:
            raise QualificationError("qualification spec schema is not supported", {"schema": schema})
        version = value.get("schema_version")
        if version is not None and (
            isinstance(version, bool) or not isinstance(version, int) or version != QUALIFICATION_SCHEMA_VERSION
        ):
            raise QualificationError("qualification spec schema_version is not supported", {"version": version})

        def integer(name: str, alias: str | None, default: int) -> int:
            raw = value.get(name, value.get(alias, default) if alias else default)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise QualificationError(f"{name} must be an integer")
            return raw

        def number(name: str, alias: str | None, default: float) -> float:
            raw = value.get(name, value.get(alias, default) if alias else default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise QualificationError(f"{name} must be a number")
            return float(raw)

        candidates_raw = value.get("physical_batch_candidates", DEFAULT_PHYSICAL_BATCH_CANDIDATES)
        if not isinstance(candidates_raw, (list, tuple)):
            raise QualificationError("physical_batch_candidates must be a list of integers")
        candidates: list[int] = []
        for candidate in candidates_raw:
            if isinstance(candidate, bool) or not isinstance(candidate, int):
                raise QualificationError("physical_batch_candidates must contain integers")
            candidates.append(candidate)

        output_raw = value.get("output")
        output = None if output_raw is None else Path(str(output_raw))
        parsed = cls(
            warmup_optimizer_updates=integer("warmup_optimizer_updates", "warmup_updates", DEFAULT_WARMUP_UPDATES),
            measured_optimizer_updates=integer(
                "measured_optimizer_updates", "measured_updates", DEFAULT_MEASURED_UPDATES
            ),
            effective_batch=integer("effective_batch", None, DEFAULT_EFFECTIVE_BATCH),
            physical_batch_candidates=tuple(candidates),
            planned_updates_per_cycle=integer(
                "planned_updates_per_cycle", "updates_per_cycle", DEFAULT_UPDATES_PER_CYCLE
            ),
            planned_max_cycles=integer("planned_max_cycles", "max_cycles", DEFAULT_MAX_CYCLES),
            minimum_memory_headroom=number("minimum_memory_headroom", "memory_headroom_fraction", DEFAULT_HEADROOM),
            output=output,
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        """Reject values that cannot describe a bounded qualification."""

        positive = (
            ("warmup_optimizer_updates", self.warmup_optimizer_updates),
            ("measured_optimizer_updates", self.measured_optimizer_updates),
            ("effective_batch", self.effective_batch),
            ("planned_updates_per_cycle", self.planned_updates_per_cycle),
            ("planned_max_cycles", self.planned_max_cycles),
        )
        for name, value in positive:
            if value <= 0:
                raise QualificationError(f"{name} must be positive", {name: value})
        if not self.physical_batch_candidates:
            raise QualificationError("physical_batch_candidates cannot be empty")
        if len(set(self.physical_batch_candidates)) != len(self.physical_batch_candidates):
            raise QualificationError("physical_batch_candidates cannot contain duplicates")
        if any(value <= 0 for value in self.physical_batch_candidates):
            raise QualificationError("physical_batch_candidates must be positive")
        if any(self.effective_batch % value for value in self.physical_batch_candidates):
            raise QualificationError(
                "effective_batch must be divisible by every physical batch candidate",
                {"effective_batch": self.effective_batch, "candidates": list(self.physical_batch_candidates)},
            )
        if not DEFAULT_HEADROOM <= self.minimum_memory_headroom < 1:
            raise QualificationError(
                "minimum_memory_headroom must be at least fifteen percent and below one",
                {"minimum_memory_headroom": self.minimum_memory_headroom},
            )

    def accumulation_for(self, physical_batch: int) -> int:
        """Return gradient accumulation for one physical batch candidate."""

        if physical_batch <= 0 or self.effective_batch % physical_batch:
            raise QualificationError(
                "physical batch must divide effective batch",
                {"physical_batch": physical_batch, "effective_batch": self.effective_batch},
            )
        return self.effective_batch // physical_batch

    def identity(self) -> dict[str, object]:
        """Return a stable, JSON-compatible identity for hashing."""

        value = asdict(self)
        if self.output is not None:
            value["output"] = self.output.as_posix()
        return value


@dataclass(frozen=True)
class AttemptResult:
    """Observed facts from one probe or measured run."""

    physical_batch: int
    accumulation: int
    ok: bool
    total_memory_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    update_seconds: tuple[float, ...] = ()
    measured_update_count: int = 0
    examples_processed: int = 0
    checkpoint_write_reload: Mapping[str, object] | None = None
    error_code: str | None = None
    error_message: str | None = None
    headroom_fraction_override: float | None = None

    @property
    def headroom_fraction(self) -> float:
        """Return free VRAM fraction based on allocator-reserved peak."""

        if self.headroom_fraction_override is not None:
            return float(self.headroom_fraction_override)
        if self.total_memory_bytes <= 0:
            return 0.0
        return max(0.0, (self.total_memory_bytes - self.peak_reserved_bytes) / self.total_memory_bytes)

    def as_json(self) -> dict[str, object]:
        """Return JSON-safe attempt facts."""

        value = {
            "physical_batch": self.physical_batch,
            "accumulation": self.accumulation,
            "ok": self.ok,
            "total_memory_bytes": self.total_memory_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "headroom_fraction": self.headroom_fraction,
            "measured_update_count": self.measured_update_count,
            "examples_processed": self.examples_processed,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }
        if self.update_seconds:
            value.update(
                {
                    "p50_optimizer_update_seconds": percentile(self.update_seconds, 50),
                    "p95_optimizer_update_seconds": percentile(self.update_seconds, 95),
                    "examples_per_second": (self.examples_processed or len(self.update_seconds))
                    / sum(self.update_seconds),
                }
            )
        if self.checkpoint_write_reload is not None:
            value["checkpoint_write_reload"] = dict(self.checkpoint_write_reload)
        return value


def percentile(values: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    if not values:
        raise QualificationError("cannot compute a percentile from no timings")
    if not 0 <= percent <= 100:
        raise QualificationError("percent must be between zero and one hundred")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def estimate_training_hours(
    seconds_per_optimizer_update: float,
    planned_updates_per_cycle: int = DEFAULT_UPDATES_PER_CYCLE,
    planned_max_cycles: int = DEFAULT_MAX_CYCLES,
) -> dict[str, float]:
    """Estimate training-only time from optimizer-update time."""

    if not math.isfinite(seconds_per_optimizer_update) or seconds_per_optimizer_update <= 0:
        raise QualificationError("seconds_per_optimizer_update must be finite and positive")
    if planned_updates_per_cycle <= 0 or planned_max_cycles <= 0:
        raise QualificationError("planned cycle values must be positive")
    cycle_hours = seconds_per_optimizer_update * planned_updates_per_cycle / 3600.0
    return {
        "hours_per_cycle": cycle_hours,
        "hours_for_max_cycles": cycle_hours * planned_max_cycles,
    }


def select_largest_batch(
    probes: Iterable[AttemptResult], minimum_memory_headroom: float = DEFAULT_HEADROOM
) -> AttemptResult:
    """Select the largest successful candidate with the required VRAM headroom."""

    if not DEFAULT_HEADROOM <= minimum_memory_headroom < 1:
        raise QualificationError("memory headroom must be at least fifteen percent and below one")
    candidates = sorted(
        (probe for probe in probes if probe.ok and probe.headroom_fraction >= minimum_memory_headroom),
        key=lambda probe: probe.physical_batch,
        reverse=True,
    )
    if not candidates:
        raise QualificationError(
            "no physical batch candidate met the memory-headroom requirement",
            {"minimum_memory_headroom": minimum_memory_headroom},
        )
    return candidates[0]


class QualificationHooks:
    """Small duck-typed hook object used by the existing trainer loop."""

    def __init__(self, trainer: Any, *, warmup_updates: int, measured_updates: int) -> None:
        self.trainer = trainer
        self.warmup_updates = warmup_updates
        self.measured_updates = measured_updates
        self.optimizer_updates = 0
        self._pending_durations: list[float] = []
        self._microbatch_started: float | None = None
        self.update_seconds: list[float] = []

    def start_microbatch(self) -> None:
        """Start timing the current accumulated optimizer update."""

        if self._microbatch_started is None:
            self._microbatch_started = time.perf_counter()

    def finish_microbatch(self, synchronized: bool) -> None:
        """Queue one accumulated-update duration at the synchronization boundary."""

        if not synchronized:
            return
        if self._microbatch_started is None:
            return
        self._pending_durations.append(time.perf_counter() - self._microbatch_started)
        self._microbatch_started = None

    def example_loss_weight(self, _batch: Any) -> float:
        """Keep every real training example in the bounded qualification."""

        return 1.0

    def acknowledge_update(self, _batch: Any, *, optimizer_updated: bool, **_kwargs: Any) -> None:
        """Record optimizer updates and stop at the requested boundary."""

        duration = self._pending_durations.pop(0) if self._pending_durations else None
        if not optimizer_updated:
            return
        self.optimizer_updates += 1
        if self.optimizer_updates > self.warmup_updates:
            if duration is None:
                duration = 0.0
            self.update_seconds.append(float(duration))
        if self.optimizer_updates >= self.warmup_updates + self.measured_updates:
            self.trainer._stop_signal = "qualification_complete"

    def should_snapshot(self, _updates: int) -> bool:
        """Disable recovery snapshots during the short timing run."""

        return False

    def mark_snapshot(self, _updates: int) -> None:
        """Implement the trainer hook protocol."""

    def coverage_complete(self) -> bool:
        """Tell the trainer that this bounded run has no coverage gate."""

        return True

    def can_train(self) -> bool:
        """Allow a fresh qualification attempt to train."""

        return True

    def state_dict(self) -> dict[str, object]:
        """Return minimal hook state for the trainer checkpoint."""

        return {
            "optimizer_updates": self.optimizer_updates,
            "warmup_updates": self.warmup_updates,
            "measured_updates": self.measured_updates,
        }


def _is_cuda_oom(error: BaseException) -> bool:
    """Return whether an exception represents a CUDA out-of-memory condition."""

    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _cleanup_cuda(torch_module: Any) -> None:
    """Release references and allocator blocks after a rejected candidate."""

    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    ipc_collect = getattr(cuda, "ipc_collect", None)
    if callable(ipc_collect):
        ipc_collect()


def _bind_collate(collate_fn: Callable[..., Any], max_speakers_per_chunk: int) -> Callable[[Any], Any]:
    """Bind the speaker limit without creating an unpicklable local function."""

    return partial(collate_fn, max_speakers_per_chunk=max_speakers_per_chunk)


def _device_facts(torch_module: Any) -> dict[str, object]:
    """Read CUDA facts without making any provider or network call."""

    cuda = torch_module.cuda
    available = bool(cuda.is_available())
    if not available:
        raise QualificationError("CUDA is required for GPU qualification")
    index = int(cuda.current_device()) if hasattr(cuda, "current_device") else 0
    name = str(cuda.get_device_name(index))
    properties = cuda.get_device_properties(index)
    total_memory = int(getattr(properties, "total_memory", 0))
    facts: dict[str, object] = {
        "device_index": index,
        "device_name": name,
        "cuda_available": available,
        "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
        "torch_version": getattr(torch_module, "__version__", None),
        "total_memory_bytes": total_memory,
        "compute_capability": [int(properties.major), int(properties.minor)]
        if hasattr(properties, "major") and hasattr(properties, "minor")
        else None,
    }
    driver_version = getattr(cuda, "driver_version", None)
    if driver_version is not None:
        facts["driver_version"] = driver_version
    return facts


def _code_hash(config: Mapping[str, Any] | None = None, config_path: Path | None = None) -> tuple[str, dict[str, str]]:
    """Hash every local implementation that constructs or trains the real batch."""

    repository_root = Path(__file__).resolve().parents[3]
    paths = {
        Path(__file__).resolve(),
        Path(__file__).with_name("cli.py").resolve(),
        (repository_root / "diarizen" / "trainer_dual_opt.py").resolve(),
        (repository_root / "diarizen" / "trainer_utils.py").resolve(),
        (repository_root / "diarizen" / "models" / "eend" / "model_wavlm_conformer.py").resolve(),
    }
    real_components = {"model", "trainer", "train_dataset"}
    if config is not None and config_path is not None and real_components.issubset(config):
        training_root = _training_root(config_path)
        paths.update({(training_root / "trainer_dual_opt.py").resolve(), (training_root / "dataset.py").resolve()})
    missing = sorted(path.as_posix() for path in paths if not path.is_file())
    if missing:
        raise QualificationError("qualification implementation source is missing", {"missing": missing})
    records = {path.as_posix(): sha256_file(path) for path in sorted(paths)}
    return sha256_json(records), records


def _hash_file_or_text(path: Path) -> str:
    """Hash the trainer config or its unresolved path for failure reports."""

    return sha256_file(path) if path.is_file() else sha256_text(path.as_posix())


def _load_config(path: Path) -> dict[str, Any]:
    """Load one trainer TOML without filling in synthetic data defaults."""

    try:
        value = toml.load(path.as_posix())
    except (OSError, TypeError, toml.TomlDecodeError) as error:
        raise QualificationError("cannot load trainer TOML", {"path": path.as_posix(), "error": str(error)}) from error
    if not isinstance(value, dict):
        raise QualificationError("trainer TOML must contain an object")
    return value


def _training_root(config_path: Path) -> Path:
    """Find the recipe working directory used by the existing trainer command."""

    for parent in (config_path.parent, *config_path.parents):
        candidate = parent / "diar_ssl"
        if candidate.is_dir():
            return candidate
    return config_path.parent


def _configured_path(raw: Any, root: Path, label: str) -> Path:
    """Resolve a required path from the trainer TOML."""

    if not isinstance(raw, str) or not raw:
        raise QualificationError(f"{label} must be a non-empty path in trainer TOML")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _validate_qualification_bundle(
    config: Mapping[str, Any],
    root: Path,
    binding: QualificationBinding,
    input_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Verify the staged four-source bundle used by the real trainer."""

    qualification = config.get("qualification")
    if not isinstance(qualification, Mapping):
        raise QualificationError("trainer TOML must contain qualification binding metadata")
    if qualification.get("binding_sha256") != binding.binding_sha256:
        raise QualificationError("trainer TOML qualification binding digest differs from the supplied binding")
    manifest_path = _configured_path(qualification.get("bundle_manifest"), root, "qualification.bundle_manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError(
            "cannot load qualification bundle manifest",
            {"path": manifest_path.as_posix(), "error": str(error)},
        ) from error
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "speakrs-qualification-bundle-v1":
        raise QualificationError("qualification bundle manifest has an unsupported schema")
    if sha256_file(manifest_path) != binding.bundle_manifest_sha256:
        raise QualificationError("qualification bundle manifest differs from the supplied binding")
    if set(manifest) != {
        "schema",
        "release_sha256",
        "restore_receipt_sha256",
        "required_sources",
        "wav_prefix",
        "selection_plan_sha256",
        "selected_batches",
        "manifests",
        "recordings",
    }:
        raise QualificationError("qualification bundle manifest fields are invalid")
    expected_identity = {
        "release_sha256": binding.release_sha256,
        "restore_receipt_sha256": binding.restore_receipt_sha256,
        "required_sources": list(binding.required_sources),
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected_identity.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise QualificationError("qualification bundle identity differs from the supplied binding", mismatches)
    selected_batches = manifest.get("selected_batches")
    if not isinstance(selected_batches, Mapping) or set(selected_batches) != set(binding.required_sources):
        raise QualificationError("qualification bundle selected batches are incomplete")
    for source in binding.required_sources:
        selected_batch = selected_batches[source]
        if not isinstance(selected_batch, Mapping) or set(selected_batch) != {
            "batch_sha256",
            "acceptance_sha256",
            "restore_receipt",
            "restore_receipt_sha256",
        }:
            raise QualificationError("qualification bundle selected batch is malformed", {"source": source})
        receipt_relative = selected_batch.get("restore_receipt")
        receipt_sha256 = selected_batch.get("restore_receipt_sha256")
        if not isinstance(receipt_relative, str) or not receipt_relative or not isinstance(receipt_sha256, str):
            raise QualificationError("qualification bundle selected batch receipt is invalid", {"source": source})
        receipt_path = (manifest_path.parent / receipt_relative).resolve()
        if (
            Path(receipt_relative).is_absolute()
            or not receipt_path.is_relative_to(manifest_path.parent.resolve())
            or not receipt_path.is_file()
            or sha256_file(receipt_path) != receipt_sha256
        ):
            raise QualificationError("qualification bundle selected batch receipt changed", {"source": source})

    declared_manifests = manifest.get("manifests")
    if not isinstance(declared_manifests, Mapping) or set(declared_manifests) != set(input_paths):
        raise QualificationError("qualification bundle manifest hashes are incomplete")
    for name, path in input_paths.items():
        if declared_manifests.get(name) != sha256_file(path):
            raise QualificationError(
                "qualification training manifest changed after bundle creation",
                {"manifest": name, "path": path.as_posix()},
            )

    wav_rows: dict[str, Path] = {}
    for line_number, line in enumerate(input_paths["wav_scp"].read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in wav_rows:
            raise QualificationError("qualification wav.scp is malformed", {"line": line_number})
        audio = Path(fields[1]).expanduser()
        wav_rows[fields[0]] = audio if audio.is_absolute() else (root / audio).resolve()

    recordings = manifest.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        raise QualificationError("qualification bundle has no recordings")
    seen: set[str] = set()
    seen_sources: set[str] = set()
    audio_identity: list[dict[str, str]] = []
    bundle_root = manifest_path.parent
    for index, item in enumerate(recordings):
        if not isinstance(item, Mapping) or set(item) != {
            "recording_id",
            "source",
            "batch_sha256",
            "audio_path",
            "audio_sha256",
            "audio_size",
            "rttm_sha256",
            "uem_sha256",
        }:
            raise QualificationError("qualification bundle recording is malformed", {"index": index})
        recording_id = item.get("recording_id")
        source = item.get("source")
        batch_sha256 = item.get("batch_sha256")
        relative_path = item.get("audio_path")
        digest = item.get("audio_sha256")
        if not all(
            isinstance(value, str) and value
            for value in (
                recording_id,
                source,
                batch_sha256,
                relative_path,
                digest,
                item.get("rttm_sha256"),
                item.get("uem_sha256"),
            )
        ):
            raise QualificationError(
                "qualification bundle recording fields must be non-empty strings", {"index": index}
            )
        assert isinstance(recording_id, str)
        assert isinstance(source, str)
        assert isinstance(batch_sha256, str)
        assert isinstance(relative_path, str)
        assert isinstance(digest, str)
        audio_size = item.get("audio_size")
        selected_batch = selected_batches.get(source)
        if (
            isinstance(audio_size, bool)
            or not isinstance(audio_size, int)
            or audio_size <= 0
            or recording_id in seen
            or source not in binding.required_sources
            or not isinstance(selected_batch, Mapping)
            or selected_batch.get("batch_sha256") != batch_sha256
        ):
            raise QualificationError("qualification bundle recording membership is invalid", {"index": index})
        audio_relative = Path(relative_path)
        expected_audio = (bundle_root / audio_relative).resolve()
        if audio_relative.is_absolute() or not expected_audio.is_relative_to(bundle_root.resolve()):
            raise QualificationError("qualification bundle audio path escapes the bundle", {"index": index})
        if wav_rows.get(recording_id) != expected_audio or not expected_audio.is_file():
            raise QualificationError(
                "qualification bundle audio path is missing or differs from wav.scp", {"index": index}
            )
        if expected_audio.stat().st_size != audio_size or sha256_file(expected_audio) != digest:
            raise QualificationError("qualification bundle audio digest differs from its file", {"index": index})
        seen.add(recording_id)
        seen_sources.add(source)
        audio_identity.append({"recording_id": recording_id, "source": source, "sha256": digest})
    if seen != set(wav_rows) or seen_sources != set(binding.required_sources):
        raise QualificationError("qualification bundle does not cover the exact staged recording inventory")
    return {
        "bundle_manifest": manifest_path.as_posix(),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "audio_identity_sha256": sha256_json(audio_identity),
        "recordings": len(audio_identity),
        "sources": list(binding.required_sources),
    }


def _validate_real_config(
    config: Mapping[str, Any],
    root: Path,
    qualification_binding: QualificationBinding | None = None,
) -> dict[str, object]:
    """Validate the selected real model and input files before constructing a trainer."""

    model = config.get("model")
    model_args = model.get("args") if isinstance(model, Mapping) else None
    if not isinstance(model_args, Mapping):
        raise QualificationError("trainer TOML must contain model.args")
    required_model_values = {
        "chunk_size": 8,
        "max_speakers_per_chunk": 4,
        "max_speakers_per_frame": 2,
        "strict_wavlm_load": True,
        "wavlm_layer_num": 25,
        "wavlm_feat_dim": 1024,
    }
    mismatches = {
        key: {"expected": expected, "actual": model_args.get(key)}
        for key, expected in required_model_values.items()
        if model_args.get(key) != expected
    }
    if mismatches:
        raise QualificationError("trainer TOML does not match the selected WavLM Large profile", mismatches)
    model_path = config.get("model", {}).get("path") if isinstance(config.get("model"), Mapping) else None
    if model_path != EXPECTED_MODEL_CLASS:
        raise QualificationError(
            "trainer TOML must select the exact DiariZen WavLM model",
            {"expected": EXPECTED_MODEL_CLASS, "actual": model_path},
        )

    trainer = config.get("trainer")
    trainer_path = trainer.get("path") if isinstance(trainer, Mapping) else None
    if trainer_path != EXPECTED_TRAINER_CLASS:
        raise QualificationError(
            "trainer TOML must select the exact dual-optimizer trainer",
            {"expected": EXPECTED_TRAINER_CLASS, "actual": trainer_path},
        )
    for key in ("optimizer_small", "optimizer_big"):
        optimizer = config.get(key)
        if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("path"), str):
            raise QualificationError(f"trainer TOML must contain {key}")

    train_dataset = config.get("train_dataset")
    dataset_args = train_dataset.get("args") if isinstance(train_dataset, Mapping) else None
    if not isinstance(dataset_args, Mapping):
        raise QualificationError("trainer TOML must contain train_dataset.args")
    dataset_path = train_dataset.get("path") if isinstance(train_dataset, Mapping) else None
    if dataset_path != EXPECTED_DATASET_CLASS:
        raise QualificationError(
            "trainer TOML must select the exact DiarizationDataset reader",
            {"expected": EXPECTED_DATASET_CLASS, "actual": dataset_path},
        )
    dataset_mismatches = {
        key: {"expected": expected, "actual": dataset_args.get(key)}
        for key, expected in {"chunk_size": 8, "sample_rate": 16_000}.items()
        if dataset_args.get(key) != expected
    }
    if dataset_mismatches:
        raise QualificationError(
            "trainer TOML does not use the selected eight-second input profile", dataset_mismatches
        )
    input_paths = {
        "wav_scp": _configured_path(dataset_args.get("scp_file"), root, "train_dataset.args.scp_file"),
        "rttm": _configured_path(dataset_args.get("rttm_file"), root, "train_dataset.args.rttm_file"),
        "uem": _configured_path(dataset_args.get("uem_file"), root, "train_dataset.args.uem_file"),
    }
    missing = {name: path.as_posix() for name, path in input_paths.items() if not path.is_file()}
    if missing:
        raise QualificationError("trainer TOML input file is missing", {"missing": missing})
    required_dataset_values = {"chunk_size": 8, "chunk_shift": 6, "sample_rate": 16_000}
    dataset_mismatches = {
        key: {"expected": expected, "actual": dataset_args.get(key)}
        for key, expected in required_dataset_values.items()
        if dataset_args.get(key) != expected
    }
    if dataset_mismatches:
        raise QualificationError(
            "trainer TOML training data settings differ from the selected profile", dataset_mismatches
        )
    wavlm_src = _configured_path(model_args.get("wavlm_src"), root, "model.args.wavlm_src")
    if not wavlm_src.is_file():
        raise QualificationError("WavLM initializer is missing", {"path": wavlm_src.as_posix()})
    wavlm_initializer_sha256 = sha256_file(wavlm_src)
    if (
        qualification_binding is not None
        and wavlm_initializer_sha256 != qualification_binding.wavlm_initializer_sha256
    ):
        raise QualificationError(
            "WavLM initializer differs from the qualification binding",
            {
                "expected": qualification_binding.wavlm_initializer_sha256,
                "actual": wavlm_initializer_sha256,
            },
        )
    facts = {
        "input_paths": {name: path.as_posix() for name, path in input_paths.items()},
        "input_hashes": {name: sha256_file(path) for name, path in input_paths.items()},
        "wavlm_initializer": wavlm_src.as_posix(),
        "wavlm_initializer_sha256": wavlm_initializer_sha256,
    }
    if qualification_binding is not None:
        facts["qualification_bundle"] = _validate_qualification_bundle(
            config,
            root,
            qualification_binding,
            input_paths,
        )
    return facts


def _normalise_attempt(
    value: AttemptResult | Mapping[str, Any], physical_batch: int, accumulation: int
) -> AttemptResult:
    """Normalize injected runner results for deterministic CPU tests."""

    if isinstance(value, AttemptResult):
        return value
    if not isinstance(value, Mapping):
        raise QualificationError("attempt runner must return AttemptResult or an object")
    timings = tuple(float(item) for item in value.get("update_seconds", ()))
    total = int(value.get("total_memory_bytes", 0))
    reserved = int(value.get("peak_reserved_bytes", 0))
    return AttemptResult(
        physical_batch=int(value.get("physical_batch", physical_batch)),
        accumulation=int(value.get("accumulation", accumulation)),
        ok=bool(value.get("ok", False)),
        total_memory_bytes=total,
        peak_allocated_bytes=int(value.get("peak_allocated_bytes", 0)),
        peak_reserved_bytes=reserved,
        update_seconds=timings,
        measured_update_count=int(value.get("measured_update_count", len(timings))),
        examples_processed=int(value.get("examples_processed", 0)),
        checkpoint_write_reload=value.get("checkpoint_write_reload"),
        error_code=value.get("error_code"),
        error_message=value.get("error_message"),
        headroom_fraction_override=(
            None if value.get("headroom_fraction") is None else float(value["headroom_fraction"])
        ),
    )


def _invoke_attempt_runner(
    runner: Callable[..., AttemptResult | Mapping[str, Any]],
    *,
    physical_batch: int,
    phase: str,
    spec: QualificationSpec,
    config: Mapping[str, Any],
) -> AttemptResult:
    """Call one runner with the complete qualification attempt contract."""

    accumulation = spec.accumulation_for(physical_batch)
    updates = spec.measured_optimizer_updates
    try:
        result = runner(
            physical_batch=physical_batch,
            accumulation=accumulation,
            phase=phase,
            warmup_updates=spec.warmup_optimizer_updates,
            measured_updates=updates,
            config=config,
        )
    except BaseException as error:  # noqa: BLE001 - classify only CUDA OOM here
        if _is_cuda_oom(error):
            return AttemptResult(
                physical_batch=physical_batch,
                accumulation=accumulation,
                ok=False,
                error_code="cuda_oom",
                error_message=str(error),
            )
        raise
    return _normalise_attempt(result, physical_batch, accumulation)


def _coerce_qualification_binding(
    value: QualificationBinding | Mapping[str, Any] | None,
) -> QualificationBinding | None:
    """Parse one optional qualification binding before a worker run."""

    if value is None:
        return None
    if isinstance(value, QualificationBinding):
        return value
    return parse_qualification_binding(value)


def _attach_qualification_control(
    report: dict[str, object],
    qualification_control: Mapping[str, object] | None,
    binding: QualificationBinding | None,
) -> None:
    """Bind a worker report to the trusted qualification lease check."""

    if qualification_control is None:
        return
    if binding is None:
        raise QualificationError("qualification lease requires a qualification binding")
    if qualification_control.get("qualification_binding_sha256") != binding.binding_sha256:
        raise QualificationError("qualification control result has a different binding digest")
    lease_id = qualification_control.get("lease_id")
    spend_ceiling = qualification_control.get("spend_ceiling_usd")
    hourly_rate = qualification_control.get("hourly_rate_usd")
    hard_deadline = qualification_control.get("hard_deadline")
    deadline_epoch_seconds = qualification_control.get("deadline_epoch_seconds")
    max_runtime_seconds = qualification_control.get("max_runtime_seconds")
    if not isinstance(lease_id, str) or not lease_id:
        raise QualificationError("qualification control result has no lease identity")
    if isinstance(spend_ceiling, bool) or not isinstance(spend_ceiling, (int, float)):
        raise QualificationError("qualification control result has no spend ceiling")
    numeric_fields = (hourly_rate, deadline_epoch_seconds, max_runtime_seconds)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric_fields):
        raise QualificationError("qualification control result has invalid runtime bounds")
    if not isinstance(hard_deadline, str) or not hard_deadline or float(max_runtime_seconds) <= 0:
        raise QualificationError("qualification control result has invalid runtime bounds")
    report["qualification_lease"] = {
        "lease_id": lease_id,
        "spend_ceiling_usd": float(spend_ceiling),
        "hourly_rate_usd": float(hourly_rate),
        "hard_deadline": hard_deadline,
        "deadline_epoch_seconds": float(deadline_epoch_seconds),
        "max_runtime_seconds": float(max_runtime_seconds),
        "qualification_binding_sha256": binding.binding_sha256,
    }


def _qualification_timeout_seconds(
    qualification_control: Mapping[str, object] | None,
    binding: QualificationBinding | None,
) -> float | None:
    """Return the paid-run limit after validating the trusted control result."""

    if qualification_control is None:
        return None
    probe: dict[str, object] = {}
    _attach_qualification_control(probe, qualification_control, binding)
    lease = probe["qualification_lease"]
    assert isinstance(lease, Mapping)
    return float(lease["max_runtime_seconds"])


def _build_report(
    *,
    spec: QualificationSpec,
    config_path: Path,
    config: Mapping[str, Any] | None,
    gpu_profile: str,
    device: Mapping[str, object],
    probes: Sequence[AttemptResult],
    selected: AttemptResult | None,
    measured: AttemptResult | None,
    config_facts: Mapping[str, object] | None,
    qualification_binding: QualificationBinding | None = None,
    failure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the stable report payload shared by success and failure paths."""

    spec_hash = sha256_json(spec.identity())
    config_hash = _hash_file_or_text(config_path)
    code_hash, code_files = _code_hash(config, config_path)
    report: dict[str, object] = {
        "schema": QUALIFICATION_SCHEMA,
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "gpu_profile": gpu_profile,
        "qualification_spec": spec.identity(),
        "warmup_optimizer_updates": spec.warmup_optimizer_updates,
        "measured_optimizer_updates_requested": spec.measured_optimizer_updates,
        "trainer_config": config_path.resolve().as_posix(),
        "device": dict(device),
        "config_facts": dict(config_facts or {}),
        "content_hashes": {
            "spec_sha256": spec_hash,
            "config_sha256": config_hash,
            "code_sha256": code_hash,
            "code_files": code_files,
        },
        "spec_sha256": spec_hash,
        "config_sha256": config_hash,
        "code_sha256": code_hash,
        "physical_batch_candidates": [probe.as_json() for probe in probes],
        "unmeasured_overhead": list(UNMEASURED_OVERHEAD),
        "precision": {
            "requested": "bf16",
            "path": "accelerate",
        },
        "training_ready": False,
        "qualification_only": qualification_binding is not None,
    }
    if qualification_binding is not None:
        report["qualification_binding_sha256"] = qualification_binding.binding_sha256
        report["capacity_policy"] = qualification_binding.capacity_policy
        report["capacity_admitted"] = qualification_binding.capacity_admitted
        report["qualification_ready"] = qualification_binding.qualification_ready
    if selected is not None:
        report["physical_batch"] = selected.physical_batch
        report["accumulation"] = selected.accumulation
        report["memory"] = {
            "total_bytes": selected.total_memory_bytes,
            "peak_allocated_bytes": selected.peak_allocated_bytes,
            "peak_reserved_bytes": selected.peak_reserved_bytes,
            "headroom_fraction": selected.headroom_fraction,
            "minimum_headroom_fraction": spec.minimum_memory_headroom,
        }
        report["peak_allocated_memory_bytes"] = selected.peak_allocated_bytes
        report["peak_reserved_memory_bytes"] = selected.peak_reserved_bytes
        report["memory_headroom_fraction"] = selected.headroom_fraction
    if measured is not None:
        timings = tuple(measured.update_seconds)
        if timings:
            p50 = percentile(timings, 50)
            p95 = percentile(timings, 95)
            p50_estimates = estimate_training_hours(p50, spec.planned_updates_per_cycle, spec.planned_max_cycles)
            p95_estimates = estimate_training_hours(p95, spec.planned_updates_per_cycle, spec.planned_max_cycles)
            examples = measured.examples_processed or len(timings) * spec.effective_batch
            report.update(
                {
                    "p50_optimizer_update_seconds": p50,
                    "p95_optimizer_update_seconds": p95,
                    "examples_per_second": examples / sum(timings),
                    "measured_update_count": measured.measured_update_count or len(timings),
                    "estimated_training_hours_per_cycle": p50_estimates["hours_per_cycle"],
                    "estimated_training_hours_for_max_cycles": p50_estimates["hours_for_max_cycles"],
                    "estimated_training_hours_per_cycle_p95": p95_estimates["hours_per_cycle"],
                    "estimated_training_hours_for_max_cycles_p95": p95_estimates["hours_for_max_cycles"],
                    "training_only_estimate": {
                        "basis": "measured optimizer-update time",
                        "planned_updates_per_cycle": spec.planned_updates_per_cycle,
                        "planned_max_cycles": spec.planned_max_cycles,
                        "p50": p50_estimates,
                        "p95": p95_estimates,
                    },
                    "checkpoint_write_reload": dict(measured.checkpoint_write_reload or {"feasible": False}),
                }
            )
    if failure:
        report["error"] = dict(failure)
    report["gpu_qualification_status"] = "qualified" if failure is None else "failed"
    report["ok"] = failure is None
    return report


def run_qualification(
    trainer_config_path: Path,
    gpu_profile: str,
    spec: QualificationSpec,
    *,
    attempt_runner: Callable[..., AttemptResult | Mapping[str, Any]] | None = None,
    device_facts: Mapping[str, object] | None = None,
    qualification_binding: QualificationBinding | Mapping[str, Any] | None = None,
    qualification_control: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run bounded worker qualification and return a report.

    When ``attempt_runner`` is omitted, the real DiariZen model, dataset reader,
    dual optimizers, and Accelerate BF16 path are constructed. Tests can inject
    an attempt runner and device facts without requiring CUDA.
    """

    spec.validate()
    binding = _coerce_qualification_binding(qualification_binding)
    timeout_seconds = _qualification_timeout_seconds(qualification_control, binding)
    trainer_config_path = Path(trainer_config_path).expanduser().resolve()
    if gpu_profile != "4090":
        raise QualificationError("unsupported gpu_profile", {"gpu_profile": gpu_profile, "supported": ["4090"]})
    config = _load_config(trainer_config_path)

    if device_facts is None:
        try:
            import torch
        except ImportError as error:
            raise QualificationError("PyTorch is required for GPU qualification") from error
        device = _device_facts(torch)
    else:
        device = dict(device_facts)
        if not bool(device.get("cuda_available")):
            raise QualificationError("CUDA is required for GPU qualification")
    device_name = str(device.get("device_name", ""))
    if "4090" not in device_name.lower():
        raise QualificationError(
            "gpu_profile=4090 requires an actual device name containing 4090",
            {"device_name": device_name, "gpu_profile": gpu_profile},
        )
    if int(device.get("total_memory_bytes", 0)) <= 0:
        raise QualificationError("CUDA device did not report total memory")

    config_facts: dict[str, object] = {}
    if attempt_runner is None:
        if binding is None:
            raise QualificationError("real GPU qualification requires a content-bound qualification input")
        if timeout_seconds is None:
            raise QualificationError("real GPU qualification requires a paid-run deadline")
        root = _training_root(trainer_config_path)
        config_facts = _validate_real_config(config, root, binding)
        deadline = time.monotonic() + timeout_seconds

        def real_runner(
            *,
            physical_batch: int,
            accumulation: int,
            phase: str,
            warmup_updates: int,
            measured_updates: int,
            config: Mapping[str, Any],
        ) -> AttemptResult:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return AttemptResult(
                    physical_batch=physical_batch,
                    accumulation=accumulation,
                    ok=False,
                    error_code="qualification_deadline_exceeded",
                    error_message="qualification runtime limit expired before this candidate",
                )
            return _run_isolated_real_attempt(
                config_path=trainer_config_path,
                config=config,
                training_root=root,
                spec=spec,
                device=device,
                work_root=work_root,
                timeout_seconds=remaining,
                physical_batch=physical_batch,
                accumulation=accumulation,
                phase=phase,
                warmup_updates=warmup_updates,
                measured_updates=measured_updates,
            )

        attempt_runner = real_runner

    probes: list[AttemptResult] = []
    work_root: Path | None = None
    temporary_root: tempfile.TemporaryDirectory[str] | None = None
    if attempt_runner is not None and getattr(attempt_runner, "__name__", "") == "real_runner":
        temporary_root = tempfile.TemporaryDirectory(
            prefix="speakrs-gpu-qualification-", dir=str(trainer_config_path.parent)
        )
        work_root = Path(temporary_root.name)
    try:
        for physical_batch in spec.physical_batch_candidates:
            probe = _invoke_attempt_runner(
                attempt_runner,
                physical_batch=physical_batch,
                phase="probe",
                spec=spec,
                config=config,
            )
            if probe.total_memory_bytes <= 0 and probe.ok:
                probe = AttemptResult(
                    **{
                        **asdict(probe),
                        "total_memory_bytes": int(device["total_memory_bytes"]),
                    }
                )
            probes.append(probe)
            if probe.error_code == "qualification_deadline_exceeded":
                return _build_report(
                    spec=spec,
                    config_path=trainer_config_path,
                    config=config,
                    gpu_profile=gpu_profile,
                    device=device,
                    probes=probes,
                    selected=None,
                    measured=None,
                    config_facts=config_facts,
                    qualification_binding=binding,
                    failure={
                        "code": probe.error_code,
                        "message": probe.error_message or "qualification deadline exceeded",
                    },
                )
        try:
            selected = select_largest_batch(probes, spec.minimum_memory_headroom)
        except QualificationError as error:
            return _build_report(
                spec=spec,
                config_path=trainer_config_path,
                config=config,
                gpu_profile=gpu_profile,
                device=device,
                probes=probes,
                selected=None,
                measured=None,
                config_facts=config_facts,
                qualification_binding=binding,
                failure={"code": error.code, "message": error.message, "details": error.details},
            )
        # each candidate already ran the complete warmup and measured window in
        # its isolated process; only the selected candidate's timings are
        # promoted into the final training estimate
        measured = selected
        if not measured.ok:
            return _build_report(
                spec=spec,
                config_path=trainer_config_path,
                config=config,
                gpu_profile=gpu_profile,
                device=device,
                probes=probes,
                selected=selected,
                measured=measured,
                config_facts=config_facts,
                qualification_binding=binding,
                failure={
                    "code": measured.error_code or "measurement_failed",
                    "message": measured.error_message or "selected batch measurement failed",
                },
            )
        if len(measured.update_seconds) < spec.measured_optimizer_updates:
            failure = {
                "code": "insufficient_measurements",
                "message": "measured optimizer update count is below the requested count",
                "details": {
                    "requested": spec.measured_optimizer_updates,
                    "observed": len(measured.update_seconds),
                },
            }
            return _build_report(
                spec=spec,
                config_path=trainer_config_path,
                config=config,
                gpu_profile=gpu_profile,
                device=device,
                probes=probes,
                selected=selected,
                measured=measured,
                config_facts=config_facts,
                qualification_binding=binding,
                failure=failure,
            )
        checkpoint = measured.checkpoint_write_reload
        if not checkpoint or not (
            bool(checkpoint.get("feasible")) and bool(checkpoint.get("write")) and bool(checkpoint.get("reload"))
        ):
            return _build_report(
                spec=spec,
                config_path=trainer_config_path,
                config=config,
                gpu_profile=gpu_profile,
                device=device,
                probes=probes,
                selected=selected,
                measured=measured,
                config_facts=config_facts,
                qualification_binding=binding,
                failure={
                    "code": "checkpoint_roundtrip_failed",
                    "message": "checkpoint write/reload did not pass",
                    "details": dict(checkpoint or {}),
                },
            )
        return _build_report(
            spec=spec,
            config_path=trainer_config_path,
            config=config,
            gpu_profile=gpu_profile,
            device=device,
            probes=probes,
            selected=selected,
            measured=measured,
            config_facts=config_facts,
            qualification_binding=binding,
        )
    finally:
        if temporary_root is not None:
            temporary_root.cleanup()


def _real_attempt_process(result_queue: Any, payload: Mapping[str, Any]) -> None:
    """Run one real candidate in a clean child process."""

    try:
        result_queue.put({"ok": True, "result": _run_real_attempt(**dict(payload))})
    except BaseException as error:  # noqa: BLE001 - parent turns this into a report
        result_queue.put(
            {
                "ok": False,
                "error": {
                    "code": "worker_process_failed",
                    "message": str(error),
                    "type": type(error).__name__,
                },
            }
        )


def _run_isolated_real_attempt(*, timeout_seconds: float | None = None, **kwargs: Any) -> AttemptResult:
    """Run one candidate in a fresh interpreter to release CUDA state."""

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_real_attempt_process, args=(result_queue, kwargs))
    process.start()
    process.join(timeout=timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=10.0)
        result_queue.close()
        return AttemptResult(
            physical_batch=int(kwargs["physical_batch"]),
            accumulation=int(kwargs["accumulation"]),
            ok=False,
            error_code="qualification_deadline_exceeded",
            error_message="qualification worker exceeded the paid-run deadline",
        )
    try:
        envelope = result_queue.get(timeout=1.0)
    except (EOFError, OSError, queue.Empty):
        return AttemptResult(
            physical_batch=int(kwargs["physical_batch"]),
            accumulation=int(kwargs["accumulation"]),
            ok=False,
            error_code="worker_process_failed",
            error_message=f"qualification worker exited with status {process.exitcode}",
        )
    finally:
        result_queue.close()
    if envelope.get("ok"):
        return envelope["result"]
    error = envelope.get("error") or {}
    return AttemptResult(
        physical_batch=int(kwargs["physical_batch"]),
        accumulation=int(kwargs["accumulation"]),
        ok=False,
        error_code=str(error.get("code", "worker_process_failed")),
        error_message=str(error.get("message", "qualification worker failed")),
    )


def _real_attempt_config(
    config: Mapping[str, Any],
    spec: QualificationSpec,
    *,
    physical_batch: int,
    accumulation: int,
    phase: str,
    work_root: Path | None,
) -> dict[str, Any]:
    """Build one isolated trainer configuration for the full timing window."""

    runtime_config = copy.deepcopy(dict(config))
    trainer_args = runtime_config.setdefault("trainer", {}).setdefault("args", {})
    trainer_args["gradient_accumulation_steps"] = accumulation
    trainer_args["max_steps"] = spec.warmup_optimizer_updates + spec.measured_optimizer_updates
    trainer_args["max_epochs"] = 1
    trainer_args["validation_before_training"] = False
    trainer_args["validation_interval"] = 2**31 - 1
    trainer_args["save_ckpt_interval"] = 1
    train_loader_config = runtime_config["train_dataset"]["dataloader"]
    train_loader_config["batch_size"] = physical_batch
    train_loader_config["drop_last"] = True
    if work_root is not None:
        runtime_config.setdefault("meta", {})["save_dir"] = str(work_root)
        runtime_config["meta"]["exp_id"] = f"{phase}-batch-{physical_batch}"
    runtime_config.setdefault("meta", {}).setdefault("exp_id", f"qualification-{phase}-{physical_batch}")
    return runtime_config


def _run_real_attempt(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    training_root: Path,
    spec: QualificationSpec,
    device: Mapping[str, object],
    work_root: Path | None,
    physical_batch: int,
    phase: str,
    accumulation: int,
    warmup_updates: int,
    measured_updates: int,
    **_kwargs: Any,
) -> AttemptResult:
    """Construct and run one real DiariZen attempt through the existing trainer."""

    import torch
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import GradientAccumulationPlugin, set_seed
    from torch.utils.data import DataLoader

    from diarizen.utils import instantiate

    runtime_config = _real_attempt_config(
        config,
        spec,
        physical_batch=physical_batch,
        accumulation=accumulation,
        phase=phase,
        work_root=work_root,
    )
    train_loader_config = runtime_config["train_dataset"]["dataloader"]

    current_directory = Path.cwd()
    path_inserted = False
    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
        path_inserted = True
    try:
        os.chdir(training_root)
        from dataset import _collate_fn

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accumulation_plugin = GradientAccumulationPlugin(
            num_steps=accumulation,
            sync_with_dataloader=False,
        )
        accelerator = Accelerator(
            gradient_accumulation_plugin=accumulation_plugin,
            mixed_precision="bf16",
            kwargs_handlers=[ddp_kwargs],
        )
        if getattr(accelerator, "mixed_precision", None) != "bf16":
            raise QualificationError("Accelerate did not enable BF16 mixed precision")
        set_seed(int(runtime_config.get("meta", {}).get("seed", 3407)), device_specific=True)

        model = instantiate(runtime_config["model"]["path"], args=runtime_config["model"]["args"])
        powerset = getattr(model, "powerset", None)
        classes = getattr(powerset, "num_powerset_classes", None)
        if classes != 11:
            raise QualificationError("real model is not configured for 11 powerset classes", {"actual": classes})
        model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info
        optimizer_small = instantiate(
            runtime_config["optimizer_small"]["path"],
            args={"params": model.wavlm_model.parameters()} | runtime_config["optimizer_small"]["args"],
        )
        optimizer_big = instantiate(
            runtime_config["optimizer_big"]["path"],
            args={"params": model.non_wavlm_parameters()} | runtime_config["optimizer_big"]["args"],
        )
        model, optimizer_small, optimizer_big = accelerator.prepare(model, optimizer_small, optimizer_big)

        train_args = runtime_config["train_dataset"]["args"]
        train_args["model_num_frames"] = model_num_frames
        train_args["model_rf_duration"] = model_rf_duration
        train_args["model_rf_step"] = model_rf_step
        collate = _bind_collate(
            _collate_fn,
            runtime_config["model"]["args"]["max_speakers_per_chunk"],
        )
        dataset = instantiate(runtime_config["train_dataset"]["path"], args=train_args)
        dataloader = DataLoader(dataset=dataset, collate_fn=collate, shuffle=True, **train_loader_config)
        dataloader = accelerator.prepare(dataloader)

        trainer_class = instantiate(runtime_config["trainer"]["path"], initialize=False)
        trainer = trainer_class(
            accelerator=accelerator,
            config=runtime_config,
            resume=False,
            model=model,
            optimizer_small=optimizer_small,
            optimizer_big=optimizer_big,
        )
        hooks = QualificationHooks(trainer, warmup_updates=warmup_updates, measured_updates=measured_updates)
        trainer.run_hooks = hooks
        original_training_step = trainer.training_step
        cuda = torch.cuda
        index = int(device.get("device_index", 0))

        @wraps(original_training_step)
        def timed_training_step(batch: Any, batch_idx: int) -> Any:
            if hooks._microbatch_started is None:
                cuda.synchronize(device=index)
            hooks.start_microbatch()
            try:
                autocast = getattr(accelerator, "autocast", None)
                context = autocast() if callable(autocast) else contextlib.nullcontext()
                with context:
                    return original_training_step(batch, batch_idx)
            finally:
                synchronized = bool(getattr(accelerator, "sync_gradients", True))
                if synchronized:
                    cuda.synchronize(device=index)
                hooks.finish_microbatch(synchronized)

        trainer.training_step = timed_training_step
        cuda.reset_peak_memory_stats(index)
        trainer.train(dataloader, None)
        update_count = len(hooks.update_seconds)
        checkpoint_result: dict[str, object] = {"feasible": False}
        if hasattr(trainer, "_find_latest_ckpt_path") and hasattr(trainer, "_load_checkpoint"):
            try:
                latest = trainer._find_latest_ckpt_path()
                trainer._load_checkpoint("latest")
                checkpoint_result = {
                    "feasible": True,
                    "write": True,
                    "reload": True,
                    "path": str(latest),
                }
            except (OSError, RuntimeError, ValueError) as error:
                checkpoint_result = {"feasible": True, "write": False, "reload": False, "error": str(error)}
        peak_allocated = int(cuda.max_memory_allocated(index))
        peak_reserved = int(cuda.max_memory_reserved(index))
        examples = update_count * spec.effective_batch
        return AttemptResult(
            physical_batch=physical_batch,
            accumulation=accumulation,
            ok=update_count >= measured_updates,
            total_memory_bytes=int(device["total_memory_bytes"]),
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            update_seconds=tuple(hooks.update_seconds),
            measured_update_count=update_count,
            examples_processed=examples,
            checkpoint_write_reload=checkpoint_result,
        )
    except BaseException as error:  # noqa: BLE001 - classify CUDA OOM and preserve real failures
        if _is_cuda_oom(error):
            _cleanup_cuda(torch)
            return AttemptResult(
                physical_batch=physical_batch,
                accumulation=accumulation,
                ok=False,
                total_memory_bytes=int(device["total_memory_bytes"]),
                error_code="cuda_oom",
                error_message=str(error),
            )
        raise
    finally:
        os.chdir(current_directory)
        if path_inserted:
            try:
                sys.path.remove(str(training_root))
            except ValueError:
                pass
        _cleanup_cuda(torch)


def execute_qualification(
    trainer_config_path: Path,
    gpu_profile: str,
    spec: QualificationSpec,
    output: Path,
    *,
    attempt_runner: Callable[..., AttemptResult | Mapping[str, Any]] | None = None,
    device_facts: Mapping[str, object] | None = None,
    qualification_binding: QualificationBinding | Mapping[str, Any] | None = None,
    qualification_control: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run qualification, atomically publish the report, and return it."""

    trainer_config_path = Path(trainer_config_path).expanduser()
    output = Path(output).expanduser()
    binding = _coerce_qualification_binding(qualification_binding)
    try:
        report = run_qualification(
            trainer_config_path,
            gpu_profile,
            spec,
            attempt_runner=attempt_runner,
            device_facts=device_facts,
            qualification_binding=binding,
            qualification_control=qualification_control,
        )
    except LargeError as error:
        config = None
        try:
            config = _load_config(trainer_config_path.resolve())
        except LargeError:
            pass
        report = _build_report(
            spec=spec,
            config_path=trainer_config_path.resolve(),
            config=config,
            gpu_profile=gpu_profile,
            device=dict(device_facts or {"cuda_available": False}),
            probes=(),
            selected=None,
            measured=None,
            config_facts={},
            qualification_binding=binding,
            failure={"code": error.code, "message": error.message, "details": error.details},
        )
    except BaseException as error:  # noqa: BLE001 - worker output must remain machine-readable
        report = _build_report(
            spec=spec,
            config_path=trainer_config_path.resolve(),
            config=None,
            gpu_profile=gpu_profile,
            device=dict(device_facts or {"cuda_available": False}),
            probes=(),
            selected=None,
            measured=None,
            config_facts={},
            qualification_binding=binding,
            failure={"code": "internal", "message": str(error), "type": type(error).__name__},
        )
    _attach_qualification_control(report, qualification_control, binding)
    write_json(output, report)
    return report


def write_failed_qualification(
    trainer_config_path: Path,
    gpu_profile: str,
    output: Path,
    error: LargeError,
    *,
    qualification_binding: QualificationBinding | Mapping[str, Any] | None = None,
    qualification_control: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically publish a failed report when parsing stops before execution."""

    trainer_config_path = Path(trainer_config_path).expanduser().resolve()
    binding = _coerce_qualification_binding(qualification_binding)
    config = None
    try:
        config = _load_config(trainer_config_path)
    except LargeError:
        pass
    report = _build_report(
        spec=QualificationSpec(),
        config_path=trainer_config_path,
        config=config,
        gpu_profile=gpu_profile,
        device={"cuda_available": False},
        probes=(),
        selected=None,
        measured=None,
        config_facts={},
        qualification_binding=binding,
        failure={"code": error.code, "message": error.message, "details": error.details},
    )
    _attach_qualification_control(report, qualification_control, binding)
    write_json(Path(output).expanduser(), report)
    return report


def load_qualification_spec(path: Path | None, overrides: Mapping[str, Any] | None = None) -> QualificationSpec:
    """Load an optional JSON qualification spec and apply explicit CLI values."""

    payload: dict[str, Any] = {}
    if path is not None:
        try:
            import json

            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise QualificationError(
                "cannot load qualification JSON spec", {"path": str(path), "error": str(error)}
            ) from error
        if not isinstance(raw, Mapping):
            raise QualificationError("qualification JSON spec must be an object")
        payload.update(raw)
    if overrides:
        payload.update({key: value for key, value in overrides.items() if value is not None})
    return QualificationSpec.from_mapping(payload)
