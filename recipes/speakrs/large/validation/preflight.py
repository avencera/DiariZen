"""Pure evaluator-host qualification and artifact-gated allocation preflight.

The module deliberately has no CUDA or model imports.  A worker can use it before
loading a framework or reserving GPU memory, and tests can provide probes without
touching the host's GPU, network, or cgroup files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Generic, Iterable, Mapping, Protocol, TypeVar

from .contracts import Sha256Digest


GIB = 1024**3
MIB = 1024**2
PROFILE_SCHEMA = "speakrs-evaluator-capability-profile-v1"
DEFAULT_GPU_NAME = "RTX 5060 Ti"
DEFAULT_MIN_GPU_MEMORY_BYTES = 16 * GIB
DEFAULT_MIN_CUDA_VERSION = "12.8"
DEFAULT_MIN_SYSTEM_RAM_BYTES = 32 * GIB
DEFAULT_MIN_CPU_CORES = 4

# leaves room for the image, the frozen 44-recording development
# cache, one approximately 1.2 GiB model, staging, and bounded evaluator logs
DEFAULT_MIN_FREE_DISK_BYTES = 32 * GIB

_CUDA_RE = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)(?:\.(?P<patch>[0-9]+))?$")
_MEMORY_RE = re.compile(r"^(?P<value>[0-9]+)(?:\s+MiB)?$")
_INTEGER_RE = re.compile(r"^[0-9]+$")
_CPUSET_PART_RE = re.compile(r"^(?P<first>[0-9]+)(?:-(?P<last>[0-9]+))?$")


class PreflightFailureReason(str, Enum):
    """Closed reasons for a rejected evaluator preflight."""

    NVIDIA_SMI_UNAVAILABLE = "nvidia_smi_unavailable"
    NVIDIA_SMI_MALFORMED = "nvidia_smi_malformed"
    CGROUP_MALFORMED = "cgroup_malformed"
    SYSTEM_MEMORY_UNAVAILABLE = "system_memory_unavailable"
    CPU_COUNT_UNAVAILABLE = "cpu_count_unavailable"
    DISK_SPACE_UNAVAILABLE = "disk_space_unavailable"
    GPU_NOT_FOUND = "gpu_not_found"
    GPU_NAME_MISMATCH = "gpu_name_mismatch"
    GPU_MEMORY_TOO_SMALL = "gpu_memory_too_small"
    CUDA_VERSION_TOO_OLD = "cuda_version_too_old"
    SYSTEM_RAM_TOO_SMALL = "system_ram_too_small"
    CPU_CORES_TOO_FEW = "cpu_cores_too_few"
    FREE_DISK_TOO_SMALL = "free_disk_too_small"
    INVALID_PROFILE = "invalid_profile"
    INVALID_ARTIFACT_SET = "invalid_artifact_set"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_LENGTH_MISMATCH = "artifact_length_mismatch"
    ARTIFACT_DIGEST_MISMATCH = "artifact_digest_mismatch"
    ARTIFACT_PATH_UNSAFE = "artifact_path_unsafe"
    ALLOCATION_FAILED = "allocation_failed"


class PreflightError(RuntimeError):
    """A preflight failure with a closed, non-secret reason."""

    def __init__(self, reason: PreflightFailureReason, detail: str | None = None) -> None:
        self.reason = reason
        self.detail = detail if detail is not None else reason.value
        super().__init__(self.detail)

    def to_dict(self) -> dict[str, str]:
        """Return a safe machine-readable failure without host paths or secrets."""

        return {"reason": self.reason.value, "detail": self.detail}


def _require_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _normalize_gpu_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("GPU name must be a non-empty string")
    if any(character in value for character in "\r\n\x00"):
        raise ValueError("GPU name contains a control character")
    normalized = " ".join(value.split())
    if normalized.startswith("NVIDIA GeForce "):
        normalized = normalized.removeprefix("NVIDIA GeForce ")
    return normalized


@dataclass(frozen=True, order=True)
class GpuName:
    """One normalized GPU product name."""

    value: str

    def __post_init__(self) -> None:
        normalized = _normalize_gpu_name(self.value)
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class CudaVersion:
    """One comparable CUDA runtime version."""

    major: int
    minor: int
    patch: int = 0

    def __post_init__(self) -> None:
        _require_integer(self.major, "CUDA major version")
        _require_integer(self.minor, "CUDA minor version")
        _require_integer(self.patch, "CUDA patch version")

    @classmethod
    def parse(cls, value: object) -> CudaVersion:
        """Parse a strict ``major.minor[.patch]`` version."""

        if not isinstance(value, str):
            raise ValueError("CUDA version must be a string")
        match = _CUDA_RE.fullmatch(value.strip())
        if match is None:
            raise ValueError("CUDA version must use major.minor or major.minor.patch")
        return cls(
            int(match.group("major")),
            int(match.group("minor")),
            int(match.group("patch") or 0),
        )

    def __str__(self) -> str:
        if self.patch:
            return f"{self.major}.{self.minor}.{self.patch}"
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class EvaluatorCapabilityProfile:
    """Typed host capabilities required by the standalone evaluator."""

    accepted_gpu_names: tuple[GpuName, ...] = (GpuName(DEFAULT_GPU_NAME),)
    minimum_gpu_memory_bytes: int = DEFAULT_MIN_GPU_MEMORY_BYTES
    minimum_cuda_version: CudaVersion = CudaVersion.parse(DEFAULT_MIN_CUDA_VERSION)
    minimum_system_ram_bytes: int = DEFAULT_MIN_SYSTEM_RAM_BYTES
    minimum_cpu_cores: int = DEFAULT_MIN_CPU_CORES
    minimum_free_disk_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES

    def __post_init__(self) -> None:
        names = tuple(name if isinstance(name, GpuName) else GpuName(name) for name in self.accepted_gpu_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("accepted_gpu_names must contain at least one unique name")
        object.__setattr__(self, "accepted_gpu_names", names)
        if not isinstance(self.minimum_cuda_version, CudaVersion):
            object.__setattr__(self, "minimum_cuda_version", CudaVersion.parse(self.minimum_cuda_version))
        for field in (
            "minimum_gpu_memory_bytes",
            "minimum_system_ram_bytes",
            "minimum_free_disk_bytes",
        ):
            _require_integer(getattr(self, field), field, minimum=1)
        _require_integer(self.minimum_cpu_cores, "minimum_cpu_cores", minimum=1)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "schema": PROFILE_SCHEMA,
            "accepted_gpu_names": [name.value for name in self.accepted_gpu_names],
            "minimum_gpu_memory_bytes": self.minimum_gpu_memory_bytes,
            "minimum_cuda_version": str(self.minimum_cuda_version),
            "minimum_system_ram_bytes": self.minimum_system_ram_bytes,
            "minimum_cpu_cores": self.minimum_cpu_cores,
            "minimum_free_disk_bytes": self.minimum_free_disk_bytes,
        }

    def to_json(self) -> str:
        """Return deterministic JSON for the profile."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: object) -> EvaluatorCapabilityProfile:
        """Parse one profile while rejecting unknown or missing fields."""

        if not isinstance(value, Mapping):
            raise ValueError("evaluator capability profile must be an object")
        required = {
            "schema",
            "accepted_gpu_names",
            "minimum_gpu_memory_bytes",
            "minimum_cuda_version",
            "minimum_system_ram_bytes",
            "minimum_cpu_cores",
            "minimum_free_disk_bytes",
        }
        actual = set(value)
        missing = required - actual
        extra = actual - required
        if missing or extra:
            raise ValueError(f"profile fields are not exact: missing={sorted(missing)}, extra={sorted(extra)}")
        if value["schema"] != PROFILE_SCHEMA:
            raise ValueError("evaluator capability profile schema is not supported")
        names = value["accepted_gpu_names"]
        if not isinstance(names, (list, tuple)):
            raise ValueError("accepted_gpu_names must be an array")
        try:
            parsed_names = tuple(GpuName(name) for name in names)
            parsed_cuda = CudaVersion.parse(value["minimum_cuda_version"])
            return cls(
                accepted_gpu_names=parsed_names,
                minimum_gpu_memory_bytes=_require_integer(
                    value["minimum_gpu_memory_bytes"], "minimum_gpu_memory_bytes", minimum=1
                ),
                minimum_cuda_version=parsed_cuda,
                minimum_system_ram_bytes=_require_integer(
                    value["minimum_system_ram_bytes"], "minimum_system_ram_bytes", minimum=1
                ),
                minimum_cpu_cores=_require_integer(value["minimum_cpu_cores"], "minimum_cpu_cores", minimum=1),
                minimum_free_disk_bytes=_require_integer(
                    value["minimum_free_disk_bytes"], "minimum_free_disk_bytes", minimum=1
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("evaluator capability profile contains invalid values") from error

    @classmethod
    def from_json(cls, value: str) -> EvaluatorCapabilityProfile:
        """Parse one strict JSON profile."""

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("evaluator capability profile JSON is invalid") from error
        return cls.from_dict(decoded)


DEFAULT_EVALUATOR_PROFILE = EvaluatorCapabilityProfile()


@dataclass(frozen=True)
class GpuInventory:
    """Observed facts for one GPU returned by the boundary probe."""

    name: GpuName
    memory_bytes: int
    cuda_runtime: CudaVersion

    def __post_init__(self) -> None:
        if not isinstance(self.name, GpuName):
            object.__setattr__(self, "name", GpuName(self.name))
        _require_integer(self.memory_bytes, "GPU memory bytes", minimum=1)
        if not isinstance(self.cuda_runtime, CudaVersion):
            object.__setattr__(self, "cuda_runtime", CudaVersion.parse(self.cuda_runtime))


@dataclass(frozen=True)
class CgroupLimits:
    """Optional cgroup limits discovered for the current process."""

    memory_limit_bytes: int | None = None
    cpu_quota_cores: Fraction | None = None
    cpuset_cpu_count: int | None = None

    def __post_init__(self) -> None:
        if self.memory_limit_bytes is not None:
            _require_integer(self.memory_limit_bytes, "cgroup memory limit", minimum=1)
        if self.cpu_quota_cores is not None:
            if not isinstance(self.cpu_quota_cores, Fraction) or self.cpu_quota_cores <= 0:
                raise ValueError("cgroup CPU quota must be a positive fraction")
        if self.cpuset_cpu_count is not None:
            _require_integer(self.cpuset_cpu_count, "cgroup cpuset CPU count", minimum=1)


@dataclass(frozen=True)
class HostInventory:
    """Effective evaluator-host facts after cgroup limits are applied."""

    gpus: tuple[GpuInventory, ...]
    effective_system_ram_bytes: int
    effective_cpu_cores: int
    free_disk_bytes: int
    cgroup_limits: CgroupLimits = CgroupLimits()

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpus", tuple(self.gpus))
        _require_integer(self.effective_system_ram_bytes, "effective system RAM bytes", minimum=1)
        _require_integer(self.effective_cpu_cores, "effective CPU cores", minimum=1)
        _require_integer(self.free_disk_bytes, "free disk bytes", minimum=0)


class NvidiaSmiProbe(Protocol):
    """Boundary protocol returning strict machine-readable nvidia-smi facts."""

    def query(self) -> str:
        """Return nvidia-smi output for parsing."""


class SystemProbe(Protocol):
    """Boundary protocol for host totals, disk, and cgroup file reads."""

    def cpu_count(self) -> int | None:
        """Return host-visible logical CPU count."""

    def memory_total_bytes(self) -> int:
        """Return host-visible memory bytes."""

    def disk_free_bytes(self, path: Path) -> int:
        """Return free bytes for one filesystem."""

    def read_text(self, path: Path) -> str | None:
        """Read one optional cgroup or proc file."""


class NvidiaSmiCommandProbe:
    """Default nvidia-smi probe using subprocess without shell interpolation."""

    def __init__(self, executable: str = "nvidia-smi") -> None:
        if not executable or any(character in executable for character in "\r\n\x00"):
            raise ValueError("nvidia-smi executable is invalid")
        self._executable = executable

    def query(self) -> str:
        """Return one strict CSV snapshot with GPU and CUDA facts."""

        query_command = (
            self._executable,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        )
        try:
            rows = subprocess.run(
                query_command,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            summary = subprocess.run(
                (self._executable,),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise PreflightError(PreflightFailureReason.NVIDIA_SMI_UNAVAILABLE) from error
        cuda_match = re.search(r"\bCUDA Version:\s*(\d+\.\d+(?:\.\d+)?)\b", summary)
        if cuda_match is None:
            raise PreflightError(PreflightFailureReason.NVIDIA_SMI_MALFORMED)
        lines = [f"{line}, {cuda_match.group(1)}" for line in rows.splitlines() if line.strip()]
        if not lines:
            raise PreflightError(PreflightFailureReason.NVIDIA_SMI_MALFORMED)
        return "\n".join(lines)


class _OperatingSystemProbe:
    """Default host probe kept private to the single collector boundary."""

    def cpu_count(self) -> int | None:
        return os.cpu_count()

    def memory_total_bytes(self) -> int:
        path = Path("/proc/meminfo")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise PreflightError(PreflightFailureReason.SYSTEM_MEMORY_UNAVAILABLE) from error
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB" and _INTEGER_RE.fullmatch(fields[1]):
                    return int(fields[1]) * 1024
                break
        raise PreflightError(PreflightFailureReason.SYSTEM_MEMORY_UNAVAILABLE)

    def disk_free_bytes(self, path: Path) -> int:
        try:
            return shutil.disk_usage(path).free
        except OSError as error:
            raise PreflightError(PreflightFailureReason.DISK_SPACE_UNAVAILABLE) from error

    def read_text(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED) from error


def _probe_output(probe: NvidiaSmiProbe) -> str:
    """Call the typed probe while allowing a simple callable test double."""

    try:
        return probe.query()
    except AttributeError:
        if callable(probe):
            return probe()  # type: ignore[operator]
        raise


def parse_nvidia_smi_output(value: str) -> tuple[GpuInventory, ...]:
    """Parse strict ``name,memory MiB,cuda version`` nvidia-smi output."""

    if not isinstance(value, str):
        raise PreflightError(PreflightFailureReason.NVIDIA_SMI_MALFORMED)
    rows = [line.strip() for line in value.splitlines() if line.strip()]
    if not rows:
        raise PreflightError(PreflightFailureReason.NVIDIA_SMI_MALFORMED)
    parsed: list[GpuInventory] = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3:
            raise PreflightError(PreflightFailureReason.NVIDIA_SMI_MALFORMED)
        name, memory, cuda = fields
        if name.lower() == "name" and memory.lower().startswith("memory.total"):
            continue
        memory_match = _MEMORY_RE.fullmatch(memory)
        if memory_match is None:
            raise PreflightError(PreflightFailureReason.NVIDIA_SMI_MALFORMED)
        try:
            parsed.append(
                GpuInventory(
                    name=GpuName(name),
                    memory_bytes=int(memory_match.group("value")) * MIB,
                    cuda_runtime=CudaVersion.parse(cuda),
                )
            )
        except ValueError as error:
            raise PreflightError(PreflightFailureReason.NVIDIA_SMI_MALFORMED) from error
    if not parsed:
        raise PreflightError(PreflightFailureReason.NVIDIA_SMI_MALFORMED)
    return tuple(parsed)


def _parse_optional_integer(value: str | None, field: str, *, allow_negative_one: bool = False) -> int | None:
    if value is not None:
        value = value.strip()
    if value is None or value == "max":
        return None
    if allow_negative_one and value == "-1":
        return None
    if _INTEGER_RE.fullmatch(value) is None:
        raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
    parsed = int(value)
    if parsed <= 0:
        raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
    return parsed


def _parse_cpu_quota(value: str | None, *, v2: bool) -> Fraction | None:
    if value is None:
        return None
    fields = value.strip().split()
    if v2:
        if len(fields) != 2:
            raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
        quota, period = fields
    else:
        if len(fields) != 1:
            raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
        quota = fields[0]
        period = None
    if quota == "max":
        return None
    if _INTEGER_RE.fullmatch(quota) is None or period is None or _INTEGER_RE.fullmatch(period) is None:
        raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
    quota_value = int(quota)
    period_value = int(period)
    if quota_value <= 0 or period_value <= 0:
        raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
    return Fraction(quota_value, period_value)


def _parse_cpuset(value: str | None) -> int | None:
    if value is None or not value:
        return None
    value = value.strip()
    if not value:
        return None
    total = 0
    seen: set[int] = set()
    for part in value.split(","):
        match = _CPUSET_PART_RE.fullmatch(part)
        if match is None:
            raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
        first = int(match.group("first"))
        last = int(match.group("last") or match.group("first"))
        if last < first:
            raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
        for index in range(first, last + 1):
            if index in seen:
                raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
            seen.add(index)
        total += last - first + 1
    return total


def read_cgroup_limits(root: Path, probe: SystemProbe) -> CgroupLimits:
    """Read cgroup v2 or v1 limits from one injected filesystem boundary."""

    v2_memory = probe.read_text(root / "memory.max")
    v2_cpu = probe.read_text(root / "cpu.max")
    v2_cpuset = probe.read_text(root / "cpuset.cpus.effective") or probe.read_text(root / "cpuset.cpus")
    if v2_memory is not None or v2_cpu is not None or v2_cpuset is not None:
        memory = _parse_optional_integer(v2_memory, "memory.max")
        quota = _parse_cpu_quota(v2_cpu, v2=True)
        cpuset = _parse_cpuset(v2_cpuset)
        return CgroupLimits(memory, quota, cpuset)

    memory = _parse_optional_integer(
        probe.read_text(root / "memory" / "memory.limit_in_bytes"),
        "memory limit",
        allow_negative_one=True,
    )
    quota_raw = probe.read_text(root / "cpu" / "cpu.cfs_quota_us")
    period_raw = probe.read_text(root / "cpu" / "cpu.cfs_period_us")
    if quota_raw is None and period_raw is not None:
        raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
    quota = None
    if quota_raw is not None:
        if period_raw is None:
            raise PreflightError(PreflightFailureReason.CGROUP_MALFORMED)
        if quota_raw.strip() == "-1":
            quota = None
        else:
            quota = _parse_cpu_quota(f"{quota_raw} {period_raw}", v2=True)
    cpuset = _parse_cpuset(
        probe.read_text(root / "cpuset" / "cpuset.cpus.effective") or probe.read_text(root / "cpuset" / "cpuset.cpus")
    )
    return CgroupLimits(memory, quota, cpuset)


def _effective_cpu_cores(host_count: int | None, limits: CgroupLimits) -> int:
    if host_count is None or isinstance(host_count, bool) or host_count <= 0:
        raise PreflightError(PreflightFailureReason.CPU_COUNT_UNAVAILABLE)
    effective = host_count
    if limits.cpu_quota_cores is not None:
        effective = min(effective, math.floor(limits.cpu_quota_cores))
    if limits.cpuset_cpu_count is not None:
        effective = min(effective, limits.cpuset_cpu_count)
    if effective <= 0:
        raise PreflightError(PreflightFailureReason.CPU_COUNT_UNAVAILABLE)
    return effective


def collect_host_inventory(
    *,
    nvidia_smi_probe: NvidiaSmiProbe | None = None,
    system_probe: SystemProbe | None = None,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    disk_path: Path = Path("."),
) -> HostInventory:
    """Collect GPU, cgroup-aware CPU/RAM, and free-disk facts at one boundary."""

    gpu_probe = nvidia_smi_probe or NvidiaSmiCommandProbe()
    host_probe = system_probe or _OperatingSystemProbe()
    try:
        gpus = parse_nvidia_smi_output(_probe_output(gpu_probe))
    except PreflightError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise PreflightError(PreflightFailureReason.NVIDIA_SMI_MALFORMED) from error
    limits = read_cgroup_limits(cgroup_root, host_probe)
    try:
        host_memory = host_probe.memory_total_bytes()
    except PreflightError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise PreflightError(PreflightFailureReason.SYSTEM_MEMORY_UNAVAILABLE) from error
    if isinstance(host_memory, bool) or not isinstance(host_memory, int) or host_memory <= 0:
        raise PreflightError(PreflightFailureReason.SYSTEM_MEMORY_UNAVAILABLE)
    effective_memory = min(host_memory, limits.memory_limit_bytes or host_memory)
    try:
        host_cpu_count = host_probe.cpu_count()
        effective_cpu = _effective_cpu_cores(host_cpu_count, limits)
        free_disk = host_probe.disk_free_bytes(disk_path)
    except PreflightError:
        raise
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise PreflightError(PreflightFailureReason.DISK_SPACE_UNAVAILABLE) from error
    if isinstance(free_disk, bool) or not isinstance(free_disk, int) or free_disk < 0:
        raise PreflightError(PreflightFailureReason.DISK_SPACE_UNAVAILABLE)
    return HostInventory(gpus, effective_memory, effective_cpu, free_disk, limits)


@dataclass(frozen=True)
class QualificationPassed:
    """Successful typed evaluator-host qualification."""

    profile: EvaluatorCapabilityProfile
    inventory: HostInventory
    gpu: GpuInventory


@dataclass(frozen=True)
class QualificationRejected:
    """Failed typed evaluator-host qualification."""

    reason: PreflightFailureReason
    profile: EvaluatorCapabilityProfile
    inventory: HostInventory
    observed_gpu: GpuInventory | None = None


QualificationResult = QualificationPassed | QualificationRejected


def qualify_host(
    inventory: HostInventory,
    profile: EvaluatorCapabilityProfile = DEFAULT_EVALUATOR_PROFILE,
) -> QualificationResult:
    """Compare effective inventory facts with a capability profile."""

    if not inventory.gpus:
        return QualificationRejected(PreflightFailureReason.GPU_NOT_FOUND, profile, inventory)
    named = tuple(gpu for gpu in inventory.gpus if gpu.name in profile.accepted_gpu_names)
    if not named:
        return QualificationRejected(PreflightFailureReason.GPU_NAME_MISMATCH, profile, inventory, inventory.gpus[0])
    gpu = named[0]
    if gpu.memory_bytes < profile.minimum_gpu_memory_bytes:
        return QualificationRejected(PreflightFailureReason.GPU_MEMORY_TOO_SMALL, profile, inventory, gpu)
    if gpu.cuda_runtime < profile.minimum_cuda_version:
        return QualificationRejected(PreflightFailureReason.CUDA_VERSION_TOO_OLD, profile, inventory, gpu)
    if inventory.effective_system_ram_bytes < profile.minimum_system_ram_bytes:
        return QualificationRejected(PreflightFailureReason.SYSTEM_RAM_TOO_SMALL, profile, inventory, gpu)
    if inventory.effective_cpu_cores < profile.minimum_cpu_cores:
        return QualificationRejected(PreflightFailureReason.CPU_CORES_TOO_FEW, profile, inventory, gpu)
    if inventory.free_disk_bytes < profile.minimum_free_disk_bytes:
        return QualificationRejected(PreflightFailureReason.FREE_DISK_TOO_SMALL, profile, inventory, gpu)
    return QualificationPassed(profile, inventory, gpu)


@dataclass(frozen=True, order=True)
class RelativeArtifactPath:
    """One safe POSIX relative artifact path."""

    value: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or not self.value
            or any(character in self.value for character in "\r\n\x00")
        ):
            raise ValueError("artifact path is invalid")
        if (
            self.value.startswith("/")
            or PureWindowsPath(self.value).is_absolute()
            or PureWindowsPath(self.value).drive
        ):
            raise ValueError("artifact path must be relative")
        if "://" in self.value:
            raise ValueError("artifact path must not be a URL")
        path = PurePosixPath(self.value)
        if path == PurePosixPath(".") or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("artifact path contains traversal")
        if "\\" in self.value or path.as_posix() != self.value:
            raise ValueError("artifact path must use canonical POSIX separators")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ExpectedArtifact:
    """One sealed artifact identified by relative path, length, and digest."""

    relative_path: RelativeArtifactPath
    length_bytes: int
    sha256: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, RelativeArtifactPath):
            object.__setattr__(self, "relative_path", RelativeArtifactPath(self.relative_path))
        if not isinstance(self.sha256, Sha256Digest):
            object.__setattr__(self, "sha256", Sha256Digest(self.sha256))
        _require_integer(self.length_bytes, "artifact length bytes", minimum=0)


@dataclass(frozen=True, init=False)
class ExpectedArtifactSet:
    """Immutable exact artifact set with no duplicate or unsafe paths."""

    artifacts: tuple[ExpectedArtifact, ...]

    def __init__(self, artifacts: Iterable[ExpectedArtifact]) -> None:
        parsed = tuple(
            artifact if isinstance(artifact, ExpectedArtifact) else ExpectedArtifact(*artifact)
            for artifact in artifacts
        )
        paths = [artifact.relative_path.value for artifact in parsed]
        if len(paths) != len(set(paths)):
            raise ValueError("expected artifact paths must be unique")
        object.__setattr__(self, "artifacts", parsed)

    def __iter__(self):
        return iter(self.artifacts)

    def __len__(self) -> int:
        return len(self.artifacts)


@dataclass(frozen=True)
class VerifiedArtifact:
    """One artifact verified against its sealed identity."""

    relative_path: RelativeArtifactPath
    length_bytes: int
    sha256: Sha256Digest


def _artifact_failure(reason: PreflightFailureReason, path: RelativeArtifactPath) -> PreflightError:
    """Build a safe artifact failure without including the absolute root."""

    return PreflightError(reason, f"{reason.value}: {path.value}")


def verify_artifacts(root: Path, expected: ExpectedArtifactSet) -> tuple[VerifiedArtifact, ...]:
    """Verify every expected artifact, including files already present in cache."""

    root_path = Path(root)
    verified: list[VerifiedArtifact] = []
    for artifact in expected:
        relative = artifact.relative_path
        path = root_path / relative.value
        try:
            if path.is_symlink() or not path.is_file():
                raise _artifact_failure(PreflightFailureReason.ARTIFACT_MISSING, relative)
            length = path.stat().st_size
        except PreflightError:
            raise
        except OSError as error:
            raise _artifact_failure(PreflightFailureReason.ARTIFACT_MISSING, relative) from error
        if length != artifact.length_bytes:
            raise _artifact_failure(PreflightFailureReason.ARTIFACT_LENGTH_MISMATCH, relative)
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            after_length = path.stat().st_size
        except OSError as error:
            raise _artifact_failure(PreflightFailureReason.ARTIFACT_MISSING, relative) from error
        if after_length != length:
            raise _artifact_failure(PreflightFailureReason.ARTIFACT_LENGTH_MISMATCH, relative)
        if digest.hexdigest() != artifact.sha256.value:
            raise _artifact_failure(PreflightFailureReason.ARTIFACT_DIGEST_MISMATCH, relative)
        verified.append(VerifiedArtifact(relative, length, Sha256Digest(digest.hexdigest())))
    return tuple(verified)


T = TypeVar("T")


@dataclass(frozen=True)
class PreflightSuccess(Generic[T]):
    """Successful host, artifact, and allocation preflight."""

    qualification: QualificationPassed
    artifacts: tuple[VerifiedArtifact, ...]
    allocation: T


def run_evaluator_preflight(
    *,
    root: Path,
    expected_artifacts: ExpectedArtifactSet,
    allocation_callback: Callable[[], T],
    inventory: HostInventory | None = None,
    profile: EvaluatorCapabilityProfile = DEFAULT_EVALUATOR_PROFILE,
    nvidia_smi_probe: NvidiaSmiProbe | None = None,
    system_probe: SystemProbe | None = None,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    disk_path: Path = Path("."),
) -> PreflightSuccess[T]:
    """Qualify, verify all artifacts, then invoke allocation exactly once.

    The callback is unreachable until artifact verification succeeds.  It is
    intentionally opaque so this module cannot import CUDA or a model.
    """

    try:
        inventory_value = (
            inventory
            if inventory is not None
            else collect_host_inventory(
                nvidia_smi_probe=nvidia_smi_probe,
                system_probe=system_probe,
                cgroup_root=cgroup_root,
                disk_path=disk_path,
            )
        )
        qualification = qualify_host(inventory_value, profile)
    except ValueError as error:
        raise PreflightError(PreflightFailureReason.INVALID_PROFILE) from error
    if isinstance(qualification, QualificationRejected):
        raise PreflightError(qualification.reason)
    artifacts = verify_artifacts(root, expected_artifacts)
    try:
        allocation = allocation_callback()
    except Exception as error:
        raise PreflightError(PreflightFailureReason.ALLOCATION_FAILED) from error
    return PreflightSuccess(qualification, artifacts, allocation)


__all__ = [
    "CudaVersion",
    "CgroupLimits",
    "DEFAULT_EVALUATOR_PROFILE",
    "DEFAULT_GPU_NAME",
    "DEFAULT_MIN_CPU_CORES",
    "DEFAULT_MIN_CUDA_VERSION",
    "DEFAULT_MIN_FREE_DISK_BYTES",
    "DEFAULT_MIN_GPU_MEMORY_BYTES",
    "DEFAULT_MIN_SYSTEM_RAM_BYTES",
    "EvaluatorCapabilityProfile",
    "GIB",
    "ExpectedArtifact",
    "ExpectedArtifactSet",
    "GpuInventory",
    "GpuName",
    "HostInventory",
    "MIB",
    "NvidiaSmiCommandProbe",
    "NvidiaSmiProbe",
    "PreflightError",
    "PreflightFailureReason",
    "PreflightSuccess",
    "PROFILE_SCHEMA",
    "QualificationPassed",
    "QualificationRejected",
    "QualificationResult",
    "RelativeArtifactPath",
    "Sha256Digest",
    "SystemProbe",
    "VerifiedArtifact",
    "collect_host_inventory",
    "parse_nvidia_smi_output",
    "qualify_host",
    "read_cgroup_limits",
    "run_evaluator_preflight",
    "verify_artifacts",
]
