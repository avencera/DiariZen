"""CPU-only tests for the standalone evaluator preflight."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from recipes.speakrs.large.validation import preflight as preflight_module
from recipes.speakrs.large.validation.contracts import Sha256Digest
from recipes.speakrs.large.validation.preflight import (
    DEFAULT_EVALUATOR_PROFILE,
    DEFAULT_MIN_FREE_DISK_BYTES,
    GIB,
    CudaVersion,
    EvaluatorCapabilityProfile,
    ExpectedArtifact,
    ExpectedArtifactSet,
    GpuInventory,
    GpuName,
    HostInventory,
    PreflightError,
    PreflightFailureReason,
    QualificationPassed,
    RelativeArtifactPath,
    collect_host_inventory,
    parse_nvidia_smi_output,
    qualify_host,
    run_evaluator_preflight,
)


@dataclass
class FakeSystemProbe:
    """In-memory system and cgroup boundary for deterministic tests."""

    host_cpu_count: int = 16
    host_memory_bytes: int = 64 * GIB
    free_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES
    files: dict[Path, str] | None = None

    def cpu_count(self) -> int:
        return self.host_cpu_count

    def memory_total_bytes(self) -> int:
        return self.host_memory_bytes

    def disk_free_bytes(self, path: Path) -> int:
        return self.free_bytes

    def read_text(self, path: Path) -> str | None:
        return (self.files or {}).get(path)


class FakeNvidiaSmiProbe:
    """In-memory nvidia-smi output probe."""

    def __init__(self, output: str) -> None:
        self.output = output

    def query(self) -> str:
        return self.output


def _gpu(name: str = "RTX 5060 Ti", memory_bytes: int = 16 * GIB, cuda: str = "12.8") -> GpuInventory:
    return GpuInventory(GpuName(name), memory_bytes, CudaVersion.parse(cuda))


def _inventory(
    *,
    gpu: GpuInventory | None = None,
    ram: int = 32 * GIB,
    cpus: int = 4,
    disk: int = DEFAULT_MIN_FREE_DISK_BYTES,
) -> HostInventory:
    return HostInventory((gpu or _gpu(),), ram, cpus, disk)


def _artifact(root: Path, relative_path: str = "cache/model.bin", payload: bytes = b"model") -> ExpectedArtifact:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ExpectedArtifact(relative_path, len(payload), Sha256Digest(hashlib.sha256(payload).hexdigest()))


def test_default_profile_requires_exact_spaced_name_and_round_trips_strictly() -> None:
    assert DEFAULT_EVALUATOR_PROFILE.accepted_gpu_names == (GpuName("RTX 5060 Ti"),)
    assert DEFAULT_EVALUATOR_PROFILE.minimum_gpu_memory_bytes == 16 * GIB
    assert DEFAULT_EVALUATOR_PROFILE.minimum_cuda_version == CudaVersion(12, 8)
    assert DEFAULT_EVALUATOR_PROFILE.minimum_system_ram_bytes == 32 * GIB
    assert DEFAULT_EVALUATOR_PROFILE.minimum_cpu_cores == 4
    assert DEFAULT_EVALUATOR_PROFILE.minimum_free_disk_bytes == DEFAULT_MIN_FREE_DISK_BYTES

    encoded = DEFAULT_EVALUATOR_PROFILE.to_dict()
    assert EvaluatorCapabilityProfile.from_dict(encoded) == DEFAULT_EVALUATOR_PROFILE
    encoded["unexpected"] = "rejected"
    with pytest.raises(ValueError, match="fields are not exact"):
        EvaluatorCapabilityProfile.from_dict(encoded)


@pytest.mark.parametrize(
    ("name", "reason"),
    (
        ("RTX_5060_Ti", PreflightFailureReason.GPU_NAME_MISMATCH),
        ("RTX 5060", PreflightFailureReason.GPU_NAME_MISMATCH),
    ),
)
def test_gpu_name_is_exact_after_whitespace_normalization(name: str, reason: PreflightFailureReason) -> None:
    inventory = _inventory(gpu=_gpu(name))

    result = qualify_host(inventory)

    assert getattr(result, "reason", None) is reason


def test_gpu_name_normalizes_only_the_standard_nvidia_vendor_prefix() -> None:
    assert GpuName("NVIDIA GeForce RTX 5060 Ti").value == "RTX 5060 Ti"
    assert GpuName("RTX   5060 Ti").value == "RTX 5060 Ti"


def test_profile_can_add_a_proven_sixteen_gib_rtx_5060_without_lowering_default_memory() -> None:
    profile = EvaluatorCapabilityProfile(
        accepted_gpu_names=(GpuName("RTX 5060 Ti"), GpuName("RTX 5060")),
    )

    assert getattr(qualify_host(_inventory(gpu=_gpu("RTX 5060", 16 * GIB)), profile), "gpu", None) is not None
    assert getattr(qualify_host(_inventory(gpu=_gpu("RTX 5060", 8 * GIB)), profile), "reason", None) is (
        PreflightFailureReason.GPU_MEMORY_TOO_SMALL
    )


@pytest.mark.parametrize(
    ("inventory", "reason"),
    (
        (_inventory(gpu=_gpu(memory_bytes=16 * GIB - 1)), PreflightFailureReason.GPU_MEMORY_TOO_SMALL),
        (_inventory(gpu=_gpu(cuda="12.7")), PreflightFailureReason.CUDA_VERSION_TOO_OLD),
        (_inventory(ram=32 * GIB - 1), PreflightFailureReason.SYSTEM_RAM_TOO_SMALL),
        (_inventory(cpus=3), PreflightFailureReason.CPU_CORES_TOO_FEW),
        (_inventory(disk=DEFAULT_MIN_FREE_DISK_BYTES - 1), PreflightFailureReason.FREE_DISK_TOO_SMALL),
    ),
)
def test_default_profile_rejects_each_lower_bound(inventory: HostInventory, reason: PreflightFailureReason) -> None:
    result = qualify_host(inventory)

    assert getattr(result, "reason", None) is reason


def test_nvidia_smi_parser_accepts_spaces_and_rejects_wrong_shape() -> None:
    inventory = parse_nvidia_smi_output("name, memory.total [MiB], cuda_version\nRTX 5060 Ti, 16384, 12.8")

    assert inventory == (_gpu(),)
    with pytest.raises(PreflightError) as error:
        parse_nvidia_smi_output("RTX 5060 Ti, 16384 MiB")
    assert error.value.reason is PreflightFailureReason.NVIDIA_SMI_MALFORMED


def test_collector_applies_cgroup_v2_memory_cpu_and_cpuset_limits(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    files = {
        cgroup / "memory.max": str(32 * GIB),
        cgroup / "cpu.max": "400000 100000",
        cgroup / "cpuset.cpus.effective": "0-7",
    }
    probe = FakeSystemProbe(files=files)

    inventory = collect_host_inventory(
        nvidia_smi_probe=FakeNvidiaSmiProbe("RTX 5060 Ti, 16384, 12.8"),
        system_probe=probe,
        cgroup_root=cgroup,
    )

    assert inventory.effective_system_ram_bytes == 32 * GIB
    assert inventory.effective_cpu_cores == 4
    assert inventory.cgroup_limits.cpuset_cpu_count == 8


def test_collector_applies_cgroup_v1_limits(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    files = {
        cgroup / "memory" / "memory.limit_in_bytes": str(32 * GIB),
        cgroup / "cpu" / "cpu.cfs_quota_us": "300000",
        cgroup / "cpu" / "cpu.cfs_period_us": "100000",
        cgroup / "cpuset" / "cpuset.cpus": "2-5",
    }
    probe = FakeSystemProbe(files=files)

    inventory = collect_host_inventory(
        nvidia_smi_probe=FakeNvidiaSmiProbe("RTX 5060 Ti, 16384, 12.8"),
        system_probe=probe,
        cgroup_root=cgroup,
    )

    assert inventory.effective_system_ram_bytes == 32 * GIB
    assert inventory.effective_cpu_cores == 3
    assert inventory.cgroup_limits.cpuset_cpu_count == 4


def test_artifact_corruption_is_rejected_before_gpu_allocation(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, payload=b"known bytes")
    artifact_path = tmp_path / artifact.relative_path.value
    artifact_path.write_bytes(b"known bytez")
    calls: list[str] = []

    with pytest.raises(PreflightError) as error:
        run_evaluator_preflight(
            root=tmp_path,
            expected_artifacts=ExpectedArtifactSet((artifact,)),
            inventory=_inventory(),
            allocation_callback=lambda: calls.append("allocated"),
        )

    assert error.value.reason is PreflightFailureReason.ARTIFACT_DIGEST_MISMATCH
    assert calls == []


def test_missing_artifact_is_rejected_before_gpu_allocation(tmp_path: Path) -> None:
    artifact = ExpectedArtifact("cache/missing.bin", 3, "a" * 64)
    calls: list[str] = []

    with pytest.raises(PreflightError) as error:
        run_evaluator_preflight(
            root=tmp_path,
            expected_artifacts=ExpectedArtifactSet((artifact,)),
            inventory=_inventory(),
            allocation_callback=lambda: calls.append("allocated"),
        )

    assert error.value.reason is PreflightFailureReason.ARTIFACT_MISSING
    assert calls == []


def test_artifact_length_change_is_rejected_before_gpu_allocation(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, payload=b"known bytes")
    (tmp_path / artifact.relative_path.value).write_bytes(b"known bytes plus")
    calls: list[str] = []

    with pytest.raises(PreflightError) as error:
        run_evaluator_preflight(
            root=tmp_path,
            expected_artifacts=ExpectedArtifactSet((artifact,)),
            inventory=_inventory(),
            allocation_callback=lambda: calls.append("allocated"),
        )

    assert error.value.reason is PreflightFailureReason.ARTIFACT_LENGTH_MISMATCH
    assert calls == []


def test_successful_preflight_verifies_cache_before_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _artifact(tmp_path)
    events: list[str] = []
    original_verify = preflight_module.verify_artifacts

    def verify(root: Path, expected: ExpectedArtifactSet):
        events.append("verify")
        return original_verify(root, expected)

    monkeypatch.setattr(preflight_module, "verify_artifacts", verify)

    result = run_evaluator_preflight(
        root=tmp_path,
        expected_artifacts=ExpectedArtifactSet((artifact,)),
        inventory=_inventory(),
        allocation_callback=lambda: events.append("allocate") or "gpu-context",
    )

    assert isinstance(result.qualification, QualificationPassed)
    assert result.allocation == "gpu-context"
    assert events == ["verify", "allocate"]


@pytest.mark.parametrize("path", ("../escape.bin", "/absolute.bin", "https://signed.example/object"))
def test_artifact_set_rejects_traversal_and_urls(path: str) -> None:
    with pytest.raises(ValueError):
        ExpectedArtifact(path, 1, "a" * 64)


def test_artifact_set_rejects_duplicate_paths() -> None:
    first = ExpectedArtifact("cache/model.bin", 1, "a" * 64)
    second = ExpectedArtifact("cache/model.bin", 1, "b" * 64)

    with pytest.raises(ValueError, match="unique"):
        ExpectedArtifactSet((first, second))


def test_failure_detail_does_not_include_absolute_root_or_digest() -> None:
    error = PreflightError(PreflightFailureReason.ARTIFACT_DIGEST_MISMATCH)

    assert "/private/secret" not in str(error)
    assert "https://" not in str(error)
    assert error.to_dict() == {"reason": "artifact_digest_mismatch", "detail": "artifact_digest_mismatch"}


def test_relative_path_type_is_canonical() -> None:
    assert RelativeArtifactPath("cache/model.bin").value == "cache/model.bin"
    with pytest.raises(ValueError):
        RelativeArtifactPath("cache\\model.bin")
