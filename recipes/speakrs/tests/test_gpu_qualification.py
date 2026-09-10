"""CPU tests for the worker-side GPU qualification orchestration."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from recipes.speakrs.large import qualification as qualification_module
from recipes.speakrs.large.qualification import (
    AttemptResult,
    QualificationError,
    QualificationSpec,
    _bind_collate,
    _code_hash,
    _run_isolated_real_attempt,
    estimate_training_hours,
    execute_qualification,
    load_qualification_spec,
    percentile,
    run_qualification,
    select_largest_batch,
)


def _collate_stub(batch, *, max_speakers_per_chunk):
    return batch, max_speakers_per_chunk


def _trainer_toml(tmp_path: Path) -> Path:
    path = tmp_path / "trainer.toml"
    path.write_text("[meta]\nseed = 3407\n", encoding="utf-8")
    return path


def test_defaults_match_first_run_profile() -> None:
    spec = QualificationSpec()

    assert spec.warmup_optimizer_updates == 20
    assert spec.measured_optimizer_updates == 100
    assert spec.effective_batch == 64
    assert spec.physical_batch_candidates == (2, 4, 8)
    assert spec.planned_updates_per_cycle == 2000
    assert spec.planned_max_cycles == 100
    assert spec.accumulation_for(2) == 32
    assert spec.accumulation_for(8) == 8


def test_bound_collate_is_picklable_for_data_loader_workers() -> None:
    bound = _bind_collate(_collate_stub, 4)

    restored = pickle.loads(pickle.dumps(bound))

    assert restored(["audio"]) == (["audio"], 4)


def test_real_code_hash_covers_model_dataset_and_trainers() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    config_path = repository_root / "recipes" / "speakrs" / "conf" / "large_wavlm_large_4090.toml"

    _digest_value, files = _code_hash(
        {"model": {}, "trainer": {}, "train_dataset": {}},
        config_path,
    )

    suffixes = {Path(path).relative_to(repository_root).as_posix() for path in files}
    assert "diarizen/models/eend/model_wavlm_conformer.py" in suffixes
    assert "diarizen/trainer_dual_opt.py" in suffixes
    assert "diarizen/trainer_utils.py" in suffixes
    assert "recipes/diar_ssl/dataset.py" in suffixes
    assert "recipes/diar_ssl/trainer_dual_opt.py" in suffixes


def test_parser_rejects_invalid_candidate() -> None:
    with pytest.raises(QualificationError):
        QualificationSpec.from_mapping({"effective_batch": 64, "physical_batch_candidates": [3]})


def test_percentile_and_training_estimate() -> None:
    assert percentile((1.0, 2.0, 8.0, 4.0), 50) == 3.0
    estimate = estimate_training_hours(1.8)
    assert estimate["hours_per_cycle"] == pytest.approx(1.0)
    assert estimate["hours_for_max_cycles"] == pytest.approx(100.0)


def test_select_largest_batch_requires_headroom() -> None:
    probes = [
        AttemptResult(2, 32, True, total_memory_bytes=100, peak_reserved_bytes=80),
        AttemptResult(4, 16, True, total_memory_bytes=100, peak_reserved_bytes=80),
        AttemptResult(8, 8, True, total_memory_bytes=100, peak_reserved_bytes=90),
    ]

    selected = select_largest_batch(probes)

    assert selected.physical_batch == 4


def test_cpu_fake_runner_probes_then_measures_only_selected(tmp_path: Path) -> None:
    config = _trainer_toml(tmp_path)
    spec = QualificationSpec(warmup_optimizer_updates=1, measured_optimizer_updates=3)
    calls: list[tuple[str, int]] = []
    windows: list[tuple[int, int]] = []

    def runner(**kwargs):
        phase = kwargs["phase"]
        physical_batch = kwargs["physical_batch"]
        calls.append((phase, physical_batch))
        windows.append((kwargs["warmup_updates"], kwargs["measured_updates"]))
        if phase == "probe" and physical_batch == 8:
            raise RuntimeError("CUDA out of memory")
        count = 3
        return {
            "ok": True,
            "total_memory_bytes": 100,
            "peak_reserved_bytes": 70,
            "update_seconds": [1.0] * count,
            "measured_update_count": count,
            "checkpoint_write_reload": {"feasible": True, "write": True, "reload": True},
        }

    report = run_qualification(
        config,
        "4090",
        spec,
        attempt_runner=runner,
        device_facts={"cuda_available": True, "device_name": "RTX 4090", "total_memory_bytes": 100},
    )

    assert report["ok"] is True
    assert report["physical_batch"] == 4
    assert calls == [("probe", 2), ("probe", 4), ("probe", 8)]
    assert windows == [(1, 3), (1, 3), (1, 3)]


def test_default_real_runner_forwards_attempt_contract_once(monkeypatch, tmp_path: Path) -> None:
    config = _trainer_toml(tmp_path)
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(qualification_module, "_coerce_qualification_binding", lambda _value: object())
    monkeypatch.setattr(qualification_module, "_qualification_timeout_seconds", lambda _control, _binding: 60.0)
    monkeypatch.setattr(qualification_module, "_validate_real_config", lambda _config, _root, _binding: {})
    monkeypatch.setattr(qualification_module, "_training_root", lambda _path: tmp_path)

    def isolated_attempt(**kwargs):
        calls.append(kwargs)
        return AttemptResult(
            physical_batch=int(kwargs["physical_batch"]),
            accumulation=int(kwargs["accumulation"]),
            ok=True,
            total_memory_bytes=100,
            peak_reserved_bytes=70,
            update_seconds=(1.0,),
            measured_update_count=1,
            checkpoint_write_reload={"feasible": True, "write": True, "reload": True},
        )

    monkeypatch.setattr(qualification_module, "_run_isolated_real_attempt", isolated_attempt)
    monkeypatch.setattr(
        qualification_module,
        "_build_report",
        lambda **kwargs: {"ok": kwargs.get("failure") is None},
    )

    report = run_qualification(
        config,
        "4090",
        QualificationSpec(
            warmup_optimizer_updates=1,
            measured_optimizer_updates=1,
            physical_batch_candidates=(2,),
        ),
        qualification_binding={},
        qualification_control={},
        device_facts={"cuda_available": True, "device_name": "RTX 4090", "total_memory_bytes": 100},
    )

    assert report == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["physical_batch"] == 2
    assert calls[0]["accumulation"] == 32
    assert calls[0]["phase"] == "probe"
    assert calls[0]["warmup_updates"] == 1
    assert calls[0]["measured_updates"] == 1
    assert calls[0]["config"] == {"meta": {"seed": 3407}}


def test_invalid_profile_and_no_cuda_publish_failed_report(tmp_path: Path) -> None:
    config = _trainer_toml(tmp_path)
    output = tmp_path / "qualification.json"
    report = execute_qualification(
        config,
        "4090",
        QualificationSpec(measured_optimizer_updates=1),
        output,
        device_facts={"cuda_available": False},
    )

    assert report["ok"] is False
    assert report["gpu_qualification_status"] == "failed"
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == "speakrs-gpu-qualification-v1"


def test_invalid_gpu_profile_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(QualificationError, match="unsupported gpu_profile"):
        run_qualification(
            _trainer_toml(tmp_path),
            "5090",
            QualificationSpec(measured_optimizer_updates=1),
            device_facts={"cuda_available": True, "device_name": "RTX 5090", "total_memory_bytes": 100},
        )


def test_checkpoint_roundtrip_failure_does_not_qualify(tmp_path: Path) -> None:
    config = _trainer_toml(tmp_path)
    spec = QualificationSpec(measured_optimizer_updates=2)

    def runner(**kwargs):
        return {
            "ok": True,
            "total_memory_bytes": 100,
            "peak_reserved_bytes": 70,
            "update_seconds": [1.0, 1.0],
            "checkpoint_write_reload": {"feasible": True, "write": True, "reload": False},
        }

    report = run_qualification(
        config,
        "4090",
        spec,
        attempt_runner=runner,
        device_facts={"cuda_available": True, "device_name": "RTX 4090", "total_memory_bytes": 100},
    )

    assert report["ok"] is False
    assert report["error"]["code"] == "checkpoint_roundtrip_failed"


def test_missing_checkpoint_roundtrip_does_not_qualify(tmp_path: Path) -> None:
    config = _trainer_toml(tmp_path)

    report = run_qualification(
        config,
        "4090",
        QualificationSpec(warmup_optimizer_updates=1, measured_optimizer_updates=1),
        attempt_runner=lambda **_kwargs: {
            "ok": True,
            "total_memory_bytes": 100,
            "peak_reserved_bytes": 70,
            "update_seconds": [1.0],
        },
        device_facts={"cuda_available": True, "device_name": "RTX 4090", "total_memory_bytes": 100},
    )

    assert report["ok"] is False
    assert report["error"]["code"] == "checkpoint_roundtrip_failed"


def test_json_spec_can_override_defaults(tmp_path: Path) -> None:
    path = tmp_path / "qualification.json"
    path.write_text(
        json.dumps(
            {
                "schema": "speakrs-gpu-qualification-v1",
                "warmup_optimizer_updates": 2,
                "measured_optimizer_updates": 4,
                "physical_batch_candidates": [2, 4],
            }
        ),
        encoding="utf-8",
    )

    spec = load_qualification_spec(path)

    assert spec.warmup_optimizer_updates == 2
    assert spec.measured_optimizer_updates == 4
    assert spec.physical_batch_candidates == (2, 4)


def test_isolated_worker_is_terminated_at_paid_run_deadline(monkeypatch) -> None:
    state = {"terminated": False}

    class FakeQueue:
        def close(self) -> None:
            pass

    class FakeProcess:
        exitcode = None

        def __init__(self) -> None:
            self.alive = True

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            state["terminated"] = True
            self.alive = False
            self.exitcode = -15

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, *, target, args):
            del target, args
            return FakeProcess()

    monkeypatch.setattr("recipes.speakrs.large.qualification.mp.get_context", lambda _method: FakeContext())

    result = _run_isolated_real_attempt(physical_batch=2, accumulation=32, timeout_seconds=0.01)

    assert state["terminated"] is True
    assert result.error_code == "qualification_deadline_exceeded"
