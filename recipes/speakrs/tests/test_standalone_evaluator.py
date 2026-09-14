"""Tests for the isolated, digest-bound validation evaluator."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from diarizen.validation_metrics import ValidationMetrics
from recipes.speakrs.large.dev_bundle import DevBundleFileRole, parse_dev_bundle_manifest, verify_dev_bundle
from recipes.speakrs.large.validation.contracts import PublishedSnapshot, Sha256Digest
from recipes.speakrs.large.validation.evaluator import (
    EVALUATOR_REQUEST_SCHEMA,
    EvaluatorError,
    EvaluatorRequest,
    LocalDevBundle,
    LocalModelArtifact,
    LocalTrainerConfiguration,
    evaluate_request,
    parse_request_json,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(path: Path) -> Sha256Digest:
    return Sha256Digest(_sha256(path))


def _make_request(tmp_path: Path, *, model_bytes: bytes = b"model-state") -> EvaluatorRequest:
    dev_root = tmp_path / "dev"
    dev_root.mkdir()
    recordings = []
    sources = ("AMI",) * 15 + ("AliMeeting",) * 15 + ("AISHELL-5",) * 14
    for index, source in enumerate(sources):
        recording_id = f"dev-{index:03d}"
        logical_id = f"{source}:{recording_id}"
        files = []
        for role in DevBundleFileRole:
            content = f"{logical_id}:{role.value}".encode()
            if role is DevBundleFileRole.AUDIO:
                relative = f"audio/{source}/{index:03d}-{role.value}"
            else:
                relative = f"annotations/{source}/{index:03d}-{role.value}"
            path = dev_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            files.append(
                {
                    "logical_id": f"{logical_id}:{role.value}",
                    "path": relative,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "byte_length": len(content),
                    "role": role.value,
                }
            )
        recordings.append(
            {
                "logical_id": logical_id,
                "recording_id": recording_id,
                "source": source,
                "files": files,
            }
        )
    bundle = parse_dev_bundle_manifest(
        {"schema": "speakrs-frozen-dev-bundle-v2", "recordings": recordings},
        expected_recordings=44,
    )
    (dev_root / "bundle.json").write_bytes(bundle.canonical_bytes())
    verify_dev_bundle(dev_root, manifest=bundle, expected_recordings=44)

    model_path = tmp_path / "pytorch_model.bin"
    model_path.write_bytes(model_bytes)
    configuration_path = tmp_path / "config.toml"
    configuration_path.write_text("[model]\npath = 'example.Model'\n", encoding="utf-8")

    snapshot = PublishedSnapshot(
        campaign_id=Sha256Digest("a" * 64),
        training_launch_id="launch-v1",
        slot_id=Sha256Digest("b" * 64),
        updates=0,
        model_digest=_digest(model_path),
        model_length=model_path.stat().st_size,
        trainer_configuration_digest=_digest(configuration_path),
        dev_bundle_digest=Sha256Digest(bundle.identity_sha256()),
        evaluator_image_identity="registry.example/evaluator@sha256:" + "c" * 64,
        evaluator_implementation_digest=Sha256Digest("d" * 64),
        recovery_generation_id=Sha256Digest("e" * 64),
        generation_sequence=1,
        progress_digest=Sha256Digest("f" * 64),
        publication_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return EvaluatorRequest(
        snapshot=snapshot,
        model=LocalModelArtifact(str(model_path), snapshot.model_digest, snapshot.model_length),
        dev_bundle=LocalDevBundle(str(dev_root / "bundle.json"), str(dev_root), snapshot.dev_bundle_digest),
        trainer_configuration=LocalTrainerConfiguration(
            str(configuration_path), snapshot.trainer_configuration_digest
        ),
        evaluator_image_identity=snapshot.evaluator_image_identity,
        evaluator_implementation_digest=snapshot.evaluator_implementation_digest,
    )


class _Metric:
    def __init__(self) -> None:
        self.reset_calls = 0

    def compute(self):
        return {
            "DiarizationErrorRate": 0.5,
            "DiarizationErrorRate/FalseAlarm": 0.1,
            "DiarizationErrorRate/Miss": 0.2,
            "DiarizationErrorRate/Confusion": 0.2,
        }

    def reset(self) -> None:
        self.reset_calls += 1


def test_request_rejects_unknown_fields(tmp_path: Path) -> None:
    request = _make_request(tmp_path)
    payload = request.to_dict()
    payload["unexpected"] = True

    with pytest.raises(EvaluatorError, match="fields are not exact"):
        EvaluatorRequest.from_dict(payload)


@pytest.mark.parametrize("field", ["model", "dev_bundle", "trainer_configuration", "evaluator_image_identity"])
def test_request_rejects_identity_mismatch(tmp_path: Path, field: str) -> None:
    request = _make_request(tmp_path)
    payload = request.to_dict()
    if field == "model":
        payload[field]["digest"] = "0" * 64
    elif field == "dev_bundle":
        payload[field]["digest"] = "0" * 64
    elif field == "trainer_configuration":
        payload[field]["digest"] = "0" * 64
    else:
        payload[field] = "other-image"

    with pytest.raises(EvaluatorError, match="match|identity"):
        EvaluatorRequest.from_dict(payload)


@pytest.mark.parametrize("rewrite", [b"changed", b"not a torch state dict"])
def test_changed_or_incomplete_model_is_rejected_before_factory(tmp_path: Path, rewrite: bytes) -> None:
    request = _make_request(tmp_path)
    Path(request.model.path).write_bytes(rewrite)
    called = False

    def factory(_configuration):
        nonlocal called
        called = True
        raise AssertionError("factory must not run for failed model preflight")

    with pytest.raises(EvaluatorError, match="changed from its verified identity"):
        evaluate_request(request, model_factory=factory)
    assert not called


def test_bundle_artifact_corruption_is_rejected_before_factory_or_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _make_request(tmp_path)
    audio_path = next(path for path in (Path(request.dev_bundle.root) / "audio").rglob("*") if path.is_file())
    audio_path.write_bytes(audio_path.read_bytes() + b"corruption")
    called = False

    def factory(_configuration):
        nonlocal called
        called = True
        raise AssertionError("factory must not run for failed bundle preflight")

    def cuda_check():
        raise AssertionError("CUDA must not be checked for failed bundle preflight")

    from recipes.speakrs.large.validation import evaluator

    monkeypatch.setattr(evaluator, "_require_cuda", cuda_check)
    with pytest.raises(EvaluatorError, match="development bundle failed canonical verification"):
        evaluate_request(request, model_factory=factory)
    assert not called


def test_bundle_requires_exactly_forty_four_recordings_before_factory(tmp_path: Path) -> None:
    request = _make_request(tmp_path)
    manifest_path = Path(request.dev_bundle.manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["recordings"].pop()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    called = False

    def factory(_configuration):
        nonlocal called
        called = True
        raise AssertionError("factory must not run for an incomplete bundle")

    with pytest.raises(EvaluatorError, match="development bundle failed canonical verification"):
        evaluate_request(request, model_factory=factory)
    assert not called


def test_portable_v2_bundle_is_accepted_by_injected_runner(tmp_path: Path) -> None:
    request = _make_request(tmp_path)
    metrics = ValidationMetrics(1.25, 0.5, 0.1, 0.2, 0.2)

    result = evaluate_request(request, runner=lambda _request: metrics)

    assert result.loss == metrics.loss
    assert result.dev_bundle_digest == request.snapshot.dev_bundle_digest


def test_unqualified_host_is_rejected_before_model_factory_or_cuda(tmp_path: Path, monkeypatch) -> None:
    from recipes.speakrs.large.validation import evaluator
    from recipes.speakrs.large.validation.preflight import (
        CudaVersion,
        GpuInventory,
        GpuName,
        HostInventory,
    )

    request = _make_request(tmp_path)
    inventory = HostInventory(
        (GpuInventory(GpuName("wrong GPU"), 16 * 1024**3, CudaVersion.parse("12.8")),),
        32 * 1024**3,
        4,
        32 * 1024**3,
    )
    called = False

    def factory(_configuration):
        nonlocal called
        called = True
        raise AssertionError("factory must not run on an unqualified host")

    def cuda_check():
        raise AssertionError("CUDA must not be checked on an unqualified host")

    monkeypatch.setattr(evaluator, "_require_cuda", cuda_check)
    with pytest.raises(EvaluatorError, match="host or development artifact preflight"):
        evaluate_request(request, model_factory=factory, host_inventory=inventory)
    assert not called


def test_production_path_has_no_cpu_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from recipes.speakrs.large.validation import evaluator
    from recipes.speakrs.large.validation.preflight import (
        CudaVersion,
        GpuInventory,
        GpuName,
        HostInventory,
    )

    request = _make_request(tmp_path)
    inventory = HostInventory(
        (GpuInventory(GpuName("RTX 5060 Ti"), 16 * 1024**3, CudaVersion.parse("12.8")),),
        32 * 1024**3,
        4,
        32 * 1024**3,
    )
    moved = False

    class Model:
        def state_dict(self):
            return {"expected": object()}

        def load_state_dict(self, _state, strict=True):
            return None

        def to(self, _device):
            nonlocal moved
            moved = True
            return self

    monkeypatch.setattr(evaluator, "_load_state_dict", lambda _path: {"expected": object()})
    monkeypatch.setattr(evaluator, "_ensure_derived_views", lambda _root, *, required: None)
    monkeypatch.setattr(
        evaluator,
        "_require_cuda",
        lambda: (_ for _ in ()).throw(EvaluatorError("CUDA is required for production evaluation")),
    )
    with pytest.raises(EvaluatorError, match="CUDA is required"):
        evaluate_request(request, model_factory=lambda _configuration: Model(), host_inventory=inventory)
    assert not moved


def test_standalone_evaluator_matches_inline_trainer_metrics_with_tolerance(tmp_path: Path) -> None:
    import torch

    from diarizen.validation_metrics import update_validation_batch_metrics
    from recipes.diar_ssl.trainer_dual_opt import Trainer

    class _Powerset:
        def to_multilabel(self, y_pred):
            return y_pred

        def to_powerset(self, target):
            return target

    class _Metric:
        def __init__(self) -> None:
            self.false_alarm = 0.0
            self.missed_detection = 0.0
            self.speech_total = 0.0

        def update(self, predictions, target) -> None:
            predictions = predictions >= 0.5
            target = target.bool()
            self.false_alarm += float((predictions & ~target).sum())
            self.missed_detection += float((~predictions & target).sum())
            self.speech_total += float(target.sum())

        def compute(self):
            denominator = self.speech_total + 1e-8
            der = (self.false_alarm + self.missed_detection) / denominator
            return {
                "DiarizationErrorRate": torch.tensor(der),
                "DiarizationErrorRate/FalseAlarm": torch.tensor(self.false_alarm / denominator),
                "DiarizationErrorRate/Miss": torch.tensor(self.missed_detection / denominator),
                "DiarizationErrorRate/Confusion": torch.tensor(0.0),
            }

        def reset(self) -> None:
            self.false_alarm = 0.0
            self.missed_detection = 0.0
            self.speech_total = 0.0

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.powerset = _Powerset()
            self.validation_metric = _Metric()

        def forward(self, xs):
            return torch.nn.functional.log_softmax(xs, dim=-1)

    frozen_xs = torch.tensor([[[2.0, -1.0], [-1.0, 2.0]]], dtype=torch.float32)
    frozen_target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    batch = {"xs": frozen_xs, "ts": frozen_target}
    request = _make_request(tmp_path)
    standalone_model = _Model()

    def runner(_request):
        loss = update_validation_batch_metrics(
            model=standalone_model,
            powerset=standalone_model.powerset,
            metric=standalone_model.validation_metric,
            features=batch["xs"],
            target=batch["ts"],
        )
        from diarizen.validation_metrics import finalize_validation_metrics

        return finalize_validation_metrics([loss], standalone_model.validation_metric)

    standalone = evaluate_request(request, runner=runner)

    inline_model = _Model()
    captured: dict[str, float] = {}

    class _InlineTrainer(Trainer):
        def __init__(self) -> None:
            self.model = inline_model
            self.unwrap_model = inline_model
            self.accelerator = type("A", (), {"is_local_main_process": True})()
            self.state = type("S", (), {"epochs_trained": 0})()
            self.writer = type("W", (), {"add_scalar": staticmethod(lambda *_args, **_kwargs: None)})()
            self._write_validation_metrics = captured.update

    trainer = _InlineTrainer()
    step = trainer.validation_step(batch, 0)
    score = trainer.validation_epoch_end([step])

    assert standalone.loss == pytest.approx(score, abs=1e-12)
    assert standalone.loss == pytest.approx(captured["Loss"], abs=1e-12)
    assert standalone.der == pytest.approx(captured["DER"], abs=1e-12)
    assert standalone.false_alarm == pytest.approx(captured["FA"], abs=1e-12)
    assert standalone.miss == pytest.approx(captured["Miss"], abs=1e-12)
    assert standalone.confusion == pytest.approx(captured["Confusion"], abs=1e-12)


def test_strict_state_load_rejects_missing_key_before_device_move(tmp_path: Path) -> None:
    import torch

    from recipes.speakrs.large.validation.evaluator import _load_state_dict, _strict_load_state

    output = io.BytesIO()
    torch.save({"unexpected": torch.tensor(1)}, output)
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(output.getvalue())

    class Model:
        def state_dict(self):
            return {"expected": object()}

        def load_state_dict(self, _state, strict=True):
            raise AssertionError("strict key comparison should fail first")

    with pytest.raises(EvaluatorError, match="incomplete or has unexpected keys"):
        _strict_load_state(Model(), _load_state_dict(model_path))


def test_standalone_cli_emits_only_one_json_document(tmp_path: Path, monkeypatch, capsys) -> None:
    request = _make_request(tmp_path)
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(request.to_dict()), encoding="utf-8")
    expected = request.snapshot.to_dict()

    # Verify strict request parsing without importing or running Torch in this process.
    assert parse_request_json(request_file.read_text(encoding="utf-8")).to_dict() == request.to_dict()
    assert EVALUATOR_REQUEST_SCHEMA in request_file.read_text(encoding="utf-8")

    from recipes.speakrs.large.validation import evaluator

    monkeypatch.setattr(
        evaluator,
        "evaluate_request",
        lambda value: evaluator.build_validation_result(
            value,
            evaluator.ValidationMetrics(1.0, 0.5, 0.1, 0.2, 0.2),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 1, tzinfo=timezone.utc),
        ),
    )
    assert evaluator.main(["--request-file", str(request_file)]) == 0
    captured = capsys.readouterr()
    decoded = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert decoded["snapshot_id"] == expected["snapshot_id"]
    assert captured.err == ""


def test_architecture_factory_does_not_use_initializer_path() -> None:
    # The production factory is tested through its dimension-only source choice.
    from recipes.speakrs.large.validation.evaluator import _wavlm_architecture_source

    assert _wavlm_architecture_source({"wavlm_layer_num": 13, "wavlm_feat_dim": 768}) == "wavlm_base"
