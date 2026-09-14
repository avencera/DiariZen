"""Standalone, digest-bound evaluator for one published DiariZen snapshot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit

from diarizen.validation_metrics import (
    ValidationMetrics,
    finalize_validation_metrics,
    update_validation_batch_metrics,
)


if TYPE_CHECKING:
    from .preflight import HostInventory


try:  # allow both ``python -m ...`` and direct execution from the repository
    from .contracts import (
        PublishedSnapshot,
        Sha256Digest,
        ValidationContractError,
        ValidationResult,
    )
except ImportError:  # pragma: no cover - only used by direct script execution
    from recipes.speakrs.large.validation.contracts import (
        PublishedSnapshot,
        Sha256Digest,
        ValidationContractError,
        ValidationResult,
    )


EVALUATOR_REQUEST_SCHEMA = "diarizen-standalone-evaluator-request-v1"
_LOGGER = logging.getLogger("diarizen.validation.evaluator")


class EvaluatorError(ValueError):
    """A request or one of its local verified inputs is invalid."""


def _local_path(value: object, field: str) -> str:
    """Validate one local filesystem path while preserving its JSON spelling."""

    path = _string(value, field)
    if urlsplit(path).scheme:
        raise EvaluatorError(f"{field} must be a local filesystem path")
    try:
        Path(path)
    except (TypeError, ValueError) as error:
        raise EvaluatorError(f"{field} must be a local filesystem path") from error
    return path


def _exact_fields(value: Mapping[str, object], required: frozenset[str], context: str) -> None:
    actual = frozenset(value)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        raise EvaluatorError(f"{context} fields are not exact: missing={sorted(missing)}, extra={sorted(extra)}")


def _string(value: object, field: str, maximum: int = 4_096) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise EvaluatorError(f"{field} must be a non-empty string of at most {maximum} UTF-8 bytes")
    if any(not character.isprintable() for character in value):
        raise EvaluatorError(f"{field} cannot contain control characters")
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluatorError(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _digest(value: object, field: str) -> Sha256Digest:
    try:
        return Sha256Digest.parse(value, field)
    except ValidationContractError as error:
        raise EvaluatorError(str(error)) from error


@dataclass(frozen=True)
class LocalModelArtifact:
    """The exact local model file authorized by a published snapshot."""

    path: str
    digest: Sha256Digest
    length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _local_path(self.path, "model path"))
        _integer(self.length, "model length", minimum=1)

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"path": self.path, "digest": self.digest.value, "length": self.length}

    @classmethod
    def from_dict(cls, value: object) -> LocalModelArtifact:
        """Parse one exact model-file declaration."""

        if not isinstance(value, Mapping):
            raise EvaluatorError("model artifact must be an object")
        _exact_fields(value, frozenset({"path", "digest", "length"}), "model artifact")
        return cls(
            path=_string(value["path"], "model path"),
            digest=_digest(value["digest"], "model digest"),
            length=_integer(value["length"], "model length", minimum=1),
        )


@dataclass(frozen=True)
class LocalDevBundle:
    """The frozen development manifest and the root containing its payloads."""

    manifest_path: str
    root: str
    digest: Sha256Digest

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", _local_path(self.manifest_path, "development manifest path"))
        object.__setattr__(self, "root", _local_path(self.root, "development root"))

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"manifest_path": self.manifest_path, "root": self.root, "digest": self.digest.value}

    @classmethod
    def from_dict(cls, value: object) -> LocalDevBundle:
        """Parse one exact frozen-development declaration."""

        if not isinstance(value, Mapping):
            raise EvaluatorError("development bundle must be an object")
        _exact_fields(value, frozenset({"manifest_path", "root", "digest"}), "development bundle")
        return cls(
            manifest_path=_string(value["manifest_path"], "development manifest path"),
            root=_string(value["root"], "development root"),
            digest=_digest(value["digest"], "development bundle digest"),
        )


@dataclass(frozen=True)
class LocalTrainerConfiguration:
    """The exact trainer configuration used to construct the architecture."""

    path: str
    digest: Sha256Digest

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _local_path(self.path, "trainer configuration path"))

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {"path": self.path, "digest": self.digest.value}

    @classmethod
    def from_dict(cls, value: object) -> LocalTrainerConfiguration:
        """Parse one exact trainer-configuration declaration."""

        if not isinstance(value, Mapping):
            raise EvaluatorError("trainer configuration must be an object")
        _exact_fields(value, frozenset({"path", "digest"}), "trainer configuration")
        return cls(
            path=_string(value["path"], "trainer configuration path"),
            digest=_digest(value["digest"], "trainer configuration digest"),
        )


@dataclass(frozen=True)
class EvaluatorRequest:
    """One strict request for evaluating one immutable published snapshot."""

    snapshot: PublishedSnapshot
    model: LocalModelArtifact
    dev_bundle: LocalDevBundle
    trainer_configuration: LocalTrainerConfiguration
    evaluator_image_identity: str
    evaluator_implementation_digest: Sha256Digest

    def __post_init__(self) -> None:
        if self.model.digest != self.snapshot.model_digest or self.model.length != self.snapshot.model_length:
            raise EvaluatorError("model artifact does not match the published snapshot")
        if self.dev_bundle.digest != self.snapshot.dev_bundle_digest:
            raise EvaluatorError("development bundle does not match the published snapshot")
        if self.trainer_configuration.digest != self.snapshot.trainer_configuration_digest:
            raise EvaluatorError("trainer configuration does not match the published snapshot")
        if self.evaluator_image_identity != self.snapshot.evaluator_image_identity:
            raise EvaluatorError("evaluator image identity does not match the published snapshot")
        if self.evaluator_implementation_digest != self.snapshot.evaluator_implementation_digest:
            raise EvaluatorError("evaluator implementation identity does not match the published snapshot")

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""

        return {
            "schema": EVALUATOR_REQUEST_SCHEMA,
            "snapshot": self.snapshot.to_dict(),
            "model": self.model.to_dict(),
            "dev_bundle": self.dev_bundle.to_dict(),
            "trainer_configuration": self.trainer_configuration.to_dict(),
            "evaluator_image_identity": self.evaluator_image_identity,
            "evaluator_implementation_digest": self.evaluator_implementation_digest.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> EvaluatorRequest:
        """Parse one strict request and validate all copied snapshot identities."""

        if not isinstance(value, Mapping):
            raise EvaluatorError("evaluator request must be an object")
        _exact_fields(
            value,
            frozenset(
                {
                    "schema",
                    "snapshot",
                    "model",
                    "dev_bundle",
                    "trainer_configuration",
                    "evaluator_image_identity",
                    "evaluator_implementation_digest",
                }
            ),
            "evaluator request",
        )
        if value["schema"] != EVALUATOR_REQUEST_SCHEMA:
            raise EvaluatorError("evaluator request schema is not supported")
        try:
            snapshot = PublishedSnapshot.from_dict(value["snapshot"])
        except ValidationContractError as error:
            raise EvaluatorError(str(error)) from error
        return cls(
            snapshot=snapshot,
            model=LocalModelArtifact.from_dict(value["model"]),
            dev_bundle=LocalDevBundle.from_dict(value["dev_bundle"]),
            trainer_configuration=LocalTrainerConfiguration.from_dict(value["trainer_configuration"]),
            evaluator_image_identity=_string(value["evaluator_image_identity"], "evaluator image identity", 512),
            evaluator_implementation_digest=_digest(
                value["evaluator_implementation_digest"], "evaluator implementation digest"
            ),
        )


class EvaluationRunner(Protocol):
    """An injectable runner for tests and alternate execution environments."""

    def __call__(self, request: EvaluatorRequest) -> ValidationMetrics:
        """Evaluate the verified request and return typed metric values."""


class ModelFactory(Protocol):
    """A factory that constructs the configured architecture without its initializer."""

    def __call__(self, configuration: Mapping[str, object]) -> object:
        """Construct one CPU model architecture."""


@dataclass(frozen=True)
class _VerifiedInputs:
    """Resolved local inputs after digest and shape checks."""

    model_path: Path
    dev_manifest_path: Path
    dev_root: Path
    configuration_path: Path
    dev_artifacts: tuple[tuple[str, int, Sha256Digest], ...]


def _reject_json_constant(value: str) -> None:
    raise EvaluatorError(f"JSON constant is not supported: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluatorError("JSON object contains duplicate fields")
        result[key] = value
    return result


def _parse_json(text: str, context: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (EvaluatorError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise EvaluatorError(f"{context} is not valid strict JSON") from error


def parse_request_json(text: str) -> EvaluatorRequest:
    """Parse one strict JSON request document."""

    return EvaluatorRequest.from_dict(_parse_json(text, "evaluator request"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_file(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise EvaluatorError(f"{field} cannot be a symbolic link")
    if not path.is_file():
        raise EvaluatorError(f"{field} is missing or is not a regular file")
    return path.resolve()


def _local_directory(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise EvaluatorError(f"{field} cannot be a symbolic link")
    if not path.is_dir():
        raise EvaluatorError(f"{field} is missing or is not a directory")
    return path.resolve()


def _verify_file(path: Path, expected_digest: Sha256Digest, expected_length: int, field: str) -> None:
    try:
        actual_length = path.stat().st_size
        actual_digest = _sha256_file(path)
    except OSError as error:
        raise EvaluatorError(f"cannot read {field}") from error
    if actual_length != expected_length or actual_digest != expected_digest.value:
        raise EvaluatorError(f"{field} changed from its verified identity")


def _verify_dev_bundle(
    request: EvaluatorRequest,
) -> tuple[Path, Path, tuple[tuple[str, int, Sha256Digest], ...]]:
    """Load and verify the location-neutral v2 bundle before framework imports."""

    root = _local_directory(request.dev_bundle.root, "development root")
    manifest = _local_file(request.dev_bundle.manifest_path, "development manifest")
    if manifest.parent != root:
        raise EvaluatorError("development manifest is not directly inside its verified root")

    try:
        bundle_module = importlib.import_module("recipes.speakrs.large.dev_bundle")
        bundle = bundle_module.load_dev_bundle(manifest, expected_recordings=44)
        identity = Sha256Digest(bundle.identity_sha256())
        if identity != request.dev_bundle.digest:
            raise EvaluatorError("development bundle identity does not match the published snapshot")
        bundle_module.verify_dev_bundle(root, manifest=bundle, expected_recordings=44)
        _ensure_derived_views(root, required=False)
    except EvaluatorError:
        raise
    except Exception as error:
        raise EvaluatorError("development bundle failed canonical verification") from error

    artifacts = tuple((file.path, file.byte_length, file.sha256) for file in bundle.files)
    return manifest, root, artifacts


def _ensure_derived_views(root: Path, *, required: bool) -> None:
    """Require complete regular-file path views when real evaluation will use them."""

    paths = tuple(root / name for name in ("bundle.paths.json", "wav.scp", "all.rttm", "all.uem"))
    present = tuple(path for path in paths if path.exists() or path.is_symlink())
    if not present:
        if required:
            raise EvaluatorError("verified development derived views are missing")
        return
    if len(present) != len(paths) or any(path.is_symlink() or not path.is_file() for path in paths):
        raise EvaluatorError("verified development derived views are incomplete")
    if not (root / "bundle.paths.json").is_file():
        raise EvaluatorError("verified development derived views have no identity manifest")


def _preflight(request: EvaluatorRequest) -> _VerifiedInputs:
    """Verify every local input before importing Torch or constructing a model."""

    model_path = _local_file(request.model.path, "model file")
    _verify_file(model_path, request.model.digest, request.model.length, "model file")

    configuration_path = _local_file(request.trainer_configuration.path, "trainer configuration")
    _verify_file(
        configuration_path,
        request.trainer_configuration.digest,
        configuration_path.stat().st_size,
        "trainer configuration",
    )

    dev_manifest_path, dev_root, dev_artifacts = _verify_dev_bundle(request)
    return _VerifiedInputs(model_path, dev_manifest_path, dev_root, configuration_path, dev_artifacts)


def _run_host_preflight(verified: _VerifiedInputs, inventory: HostInventory | None) -> None:
    """Qualify the host and recheck every sealed bundle payload before CUDA use."""

    try:
        preflight = importlib.import_module("recipes.speakrs.large.validation.preflight")
        expected = preflight.ExpectedArtifactSet(
            preflight.ExpectedArtifact(path, length, digest) for path, length, digest in verified.dev_artifacts
        )
        preflight.run_evaluator_preflight(
            root=verified.dev_root,
            expected_artifacts=expected,
            allocation_callback=lambda: None,
            inventory=inventory,
            profile=preflight.DEFAULT_EVALUATOR_PROFILE,
        )
    except EvaluatorError:
        raise
    except Exception as error:
        raise EvaluatorError("evaluator host or development artifact preflight failed") from error


def _require_cuda() -> object:
    """Import Torch only after preflight and reject CPU fallback."""

    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
    except Exception as error:
        raise EvaluatorError("CUDA is required for production evaluation") from error
    if not available:
        raise EvaluatorError("CUDA is required for production evaluation")
    return torch


def _load_configuration(path: Path) -> Mapping[str, object]:
    """Load the already verified TOML trainer configuration."""

    try:
        toml = importlib.import_module("toml")
        configuration = toml.load(path.as_posix())
    except Exception as error:  # pragma: no cover - dependency and malformed config are runtime failures
        raise EvaluatorError("trainer configuration could not be loaded") from error
    if not isinstance(configuration, Mapping):
        raise EvaluatorError("trainer configuration must be an object")
    return configuration


def _wavlm_architecture_source(args: Mapping[str, object]) -> str:
    """Choose a built-in WavLM config from the serialized architecture dimensions."""

    layer_count = args.get("wavlm_layer_num", 13)
    feature_dimension = args.get("wavlm_feat_dim", 768)
    if layer_count == 13 and feature_dimension == 768:
        return "wavlm_base"
    if layer_count == 25 and feature_dimension == 1024:
        return "wavlm_large"
    raise EvaluatorError("WavLM initializer is not replaceable for this architecture")


def architecture_only_model_factory(configuration: Mapping[str, object]) -> object:
    """Construct the configured model without loading its training initializer.

    WavLM model recipes encode the initializer path in ``wavlm_src``. The
    published state dict already contains the complete model, so only the
    matching built-in architecture configuration is used during construction.
    """

    try:
        model_spec = configuration["model"]
        if not isinstance(model_spec, Mapping):
            raise EvaluatorError("model configuration must be an object")
        model_path = _string(model_spec["path"], "model configuration path")
        raw_args = model_spec.get("args", {})
        if not isinstance(raw_args, Mapping):
            raise EvaluatorError("model configuration arguments must be an object")
        model_args = copy.deepcopy(dict(raw_args))
        if "wavlm_src" in model_args:
            model_args["wavlm_src"] = _wavlm_architecture_source(model_args)
        module_name, symbol_name = model_path.rsplit(".", 1)
        model_type = getattr(importlib.import_module(module_name), symbol_name)
        return model_type(**model_args)
    except EvaluatorError:
        raise
    except (AttributeError, ImportError, TypeError, ValueError) as error:
        raise EvaluatorError("configured model architecture could not be constructed") from error


def _load_state_dict(path: Path) -> Mapping[str, object]:
    try:
        torch = importlib.import_module("torch")
        try:
            state = torch.load(path.as_posix(), map_location="cpu", weights_only=True)
        except TypeError:  # older Torch versions do not have ``weights_only``
            state = torch.load(path.as_posix(), map_location="cpu")
    except Exception as error:
        raise EvaluatorError("model state could not be loaded") from error
    if not isinstance(state, Mapping) or not state:
        raise EvaluatorError("model file must contain a non-empty state dict")
    if any(not isinstance(key, str) for key in state):
        raise EvaluatorError("model state keys must be strings")
    return state


def _strict_load_state(model: object, state: Mapping[str, object]) -> None:
    load_state_dict = getattr(model, "load_state_dict", None)
    if not callable(load_state_dict):
        raise EvaluatorError("configured model does not support state loading")
    state_dict = getattr(model, "state_dict", None)
    if callable(state_dict):
        try:
            expected = state_dict()
        except Exception as error:
            raise EvaluatorError("configured model state could not be inspected") from error
        if isinstance(expected, Mapping):
            missing = tuple(sorted(set(expected) - set(state)))
            unexpected = tuple(sorted(set(state) - set(expected)))
            if missing or unexpected:
                raise EvaluatorError(
                    "published model state is incomplete or has unexpected keys"
                    f": missing={list(missing)}, unexpected={list(unexpected)}"
                )
    try:
        incompatible = load_state_dict(state, strict=True)
    except Exception as error:
        raise EvaluatorError("published model state does not match the configured architecture") from error
    missing = tuple(getattr(incompatible, "missing_keys", ()))
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        raise EvaluatorError(
            "published model state is incomplete or has unexpected keys"
            f": missing={list(missing)}, unexpected={list(unexpected)}"
        )


def _resolve_symbol(path: object, label: str) -> object:
    if not isinstance(path, str) or "." not in path:
        raise EvaluatorError(f"{label} path is invalid")
    module_name, symbol_name = path.rsplit(".", 1)
    if module_name == "dataset":
        module_name = "recipes.diar_ssl.dataset"
    try:
        return getattr(importlib.import_module(module_name), symbol_name)
    except (AttributeError, ImportError) as error:
        raise EvaluatorError(f"{label} could not be imported") from error


def _run_model_evaluation(
    model: object,
    configuration: Mapping[str, object],
    dev_root: Path,
) -> ValidationMetrics:
    """Run the same model, loss, permutation, and frozen-dev step as training."""

    try:
        torch = importlib.import_module("torch")
        from functools import partial

        from torch.utils.data import DataLoader
    except ImportError as error:  # pragma: no cover - evaluator image dependency failure
        raise EvaluatorError("evaluation dependencies could not be imported") from error

    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info
    validate_spec = configuration.get("validate_dataset")
    if not isinstance(validate_spec, Mapping):
        raise EvaluatorError("validate_dataset configuration is missing")
    dataset_args = validate_spec.get("args", {})
    dataloader_args = validate_spec.get("dataloader", {})
    if not isinstance(dataset_args, Mapping) or not isinstance(dataloader_args, Mapping):
        raise EvaluatorError("validate_dataset configuration is invalid")

    _ensure_derived_views(dev_root, required=True)

    dataset_args = dict(dataset_args)
    dataset_args.update(
        {
            "scp_file": (dev_root / "wav.scp").as_posix(),
            "rttm_file": (dev_root / "all.rttm").as_posix(),
            "uem_file": (dev_root / "all.uem").as_posix(),
            "model_num_frames": model_num_frames,
            "model_rf_duration": model_rf_duration,
            "model_rf_step": model_rf_step,
        }
    )
    dataset_type = _resolve_symbol(validate_spec.get("path"), "validation dataset")
    dataset = dataset_type(**dataset_args)
    collate_fn = partial(
        importlib.import_module("recipes.diar_ssl.dataset")._collate_fn,
        max_speakers_per_chunk=configuration["model"]["args"]["max_speakers_per_chunk"],
    )
    loader = DataLoader(dataset=dataset, collate_fn=collate_fn, shuffle=False, **dict(dataloader_args))
    _require_cuda()
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    try:
        metric = model.validation_metric
    except AttributeError as error:
        raise EvaluatorError("configured model has no validation metric") from error
    if metric is None:
        raise EvaluatorError("configured model has no validation metric")
    losses: list[object] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            try:
                xs = batch["xs"].to(device)
                target = batch["ts"].to(device)
            except (AttributeError, KeyError, TypeError) as error:
                raise EvaluatorError(f"validation batch {batch_index} is malformed") from error
            loss = update_validation_batch_metrics(
                model=model,
                powerset=model.powerset,
                metric=metric,
                features=xs,
                target=target,
            )
            losses.append(loss.detach().float())
    return finalize_validation_metrics(losses, metric)


def build_validation_result(
    request: EvaluatorRequest,
    metrics: ValidationMetrics,
    started_at: datetime,
    completed_at: datetime,
) -> ValidationResult:
    """Build and re-check a result bound to every request snapshot identity."""

    snapshot = request.snapshot
    try:
        result = ValidationResult(
            snapshot_id=snapshot.snapshot_id,
            campaign_id=snapshot.campaign_id,
            training_launch_id=snapshot.training_launch_id,
            slot_id=snapshot.slot_id,
            updates=snapshot.updates,
            model_digest=snapshot.model_digest,
            model_length=snapshot.model_length,
            trainer_configuration_digest=snapshot.trainer_configuration_digest,
            dev_bundle_digest=snapshot.dev_bundle_digest,
            evaluator_image_identity=snapshot.evaluator_image_identity,
            evaluator_implementation_digest=snapshot.evaluator_implementation_digest,
            recovery_generation_id=snapshot.recovery_generation_id,
            generation_sequence=snapshot.generation_sequence,
            progress_digest=snapshot.progress_digest,
            publication_time=snapshot.publication_time,
            loss=metrics.loss,
            der=metrics.der,
            false_alarm=metrics.false_alarm,
            miss=metrics.miss,
            confusion=metrics.confusion,
            started_at=started_at.astimezone(timezone.utc),
            completed_at=completed_at.astimezone(timezone.utc),
        )
    except ValidationContractError as error:
        raise EvaluatorError(str(error)) from error
    if not result.matches_snapshot(snapshot):
        raise EvaluatorError("evaluation result does not match the request snapshot")
    if result.evaluator_image_identity != request.evaluator_image_identity:
        raise EvaluatorError("evaluation result image identity does not match the request")
    if result.evaluator_implementation_digest != request.evaluator_implementation_digest:
        raise EvaluatorError("evaluation result implementation identity does not match the request")
    return result


def evaluate_request(
    request: EvaluatorRequest,
    *,
    runner: EvaluationRunner | None = None,
    model_factory: ModelFactory | None = None,
    clock: Callable[[], datetime] | None = None,
    host_inventory: HostInventory | None = None,
) -> ValidationResult:
    """Evaluate one request after local identity preflight."""

    started_at = (clock or (lambda: datetime.now(timezone.utc)))().astimezone(timezone.utc)
    verified = _preflight(request)
    if runner is not None:
        metrics = runner(request)
        if not isinstance(metrics, ValidationMetrics):
            raise EvaluatorError("evaluation runner must return typed validation metrics")
    else:
        _run_host_preflight(verified, host_inventory)
        configuration = _load_configuration(verified.configuration_path)
        _verify_file(
            verified.configuration_path,
            request.trainer_configuration.digest,
            verified.configuration_path.stat().st_size,
            "trainer configuration",
        )
        state = _load_state_dict(verified.model_path)
        _verify_file(verified.model_path, request.model.digest, request.model.length, "model file")
        factory = model_factory or architecture_only_model_factory
        model = factory(configuration)
        _strict_load_state(model, state)
        _verify_file(verified.model_path, request.model.digest, request.model.length, "model file")
        _ensure_derived_views(verified.dev_root, required=True)
        _require_cuda()
        metrics = _run_model_evaluation(model, configuration, verified.dev_root)
    completed_at = (clock or (lambda: datetime.now(timezone.utc)))().astimezone(timezone.utc)
    return build_validation_result(request, metrics, started_at, completed_at)


def result_json(result: ValidationResult) -> str:
    """Encode one compact result document without local paths or environment data."""

    return json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _request_text(request_file: Path | None) -> str:
    if request_file is None:
        return sys.stdin.read()
    return request_file.read_text(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the one-request evaluator CLI."""

    parser = argparse.ArgumentParser(description="Evaluate one published DiariZen validation snapshot")
    parser.add_argument("--request-file", type=Path, help="read the strict request JSON from this file")
    args = parser.parse_args(argv)
    try:
        request = parse_request_json(_request_text(args.request_file))
        _LOGGER.info("Starting validation for snapshot %s", request.snapshot.snapshot_id.value)
        result = evaluate_request(request)
        sys.stdout.write(result_json(result) + "\n")
        sys.stdout.flush()
        _LOGGER.info("Validation completed for snapshot %s", request.snapshot.snapshot_id.value)
        return 0
    except Exception as error:
        print(f"validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())


__all__ = [
    "EVALUATOR_REQUEST_SCHEMA",
    "EvaluationRunner",
    "EvaluatorError",
    "EvaluatorRequest",
    "LocalDevBundle",
    "LocalModelArtifact",
    "LocalTrainerConfiguration",
    "ModelFactory",
    "architecture_only_model_factory",
    "build_validation_result",
    "evaluate_request",
    "main",
    "parse_request_json",
    "result_json",
]
