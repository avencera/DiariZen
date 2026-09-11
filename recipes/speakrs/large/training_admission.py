"""Typed admission and launch lock for the authorized four-source stage."""

from __future__ import annotations

import math
import posixpath
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import toml

from .errors import ContractError, RuntimeGateError
from .hashing import require_mapping, sha256_file, sha256_json
from .jsonio import read_json, write_json


AUTHORIZATION_SCHEMA = "speakrs-training-authorization-v1"
LAUNCH_SCHEMA = "speakrs-four-source-launch-v1"
CAPACITY_ACCEPTANCE = "accept-qualified-diagnostic-capacity"
DESTROY_REQUEST_RESERVE_SECONDS = 60


def _text(payload: Mapping[str, Any], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label}.{field} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, Any], field: str, label: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label}.{field} must be a positive integer")
    return value


def _number(payload: Mapping[str, Any], field: str, label: str, *, allow_zero: bool = False) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label}.{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not allow_zero and number == 0:
        raise ContractError(f"{label}.{field} must be finite and positive")
    return number


def _timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ContractError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _digest(payload: Mapping[str, Any], field: str, label: str) -> str:
    value = _text(payload, field, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{label}.{field} must be a SHA-256 digest")
    return value


def _oci_digest(payload: Mapping[str, Any], field: str, label: str) -> str:
    value = _text(payload, field, label)
    if not value.startswith("sha256:"):
        raise ContractError(f"{label}.{field} must be an OCI SHA-256 digest")
    _digest({"digest": value.removeprefix("sha256:")}, "digest", label)
    return value


@dataclass(frozen=True)
class TrainingAuthorization:
    """One user-approved bounded training stage."""

    authorization_id: str
    user_authorization_ref: str
    approved_at: str
    spend_ceiling_usd: float
    max_cycles: int
    updates_per_cycle: int
    max_updates: int
    capacity_acceptance: str
    qualification_sha256: str
    qualification_binding_sha256: str
    release_sha256: str
    frozen_dev_sha256: str

    @classmethod
    def parse(cls, payload: Any) -> "TrainingAuthorization":
        """Parse an exact stage authorization and exclude incompatible values."""

        data = require_mapping(payload, "training authorization")
        allowed = {
            "schema",
            "authorization_id",
            "user_authorization_ref",
            "approved_at",
            "spend_ceiling_usd",
            "max_cycles",
            "updates_per_cycle",
            "max_updates",
            "capacity_acceptance",
            "qualification_sha256",
            "qualification_binding_sha256",
            "release_sha256",
            "frozen_dev_sha256",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ContractError("training authorization has unknown keys", {"unknown": unknown})
        if data.get("schema") != AUTHORIZATION_SCHEMA:
            raise ContractError(f"training authorization schema must be {AUTHORIZATION_SCHEMA}")
        cycles = _integer(data, "max_cycles", "training authorization")
        updates_per_cycle = _integer(data, "updates_per_cycle", "training authorization")
        max_updates = _integer(data, "max_updates", "training authorization")
        if max_updates != cycles * updates_per_cycle:
            raise ContractError("training authorization max_updates must equal cycles times updates_per_cycle")
        capacity_acceptance = _text(data, "capacity_acceptance", "training authorization")
        if capacity_acceptance != CAPACITY_ACCEPTANCE:
            raise ContractError(f"training authorization capacity_acceptance must be {CAPACITY_ACCEPTANCE}")
        approved_at = _text(data, "approved_at", "training authorization")
        _timestamp(approved_at, "training authorization approved_at")
        spend_ceiling = _number(data, "spend_ceiling_usd", "training authorization")
        if spend_ceiling != 100.0 or cycles != 30 or updates_per_cycle != 2_000:
            raise ContractError("training authorization must match the approved $100, 30-cycle stage")
        return cls(
            authorization_id=_text(data, "authorization_id", "training authorization"),
            user_authorization_ref=_text(data, "user_authorization_ref", "training authorization"),
            approved_at=approved_at,
            spend_ceiling_usd=spend_ceiling,
            max_cycles=cycles,
            updates_per_cycle=updates_per_cycle,
            max_updates=max_updates,
            capacity_acceptance=capacity_acceptance,
            qualification_sha256=_digest(data, "qualification_sha256", "training authorization"),
            qualification_binding_sha256=_digest(data, "qualification_binding_sha256", "training authorization"),
            release_sha256=_digest(data, "release_sha256", "training authorization"),
            frozen_dev_sha256=_digest(data, "frozen_dev_sha256", "training authorization"),
        )


@dataclass(frozen=True)
class RentalOffer:
    """Actual rented worker identity, rates, and bounded lifetime."""

    provider: str
    instance_id: str
    gpu_profile: str
    gpu_name: str
    gpu_memory_bytes: int
    ssh_host: str
    ssh_port: int
    ssh_user: str
    rented_at: str
    hard_deadline: str
    destroy_at: str
    prior_spend_usd: float
    resume_from_attempt_id: str | None
    gpu_usd_per_hour: float
    disk_usd_per_hour: float

    @classmethod
    def parse(cls, payload: Any, authorization: TrainingAuthorization) -> "RentalOffer":
        """Parse an offer whose deadline cannot exceed the stage budget."""

        data = require_mapping(payload, "rental offer")
        allowed = {
            "provider",
            "instance_id",
            "gpu_profile",
            "gpu_name",
            "gpu_memory_bytes",
            "ssh_host",
            "ssh_port",
            "ssh_user",
            "rented_at",
            "hard_deadline",
            "destroy_at",
            "prior_spend_usd",
            "resume_from_attempt_id",
            "rates",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ContractError("rental offer has unknown keys", {"unknown": unknown})
        rates = require_mapping(data.get("rates"), "rental offer rates")
        if set(rates) != {"gpu_usd_per_hour", "disk_usd_per_hour"}:
            raise ContractError("rental offer rate fields are not exact")
        rented_at = _text(data, "rented_at", "rental offer")
        hard_deadline = _text(data, "hard_deadline", "rental offer")
        destroy_at = _text(data, "destroy_at", "rental offer")
        started = _timestamp(rented_at, "rental offer rented_at")
        deadline = _timestamp(hard_deadline, "rental offer hard_deadline")
        destruction = _timestamp(destroy_at, "rental offer destroy_at")
        if deadline <= datetime.now(timezone.utc):
            raise ContractError("rental offer hard deadline has passed")
        gpu_rate = _number(rates, "gpu_usd_per_hour", "rental offer rates")
        disk_rate = _number(rates, "disk_usd_per_hour", "rental offer rates", allow_zero=True)
        prior_spend = _number(data, "prior_spend_usd", "rental offer", allow_zero=True)
        if prior_spend >= authorization.spend_ceiling_usd:
            raise ContractError("rental offer has no authorized spend remaining")
        resume_from = data.get("resume_from_attempt_id")
        if resume_from is not None:
            _digest({"digest": resume_from}, "digest", "rental offer resume_from_attempt_id")
        if (prior_spend == 0) != (resume_from is None):
            raise ContractError("rental offer recovery fields are inconsistent")
        remaining_spend = authorization.spend_ceiling_usd - prior_spend
        affordable_deadline = started.timestamp() + remaining_spend / (gpu_rate + disk_rate) * 3600
        if deadline.timestamp() > affordable_deadline + 1:
            raise ContractError("rental offer hard deadline exceeds the authorized spend ceiling")
        if (
            destruction - deadline
        ).total_seconds() < DESTROY_REQUEST_RESERVE_SECONDS or destruction.timestamp() > affordable_deadline + 1:
            raise ContractError("rental offer destroy_at is outside the authorized worker lifetime")
        if (destruction - deadline).total_seconds() > 900:
            raise ContractError("rental offer destroy_at exceeds the bounded backup grace")
        provider = _text(data, "provider", "rental offer")
        if provider != "vast.ai":
            raise ContractError("rental offer provider must be vast.ai")
        instance_id = _text(data, "instance_id", "rental offer")
        if not instance_id.isdecimal():
            raise ContractError("rental offer instance_id must contain only digits")
        return cls(
            provider=provider,
            instance_id=instance_id,
            gpu_profile=_text(data, "gpu_profile", "rental offer"),
            gpu_name=_text(data, "gpu_name", "rental offer"),
            gpu_memory_bytes=_integer(data, "gpu_memory_bytes", "rental offer"),
            ssh_host=_text(data, "ssh_host", "rental offer"),
            ssh_port=_integer(data, "ssh_port", "rental offer"),
            ssh_user=_text(data, "ssh_user", "rental offer"),
            rented_at=rented_at,
            hard_deadline=hard_deadline,
            destroy_at=destroy_at,
            prior_spend_usd=prior_spend,
            resume_from_attempt_id=resume_from,
            gpu_usd_per_hour=gpu_rate,
            disk_usd_per_hour=disk_rate,
        )

    def identity(self) -> dict[str, object]:
        """Return the complete non-secret rental identity."""

        return {
            "provider": self.provider,
            "instance_id": self.instance_id,
            "gpu_profile": self.gpu_profile,
            "gpu_name": self.gpu_name,
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "ssh_user": self.ssh_user,
            "rented_at": self.rented_at,
            "hard_deadline": self.hard_deadline,
            "destroy_at": self.destroy_at,
            "prior_spend_usd": self.prior_spend_usd,
            "resume_from_attempt_id": self.resume_from_attempt_id,
            "rates": {
                "gpu_usd_per_hour": self.gpu_usd_per_hour,
                "disk_usd_per_hour": self.disk_usd_per_hour,
                "total_usd_per_hour": self.gpu_usd_per_hour + self.disk_usd_per_hour,
            },
        }


def _artifact(path: Path, schema: str | None = None) -> tuple[dict[str, object], Mapping[str, Any] | None]:
    if not path.is_file():
        raise RuntimeGateError("launch artifact is missing", {"path": str(path)})
    payload = read_json(path) if schema is not None else None
    if schema is not None and (not isinstance(payload, Mapping) or payload.get("schema") != schema):
        raise RuntimeGateError("launch artifact has the wrong schema", {"path": str(path), "schema": schema})
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}, payload


def _worker_path(layout: Mapping[str, Any], field: str) -> str:
    value = _text(layout, field, "worker layout")
    if not value.startswith("/"):
        raise ContractError(f"worker layout.{field} must be an absolute path")
    return posixpath.normpath(value)


def _resolved_config_path(training_root: str, path: str) -> str:
    if posixpath.isabs(path):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(training_root, path))


def _require_exact_values(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for field, value in expected.items():
        if actual.get(field) != value:
            raise RuntimeGateError(
                f"{label} differs from the qualified training recipe",
                {"field": field, "expected": value, "actual": actual.get(field)},
            )


def _bundle_payload_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeGateError("bundle payload path is not a safe relative path", {"path": relative})
    candidate = root.joinpath(*path.parts)
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeGateError("bundle payload path escapes its root", {"path": relative}) from error
    return candidate


def verify_bundle_contents(root: Path, bundle: Mapping[str, Any], expected_recordings: int) -> None:
    """Verify every manifest and audio payload named by one trainer bundle."""

    manifests = require_mapping(bundle.get("manifests"), "bundle manifests")
    manifest_names = {"wav_scp": "wav.scp", "rttm": "all.rttm", "uem": "all.uem"}
    if set(manifests) != set(manifest_names):
        raise RuntimeGateError("bundle manifest fields are not exact")
    for field, filename in manifest_names.items():
        expected_digest = _digest(manifests, field, "bundle manifests")
        path = root / filename
        if not path.is_file() or sha256_file(path) != expected_digest:
            raise RuntimeGateError("bundle manifest payload differs from its identity", {"path": str(path)})

    recordings = bundle.get("recordings")
    if not isinstance(recordings, list) or len(recordings) != expected_recordings:
        raise RuntimeGateError(f"bundle must contain exactly {expected_recordings} recordings")
    wav_prefix = _text(bundle, "wav_prefix", "bundle")
    expected_wav: dict[str, str] = {}
    sources_by_recording: dict[str, str] = {}
    for row_value in recordings:
        row = require_mapping(row_value, "bundle recording")
        recording_id = _text(row, "recording_id", "bundle recording")
        source = _text(row, "source", "bundle recording")
        if recording_id in expected_wav:
            raise RuntimeGateError("bundle has duplicate recording ids", {"recording_id": recording_id})
        relative = _text(row, "audio_path", "bundle recording")
        audio = _bundle_payload_path(root, relative)
        expected_digest = _digest(row, "audio_sha256", "bundle recording")
        expected_size = _integer(row, "audio_size", "bundle recording")
        if not audio.is_file() or audio.stat().st_size != expected_size or sha256_file(audio) != expected_digest:
            raise RuntimeGateError("bundle audio differs from its identity", {"recording_id": recording_id})
        expected_wav[recording_id] = (PurePosixPath(wav_prefix) / PurePosixPath(relative)).as_posix()
        sources_by_recording[recording_id] = source

    actual_wav: dict[str, str] = {}
    for line in (root / "wav.scp").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in actual_wav:
            raise RuntimeGateError("bundle wav.scp is invalid")
        actual_wav[fields[0]] = fields[1]
    if actual_wav != expected_wav:
        raise RuntimeGateError("bundle wav.scp differs from its recording identities")

    rttm_ids: set[str] = set()
    for line in (root / "all.rttm").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) not in {9, 10} or fields[0] != "SPEAKER" or fields[1] not in expected_wav:
            raise RuntimeGateError("bundle RTTM is invalid")
        rttm_ids.add(fields[1])
    if rttm_ids != set(expected_wav):
        raise RuntimeGateError("bundle RTTM does not cover every recording")

    uem_ids: set[str] = set()
    for line in (root / "all.uem").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 4 or fields[0] in uem_ids or fields[0] not in expected_wav:
            raise RuntimeGateError("bundle UEM is invalid")
        uem_ids.add(fields[0])
    if uem_ids != set(expected_wav):
        raise RuntimeGateError("bundle UEM does not cover every recording")

    if bundle.get("schema") == "speakrs-training-bundle-v1":
        required_sources = bundle.get("required_sources")
        if not isinstance(required_sources, list) or set(required_sources) != {
            "AMI",
            "AliMeeting",
            "AISHELL-5",
            "VoxConverse",
        }:
            raise RuntimeGateError("training bundle does not contain the exact four sources")
        if set(sources_by_recording.values()) != set(required_sources):
            raise RuntimeGateError("training bundle recording sources are incomplete")
    elif bundle.get("schema") == "speakrs-frozen-dev-bundle-v1":
        frozen_splits = require_mapping(bundle.get("frozen_splits"), "frozen development splits")
        if any(
            not isinstance(source, str)
            or not source
            or not isinstance(values, list)
            or any(not isinstance(recording_id, str) or not recording_id for recording_id in values)
            for source, values in frozen_splits.items()
        ):
            raise RuntimeGateError("frozen development split fields are invalid")
        expected_pairs = {
            (source, recording_id)
            for source, values in frozen_splits.items()
            if isinstance(source, str) and isinstance(values, list)
            for recording_id in values
            if isinstance(recording_id, str)
        }
        actual_pairs = {(source, recording_id) for recording_id, source in sources_by_recording.items()}
        if expected_pairs != actual_pairs:
            raise RuntimeGateError("development bundle recordings differ from its frozen splits")


def freeze_training_launch(
    authorization_path: Path,
    qualification_path: Path,
    qualification_binding_path: Path,
    offer_path: Path,
    train_bundle_path: Path,
    dev_bundle_path: Path,
    trainer_config_path: Path,
    initializer_path: Path,
    image_identity_path: Path,
    worker_layout_path: Path,
    predecessor_launch_path: Path | None,
    predecessor_receipt_path: Path | None,
    output: Path,
) -> dict[str, object]:
    """Validate all stage identities and publish one immutable launch lock."""

    authorization_payload = read_json(authorization_path)
    authorization = TrainingAuthorization.parse(authorization_payload)
    offer = RentalOffer.parse(read_json(offer_path), authorization)
    qualification_identity, qualification = _artifact(qualification_path, "speakrs-gpu-qualification-v1")
    assert qualification is not None
    if not qualification.get("ok") or qualification.get("gpu_qualification_status") != "qualified":
        raise RuntimeGateError("launch requires a successful real GPU qualification")
    if qualification.get("qualification_only") is not True or qualification.get("training_ready") is not False:
        raise RuntimeGateError("launch promotion requires the unmodified diagnostic qualification artifact")
    if qualification_identity["sha256"] != authorization.qualification_sha256:
        raise RuntimeGateError("training authorization is bound to a different qualification report")
    if qualification.get("qualification_binding_sha256") != authorization.qualification_binding_sha256:
        raise RuntimeGateError("training authorization is bound to a different qualification input")
    binding_identity, qualification_binding = _artifact(qualification_binding_path, "speakrs-qualification-input-v1")
    assert qualification_binding is not None
    if qualification_binding.get("qualification_binding_sha256") != authorization.qualification_binding_sha256:
        raise RuntimeGateError("qualification binding artifact differs from the authorized input")
    if qualification_binding.get("release_sha256") != authorization.release_sha256:
        raise RuntimeGateError("qualification binding belongs to a different authorized release")
    if qualification.get("gpu_profile") != offer.gpu_profile:
        raise RuntimeGateError("rental GPU profile differs from the qualification")
    physical_batch = _integer(qualification, "physical_batch", "qualification")
    accumulation = _integer(qualification, "accumulation", "qualification")
    train_identity, train_bundle = _artifact(train_bundle_path, "speakrs-training-bundle-v1")
    dev_identity, dev_bundle = _artifact(dev_bundle_path, "speakrs-frozen-dev-bundle-v1")
    assert train_bundle is not None and dev_bundle is not None
    if train_bundle.get("release_sha256") != authorization.release_sha256:
        raise RuntimeGateError("training bundle belongs to a different authorized release")
    frozen_splits = require_mapping(dev_bundle.get("frozen_splits"), "frozen development splits")
    if sha256_json(frozen_splits) != authorization.frozen_dev_sha256:
        raise RuntimeGateError("development bundle differs from the authorized frozen split")
    verify_bundle_contents(train_bundle_path.parent, train_bundle, 1_127)
    verify_bundle_contents(dev_bundle_path.parent, dev_bundle, 44)
    train_recordings = train_bundle["recordings"]
    dev_recordings = dev_bundle["recordings"]
    train_ids = {row["recording_id"] for row in train_recordings}
    dev_ids = {row["recording_id"] for row in dev_recordings}
    train_audio = {row["audio_sha256"] for row in train_recordings}
    dev_audio = {row["audio_sha256"] for row in dev_recordings}
    if train_ids & dev_ids or train_audio & dev_audio:
        raise RuntimeGateError("training and development bundles overlap")
    config_identity, _ = _artifact(trainer_config_path)
    initializer_identity, _ = _artifact(initializer_path)
    if initializer_identity["sha256"] != qualification_binding.get("wavlm_initializer_sha256"):
        raise RuntimeGateError("WavLM initializer differs from the qualified initializer")
    image_identity, image = _artifact(image_identity_path, "speakrs-main-training-image-v1")
    assert image is not None
    source_commit = _text(image, "source_commit", "image identity")
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise RuntimeGateError("image source commit is invalid")
    _oci_digest(image, "index_digest", "image identity")
    _oci_digest(image, "linux_amd64_manifest_digest", "image identity")
    if image.get("anonymous_manifest_get_status") != 200:
        raise RuntimeGateError("image identity has no successful anonymous manifest check")
    runtime_versions = require_mapping(image.get("runtime_versions"), "image runtime versions")
    if set(runtime_versions) != {"python", "torch", "accelerate"} or any(
        not isinstance(value, str) or not value for value in runtime_versions.values()
    ):
        raise RuntimeGateError("image runtime version fields are not exact")
    runtime_code = image.get("runtime_code_sha256")
    required_runtime_code = {
        "diarizen/trainer_dual_opt.py",
        "diarizen/trainer_utils.py",
        "recipes/diar_ssl/run_dual_opt.py",
        "recipes/diar_ssl/trainer_dual_opt.py",
        "recipes/diar_ssl/dataset.py",
        "diarizen/models/eend/model_wavlm_conformer.py",
        "recipes/speakrs/large/cli.py",
        "recipes/speakrs/large/controller.py",
        "recipes/speakrs/large/remote_backup.py",
        "recipes/speakrs/large/training_admission.py",
        "recipes/speakrs/large/training_supervisor.py",
        "recipes/speakrs/large/vast_guard.py",
        "recipes/speakrs/large_run.py",
    }
    if not isinstance(runtime_code, Mapping) or set(runtime_code) != required_runtime_code:
        raise RuntimeGateError("image identity has no exact training runtime code set")
    for relative, digest in runtime_code.items():
        _digest({"digest": digest}, "digest", f"image runtime code {relative}")
    layout = require_mapping(read_json(worker_layout_path), "worker layout")
    allowed_layout = {
        "schema",
        "training_root",
        "trainer_config",
        "train_bundle",
        "dev_bundle",
        "initializer",
        "experiment_root",
        "checkpoint_root",
        "trusted_backup_root",
        "repository_root",
    }
    if layout.get("schema") != "speakrs-training-worker-layout-v1" or set(layout) != allowed_layout:
        raise ContractError("worker layout fields are not exact")
    worker_paths = {field: _worker_path(layout, field) for field in allowed_layout - {"schema"}}
    expected_training_root = posixpath.join(worker_paths["repository_root"], "recipes", "diar_ssl")
    if worker_paths["training_root"] != expected_training_root:
        raise RuntimeGateError("worker training root differs from the repository training entry point")
    expected_config_parent = posixpath.join(worker_paths["repository_root"], "recipes", "speakrs", "conf")
    if posixpath.dirname(worker_paths["trainer_config"]) != expected_config_parent:
        raise RuntimeGateError("worker trainer config is outside the repository config directory")
    if trainer_config_path.name != posixpath.basename(worker_paths["trainer_config"]):
        raise RuntimeGateError("local and worker trainer config names differ")
    config = toml.load(trainer_config_path)
    if require_mapping(config.get("finetune"), "trainer finetune").get("finetune") is not False:
        raise RuntimeGateError("trainer must start from the qualified WavLM initializer")
    if require_mapping(config.get("trainer"), "trainer config trainer").get("path") != "trainer_dual_opt.Trainer":
        raise RuntimeGateError("trainer class differs from the qualified training recipe")
    trainer_args = require_mapping(
        require_mapping(config.get("trainer"), "trainer config trainer").get("args"), "trainer args"
    )
    if trainer_args.get("max_steps") != authorization.max_updates:
        raise RuntimeGateError("trainer max_steps differs from the authorized stage")
    if trainer_args.get("snapshot_every_updates") != authorization.updates_per_cycle:
        raise RuntimeGateError("trainer snapshot interval differs from the authorized cycle")
    if trainer_args.get("max_update_checkpoints") != 3:
        raise RuntimeGateError("trainer must retain exactly three worker update checkpoints")
    if trainer_args.get("save_ckpt_interval") != 0 or trainer_args.get("max_num_checkpoints") != 0:
        raise RuntimeGateError("fixed-update training must disable duplicate full recovery checkpoints")
    if trainer_args.get("ranked_checkpoint_count") != 5:
        raise RuntimeGateError("trainer must retain exactly five model-only selection checkpoints")
    _require_exact_values(
        trainer_args,
        {
            "gradient_percentile": 90,
            "gradient_history_size": 1_000,
            "save_max_score": False,
            "validation_interval": 1,
            "validation_before_training": True,
            "freeze_wavlm": False,
            "lr_decay": False,
            "use_one_cycle_lr": False,
            "warmup_steps": 0,
        },
        "trainer controls",
    )
    train_config = require_mapping(config.get("train_dataset"), "train dataset")
    train_args = require_mapping(train_config.get("args"), "train dataset args")
    train_loader = require_mapping(train_config.get("dataloader"), "train dataloader")
    dev_args = require_mapping(
        require_mapping(config.get("validate_dataset"), "validate dataset").get("args"), "dev args"
    )
    meta = require_mapping(config.get("meta"), "trainer config meta")
    save_dir = _resolved_config_path(worker_paths["training_root"], _text(meta, "save_dir", "trainer config meta"))
    if save_dir != worker_paths["experiment_root"]:
        raise RuntimeGateError("trainer save directory differs from the worker experiment root")
    expected_checkpoint_root = posixpath.join(
        worker_paths["experiment_root"],
        PurePosixPath(worker_paths["trainer_config"]).stem,
        "checkpoints",
    )
    if worker_paths["checkpoint_root"] != expected_checkpoint_root:
        raise RuntimeGateError("worker checkpoint root differs from the trainer output path")
    model_args = require_mapping(require_mapping(config.get("model"), "model").get("args"), "model args")
    if require_mapping(config.get("model"), "model").get("path") != (
        "diarizen.models.eend.model_wavlm_conformer.Model"
    ):
        raise RuntimeGateError("model class differs from the qualified training recipe")
    _require_exact_values(
        model_args,
        {
            "strict_wavlm_load": True,
            "wavlm_layer_num": 25,
            "wavlm_feat_dim": 1_024,
            "attention_in": 256,
            "ffn_hidden": 1_024,
            "num_head": 4,
            "num_layer": 4,
            "kernel_size": 31,
            "dropout": 0.1,
            "chunk_size": 8,
            "use_posi": False,
            "output_activate_function": False,
            "selected_channel": 0,
            "max_speakers_per_chunk": 4,
            "max_speakers_per_frame": 2,
        },
        "model configuration",
    )
    for section_name, section, expected_shift in (
        ("training dataset", train_config, 6),
        ("development dataset", require_mapping(config.get("validate_dataset"), "validate dataset"), 8),
    ):
        if section.get("path") != "dataset.DiarizationDataset":
            raise RuntimeGateError(f"{section_name} class differs from the qualified training recipe")
        _require_exact_values(
            require_mapping(section.get("args"), f"{section_name} args"),
            {"chunk_size": 8, "chunk_shift": expected_shift, "sample_rate": 16_000},
            section_name,
        )
    _require_exact_values(
        train_loader,
        {"num_workers": 4, "drop_last": True, "pin_memory": True},
        "training dataloader",
    )
    dev_loader = require_mapping(
        require_mapping(config.get("validate_dataset"), "validate dataset").get("dataloader"),
        "development dataloader",
    )
    _require_exact_values(
        dev_loader,
        {"num_workers": 4, "drop_last": False, "pin_memory": True},
        "development dataloader",
    )
    dev_batch_size = dev_loader.get("batch_size")
    if (
        isinstance(dev_batch_size, bool)
        or not isinstance(dev_batch_size, int)
        or not 0 < dev_batch_size <= physical_batch
    ):
        raise RuntimeGateError("development batch size is outside the qualified physical batch")
    expected_paths = (
        ("train_bundle", train_args, "scp_file", "wav.scp"),
        ("train_bundle", train_args, "rttm_file", "all.rttm"),
        ("train_bundle", train_args, "uem_file", "all.uem"),
        ("dev_bundle", dev_args, "scp_file", "wav.scp"),
        ("dev_bundle", dev_args, "rttm_file", "all.rttm"),
        ("dev_bundle", dev_args, "uem_file", "all.uem"),
        ("initializer", model_args, "wavlm_src", ""),
    )
    for layout_field, section, config_field, suffix in expected_paths:
        config_path = _resolved_config_path(
            worker_paths["training_root"], _text(section, config_field, "trainer config")
        )
        expected = worker_paths[layout_field] if not suffix else posixpath.join(worker_paths[layout_field], suffix)
        if config_path != expected:
            raise RuntimeGateError("trainer path differs from the worker layout", {"field": config_field})
    if (
        train_loader.get("batch_size") != physical_batch
        or trainer_args.get("gradient_accumulation_steps") != accumulation
    ):
        raise RuntimeGateError("trainer batch plan differs from the qualified plan")
    qualification_spec = require_mapping(qualification.get("qualification_spec"), "qualification spec")
    if qualification_spec.get("effective_batch") != physical_batch * accumulation or (
        require_mapping(qualification.get("precision"), "qualification precision").get("requested") != "bf16"
    ):
        raise RuntimeGateError("trainer execution plan differs from the qualified plan")
    optimizer_expected = {
        "optimizer_small": {"path": "torch.optim.AdamW", "lr": 1e-5},
        "optimizer_big": {"path": "torch.optim.AdamW", "lr": 1e-3},
    }
    for section_name, expected in optimizer_expected.items():
        optimizer = require_mapping(config.get(section_name), section_name)
        if optimizer.get("path") != expected["path"]:
            raise RuntimeGateError(f"{section_name} class differs from the qualified training recipe")
        _require_exact_values(
            require_mapping(optimizer.get("args"), f"{section_name} args"),
            {
                "lr": expected["lr"],
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
            },
            section_name,
        )
    artifacts = {
        "qualification": qualification_identity,
        "qualification_binding": binding_identity,
        "train_bundle": train_identity,
        "dev_bundle": dev_identity,
        "trainer_config": config_identity,
        "initializer": initializer_identity,
        "image_identity": image_identity,
    }
    durable_core = {
        "schema": LAUNCH_SCHEMA,
        "authorization": dict(authorization_payload),
        "authorization_sha256": sha256_file(authorization_path),
        "qualification_binding_sha256": qualification.get("qualification_binding_sha256"),
        "image": dict(image),
        "artifacts": artifacts,
        "worker_paths": worker_paths,
        "physical_batch": physical_batch,
        "accumulation": accumulation,
        "max_cycles": authorization.max_cycles,
        "updates_per_cycle": authorization.updates_per_cycle,
        "max_updates": authorization.max_updates,
        "spend_ceiling_usd": authorization.spend_ceiling_usd,
    }
    launch_id = sha256_json(durable_core)
    recovery: dict[str, object] | None = None
    if offer.resume_from_attempt_id is None:
        if predecessor_launch_path is not None or predecessor_receipt_path is not None:
            raise RuntimeGateError("initial rental cannot include predecessor artifacts")
    else:
        if predecessor_launch_path is None or predecessor_receipt_path is None:
            raise RuntimeGateError("replacement rental requires predecessor launch and spend receipt")
        predecessor = parse_launch_lock(read_json(predecessor_launch_path))
        if predecessor.get("attempt_id") != offer.resume_from_attempt_id:
            raise RuntimeGateError("replacement rental names a different predecessor attempt")
        if predecessor.get("launch_id") != launch_id:
            raise RuntimeGateError("replacement rental belongs to a different durable training run")
        receipt = require_mapping(read_json(predecessor_receipt_path), "predecessor rental receipt")
        required_receipt = {
            "schema",
            "attempt_id",
            "launch_id",
            "instance_id",
            "rented_at",
            "ended_at",
            "rates",
            "deletion_outcome",
            "spend_usd",
        }
        if set(receipt) != required_receipt or receipt.get("schema") != "speakrs-rental-attempt-receipt-v1":
            raise RuntimeGateError("predecessor rental receipt fields are not exact")
        previous_offer = require_mapping(predecessor.get("offer"), "predecessor offer")
        if (
            receipt.get("attempt_id") != predecessor.get("attempt_id")
            or receipt.get("launch_id") != launch_id
            or receipt.get("instance_id") != previous_offer.get("instance_id")
            or receipt.get("rented_at") != previous_offer.get("rented_at")
            or receipt.get("rates") != previous_offer.get("rates")
            or receipt.get("deletion_outcome") not in {"destroyed", "already-absent"}
        ):
            raise RuntimeGateError("predecessor rental receipt does not match the prior attempt")
        ended_at = _timestamp(_text(receipt, "ended_at", "predecessor rental receipt"), "predecessor ended_at")
        rented_at = _timestamp(_text(previous_offer, "rented_at", "predecessor offer"), "predecessor rented_at")
        if ended_at < rented_at:
            raise RuntimeGateError("predecessor rental receipt has an invalid lifetime")
        rates = require_mapping(previous_offer.get("rates"), "predecessor rates")
        previous_spend = _number(previous_offer, "prior_spend_usd", "predecessor offer", allow_zero=True)
        calculated_spend = previous_spend + (ended_at - rented_at).total_seconds() / 3600 * _number(
            rates, "total_usd_per_hour", "predecessor rates"
        )
        receipt_spend = _number(receipt, "spend_usd", "predecessor rental receipt", allow_zero=True)
        if (
            receipt_spend >= authorization.spend_ceiling_usd
            or not math.isclose(receipt_spend, calculated_spend, abs_tol=0.01)
            or not math.isclose(offer.prior_spend_usd, receipt_spend, abs_tol=0.01)
        ):
            raise RuntimeGateError("replacement rental does not preserve the prior spend")
        recovery = {
            "predecessor_launch": {"sha256": sha256_file(predecessor_launch_path)},
            "predecessor_receipt": {"sha256": sha256_file(predecessor_receipt_path)},
        }
    attempt_core = {
        **durable_core,
        "launch_id": launch_id,
        "offer": offer.identity(),
        "recovery": recovery,
        "hard_deadline": offer.hard_deadline,
        "state": "launch-locked",
    }
    launch = {**attempt_core, "attempt_id": sha256_json(attempt_core)}
    write_json(output, launch)
    return {
        "ok": True,
        "command": "freeze-run",
        "launch_id": launch["launch_id"],
        "attempt_id": launch["attempt_id"],
        "output": str(output),
    }


def parse_launch_lock(payload: Any) -> Mapping[str, Any]:
    """Validate the immutable launch identifier and return its fields."""

    data = require_mapping(payload, "launch lock")
    if data.get("schema") != LAUNCH_SCHEMA:
        raise ContractError(f"launch schema must be {LAUNCH_SCHEMA}")
    attempt_id = _text(data, "attempt_id", "launch lock")
    attempt_core = dict(data)
    attempt_core.pop("attempt_id")
    if sha256_json(attempt_core) != attempt_id:
        raise ContractError("launch attempt digest does not match its content")
    launch_id = _text(data, "launch_id", "launch lock")
    durable_core = dict(attempt_core)
    for field in ("launch_id", "offer", "recovery", "hard_deadline", "state"):
        durable_core.pop(field)
    if sha256_json(durable_core) != launch_id:
        raise ContractError("durable launch digest does not match its content")
    TrainingAuthorization.parse(require_mapping(data.get("authorization"), "launch authorization"))
    return data
