"""Focused tests for the qualification-only release binding."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from recipes.speakrs.large import data as data_module
from recipes.speakrs.large.budget import BudgetLedger
from recipes.speakrs.large.contracts import (
    DEFAULT_BUDGET,
    DEFAULT_MODEL,
    QUALIFICATION_BINDING_SCHEMA,
    QUALIFICATION_CAPACITY_LIMIT,
    QUALIFICATION_SOURCES,
    QUALIFICATION_TARGET_PROFILE,
    parse_qualification_binding,
)
from recipes.speakrs.large.controller import Controller, FakeProvider, lease_from_offer
from recipes.speakrs.large.errors import ContractError, LargeError
from recipes.speakrs.large.hashing import sha256_file, sha256_json
from recipes.speakrs.large.qualification import QualificationSpec, _validate_real_config, execute_qualification
from recipes.speakrs.large.recovery import LocalTransport


def _digest(label: str) -> str:
    """Return a non-placeholder test digest."""

    return sha256_json({"label": label})


def _payload() -> dict[str, object]:
    """Return one valid four-source diagnostic binding payload."""

    rows = {
        source: {
            "chunk_seconds": 8,
            "max_overlap": 2,
            "local_slots": 4,
            "speaker_seconds": 100.0,
            "lost_seconds": 1.19 if source == "AMI" else 0.1,
            "loss_fraction": 0.0119 if source == "AMI" else 0.001,
            "admitted": source != "AMI",
        }
        for source in QUALIFICATION_SOURCES
    }
    return {
        "schema": QUALIFICATION_BINDING_SCHEMA,
        "purpose": "gpu_qualification",
        "release_sha256": _digest("release"),
        "restore_receipt_sha256": _digest("restore"),
        "bundle_manifest_sha256": _digest("bundle"),
        "wavlm_initializer_sha256": _digest("wavlm-initializer"),
        "required_sources": list(QUALIFICATION_SOURCES),
        "model_identity": DEFAULT_MODEL.identity(),
        "target_profile": dict(QUALIFICATION_TARGET_PROFILE),
        "source_capacity": rows,
        "capacity_policy": "diagnostic",
        "capacity_limit": QUALIFICATION_CAPACITY_LIMIT,
        "authorization_reference": "approval-1",
    }


def test_binding_derives_diagnostic_readiness_and_preserves_ami_loss() -> None:
    binding = parse_qualification_binding(_payload())

    assert binding.qualification_ready is True
    assert binding.training_ready is False
    assert binding.gpu_qualification_status == "not_run"
    assert binding.capacity_admitted is False
    assert binding.source_capacity["AMI"]["lost_seconds"] == pytest.approx(1.19)
    assert binding.source_capacity["AMI"]["admitted"] is False
    with pytest.raises(TypeError):
        binding.target_profile["chunk_seconds"] = 10  # type: ignore[index]


def test_four_overlap_profile_cannot_substitute_for_two_overlap() -> None:
    payload = _payload()
    payload["target_profile"] = {"chunk_seconds": 8, "max_overlap": 4, "local_slots": 4}

    with pytest.raises(ContractError):
        parse_qualification_binding(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("purpose", "training"),
        ("required_sources", ["AMI", "AliMeeting", "AISHELL4", "VoxConverse"]),
        ("capacity_policy", "admission"),
        ("authorization_reference", ""),
    ),
)
def test_binding_rejects_wrong_contract_inputs(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ContractError):
        parse_qualification_binding(payload)


def test_binding_digest_covers_release_and_source_totals() -> None:
    payload = _payload()
    binding = parse_qualification_binding(payload)
    sealed = binding.as_dict()

    changed_release = copy.deepcopy(sealed)
    changed_release["release_sha256"] = _digest("other-release")
    with pytest.raises(ContractError):
        parse_qualification_binding(changed_release)

    changed_capacity = copy.deepcopy(sealed)
    changed_capacity["source_capacity"]["AMI"]["lost_seconds"] = 2.0  # type: ignore[index]
    changed_capacity["source_capacity"]["AMI"]["loss_fraction"] = 0.02  # type: ignore[index]
    changed_capacity["source_capacity"]["AMI"]["admitted"] = False  # type: ignore[index]
    with pytest.raises(ContractError):
        parse_qualification_binding(changed_capacity)


def test_lease_and_gpu_report_retain_binding_digest(tmp_path: Path) -> None:
    binding = parse_qualification_binding(_payload())
    lease = lease_from_offer(
        {
            "offer_id": "o1",
            "gpu_profile": "4090",
            "usd_per_hour": 0.3,
            "disk_usd_per_hour": 0.01,
            "hard_deadline": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
        "prep",
        "/tmp/backup",
        binding.binding_sha256,
    )
    controller = Controller(BudgetLedger(DEFAULT_BUDGET), FakeProvider(), LocalTransport())
    result = controller.control_qualification(lease, binding)
    assert lease["qualification_binding_sha256"] == binding.binding_sha256
    assert result["qualification_binding_sha256"] == binding.binding_sha256

    config = tmp_path / "trainer.toml"
    config.write_text("[meta]\nseed = 3407\n", encoding="utf-8")
    qualification_control = controller.control_qualification(lease, binding)
    report = execute_qualification(
        config,
        "4090",
        QualificationSpec(warmup_optimizer_updates=1, measured_optimizer_updates=1),
        tmp_path / "qualification.json",
        qualification_binding=binding,
        qualification_control=qualification_control,
        attempt_runner=lambda **_kwargs: {
            "ok": True,
            "total_memory_bytes": 100,
            "peak_reserved_bytes": 70,
            "update_seconds": [1.0],
            "measured_update_count": 1,
            "checkpoint_write_reload": {"feasible": True, "write": True, "reload": True},
        },
        device_facts={"cuda_available": True, "device_name": "RTX 4090", "total_memory_bytes": 100},
    )
    assert report["qualification_binding_sha256"] == binding.binding_sha256
    assert report["training_ready"] is False
    report_lease = report["qualification_lease"]
    assert report_lease["lease_id"] == lease["lease_id"]  # type: ignore[index]
    assert report_lease["spend_ceiling_usd"] == 5.0  # type: ignore[index]
    assert report_lease["qualification_binding_sha256"] == binding.binding_sha256  # type: ignore[index]
    assert report_lease["max_runtime_seconds"] > 0  # type: ignore[index]


@pytest.mark.parametrize(("field", "value"), (("spend_ceiling_usd", float("nan")), ("gpu_rate", float("nan"))))
def test_qualification_control_rejects_non_finite_budget_values(field: str, value: float) -> None:
    binding = parse_qualification_binding(_payload())
    lease = lease_from_offer(
        {
            "offer_id": "o1",
            "gpu_profile": "4090",
            "usd_per_hour": 0.3,
            "disk_usd_per_hour": 0.01,
            "hard_deadline": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
        "prep",
        "/tmp/backup",
        binding.binding_sha256,
    )
    if field == "spend_ceiling_usd":
        lease["spend_ceiling_usd"] = value
    else:
        lease["rates"]["gpu_usd_per_hour"] = value  # type: ignore[index]

    controller = Controller(BudgetLedger(DEFAULT_BUDGET), FakeProvider(), LocalTransport())
    with pytest.raises(LargeError):
        controller.control_qualification(lease, binding)


def test_real_config_requires_the_content_bound_four_source_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    audio = bundle / "audio"
    receipts = bundle / "restore-receipts"
    audio.mkdir(parents=True)
    receipts.mkdir()
    payload = _payload()
    selected_batches: dict[str, dict[str, object]] = {}
    recordings: list[dict[str, object]] = []
    wav_rows: list[str] = []
    for index, source in enumerate(QUALIFICATION_SOURCES):
        batch_sha256 = _digest(f"batch-{source}")
        audio_path = audio / f"{index}.flac"
        audio_path.write_bytes(f"audio-{source}".encode())
        receipt = receipts / f"{source}.json"
        receipt.write_text(f"receipt-{source}\n", encoding="utf-8")
        selected_batches[source] = {
            "batch_sha256": batch_sha256,
            "acceptance_sha256": _digest(f"acceptance-{source}"),
            "restore_receipt": receipt.relative_to(bundle).as_posix(),
            "restore_receipt_sha256": sha256_file(receipt),
        }
        recording_id = f"recording-{index}"
        relative_audio = audio_path.relative_to(bundle).as_posix()
        wav_rows.append(f"{recording_id} bundle/{relative_audio}\n")
        recordings.append(
            {
                "recording_id": recording_id,
                "source": source,
                "batch_sha256": batch_sha256,
                "audio_path": relative_audio,
                "audio_sha256": sha256_file(audio_path),
                "audio_size": audio_path.stat().st_size,
                "rttm_sha256": _digest(f"rttm-{source}"),
                "uem_sha256": _digest(f"uem-{source}"),
            }
        )
    manifests = {
        "wav_scp": bundle / "wav.scp",
        "rttm": bundle / "all.rttm",
        "uem": bundle / "all.uem",
    }
    manifests["wav_scp"].write_text("".join(wav_rows), encoding="utf-8")
    manifests["rttm"].write_text("rttm\n", encoding="utf-8")
    manifests["uem"].write_text("uem\n", encoding="utf-8")
    manifest = {
        "schema": "speakrs-qualification-bundle-v1",
        "release_sha256": payload["release_sha256"],
        "restore_receipt_sha256": payload["restore_receipt_sha256"],
        "required_sources": list(QUALIFICATION_SOURCES),
        "wav_prefix": "bundle",
        "selection_plan_sha256": _digest("selection"),
        "selected_batches": selected_batches,
        "manifests": {name: sha256_file(path) for name, path in manifests.items()},
        "recordings": recordings,
    }
    manifest_path = bundle / "bundle.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload["bundle_manifest_sha256"] = sha256_file(manifest_path)
    wavlm = tmp_path / "wavlm.pt"
    wavlm.write_bytes(b"wavlm")
    payload["wavlm_initializer_sha256"] = sha256_file(wavlm)
    binding = parse_qualification_binding(payload)
    config = {
        "model": {
            "path": "diarizen.models.eend.model_wavlm_conformer.Model",
            "args": {
                "chunk_size": 8,
                "max_speakers_per_chunk": 4,
                "max_speakers_per_frame": 2,
                "strict_wavlm_load": True,
                "wavlm_layer_num": 25,
                "wavlm_feat_dim": 1024,
                "wavlm_src": "wavlm.pt",
            },
        },
        "trainer": {"path": "trainer_dual_opt.Trainer"},
        "optimizer_small": {"path": "torch.optim.AdamW"},
        "optimizer_big": {"path": "torch.optim.AdamW"},
        "train_dataset": {
            "path": "dataset.DiarizationDataset",
            "args": {
                "chunk_size": 8,
                "chunk_shift": 6,
                "sample_rate": 16000,
                "scp_file": "bundle/wav.scp",
                "rttm_file": "bundle/all.rttm",
                "uem_file": "bundle/all.uem",
            },
        },
        "qualification": {
            "binding_sha256": binding.binding_sha256,
            "bundle_manifest": "bundle/bundle.json",
        },
    }

    facts = _validate_real_config(config, tmp_path, binding)

    assert facts["qualification_bundle"]["recordings"] == len(QUALIFICATION_SOURCES)  # type: ignore[index]
    wrong_model = copy.deepcopy(config)
    wrong_model["model"]["path"] = "alternate.wavlm.Model"  # type: ignore[index]
    with pytest.raises(LargeError, match="exact DiariZen WavLM model"):
        _validate_real_config(wrong_model, tmp_path, binding)
    wrong_dataset = copy.deepcopy(config)
    wrong_dataset["train_dataset"]["path"] = "alternate.Dataset"  # type: ignore[index]
    with pytest.raises(LargeError, match="exact DiarizationDataset reader"):
        _validate_real_config(wrong_dataset, tmp_path, binding)
    wavlm.write_bytes(b"changed")
    with pytest.raises(LargeError, match="initializer differs"):
        _validate_real_config(config, tmp_path, binding)
    wavlm.write_bytes(b"wavlm")
    (audio / "0.flac").write_bytes(b"changed")
    with pytest.raises(LargeError, match="audio digest"):
        _validate_real_config(config, tmp_path, binding)


def test_qualification_bundle_builds_real_trainer_inputs(monkeypatch, tmp_path: Path) -> None:
    release = tmp_path / "release"
    seal = tmp_path / "seal.json"
    final_restore = tmp_path / "final-restore.json"
    selection = tmp_path / "selection.json"
    output = tmp_path / "bundle"
    seal.write_text("{}\n", encoding="utf-8")
    final_restore.write_text("final restore\n", encoding="utf-8")
    batches: dict[str, dict[str, object]] = {}
    capacity: dict[str, tuple[str, object]] = {}
    portable: dict[str, dict[str, object]] = {}
    restore_payloads: dict[str, dict[str, object]] = {}
    selection_rows: list[dict[str, str]] = []
    for index, source in enumerate(QUALIFICATION_SOURCES):
        batch_sha256 = _digest(f"bundle-batch-{source}")
        acceptance_sha256 = _digest(f"bundle-acceptance-{source}")
        remote = tmp_path / f"remote-{index}.json"
        remote.write_text("{}\n", encoding="utf-8")
        selection_rows.append({"batch_sha256": batch_sha256, "path": str(remote)})
        recording_id = f"recording-{index}"
        identities: dict[str, dict[str, object]] = {}
        restored_objects: list[dict[str, object]] = []
        for field, suffix, content in (
            ("audio", ".flac", f"audio-{source}".encode()),
            ("rttm", ".rttm", f"SPEAKER {recording_id} 1 0 1 <NA> <NA> spk <NA> <NA>\n".encode()),
            ("uem", ".uem", f"{recording_id} 1 0 1\n".encode()),
        ):
            path = tmp_path / f"{recording_id}{suffix}"
            path.write_bytes(content)
            key = f"key/{source}/{field}"
            identity = {"key": key, "sha256": sha256_file(path), "size": path.stat().st_size}
            identities[field] = identity
            restored_objects.append({**identity, "restored_path": str(path)})
        batches[batch_sha256] = {
            "batch_sha256": batch_sha256,
            "acceptance_sha256": acceptance_sha256,
        }
        capacity[batch_sha256] = (source, object())
        portable[batch_sha256] = {
            "recordings": [{"recording_id": recording_id, **identities}],
        }
        restore_payloads[str(remote)] = {"objects": restored_objects}
    selection.write_text(
        json.dumps(
            {
                "schema": "speakrs-qualification-restore-plan-v1",
                "purpose": "gpu_qualification",
                "batches": selection_rows,
            }
        ),
        encoding="utf-8",
    )
    context = {
        "release_sha256": _digest("bundle-release"),
        "batch_by_hash": batches,
        "capacity_by_batch": capacity,
        "batch_marker_by_hash": {digest: {} for digest in batches},
        "portable_by_batch": portable,
    }

    monkeypatch.setattr(data_module, "_backend_from_spec", lambda _spec, _backend: object())
    monkeypatch.setattr(
        data_module,
        "_qualification_release_context",
        lambda *_args: (context, {}),
    )

    def restore_check(_spec, remote_path, receipt_path, *, backend):
        del backend
        payload = restore_payloads[str(remote_path)]
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(data_module, "restore_check", restore_check)
    monkeypatch.setattr(data_module, "_batch_restore_matches", lambda *_args, **_kwargs: True)

    result = data_module.qualification_bundle_data(
        object(),  # type: ignore[arg-type]
        release,
        seal,
        final_restore,
        selection,
        "../speakrs/data/qualification-v1",
        output,
    )

    manifest = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert result["recordings"] == 4
    assert set(manifest["selected_batches"]) == set(QUALIFICATION_SOURCES)
    assert len((output / "wav.scp").read_text(encoding="utf-8").splitlines()) == 4
    assert "../speakrs/data/qualification-v1/audio/AMI/" in (output / "wav.scp").read_text(encoding="utf-8")
    assert len((output / "all.rttm").read_text(encoding="utf-8").splitlines()) == 4
    assert len((output / "all.uem").read_text(encoding="utf-8").splitlines()) == 4

    original_rttm = (output / "all.rttm").read_text(encoding="utf-8")
    (output / "all.rttm").write_text("forged\n", encoding="utf-8")
    manifest["manifests"]["rttm"] = sha256_file(output / "all.rttm")
    (output / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LargeError, match="labels differ"):
        data_module._validate_qualification_bundle_against_release(output / "bundle.json", context)

    (output / "all.rttm").write_text(original_rttm, encoding="utf-8")
    manifest["manifests"]["rttm"] = sha256_file(output / "all.rttm")
    manifest["selected_batches"]["AMI"]["batch_sha256"] = _digest("forged-batch")
    (output / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LargeError, match="outside the final release"):
        data_module._validate_qualification_bundle_against_release(output / "bundle.json", context)
