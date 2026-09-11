"""Tests for typed promotion from diagnostic qualification to a bounded stage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import toml

from recipes.speakrs.large.errors import RuntimeGateError
from recipes.speakrs.large.hashing import sha256_file, sha256_json
from recipes.speakrs.large.jsonio import read_json, write_json
from recipes.speakrs.large.training_admission import freeze_training_launch, parse_launch_lock


def _write_bundle(
    root: Path,
    schema: str,
    recordings: list[tuple[str, str]],
    *,
    release_sha256: str | None = None,
    frozen_splits: dict[str, list[str]] | None = None,
) -> Path:
    root.mkdir()
    wav_rows = []
    rttm_rows = []
    uem_rows = []
    rows = []
    for index, (recording_id, source) in enumerate(recordings):
        audio = root / "audio" / f"{index}.flac"
        audio.parent.mkdir(exist_ok=True)
        audio.write_bytes(f"{root.name}-audio-{index}".encode())
        relative = audio.relative_to(root).as_posix()
        wav_rows.append(f"{recording_id} ../speakrs/data/{root.name}/{relative}\n")
        rttm_rows.append(f"SPEAKER {recording_id} 1 0.000000 1.000000 <NA> <NA> speaker <NA> <NA>\n")
        uem_rows.append(f"{recording_id} 1 0.000000 1.000000\n")
        rows.append(
            {
                "recording_id": recording_id,
                "source": source,
                "audio_path": relative,
                "audio_sha256": sha256_file(audio),
                "audio_size": audio.stat().st_size,
            }
        )
    manifest_paths = {"wav_scp": root / "wav.scp", "rttm": root / "all.rttm", "uem": root / "all.uem"}
    for path, contents in zip(manifest_paths.values(), (wav_rows, rttm_rows, uem_rows), strict=True):
        path.write_text("".join(contents), encoding="utf-8")
    payload = {
        "schema": schema,
        "wav_prefix": f"../speakrs/data/{root.name}",
        "manifests": {name: sha256_file(path) for name, path in manifest_paths.items()},
        "recordings": rows,
    }
    if release_sha256 is not None:
        payload["release_sha256"] = release_sha256
        payload["required_sources"] = ["AMI", "AliMeeting", "AISHELL-5", "VoxConverse"]
    if frozen_splits is not None:
        payload["frozen_splits"] = frozen_splits
    bundle = root / "bundle.json"
    write_json(bundle, payload)
    return bundle


def _inputs(root: Path) -> dict[str, Path]:
    binding = "d" * 64
    release = "e" * 64
    initializer = root / "initializer.pt"
    initializer.write_bytes(b"initializer")
    frozen_splits = {
        "AMI": [f"ami-{index}" for index in range(15)],
        "AliMeeting": [f"ali-{index}" for index in range(15)],
        "AISHELL-5": [f"aishell-{index}" for index in range(14)],
    }
    qualification = root / "qualification.json"
    write_json(
        qualification,
        {
            "schema": "speakrs-gpu-qualification-v1",
            "ok": True,
            "gpu_qualification_status": "qualified",
            "qualification_only": True,
            "training_ready": False,
            "gpu_profile": "4090",
            "physical_batch": 8,
            "accumulation": 8,
            "qualification_binding_sha256": binding,
            "qualification_spec": {"effective_batch": 64},
            "precision": {"requested": "bf16"},
        },
    )
    qualification_binding = root / "qualification-binding.json"
    write_json(
        qualification_binding,
        {
            "schema": "speakrs-qualification-input-v1",
            "qualification_binding_sha256": binding,
            "release_sha256": release,
            "wavlm_initializer_sha256": sha256_file(initializer),
        },
    )
    authorization = root / "authorization.json"
    write_json(
        authorization,
        {
            "schema": "speakrs-training-authorization-v1",
            "authorization_id": "four-source-stage-1",
            "user_authorization_ref": "approved-30-cycle-stage",
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "spend_ceiling_usd": 100.0,
            "max_cycles": 30,
            "updates_per_cycle": 2_000,
            "max_updates": 60_000,
            "capacity_acceptance": "accept-qualified-diagnostic-capacity",
            "qualification_sha256": sha256_file(qualification),
            "qualification_binding_sha256": binding,
            "release_sha256": release,
            "frozen_dev_sha256": sha256_json(frozen_splits),
        },
    )
    offer = root / "offer.json"
    now = datetime.now(timezone.utc)
    write_json(
        offer,
        {
            "provider": "vast.ai",
            "instance_id": "50520000",
            "gpu_profile": "4090",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_memory_bytes": 24_000_000_000,
            "ssh_host": "203.0.113.10",
            "ssh_port": 2222,
            "ssh_user": "root",
            "rented_at": (now - timedelta(hours=1)).isoformat(),
            "hard_deadline": (now + timedelta(hours=2)).isoformat(),
            "destroy_at": (now + timedelta(hours=2, minutes=10)).isoformat(),
            "prior_spend_usd": 0.0,
            "resume_from_attempt_id": None,
            "rates": {"gpu_usd_per_hour": 0.35, "disk_usd_per_hour": 0.02},
        },
    )
    sources = ("AMI", "AliMeeting", "AISHELL-5", "VoxConverse")
    train_bundle = _write_bundle(
        root / "training-v1",
        "speakrs-training-bundle-v1",
        [(f"train-{index}", sources[index % len(sources)]) for index in range(1_127)],
        release_sha256=release,
    )
    dev_bundle = _write_bundle(
        root / "dev-v1",
        "speakrs-frozen-dev-bundle-v1",
        [(recording_id, source) for source, values in frozen_splits.items() for recording_id in values],
        frozen_splits=frozen_splits,
    )
    image = root / "image.json"
    write_json(
        image,
        {
            "schema": "speakrs-main-training-image-v1",
            "source_commit": "a" * 40,
            "index_digest": "sha256:" + "b" * 64,
            "linux_amd64_manifest_digest": "sha256:" + "c" * 64,
            "anonymous_manifest_get_status": 200,
            "runtime_versions": {"python": "3.10.0", "torch": "2.11.0", "accelerate": "1.6.0"},
            "runtime_code_sha256": {
                "diarizen/trainer_dual_opt.py": "1" * 64,
                "diarizen/trainer_utils.py": "2" * 64,
                "recipes/diar_ssl/run_dual_opt.py": "3" * 64,
                "recipes/diar_ssl/trainer_dual_opt.py": "4" * 64,
                "recipes/diar_ssl/dataset.py": "5" * 64,
                "diarizen/models/eend/model_wavlm_conformer.py": "6" * 64,
                "recipes/speakrs/large/cli.py": "7" * 64,
                "recipes/speakrs/large/controller.py": "b" * 64,
                "recipes/speakrs/large/remote_backup.py": "c" * 64,
                "recipes/speakrs/large/training_admission.py": "8" * 64,
                "recipes/speakrs/large/training_supervisor.py": "9" * 64,
                "recipes/speakrs/large/vast_guard.py": "d" * 64,
                "recipes/speakrs/large_run.py": "a" * 64,
            },
        },
    )
    config = root / "trainer.toml"
    config.write_text(
        toml.dumps(
            {
                "meta": {"save_dir": "../speakrs/exp_main"},
                "finetune": {"finetune": False},
                "trainer": {
                    "path": "trainer_dual_opt.Trainer",
                    "args": {
                        "max_steps": 60_000,
                        "snapshot_every_updates": 2_000,
                        "max_update_checkpoints": 3,
                        "save_ckpt_interval": 0,
                        "max_num_checkpoints": 0,
                        "ranked_checkpoint_count": 5,
                        "gradient_accumulation_steps": 8,
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
                },
                "train_dataset": {
                    "path": "dataset.DiarizationDataset",
                    "args": {
                        "scp_file": "../speakrs/data/training-v1/wav.scp",
                        "rttm_file": "../speakrs/data/training-v1/all.rttm",
                        "uem_file": "../speakrs/data/training-v1/all.uem",
                        "chunk_size": 8,
                        "chunk_shift": 6,
                        "sample_rate": 16_000,
                    },
                    "dataloader": {
                        "batch_size": 8,
                        "num_workers": 4,
                        "drop_last": True,
                        "pin_memory": True,
                    },
                },
                "validate_dataset": {
                    "path": "dataset.DiarizationDataset",
                    "args": {
                        "scp_file": "../speakrs/data/dev-v1/wav.scp",
                        "rttm_file": "../speakrs/data/dev-v1/all.rttm",
                        "uem_file": "../speakrs/data/dev-v1/all.uem",
                        "chunk_size": 8,
                        "chunk_shift": 8,
                        "sample_rate": 16_000,
                    },
                    "dataloader": {
                        "batch_size": 2,
                        "num_workers": 4,
                        "drop_last": False,
                        "pin_memory": True,
                    },
                },
                "model": {
                    "path": "diarizen.models.eend.model_wavlm_conformer.Model",
                    "args": {
                        "wavlm_src": "../speakrs/artifacts/wavlm-large-torchaudio.pt",
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
                },
                "optimizer_small": {
                    "path": "torch.optim.AdamW",
                    "args": {
                        "lr": 1e-5,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                    },
                },
                "optimizer_big": {
                    "path": "torch.optim.AdamW",
                    "args": {
                        "lr": 1e-3,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    layout = root / "layout.json"
    write_json(
        layout,
        {
            "schema": "speakrs-training-worker-layout-v1",
            "training_root": "/opt/diarizen/recipes/diar_ssl",
            "trainer_config": "/opt/diarizen/recipes/speakrs/conf/trainer.toml",
            "train_bundle": "/opt/diarizen/recipes/speakrs/data/training-v1",
            "dev_bundle": "/opt/diarizen/recipes/speakrs/data/dev-v1",
            "initializer": "/opt/diarizen/recipes/speakrs/artifacts/wavlm-large-torchaudio.pt",
            "experiment_root": "/opt/diarizen/recipes/speakrs/exp_main",
            "checkpoint_root": "/opt/diarizen/recipes/speakrs/exp_main/trainer/checkpoints",
            "trusted_backup_root": "/trusted/checkpoints",
            "repository_root": "/opt/diarizen",
        },
    )
    return {
        "authorization": authorization,
        "qualification": qualification,
        "qualification_binding": qualification_binding,
        "offer": offer,
        "train_bundle": train_bundle,
        "dev_bundle": dev_bundle,
        "trainer_config": config,
        "initializer": initializer,
        "image_identity": image,
        "worker_layout": layout,
    }


def _freeze(
    paths: dict[str, Path],
    output: Path,
    predecessor_launch: Path | None = None,
    predecessor_receipt: Path | None = None,
):
    return freeze_training_launch(
        paths["authorization"],
        paths["qualification"],
        paths["qualification_binding"],
        paths["offer"],
        paths["train_bundle"],
        paths["dev_bundle"],
        paths["trainer_config"],
        paths["initializer"],
        paths["image_identity"],
        paths["worker_layout"],
        predecessor_launch,
        predecessor_receipt,
        output,
    )


def test_diagnostic_qualification_promotes_only_through_bounded_authorization(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    output = tmp_path / "launch.json"

    result = _freeze(paths, output)

    parsed = parse_launch_lock(read_json(output))
    assert result["launch_id"] == parsed["launch_id"]
    assert parsed["max_updates"] == 60_000
    assert parsed["physical_batch"] == 8
    assert parsed["artifacts"]["qualification"]["sha256"] == sha256_file(paths["qualification"])


def test_modified_qualification_cannot_use_existing_authorization(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    qualification = read_json(paths["qualification"])
    qualification["physical_batch"] = 4
    write_json(paths["qualification"], qualification)

    with pytest.raises(RuntimeGateError, match="different qualification report"):
        _freeze(paths, tmp_path / "launch.json")


def test_launch_rejects_different_release_and_tampered_bundle_payload(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    train_bundle = read_json(paths["train_bundle"])
    train_bundle["release_sha256"] = "f" * 64
    write_json(paths["train_bundle"], train_bundle)
    with pytest.raises(RuntimeGateError, match="different authorized release"):
        _freeze(paths, tmp_path / "wrong-release.json")

    train_bundle["release_sha256"] = "e" * 64
    write_json(paths["train_bundle"], train_bundle)
    audio = paths["train_bundle"].parent / train_bundle["recordings"][0]["audio_path"]
    audio.write_bytes(b"changed")
    with pytest.raises(RuntimeGateError, match="bundle audio differs"):
        _freeze(paths, tmp_path / "tampered-audio.json")


def test_launch_rejects_worker_output_path_mismatch(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    layout = read_json(paths["worker_layout"])
    layout["checkpoint_root"] = "/opt/diarizen/recipes/speakrs/exp_main/wrong/checkpoints"
    write_json(paths["worker_layout"], layout)

    with pytest.raises(RuntimeGateError, match="trainer output path"):
        _freeze(paths, tmp_path / "launch.json")


def test_launch_rejects_unqualified_initializer_and_model_capacity(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    paths["initializer"].write_bytes(b"different initializer")
    with pytest.raises(RuntimeGateError, match="qualified initializer"):
        _freeze(paths, tmp_path / "wrong-initializer.json")

    second = tmp_path / "second"
    second.mkdir()
    paths = _inputs(second)
    config = toml.load(paths["trainer_config"])
    config["model"]["args"]["max_speakers_per_chunk"] = 5
    paths["trainer_config"].write_text(toml.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeGateError, match="qualified training recipe"):
        _freeze(paths, tmp_path / "wrong-capacity.json")


def test_replacement_rental_preserves_run_identity_and_prior_spend(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    first_path = tmp_path / "first-launch.json"
    _freeze(paths, first_path)
    first = read_json(first_path)
    rented_at = datetime.fromisoformat(first["offer"]["rented_at"])
    ended_at = rented_at + timedelta(hours=4)
    spent = 4 * first["offer"]["rates"]["total_usd_per_hour"]
    receipt_path = tmp_path / "first-receipt.json"
    write_json(
        receipt_path,
        {
            "schema": "speakrs-rental-attempt-receipt-v1",
            "attempt_id": first["attempt_id"],
            "launch_id": first["launch_id"],
            "instance_id": first["offer"]["instance_id"],
            "rented_at": first["offer"]["rented_at"],
            "ended_at": ended_at.isoformat(),
            "rates": first["offer"]["rates"],
            "deletion_outcome": "destroyed",
            "spend_usd": spent,
        },
    )
    offer = read_json(paths["offer"])
    now = datetime.now(timezone.utc)
    offer.update(
        {
            "instance_id": "50520001",
            "rented_at": now.isoformat(),
            "hard_deadline": (now + timedelta(hours=2)).isoformat(),
            "destroy_at": (now + timedelta(hours=2, minutes=10)).isoformat(),
            "prior_spend_usd": spent,
            "resume_from_attempt_id": first["attempt_id"],
        }
    )
    write_json(paths["offer"], offer)
    second_path = tmp_path / "second-launch.json"

    _freeze(paths, second_path, first_path, receipt_path)

    second = read_json(second_path)
    assert second["launch_id"] == first["launch_id"]
    assert second["attempt_id"] != first["attempt_id"]
    assert second["recovery"]["predecessor_receipt"]["sha256"] == sha256_file(receipt_path)
