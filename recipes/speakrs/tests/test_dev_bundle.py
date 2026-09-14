"""Tests for the frozen four-source development bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from recipes.speakrs.large.dev_bundle import (
    EXPECTED_DEV_RECORDINGS,
    DevBundleFileRole,
    _frozen_dev_ids,
    build_dev_bundle,
    parse_dev_bundle_manifest,
    verify_dev_bundle,
)
from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.validation.contracts import Sha256Digest


def _write_audio(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000)


def _spec():
    return SimpleNamespace(
        frozen_splits={
            "AMI": {"train": ("ami-train",), "dev": ("ami-dev",), "test": ()},
            "AliMeeting": {"train": (), "dev": ("ali-dev",), "test": ("ali-test",)},
            "VoxConverse": {"train": ("vox-train",), "dev": (), "test": ("vox-test",)},
            "AISHELL-5": {"train": ("train-001",), "dev": ("dev-001",), "test": ("eval1-001",)},
        }
    )


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    audio_root = root / "audio"
    _write_audio(audio_root / "AMI" / "ami-dev.flac")
    _write_audio(audio_root / "AliMeeting" / "ali-dev.flac")
    established = root / "established"
    established.mkdir()
    established.joinpath("rttm").write_text(
        "SPEAKER ami-dev 1 0.000000 0.500000 <NA> <NA> ami-speaker <NA> <NA>\n"
        "SPEAKER ali-dev 1 0.100000 0.500000 <NA> <NA> ali-speaker <NA> <NA>\n",
        encoding="utf-8",
    )
    established.joinpath("all.uem").write_text(
        "ami-dev 1 0.000000 0.900000\nali-dev 1 0.100000 0.950000\n",
        encoding="utf-8",
    )
    aishell_root = root / "aishell"
    aishell = aishell_root / "001"
    _write_audio(aishell / "DX01C01.wav")
    aishell.joinpath("DX01C01.TextGrid").write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 1.0\n'
        'item [1]:\nclass = "IntervalTier"\nname = "P0247"\nxmin = 0\nxmax = 1.0\n'
        'intervals [1]:\nxmin = 0.25\nxmax = 0.75\ntext = "spoken words"\n',
        encoding="utf-8",
    )
    return audio_root, established, aishell_root


def _build(root: Path, prefix: str = "../speakrs/data/dev-v1", name: str = "dev") -> Path:
    audio_root, established, aishell = _inputs(root / f"inputs-{name}")
    output = root / name
    build_dev_bundle(_spec(), audio_root, established, aishell, prefix, output)
    return output


def _forty_four_recording_payloads() -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    sources = ("AMI",) * 15 + ("AliMeeting",) * 15 + ("AISHELL-5",) * 14
    for index, source in enumerate(sources):
        recording_id = f"dev-{index:03d}"
        logical_id = f"{source}:{recording_id}"
        files = []
        for role in DevBundleFileRole:
            content = f"{logical_id}:{role.value}".encode()
            files.append(
                {
                    "logical_id": f"{logical_id}:{role.value}",
                    "path": f"{role.value}/{source}/{index:03d}-{role.value}",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "byte_length": len(content),
                    "role": role.value,
                }
            )
        payloads.append(
            {
                "logical_id": logical_id,
                "recording_id": recording_id,
                "source": source,
                "files": files,
            }
        )
    return payloads


def test_dev_bundle_uses_frozen_ids_and_speaker_tiers(tmp_path: Path) -> None:
    audio_root, established, aishell_root = _inputs(tmp_path)
    output = tmp_path / "dev"

    result = build_dev_bundle(
        _spec(),  # type: ignore[arg-type]
        audio_root,
        established,
        aishell_root,
        "../speakrs/data/dev-v1",
        output,
    )

    manifest = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    parsed = parse_dev_bundle_manifest(manifest, expected_recordings=None)
    assert result["recordings"] == 3
    assert isinstance(parsed.recordings[0].files[0].sha256, Sha256Digest)
    assert [row["recording_id"] for row in manifest["recordings"]] == ["ami-dev", "ali-dev", "dev-001"]
    assert "SPEAKER dev-001 1 0.250000 0.500000 <NA> <NA> P0247" in (output / "all.rttm").read_text()
    assert "spoken words" not in (output / "all.rttm").read_text()
    uem = (output / "all.uem").read_text()
    assert "ami-dev 1 0.000000 0.900000" in uem
    assert "ali-dev 1 0.100000 0.950000" in uem


def test_moving_or_changing_path_view_preserves_identity(tmp_path: Path) -> None:
    first = _build(tmp_path, "../first-root", "first")
    second = _build(tmp_path, "../second-root", "second")
    first_manifest = json.loads((first / "bundle.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "bundle.json").read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    assert "wav_prefix" not in first_manifest
    first_paths = json.loads((first / "bundle.paths.json").read_text(encoding="utf-8"))
    second_paths = json.loads((second / "bundle.paths.json").read_text(encoding="utf-8"))
    assert first_paths["wav_prefix"] != second_paths["wav_prefix"]
    moved = tmp_path / "moved"
    shutil.copytree(first, moved)
    first_identity = verify_dev_bundle(first, expected_recordings=None)["identity_sha256"]
    assert verify_dev_bundle(moved, expected_recordings=None)["identity_sha256"] == first_identity


def test_verifier_rejects_one_byte_corruption(tmp_path: Path) -> None:
    output = _build(tmp_path)
    audio_path = next(path for path in (output / "audio").rglob("*") if path.is_file())
    data = bytearray(audio_path.read_bytes())
    data[0] ^= 1
    audio_path.write_bytes(data)
    with pytest.raises(PreparationError, match="differs from its identity"):
        verify_dev_bundle(output, expected_recordings=None)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update({"unknown": True}), "unknown fields"),
        (lambda payload: payload["recordings"][0]["files"][0].update({"path": "../escape"}), "traversal"),
        (lambda payload: payload["recordings"][0]["files"][0].update({"sha256": "x" * 64}), "SHA-256"),
        (lambda payload: payload["recordings"][0]["files"][0].update({"byte_length": -1}), "non-negative"),
        (lambda payload: payload["recordings"][0]["files"].pop(), "exactly one"),
        (lambda payload: payload["recordings"].append(payload["recordings"][0]), "duplicate recording"),
    ),
)
def test_manifest_rejects_malformed_identity(tmp_path: Path, mutation, message: str) -> None:
    payload = json.loads((_build(tmp_path) / "bundle.json").read_text(encoding="utf-8"))
    mutation(payload)
    with pytest.raises(PreparationError, match=message):
        parse_dev_bundle_manifest(payload)


def test_manifest_rejects_duplicate_file_path(tmp_path: Path) -> None:
    payload = json.loads((_build(tmp_path) / "bundle.json").read_text(encoding="utf-8"))
    files = payload["recordings"][0]["files"]
    files[1]["path"] = files[0]["path"]
    with pytest.raises(PreparationError, match="duplicate file paths"):
        parse_dev_bundle_manifest(payload)


def test_expected_forty_four_recording_contract_is_ordered(tmp_path: Path) -> None:
    recordings = _forty_four_recording_payloads()
    forward = {"schema": "speakrs-frozen-dev-bundle-v2", "recordings": recordings}
    reverse = {"schema": "speakrs-frozen-dev-bundle-v2", "recordings": list(reversed(recordings))}
    first = parse_dev_bundle_manifest(forward, expected_recordings=EXPECTED_DEV_RECORDINGS)
    second = parse_dev_bundle_manifest(reverse, expected_recordings=EXPECTED_DEV_RECORDINGS)
    assert len(first.recordings) == EXPECTED_DEV_RECORDINGS
    assert first.identity_sha256() == second.identity_sha256()
    assert [recording.source for recording in first.recordings[:15]] == ["AMI"] * 15
    assert all({file.role for file in recording.files} == set(DevBundleFileRole) for recording in first.recordings)


def test_dev_bundle_rejects_train_overlap() -> None:
    spec = _spec()
    spec.frozen_splits["AMI"]["train"] = ("ami-dev",)

    with pytest.raises(PreparationError, match="overlaps train or test"):
        _frozen_dev_ids(spec)  # type: ignore[arg-type]
