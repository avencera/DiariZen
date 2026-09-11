"""Tests for the frozen four-source development bundle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from recipes.speakrs.large.dev_bundle import _frozen_dev_ids, build_dev_bundle
from recipes.speakrs.large.errors import PreparationError


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


def test_dev_bundle_uses_frozen_ids_and_speaker_tiers(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    _write_audio(audio_root / "AMI" / "ami-dev.flac")
    _write_audio(audio_root / "AliMeeting" / "ali-dev.flac")
    established = tmp_path / "established"
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
    aishell = tmp_path / "aishell" / "001"
    _write_audio(aishell / "DX01C01.wav")
    aishell.joinpath("DX01C01.TextGrid").write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 1.0\n'
        'item [1]:\nclass = "IntervalTier"\nname = "P0247"\nxmin = 0\nxmax = 1.0\n'
        'intervals [1]:\nxmin = 0.25\nxmax = 0.75\ntext = "spoken words"\n',
        encoding="utf-8",
    )
    output = tmp_path / "dev"

    result = build_dev_bundle(
        _spec(),  # type: ignore[arg-type]
        audio_root,
        established,
        tmp_path / "aishell",
        "../speakrs/data/dev-v1",
        output,
    )

    manifest = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert result["recordings"] == 3
    assert [row["recording_id"] for row in manifest["recordings"]] == ["ami-dev", "ali-dev", "dev-001"]
    assert "SPEAKER dev-001 1 0.250000 0.500000 <NA> <NA> P0247" in (output / "all.rttm").read_text()
    assert "spoken words" not in (output / "all.rttm").read_text()
    uem = (output / "all.uem").read_text()
    assert "ami-dev 1 0.000000 0.900000" in uem
    assert "ali-dev 1 0.100000 0.950000" in uem


def test_dev_bundle_rejects_train_overlap() -> None:
    spec = _spec()
    spec.frozen_splits["AMI"]["train"] = ("ami-dev",)

    with pytest.raises(PreparationError, match="overlaps train or test"):
        _frozen_dev_ids(spec)  # type: ignore[arg-type]
