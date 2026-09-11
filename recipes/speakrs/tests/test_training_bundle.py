"""Tests for the complete sealed-release trainer bundle."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from recipes.speakrs.large import training_bundle as bundle_module
from recipes.speakrs.large.errors import UnresolvedInputError
from recipes.speakrs.large.hashing import sha256_bytes, sha256_file


class InterruptibleBackend:
    """Small streaming backend that can interrupt one object before yielding."""

    def __init__(self, objects: dict[str, bytes], fail_once: str | None = None):
        self.objects = objects
        self.fail_once = fail_once
        self.calls: Counter[str] = Counter()

    def iter_bytes(self, key: str, *, chunk_size: int = 1024 * 1024, max_bytes: int | None = None):
        del chunk_size, max_bytes
        self.calls[key] += 1
        if self.fail_once == key:
            self.fail_once = None
            raise UnresolvedInputError("injected transfer interruption")
        yield self.objects[key]


def _fixture_context() -> tuple[dict[str, object], dict[str, bytes]]:
    objects: dict[str, bytes] = {}
    inventory: dict[str, dict[str, object]] = {}
    for source, recording_id in (("AMI", "ES2002a"), ("VoxConverse", "dev_001")):
        for codec, purpose, payload in (
            ("flac", "train-audio", f"audio-{recording_id}".encode()),
            ("rttm", "train-label", f"SPEAKER {recording_id} 1 0 1 <NA> <NA> spk <NA> <NA>\n".encode()),
            ("uem", "train-label", f"{recording_id} 1 0 1\n".encode()),
        ):
            digest = sha256_bytes(payload)
            key = f"datasets/test/{source}/{recording_id}/{digest}.{codec}"
            objects[key] = payload
            inventory[key] = {
                "key": key,
                "sha256": digest,
                "size": len(payload),
                "codec": codec,
                "purpose": purpose,
                "source": source,
                "parent_id": recording_id,
            }
    return (
        {
            "release_sha256": hashlib.sha256(b"release").hexdigest(),
            "release_by_key": inventory,
            "required_sources": ["AMI", "VoxConverse"],
        },
        objects,
    )


def test_training_bundle_restores_every_recording_and_resumes(monkeypatch, tmp_path: Path) -> None:
    context, objects = _fixture_context()
    backend = InterruptibleBackend(objects, fail_once=next(key for key in objects if "dev_001" in key))
    seal = tmp_path / "seal.json"
    receipt = tmp_path / "restore.json"
    seal.write_text("{}\n", encoding="utf-8")
    receipt.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "bundle"
    monkeypatch.setattr(bundle_module, "_backend_from_spec", lambda _spec, supplied: supplied)
    monkeypatch.setattr(bundle_module, "_qualification_release_context", lambda *_args: (context, {}))

    with pytest.raises(UnresolvedInputError, match="interruption"):
        bundle_module.training_bundle_data(
            object(),  # type: ignore[arg-type]
            tmp_path / "release",
            seal,
            receipt,
            "../speakrs/data/training-v1",
            output,
            backend=backend,
        )

    calls_before_resume = dict(backend.calls)
    result = bundle_module.training_bundle_data(
        object(),  # type: ignore[arg-type]
        tmp_path / "release",
        seal,
        receipt,
        "../speakrs/data/training-v1",
        output,
        backend=backend,
    )

    manifest = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert result["recordings"] == 2
    assert len(manifest["recordings"]) == 2
    assert (output / "wav.scp").read_text(encoding="utf-8").splitlines() == [
        f"ES2002a ../speakrs/data/training-v1/audio/AMI/{sha256_bytes(b'audio-ES2002a')}.flac",
        f"dev_001 ../speakrs/data/training-v1/audio/VoxConverse/{sha256_bytes(b'audio-dev_001')}.flac",
    ]
    assert manifest["manifests"]["wav_scp"] == sha256_file(output / "wav.scp")
    assert not (tmp_path / ".bundle.partial").exists()
    first_audio = next(key for key in objects if "ES2002a" in key and key.endswith(".flac"))
    assert backend.calls[first_audio] == calls_before_resume[first_audio]


def test_training_bundle_rejects_duplicate_recording_ids_across_sources() -> None:
    context, _ = _fixture_context()
    inventory = context["release_by_key"]
    assert isinstance(inventory, dict)
    for item in inventory.values():
        if item["source"] == "VoxConverse":
            item["parent_id"] = "ES2002a"

    with pytest.raises(bundle_module.PreparationError, match="globally unique"):
        bundle_module._training_recordings(context)
