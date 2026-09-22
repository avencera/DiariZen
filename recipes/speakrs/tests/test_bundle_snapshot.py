"""Focused tests for immutable trainer-bundle snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recipes.speakrs.large import bundle_snapshot
from recipes.speakrs.large.bundle_snapshot import (
    _publish_absent_bytes,
    _safe_relative_path,
    publish_bundle_snapshot,
    restore_bundle_snapshot,
    verify_bundle_snapshot,
)
from recipes.speakrs.large.contracts import ObjectStoreDestination
from recipes.speakrs.large.errors import ContractError, PreparationError
from recipes.speakrs.large.hashing import sha256_file
from recipes.speakrs.large.storage import validate_content_addressed_write


class MemoryStore:
    """Small private object store used by snapshot contract tests."""

    def __init__(self, destination: ObjectStoreDestination | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.destination = destination

    def exists(self, key: str) -> bool:
        return key in self.objects

    def head(self, key: str) -> dict[str, object]:
        return {"size": len(self.objects[key])}

    def put_bytes(self, key: str, payload: bytes, *, content_type: str = "application/octet-stream") -> dict[str, str]:
        if self.destination is not None:
            validate_content_addressed_write(self.destination, key, payload, content_type=content_type)
        self.objects[key] = payload
        return {"key": key, "etag": hashlib.md5(payload, usedforsecurity=False).hexdigest()}

    def iter_bytes(self, key: str, *, chunk_size: int = 1024 * 1024, max_bytes: int | None = None):
        payload = self.objects[key]
        assert max_bytes is None or len(payload) <= max_bytes
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]


def _destination() -> ObjectStoreDestination:
    return ObjectStoreDestination(
        provider="r2",
        endpoint="https://example.invalid",
        bucket="private",
        prefix="datasets/diarization-data-verification",
        credential_reference="wrangler",
    )


def _bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    bundle = tmp_path / "indic-diarbench-union-v1"
    (bundle / "audio" / "source").mkdir(parents=True)
    (bundle / "audio" / "source" / "first.flac").write_bytes(b"first-audio")
    (bundle / "audio" / "source" / "second.flac").write_bytes(b"second-audio")
    (bundle / "wav.scp").write_text(
        "first /opt/diarizen/recipes/speakrs/data/indic-diarbench-union-v1/audio/source/first.flac\n",
        encoding="utf-8",
    )
    (bundle / "all.rttm").write_text("SPEAKER first 1 0 1 <NA> <NA> spk <NA> <NA>\n", encoding="utf-8")
    (bundle / "all.uem").write_text("first 1 0 1\n", encoding="utf-8")
    (bundle / "bundle.json").write_text('{"schema":"speakrs-training-bundle-v1"}\n', encoding="utf-8")
    descriptor = tmp_path / "indic-diarbench-next-run-input-v1.json"
    descriptor.write_text('{"training_started":false}\n', encoding="utf-8")
    return bundle, descriptor, sha256_file(bundle / "bundle.json")


def _set_wav_target(bundle: Path, restore_parent: Path) -> None:
    (bundle / "wav.scp").write_text(
        f"first {restore_parent.as_posix()}/{bundle.name}/audio/source/first.flac\n",
        encoding="utf-8",
    )


def test_publish_and_restore_preserve_exact_layout_and_identities(monkeypatch, tmp_path: Path) -> None:
    bundle, descriptor, bundle_identity = _bundle(tmp_path)
    publication = tmp_path / "publication.json"
    restore_parent = tmp_path / "trainer" / "data"
    _set_wav_target(bundle, restore_parent)
    store = MemoryStore(_destination())

    result = publish_bundle_snapshot(
        _destination(),
        bundle,
        descriptor,
        bundle_identity,
        restore_parent,
        publication,
        temporary_root=tmp_path / "temporary",
        config_path=tmp_path / "config.json",
        max_shard_bytes=20 * 1024,
        backend=store,
    )

    manifest = json.loads(publication.read_text(encoding="utf-8"))
    assert result["shards"] >= 1
    assert manifest["bundle"]["bundle_manifest_sha256"] == bundle_identity
    assert manifest["restore"]["parent"] == str(restore_parent)
    monkeypatch.setattr(bundle_snapshot, "assert_private_access", lambda *_args: {"private": True})
    verification = verify_bundle_snapshot(
        _destination(), result["manifest_key"], tmp_path / "verification.json", backend=store
    )
    assert verification["privacy"] == {"private": True}

    restore = restore_bundle_snapshot(
        _destination(),
        result["manifest_key"],
        restore_parent,
        tmp_path / "restore.json",
        backend=store,
    )

    restored_bundle = Path(restore["bundle"])
    assert sha256_file(restored_bundle / "bundle.json") == bundle_identity
    assert (restored_bundle / "wav.scp").read_bytes() == (bundle / "wav.scp").read_bytes()
    assert (restore_parent / descriptor.name).read_bytes() == descriptor.read_bytes()


@pytest.mark.parametrize("value", ("../escape", "/absolute", "safe/../../escape", "."))
def test_snapshot_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(ContractError, match="safe relative path"):
        _safe_relative_path(value, "test path")


def test_snapshot_rejects_symbolic_links(tmp_path: Path) -> None:
    bundle, descriptor, bundle_identity = _bundle(tmp_path)
    _set_wav_target(bundle, tmp_path / "restore")
    (bundle / "linked").symlink_to(bundle / "wav.scp")

    with pytest.raises(PreparationError, match="symbolic links"):
        publish_bundle_snapshot(
            _destination(),
            bundle,
            descriptor,
            bundle_identity,
            tmp_path / "restore",
            tmp_path / "publication.json",
            temporary_root=tmp_path / "temporary",
            config_path=tmp_path / "config.json",
            backend=MemoryStore(),
        )


def test_existing_different_object_is_a_conflict() -> None:
    store = MemoryStore()
    store.objects["snapshot/object"] = b"different"

    with pytest.raises(PreparationError, match="different bytes"):
        _publish_absent_bytes(store, "snapshot/object", b"expected", content_type="application/octet-stream")
    assert store.objects["snapshot/object"] == b"different"


def test_restore_requires_the_recorded_wav_parent(tmp_path: Path) -> None:
    bundle, descriptor, bundle_identity = _bundle(tmp_path)
    expected_parent = tmp_path / "expected"
    _set_wav_target(bundle, expected_parent)
    store = MemoryStore(_destination())
    result = publish_bundle_snapshot(
        _destination(),
        bundle,
        descriptor,
        bundle_identity,
        expected_parent,
        tmp_path / "publication.json",
        temporary_root=tmp_path / "temporary",
        config_path=tmp_path / "config.json",
        backend=store,
    )

    with pytest.raises(PreparationError, match="wav.scp"):
        restore_bundle_snapshot(
            _destination(),
            result["manifest_key"],
            tmp_path / "wrong",
            tmp_path / "restore.json",
            backend=store,
        )
