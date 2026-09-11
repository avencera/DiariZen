"""Resumable full-release restoration for WavLM Large training."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

from .contracts import DataPreparationSpec
from .data import _backend_from_spec, _qualification_release_context
from .errors import PreparationError
from .hashing import sha256_file
from .jsonio import read_json, write_json
from .storage import StorageBackend


def _download_training_object(store: StorageBackend, identity: Mapping[str, object], destination: Path) -> None:
    """Restore one content-addressed object through an atomic local path."""

    key = identity.get("key")
    digest = identity.get("sha256")
    size = identity.get("size")
    if not isinstance(key, str) or not isinstance(digest, str) or isinstance(size, bool) or not isinstance(size, int):
        raise PreparationError("training bundle object identity is invalid")
    if destination.is_file() and destination.stat().st_size == size and sha256_file(destination) == digest:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    actual = hashlib.sha256()
    restored_size = 0
    try:
        with temporary.open("wb") as handle:
            for chunk in store.iter_bytes(key, max_bytes=size):
                handle.write(chunk)
                actual.update(chunk)
                restored_size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if restored_size != size or actual.hexdigest() != digest:
        temporary.unlink(missing_ok=True)
        raise PreparationError(
            "training bundle object differs from the sealed release",
            {"key": key, "expected_size": size, "actual_size": restored_size},
        )
    temporary.replace(destination)


def _training_recordings(context: Mapping[str, object]) -> list[dict[str, object]]:
    """Return the exact audio, RTTM, and UEM triples in one sealed release."""

    release_by_key = context.get("release_by_key")
    if not isinstance(release_by_key, Mapping):
        raise PreparationError("training release has no object inventory")
    grouped: dict[tuple[str, str], dict[str, Mapping[str, object]]] = {}
    codec_fields = {"flac": "audio", "rttm": "rttm", "uem": "uem"}
    for value in release_by_key.values():
        if not isinstance(value, Mapping) or value.get("purpose") == "manifest":
            continue
        source = value.get("source")
        recording_id = value.get("parent_id")
        codec = value.get("codec")
        if not isinstance(source, str) or not isinstance(recording_id, str) or codec not in codec_fields:
            raise PreparationError("training release object cannot be assigned to a recording")
        field = codec_fields[str(codec)]
        row = grouped.setdefault((source, recording_id), {})
        if field in row:
            raise PreparationError(
                "training release has duplicate recording objects",
                {"source": source, "recording_id": recording_id, "field": field},
            )
        row[field] = value

    recordings: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for (source, recording_id), identities in sorted(grouped.items()):
        if set(identities) != {"audio", "rttm", "uem"}:
            raise PreparationError(
                "training release recording is incomplete",
                {"source": source, "recording_id": recording_id, "fields": sorted(identities)},
            )
        if recording_id in seen_ids:
            raise PreparationError("training recording ids are not globally unique", {"recording_id": recording_id})
        seen_ids.add(recording_id)
        recordings.append(
            {
                "recording_id": recording_id,
                "source": source,
                **{field: dict(identity) for field, identity in identities.items()},
            }
        )
    if not recordings:
        raise PreparationError("training release contains no recordings")
    return recordings


def training_bundle_data(
    spec: DataPreparationSpec,
    release: Path,
    seal_path: Path,
    restore_receipt_path: Path,
    wav_prefix: str,
    output: Path,
    *,
    backend: StorageBackend | None = None,
) -> dict[str, object]:
    """Restore every sealed training recording into a resumable bundle."""

    release = Path(release)
    seal_path = Path(seal_path)
    restore_receipt_path = Path(restore_receipt_path)
    output = Path(output)
    if not isinstance(wav_prefix, str) or not wav_prefix.strip():
        raise PreparationError("training bundle requires a non-empty trainer-relative wav prefix")
    if output.exists():
        raise PreparationError("training bundle output already exists", {"path": str(output)})

    store = _backend_from_spec(spec, backend)
    context, _ = _qualification_release_context(spec, release, seal_path, restore_receipt_path, store)
    recordings = _training_recordings(context)
    partial = output.with_name(f".{output.name}.partial")
    partial.mkdir(parents=True, exist_ok=True)
    state_path = partial / "restore-state.json"
    state_identity = {
        "schema": "speakrs-training-bundle-restore-v1",
        "release_sha256": context["release_sha256"],
        "restore_receipt_sha256": sha256_file(restore_receipt_path),
        "wav_prefix": wav_prefix,
        "recordings": len(recordings),
    }
    if state_path.is_file() and read_json(state_path) != state_identity:
        raise PreparationError("partial training bundle belongs to different inputs", {"path": str(partial)})
    write_json(state_path, state_identity)

    restored: list[dict[str, object]] = []
    wav_rows: list[str] = []
    rttm_rows: list[str] = []
    uem_rows: list[str] = []
    for index, row in enumerate(recordings, 1):
        source = str(row["source"])
        recording_id = str(row["recording_id"])
        local: dict[str, Path] = {}
        for field, directory in (("audio", "audio"), ("rttm", "labels"), ("uem", "labels")):
            identity = row[field]
            assert isinstance(identity, Mapping)
            codec = str(identity["codec"])
            destination = partial / directory / source / f"{identity['sha256']}.{codec}"
            _download_training_object(store, identity, destination)
            local[field] = destination

        audio_relative = local["audio"].relative_to(partial)
        wav_rows.append(f"{recording_id} {(Path(wav_prefix) / audio_relative).as_posix()}\n")
        for field, rows in (("rttm", rttm_rows), ("uem", uem_rows)):
            text = local[field].read_text(encoding="utf-8")
            rows.append(text if text.endswith("\n") else text + "\n")
        restored.append(
            {
                "recording_id": recording_id,
                "source": source,
                "audio_path": audio_relative.as_posix(),
                "audio_sha256": row["audio"]["sha256"],
                "audio_size": row["audio"]["size"],
                "rttm_sha256": row["rttm"]["sha256"],
                "uem_sha256": row["uem"]["sha256"],
            }
        )
        if index == 1 or index % 25 == 0 or index == len(recordings):
            write_json(partial / "progress.json", {**state_identity, "completed_recordings": index})

    manifest_paths = {"wav_scp": partial / "wav.scp", "rttm": partial / "all.rttm", "uem": partial / "all.uem"}
    manifest_paths["wav_scp"].write_text("".join(wav_rows), encoding="utf-8")
    manifest_paths["rttm"].write_text("".join(rttm_rows), encoding="utf-8")
    manifest_paths["uem"].write_text("".join(uem_rows), encoding="utf-8")
    manifest = {
        "schema": "speakrs-training-bundle-v1",
        "release_sha256": context["release_sha256"],
        "seal_sha256": sha256_file(seal_path),
        "restore_receipt_sha256": sha256_file(restore_receipt_path),
        "required_sources": list(context["required_sources"]),
        "wav_prefix": wav_prefix,
        "manifests": {name: sha256_file(path) for name, path in manifest_paths.items()},
        "recordings": restored,
    }
    write_json(partial / "bundle.json", manifest)
    state_path.unlink()
    (partial / "progress.json").unlink()
    partial.replace(output)
    return {
        "ok": True,
        "command": "training-bundle",
        "bundle": str(output),
        "bundle_manifest": str(output / "bundle.json"),
        "bundle_manifest_sha256": sha256_file(output / "bundle.json"),
        "recordings": len(restored),
        "sources": list(context["required_sources"]),
    }
