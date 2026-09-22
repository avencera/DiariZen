"""Immutable, restore-ready snapshots of complete trainer bundles."""

from __future__ import annotations

import hashlib
import io
import os
import shlex
import shutil
import tarfile
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from .contracts import ObjectStoreDestination, parse_r2_destination
from .errors import ContractError, PreparationError
from .hashing import sha256_file, sha256_json
from .jsonio import write_json
from .storage import StorageBackend, assert_private_access, backend_from_destination


SNAPSHOT_SCHEMA = "speakrs-training-bundle-snapshot-v1"
DEFAULT_SHARD_BYTES = 240 * 1024**2
TAR_BLOCK_BYTES = 512
TAR_END_BYTES = 2 * TAR_BLOCK_BYTES


class SnapshotStore(StorageBackend, Protocol):
    """Object-store operations required by bundle snapshots."""

    def exists(self, key: str) -> bool:
        """Return whether a key exists."""

    def head(self, key: str) -> dict[str, object]:
        """Return object metadata without downloading its body."""

    def put_bytes(self, key: str, payload: bytes, *, content_type: str = "application/octet-stream") -> dict[str, str]:
        """Write one absent object."""


def load_snapshot_destination(path: Path) -> ObjectStoreDestination:
    """Load only the R2 boundary from a full data-preparation document."""

    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ContractError("snapshot configuration must be a JSON object")
    return parse_r2_destination(value.get("r2"))


@dataclass(frozen=True)
class SnapshotFile:
    """One regular source file with a safe archive path."""

    source: Path
    archive_path: PurePosixPath
    size: int


@dataclass(frozen=True)
class SnapshotShard:
    """One immutable tar shard in the snapshot."""

    key: str
    sha256: str
    size: int
    files: int

    def to_json(self) -> dict[str, object]:
        """Return the stable manifest representation."""

        return {"key": self.key, "sha256": self.sha256, "size": self.size, "files": self.files}


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError(f"{label} must be a safe relative path", {label: value})
    return path


def _snapshot_files(bundle: Path, launch_descriptor: Path) -> tuple[SnapshotFile, ...]:
    """Inventory regular files without following links or crossing roots."""

    bundle = Path(bundle)
    launch_descriptor = Path(launch_descriptor)
    if bundle.is_symlink():
        raise PreparationError("training bundle root cannot be a symbolic link", {"path": str(bundle)})
    if launch_descriptor.is_symlink():
        raise PreparationError("launch descriptor cannot be a symbolic link", {"path": str(launch_descriptor)})
    bundle = bundle.resolve()
    launch_descriptor = launch_descriptor.resolve()
    if not bundle.is_dir():
        raise PreparationError("training bundle is not a directory", {"path": str(bundle)})
    if not launch_descriptor.is_file():
        raise PreparationError("launch descriptor must be a regular file", {"path": str(launch_descriptor)})
    if launch_descriptor.parent != bundle.parent:
        raise PreparationError("launch descriptor must be adjacent to the training bundle")
    _safe_relative_path(bundle.name, "bundle name")
    _safe_relative_path(launch_descriptor.name, "launch descriptor name")

    files: list[SnapshotFile] = []
    for root, directories, names in os.walk(bundle, followlinks=False):
        directories.sort()
        names.sort()
        root_path = Path(root)
        for name in directories:
            if (root_path / name).is_symlink():
                raise PreparationError(
                    "training bundle cannot contain symbolic links", {"path": str(root_path / name)}
                )
        for name in names:
            source = root_path / name
            if source.is_symlink():
                raise PreparationError("training bundle cannot contain symbolic links", {"path": str(source)})
            if not source.is_file():
                raise PreparationError("training bundle can contain only regular files", {"path": str(source)})
            relative = source.relative_to(bundle)
            archive_path = _safe_relative_path((PurePosixPath(bundle.name) / relative.as_posix()).as_posix(), "file")
            files.append(SnapshotFile(source, archive_path, source.stat().st_size))
    files.append(
        SnapshotFile(launch_descriptor, PurePosixPath(launch_descriptor.name), launch_descriptor.stat().st_size)
    )
    return tuple(sorted(files, key=lambda item: item.archive_path.as_posix()))


def _tar_member_bytes(size: int) -> int:
    return TAR_BLOCK_BYTES + ((size + TAR_BLOCK_BYTES - 1) // TAR_BLOCK_BYTES) * TAR_BLOCK_BYTES


def _group_shards(files: tuple[SnapshotFile, ...], max_shard_bytes: int) -> tuple[tuple[SnapshotFile, ...], ...]:
    if max_shard_bytes <= TAR_END_BYTES + TAR_BLOCK_BYTES:
        raise ContractError("snapshot shard limit is too small")
    groups: list[tuple[SnapshotFile, ...]] = []
    current: list[SnapshotFile] = []
    current_bytes = TAR_END_BYTES
    for item in files:
        member_bytes = _tar_member_bytes(item.size)
        if member_bytes + TAR_END_BYTES > max_shard_bytes:
            raise PreparationError(
                "one bundle file exceeds the snapshot shard limit",
                {"path": item.archive_path.as_posix(), "size": item.size, "max_shard_bytes": max_shard_bytes},
            )
        if current and current_bytes + member_bytes > max_shard_bytes:
            groups.append(tuple(current))
            current = []
            current_bytes = TAR_END_BYTES
        current.append(item)
        current_bytes += member_bytes
    if current:
        groups.append(tuple(current))
    if not groups:
        raise PreparationError("training bundle snapshot cannot be empty")
    return tuple(groups)


def _write_tar_shard(path: Path, files: tuple[SnapshotFile, ...]) -> None:
    """Write one deterministic, uncompressed tar shard."""

    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        for item in files:
            info = tarfile.TarInfo(item.archive_path.as_posix())
            info.size = item.size
            info.mode = item.source.stat().st_mode & 0o777
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with item.source.open("rb") as source:
                archive.addfile(info, source)


def _tree_identity(root: Path) -> tuple[str, int, int]:
    """Hash every regular file identity and its safe relative path."""

    rows: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PreparationError("restored bundle contains a symbolic link", {"path": str(path)})
        if not path.is_file():
            continue
        relative = _safe_relative_path(path.relative_to(root).as_posix(), "restored file")
        size = path.stat().st_size
        rows.append({"path": relative.as_posix(), "size": size, "sha256": sha256_file(path)})
        total_bytes += size
    return sha256_json(rows), len(rows), total_bytes


def _tar_file_identities(path: Path) -> list[dict[str, object]]:
    """Hash member bytes from the exact packaged tar stream."""

    rows: list[dict[str, object]] = []
    with tarfile.open(path, "r:") as archive:
        for member in _safe_tar_members(archive):
            source = archive.extractfile(member)
            if source is None:
                raise PreparationError("snapshot tar member has no readable body", {"member": member.name})
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            if size != member.size:
                raise PreparationError("snapshot tar member size changed while packaging", {"member": member.name})
            rows.append({"path": member.name, "size": size, "sha256": digest.hexdigest()})
    return rows


def _remote_identity(store: SnapshotStore, key: str, *, max_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in store.iter_bytes(key, max_bytes=max_bytes):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _publish_absent_bytes(
    store: SnapshotStore,
    key: str,
    payload: bytes,
    *,
    content_type: str,
) -> str:
    """Create one absent object or prove that its existing bytes match."""

    expected = hashlib.sha256(payload).hexdigest()
    if store.exists(key):
        metadata = store.head(key)
        try:
            remote_size = int(metadata.get("size", -1))
        except (TypeError, ValueError):
            remote_size = -1
        if remote_size != len(payload):
            raise PreparationError(
                "snapshot key already contains different bytes",
                {"key": key, "expected_size": len(payload), "actual_size": remote_size},
            )
        actual, size = _remote_identity(store, key, max_bytes=len(payload))
        if actual != expected or size != len(payload):
            raise PreparationError(
                "snapshot key already contains different bytes",
                {"key": key, "expected_sha256": expected, "actual_sha256": actual},
            )
        return "already-present"
    store.put_bytes(key, payload, content_type=content_type)
    return "created"


def _object_key(destination: ObjectStoreDestination, bundle_sha256: str, digest: str, extension: str) -> str:
    return (
        f"{destination.prefix}/datasets/training-bundle-snapshot/{bundle_sha256}/"
        f"restore-ready/objects/{digest}.{extension}"
    )


def resolve_snapshot_manifest_key(
    destination: ObjectStoreDestination,
    bundle_sha256: str,
    *,
    backend: SnapshotStore | None = None,
) -> str:
    """Resolve the only immutable manifest for one bundle identity."""

    store = backend or backend_from_destination(destination)
    prefix = f"{destination.prefix}/datasets/training-bundle-snapshot/{bundle_sha256}/restore-ready/objects"
    keys = [key for key in store.list_prefix(prefix, max_keys=10_000) if key.endswith(".json")]
    if len(keys) != 1:
        raise PreparationError(
            "bundle snapshot must have exactly one manifest",
            {"bundle_sha256": bundle_sha256, "manifest_count": len(keys)},
        )
    return keys[0]


def _validate_wav_restore_target(bundle: Path, restore_parent: Path) -> None:
    """Prove that every absolute wav path uses the recorded restore target."""

    expected = (restore_parent / bundle.name).as_posix().rstrip("/") + "/"
    wav_scp = bundle / "wav.scp"
    rows = 0
    with wav_scp.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split(maxsplit=1)
            if len(fields) != 2 or not fields[1].startswith(expected):
                raise PreparationError(
                    "wav.scp path does not match the requested restore target",
                    {"line": line_number, "expected_prefix": expected},
                )
            rows += 1
    if rows == 0:
        raise PreparationError("wav.scp cannot be empty")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _manifest_payload(manifest: Mapping[str, object]) -> bytes:
    buffer = io.StringIO()
    import json

    json.dump(manifest, buffer, indent=2, sort_keys=True)
    buffer.write("\n")
    return buffer.getvalue().encode("utf-8")


def publish_bundle_snapshot(
    destination: ObjectStoreDestination,
    bundle: Path,
    launch_descriptor: Path,
    expected_bundle_sha256: str,
    restore_parent: Path,
    output: Path,
    *,
    temporary_root: Path,
    config_path: Path,
    max_shard_bytes: int = DEFAULT_SHARD_BYTES,
    backend: SnapshotStore | None = None,
) -> dict[str, object]:
    """Package and publish an exact bundle through bounded uncompressed shards."""

    expected_bundle_sha256 = expected_bundle_sha256.strip().lower()
    if len(expected_bundle_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_bundle_sha256
    ):
        raise ContractError("expected bundle identity must be a SHA-256 digest")
    bundle_manifest = bundle / "bundle.json"
    if sha256_file(bundle_manifest) != expected_bundle_sha256:
        raise PreparationError("bundle manifest does not match the expected bundle identity")
    if _path_is_within(output, bundle) or output.resolve() == launch_descriptor.resolve():
        raise PreparationError("snapshot manifest output cannot modify the source bundle")
    _validate_wav_restore_target(bundle, restore_parent)
    descriptor_sha256 = sha256_file(launch_descriptor)
    files = _snapshot_files(bundle, launch_descriptor)
    groups = _group_shards(files, max_shard_bytes)
    store = backend or backend_from_destination(destination)
    temporary_root.mkdir(parents=True, exist_ok=True)
    shards: list[SnapshotShard] = []
    outcomes: list[str] = []
    packaged_files: list[dict[str, object]] = []
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="speakrs-bundle-snapshot-", dir=temporary_root) as temp_directory:
        temp = Path(temp_directory)
        for index, group in enumerate(groups, 1):
            shard_path = temp / f"shard-{index:04d}.tar"
            _write_tar_shard(shard_path, group)
            packaged_files.extend(_tar_file_identities(shard_path))
            digest = sha256_file(shard_path)
            size = shard_path.stat().st_size
            if size > max_shard_bytes:
                raise PreparationError(
                    "packaged tar exceeds the snapshot shard limit",
                    {"size": size, "max_shard_bytes": max_shard_bytes},
                )
            key = _object_key(destination, expected_bundle_sha256, digest, "tar")
            payload = shard_path.read_bytes()
            outcomes.append(_publish_absent_bytes(store, key, payload, content_type="application/x-tar"))
            shards.append(SnapshotShard(key, digest, size, len(group)))
            shard_path.unlink()

    bundle_prefix = f"{bundle.name}/"
    bundle_rows = [
        {"path": str(row["path"])[len(bundle_prefix) :], "size": row["size"], "sha256": row["sha256"]}
        for row in packaged_files
        if str(row["path"]).startswith(bundle_prefix)
    ]
    descriptor_rows = [row for row in packaged_files if row["path"] == launch_descriptor.name]
    if len(bundle_rows) + len(descriptor_rows) != len(packaged_files) or len(descriptor_rows) != 1:
        raise PreparationError("packaged snapshot layout differs from the requested bundle")
    bundle_rows.sort(key=lambda row: str(row["path"]))
    bundle_tree_sha256 = sha256_json(bundle_rows)
    bundle_file_count = len(bundle_rows)
    bundle_bytes = sum(int(row["size"]) for row in bundle_rows)
    if descriptor_rows[0]["sha256"] != descriptor_sha256:
        raise PreparationError("launch descriptor changed while packaging")
    packaged_bundle_manifests = [row for row in bundle_rows if row["path"] == "bundle.json"]
    if len(packaged_bundle_manifests) != 1 or packaged_bundle_manifests[0]["sha256"] != expected_bundle_sha256:
        raise PreparationError("bundle manifest changed while packaging")
    restore_command = (
        "python -m recipes.speakrs.large.cli data bundle-snapshot-restore "
        f"--bundle-sha256 {expected_bundle_sha256} "
        f"--config {shlex.quote(str(config_path))} --restore-parent {shlex.quote(str(restore_parent))}"
    )
    manifest: dict[str, object] = {
        "schema": SNAPSHOT_SCHEMA,
        "bundle": {
            "directory": bundle.name,
            "bundle_manifest_sha256": expected_bundle_sha256,
            "tree_sha256": bundle_tree_sha256,
            "files": bundle_file_count,
            "bytes": bundle_bytes,
        },
        "launch_descriptor": {
            "name": launch_descriptor.name,
            "sha256": descriptor_sha256,
            "size": launch_descriptor.stat().st_size,
        },
        "restore": {"parent": str(restore_parent), "command": restore_command},
        "shards": [shard.to_json() for shard in shards],
        "total_shard_bytes": sum(shard.size for shard in shards),
    }
    manifest_bytes = _manifest_payload(manifest)
    manifest_key = _object_key(
        destination,
        expected_bundle_sha256,
        hashlib.sha256(manifest_bytes).hexdigest(),
        "json",
    )
    manifest_outcome = _publish_absent_bytes(store, manifest_key, manifest_bytes, content_type="application/json")
    write_json(output, manifest)
    return {
        "ok": True,
        "command": "bundle-snapshot-publish",
        "manifest_key": manifest_key,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "shards": len(shards),
        "shard_bytes": sum(shard.size for shard in shards),
        "bundle_bytes": bundle_bytes,
        "created_shards": outcomes.count("created"),
        "existing_shards": outcomes.count("already-present"),
        "manifest_outcome": manifest_outcome,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _read_remote_json(store: SnapshotStore, key: str) -> tuple[dict[str, object], bytes]:
    payload = b"".join(store.iter_bytes(key, max_bytes=16 * 1024**2))
    import json

    value = json.loads(payload)
    if not isinstance(value, dict):
        raise PreparationError("snapshot manifest must be a JSON object")
    return value, payload


def _parse_manifest(value: Mapping[str, object]) -> tuple[dict[str, object], list[SnapshotShard]]:
    if value.get("schema") != SNAPSHOT_SCHEMA:
        raise PreparationError("snapshot manifest schema is invalid")
    raw_shards = value.get("shards")
    bundle = value.get("bundle")
    descriptor = value.get("launch_descriptor")
    restore = value.get("restore")
    if not isinstance(bundle, Mapping) or not isinstance(descriptor, Mapping) or not isinstance(restore, Mapping):
        raise PreparationError("snapshot manifest restore identity is invalid")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise PreparationError("snapshot manifest has no shards")
    shards: list[SnapshotShard] = []
    for index, raw in enumerate(raw_shards, 1):
        if not isinstance(raw, Mapping):
            raise PreparationError("snapshot shard identity is invalid")
        key = raw.get("key")
        digest = raw.get("sha256")
        size = raw.get("size")
        files = raw.get("files")
        if (
            not isinstance(key, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or isinstance(files, bool)
            or not isinstance(files, int)
            or files <= 0
        ):
            raise PreparationError("snapshot shard identity is invalid", {"index": index})
        shards.append(SnapshotShard(key, digest, size, files))
    if len({shard.key for shard in shards}) != len(shards):
        raise PreparationError("snapshot manifest has duplicate shard keys")
    return dict(value), shards


def _validate_manifest_locations(
    destination: ObjectStoreDestination,
    manifest: Mapping[str, object],
    shards: list[SnapshotShard],
    manifest_key: str,
    manifest_bytes: bytes,
) -> None:
    bundle = manifest["bundle"]
    assert isinstance(bundle, Mapping)
    bundle_sha256 = bundle.get("bundle_manifest_sha256")
    if not isinstance(bundle_sha256, str):
        raise PreparationError("snapshot bundle identity is invalid")
    object_prefix = f"{destination.prefix}/datasets/training-bundle-snapshot/{bundle_sha256}/restore-ready/objects/"
    if not manifest_key.startswith(object_prefix) or not manifest_key.endswith(
        f"/{hashlib.sha256(manifest_bytes).hexdigest()}.json"
    ):
        raise PreparationError("snapshot manifest key does not match its content identity")
    for shard in shards:
        if not shard.key.startswith(object_prefix) or not shard.key.endswith(f"/{shard.sha256}.tar"):
            raise PreparationError("snapshot shard key does not match its content identity", {"key": shard.key})


def verify_bundle_snapshot(
    destination: ObjectStoreDestination,
    manifest_key: str,
    output: Path,
    *,
    backend: SnapshotStore | None = None,
) -> dict[str, object]:
    """Stream every remote shard once and verify privacy and immutable identities."""

    store = backend or backend_from_destination(destination)
    raw_manifest, manifest_bytes = _read_remote_json(store, manifest_key)
    manifest, shards = _parse_manifest(raw_manifest)
    _validate_manifest_locations(destination, manifest, shards, manifest_key, manifest_bytes)
    started = time.monotonic()
    for shard in shards:
        digest, size = _remote_identity(store, shard.key, max_bytes=shard.size)
        if digest != shard.sha256 or size != shard.size:
            raise PreparationError("remote snapshot shard identity differs", {"key": shard.key})
    privacy = assert_private_access(store, manifest_key, destination.prefix)
    receipt = {
        "schema": "speakrs-training-bundle-snapshot-verification-v1",
        "manifest_key": manifest_key,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "shards": len(shards),
        "bytes": sum(shard.size for shard in shards),
        "privacy": privacy,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output, receipt)
    return {"ok": True, "command": "bundle-snapshot-verify", **receipt}


def _safe_tar_members(archive: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    for member in archive:
        _safe_relative_path(member.name, "tar member")
        if not member.isfile():
            raise PreparationError("snapshot tar can contain only regular files", {"member": member.name})
        yield member


def _extract_regular_members(archive: tarfile.TarFile, members: list[tarfile.TarInfo], root: Path) -> None:
    """Extract validated regular files without link or special-file handling."""

    for member in members:
        destination = root.joinpath(*PurePosixPath(member.name).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise PreparationError("snapshot tar member has no readable body", {"member": member.name})
        with destination.open("xb") as handle:
            shutil.copyfileobj(source, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        destination.chmod(member.mode & 0o777)


def restore_bundle_snapshot(
    destination: ObjectStoreDestination,
    manifest_key: str,
    restore_parent: Path,
    output: Path,
    *,
    backend: SnapshotStore | None = None,
) -> dict[str, object]:
    """Directly download, unpack, and verify one complete snapshot."""

    store = backend or backend_from_destination(destination)
    raw_manifest, manifest_bytes = _read_remote_json(store, manifest_key)
    manifest, shards = _parse_manifest(raw_manifest)
    _validate_manifest_locations(destination, manifest, shards, manifest_key, manifest_bytes)
    restore = manifest.get("restore")
    bundle = manifest.get("bundle")
    descriptor = manifest.get("launch_descriptor")
    if not isinstance(restore, Mapping) or not isinstance(bundle, Mapping) or not isinstance(descriptor, Mapping):
        raise PreparationError("snapshot manifest restore identity is invalid")
    if str(restore_parent) != restore.get("parent"):
        raise PreparationError(
            "restore parent differs from the path required by wav.scp",
            {"expected": restore.get("parent"), "actual": str(restore_parent)},
        )
    bundle_name = _safe_relative_path(str(bundle.get("directory")), "bundle directory")
    descriptor_name = _safe_relative_path(str(descriptor.get("name")), "launch descriptor")
    final_bundle = restore_parent / bundle_name.as_posix()
    final_descriptor = restore_parent / descriptor_name.as_posix()
    if _path_is_within(output, final_bundle) or output.resolve() == final_descriptor.resolve():
        raise PreparationError("restore receipt cannot modify restored snapshot contents")
    if final_bundle.exists() or final_descriptor.exists():
        raise PreparationError("snapshot restore target already exists", {"bundle": str(final_bundle)})
    restore_parent.mkdir(parents=True, exist_ok=True)
    partial = restore_parent / f".{bundle_name.name}.snapshot-partial"
    if partial.exists():
        raise PreparationError("snapshot partial restore already exists", {"path": str(partial)})
    partial.mkdir()
    started = time.monotonic()
    extracted: set[str] = set()
    moved_bundle = False
    moved_descriptor = False

    try:
        for shard in shards:
            temporary = partial / ".snapshot-shard.tar"
            digest = hashlib.sha256()
            size = 0
            with temporary.open("wb") as handle:
                for chunk in store.iter_bytes(shard.key, max_bytes=shard.size):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if digest.hexdigest() != shard.sha256 or size != shard.size:
                raise PreparationError("downloaded snapshot shard identity differs", {"key": shard.key})
            with tarfile.open(temporary, "r:") as archive:
                members = list(_safe_tar_members(archive))
                names = {member.name for member in members}
                if len(names) != len(members) or names & extracted:
                    raise PreparationError("snapshot tar contains a duplicate member")
                if len(members) != shard.files:
                    raise PreparationError("snapshot tar file count differs from its identity", {"key": shard.key})
                extracted.update(names)
                _extract_regular_members(archive, members, partial)
            temporary.unlink()

        restored_bundle = partial / bundle_name.as_posix()
        restored_descriptor = partial / descriptor_name.as_posix()
        tree_sha256, files, size = _tree_identity(restored_bundle)
        if tree_sha256 != bundle.get("tree_sha256") or files != bundle.get("files") or size != bundle.get("bytes"):
            raise PreparationError("restored bundle tree identity differs from the snapshot")
        if sha256_file(restored_bundle / "bundle.json") != bundle.get("bundle_manifest_sha256"):
            raise PreparationError("restored bundle manifest identity differs from the snapshot")
        if sha256_file(restored_descriptor) != descriptor.get("sha256"):
            raise PreparationError("restored launch descriptor identity differs from the snapshot")
        restored_bundle.replace(final_bundle)
        moved_bundle = True
        restored_descriptor.replace(final_descriptor)
        moved_descriptor = True
        partial.rmdir()
    except BaseException:
        if moved_descriptor:
            final_descriptor.unlink(missing_ok=True)
        if moved_bundle:
            shutil.rmtree(final_bundle)
        shutil.rmtree(partial, ignore_errors=True)
        raise

    receipt = {
        "schema": "speakrs-training-bundle-snapshot-restore-v1",
        "manifest_key": manifest_key,
        "bundle": str(final_bundle),
        "launch_descriptor": str(final_descriptor),
        "bundle_manifest_sha256": bundle["bundle_manifest_sha256"],
        "tree_sha256": bundle["tree_sha256"],
        "files": bundle["files"],
        "bytes": bundle["bytes"],
        "shards": len(shards),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output, receipt)
    return {"ok": True, "command": "bundle-snapshot-restore", **receipt}
