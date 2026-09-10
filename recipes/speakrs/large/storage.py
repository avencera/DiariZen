"""Local staging, remote transfer, and eviction owned by this module."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree

from .contracts import (
    BatchState,
    LocalCopyState,
    ObjectState,
    ObjectStoreDestination,
    RemoteReleaseState,
    SelectionState,
    SourceMembership,
    assert_state_transition,
    is_placeholder_hash,
    parse_object_receipt,
    require_content_hash,
)
from .errors import ContractError, PreparationError, UnresolvedInputError
from .hashing import sha256_bytes, sha256_file, sha256_json
from .jsonio import read_json, write_json


DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_OBJECT_BYTES = 8 * 1024**3
DEFAULT_MAX_INVENTORY_KEYS = 100_000
DEFAULT_MAX_INVENTORY_RESPONSE_BYTES = 64 * 1024**2
OPEN_FILE_CHECK_TIMEOUT_SECONDS = 5
CLOUDFLARE_R2_DEFAULT_ENCRYPTION = "AES-256"
WRANGLER_R2_REST_MAX_UPLOAD_BYTES = 300 * 1024**2
WRANGLER_VERSION = "4.129.0"
MARKER_DIRECTORY = "_commits"
_R2_CONTROL_CANARIES = (
    ("_canary/owner-probe.bin", b"speakrs-diarization-r2-owner-canary"),
    ("_canary/probe.txt", b"speakrs-diarization-r2-canary"),
)
CONSUMED_SOURCE_STATE = "consumed-source"
DELETION_JOURNAL_SCHEMA = "speakrs-deletion-journal-v1"
DELETION_JOURNAL_DIRECTORY = ".deletion-journal"
DELETION_INTENT_STATE = "intent"
DELETION_COMPLETED_STATE = "completed"
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ACCOUNT_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def probe_writable_root(root: Path) -> dict[str, object]:
    """Create, write, fsync, rename, read, and delete a task-owned probe."""

    root = root.expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PreparationError(
            f"storage root is not writable: {root}",
            {"error": str(error)},
        ) from error

    probe_dir = root / ".large-write-probe"
    probe_dir.mkdir(exist_ok=True)
    source = probe_dir / "write.bin"
    renamed = probe_dir / "write-renamed.bin"
    try:
        with source.open("wb") as handle:
            handle.write(b"wavlm-large-probe")
            handle.flush()
            os.fsync(handle.fileno())
        source.replace(renamed)
        payload = renamed.read_bytes()
        if payload != b"wavlm-large-probe":
            raise PreparationError("probe read-back mismatch", {"path": str(root)})
        usage = os.statvfs(root)
        free_bytes = usage.f_frsize * usage.f_bavail
        result = {
            "path": str(root.resolve()),
            "writable": True,
            "free_bytes": free_bytes,
            "free_gib": round(free_bytes / 1024**3, 2),
        }
    except OSError as error:
        raise PreparationError(
            f"storage probe failed: {root}",
            {"error": str(error)},
        ) from error
    finally:
        renamed.unlink(missing_ok=True)
        source.unlink(missing_ok=True)
        try:
            probe_dir.rmdir()
        except OSError:
            pass
    return result


def require_free_gib(root: Path, minimum_gib: float) -> dict[str, object]:
    """Fail when the probed root does not have enough free space."""

    probe = probe_writable_root(root)
    if float(probe["free_gib"]) < minimum_gib:
        raise PreparationError(
            f"only {probe['free_gib']} GiB is free; {minimum_gib} GiB is required",
            probe,
        )
    return probe


def write_resource_plan(path: Path, plan: dict[str, object]) -> None:
    """Write the provisional resource plan before bulk transfers."""

    write_json(path, plan)


def load_resource_plan(path: Path) -> dict[str, object]:
    """Read a previously written resource plan."""

    return json.loads(path.read_text(encoding="utf-8"))


OBJECT_TRANSITIONS = {
    ObjectState.PLANNED: {ObjectState.UPLOADING},
    ObjectState.UPLOADING: {ObjectState.UPLOADED, ObjectState.PLANNED},
    ObjectState.UPLOADED: {ObjectState.READBACK_VERIFIED, ObjectState.UPLOADING},
    ObjectState.READBACK_VERIFIED: set(),
}
BATCH_TRANSITIONS = {
    BatchState.DRAFT: {BatchState.OBJECTS_VERIFIED},
    BatchState.OBJECTS_VERIFIED: {BatchState.COMMITTED},
    BatchState.COMMITTED: set(),
}
RELEASE_TRANSITIONS = {
    RemoteReleaseState.DRAFT: {RemoteReleaseState.BATCHES_ACCOUNTED},
    RemoteReleaseState.BATCHES_ACCOUNTED: {RemoteReleaseState.COMMITTED},
    RemoteReleaseState.COMMITTED: set(),
}
LOCAL_TRANSITIONS = {
    LocalCopyState.RETAINED: {LocalCopyState.EVICTION_ELIGIBLE},
    LocalCopyState.EVICTION_ELIGIBLE: {LocalCopyState.EVICTED, LocalCopyState.RETAINED},
    LocalCopyState.EVICTED: set(),
}


def _validate_key(key: str, label: str = "object key") -> str:
    """Validate a relative object key before passing it to a provider."""

    if not isinstance(key, str) or not key or key.startswith("/"):
        raise ContractError(f"{label} must be a relative non-empty key")
    parts = key.split("/")
    if any(part in {"", ".", ".."} for part in parts) or "?" in key or "#" in key:
        raise ContractError(f"{label} contains an invalid path segment", {"key": key})
    return key


def _validate_prefix(prefix: str, label: str = "prefix") -> str:
    """Validate and normalize a non-empty task-owned object prefix."""

    if not isinstance(prefix, str) or not prefix:
        raise ContractError(f"{label} must be a non-empty prefix")
    normalized = prefix.strip("/")
    if (
        not normalized
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
        or "?" in normalized
        or "#" in normalized
    ):
        raise ContractError(f"{label} contains an invalid path segment", {"prefix": prefix})
    return normalized


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a non-negative integer")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


_OBJECT_KEY_RE = re.compile(
    r"^datasets/(?P<source>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<selection>[A-Za-z0-9][A-Za-z0-9_.-]*)/objects/"
    r"(?P<digest>[0-9a-fA-F]{64})\.(?P<extension>[A-Za-z0-9][A-Za-z0-9_.-]*)$"
)
_MARKER_KEY_RE = re.compile(
    r"^(?:(?P<scope>datasets/[A-Za-z0-9][A-Za-z0-9_.-]*/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)/)?"
    r"_commits/(?P<kind>batches|releases)/(?P<digest>[0-9a-fA-F]{64})\.json$"
)
_MARKER_DEFINITIONS: dict[str, tuple[str, str, frozenset[str]]] = {
    "batches": (
        "speakrs-remote-batch-v1",
        "batch_sha256",
        frozenset(
            {
                "schema",
                "state",
                "objects",
                "label_policy_id",
                "split_id",
                "qa_policy_sha256",
                "acceptance_sha256",
                "inventory",
                "batch_sha256",
            }
        ),
    ),
    "releases": (
        "speakrs-remote-release-v1",
        "release_sha256",
        frozenset(
            {
                "schema",
                "state",
                "batches",
                "objects",
                "required_sources",
                "release_identity",
                "inventory",
                "release_sha256",
            }
        ),
    ),
}


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_canonical_marker(payload: bytes, *, kind: str, digest: str) -> Mapping[str, object]:
    """Validate the canonical bytes and schema identity of one commit marker."""

    try:
        decoded = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError("content-addressed marker payload is not valid JSON") from error
    if not isinstance(decoded, Mapping):
        raise ContractError("content-addressed marker payload must be a JSON object")
    if _canonical_json_bytes(decoded) != payload:
        raise ContractError("content-addressed marker payload is not canonical JSON")
    definition = _MARKER_DEFINITIONS[kind]
    schema, digest_field, fields = definition
    if set(decoded) != fields:
        raise ContractError(
            "content-addressed marker schema has unexpected fields",
            {"kind": kind, "unknown": sorted(set(decoded) - fields), "missing": sorted(fields - set(decoded))},
        )
    if decoded.get("schema") != schema:
        raise ContractError("content-addressed marker schema is not supported", {"kind": kind})
    if decoded.get("state") != RemoteReleaseState.COMMITTED.value:
        raise ContractError("content-addressed marker must be committed", {"kind": kind})
    marker_digest = require_content_hash(decoded.get(digest_field), f"{kind} marker digest")
    if marker_digest != digest:
        raise ContractError(
            "content-addressed marker key does not match its digest field",
            {"kind": kind, "key_digest": digest, "payload_digest": marker_digest},
        )
    base = dict(decoded)
    del base[digest_field]
    if sha256_json(base) != digest:
        raise ContractError("content-addressed marker digest does not match its canonical identity", {"kind": kind})

    objects = decoded.get("objects")
    if not isinstance(objects, list):
        raise ContractError("content-addressed marker objects must be an array", {"kind": kind})
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise ContractError("content-addressed marker contains an invalid object", {"kind": kind, "index": index})
        _object_identity(item, f"{kind} marker objects[{index}]")
        if item.get("state") != ObjectState.READBACK_VERIFIED.value:
            raise ContractError(
                "content-addressed marker object is not readback-verified", {"kind": kind, "index": index}
            )
        if not isinstance(item.get("encryption"), str) or not item.get("encryption"):
            raise ContractError(
                "content-addressed marker object lacks encryption evidence", {"kind": kind, "index": index}
            )
        if item.get("public") is not False:
            raise ContractError("content-addressed marker object is public", {"kind": kind, "index": index})

    inventory = decoded.get("inventory")
    if not isinstance(inventory, Mapping):
        raise ContractError("content-addressed marker inventory must be an object", {"kind": kind})
    if inventory.get("complete") is not True:
        raise ContractError("content-addressed marker inventory is incomplete", {"kind": kind})
    return decoded


def _validate_content_addressed_parts(
    key: str,
    relative: str,
    payload: bytes,
    content_type: str,
) -> tuple[str, str]:
    """Validate one content-addressed path after its task prefix is removed."""

    if not isinstance(payload, bytes):
        raise ContractError("content-addressed write payload must be bytes", {"key": key})
    if not isinstance(content_type, str) or not content_type.strip():
        raise ContractError("content-addressed write content type must be a non-empty string", {"key": key})
    object_match = _OBJECT_KEY_RE.fullmatch(relative)
    if object_match is not None:
        digest = require_content_hash(object_match.group("digest"), "object key digest")
        actual = sha256_bytes(payload)
        if actual != digest:
            raise ContractError(
                "content-addressed object key does not match its payload",
                {"key": key, "expected": digest, "actual": actual},
            )
        return "object", digest

    marker_match = _MARKER_KEY_RE.fullmatch(relative)
    if marker_match is None:
        raise ContractError(
            "content-addressed write key is not an owned object or commit marker",
            {"key": key},
        )
    kind = marker_match.group("kind")
    digest = require_content_hash(marker_match.group("digest"), f"{kind} marker key digest")
    if kind == "releases" and marker_match.group("scope") is not None:
        raise ContractError("release markers must stay at the configured task prefix", {"key": key})
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise ContractError(
            "commit markers must use the application/json content type",
            {"key": key, "content_type": content_type},
        )
    _load_canonical_marker(payload, kind=kind, digest=digest)
    return f"{kind}-marker", digest


@dataclass(frozen=True)
class ContentAddressedWrite:
    """Validated immutable bytes for an owned R2 object or commit marker."""

    key: str
    payload: bytes
    content_type: str
    kind: str
    digest: str

    @classmethod
    def validate(
        cls,
        destination: ObjectStoreDestination,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> ContentAddressedWrite:
        """Validate an object key and its bytes before any remote request."""

        prefix = _validate_prefix(destination.prefix, "destination.prefix")
        key = _validate_key(key)
        if not key.startswith(prefix + "/"):
            raise ContractError(
                "content-addressed write must stay under the configured destination prefix",
                {"key": key, "prefix": prefix},
            )
        relative = key[len(prefix) + 1 :]
        kind, digest = _validate_content_addressed_parts(key, relative, payload, content_type)
        return cls(key=key, payload=payload, content_type=content_type, kind=kind, digest=digest)


def validate_content_addressed_write(
    destination: ObjectStoreDestination,
    key: str,
    payload: bytes,
    *,
    content_type: str = "application/octet-stream",
) -> ContentAddressedWrite:
    """Validate content-addressed bytes before passing them to a remote backend."""

    return ContentAddressedWrite.validate(destination, key, payload, content_type=content_type)


def put_content_addressed(
    backend: StorageBackend,
    destination: ObjectStoreDestination,
    key: str,
    payload: bytes,
    *,
    content_type: str = "application/octet-stream",
) -> dict[str, str]:
    """Validate an immutable write before handing it to a cooperating backend."""

    write = validate_content_addressed_write(destination, key, payload, content_type=content_type)
    return backend.put_content_addressed(write.key, write.payload, content_type=write.content_type)


def _bounded_chunks(chunks: Iterator[bytes], *, max_bytes: int | None) -> Iterator[bytes]:
    total = 0
    for chunk in chunks:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise PreparationError("remote reader returned a non-byte chunk")
        value = bytes(chunk)
        total += len(value)
        if max_bytes is not None and total > max_bytes:
            raise PreparationError(
                "remote object exceeds the configured read bound",
                {"max_bytes": max_bytes},
            )
        if value:
            yield value


def _read_response_bounded(response: Any, *, max_bytes: int) -> bytes:
    """Read an HTTP response in bounded chunks."""

    max_bytes = _positive_int(max_bytes, "max_bytes")
    return _collect_chunks(
        iter(lambda: response.read(DEFAULT_CHUNK_SIZE), b""),
        max_bytes=max_bytes,
    )


def _iter_backend_bytes(
    backend: StorageBackend,
    key: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
) -> Iterator[bytes]:
    """Use a backend streaming method when available and enforce a hard byte bound."""

    chunk_size = _positive_int(chunk_size, "chunk_size")
    if max_bytes is not None:
        max_bytes = _positive_int(max_bytes, "max_bytes")
    iterator = getattr(backend, "iter_bytes", None)
    if callable(iterator):
        try:
            chunks = iterator(key, chunk_size=chunk_size, max_bytes=max_bytes)
        except TypeError:
            chunks = iterator(key)
        yield from _bounded_chunks(iter(chunks), max_bytes=max_bytes)
        return
    payload = backend.get_bytes(key)
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    yield from _bounded_chunks(iter((payload,)), max_bytes=max_bytes)


def _collect_chunks(chunks: Iterator[bytes], *, max_bytes: int | None) -> bytes:
    return b"".join(_bounded_chunks(chunks, max_bytes=max_bytes))


@dataclass(frozen=True)
class ReadbackProof:
    """Full remote readback identity for one object."""

    key: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe readback proof."""

        return {"key": self.key, "sha256": self.sha256, "size": self.size}


class StorageBackend(Protocol):
    """Remote object operations. This backend does not decide acceptance."""

    atomic_create_supported: bool

    def put_bytes(self, key: str, payload: bytes, *, content_type: str = "application/octet-stream") -> dict[str, str]:
        """Store immutable bytes. Same key with different bytes must fail."""

    def put_content_addressed(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> dict[str, str]:
        """Validate an immutable key and payload before storing the bytes."""

    def get_bytes(self, key: str) -> bytes:
        """Read the full remote object."""

    def iter_bytes(
        self,
        key: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
    ) -> Iterator[bytes]:
        """Read an object in bounded chunks."""

    def head(self, key: str) -> dict[str, str]:
        """Return metadata. Metadata is not a content proof."""

    def list_prefix(self, prefix: str, *, max_keys: int | None = DEFAULT_MAX_INVENTORY_KEYS) -> list[str]:
        """List keys under a scoped prefix."""

    def exists(self, key: str) -> bool:
        """Return whether a key exists."""

    def encryption_evidence(self, key: str) -> str | None:
        """Return encryption-at-rest evidence if present."""

    def is_public(self, key: str) -> bool:
        """Return whether anonymous clients can read the object."""

    def anonymous_get(self, key: str) -> tuple[int, bytes]:
        """GET without credentials. Private objects must not disclose contents."""

    def anonymous_list(self, prefix: str) -> tuple[int, list[str]]:
        """List without credentials."""


@dataclass
class StoredObject:
    """In-memory object used by tests and local dry-runs."""

    payload: bytes
    etag: str
    encryption: str | None = "AES256"
    public: bool = False
    content_type: str = "application/octet-stream"


@dataclass
class MemoryBackend:
    """Deterministic backend for failure tests. Not a remote proof."""

    atomic_create_supported = False

    objects: dict[str, StoredObject] = field(default_factory=dict)
    fail_get_keys: set[str] = field(default_factory=set)
    fail_head_keys: set[str] = field(default_factory=set)
    fail_list: bool = False
    corrupt_get_keys: set[str] = field(default_factory=set)
    expired: bool = False
    encryption_default: str | None = "AES256"
    allow_anonymous: bool = False
    put_interrupt_after: int | None = None
    _put_count: int = 0

    def put_bytes(self, key: str, payload: bytes, *, content_type: str = "application/octet-stream") -> dict[str, str]:
        _validate_key(key)
        if self.expired:
            raise PreparationError("expired credentials")
        payload = bytes(payload)
        existing = self.objects.get(key)
        if existing is not None:
            if existing.payload != payload:
                raise PreparationError("same-key different-content fails", {"key": key})
            return {"etag": existing.etag, "key": key}
        self._put_count += 1
        if self.put_interrupt_after is not None and self._put_count > self.put_interrupt_after:
            raise PreparationError("interrupted upload", {"key": key})
        etag = hashlib.md5(payload).hexdigest()
        self.objects[key] = StoredObject(
            payload=payload,
            etag=etag,
            encryption=self.encryption_default,
            public=False,
            content_type=content_type,
        )
        return {"etag": etag, "key": key}

    def put_content_addressed(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> dict[str, str]:
        """Validate content-addressed bytes before using the in-memory writer."""

        marker_match = re.search(
            r"(?:^|/)(?P<relative>_commits/(?P<kind>batches|releases)/[0-9a-fA-F]{64}\.json)$",
            key,
        )
        object_start = key.rfind("datasets/")
        object_relative = key[object_start:] if object_start >= 0 else ""
        relative = (
            object_relative
            if _OBJECT_KEY_RE.fullmatch(object_relative)
            else marker_match.group("relative")
            if marker_match
            else key
        )
        kind, _ = _validate_content_addressed_parts(key, relative, payload, content_type)
        if kind not in {"object", "batches-marker", "releases-marker"}:
            raise ContractError("content-addressed write kind is unknown", {"key": key})
        return self.put_bytes(key, payload, content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        return _collect_chunks(
            self.iter_bytes(key, chunk_size=DEFAULT_CHUNK_SIZE, max_bytes=DEFAULT_MAX_OBJECT_BYTES),
            max_bytes=DEFAULT_MAX_OBJECT_BYTES,
        )

    def iter_bytes(
        self,
        key: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
    ) -> Iterator[bytes]:
        _validate_key(key)
        chunk_size = _positive_int(chunk_size, "chunk_size")
        if max_bytes is not None:
            max_bytes = _positive_int(max_bytes, "max_bytes")
        if self.expired:
            raise PreparationError("expired credentials")
        if key in self.fail_get_keys or key not in self.objects:
            raise PreparationError("missing remote object", {"key": key})
        payload = self.objects[key].payload
        if key in self.corrupt_get_keys:
            payload = payload[:-1] if payload else b"x"
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]

    def head(self, key: str) -> dict[str, str]:
        _validate_key(key)
        if self.expired:
            raise PreparationError("expired credentials")
        if key in self.fail_head_keys:
            raise PreparationError("metadata unavailable", {"key": key})
        stored = self.objects.get(key)
        if stored is None:
            raise PreparationError("missing remote object", {"key": key})
        return {
            "etag": stored.etag,
            "size": str(len(stored.payload)),
            "encryption": stored.encryption or "",
            "content_type": stored.content_type,
        }

    def list_prefix(self, prefix: str, *, max_keys: int | None = DEFAULT_MAX_INVENTORY_KEYS) -> list[str]:
        _validate_prefix(prefix)
        if self.fail_list:
            raise PreparationError("remote inventory is unavailable", {"prefix": prefix})
        if max_keys is not None:
            max_keys = _positive_int(max_keys, "max_keys")
        keys = sorted(key for key in self.objects if key.startswith(prefix))
        if max_keys is not None and len(keys) > max_keys:
            raise PreparationError("remote inventory exceeds the configured key bound", {"max_keys": max_keys})
        return keys

    def exists(self, key: str) -> bool:
        _validate_key(key)
        return key in self.objects

    def encryption_evidence(self, key: str) -> str | None:
        stored = self.objects.get(key)
        return None if stored is None else stored.encryption

    def is_public(self, key: str) -> bool:
        stored = self.objects.get(key)
        return bool(stored and stored.public)

    def anonymous_get(self, key: str) -> tuple[int, bytes]:
        stored = self.objects.get(key)
        if stored is None:
            return 404, b""
        if self.allow_anonymous or stored.public:
            return 200, stored.payload
        return 403, b""

    def anonymous_list(self, prefix: str) -> tuple[int, list[str]]:
        if self.allow_anonymous:
            return 200, self.list_prefix(prefix)
        return 403, []


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _canonical_query(query: Mapping[str, str] | None) -> str:
    if not query:
        return ""
    encoded = [
        (
            urllib.parse.quote(str(name), safe="-_.~"),
            urllib.parse.quote(str(value), safe="-_.~"),
        )
        for name, value in query.items()
    ]
    return "&".join(f"{name}={value}" for name, value in sorted(encoded))


def _sigv4_headers(
    *,
    method: str,
    endpoint: str,
    bucket: str,
    key: str,
    payload: bytes,
    access_key: str,
    secret_key: str,
    region: str,
    extra_headers: dict[str, str] | None = None,
    query: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    parsed = urllib.parse.urlparse(endpoint)
    host = parsed.netloc
    amz_date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    datestamp = amz_date[:8]
    payload_hash = hashlib.sha256(payload).hexdigest()
    object_path = f"/{bucket}/{key}" if key else f"/{bucket}"
    canonical_uri = urllib.parse.quote(object_path, safe="/-_.~")
    canonical_query = _canonical_query(query)
    headers = {
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    if extra_headers:
        headers.update({name.lower(): value.strip() for name, value in extra_headers.items()})
    signed_header_names = ";".join(sorted(headers))
    canonical_headers = "".join(f"{name}:{headers[name]}\n" for name in sorted(headers))
    canonical_request = "\n".join(
        [method, canonical_uri, canonical_query, canonical_headers, signed_header_names, payload_hash]
    )
    credential_scope = f"{datestamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signing_key = _sign(
        _sign(_sign(_sign(("AWS4" + secret_key).encode("utf-8"), datestamp), region), "s3"),
        "aws4_request",
    )
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_header_names}, Signature={signature}"
    )
    url = f"{parsed.scheme}://{host}{canonical_uri}"
    if canonical_query:
        url += f"?{canonical_query}"
    return url, headers


def load_rclone_s3_remote(name: str) -> dict[str, str]:
    """Load endpoint and keys from rclone.conf without logging secrets."""

    config = Path.home() / ".config" / "rclone" / "rclone.conf"
    if not config.is_file():
        raise UnresolvedInputError("rclone credentials are not available", {"reference": name})
    current = None
    values: dict[str, dict[str, str]] = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            values[current] = {}
            continue
        if current is None or "=" not in stripped or stripped.startswith("#"):
            continue
        key, _, raw = stripped.partition("=")
        values[current][key.strip().lower()] = raw.strip()
    remote = values.get(name)
    if remote is None:
        raise UnresolvedInputError("named rclone remote is missing", {"reference": name})
    access = remote.get("access_key_id") or remote.get("access_key")
    secret = remote.get("secret_access_key") or remote.get("secret_key")
    endpoint = remote.get("endpoint")
    if not access or not secret or not endpoint:
        raise UnresolvedInputError("rclone remote is missing S3 fields", {"reference": name})
    if not endpoint.startswith("http"):
        endpoint = "https://" + endpoint
    return {
        "access_key": access,
        "secret_key": secret,
        "endpoint": endpoint,
        "region": remote.get("region") or "auto",
    }


@dataclass
class S3CompatibleBackend:
    """Signed S3-compatible adapter. Does not decide source permission."""

    atomic_create_supported = False

    destination: ObjectStoreDestination
    access_key: str
    secret_key: str
    timeout: int = 60

    @classmethod
    def from_rclone(cls, destination: ObjectStoreDestination) -> S3CompatibleBackend:
        """Resolve rclone S3 credentials."""

        reference = destination.credential_reference.split(":", 1)
        if len(reference) != 2 or not reference[1]:
            raise UnresolvedInputError("rclone credential reference is incomplete")
        loaded = load_rclone_s3_remote(reference[1])
        endpoint = destination.endpoint or loaded["endpoint"]
        return cls(
            destination=ObjectStoreDestination(
                provider=destination.provider,
                endpoint=endpoint,
                bucket=destination.bucket,
                prefix=destination.prefix,
                credential_reference=destination.credential_reference,
                region=destination.region or loaded["region"],
            ),
            access_key=loaded["access_key"],
            secret_key=loaded["secret_key"],
        )

    def _request(
        self,
        method: str,
        key: str,
        payload: bytes = b"",
        extra_headers: dict[str, str] | None = None,
        query: Mapping[str, str] | None = None,
        *,
        max_body_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
    ) -> tuple[int, bytes, dict[str, str]]:
        if key:
            _validate_key(key)
        url, headers = _sigv4_headers(
            method=method,
            endpoint=self.destination.endpoint,
            bucket=self.destination.bucket,
            key=key,
            payload=payload,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.destination.region or "auto",
            extra_headers=extra_headers,
            query=query,
        )
        request = urllib.request.Request(url, data=payload if method in {"PUT", "POST"} else None, method=method)
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read() if method != "HEAD" else b""
                if max_body_bytes is not None and len(body) > max_body_bytes:
                    raise PreparationError(
                        "object-store response exceeds the configured read bound",
                        {"max_bytes": max_body_bytes},
                    )
                return response.status, body, {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as error:
            body = error.read(DEFAULT_CHUNK_SIZE)
            if error.code in {401, 403}:
                raise UnresolvedInputError(
                    "object store rejected authenticated access to the named bucket",
                    {"bucket": self.destination.bucket, "status": error.code},
                ) from error
            raise PreparationError(
                "object store request failed",
                {"status": error.code, "key": key},
            ) from error
        except urllib.error.URLError as error:
            raise UnresolvedInputError(
                "object store request failed to connect",
                {"reason": str(error.reason)},
            ) from error

    def put_bytes(self, key: str, payload: bytes, *, content_type: str = "application/octet-stream") -> dict[str, str]:
        write = ContentAddressedWrite.validate(self.destination, key, payload, content_type=content_type)
        payload = write.payload
        try:
            if self.exists(write.key):
                remote = self.get_bytes(write.key)
                if remote != payload:
                    raise PreparationError("same-key different-content fails", {"key": write.key})
                return {"key": write.key, "etag": ""}
        except PreparationError as error:
            if (error.details or {}).get("status") != 404:
                raise
        status, _, headers = self._request(
            "PUT",
            write.key,
            payload,
            extra_headers={"content-type": content_type},
        )
        if status not in {200, 201}:
            raise PreparationError("upload failed", {"status": status, "key": write.key})
        return {"key": write.key, "etag": headers.get("etag", "").strip('"')}

    def put_content_addressed(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> dict[str, str]:
        """Validate content-addressed bytes before using the S3 writer."""

        write = ContentAddressedWrite.validate(self.destination, key, payload, content_type=content_type)
        return self.put_bytes(write.key, write.payload, content_type=write.content_type)

    def iter_bytes(
        self,
        key: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
    ) -> Iterator[bytes]:
        _validate_key(key)
        chunk_size = _positive_int(chunk_size, "chunk_size")
        if max_bytes is not None:
            max_bytes = _positive_int(max_bytes, "max_bytes")
        url, headers = _sigv4_headers(
            method="GET",
            endpoint=self.destination.endpoint,
            bucket=self.destination.bucket,
            key=key,
            payload=b"",
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.destination.region or "auto",
        )
        request = urllib.request.Request(url, method="GET")
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            error.read(DEFAULT_CHUNK_SIZE)
            if error.code in {401, 403}:
                raise UnresolvedInputError(
                    "object store rejected authenticated access to the named bucket",
                    {"bucket": self.destination.bucket, "status": error.code},
                ) from error
            raise PreparationError("missing remote object", {"key": key, "status": error.code}) from error
        except urllib.error.URLError as error:
            raise UnresolvedInputError(
                "object store request failed to connect",
                {"reason": str(error.reason)},
            ) from error
        with response:
            if response.status != 200:
                raise PreparationError("missing remote object", {"key": key, "status": response.status})
            total = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise PreparationError(
                        "remote object exceeds the configured read bound",
                        {"max_bytes": max_bytes, "key": key},
                    )
                yield chunk

    def get_bytes(self, key: str) -> bytes:
        return _collect_chunks(
            self.iter_bytes(key, chunk_size=DEFAULT_CHUNK_SIZE, max_bytes=DEFAULT_MAX_OBJECT_BYTES),
            max_bytes=DEFAULT_MAX_OBJECT_BYTES,
        )

    def head(self, key: str) -> dict[str, str]:
        _validate_key(key)
        status, _, headers = self._request("HEAD", key)
        if status != 200:
            raise PreparationError("missing remote object", {"key": key, "status": status})
        return {
            "etag": headers.get("etag", "").strip('"'),
            "size": headers.get("content-length", ""),
            "encryption": headers.get("x-amz-server-side-encryption", ""),
            "content_type": headers.get("content-type", ""),
        }

    def list_prefix(self, prefix: str, *, max_keys: int | None = DEFAULT_MAX_INVENTORY_KEYS) -> list[str]:
        prefix = _validate_prefix(prefix)
        if max_keys is not None:
            max_keys = _positive_int(max_keys, "max_keys")
        keys: list[str] = []
        token: str | None = None
        while True:
            page_limit = 1000 if max_keys is None else min(1000, max_keys - len(keys))
            if page_limit <= 0:
                raise PreparationError("remote inventory exceeds the configured key bound", {"max_keys": max_keys})
            query: dict[str, str] = {"list-type": "2", "prefix": prefix, "max-keys": str(page_limit)}
            if token:
                query["continuation-token"] = token
            status, body, _ = self._request("GET", "", query=query, max_body_bytes=DEFAULT_MAX_OBJECT_BYTES)
            if status != 200:
                raise PreparationError("list failed", {"status": status, "prefix": prefix})
            try:
                root = ElementTree.fromstring(body)
            except ElementTree.ParseError as error:
                raise PreparationError("object inventory response is not valid XML", {"prefix": prefix}) from error
            namespace = root.tag.split("}", 1)[0] + "}" if root.tag.startswith("{") else ""
            page = [item.text or "" for item in root.findall(f".//{namespace}Key") if item.text]
            keys.extend(page)
            if max_keys is not None and len(keys) > max_keys:
                raise PreparationError("remote inventory exceeds the configured key bound", {"max_keys": max_keys})
            truncated = (root.findtext(f"{namespace}IsTruncated") or "").lower() == "true"
            if not truncated:
                break
            token = root.findtext(f"{namespace}NextContinuationToken")
            if not token:
                raise UnresolvedInputError("object inventory pagination is incomplete", {"prefix": prefix})
        if len(set(keys)) != len(keys):
            raise PreparationError("object inventory contains duplicate keys", {"prefix": prefix})
        return keys

    def exists(self, key: str) -> bool:
        try:
            self.head(key)
        except UnresolvedInputError:
            raise
        except PreparationError as error:
            if (error.details or {}).get("status") == 404:
                return False
            raise
        return True

    def encryption_evidence(self, key: str) -> str | None:
        try:
            return self.head(key).get("encryption") or None
        except UnresolvedInputError:
            raise
        except PreparationError:
            return None

    def is_public(self, key: str) -> bool:
        status, _ = self.anonymous_get(key)
        return 200 <= status < 300

    def _anonymous_url(self, key: str = "", query: Mapping[str, str] | None = None) -> str:
        parsed = urllib.parse.urlparse(self.destination.endpoint)
        path = f"/{self.destination.bucket}"
        if key:
            path += "/" + urllib.parse.quote(key, safe="/-_.~")
        url = f"{parsed.scheme}://{parsed.netloc}{path}"
        encoded = urllib.parse.urlencode(query or {})
        return f"{url}?{encoded}" if encoded else url

    def anonymous_get(self, key: str) -> tuple[int, bytes]:
        _validate_key(key)
        request = urllib.request.Request(self._anonymous_url(key), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read(DEFAULT_CHUNK_SIZE)
        except urllib.error.HTTPError as error:
            return error.code, error.read(DEFAULT_CHUNK_SIZE)
        except urllib.error.URLError as error:
            raise UnresolvedInputError(
                "anonymous object-store GET failed to connect",
                {"reason": str(error.reason)},
            ) from error

    def anonymous_list(self, prefix: str) -> tuple[int, list[str]]:
        prefix = _validate_prefix(prefix)
        request = urllib.request.Request(
            self._anonymous_url(query={"list-type": "2", "prefix": prefix}),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read(DEFAULT_CHUNK_SIZE)
                if not 200 <= response.status < 300:
                    return response.status, []
                try:
                    root = ElementTree.fromstring(body)
                except ElementTree.ParseError:
                    return response.status, ["disclosed"]
                namespace = root.tag.split("}", 1)[0] + "}" if root.tag.startswith("{") else ""
                keys = [item.text or "" for item in root.findall(f".//{namespace}Key") if item.text]
                return response.status, keys or ["disclosed"]
        except urllib.error.HTTPError as error:
            return error.code, []
        except urllib.error.URLError as error:
            raise UnresolvedInputError(
                "anonymous object-store listing failed to connect",
                {"reason": str(error.reason)},
            ) from error


def _run_wrangler(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["wrangler", *args], check=False, capture_output=True, text=True)


def assert_no_open_file_handles(path: Path) -> dict[str, object]:
    """Require OS evidence that no process currently holds a local file open."""

    path = path.expanduser().resolve()
    executable = shutil.which("lsof")
    if executable is None:
        raise UnresolvedInputError(
            "eviction cannot verify OS open-file state",
            {"path": str(path), "check": "lsof-unavailable"},
        )
    try:
        completed = subprocess.run(
            [executable, "-F", "p", "--", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=OPEN_FILE_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise UnresolvedInputError(
            "eviction cannot verify OS open-file state",
            {"path": str(path), "check": "lsof-timeout"},
        ) from error
    except OSError as error:
        raise UnresolvedInputError(
            "eviction cannot verify OS open-file state",
            {"path": str(path), "check": "lsof-failed"},
        ) from error

    pids: list[int] = []
    for line in completed.stdout.splitlines():
        if not line:
            continue
        if line.startswith("p"):
            raw_pid = line[1:]
            if not raw_pid.isdigit() or int(raw_pid) <= 0:
                raise UnresolvedInputError(
                    "eviction cannot verify OS open-file state",
                    {"path": str(path), "check": "lsof-invalid-output"},
                )
            pids.append(int(raw_pid))
            continue
        if line[0] in {"f", "n", "c", "a", "t", "u", "g", "G", "R", "T"}:
            continue
        raise UnresolvedInputError(
            "eviction cannot verify OS open-file state",
            {"path": str(path), "check": "lsof-invalid-output"},
        )
    if completed.returncode not in {0, 1} or (completed.returncode == 1 and completed.stderr.strip()):
        raise UnresolvedInputError(
            "eviction cannot verify OS open-file state",
            {"path": str(path), "check": "lsof-failed", "status": completed.returncode},
        )
    if completed.returncode == 0 and not pids:
        raise UnresolvedInputError(
            "eviction cannot verify OS open-file state",
            {"path": str(path), "check": "lsof-invalid-output"},
        )
    if pids:
        raise PreparationError(
            "open file handles block unsafe deletion",
            {"path": str(path), "open_pids": sorted(set(pids))},
        )
    return {"path": str(path), "check": "lsof", "open_pids": []}


def _wrangler_auth_paths(profile: str) -> tuple[Path, ...]:
    """Return the platform paths used by Wrangler's plaintext auth store."""

    if not isinstance(profile, str) or not profile or not _SEGMENT_RE.fullmatch(profile):
        raise ContractError("Wrangler profile must be a valid non-empty name")
    candidates: list[Path] = []
    configured = os.environ.get("WRANGLER_CONFIG_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            Path.home() / ".wrangler",
            Path.home() / "Library" / "Preferences" / ".wrangler",
            Path.home() / ".config" / ".wrangler",
        )
    )
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        candidates.append(Path(xdg_config).expanduser() / ".wrangler")
    profile_name = f"{profile}.toml"
    paths: list[Path] = []
    for candidate in candidates:
        for path in (candidate / "config" / profile_name, candidate / profile_name):
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def _load_wrangler_oauth_state(profile: str) -> tuple[str | None, datetime | None]:
    """Read the OAuth token and expiry from Wrangler's local auth profile."""

    for path in _wrangler_auth_paths(profile):
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                try:
                    import tomli as tomllib
                except ModuleNotFoundError:
                    import toml as tomllib

            parsed = tomllib.loads(raw)
            token = parsed.get("oauth_token") if isinstance(parsed, Mapping) else None
            expiration = parsed.get("expiration_time") if isinstance(parsed, Mapping) else None
        except (ImportError, ValueError, TypeError):
            token, expiration = None, None
        if isinstance(token, str) and token.strip():
            expiry: datetime | None
            if isinstance(expiration, datetime):
                expiry = expiration
            elif isinstance(expiration, str) and expiration.strip():
                normalized = expiration.strip()
                if normalized.endswith("Z"):
                    normalized = normalized[:-1] + "+00:00"
                try:
                    expiry = datetime.fromisoformat(normalized)
                except ValueError:
                    expiry = None
            else:
                expiry = None
            if expiry is not None:
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                expiry = expiry.astimezone(timezone.utc)
            return token.strip(), expiry
    return None, None


def _account_id_from_endpoint(endpoint: str) -> str | None:
    parsed = urllib.parse.urlparse(endpoint)
    hostname = parsed.hostname or ""
    first_label = hostname.split(".", 1)[0]
    return first_label if _ACCOUNT_ID_RE.fullmatch(first_label) else None


def _wrangler_whoami(profile: str) -> subprocess.CompletedProcess[str]:
    """Refresh the selected Wrangler profile and return its non-secret account output."""

    args: list[str] = []
    if profile != "default":
        args.extend(("--profile", profile))
    args.extend(("whoami", "--json"))
    return _run_wrangler(args)


def _account_id_from_wrangler_output(output: str) -> str | None:
    """Resolve the first valid account ID from Wrangler's JSON output."""

    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return None
    accounts = payload.get("accounts") if isinstance(payload, Mapping) else None
    if not isinstance(accounts, list):
        return None
    for account in accounts:
        if not isinstance(account, Mapping):
            continue
        account_id = account.get("id")
        if isinstance(account_id, str) and _ACCOUNT_ID_RE.fullmatch(account_id):
            return account_id
    return None


@dataclass
class WranglerR2Backend:
    """Cloudflare R2 adapter using Wrangler's authenticated REST API."""

    # keep false until Cloudflare documents conditional object creation
    atomic_create_supported = False

    destination: ObjectStoreDestination
    timeout: int = 60
    encryption_at_rest: str | None = None
    api_token: str | None = field(default=None, repr=False)
    account_id: str | None = None
    profile: str = "default"
    api_base_url: str = "https://api.cloudflare.com/client/v4"
    # retain the old compatibility argument; direct REST transfers create no local object files
    temporary_root: Path | None = None
    _credentials_cache: tuple[str, str, datetime | None] | None = field(default=None, init=False, repr=False)

    def _direct_object_url(self, key: str, account_id: str) -> str:
        """Build the REST object URL while preserving literal key slashes."""

        encoded_account = urllib.parse.quote(account_id, safe="")
        encoded_bucket = urllib.parse.quote(self.destination.bucket, safe="")
        encoded_key = urllib.parse.quote(_validate_key(key), safe="/-_.~")
        return (
            f"{self.api_base_url.rstrip('/')}/accounts/{encoded_account}/r2/buckets/"
            f"{encoded_bucket}/objects/{encoded_key}"
        )

    def _direct_credentials(self, *, force_refresh: bool = False) -> tuple[str, str]:
        """Resolve one cached account/token pair, refreshing Wrangler OAuth once if needed."""

        now = datetime.now(timezone.utc)
        if self._credentials_cache is not None and not force_refresh:
            account_id, token, expiry = self._credentials_cache
            if expiry is None or expiry > now:
                return account_id, token
            self._credentials_cache = None
        if self.api_token is not None and (not isinstance(self.api_token, str) or not self.api_token.strip()):
            raise ContractError("Cloudflare API token must be a non-empty string when provided")
        token = self.api_token or os.environ.get("CLOUDFLARE_API_TOKEN")
        whoami: subprocess.CompletedProcess[str] | None = None
        expiry: datetime | None = None
        if isinstance(token, str):
            token = token.strip()
        if not token:
            token, expiry = _load_wrangler_oauth_state(self.profile)
            if force_refresh or token is None or expiry is None or expiry <= now:
                whoami = _wrangler_whoami(self.profile)
                if whoami.returncode != 0:
                    raise UnresolvedInputError(
                        "Wrangler authentication could not be refreshed for R2 access",
                        {"status": whoami.returncode, "bucket": self.destination.bucket},
                    )
                token, expiry = _load_wrangler_oauth_state(self.profile)
                now = datetime.now(timezone.utc)
            if token is None or expiry is None or expiry <= now:
                raise UnresolvedInputError(
                    "Wrangler OAuth credentials are missing or expired",
                    {"bucket": self.destination.bucket},
                )
        account_id = self.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        account_id = account_id.strip() if isinstance(account_id, str) else account_id
        account_id = account_id or _account_id_from_endpoint(self.destination.endpoint)
        if not account_id:
            if whoami is None:
                whoami = _wrangler_whoami(self.profile)
            if whoami.returncode == 0:
                account_id = _account_id_from_wrangler_output(whoami.stdout)
        if not isinstance(account_id, str) or not _ACCOUNT_ID_RE.fullmatch(account_id):
            raise UnresolvedInputError(
                "Cloudflare R2 access requires a valid account ID",
                {"bucket": self.destination.bucket},
            )
        self._credentials_cache = (account_id, token, expiry)
        return account_id, token

    def _direct_headers(
        self,
        token: str,
        *,
        content_type: str | None = None,
        content_length: int | None = None,
    ) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": f"wrangler/{WRANGLER_VERSION}",
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        return headers

    @staticmethod
    def _response_status(response: Any) -> int:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if isinstance(status, bool) or not isinstance(status, int):
            raise UnresolvedInputError("Cloudflare R2 response status is invalid")
        return status

    def _direct_http_error(self, error: urllib.error.HTTPError, *, key: str) -> None:
        status = error.code
        try:
            error.read(DEFAULT_CHUNK_SIZE)
        except (OSError, http.client.HTTPException):
            pass
        if status in {401, 403}:
            raise UnresolvedInputError(
                "Cloudflare R2 rejected authenticated object access",
                {"bucket": self.destination.bucket, "status": status, "key": key},
            ) from error
        if status == 404:
            raise PreparationError("missing remote object", {"key": key, "status": 404}) from error
        raise PreparationError(
            "Cloudflare R2 object request failed",
            {"bucket": self.destination.bucket, "status": status, "key": key},
        ) from error

    def _put_direct_once(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str,
        force_refresh: bool = False,
    ) -> dict[str, str]:
        account_id, token = self._direct_credentials(force_refresh=force_refresh)
        headers = self._direct_headers(token, content_type=content_type, content_length=len(payload))
        request = urllib.request.Request(
            self._direct_object_url(key, account_id),
            data=payload,
            headers=headers,
            method="PUT",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = self._response_status(response)
                if not 200 <= status < 300:
                    if status in {401, 403}:
                        raise UnresolvedInputError(
                            "Cloudflare R2 rejected authenticated object upload",
                            {"bucket": self.destination.bucket, "status": status, "key": key},
                        )
                    raise PreparationError(
                        "Cloudflare R2 object upload failed",
                        {"bucket": self.destination.bucket, "status": status, "key": key},
                    )
                response_headers = getattr(response, "headers", {})
                etag = ""
                if hasattr(response_headers, "get"):
                    etag = str(response_headers.get("etag") or response_headers.get("ETag") or "").strip('"')
                _read_response_bounded(response, max_bytes=DEFAULT_MAX_INVENTORY_RESPONSE_BYTES)
        except urllib.error.HTTPError as error:
            self._direct_http_error(error, key=key)
        except urllib.error.URLError as error:
            raise UnresolvedInputError(
                "Cloudflare R2 object upload could not connect",
                {"bucket": self.destination.bucket, "key": key},
            ) from error
        except (OSError, http.client.HTTPException, TimeoutError) as error:
            raise UnresolvedInputError(
                "Cloudflare R2 object upload was interrupted",
                {"bucket": self.destination.bucket, "key": key},
            ) from error
        return {"key": key, "etag": etag}

    def put_bytes(self, key: str, payload: bytes, *, content_type: str = "application/octet-stream") -> dict[str, str]:
        write = ContentAddressedWrite.validate(self.destination, key, payload, content_type=content_type)
        payload = write.payload
        if len(payload) > WRANGLER_R2_REST_MAX_UPLOAD_BYTES:
            raise UnresolvedInputError(
                "Cloudflare REST R2 upload exceeds its 300 MiB limit",
                {
                    "key": key,
                    "size": len(payload),
                    "max_bytes": WRANGLER_R2_REST_MAX_UPLOAD_BYTES,
                },
            )
        if self.exists(write.key):
            remote = self.get_bytes(write.key)
            if remote != payload:
                raise PreparationError("same-key different-content fails", {"key": write.key})
            return {"key": write.key, "etag": ""}
        try:
            return self._put_direct_once(write.key, payload, content_type=content_type)
        except UnresolvedInputError as error:
            if (error.details or {}).get("status") not in {401, 403}:
                raise
            self._credentials_cache = None
            return self._put_direct_once(write.key, payload, content_type=content_type, force_refresh=True)

    def put_content_addressed(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> dict[str, str]:
        """Validate content-addressed bytes before using the direct REST writer."""

        write = ContentAddressedWrite.validate(self.destination, key, payload, content_type=content_type)
        return self.put_bytes(write.key, write.payload, content_type=write.content_type)

    def _iter_bytes_once(
        self,
        key: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
        force_refresh: bool = False,
    ) -> Iterator[bytes]:
        _validate_key(key)
        chunk_size = _positive_int(chunk_size, "chunk_size")
        if max_bytes is not None:
            max_bytes = _positive_int(max_bytes, "max_bytes")
        account_id, token = self._direct_credentials(force_refresh=force_refresh)
        request = urllib.request.Request(
            self._direct_object_url(key, account_id),
            headers=self._direct_headers(token),
            method="GET",
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            self._direct_http_error(error, key=key)
        except urllib.error.URLError as error:
            raise UnresolvedInputError(
                "Cloudflare R2 object read could not connect",
                {"bucket": self.destination.bucket, "key": key},
            ) from error
        except (OSError, http.client.HTTPException, TimeoutError) as error:
            raise UnresolvedInputError(
                "Cloudflare R2 object read was interrupted",
                {"bucket": self.destination.bucket, "key": key},
            ) from error
        with response:
            status = self._response_status(response)
            if status == 404:
                raise PreparationError("missing remote object", {"key": key, "status": 404})
            if status in {401, 403}:
                raise UnresolvedInputError(
                    "Cloudflare R2 rejected authenticated object access",
                    {"bucket": self.destination.bucket, "status": status, "key": key},
                )
            if status != 200:
                raise PreparationError(
                    "Cloudflare R2 object read failed",
                    {"bucket": self.destination.bucket, "status": status, "key": key},
                )
            total = 0
            try:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise PreparationError("Cloudflare R2 object response returned a non-byte chunk")
                    chunk = bytes(chunk)
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise PreparationError(
                            "remote object exceeds the configured read bound",
                            {"max_bytes": max_bytes, "key": key},
                        )
                    yield chunk
            except PreparationError:
                raise
            except (OSError, http.client.HTTPException, urllib.error.URLError, TimeoutError) as error:
                raise UnresolvedInputError(
                    "Cloudflare R2 object read was interrupted",
                    {"bucket": self.destination.bucket, "key": key},
                ) from error

    def iter_bytes(
        self,
        key: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
    ) -> Iterator[bytes]:
        """Read an object, retrying one authenticated request after OAuth refresh."""

        yielded = False
        try:
            for chunk in self._iter_bytes_once(key, chunk_size=chunk_size, max_bytes=max_bytes):
                yielded = True
                yield chunk
        except UnresolvedInputError as error:
            if yielded or (error.details or {}).get("status") not in {401, 403}:
                raise
            self._credentials_cache = None
            yield from self._iter_bytes_once(
                key,
                chunk_size=chunk_size,
                max_bytes=max_bytes,
                force_refresh=True,
            )

    def get_bytes(self, key: str) -> bytes:
        return _collect_chunks(
            self.iter_bytes(key, chunk_size=DEFAULT_CHUNK_SIZE, max_bytes=DEFAULT_MAX_OBJECT_BYTES),
            max_bytes=DEFAULT_MAX_OBJECT_BYTES,
        )

    def head(self, key: str) -> dict[str, str]:
        """Return only metadata that Wrangler can substantiate without inventing provider facts."""

        payload = self.get_bytes(key)
        return {
            "etag": "",
            "size": str(len(payload)),
            "encryption": self.encryption_at_rest or "",
        }

    def list_prefix(self, prefix: str, *, max_keys: int | None = DEFAULT_MAX_INVENTORY_KEYS) -> list[str]:
        """List a bounded prefix through Cloudflare's authenticated R2 API."""

        prefix = _validate_prefix(prefix)
        if max_keys is not None:
            max_keys = _positive_int(max_keys, "max_keys")
        keys: list[str] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            account_id, token = self._cloudflare_credentials()
            remaining = None if max_keys is None else max_keys - len(keys)
            if remaining is not None and remaining <= 0:
                raise PreparationError("remote inventory exceeds the configured key bound", {"max_keys": max_keys})
            query: dict[str, str] = {
                "prefix": prefix,
                "per_page": str(min(1000, remaining)) if remaining is not None else "1000",
            }
            if cursor:
                if cursor in seen_cursors:
                    raise UnresolvedInputError(
                        "Cloudflare R2 inventory pagination repeated a cursor", {"prefix": prefix}
                    )
                seen_cursors.add(cursor)
                query["cursor"] = cursor
            payload = self._cloudflare_api_list(account_id, token, query)
            result = payload.get("result")
            if not isinstance(result, list):
                raise PreparationError("Cloudflare R2 inventory result is not an array", {"prefix": prefix})
            for item in result:
                if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
                    raise PreparationError("Cloudflare R2 inventory contains an invalid object", {"prefix": prefix})
                keys.append(_validate_key(item["key"], "remote inventory key"))
                if max_keys is not None and len(keys) > max_keys:
                    raise PreparationError("remote inventory exceeds the configured key bound", {"max_keys": max_keys})
            result_info = payload.get("result_info")
            page_limit = int(query["per_page"])
            if result_info is None:
                if len(result) < page_limit:
                    break
                raise PreparationError("Cloudflare R2 inventory pagination metadata is missing", {"prefix": prefix})
            if not isinstance(result_info, Mapping):
                raise PreparationError("Cloudflare R2 inventory pagination metadata is invalid", {"prefix": prefix})
            truncated = result_info.get("is_truncated")
            if not isinstance(truncated, bool):
                if len(result) < page_limit:
                    break
                raise PreparationError("Cloudflare R2 inventory pagination metadata is invalid", {"prefix": prefix})
            if not truncated:
                break
            next_cursor = result_info.get("cursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                raise UnresolvedInputError("Cloudflare R2 inventory pagination is incomplete", {"prefix": prefix})
            cursor = next_cursor
        if len(set(keys)) != len(keys):
            raise PreparationError("Cloudflare R2 inventory contains duplicate keys", {"prefix": prefix})
        return keys

    def _cloudflare_credentials(self) -> tuple[str, str]:
        """Resolve API credentials without exposing token values in errors or receipts."""

        return self._direct_credentials()

    def _cloudflare_api_list(
        self,
        account_id: str,
        token: str,
        query: Mapping[str, str],
    ) -> dict[str, object]:
        return self._cloudflare_api_json(
            account_id,
            token,
            resource="/objects",
            query=query,
            action="inventory",
        )

    def _cloudflare_api_json(
        self,
        account_id: str,
        token: str,
        *,
        resource: str,
        query: Mapping[str, str] | None = None,
        action: str,
    ) -> dict[str, object]:
        try:
            return self._cloudflare_api_json_once(
                account_id,
                token,
                resource=resource,
                query=query,
                action=action,
            )
        except UnresolvedInputError as error:
            if (error.details or {}).get("status") not in {401, 403}:
                raise
            self._credentials_cache = None
            refreshed_account, refreshed_token = self._direct_credentials(force_refresh=True)
            return self._cloudflare_api_json_once(
                refreshed_account,
                refreshed_token,
                resource=resource,
                query=query,
                action=action,
            )

    def _cloudflare_api_json_once(
        self,
        account_id: str,
        token: str,
        *,
        resource: str,
        query: Mapping[str, str] | None = None,
        action: str,
    ) -> dict[str, object]:
        base = self.api_base_url.rstrip("/")
        path = "/accounts/{}/r2/buckets/{}{}".format(
            urllib.parse.quote(account_id, safe=""),
            urllib.parse.quote(self.destination.bucket, safe=""),
            resource,
        )
        encoded_query = urllib.parse.urlencode(query or {})
        url = f"{base}{path}"
        if encoded_query:
            url += f"?{encoded_query}"
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = _read_response_bounded(response, max_bytes=DEFAULT_MAX_INVENTORY_RESPONSE_BYTES)
                status = response.status
        except urllib.error.HTTPError as error:
            error.read(DEFAULT_CHUNK_SIZE)
            if error.code in {401, 403}:
                raise UnresolvedInputError(
                    f"Cloudflare API rejected authenticated R2 {action} access",
                    {"bucket": self.destination.bucket, "status": error.code},
                ) from error
            raise PreparationError(
                f"Cloudflare R2 {action} request failed",
                {"bucket": self.destination.bucket, "status": error.code},
            ) from error
        except urllib.error.URLError as error:
            raise UnresolvedInputError(
                f"Cloudflare R2 {action} failed to connect",
                {"reason": str(error.reason)},
            ) from error
        if not 200 <= status < 300:
            if status in {401, 403}:
                raise UnresolvedInputError(
                    f"Cloudflare API rejected authenticated R2 {action} access",
                    {"bucket": self.destination.bucket, "status": status},
                )
            raise PreparationError(
                f"Cloudflare R2 {action} request failed",
                {"bucket": self.destination.bucket, "status": status},
            )
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PreparationError(f"Cloudflare R2 {action} response is not valid JSON") from error
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise PreparationError(f"Cloudflare R2 {action} response was unsuccessful")
        return payload

    def _authenticated_bucket_evidence(self) -> dict[str, object]:
        """Prove the API credential names the configured bucket before using provider policy."""

        account_id, token = self._cloudflare_credentials()
        payload = self._cloudflare_api_json(account_id, token, resource="", action="bucket")
        result = payload.get("result")
        if not isinstance(result, Mapping) or result.get("name") != self.destination.bucket:
            raise PreparationError(
                "Cloudflare R2 bucket evidence does not name the configured bucket",
                {"bucket": self.destination.bucket},
            )
        evidence: dict[str, object] = {"name": result["name"]}
        for name in ("creation_date", "location", "jurisdiction", "storage_class"):
            value = result.get(name)
            if value is not None and not isinstance(value, str):
                raise PreparationError(
                    "Cloudflare R2 bucket evidence contains an invalid field",
                    {"bucket": self.destination.bucket, "field": name},
                )
            if value is not None:
                evidence[name] = value
        return evidence

    def _provider_encryption_evidence_after_readback(self) -> dict[str, object]:
        """Return bucket encryption policy after a caller proved object existence."""

        bucket = self._authenticated_bucket_evidence()
        configured = self.encryption_at_rest
        if configured is not None:
            if not isinstance(configured, str) or not configured.strip():
                raise ContractError("encryption_at_rest must be a non-empty string when provided")
            normalized = configured.upper().replace(" ", "").replace("_", "-")
            if normalized not in {"AES256", "AES-256", "AES-256-GCM"}:
                raise PreparationError(
                    "configured encryption evidence differs from Cloudflare R2 default policy",
                    {"bucket": self.destination.bucket},
                )
        return {
            "provider": "cloudflare-r2",
            "policy": CLOUDFLARE_R2_DEFAULT_ENCRYPTION,
            "bucket": bucket,
            "authenticated": True,
        }

    def provider_encryption_evidence(self, key: str) -> dict[str, object] | None:
        """Return provider encryption policy bound to an authenticated bucket and object."""

        _validate_key(key)
        if not self.exists(key):
            return None
        return self._provider_encryption_evidence_after_readback()

    def provider_privacy_evidence(self) -> dict[str, object]:
        """Require disabled managed public access and no custom domains."""

        account_id, token = self._cloudflare_credentials()
        managed_payload = self._cloudflare_api_json(
            account_id,
            token,
            resource="/domains/managed",
            action="managed-domain",
        )
        managed = managed_payload.get("result")
        if not isinstance(managed, Mapping) or managed.get("enabled") is not False:
            raise PreparationError(
                "Cloudflare R2 managed public access is enabled or unproven",
                {"bucket": self.destination.bucket},
            )
        custom_payload = self._cloudflare_api_json(
            account_id,
            token,
            resource="/domains/custom",
            action="custom-domain",
        )
        custom = custom_payload.get("result")
        if not isinstance(custom, Mapping) or not isinstance(custom.get("domains"), list):
            raise PreparationError(
                "Cloudflare R2 custom-domain evidence is missing",
                {"bucket": self.destination.bucket},
            )
        if custom["domains"]:
            raise PreparationError(
                "Cloudflare R2 custom domains would expose private objects",
                {"bucket": self.destination.bucket},
            )
        return {
            "provider": "cloudflare-r2",
            "authenticated": True,
            "managed_public_access_enabled": False,
            "custom_domains_count": 0,
        }

    def exists(self, key: str) -> bool:
        try:
            self.get_bytes(key)
        except UnresolvedInputError:
            raise
        except PreparationError as error:
            if (error.details or {}).get("status") != 404:
                raise
            return False
        return True

    def encryption_evidence(self, key: str) -> str | None:
        evidence = self.provider_encryption_evidence(key)
        if evidence is None:
            return None
        return json.dumps(evidence, sort_keys=True, separators=(",", ":"))

    def is_public(self, key: str) -> bool:
        status, _ = self.anonymous_get(key)
        return 200 <= status < 300

    def _anonymous_url(self, key: str = "", query: Mapping[str, str] | None = None) -> str:
        parsed = urllib.parse.urlparse(self.destination.endpoint)
        path = f"/{self.destination.bucket}"
        if key:
            path += "/" + urllib.parse.quote(_validate_key(key), safe="/-_.~")
        url = f"{parsed.scheme}://{parsed.netloc}{path}"
        encoded = urllib.parse.urlencode(query or {})
        return f"{url}?{encoded}" if encoded else url

    def anonymous_get(self, key: str) -> tuple[int, bytes]:
        request = urllib.request.Request(self._anonymous_url(key), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read(DEFAULT_CHUNK_SIZE)
        except urllib.error.HTTPError as error:
            return error.code, error.read(DEFAULT_CHUNK_SIZE)
        except urllib.error.URLError as error:
            raise UnresolvedInputError(
                "anonymous R2 GET failed to connect",
                {"reason": str(error.reason)},
            ) from error

    def anonymous_list(self, prefix: str) -> tuple[int, list[str]]:
        prefix = _validate_prefix(prefix)
        request = urllib.request.Request(
            self._anonymous_url(query={"list-type": "2", "prefix": prefix}),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read(DEFAULT_CHUNK_SIZE)
                if not 200 <= response.status < 300:
                    return response.status, []
                try:
                    root = ElementTree.fromstring(body)
                except ElementTree.ParseError:
                    return response.status, ["disclosed"]
                namespace = root.tag.split("}", 1)[0] + "}" if root.tag.startswith("{") else ""
                keys = [item.text or "" for item in root.findall(f".//{namespace}Key") if item.text]
                return response.status, keys or ["disclosed"]
        except urllib.error.HTTPError as error:
            return error.code, []
        except urllib.error.URLError as error:
            raise UnresolvedInputError(
                "anonymous R2 listing failed to connect",
                {"reason": str(error.reason)},
            ) from error


def backend_from_destination(destination: ObjectStoreDestination) -> StorageBackend:
    """Build the remote backend for a parsed destination."""

    reference = destination.credential_reference
    if reference == "wrangler" or reference.startswith("wrangler:"):
        return WranglerR2Backend(destination=destination)
    if reference.startswith("rclone:"):
        return S3CompatibleBackend.from_rclone(destination)
    raise UnresolvedInputError(
        "unsupported credential reference",
        {"reference": reference.split(":", 1)[0] if ":" in reference else "missing"},
    )


def object_key(source: str, version: str, selection: str, digest: str, ext: str) -> str:
    """Return a content-addressed immutable key."""

    for value, label in ((source, "source"), (version, "version"), (selection, "selection")):
        if not isinstance(value, str) or not _SEGMENT_RE.fullmatch(value):
            raise ContractError(f"object key {label} is invalid")
    digest = require_content_hash(digest, "object sha256")
    suffix = ext.lstrip(".")
    if not suffix or "/" in suffix or any(not _SEGMENT_RE.fullmatch(part) for part in suffix.split("/")):
        raise ContractError("object key extension is invalid")
    return f"datasets/{source}/{version}/{selection}/objects/{digest}.{suffix}"


def _readback(
    backend: StorageBackend,
    key: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
) -> ReadbackProof:
    key = _validate_key(key)
    digest = hashlib.sha256()
    size = 0
    for chunk in _iter_backend_bytes(backend, key, chunk_size=chunk_size, max_bytes=max_bytes):
        digest.update(chunk)
        size += len(chunk)
    actual = digest.hexdigest()
    if expected_size is not None and size != expected_size:
        raise PreparationError(
            "remote object size differs from the accepted identity",
            {"key": key, "actual_size": size, "expected_size": expected_size},
        )
    if expected_sha256 is not None:
        expected = require_content_hash(expected_sha256, "expected sha256")
        if actual != expected:
            raise PreparationError(
                "partial/corrupt/missing remote content cannot commit",
                {"key": key, "actual": actual, "expected": expected},
            )
    return ReadbackProof(key=key, sha256=actual, size=size)


def _readback_payload(
    backend: StorageBackend,
    key: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
) -> tuple[ReadbackProof, bytes]:
    """Read one bounded object once while retaining a small caller-selected payload."""

    key = _validate_key(key)
    digest = hashlib.sha256()
    size = 0
    payload = bytearray()
    for chunk in _iter_backend_bytes(backend, key, chunk_size=chunk_size, max_bytes=max_bytes):
        digest.update(chunk)
        size += len(chunk)
        payload.extend(chunk)
    actual = digest.hexdigest()
    if expected_size is not None and size != expected_size:
        raise PreparationError(
            "remote object size differs from the accepted identity",
            {"key": key, "actual_size": size, "expected_size": expected_size},
        )
    if expected_sha256 is not None:
        expected = require_content_hash(expected_sha256, "expected sha256")
        if actual != expected:
            raise PreparationError(
                "partial/corrupt/missing remote content cannot commit",
                {"key": key, "actual": actual, "expected": expected},
            )
    return ReadbackProof(key=key, sha256=actual, size=size), bytes(payload)


def full_readback_sha256(
    backend: StorageBackend,
    key: str,
    expected: str,
    *,
    expected_size: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
) -> str:
    """Recompute SHA-256 over full remote bytes. HEAD/ETag cannot satisfy this."""

    return _readback(
        backend,
        key,
        expected_sha256=expected,
        expected_size=expected_size,
        chunk_size=chunk_size,
        max_bytes=max_bytes,
    ).sha256


_NO_PROVIDER_EVIDENCE = object()


def _assert_anonymous_denied(status: int, label: str) -> None:
    if 200 <= status < 300:
        raise PreparationError("public/readable anonymous objects fail privacy check", {"check": label})
    if 300 <= status < 400:
        raise PreparationError("anonymous access redirected instead of being denied", {"check": label})
    if status >= 500:
        raise UnresolvedInputError("anonymous privacy check could not be resolved", {"check": label, "status": status})


def _provider_privacy_evidence(backend: StorageBackend) -> object:
    checker = getattr(backend, "provider_privacy_evidence", None)
    if not callable(checker):
        return _NO_PROVIDER_EVIDENCE
    return checker()


def _privacy_evidence_after_readback(
    backend: StorageBackend,
    proof: ReadbackProof,
    prefix: str,
    *,
    anonymous_list: tuple[int, Sequence[str]] | None = None,
    provider_evidence: object = _NO_PROVIDER_EVIDENCE,
) -> dict[str, object]:
    """Check anonymous privacy after authenticated bytes have already been proved."""

    prefix = _validate_prefix(prefix)
    if anonymous_list is None:
        list_status, _ = backend.anonymous_list(prefix)
    else:
        list_status = anonymous_list[0]
    get_status, _ = backend.anonymous_get(proof.key)
    _assert_anonymous_denied(get_status, "anonymous-get")
    _assert_anonymous_denied(list_status, "anonymous-list")
    result: dict[str, object] = {
        "authenticated_bytes": proof.size,
        "authenticated_sha256": proof.sha256,
        "anonymous_get_status": get_status,
        "anonymous_list_status": list_status,
        "denied_anonymous": True,
    }
    if provider_evidence is _NO_PROVIDER_EVIDENCE:
        provider_evidence = _provider_privacy_evidence(backend)
    if provider_evidence is not _NO_PROVIDER_EVIDENCE:
        result["provider"] = provider_evidence
    return result


def _encryption_evidence_after_readback(
    backend: StorageBackend,
    proof: ReadbackProof,
    *,
    wrangler_policy: object = _NO_PROVIDER_EVIDENCE,
) -> tuple[str, object]:
    """Read encryption metadata without re-reading an already proved Wrangler object."""

    if isinstance(backend, WranglerR2Backend):
        if wrangler_policy is _NO_PROVIDER_EVIDENCE:
            wrangler_policy = backend._provider_encryption_evidence_after_readback()
        if not isinstance(wrangler_policy, Mapping):
            raise PreparationError("missing encryption evidence", {"key": proof.key})
        return json.dumps(wrangler_policy, sort_keys=True, separators=(",", ":")), wrangler_policy
    metadata = backend.head(proof.key)
    if not isinstance(metadata, Mapping):
        raise PreparationError("remote metadata is not an object", {"key": proof.key})
    encryption = metadata.get("encryption")
    if not isinstance(encryption, str) or not encryption:
        encryption = backend.encryption_evidence(proof.key)
    if not isinstance(encryption, str) or not encryption:
        raise PreparationError("missing encryption evidence", {"key": proof.key})
    return encryption, wrangler_policy


class _RemoteEvidenceSession:
    """Cache operation-scoped privacy and provider policy evidence."""

    __slots__ = ("backend", "_anonymous_list", "_privacy_prefix", "_provider_privacy", "_wrangler_policy")

    def __init__(self, backend: StorageBackend, *, privacy_prefix: str | None = None) -> None:
        self.backend = backend
        self._privacy_prefix = (
            _validate_prefix(privacy_prefix, "privacy_prefix") if privacy_prefix is not None else None
        )
        self._anonymous_list: tuple[int, Sequence[str]] | None = None
        self._provider_privacy: object = _NO_PROVIDER_EVIDENCE
        self._wrangler_policy: object = _NO_PROVIDER_EVIDENCE

    def encryption_for(self, proof: ReadbackProof) -> str:
        encryption, self._wrangler_policy = _encryption_evidence_after_readback(
            self.backend,
            proof,
            wrangler_policy=self._wrangler_policy,
        )
        return encryption

    def privacy_for(self, proof: ReadbackProof) -> dict[str, object]:
        if self._privacy_prefix is None:
            raise ContractError("privacy evidence requires a privacy prefix")
        if self._anonymous_list is None:
            self._anonymous_list = self.backend.anonymous_list(self._privacy_prefix)
            _assert_anonymous_denied(self._anonymous_list[0], "anonymous-list")
        if self._provider_privacy is _NO_PROVIDER_EVIDENCE:
            self._provider_privacy = _provider_privacy_evidence(self.backend)
        return _privacy_evidence_after_readback(
            self.backend,
            proof,
            self._privacy_prefix,
            anonymous_list=self._anonymous_list,
            provider_evidence=self._provider_privacy,
        )


def _verified_control_keys(
    backend: StorageBackend,
    prefix: str,
    evidence: _RemoteEvidenceSession,
    inventory_keys: Sequence[str],
) -> tuple[str, ...]:
    """Verify known task-control canaries before excluding them from release data."""

    present = set(inventory_keys)
    verified = []
    for relative, payload in _R2_CONTROL_CANARIES:
        key = f"{prefix}/{relative}"
        if key not in present:
            continue
        proof = _readback(
            backend,
            key,
            expected_sha256=sha256_bytes(payload),
            expected_size=len(payload),
            max_bytes=len(payload),
        )
        evidence.encryption_for(proof)
        evidence.privacy_for(proof)
        verified.append(key)
    return tuple(verified)


def assert_private_access(backend: StorageBackend, key: str, prefix: str) -> dict[str, object]:
    """Authenticated read must work; anonymous GET/list must not disclose contents."""

    proof = _readback(backend, key)
    session = _RemoteEvidenceSession(backend, privacy_prefix=prefix)
    return session.privacy_for(proof)


def _object_identity(item: Mapping[str, object], label: str = "object") -> dict[str, object]:
    if not isinstance(item, Mapping):
        raise ContractError(f"{label} must be an object")
    raw_key = item.get("key")
    if not isinstance(raw_key, str):
        raise ContractError(f"{label}.key must be a string")
    key = _validate_key(raw_key, f"{label}.key")
    digest = require_content_hash(item.get("sha256"), f"{label}.sha256")
    size = _positive_int(item.get("size"), f"{label}.size")
    identity: dict[str, object] = {"key": key, "sha256": digest, "size": size}
    for name in ("purpose", "parent_id", "source", "version", "codec"):
        if name in item:
            value = item[name]
            if value is not None and not isinstance(value, str):
                raise ContractError(f"{label}.{name} must be a string or null")
            identity[name] = value
    return identity


def _compare_object_identity(expected: Mapping[str, object], actual: Mapping[str, object], label: str) -> None:
    expected_identity = _object_identity(expected, f"{label}.expected")
    actual_identity = _object_identity(actual, f"{label}.actual")
    if expected_identity != actual_identity:
        raise PreparationError(
            "uploaded inventory differs from accepted selection",
            {"key": expected_identity["key"], "expected": expected_identity, "actual": actual_identity},
        )


def _list_prefix_bounded(
    backend: StorageBackend,
    prefix: str,
    *,
    max_keys: int | None = DEFAULT_MAX_INVENTORY_KEYS,
) -> list[str]:
    """Call old test doubles safely while enforcing the new inventory bound."""

    try:
        result = backend.list_prefix(prefix, max_keys=max_keys)
    except TypeError:
        result = backend.list_prefix(prefix)
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise PreparationError("remote inventory is not a key array", {"prefix": prefix})
    keys = [_validate_key(str(key), "remote inventory key") for key in result]
    if max_keys is not None and len(keys) > max_keys:
        raise PreparationError("remote inventory exceeds the configured key bound", {"max_keys": max_keys})
    return keys


def _known_marker_keys(
    backend: StorageBackend,
    prefix: str,
    *,
    kind: str,
    max_keys: int | None = DEFAULT_MAX_INVENTORY_KEYS,
    inventory_keys: Sequence[str] | None = None,
) -> list[str]:
    """Return existing content-addressed markers that can be explained in a retry."""

    marker_prefix = f"{prefix}/{MARKER_DIRECTORY}/{kind}/"
    keys = (
        _list_prefix_bounded(backend, prefix, max_keys=max_keys)
        if inventory_keys is None
        else [_validate_key(key, "remote inventory key") for key in inventory_keys]
    )
    known = []
    for key in keys:
        if not key.startswith(marker_prefix) or not key.endswith(".json"):
            continue
        digest = key[len(marker_prefix) : -5]
        if not is_placeholder_hash(digest):
            marker_bytes = backend.get_bytes(key)
            _load_canonical_marker(marker_bytes, kind=kind, digest=digest)
            known.append(key)
    return sorted(known)


def _stable_inventory(inventory: Mapping[str, object], expected_keys: Sequence[str]) -> dict[str, object]:
    """Keep content-addressed marker identities stable when explained markers already exist."""

    keys = sorted(set(expected_keys))
    return {
        "prefix": inventory["prefix"],
        "expected_keys": keys,
        "allowed_keys": [],
        "actual_keys": keys,
        "expected_count": len(keys),
        "actual_count": len(keys),
        "complete": True,
    }


def verify_inventory_closure(
    backend: StorageBackend,
    expected: Sequence[Mapping[str, object]],
    *,
    prefix: str,
    allowed_keys: Sequence[str] = (),
    max_keys: int | None = DEFAULT_MAX_INVENTORY_KEYS,
) -> dict[str, object]:
    """Prove that a bounded task prefix contains exactly the expected objects and explained markers."""

    prefix = _validate_prefix(prefix)
    if max_keys is not None:
        max_keys = _positive_int(max_keys, "max_keys")
    actual = _list_prefix_bounded(backend, prefix, max_keys=max_keys)
    return _verify_inventory_closure(expected, prefix=prefix, allowed_keys=allowed_keys, actual_keys=actual)


def _verify_inventory_closure(
    expected: Sequence[Mapping[str, object]],
    *,
    prefix: str,
    allowed_keys: Sequence[str],
    actual_keys: Sequence[str],
) -> dict[str, object]:
    """Validate inventory closure against one operation-owned inventory snapshot."""

    expected_keys = []
    for index, item in enumerate(expected):
        identity = _object_identity(item, f"expected[{index}]")
        key = str(identity["key"])
        if not key.startswith(prefix + "/") and key != prefix:
            raise ContractError("expected object lies outside the inventory prefix", {"key": key, "prefix": prefix})
        expected_keys.append(key)
    if len(set(expected_keys)) != len(expected_keys):
        raise PreparationError("expected inventory contains duplicate keys")
    allowed = {_validate_key(str(key), "allowed inventory key") for key in allowed_keys}
    outside_allowed = sorted(key for key in allowed if not key.startswith(prefix + "/") and key != prefix)
    if outside_allowed:
        raise ContractError("allowed inventory key lies outside the inventory prefix", {"keys": outside_allowed[:10]})
    actual = [_validate_key(str(key), "remote inventory key") for key in actual_keys]
    if len(set(actual)) != len(actual):
        raise PreparationError("remote inventory contains duplicate keys", {"prefix": prefix})
    actual_set = set(actual)
    expected_set = set(expected_keys)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set - allowed)
    if missing or extra:
        raise PreparationError(
            "remote inventory is not an exact accepted-selection closure",
            {"prefix": prefix, "missing": missing[:20], "extra": extra[:20]},
        )
    return {
        "prefix": prefix,
        "expected_keys": sorted(expected_set),
        "allowed_keys": sorted(actual_set.intersection(allowed)),
        "actual_keys": sorted(actual_set),
        "expected_count": len(expected_set),
        "actual_count": len(actual_set),
        "complete": True,
    }


def mark_readback_verified(
    planned: Mapping[str, object],
    *,
    backend: StorageBackend,
    expected_sha256: str,
    prefix: str | None = None,
    max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
) -> dict[str, object]:
    """Advance an object to readback-verified only after bounded full-byte checks."""

    try:
        current = ObjectState(str(planned["state"]))
    except (KeyError, ValueError) as error:
        raise ContractError("object state is unknown") from error
    expected = require_content_hash(expected_sha256, "expected sha256")
    if "sha256" in planned and require_content_hash(planned["sha256"], "planned sha256") != expected:
        raise PreparationError("planned object hash differs from expected hash", {"key": planned.get("key")})
    if current is ObjectState.PLANNED:
        assert_state_transition(current, ObjectState.UPLOADING, OBJECT_TRANSITIONS, "object")
        current = ObjectState.UPLOADING
    if current is ObjectState.UPLOADING:
        assert_state_transition(current, ObjectState.UPLOADED, OBJECT_TRANSITIONS, "object")
        current = ObjectState.UPLOADED
    raw_key = planned.get("key")
    if not isinstance(raw_key, str):
        raise ContractError("planned.key must be a string")
    key = _validate_key(raw_key)
    expected_size = _positive_int(planned.get("size"), "planned.size")
    head = backend.head(key)
    head_size = head.get("size")
    if head_size:
        try:
            if int(head_size) != expected_size:
                raise PreparationError(
                    "remote object size differs from the accepted identity",
                    {"key": key, "actual_size": head_size, "expected_size": expected_size},
                )
        except ValueError as error:
            raise PreparationError("remote metadata contains an invalid object size", {"key": key}) from error
    proof = _readback(backend, key, expected_sha256=expected, expected_size=expected_size, max_bytes=max_bytes)
    evidence = _RemoteEvidenceSession(backend, privacy_prefix=prefix)
    encryption = evidence.encryption_for(proof)
    if prefix is not None:
        privacy = evidence.privacy_for(proof)
    elif backend.is_public(key):
        raise PreparationError("public/readable anonymous objects fail privacy check", {"key": key})
    else:
        privacy = None
    if current is not ObjectState.READBACK_VERIFIED:
        assert_state_transition(current, ObjectState.READBACK_VERIFIED, OBJECT_TRANSITIONS, "object")
    payload = {
        "key": key,
        "sha256": proof.sha256,
        "size": proof.size,
        "purpose": planned.get("purpose"),
        "parent_id": planned.get("parent_id"),
        "source": planned.get("source"),
        "state": ObjectState.READBACK_VERIFIED.value,
        "codec": planned.get("codec"),
        "etag": head.get("etag") or None,
        "encryption": encryption,
        "public": False,
    }
    receipt = parse_object_receipt(payload)
    result = {
        "key": receipt.key,
        "sha256": receipt.sha256,
        "size": receipt.size,
        "purpose": receipt.purpose,
        "parent_id": receipt.parent_id,
        "source": receipt.source,
        "state": receipt.state.value,
        "codec": receipt.codec,
        "etag": receipt.etag,
        "encryption": receipt.encryption,
        "public": False,
    }
    if "version" in planned:
        result["version"] = planned["version"]
    if privacy is not None:
        result["privacy"] = privacy
    return result


def upload_success_is_not_proof(upload_response: Mapping[str, object]) -> None:
    """Reject eviction or commit based on an upload-success response alone."""

    if not isinstance(upload_response, Mapping):
        raise PreparationError("upload success cannot enable eviction or commit")
    readback = upload_response.get("readback_sha256")
    if upload_response.get("ok") and not readback:
        raise PreparationError("upload success cannot enable eviction or commit")
    if upload_response.get("etag") and not readback:
        raise PreparationError("ETag alone cannot enable eviction")
    if upload_response.get("head") and not readback:
        raise PreparationError("forged HEAD metadata cannot enable eviction")


def _receipt_dict(item: Mapping[str, object]) -> dict[str, object]:
    identity = _object_identity(item)
    state = ObjectState(str(item.get("state")))
    if state is not ObjectState.READBACK_VERIFIED:
        raise PreparationError("partial/corrupt/missing remote content cannot commit", {"key": identity["key"]})
    encryption = item.get("encryption")
    if not isinstance(encryption, str) or not encryption:
        raise PreparationError("missing encryption evidence", {"key": identity["key"]})
    if item.get("public") is not False:
        raise PreparationError("public/readable anonymous objects fail privacy check", {"key": identity["key"]})
    payload = {
        **{key: value for key, value in identity.items() if key != "version"},
        "purpose": item.get("purpose"),
        "parent_id": item.get("parent_id"),
        "source": item.get("source"),
        "state": state.value,
        "codec": item.get("codec"),
        "etag": item.get("etag") if isinstance(item.get("etag"), str) else None,
        "encryption": encryption,
        "public": False,
    }
    receipt = parse_object_receipt(payload)
    result = {
        "key": receipt.key,
        "sha256": receipt.sha256,
        "size": receipt.size,
        "purpose": receipt.purpose,
        "parent_id": receipt.parent_id,
        "source": receipt.source,
        "state": receipt.state.value,
        "codec": receipt.codec,
        "etag": receipt.etag,
        "encryption": receipt.encryption,
        "public": False,
    }
    if "version" in identity:
        result["version"] = identity["version"]
    return result


def _marker_key(scope: str, kind: str, digest: str) -> str:
    if kind not in _MARKER_DEFINITIONS:
        raise ContractError("marker kind is unknown", {"kind": kind})
    return f"{_validate_prefix(scope)}/{MARKER_DIRECTORY}/{kind}/{require_content_hash(digest, 'marker digest')}.json"


def _validate_marker_publication_scope(marker_key: str, privacy_prefix: str, kind: str) -> None:
    """Bind a commit marker to the inventory prefix it explains."""

    key = _validate_key(marker_key, "marker key")
    prefix = _validate_prefix(privacy_prefix, "marker inventory prefix")
    if not key.startswith(prefix + "/"):
        raise ContractError(
            "commit marker must stay under its inventory prefix",
            {"key": key, "prefix": prefix},
        )
    relative = key[len(prefix) + 1 :]
    match = _MARKER_KEY_RE.fullmatch(relative)
    if match is None or match.group("kind") != kind or match.group("scope") is not None:
        raise ContractError(
            "commit marker path does not match its inventory prefix",
            {"key": key, "prefix": prefix, "kind": kind},
        )


def _publish_marker(
    backend: StorageBackend,
    payload: Mapping[str, object],
    *,
    marker_key: str,
    privacy_prefix: str,
    marker_prefix: str | None = None,
) -> dict[str, object]:
    marker_key = _validate_key(marker_key, "marker key")
    marker_kind = "releases" if payload.get("schema") == "speakrs-remote-release-v1" else "batches"
    _validate_marker_publication_scope(marker_key, marker_prefix or privacy_prefix, marker_kind)
    marker_bytes = _canonical_json_bytes(payload)
    marker_sha256 = sha256_bytes(marker_bytes)
    backend.put_content_addressed(marker_key, marker_bytes, content_type="application/json")
    proof = _readback(
        backend,
        marker_key,
        expected_sha256=marker_sha256,
        expected_size=len(marker_bytes),
    )
    evidence = _RemoteEvidenceSession(backend, privacy_prefix=privacy_prefix)
    encryption = evidence.encryption_for(proof)
    privacy = evidence.privacy_for(proof)
    return {
        "key": proof.key,
        "sha256": proof.sha256,
        "size": proof.size,
        "state": "committed",
        "encryption": encryption,
        "privacy": privacy,
    }


def _read_marker(backend: StorageBackend, reference: Mapping[str, object]) -> dict[str, object]:
    key = _validate_key(str(reference.get("key", "")), "marker key")
    digest = require_content_hash(reference.get("sha256"), "marker sha256")
    size = _positive_int(reference.get("size"), "marker size")
    proof, marker_bytes = _readback_payload(backend, key, expected_sha256=digest, expected_size=size)
    marker_match = re.search(
        r"(?:^|/)(?P<relative>_commits/(?P<kind>batches|releases)/[0-9a-fA-F]{64}\.json)$",
        key,
    )
    if marker_match is None:
        raise ContractError("remote commit marker key is not content-addressed", {"key": key})
    try:
        payload = _load_canonical_marker(
            marker_bytes,
            kind=marker_match.group("kind"),
            digest=key.rsplit("/", 1)[-1][:-5],
        )
    except ContractError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise PreparationError("remote commit marker is not valid JSON", {"key": key}) from error
    if proof.sha256 != digest:
        raise PreparationError("remote commit marker hash mismatch", {"key": key})
    return dict(payload)


def _marker_object_records(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    records: list[Mapping[str, object]] = []
    direct = payload.get("objects")
    if isinstance(direct, list):
        records.extend(item for item in direct if isinstance(item, Mapping))
    batches = payload.get("batches")
    if isinstance(batches, list):
        for batch in batches:
            if isinstance(batch, Mapping):
                records.extend(_marker_object_records(batch))
    return records


def _marker_acceptance_hashes(payload: Mapping[str, object]) -> set[str]:
    values: set[str] = set()
    for key in ("acceptance_sha256", "selection_acceptance_sha256"):
        value = payload.get(key)
        if isinstance(value, str) and not is_placeholder_hash(value):
            values.add(value.lower())
    identity = payload.get("release_identity")
    if isinstance(identity, Mapping):
        value = identity.get("acceptance_sha256")
        if isinstance(value, str) and not is_placeholder_hash(value):
            values.add(value.lower())
    batches = payload.get("batches")
    if isinstance(batches, list):
        for batch in batches:
            if isinstance(batch, Mapping):
                values.update(_marker_acceptance_hashes(batch))
    return values


_BATCH_SEAL_TOKEN = object()


@dataclass(frozen=True)
class _VerifiedBatch:
    """In-memory proof bundle accepted only by the private batch sealer."""

    receipts: tuple[Mapping[str, object], ...]
    inventory: Mapping[str, object]
    token: object


def _validate_batch_contract(
    *,
    state: BatchState,
    label_policy_id: str,
    split_id: str,
    qa_policy_sha256: str,
    acceptance_sha256: str | None,
    backend: StorageBackend | None,
    inventory_prefix: str | None,
) -> tuple[str, str]:
    if is_placeholder_hash(qa_policy_sha256):
        raise ContractError("placeholder hashes cannot seal a real release")
    if not isinstance(label_policy_id, str) or not label_policy_id:
        raise ContractError("label_policy_id must be a non-empty string")
    if not isinstance(split_id, str) or not split_id:
        raise ContractError("split_id must be a non-empty string")
    if state not in {BatchState.DRAFT, BatchState.OBJECTS_VERIFIED, BatchState.COMMITTED}:
        raise ContractError("batch state is unknown")
    if backend is None or inventory_prefix is None:
        raise UnresolvedInputError("a backend and scoped inventory prefix are required for a remote batch commit")
    return require_content_hash(acceptance_sha256, "acceptance sha256"), _validate_prefix(
        inventory_prefix, "inventory_prefix"
    )


def _expected_batch_inventory(
    expected_objects: Sequence[Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    if expected_objects is None:
        raise ContractError("expected_objects is required; uploaded rows cannot define the expected inventory")
    if not isinstance(expected_objects, Sequence) or isinstance(expected_objects, (str, bytes)):
        raise ContractError("expected_objects must be an array")
    identities = [_object_identity(item, f"expected_objects[{index}]") for index, item in enumerate(expected_objects)]
    if not identities:
        raise PreparationError("a remote batch must contain at least one accepted object")
    expected_by_key = {str(item["key"]): item for item in identities}
    if len(expected_by_key) != len(identities):
        raise ContractError("expected_objects contains duplicate keys")
    return expected_by_key


def _uploaded_batch_inventory(
    objects: Sequence[Mapping[str, object]],
    expected_by_key: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ContractError("uploaded objects must be an array")
    actual_by_key: dict[str, Mapping[str, object]] = {}
    for item in objects:
        identity = _object_identity(item, "uploaded object")
        key = str(identity["key"])
        if key in actual_by_key:
            raise PreparationError("uploaded inventory contains duplicate keys", {"key": key})
        actual_by_key[key] = item
        if item.get("state") != ObjectState.UPLOADED.value:
            raise PreparationError("remote verification requires UPLOADED objects", {"key": key})
        expected_item = expected_by_key.get(key)
        if expected_item is None:
            raise PreparationError(
                "remote batch contains an object outside the explicit accepted inventory", {"key": key}
            )
        _compare_object_identity(expected_item, item, "batch")
    if set(actual_by_key) != set(expected_by_key):
        raise PreparationError(
            "remote batch does not cover the explicit accepted inventory",
            {
                "missing": sorted(set(expected_by_key) - set(actual_by_key))[:20],
                "extra": sorted(set(actual_by_key) - set(expected_by_key))[:20],
            },
        )
    return actual_by_key


def _receipt_from_readback(
    item: Mapping[str, object],
    proof: ReadbackProof,
    *,
    encryption: str,
) -> dict[str, object]:
    identity = _object_identity(item, "uploaded object")
    payload = {
        "key": proof.key,
        "sha256": proof.sha256,
        "size": proof.size,
        "purpose": identity.get("purpose"),
        "parent_id": identity.get("parent_id"),
        "source": identity.get("source"),
        "state": ObjectState.READBACK_VERIFIED.value,
        "codec": identity.get("codec"),
        "etag": None,
        "encryption": encryption,
        "public": False,
    }
    receipt = parse_object_receipt(payload)
    result = {
        "key": receipt.key,
        "sha256": receipt.sha256,
        "size": receipt.size,
        "purpose": receipt.purpose,
        "parent_id": receipt.parent_id,
        "source": receipt.source,
        "state": receipt.state.value,
        "codec": receipt.codec,
        "etag": receipt.etag,
        "encryption": receipt.encryption,
        "public": False,
    }
    if "version" in identity:
        result["version"] = identity["version"]
    return result


def _seal_batch(
    verified: _VerifiedBatch,
    *,
    state: BatchState,
    label_policy_id: str,
    split_id: str,
    qa_policy_sha256: str,
    acceptance_sha256: str,
    backend: StorageBackend,
    inventory_prefix: str,
    privacy_prefix: str | None = None,
    marker_key: str | None = None,
) -> dict[str, object]:
    """Publish a batch marker from one in-memory verified operation."""

    if verified.token is not _BATCH_SEAL_TOKEN:
        raise TypeError("batch sealing requires an operation-owned verification bundle")
    marker_inventory = _stable_inventory(
        verified.inventory,
        [str(item["key"]) for item in verified.receipts],
    )
    base: dict[str, object] = {
        "schema": "speakrs-remote-batch-v1",
        "state": BatchState.COMMITTED.value,
        "objects": [dict(item) for item in verified.receipts],
        "label_policy_id": label_policy_id,
        "split_id": split_id,
        "qa_policy_sha256": str(qa_policy_sha256).lower(),
        "acceptance_sha256": acceptance_sha256,
        "inventory": marker_inventory,
    }
    batch_sha256 = sha256_json(base)
    resolved_marker_key = marker_key or _marker_key(inventory_prefix, "batches", batch_sha256)
    marker_payload = {**base, "batch_sha256": batch_sha256}
    marker = _publish_marker(
        backend,
        marker_payload,
        marker_key=resolved_marker_key,
        privacy_prefix=privacy_prefix or inventory_prefix,
        marker_prefix=inventory_prefix,
    )
    if state is BatchState.DRAFT:
        assert_state_transition(state, BatchState.OBJECTS_VERIFIED, BATCH_TRANSITIONS, "batch")
        state = BatchState.OBJECTS_VERIFIED
    if state is BatchState.OBJECTS_VERIFIED:
        assert_state_transition(state, BatchState.COMMITTED, BATCH_TRANSITIONS, "batch")
    elif state is not BatchState.COMMITTED:
        raise ContractError("batch state is unknown")
    result = dict(marker_payload)
    result["marker"] = marker
    return result


def verify_and_commit_batch(
    objects: Sequence[Mapping[str, object]],
    *,
    state: BatchState,
    label_policy_id: str,
    split_id: str,
    qa_policy_sha256: str,
    expected_objects: Sequence[Mapping[str, object]],
    acceptance_sha256: str,
    backend: StorageBackend,
    inventory_prefix: str,
    privacy_prefix: str,
    allowed_inventory_keys: Sequence[str] = (),
    marker_key: str | None = None,
    required_keys: Sequence[str] | None = None,
) -> dict[str, object]:
    """Verify uploaded objects once and publish an immutable batch marker."""

    del required_keys
    acceptance, inventory_scope = _validate_batch_contract(
        state=state,
        label_policy_id=label_policy_id,
        split_id=split_id,
        qa_policy_sha256=qa_policy_sha256,
        acceptance_sha256=acceptance_sha256,
        backend=backend,
        inventory_prefix=inventory_prefix,
    )
    expected_by_key = _expected_batch_inventory(expected_objects)
    uploaded_by_key = _uploaded_batch_inventory(objects, expected_by_key)
    evidence = _RemoteEvidenceSession(backend, privacy_prefix=privacy_prefix)
    receipts: list[Mapping[str, object]] = []
    for key in sorted(expected_by_key):
        expected = expected_by_key[key]
        proof = _readback(
            backend,
            key,
            expected_sha256=str(expected["sha256"]),
            expected_size=int(expected["size"]),
        )
        encryption = evidence.encryption_for(proof)
        evidence.privacy_for(proof)
        receipts.append(_receipt_from_readback(uploaded_by_key[key], proof, encryption=encryption))
    inventory = verify_inventory_closure(
        backend,
        receipts,
        prefix=inventory_scope,
        allowed_keys=[*allowed_inventory_keys, *_known_marker_keys(backend, inventory_scope, kind="batches")],
    )
    return _seal_batch(
        _VerifiedBatch(tuple(receipts), inventory, _BATCH_SEAL_TOKEN),
        state=state,
        label_policy_id=label_policy_id,
        split_id=split_id,
        qa_policy_sha256=qa_policy_sha256,
        acceptance_sha256=acceptance,
        backend=backend,
        inventory_prefix=inventory_scope,
        privacy_prefix=privacy_prefix,
        marker_key=marker_key,
    )


def commit_batch(
    objects: Sequence[Mapping[str, object]],
    *,
    state: BatchState,
    label_policy_id: str,
    split_id: str,
    qa_policy_sha256: str,
    expected_objects: Sequence[Mapping[str, object]] | None = None,
    acceptance_sha256: str | None = None,
    backend: StorageBackend | None = None,
    inventory_prefix: str | None = None,
    allowed_inventory_keys: Sequence[str] = (),
    marker_key: str | None = None,
    required_keys: Sequence[str] | None = None,
) -> dict[str, object]:
    """Read back an explicit accepted inventory and publish an immutable batch marker."""

    acceptance, prefix = _validate_batch_contract(
        state=state,
        label_policy_id=label_policy_id,
        split_id=split_id,
        qa_policy_sha256=qa_policy_sha256,
        acceptance_sha256=acceptance_sha256,
        backend=backend,
        inventory_prefix=inventory_prefix,
    )
    expected_by_key = _expected_batch_inventory(expected_objects)
    actual_by_key: dict[str, Mapping[str, object]] = {}
    for item in objects:
        identity = _object_identity(item, "uploaded object")
        key = str(identity["key"])
        if key in actual_by_key:
            raise PreparationError("uploaded inventory contains duplicate keys", {"key": key})
        actual_by_key[key] = item
        try:
            item_state = ObjectState(str(item.get("state")))
        except ValueError as error:
            raise ContractError("object state is unknown", {"key": key}) from error
        if item_state is not ObjectState.READBACK_VERIFIED:
            raise PreparationError("partial/corrupt/missing remote content cannot commit", {"key": key})
        expected_item = expected_by_key.get(key)
        if expected_item is None:
            raise PreparationError(
                "remote batch contains an object outside the explicit accepted inventory", {"key": key}
            )
        _compare_object_identity(expected_item, item, "batch")
    if set(actual_by_key) != set(expected_by_key):
        raise PreparationError(
            "remote batch does not cover the explicit accepted inventory",
            {
                "missing": sorted(set(expected_by_key) - set(actual_by_key))[:20],
                "extra": sorted(set(actual_by_key) - set(expected_by_key))[:20],
            },
        )
    expected_receipts = []
    evidence = _RemoteEvidenceSession(backend, privacy_prefix=prefix)
    for key in sorted(expected_by_key):
        item = actual_by_key[key]
        receipt = _receipt_dict(item)
        proof = _readback(
            backend,
            key,
            expected_sha256=str(receipt["sha256"]),
            expected_size=int(receipt["size"]),
        )
        encryption = evidence.encryption_for(proof)
        if encryption != receipt.get("encryption"):
            raise PreparationError("remote encryption evidence differs from the object receipt", {"key": key})
        evidence.privacy_for(proof)
        expected_receipts.append(receipt)
    inventory = verify_inventory_closure(
        backend,
        expected_receipts,
        prefix=prefix,
        allowed_keys=[*allowed_inventory_keys, *_known_marker_keys(backend, prefix, kind="batches")],
    )
    return _seal_batch(
        _VerifiedBatch(tuple(expected_receipts), inventory, _BATCH_SEAL_TOKEN),
        state=state,
        label_policy_id=label_policy_id,
        split_id=split_id,
        qa_policy_sha256=qa_policy_sha256,
        acceptance_sha256=acceptance,
        backend=backend,
        inventory_prefix=prefix,
        marker_key=marker_key,
    )


def _validate_batch_marker(backend: StorageBackend, batch: Mapping[str, object]) -> dict[str, object]:
    batch_without_marker = {key: value for key, value in batch.items() if key not in {"batch_sha256", "marker"}}
    batch_hash = require_content_hash(batch.get("batch_sha256"), "batch sha256")
    if sha256_json(batch_without_marker) != batch_hash:
        raise PreparationError("batch receipt identity changed after commit")
    reference = batch.get("marker")
    if not isinstance(reference, Mapping):
        raise PreparationError("committed batch is missing its immutable marker")
    marker_payload = _read_marker(backend, reference)
    if marker_payload.get("batch_sha256") != batch_hash:
        raise PreparationError("batch marker does not match the batch receipt")
    if marker_payload.get("state") != BatchState.COMMITTED.value:
        raise PreparationError("batch marker is not committed")
    outer_objects = batch.get("objects")
    marker_objects = marker_payload.get("objects")
    if not isinstance(outer_objects, list) or not isinstance(marker_objects, list):
        raise PreparationError("committed batch is missing its object inventory")
    if len(outer_objects) != len(marker_objects):
        raise PreparationError("batch marker object inventory differs from the batch receipt")
    outer_by_key = {_object_identity(item)["key"]: item for item in outer_objects if isinstance(item, Mapping)}
    marker_by_key = {_object_identity(item)["key"]: item for item in marker_objects if isinstance(item, Mapping)}
    if set(outer_by_key) != set(marker_by_key):
        raise PreparationError("batch marker object inventory differs from the batch receipt")
    for key in outer_by_key:
        _compare_object_identity(outer_by_key[key], marker_by_key[key], "batch marker")
    return marker_payload


def commit_release(
    batches: Sequence[Mapping[str, object]],
    *,
    required_sources: Sequence[str],
    state: RemoteReleaseState,
    backend: StorageBackend | None = None,
    inventory_prefix: str | None = None,
    required_objects: Sequence[Mapping[str, object]] | None = None,
    release_identity: Mapping[str, object] | None = None,
    allowed_inventory_keys: Sequence[str] = (),
    marker_key: str | None = None,
) -> dict[str, object]:
    """Publish a final release marker only after all committed batch identities close exactly."""

    if backend is None or inventory_prefix is None:
        raise UnresolvedInputError("a backend and scoped inventory prefix are required for a release commit")
    if state not in {
        RemoteReleaseState.DRAFT,
        RemoteReleaseState.BATCHES_ACCOUNTED,
        RemoteReleaseState.COMMITTED,
    }:
        raise ContractError("release state is unknown")
    if not batches:
        raise PreparationError("release cannot commit without committed batches")
    source_names = [str(name) for name in required_sources]
    if len(set(source_names)) != len(source_names) or any(not name for name in source_names):
        raise ContractError("required_sources must contain unique non-empty names")
    identity = dict(release_identity or {})
    if not identity:
        raise ContractError("release_identity is required for a complete release seal")
    required_batch_acceptance = identity.get("required_batches")
    if not isinstance(required_batch_acceptance, Mapping) or not required_batch_acceptance:
        raise ContractError("release_identity.required_batches must map every batch hash to its acceptance hash")
    expected_batch_acceptance = {}
    for batch_hash, acceptance_hash in required_batch_acceptance.items():
        expected_batch_acceptance[require_content_hash(batch_hash, "required batch sha256")] = require_content_hash(
            acceptance_hash, "required batch acceptance sha256"
        )
    identity["required_batches"] = dict(sorted(expected_batch_acceptance.items()))
    union: dict[str, Mapping[str, object]] = {}
    marker_refs: list[str] = []
    batch_payloads: list[Mapping[str, object]] = []
    seen_batch_hashes: set[str] = set()
    for index, batch in enumerate(batches):
        if not isinstance(batch, Mapping) or batch.get("state") != BatchState.COMMITTED.value:
            raise PreparationError("release cannot commit an uncommitted batch", {"batch": index})
        marker_payload = _validate_batch_marker(backend, batch)
        batch_hash = require_content_hash(batch.get("batch_sha256"), "batch sha256")
        if batch_hash in seen_batch_hashes:
            raise PreparationError("release contains a duplicate batch", {"batch_sha256": batch_hash})
        seen_batch_hashes.add(batch_hash)
        batch_acceptance = require_content_hash(batch.get("acceptance_sha256"), "batch acceptance sha256")
        expected_acceptance = expected_batch_acceptance.get(batch_hash)
        if expected_acceptance is None or batch_acceptance != expected_acceptance:
            raise PreparationError(
                "release batch acceptance identity is not listed in release_identity.required_batches"
            )
        marker = batch.get("marker")
        if isinstance(marker, Mapping):
            marker_refs.append(_validate_key(str(marker.get("key", "")), "batch marker key"))
        batch_payloads.append(batch)
        for item in marker_payload.get("objects") or []:
            if not isinstance(item, Mapping):
                raise PreparationError("committed batch marker contains an invalid object")
            receipt = _receipt_dict(item)
            key = str(receipt["key"])
            if key in union:
                raise PreparationError("release contains duplicate object keys", {"key": key})
            union[key] = receipt
    if set(expected_batch_acceptance) != seen_batch_hashes:
        raise PreparationError(
            "release_identity.required_batches does not match the supplied committed batches",
            {
                "missing": sorted(set(expected_batch_acceptance) - seen_batch_hashes),
                "extra": sorted(seen_batch_hashes - set(expected_batch_acceptance)),
            },
        )
    required_items = list(required_objects) if required_objects is not None else list(union.values())
    expected = [_object_identity(item, f"required_objects[{index}]") for index, item in enumerate(required_items)]
    expected_by_key = {str(item["key"]): item for item in expected}
    if len(expected_by_key) != len(expected):
        raise ContractError("required_objects contains duplicate keys")
    if set(expected_by_key) != set(union):
        raise PreparationError(
            "release object union differs from the explicit accepted inventory",
            {
                "missing": sorted(set(expected_by_key) - set(union))[:20],
                "extra": sorted(set(union) - set(expected_by_key))[:20],
            },
        )
    prefix = _validate_prefix(inventory_prefix, "inventory_prefix")
    evidence = _RemoteEvidenceSession(backend, privacy_prefix=prefix)
    for key, item in union.items():
        _compare_object_identity(expected_by_key[key], item, "release")
        proof = _readback(backend, key, expected_sha256=str(item["sha256"]), expected_size=int(item["size"]))
        if evidence.encryption_for(proof) != item.get("encryption"):
            raise PreparationError("remote encryption evidence differs from the object receipt", {"key": key})
        evidence.privacy_for(proof)
    inventory_keys = _list_prefix_bounded(backend, prefix)
    control_keys = _verified_control_keys(backend, prefix, evidence, inventory_keys)
    present_sources = {str(item.get("source")) for item in union.values()}
    missing_sources = sorted(set(source_names) - present_sources)
    extra_sources = sorted(present_sources - set(source_names))
    if missing_sources or extra_sources:
        raise UnresolvedInputError(
            "required membership is incomplete",
            {"missing": missing_sources, "extra": extra_sources},
        )
    inventory = _verify_inventory_closure(
        list(union.values()),
        prefix=prefix,
        allowed_keys=[
            *allowed_inventory_keys,
            *control_keys,
            *marker_refs,
            *_known_marker_keys(backend, prefix, kind="releases", inventory_keys=inventory_keys),
        ],
        actual_keys=inventory_keys,
    )
    marker_inventory = _stable_inventory(inventory, list(union))
    batches_payload = [dict(batch) for batch in sorted(batch_payloads, key=lambda item: str(item["batch_sha256"]))]
    base: dict[str, object] = {
        "schema": "speakrs-remote-release-v1",
        "state": RemoteReleaseState.COMMITTED.value,
        "batches": batches_payload,
        "objects": [dict(union[key]) for key in sorted(union)],
        "required_sources": sorted(source_names),
        "release_identity": identity,
        "inventory": marker_inventory,
    }
    release_sha256 = sha256_json(base)
    resolved_marker_key = marker_key or _marker_key(prefix, "releases", release_sha256)
    marker_payload = {**base, "release_sha256": release_sha256}
    marker = _publish_marker(
        backend,
        marker_payload,
        marker_key=resolved_marker_key,
        privacy_prefix=prefix,
    )
    if state is RemoteReleaseState.DRAFT:
        assert_state_transition(state, RemoteReleaseState.BATCHES_ACCOUNTED, RELEASE_TRANSITIONS, "release")
        state = RemoteReleaseState.BATCHES_ACCOUNTED
    if state is RemoteReleaseState.BATCHES_ACCOUNTED:
        assert_state_transition(state, RemoteReleaseState.COMMITTED, RELEASE_TRANSITIONS, "release")
    elif state is not RemoteReleaseState.COMMITTED:
        raise ContractError("release state is unknown")
    result = dict(marker_payload)
    result["marker"] = marker
    return result


@dataclass(frozen=True)
class RemoteRestoreProof:
    """Typed proof that one local copy can be restored from a committed remote marker."""

    object_key: str
    object_sha256: str
    object_size: int
    acceptance_sha256: str
    marker_key: str
    marker_sha256: str
    restore_receipt_sha256: str
    restored_sha256: str
    restored_size: int
    restore_receipt_path: Path

    def __post_init__(self) -> None:
        _validate_key(self.object_key, "proof.object_key")
        _validate_key(self.marker_key, "proof.marker_key")
        object_sha256 = require_content_hash(self.object_sha256, "proof.object_sha256")
        acceptance_sha256 = require_content_hash(self.acceptance_sha256, "proof.acceptance_sha256")
        marker_sha256 = require_content_hash(self.marker_sha256, "proof.marker_sha256")
        restore_receipt_sha256 = require_content_hash(self.restore_receipt_sha256, "proof.restore_receipt_sha256")
        restored_sha256 = require_content_hash(self.restored_sha256, "proof.restored_sha256")
        object.__setattr__(self, "object_sha256", object_sha256)
        object.__setattr__(self, "acceptance_sha256", acceptance_sha256)
        object.__setattr__(self, "marker_sha256", marker_sha256)
        object.__setattr__(self, "restore_receipt_sha256", restore_receipt_sha256)
        object.__setattr__(self, "restored_sha256", restored_sha256)
        _positive_int(self.object_size, "proof.object_size")
        _positive_int(self.restored_size, "proof.restored_size")
        if not isinstance(self.restore_receipt_path, Path):
            raise ContractError("proof.restore_receipt_path must name a restore receipt file")
        if self.restored_sha256.lower() != self.object_sha256.lower() or self.restored_size != self.object_size:
            raise ContractError("restored identity does not match the committed object")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> RemoteRestoreProof:
        """Parse a restore proof without accepting boolean completion flags."""

        if not isinstance(payload, Mapping):
            raise ContractError("remote restore proof must be an object")
        allowed = {
            "object_key",
            "object_sha256",
            "object_size",
            "acceptance_sha256",
            "marker_key",
            "marker_sha256",
            "restore_receipt_sha256",
            "restored_sha256",
            "restored_size",
            "restore_receipt_path",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ContractError("remote restore proof has unknown keys", {"unknown": unknown})
        for name in ("restored", "remote_verified", "committed", "ok"):
            if name in payload:
                raise ContractError("remote restore proof cannot use boolean completion flags", {"field": name})
        path = payload.get("restore_receipt_path")
        if not isinstance(path, str) or not path:
            raise ContractError("remote restore proof requires restore_receipt_path")
        required_strings = (
            "object_key",
            "object_sha256",
            "acceptance_sha256",
            "marker_key",
            "marker_sha256",
            "restore_receipt_sha256",
            "restored_sha256",
        )
        for name in required_strings:
            if not isinstance(payload.get(name), str):
                raise ContractError(f"remote restore proof {name} must be a string")
        return cls(
            object_key=payload["object_key"],
            object_sha256=payload["object_sha256"],
            object_size=payload.get("object_size"),
            acceptance_sha256=payload["acceptance_sha256"],
            marker_key=payload["marker_key"],
            marker_sha256=payload["marker_sha256"],
            restore_receipt_sha256=payload["restore_receipt_sha256"],
            restored_sha256=payload["restored_sha256"],
            restored_size=payload.get("restored_size"),
            restore_receipt_path=Path(path),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe proof."""

        result = {
            "object_key": self.object_key,
            "object_sha256": self.object_sha256,
            "object_size": self.object_size,
            "acceptance_sha256": self.acceptance_sha256,
            "marker_key": self.marker_key,
            "marker_sha256": self.marker_sha256,
            "restore_receipt_sha256": self.restore_receipt_sha256,
            "restored_sha256": self.restored_sha256,
            "restored_size": self.restored_size,
        }
        result["restore_receipt_path"] = self.restore_receipt_path.as_posix()
        return result


def verify_remote_restore_proof(
    proof: RemoteRestoreProof | Mapping[str, object],
    *,
    backend: StorageBackend,
) -> RemoteRestoreProof:
    """Recheck marker, object, and optional local restore-receipt identities before deletion."""

    typed = proof if isinstance(proof, RemoteRestoreProof) else RemoteRestoreProof.from_mapping(proof)
    path = typed.restore_receipt_path
    if not path.is_file() or sha256_file(path) != typed.restore_receipt_sha256:
        raise PreparationError("restore receipt is missing or changed", {"path": str(path)})
    try:
        restore_receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError("restore receipt is not valid JSON", {"path": str(path)}) from error
    if not isinstance(restore_receipt, Mapping):
        raise PreparationError("restore receipt is not an object", {"path": str(path)})
    if (
        restore_receipt.get("schema") != "speakrs-cold-restore-v1"
        or restore_receipt.get("command") != "restore-check"
        or restore_receipt.get("ok") is not True
    ):
        raise PreparationError("restore receipt is not a successful cold-restore receipt", {"path": str(path)})
    if restore_receipt.get("acceptance_sha256") != typed.acceptance_sha256:
        raise PreparationError("restore receipt acceptance identity differs from the proof", {"path": str(path)})
    receipt_marker = restore_receipt.get("marker")
    if not isinstance(receipt_marker, Mapping):
        raise PreparationError("restore receipt is missing its committed marker", {"path": str(path)})
    if receipt_marker.get("key") != typed.marker_key or receipt_marker.get("sha256") != typed.marker_sha256:
        raise PreparationError("restore receipt marker differs from the proof", {"path": str(path)})
    receipt_objects = restore_receipt.get("objects")
    if not isinstance(receipt_objects, list):
        raise PreparationError("restore receipt is missing restored object coverage", {"path": str(path)})
    covered = False
    for item in receipt_objects:
        if not isinstance(item, Mapping):
            raise PreparationError("restore receipt contains an invalid object", {"path": str(path)})
        try:
            identity = _object_identity(item, "restore receipt object")
        except ContractError as error:
            raise PreparationError("restore receipt contains an invalid object", {"path": str(path)}) from error
        if (
            identity["key"] == typed.object_key
            and identity["sha256"] == typed.object_sha256.lower()
            and identity["size"] == typed.object_size
        ):
            covered = True
    if not covered:
        raise PreparationError("restore receipt does not cover the evicted object", {"key": typed.object_key})
    marker_ref = {
        "key": typed.marker_key,
        "sha256": typed.marker_sha256,
        "size": 1,
    }
    marker_size = backend.head(typed.marker_key).get("size")
    if not marker_size:
        raise PreparationError("committed marker size is unavailable", {"key": typed.marker_key})
    try:
        marker_ref["size"] = int(marker_size)
    except (TypeError, ValueError) as error:
        raise PreparationError("committed marker size is invalid", {"key": typed.marker_key}) from error
    marker = _read_marker(backend, marker_ref)
    if typed.acceptance_sha256.lower() not in _marker_acceptance_hashes(marker):
        raise PreparationError("restore proof acceptance identity is not bound by the committed marker")
    records = _marker_object_records(marker)
    if not any(
        _object_identity(item)["key"] == typed.object_key
        and _object_identity(item)["sha256"] == typed.object_sha256.lower()
        and _object_identity(item)["size"] == typed.object_size
        for item in records
    ):
        raise PreparationError("restore proof object is absent from the committed marker", {"key": typed.object_key})
    _readback(
        backend,
        typed.object_key,
        expected_sha256=typed.object_sha256,
        expected_size=typed.object_size,
    )
    return typed


@dataclass
class LocalCopy:
    """One task-owned local file."""

    path: Path
    state: LocalCopyState
    source: str
    sha256: str
    live_readers: int = 0
    task_owned: bool = True
    labels_accepted: bool = False
    remote_verified: bool = False
    remote_proof: RemoteRestoreProof | Mapping[str, object] | None = None
    deletion_receipt: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ConsumedSource:
    """One exact raw source file that may be discarded after canonicalization."""

    path: Path
    parent_id: str
    source_sha256: str
    task_root: Path
    live_readers: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.task_root, Path):
            object.__setattr__(self, "task_root", Path(self.task_root))
        if not isinstance(self.parent_id, str) or not self.parent_id:
            raise ContractError("consumed source parent_id must be a non-empty string")
        object.__setattr__(self, "source_sha256", require_content_hash(self.source_sha256, "source sha256"))
        _non_negative_int(self.live_readers, "source live_readers")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ConsumedSource:
        """Parse an exact source cleanup request without accepting completion flags."""

        if not isinstance(payload, Mapping):
            raise ContractError("consumed source request must be an object")
        allowed = {"path", "source_path", "parent_id", "source_sha256", "sha256", "task_root", "live_readers"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ContractError("consumed source request has unknown keys", {"unknown": unknown})
        path = payload.get("path", payload.get("source_path"))
        if not isinstance(path, str) or not path:
            raise ContractError("consumed source request requires path")
        parent_id = payload.get("parent_id")
        source_sha256 = payload.get("source_sha256", payload.get("sha256"))
        if "source_sha256" in payload and "sha256" in payload and payload["source_sha256"] != payload["sha256"]:
            raise ContractError("consumed source request has conflicting source hashes")
        task_root = payload.get("task_root")
        if not isinstance(parent_id, str) or not parent_id:
            raise ContractError("consumed source request requires parent_id")
        if not isinstance(source_sha256, str):
            raise ContractError("consumed source request requires source_sha256")
        if not isinstance(task_root, str) or not task_root:
            raise ContractError("consumed source request requires task_root")
        return cls(
            path=Path(path),
            parent_id=parent_id,
            source_sha256=source_sha256,
            task_root=Path(task_root),
            live_readers=payload.get("live_readers", 0),
        )


def enforce_cap(used_bytes: int, cap_bytes: int, label: str, *, additional_bytes: int = 0) -> None:
    """Stop before crossing a measured cap. Do not delete non-task files."""

    used = _non_negative_int(used_bytes, "used_bytes")
    cap = _positive_int(cap_bytes, "cap_bytes")
    additional = _non_negative_int(additional_bytes, "additional_bytes")
    if used + additional > cap:
        raise PreparationError(
            f"{label} cap exhausted",
            {"used_bytes": used, "additional_bytes": additional, "cap_bytes": cap},
        )


def enforce_free_space_reserve(root: Path, reserve_bytes: int) -> dict[str, int]:
    """Check free space without silently expanding the configured reserve."""

    reserve = _non_negative_int(reserve_bytes, "reserve_bytes")
    usage = os.statvfs(root)
    free_bytes = usage.f_frsize * usage.f_bavail
    if free_bytes < reserve:
        raise PreparationError("free-space reserve is exhausted", {"free_bytes": free_bytes, "reserve_bytes": reserve})
    return {"free_bytes": free_bytes, "reserve_bytes": reserve}


def directory_bytes(root: Path) -> int:
    """Return the size of files under a directory."""

    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _restore_receipt_anchor(value: Path | str, label: str = "restore receipt") -> Path:
    if isinstance(value, str):
        value = Path(value)
    if not isinstance(value, Path) or not value.name:
        raise ContractError(f"{label} must name a durable receipt file")
    return value.expanduser().resolve(strict=False)


def _deletion_intent_core(intent: Mapping[str, object]) -> dict[str, object]:
    """Return the immutable identity that selects one deletion journal."""

    if not isinstance(intent, Mapping):
        raise ContractError("deletion intent must be an object")
    operation = intent.get("operation")
    if operation not in {"eviction", "consumed-source"}:
        raise ContractError("deletion intent operation is unknown")
    raw_path = intent.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractError("deletion intent path is required")
    path = Path(raw_path).expanduser().resolve(strict=False).as_posix()
    raw_receipt_path = intent.get("restore_receipt_path")
    receipt_path = _restore_receipt_anchor(raw_receipt_path, "deletion intent restore receipt")
    receipt_sha256 = require_content_hash(
        intent.get("restore_receipt_sha256"), "deletion intent restore receipt sha256"
    )
    core: dict[str, object] = {
        "operation": operation,
        "path": path,
        "restore_receipt_path": receipt_path.as_posix(),
        "restore_receipt_sha256": receipt_sha256,
    }
    if operation == "eviction":
        object_key = intent.get("object_key")
        if not isinstance(object_key, str):
            raise ContractError("eviction deletion intent object_key is required")
        core.update(
            {
                "copy_sha256": require_content_hash(intent.get("copy_sha256"), "eviction copy sha256"),
                "object_key": _validate_key(object_key, "eviction object key"),
                "object_sha256": require_content_hash(intent.get("object_sha256"), "eviction object sha256"),
                "object_size": _positive_int(intent.get("object_size"), "eviction object size"),
                "marker_key": _validate_key(str(intent.get("marker_key", "")), "eviction marker key"),
                "marker_sha256": require_content_hash(intent.get("marker_sha256"), "eviction marker sha256"),
                "acceptance_sha256": require_content_hash(
                    intent.get("acceptance_sha256"), "eviction acceptance sha256"
                ),
            }
        )
    else:
        parent_id = intent.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise ContractError("consumed-source deletion intent parent_id is required")
        restore_marker = intent.get("restore_marker")
        if not isinstance(restore_marker, Mapping):
            raise ContractError("consumed-source deletion intent restore marker is required")
        marker_key = restore_marker.get("key")
        if not isinstance(marker_key, str):
            raise ContractError("consumed-source deletion intent marker key is required")
        core.update(
            {
                "source_sha256": require_content_hash(intent.get("source_sha256"), "source deletion sha256"),
                "parent_id": parent_id,
                "restore_marker": {
                    "key": _validate_key(marker_key, "source deletion marker key"),
                    "sha256": require_content_hash(restore_marker.get("sha256"), "source deletion marker sha256"),
                },
                "acceptance_sha256": require_content_hash(
                    intent.get("acceptance_sha256"), "source deletion acceptance sha256"
                ),
            }
        )
    return core


def _deletion_journal_path(intent: Mapping[str, object]) -> Path:
    core = _deletion_intent_core(intent)
    anchor = Path(str(core["restore_receipt_path"]))
    digest = sha256_json(core)
    return anchor.parent / DELETION_JOURNAL_DIRECTORY / f"{core['operation']}-{digest}.json"


def _deletion_journal_envelope(
    intent: Mapping[str, object],
    state: str,
    completion: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if state not in {DELETION_INTENT_STATE, DELETION_COMPLETED_STATE}:
        raise ContractError("deletion journal state is unknown")
    normalized_intent = dict(intent)
    _deletion_intent_core(normalized_intent)
    if state == DELETION_INTENT_STATE and completion is not None:
        raise ContractError("deletion intent journal cannot contain a completion")
    if state == DELETION_COMPLETED_STATE and completion is None:
        raise ContractError("completed deletion journal requires a completion")
    base: dict[str, object] = {
        "schema": DELETION_JOURNAL_SCHEMA,
        "state": state,
        "intent": normalized_intent,
        "intent_sha256": sha256_json(normalized_intent),
        "completion": dict(completion) if completion is not None else None,
    }
    base["journal_sha256"] = sha256_json(base)
    return base


def _load_deletion_journal(path: Path) -> dict[str, object]:
    try:
        payload = read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError("deletion journal is missing or invalid", {"path": str(path)}) from error
    if not isinstance(payload, Mapping):
        raise PreparationError("deletion journal is not an object", {"path": str(path)})
    allowed = {"schema", "state", "intent", "intent_sha256", "completion", "journal_sha256"}
    if set(payload) != allowed:
        raise PreparationError("deletion journal schema has unexpected fields", {"path": str(path)})
    if payload.get("schema") != DELETION_JOURNAL_SCHEMA:
        raise PreparationError("deletion journal schema is unknown", {"path": str(path)})
    state = payload.get("state")
    if state not in {DELETION_INTENT_STATE, DELETION_COMPLETED_STATE}:
        raise PreparationError("deletion journal state is unknown", {"path": str(path)})
    intent = payload.get("intent")
    if not isinstance(intent, Mapping):
        raise PreparationError("deletion journal has no intent", {"path": str(path)})
    try:
        intent_sha256 = require_content_hash(payload.get("intent_sha256"), "deletion intent sha256")
        journal_sha256 = require_content_hash(payload.get("journal_sha256"), "deletion journal sha256")
        _deletion_intent_core(intent)
    except ContractError as error:
        raise PreparationError("deletion journal intent is invalid", {"path": str(path)}) from error
    if intent_sha256 != sha256_json(intent):
        raise PreparationError("deletion journal intent changed", {"path": str(path)})
    base = {key: payload[key] for key in ("schema", "state", "intent", "intent_sha256", "completion")}
    if journal_sha256 != sha256_json(base):
        raise PreparationError("deletion journal identity changed", {"path": str(path)})
    completion = payload.get("completion")
    if state == DELETION_INTENT_STATE:
        if completion is not None:
            raise PreparationError("deletion intent journal has an unexpected completion", {"path": str(path)})
    else:
        if not isinstance(completion, Mapping):
            raise PreparationError("completed deletion journal has no receipt", {"path": str(path)})
        receipt_sha256 = completion.get("receipt_sha256")
        if not isinstance(receipt_sha256, str):
            raise PreparationError("completed deletion journal has no receipt identity", {"path": str(path)})
        if receipt_sha256 != sha256_json({key: value for key, value in completion.items() if key != "receipt_sha256"}):
            raise PreparationError("completed deletion receipt changed", {"path": str(path)})
        try:
            expected_completion = _deletion_completion_for_intent(intent)
        except (ContractError, KeyError, TypeError) as error:
            raise PreparationError("completed deletion journal completion is invalid", {"path": str(path)}) from error
        if dict(completion) != expected_completion:
            raise PreparationError("completed deletion receipt does not match its intent", {"path": str(path)})
    return dict(payload)


def _deletion_intent_matches(intent: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    try:
        normalized = _deletion_intent_core(intent)
    except ContractError:
        return False
    for key, value in expected.items():
        if key in {"path", "restore_receipt_path"}:
            if key == "path":
                expected_value = Path(str(value)).expanduser().resolve(strict=False).as_posix()
            else:
                expected_value = _restore_receipt_anchor(value, "expected restore receipt").as_posix()
        elif key.endswith("_sha256"):
            try:
                expected_value = require_content_hash(value, key)
            except ContractError:
                return False
        else:
            expected_value = value
        if normalized.get(key, intent.get(key)) != expected_value:
            return False
    return True


def recover_deletion_journal(
    restore_receipt_path: Path | str,
    *,
    operation: str,
    path: Path | str,
    expected_identity: Mapping[str, object],
) -> dict[str, object] | None:
    """Recover a validated deletion intent or completion for one missing copy.

    The helper only returns a journal whose operation, path, restore receipt, and
    supplied immutable identity fields match exactly. A caller must invoke it
    only after observing that the local copy is missing.
    """

    if operation not in {"eviction", "consumed-source"}:
        raise ContractError("deletion journal operation is unknown")
    anchor = _restore_receipt_anchor(restore_receipt_path)
    if not isinstance(expected_identity, Mapping):
        raise ContractError("deletion journal expected identity must be an object")
    expected = dict(expected_identity)
    expected.update(
        {
            "operation": operation,
            "path": Path(path).expanduser().resolve(strict=False).as_posix(),
            "restore_receipt_path": anchor.as_posix(),
        }
    )
    journal_root = anchor.parent / DELETION_JOURNAL_DIRECTORY
    if not journal_root.is_dir():
        return None
    candidates = sorted(journal_root.glob(f"{operation}-*.json"))
    matches: list[dict[str, object]] = []
    for candidate in candidates:
        journal = _load_deletion_journal(candidate)
        intent = journal["intent"]
        if not isinstance(intent, Mapping):
            raise PreparationError("deletion journal has no intent", {"path": str(candidate)})
        expected_path = _deletion_journal_path(intent)
        if expected_path != candidate:
            raise PreparationError("deletion journal filename does not match its identity", {"path": str(candidate)})
        if _deletion_intent_matches(intent, expected):
            matches.append(journal)
    if len(matches) > 1:
        raise PreparationError("multiple deletion journals match one copy", {"path": str(path)})
    return matches[0] if matches else None


def _persist_deletion_intent(intent: Mapping[str, object]) -> dict[str, object]:
    path = _deletion_journal_path(intent)
    if path.is_file():
        existing = _load_deletion_journal(path)
        if existing.get("intent") != dict(intent):
            raise PreparationError("deletion journal intent differs from the current proof", {"path": str(path)})
        return existing
    journal = _deletion_journal_envelope(intent, DELETION_INTENT_STATE)
    write_json(path, journal)
    return journal


def _persist_deletion_completion(intent: Mapping[str, object], completion: Mapping[str, object]) -> dict[str, object]:
    expected_completion = _deletion_completion_for_intent(intent)
    if dict(completion) != expected_completion:
        raise PreparationError("completed deletion receipt does not match its intent")
    path = _deletion_journal_path(intent)
    if not path.is_file():
        raise PreparationError("deletion intent was not durably recorded", {"path": str(path)})
    existing = _load_deletion_journal(path)
    if existing.get("intent") != dict(intent):
        raise PreparationError("deletion journal intent differs from the current proof", {"path": str(path)})
    if existing.get("state") == DELETION_COMPLETED_STATE:
        if existing.get("completion") != dict(completion):
            raise PreparationError("completed deletion receipt differs from the current proof", {"path": str(path)})
        return existing
    journal = _deletion_journal_envelope(intent, DELETION_COMPLETED_STATE, completion)
    write_json(path, journal)
    return journal


def _receipt_with_hash(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["receipt_sha256"] = sha256_json(result)
    return result


def _copy_proof(copy: LocalCopy, proof: RemoteRestoreProof | Mapping[str, object] | None) -> RemoteRestoreProof:
    candidate = proof if proof is not None else copy.remote_proof
    if candidate is None:
        raise PreparationError("eviction requires a typed remote restore proof", {"path": str(copy.path)})
    return candidate if isinstance(candidate, RemoteRestoreProof) else RemoteRestoreProof.from_mapping(candidate)


def _validate_copy_metadata(copy: LocalCopy) -> None:
    if copy.task_owned is not True:
        raise PreparationError("eviction cannot delete non-task files", {"path": str(copy.path)})
    if copy.labels_accepted is not True:
        raise PreparationError("eviction requires accepted labels", {"path": str(copy.path)})
    if isinstance(copy.live_readers, bool) or copy.live_readers < 0:
        raise PreparationError("local reader count is invalid", {"path": str(copy.path)})
    if copy.live_readers:
        raise PreparationError("live readers block unsafe deletion", {"path": str(copy.path)})
    if copy.path.is_symlink():
        raise PreparationError("eviction requires one existing task-owned regular file", {"path": str(copy.path)})


def _verify_copy_before_eviction(
    copy: LocalCopy,
    *,
    proof: RemoteRestoreProof | Mapping[str, object] | None,
    backend: StorageBackend | None,
) -> RemoteRestoreProof:
    _validate_copy_metadata(copy)
    if not copy.path.is_file():
        raise PreparationError("eviction requires one existing task-owned regular file", {"path": str(copy.path)})
    expected_hash = require_content_hash(copy.sha256, "local copy sha256")
    actual_hash = sha256_file(copy.path)
    if actual_hash != expected_hash:
        raise PreparationError("local copy changed after remote restore proof", {"path": str(copy.path)})
    typed = _copy_proof(copy, proof)
    if typed.object_sha256 != expected_hash:
        raise PreparationError("remote restore proof does not match the local copy", {"path": str(copy.path)})
    if copy.path.stat().st_size != typed.object_size:
        raise PreparationError("local copy size differs from the remote proof", {"path": str(copy.path)})
    if backend is None:
        raise UnresolvedInputError("eviction requires a backend to recheck the remote restore proof")
    return verify_remote_restore_proof(typed, backend=backend)


def _group_copy_proof(copy: LocalCopy, session: _CleanupVerificationSession) -> RemoteRestoreProof:
    """Bind one caller copy to the proof owned by a fresh cleanup session."""

    candidate = _copy_proof(copy, None)
    proof = session.proof_for(candidate.object_key)
    if any(
        (
            getattr(candidate, field) != getattr(proof, field)
            if field != "restore_receipt_path"
            else candidate.restore_receipt_path.expanduser().resolve(strict=False)
            != proof.restore_receipt_path.expanduser().resolve(strict=False)
        )
        for field in (
            "object_key",
            "object_sha256",
            "object_size",
            "acceptance_sha256",
            "marker_key",
            "marker_sha256",
            "restore_receipt_sha256",
            "restored_sha256",
            "restored_size",
            "restore_receipt_path",
        )
    ):
        raise PreparationError(
            "caller remote restore proof differs from the durable cleanup receipt", {"path": str(copy.path)}
        )
    return proof


def _prepare_group_eviction_copy(
    copy: LocalCopy,
    *,
    proof: RemoteRestoreProof,
) -> LocalCopy:
    """Run per-copy local checks before a grouped deletion core is entered."""

    _validate_copy_metadata(copy)
    if copy.state not in {LocalCopyState.RETAINED, LocalCopyState.EVICTION_ELIGIBLE}:
        raise PreparationError("local copy is not retained", {"path": str(copy.path)})
    expected_hash = require_content_hash(copy.sha256, "local copy sha256")
    if proof.object_sha256 != expected_hash:
        raise PreparationError("remote restore proof does not match the local copy", {"path": str(copy.path)})
    if copy.path.is_file():
        if sha256_file(copy.path) != expected_hash:
            raise PreparationError("local copy changed after remote restore proof", {"path": str(copy.path)})
        if copy.path.stat().st_size != proof.object_size:
            raise PreparationError("local copy size differs from the remote proof", {"path": str(copy.path)})
    elif copy.state is LocalCopyState.RETAINED:
        raise PreparationError("eviction requires one existing task-owned regular file", {"path": str(copy.path)})
    if copy.state is LocalCopyState.EVICTION_ELIGIBLE:
        return LocalCopy(
            path=copy.path,
            state=copy.state,
            source=copy.source,
            sha256=copy.sha256,
            live_readers=copy.live_readers,
            task_owned=copy.task_owned,
            labels_accepted=copy.labels_accepted,
            remote_verified=True,
            remote_proof=proof,
            deletion_receipt=copy.deletion_receipt,
        )
    assert_state_transition(copy.state, LocalCopyState.EVICTION_ELIGIBLE, LOCAL_TRANSITIONS, "local")
    return LocalCopy(
        path=copy.path,
        state=LocalCopyState.EVICTION_ELIGIBLE,
        source=copy.source,
        sha256=copy.sha256,
        live_readers=copy.live_readers,
        task_owned=copy.task_owned,
        labels_accepted=copy.labels_accepted,
        remote_verified=True,
        remote_proof=proof,
        deletion_receipt=copy.deletion_receipt,
    )


def _recheck_local_identity(path: Path, *, expected_sha256: str, expected_size: int, label: str) -> None:
    """Recheck local bytes and size after remote proof work and before deletion."""

    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} is no longer one existing regular file", {"path": str(path)})
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PreparationError(f"{label} changed after remote proof", {"path": str(path)})
    if path.stat().st_size != expected_size:
        raise PreparationError(f"{label} size changed after remote proof", {"path": str(path)})


def _eviction_intent(
    copy: LocalCopy,
    proof: RemoteRestoreProof,
    *,
    open_file_check: Mapping[str, object],
) -> dict[str, object]:
    return {
        "operation": "eviction",
        "path": copy.path.expanduser().resolve(strict=False).as_posix(),
        "copy_sha256": require_content_hash(copy.sha256, "local copy sha256"),
        "copy_size": proof.object_size,
        "source": copy.source,
        "object_key": proof.object_key,
        "object_sha256": proof.object_sha256,
        "object_size": proof.object_size,
        "acceptance_sha256": proof.acceptance_sha256,
        "marker_key": proof.marker_key,
        "marker_sha256": proof.marker_sha256,
        "restore_receipt_path": proof.restore_receipt_path.expanduser().resolve(strict=False).as_posix(),
        "restore_receipt_sha256": proof.restore_receipt_sha256,
        "open_file_check": dict(open_file_check),
    }


def _eviction_completion(intent: Mapping[str, object]) -> dict[str, object]:
    return _receipt_with_hash(
        {
            "schema": "speakrs-eviction-v1",
            "state": LocalCopyState.EVICTED.value,
            "path": intent["path"],
            "sha256": intent["copy_sha256"],
            "source": intent.get("source"),
            "object_key": intent["object_key"],
            "object_sha256": intent["object_sha256"],
            "object_size": intent["object_size"],
            "marker_key": intent["marker_key"],
            "marker_sha256": intent["marker_sha256"],
            "restore_receipt_path": intent["restore_receipt_path"],
            "restore_receipt_sha256": intent["restore_receipt_sha256"],
            "open_file_check": intent["open_file_check"],
            "deletion_journal_path": _deletion_journal_path(intent).as_posix(),
            "deleted": True,
        }
    )


def _evicted_copy(
    copy: LocalCopy,
    proof: RemoteRestoreProof,
    deletion: Mapping[str, object],
) -> LocalCopy:
    assert_state_transition(copy.state, LocalCopyState.EVICTED, LOCAL_TRANSITIONS, "local")
    return LocalCopy(
        path=copy.path,
        state=LocalCopyState.EVICTED,
        source=copy.source,
        sha256=copy.sha256,
        live_readers=0,
        task_owned=copy.task_owned,
        labels_accepted=copy.labels_accepted,
        remote_verified=True,
        remote_proof=proof,
        deletion_receipt=deletion,
    )


def mark_eviction_eligible(
    copy: LocalCopy,
    *,
    proof: RemoteRestoreProof | Mapping[str, object] | None = None,
    backend: StorageBackend | None = None,
) -> LocalCopy:
    """Mark a copy eligible only after labels, local identity, marker, and restore proof checks."""

    if copy.state is LocalCopyState.EVICTION_ELIGIBLE:
        _verify_copy_before_eviction(copy, proof=proof, backend=backend)
        return copy
    if copy.state is not LocalCopyState.RETAINED:
        raise PreparationError("local copy is not retained", {"path": str(copy.path)})
    typed = _verify_copy_before_eviction(copy, proof=proof, backend=backend)
    assert_state_transition(copy.state, LocalCopyState.EVICTION_ELIGIBLE, LOCAL_TRANSITIONS, "local")
    return LocalCopy(
        path=copy.path,
        state=LocalCopyState.EVICTION_ELIGIBLE,
        source=copy.source,
        sha256=copy.sha256,
        live_readers=copy.live_readers,
        task_owned=copy.task_owned,
        labels_accepted=copy.labels_accepted,
        remote_verified=copy.remote_verified,
        remote_proof=typed,
        deletion_receipt=copy.deletion_receipt,
    )


def _evict_copy_after_proof(
    copy: LocalCopy,
    *,
    proof: RemoteRestoreProof,
    preserve_on_interrupt: bool = False,
) -> LocalCopy:
    """Delete one eligible task-owned file after its remote proof is bound."""

    if copy.state is not LocalCopyState.EVICTION_ELIGIBLE:
        raise PreparationError("local copy is not eviction-eligible", {"path": str(copy.path)})
    if preserve_on_interrupt:
        raise PreparationError("interrupted eviction", {"path": str(copy.path)})
    _validate_copy_metadata(copy)
    typed = _copy_proof(copy, proof)
    expected_identity = {
        "copy_sha256": require_content_hash(copy.sha256, "local copy sha256"),
        "object_key": typed.object_key,
        "object_sha256": typed.object_sha256,
        "object_size": typed.object_size,
        "acceptance_sha256": typed.acceptance_sha256,
        "marker_key": typed.marker_key,
        "marker_sha256": typed.marker_sha256,
        "restore_receipt_sha256": typed.restore_receipt_sha256,
    }
    if not copy.path.exists():
        recovered = recover_deletion_journal(
            typed.restore_receipt_path,
            operation="eviction",
            path=copy.path,
            expected_identity=expected_identity,
        )
        if recovered is None:
            raise PreparationError("eviction requires one existing task-owned regular file", {"path": str(copy.path)})
        if recovered.get("state") == DELETION_INTENT_STATE:
            intent = recovered.get("intent")
            if not isinstance(intent, Mapping):
                raise PreparationError("deletion journal has no intent", {"path": str(copy.path)})
            completion = _eviction_completion(intent)
            _persist_deletion_completion(intent, completion)
        else:
            completion = recovered.get("completion")
            if not isinstance(completion, Mapping):
                raise PreparationError("completed deletion journal has no receipt", {"path": str(copy.path)})
        return _evicted_copy(copy, typed, dict(completion))
    _recheck_local_identity(
        copy.path,
        expected_sha256=typed.object_sha256,
        expected_size=typed.object_size,
        label="local copy",
    )
    open_file_check = assert_no_open_file_handles(copy.path)
    intent = _eviction_intent(copy, typed, open_file_check=open_file_check)
    _persist_deletion_intent(intent)
    _recheck_local_identity(
        copy.path,
        expected_sha256=typed.object_sha256,
        expected_size=typed.object_size,
        label="local copy",
    )
    try:
        copy.path.unlink()
    except OSError as error:
        raise PreparationError(
            "task-owned file could not be deleted", {"path": str(copy.path), "error": str(error)}
        ) from error
    deletion = _eviction_completion(intent)
    _persist_deletion_completion(intent, deletion)
    return _evicted_copy(copy, typed, deletion)


def evict_copy(
    copy: LocalCopy,
    *,
    proof: RemoteRestoreProof | Mapping[str, object] | None = None,
    backend: StorageBackend | None = None,
    preserve_on_interrupt: bool = False,
) -> LocalCopy:
    """Delete one eligible task-owned file after rechecking its remote replacement."""

    if copy.state is not LocalCopyState.EVICTION_ELIGIBLE:
        raise PreparationError("local copy is not eviction-eligible", {"path": str(copy.path)})
    if preserve_on_interrupt:
        raise PreparationError("interrupted eviction", {"path": str(copy.path)})
    if not copy.path.exists():
        _validate_copy_metadata(copy)
        typed = _copy_proof(copy, proof)
        if backend is None:
            raise UnresolvedInputError("eviction requires a backend to recheck the remote restore proof")
        verify_remote_restore_proof(typed, backend=backend)
        return _evict_copy_after_proof(copy, proof=typed, preserve_on_interrupt=preserve_on_interrupt)
    typed = _verify_copy_before_eviction(copy, proof=proof, backend=backend)
    return _evict_copy_after_proof(copy, proof=typed, preserve_on_interrupt=preserve_on_interrupt)


def _load_json_mapping(
    value: Mapping[str, object] | Path | str,
    label: str,
) -> tuple[dict[str, object], Path | None, str | None]:
    if isinstance(value, Mapping):
        return dict(value), None, None
    if not isinstance(value, (Path, str)):
        raise ContractError(f"{label} must be an object or JSON path")
    path = Path(value).expanduser()
    if not path.is_file():
        raise PreparationError(f"{label} is missing", {"path": str(path)})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"{label} is not valid JSON", {"path": str(path)}) from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return payload, path, sha256_file(path)


_CLEANUP_SESSION_TOKEN = object()


def _bare_object_identity(item: Mapping[str, object], label: str) -> dict[str, object]:
    identity = _object_identity(item, label)
    return {key: identity[key] for key in ("key", "sha256", "size")}


class _CleanupVerificationSession:
    """Private remote proof shared by one grouped cleanup operation."""

    __slots__ = (
        "_acceptance_sha256",
        "_marker",
        "_marker_records",
        "_manifest_bytes",
        "_manifest_payload",
        "_proofs",
        "_restore_path",
        "_restore_sha256",
        "_restored_records",
        "_token",
    )

    def __init__(
        self,
        *,
        acceptance_sha256: str,
        marker: Mapping[str, object],
        marker_records: Mapping[str, Mapping[str, object]],
        manifest_bytes: bytes,
        manifest_payload: Mapping[str, object],
        proofs: Mapping[str, RemoteRestoreProof],
        restore_path: Path,
        restore_sha256: str,
        restored_records: Mapping[str, Mapping[str, object]],
        token: object,
    ) -> None:
        if token is not _CLEANUP_SESSION_TOKEN:
            raise TypeError("cleanup verification sessions are created by grouped cleanup owners")
        self._acceptance_sha256 = acceptance_sha256
        self._marker = dict(marker)
        self._marker_records = dict(marker_records)
        self._manifest_bytes = manifest_bytes
        self._manifest_payload = dict(manifest_payload)
        self._proofs = dict(proofs)
        self._restore_path = restore_path
        self._restore_sha256 = restore_sha256
        self._restored_records = dict(restored_records)
        self._token = token

    @property
    def acceptance_sha256(self) -> str:
        return self._acceptance_sha256

    @property
    def marker(self) -> dict[str, object]:
        return dict(self._marker)

    @property
    def marker_records(self) -> dict[str, Mapping[str, object]]:
        return dict(self._marker_records)

    @property
    def manifest_bytes(self) -> bytes:
        return self._manifest_bytes

    @property
    def manifest_payload(self) -> dict[str, object]:
        return dict(self._manifest_payload)

    @property
    def restore_path(self) -> Path:
        return self._restore_path

    @property
    def restore_sha256(self) -> str:
        return self._restore_sha256

    @property
    def restored_records(self) -> dict[str, Mapping[str, object]]:
        return dict(self._restored_records)

    def proof_for(self, object_key: str) -> RemoteRestoreProof:
        """Return the proof derived from this session's durable receipt."""

        if self._token is not _CLEANUP_SESSION_TOKEN:
            raise PreparationError("cleanup verification session is invalid")
        try:
            return self._proofs[_validate_key(object_key, "cleanup object key")]
        except KeyError as error:
            raise PreparationError(
                "cleanup object is absent from the committed inventory", {"key": object_key}
            ) from error


def _build_cleanup_verification_session(
    restore_receipt: Path | str,
    *,
    backend: StorageBackend,
) -> _CleanupVerificationSession:
    """Validate one durable restore receipt and read each committed object once."""

    if not isinstance(restore_receipt, (Path, str)):
        raise ContractError("grouped cleanup requires a durable cold restore receipt path")
    restore, restore_path, restore_sha256 = _load_json_mapping(restore_receipt, "cold restore receipt")
    if restore_path is None or restore_sha256 is None:
        raise ContractError("grouped cleanup requires a durable cold restore receipt path")
    if (
        restore.get("schema") != "speakrs-cold-restore-v1"
        or restore.get("command") != "restore-check"
        or restore.get("ok") is not True
    ):
        raise PreparationError("cold restore receipt is not a successful cold-restore receipt")

    acceptance_sha256 = require_content_hash(restore.get("acceptance_sha256"), "restore acceptance sha256")
    marker_value = restore.get("marker")
    if not isinstance(marker_value, Mapping):
        raise PreparationError("cold restore receipt has no committed marker")
    marker_key = _validate_key(str(marker_value.get("key", "")), "restore marker key")
    marker_sha256 = require_content_hash(marker_value.get("sha256"), "restore marker sha256")
    marker_size_value = marker_value.get("size")
    if marker_size_value is None:
        marker_size_raw = backend.head(marker_key).get("size")
        if not marker_size_raw:
            raise PreparationError("committed marker size is unavailable", {"key": marker_key})
        try:
            marker_size_value = int(marker_size_raw)
        except (TypeError, ValueError) as error:
            raise PreparationError("committed marker size is invalid", {"key": marker_key}) from error
    marker_size = _positive_int(marker_size_value, "marker size")
    marker_reference = {"key": marker_key, "sha256": marker_sha256, "size": marker_size}
    marker_payload = _read_marker(backend, marker_reference)
    if acceptance_sha256.lower() not in _marker_acceptance_hashes(marker_payload):
        raise PreparationError("cold restore acceptance is not bound by the committed marker")
    for digest_field in ("batch_sha256", "release_sha256"):
        declared = restore.get(digest_field)
        if declared is not None and marker_payload.get(digest_field) != declared:
            raise PreparationError(
                "cold restore marker identity differs from the receipt",
                {"field": digest_field},
            )

    marker_records: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(_marker_object_records(marker_payload)):
        identity = _bare_object_identity(item, f"committed marker object[{index}]")
        key = str(identity["key"])
        if key in marker_records:
            raise PreparationError("committed marker contains duplicate object keys", {"key": key})
        marker_records[key] = item
    if not marker_records:
        raise PreparationError("committed marker contains no objects")

    restore_objects = restore.get("objects")
    if not isinstance(restore_objects, list) or not restore_objects:
        raise PreparationError("cold restore receipt has no restored object inventory")
    restored_records: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(restore_objects):
        identity = _bare_object_identity(item, f"restore object[{index}]")
        key = str(identity["key"])
        if key in restored_records:
            raise PreparationError("cold restore receipt contains duplicate objects", {"key": key})
        restored_path = item.get("restored_path")
        if not isinstance(restored_path, str) or not restored_path:
            raise PreparationError("cold restore receipt does not name every restored file", {"key": key})
        path = Path(restored_path).expanduser()
        restored_records[key] = {**identity, "restored_path": path.as_posix()}

    if set(marker_records) != set(restored_records):
        raise PreparationError("cold restore object inventory differs from the committed marker")
    for key in marker_records:
        if _bare_object_identity(marker_records[key], "committed marker object") != {
            field: restored_records[key][field] for field in ("key", "sha256", "size")
        }:
            raise PreparationError("cold restore object inventory differs from the committed marker", {"key": key})

    proofs: dict[str, RemoteRestoreProof] = {}
    manifest_keys = [key for key, item in marker_records.items() if item.get("purpose") == "manifest"]
    if len(manifest_keys) != 1:
        raise PreparationError("committed marker requires exactly one portable manifest")
    manifest_key = manifest_keys[0]
    manifest_payload: Mapping[str, object] | None = None
    manifest_bytes = b""
    evidence_session = _RemoteEvidenceSession(backend)
    for key in sorted(marker_records):
        marker_record = marker_records[key]
        identity = _bare_object_identity(marker_record, "committed marker object")
        if key == manifest_key:
            proof, payload = _readback_payload(
                backend,
                key,
                expected_sha256=str(identity["sha256"]),
                expected_size=int(identity["size"]),
            )
            manifest_bytes = payload
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PreparationError("remote portable manifest is not valid JSON", {"key": key}) from error
            if not isinstance(decoded, Mapping):
                raise PreparationError("remote portable manifest is not an object", {"key": key})
            manifest_payload = decoded
        else:
            proof = _readback(
                backend,
                key,
                expected_sha256=str(identity["sha256"]),
                expected_size=int(identity["size"]),
            )
        encryption = evidence_session.encryption_for(proof)
        if not encryption or encryption != marker_record.get("encryption"):
            raise PreparationError("remote encryption evidence differs from the committed marker", {"key": key})
        restored = restored_records[key]
        proofs[key] = RemoteRestoreProof(
            object_key=key,
            object_sha256=str(identity["sha256"]),
            object_size=int(identity["size"]),
            acceptance_sha256=acceptance_sha256,
            marker_key=marker_key,
            marker_sha256=marker_sha256,
            restore_receipt_sha256=restore_sha256,
            restored_sha256=str(restored["sha256"]),
            restored_size=int(restored["size"]),
            restore_receipt_path=restore_path,
        )
    if manifest_payload is None:
        raise PreparationError("committed marker requires exactly one portable manifest")
    return _CleanupVerificationSession(
        acceptance_sha256=acceptance_sha256,
        marker=marker_reference,
        marker_records=marker_records,
        manifest_bytes=manifest_bytes,
        manifest_payload=manifest_payload,
        proofs=proofs,
        restore_path=restore_path,
        restore_sha256=restore_sha256,
        restored_records=restored_records,
        token=_CLEANUP_SESSION_TOKEN,
    )


def _validate_restored_local_files(session: _CleanupVerificationSession) -> None:
    """Validate every cold-restored file before grouped raw-source deletion."""

    for key, record in session.restored_records.items():
        restored_path = record.get("restored_path")
        if not isinstance(restored_path, str) or not restored_path:
            raise PreparationError("cold restore receipt does not name every restored file", {"key": key})
        path = Path(restored_path).expanduser()
        if path.is_symlink() or not path.is_file():
            raise PreparationError("cold restore receipt names a missing or non-regular file", {"path": str(path)})
        if path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise PreparationError("cold restore receipt has a changed restored file", {"key": key})


def evict_copies(
    copies: Sequence[LocalCopy],
    *,
    restore_receipt: Path | str,
    backend: StorageBackend,
    preserve_on_interrupt: bool = False,
) -> list[LocalCopy]:
    """Delete grouped local copies after one fresh remote cleanup verification."""

    if not isinstance(copies, Sequence) or isinstance(copies, (str, bytes)):
        raise ContractError("grouped eviction copies must be an array")
    session = _build_cleanup_verification_session(restore_receipt, backend=backend)
    prepared: list[LocalCopy] = []
    paths: set[Path] = set()
    for copy in copies:
        if not isinstance(copy, LocalCopy):
            raise ContractError("grouped eviction copies must be LocalCopy values")
        normalized_path = copy.path.expanduser().resolve(strict=False)
        if normalized_path in paths:
            raise PreparationError("grouped eviction contains duplicate local paths", {"path": str(copy.path)})
        paths.add(normalized_path)
        proof = _group_copy_proof(copy, session)
        prepared.append(_prepare_group_eviction_copy(copy, proof=proof))
    return [
        _evict_copy_after_proof(copy, proof=_copy_proof(copy, None), preserve_on_interrupt=preserve_on_interrupt)
        for copy in prepared
    ]


def _portable_facts(value: object) -> object:
    """Remove local paths from a transform receipt in the same way as the manifest builder."""

    if isinstance(value, Mapping):
        return {
            key: _portable_facts(item)
            for key, item in value.items()
            if key not in {"path", "source_audio", "rttm_path", "uem_path", "bounds"}
        }
    if isinstance(value, list):
        return [_portable_facts(item) for item in value]
    return value


def _consumed_source_transform(
    request: ConsumedSource,
    transform_receipt: Mapping[str, object] | Path | str,
) -> tuple[dict[str, object], dict[str, object], Path | None, str | None]:
    transform, transform_path, transform_sha256 = _load_json_mapping(transform_receipt, "time-transform receipt")
    if transform.get("schema") != "speakrs-source-transforms-v1":
        raise ContractError("time-transform receipt schema is unknown")
    parents = transform.get("parents")
    if not isinstance(parents, list):
        raise ContractError("time-transform receipt must contain parent receipts")
    matches = [item for item in parents if isinstance(item, Mapping) and item.get("parent_id") == request.parent_id]
    if len(matches) != 1:
        raise PreparationError(
            "time-transform receipt must contain exactly one requested parent",
            {"parent_id": request.parent_id},
        )
    parent = dict(matches[0])
    if parent.get("schema") != "speakrs-parent-preparation" or parent.get("schema_version") != 2:
        raise PreparationError("time-transform receipt contains an unknown parent preparation receipt")
    source_audio = parent.get("source_audio")
    if not isinstance(source_audio, str) or not source_audio:
        raise PreparationError("parent transform receipt does not retain the original source path")
    if Path(source_audio).expanduser().resolve(strict=False) != request.path.expanduser().resolve(strict=False):
        raise PreparationError("source path differs from the exact path in the transform receipt")
    if require_content_hash(parent.get("source_sha256"), "transform source sha256") != request.source_sha256:
        raise PreparationError("source hash differs from the transform receipt", {"parent_id": request.parent_id})
    audio = parent.get("audio")
    transform_facts = parent.get("transform")
    if not isinstance(audio, Mapping) or not isinstance(transform_facts, Mapping) or not transform_facts:
        raise PreparationError("parent transform receipt lacks canonical audio or transform facts")
    require_content_hash(audio.get("sha256"), "canonical audio sha256")
    recorded_transform_sha256 = parent.get("transform_sha256")
    if recorded_transform_sha256 is not None and require_content_hash(
        recorded_transform_sha256, "transform sha256"
    ) != sha256_json(transform_facts):
        raise PreparationError("parent transform receipt has a changed transform")
    return transform, parent, transform_path, transform_sha256


def _consumed_source_output_refs(
    portable: Mapping[str, object],
    *,
    parent_id: str,
    canonical_sha256: str,
    label_sha256: str,
    uem_sha256: str,
) -> dict[str, dict[str, object]]:
    recordings = portable.get("recordings")
    if not isinstance(recordings, list):
        raise PreparationError("portable manifest has no recording inventory")
    matches = [item for item in recordings if isinstance(item, Mapping) and item.get("recording_id") == parent_id]
    if len(matches) != 1:
        raise PreparationError("portable manifest must contain exactly one requested parent", {"parent_id": parent_id})
    recording = matches[0]
    expected_hashes = {"audio": canonical_sha256, "rttm": label_sha256, "uem": uem_sha256}
    refs: dict[str, dict[str, object]] = {}
    for field, expected_sha256 in expected_hashes.items():
        reference = recording.get(field)
        if not isinstance(reference, Mapping):
            raise PreparationError("portable manifest parent is missing an output reference", {"field": field})
        key = reference.get("key")
        if not isinstance(key, str):
            raise ContractError(f"portable recording {field}.key must be a string")
        _validate_key(key, f"portable recording {field}.key")
        digest = require_content_hash(reference.get("sha256"), f"portable recording {field}.sha256")
        if digest != expected_sha256:
            raise PreparationError(
                "portable manifest output differs from the transform receipt",
                {"field": field, "parent_id": parent_id},
            )
        size = _positive_int(reference.get("size"), f"portable recording {field}.size")
        refs[field] = {"key": key, "sha256": digest, "size": size}
    return refs


def _consumed_source_receipts(
    accepted_outputs: Sequence[Mapping[str, object]],
    *,
    output_refs: Mapping[str, Mapping[str, object]],
    parent_id: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not isinstance(accepted_outputs, Sequence) or isinstance(accepted_outputs, (str, bytes)):
        raise ContractError("accepted output receipts must be an array")
    receipts = [_receipt_dict(item) for item in accepted_outputs]
    receipt_keys = [str(item["key"]) for item in receipts]
    if len(set(receipt_keys)) != len(receipt_keys):
        raise PreparationError("accepted output receipts contain duplicate keys")
    manifest_receipts = [item for item in receipts if item.get("purpose") == "manifest"]
    if len(manifest_receipts) != 1:
        raise PreparationError("accepted output receipts must contain exactly one portable manifest")
    for field, reference in output_refs.items():
        matches = [item for item in receipts if item.get("key") == reference["key"]]
        if len(matches) != 1:
            raise PreparationError(
                "accepted output receipts do not cover every required parent output",
                {"field": field, "parent_id": parent_id},
            )
        receipt = matches[0]
        if receipt.get("sha256") != reference["sha256"] or receipt.get("size") != reference["size"]:
            raise PreparationError("accepted output receipt differs from the portable manifest", {"field": field})
        expected_purpose = "train-audio" if field == "audio" else "train-label"
        if receipt.get("purpose") != expected_purpose or receipt.get("parent_id") != parent_id:
            raise PreparationError("accepted output receipt has the wrong parent or purpose", {"field": field})
    return receipts, manifest_receipts[0]


def _consumed_source_completion(intent: Mapping[str, object]) -> dict[str, object]:
    return _receipt_with_hash(
        {
            "schema": "speakrs-consumed-source-v1",
            "state": CONSUMED_SOURCE_STATE,
            "path": intent["path"],
            "source_path": intent["path"],
            "source_sha256": intent["source_sha256"],
            "source_size": intent["source_size"],
            "parent_id": intent["parent_id"],
            "transform_receipt_sha256": intent["transform_receipt_sha256"],
            "portable_manifest": intent["portable_manifest"],
            "accepted_outputs": intent["accepted_outputs"],
            "restore_receipt_path": intent["restore_receipt_path"],
            "restore_receipt_sha256": intent["restore_receipt_sha256"],
            "restore_marker": intent["restore_marker"],
            "restored_outputs": intent["restored_outputs"],
            "open_file_check": intent["open_file_check"],
            "deletion_journal_path": _deletion_journal_path(intent).as_posix(),
            "deleted": True,
        }
    )


def _deletion_completion_for_intent(intent: Mapping[str, object]) -> dict[str, object]:
    """Build the only completion receipt valid for one deletion intent."""

    operation = intent.get("operation")
    if operation == "eviction":
        return _eviction_completion(intent)
    if operation == "consumed-source":
        return _consumed_source_completion(intent)
    raise ContractError("deletion intent operation is unknown")


def _discard_consumed_source_with_session(
    request: ConsumedSource | Mapping[str, object],
    *,
    transform_receipt: Mapping[str, object] | Path | str,
    portable_manifest: Mapping[str, object] | Path | str,
    accepted_outputs: Sequence[Mapping[str, object]],
    session: _CleanupVerificationSession,
) -> dict[str, object]:
    """Delete one raw source file after a grouped remote proof and local checks.

    This API consumes exactly one parent file. It never consumes an archive, a directory,
    or multiple parents through one receipt.
    """

    typed = request if isinstance(request, ConsumedSource) else ConsumedSource.from_mapping(request)
    raw_path = typed.path.expanduser()
    task_root = typed.task_root.expanduser().resolve()
    source_path = raw_path.resolve(strict=False)
    if task_root == Path(task_root.anchor):
        raise ContractError("consumed source task_root cannot be the filesystem root")
    if raw_path.is_symlink():
        raise PreparationError("consumed source must be one existing regular file", {"path": str(raw_path)})
    if not source_path.is_relative_to(task_root) or source_path == task_root:
        raise PreparationError("consumed source is outside the managed staging root", {"path": str(raw_path)})
    if isinstance(typed.live_readers, bool) or typed.live_readers < 0:
        raise ContractError("source live_readers must be a non-negative integer")
    if typed.live_readers:
        raise PreparationError("active source readers block consumed-source deletion", {"path": str(raw_path)})
    source_exists = raw_path.is_file()
    source_size: int | None = None
    if source_exists:
        actual_source_sha256 = sha256_file(raw_path)
        if actual_source_sha256 != typed.source_sha256:
            raise PreparationError("source bytes differ from the consumed-source identity", {"path": str(raw_path)})
        source_size = raw_path.stat().st_size

    transform, parent, _, transform_file_sha256 = _consumed_source_transform(typed, transform_receipt)
    source_sha256 = require_content_hash(parent["source_sha256"], "parent source sha256")
    if source_sha256 != typed.source_sha256:
        raise PreparationError("parent transform source identity differs from the consumed-source request")
    audio = parent["audio"]
    canonical_sha256 = require_content_hash(audio["sha256"], "parent canonical audio sha256")
    canonical_path = audio.get("path")
    if not isinstance(canonical_path, str) or not canonical_path:
        raise PreparationError("parent transform receipt does not retain the canonical audio path")
    if Path(canonical_path).expanduser().resolve(strict=False) == source_path:
        raise PreparationError("parent transform receipt maps the source onto itself")
    label_sha256 = require_content_hash(parent.get("label_sha256"), "parent label sha256")
    uem_sha256 = require_content_hash(parent.get("uem_sha256"), "parent UEM sha256")

    portable, portable_path, portable_file_sha256 = _load_json_mapping(portable_manifest, "portable manifest")
    if portable.get("schema") != "speakrs-portable-training-selection-v1":
        raise ContractError("portable manifest schema is unknown")
    if portable.get("acceptance_sha256") is None:
        raise PreparationError("portable manifest has no acceptance identity")
    evidence_hashes = portable.get("evidence_hashes")
    if transform_file_sha256 is not None:
        if (
            not isinstance(evidence_hashes, Mapping)
            or require_content_hash(evidence_hashes.get("time_transform"), "portable time-transform sha256")
            != transform_file_sha256
        ):
            raise PreparationError("portable manifest is not bound to the local time-transform receipt")
    provenance = portable.get("provenance")
    portable_transform = provenance.get("time_transform") if isinstance(provenance, Mapping) else None
    if not isinstance(portable_transform, Mapping):
        raise PreparationError("portable manifest has no time-transform provenance")
    if portable_transform != _portable_facts(transform):
        raise PreparationError("portable manifest time-transform provenance differs from the local receipt")
    portable_parents = portable_transform.get("parents")
    if not isinstance(portable_parents, list):
        raise PreparationError("portable manifest time-transform provenance has no parent receipts")
    portable_parent_matches = [
        item for item in portable_parents if isinstance(item, Mapping) and item.get("parent_id") == typed.parent_id
    ]
    if len(portable_parent_matches) != 1:
        raise PreparationError("portable manifest lacks the requested parent transform proof")
    portable_parent = portable_parent_matches[0]
    if (
        require_content_hash(portable_parent.get("source_sha256"), "portable source sha256") != source_sha256
        or not isinstance(portable_parent.get("audio"), Mapping)
        or require_content_hash(portable_parent["audio"].get("sha256"), "portable canonical audio sha256")
        != canonical_sha256
        or not isinstance(portable_parent.get("transform"), Mapping)
        or not portable_parent["transform"]
    ):
        raise PreparationError("portable manifest does not bind source bytes to canonical transform")
    output_refs = _consumed_source_output_refs(
        portable,
        parent_id=typed.parent_id,
        canonical_sha256=canonical_sha256,
        label_sha256=label_sha256,
        uem_sha256=uem_sha256,
    )
    receipts, manifest_receipt = _consumed_source_receipts(
        accepted_outputs,
        output_refs=output_refs,
        parent_id=typed.parent_id,
    )
    manifest_source = portable.get("source")
    manifest_version = portable.get("version")
    if not isinstance(manifest_source, str) or not manifest_source:
        raise PreparationError("portable manifest has no source identity")
    if not isinstance(manifest_version, str) or not manifest_version:
        raise PreparationError("portable manifest has no version identity")
    if any(item.get("source") != manifest_source or item.get("version") != manifest_version for item in receipts):
        raise PreparationError("accepted output receipt source/version differs from the portable manifest")
    acceptance_sha256 = session.acceptance_sha256
    if require_content_hash(portable.get("acceptance_sha256"), "portable acceptance sha256") != acceptance_sha256:
        raise PreparationError("portable manifest and cold restore acceptance identities differ")
    marker_reference = session.marker
    marker_key = str(marker_reference["key"])
    marker_sha256 = str(marker_reference["sha256"])
    marker_records = session.marker_records
    for receipt in receipts:
        key = str(receipt["key"])
        marker_record = marker_records.get(key)
        if marker_record is None:
            raise PreparationError("accepted output is absent from the committed marker", {"key": key})
        _compare_object_identity(receipt, marker_record, "consumed-source marker")
        if marker_record.get("encryption") != receipt.get("encryption"):
            raise PreparationError("accepted output encryption differs from its receipt", {"key": key})
        restored = session.restored_records.get(key)
        if restored is None or any(restored[field] != receipt[field] for field in ("key", "sha256", "size")):
            raise PreparationError("cold restore proof does not cover an accepted output", {"key": key})
    restored = session.restored_records
    manifest_key = str(manifest_receipt["key"])
    if manifest_key not in session.marker_records or session.marker_records[manifest_key].get("purpose") != "manifest":
        raise PreparationError("accepted output manifest is absent from the committed marker", {"key": manifest_key})
    manifest_bytes = session.manifest_bytes
    remote_portable = session.manifest_payload
    if len(manifest_bytes) != manifest_receipt["size"] or sha256_bytes(manifest_bytes) != manifest_receipt["sha256"]:
        raise PreparationError("remote portable manifest failed full readback", {"key": manifest_key})
    if remote_portable != portable:
        raise PreparationError("portable manifest differs from the committed remote object", {"key": manifest_key})
    if portable_path is not None:
        if portable_file_sha256 != manifest_receipt["sha256"] or portable_path.read_bytes() != manifest_bytes:
            raise PreparationError("local portable manifest differs from the committed remote object")
    restore_receipt_sha256 = session.restore_sha256
    restore_receipt_path = session.restore_path
    restore_marker = {"key": marker_key, "sha256": marker_sha256}
    transform_receipt_sha256 = transform_file_sha256 or sha256_json(transform)
    accepted_output_identities = [
        {"key": item["key"], "sha256": item["sha256"], "size": item["size"]} for item in receipts
    ]
    expected_identity = {
        "source_sha256": source_sha256,
        "parent_id": typed.parent_id,
        "acceptance_sha256": acceptance_sha256,
        "restore_marker": restore_marker,
        "restore_receipt_sha256": restore_receipt_sha256,
    }
    if not source_exists:
        recovered = recover_deletion_journal(
            restore_receipt_path,
            operation="consumed-source",
            path=raw_path,
            expected_identity=expected_identity,
        )
        if recovered is None:
            raise PreparationError("consumed source must be one existing regular file", {"path": str(raw_path)})
        completion = recovered.get("completion")
        if recovered.get("state") == DELETION_INTENT_STATE:
            intent = recovered.get("intent")
            if not isinstance(intent, Mapping):
                raise PreparationError("deletion journal has no intent", {"path": str(raw_path)})
            completion = _consumed_source_completion(intent)
            _persist_deletion_completion(intent, completion)
        if not isinstance(completion, Mapping):
            raise PreparationError("completed deletion journal has no receipt", {"path": str(raw_path)})
        return dict(completion)

    if source_size is None:
        raise PreparationError("consumed source size is unavailable", {"path": str(raw_path)})
    _recheck_local_identity(
        raw_path,
        expected_sha256=typed.source_sha256,
        expected_size=source_size,
        label="consumed source",
    )
    open_file_check = assert_no_open_file_handles(raw_path)
    intent = {
        "operation": "consumed-source",
        "path": raw_path.resolve(strict=False).as_posix(),
        "source_sha256": source_sha256,
        "source_size": source_size,
        "parent_id": typed.parent_id,
        "transform_receipt_sha256": transform_receipt_sha256,
        "portable_manifest": {
            "key": manifest_key,
            "sha256": manifest_receipt["sha256"],
            "size": manifest_receipt["size"],
        },
        "accepted_outputs": accepted_output_identities,
        "acceptance_sha256": acceptance_sha256,
        "restore_receipt_path": restore_receipt_path.as_posix(),
        "restore_receipt_sha256": restore_receipt_sha256,
        "restore_marker": restore_marker,
        "restored_outputs": sorted(restored),
        "open_file_check": open_file_check,
    }
    _persist_deletion_intent(intent)
    _recheck_local_identity(
        raw_path,
        expected_sha256=typed.source_sha256,
        expected_size=source_size,
        label="consumed source",
    )
    try:
        raw_path.unlink()
    except OSError as error:
        raise PreparationError(
            "consumed source could not be deleted", {"path": str(raw_path), "error": str(error)}
        ) from error
    deletion = _consumed_source_completion(intent)
    _persist_deletion_completion(intent, deletion)
    return deletion


def discard_consumed_sources(
    requests: Sequence[ConsumedSource | Mapping[str, object]],
    *,
    transform_receipt: Mapping[str, object] | Path | str,
    portable_manifest: Mapping[str, object] | Path | str | None = None,
    accepted_outputs: Sequence[Mapping[str, object]] | None = None,
    restore_receipt: Path | str,
    backend: StorageBackend,
) -> list[dict[str, object]]:
    """Delete grouped raw source files after one fresh remote cleanup verification."""

    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        raise ContractError("grouped consumed sources must be an array")
    session = _build_cleanup_verification_session(restore_receipt, backend=backend)
    typed_requests: list[ConsumedSource] = []
    paths: set[Path] = set()
    parents: set[str] = set()
    for request in requests:
        typed = request if isinstance(request, ConsumedSource) else ConsumedSource.from_mapping(request)
        path = typed.path.expanduser().resolve(strict=False)
        if path in paths:
            raise PreparationError("grouped consumed sources contain duplicate paths", {"path": str(typed.path)})
        if typed.parent_id in parents:
            raise PreparationError(
                "grouped consumed sources contain duplicate parent ids", {"parent_id": typed.parent_id}
            )
        paths.add(path)
        parents.add(typed.parent_id)
        typed_requests.append(typed)
    transform, _, _ = _load_json_mapping(transform_receipt, "time-transform receipt")
    transform_parents = transform.get("parents")
    if transform.get("schema") != "speakrs-source-transforms-v1" or not isinstance(transform_parents, list):
        raise ContractError("time-transform receipt schema is unknown")
    transform_parent_ids = {item.get("parent_id") for item in transform_parents if isinstance(item, Mapping)}
    if len(transform_parent_ids) != len(transform_parents) or None in transform_parent_ids:
        raise PreparationError("time-transform receipt contains duplicate or invalid parents")
    if portable_manifest is None:
        portable = session.manifest_payload
        portable_path = None
        portable_sha256 = None
    else:
        portable, portable_path, portable_sha256 = _load_json_mapping(portable_manifest, "portable manifest")
    if portable != session.manifest_payload:
        raise PreparationError("portable manifest differs from the committed remote object")
    if portable_path is not None and portable_sha256 != sha256_bytes(session.manifest_bytes):
        raise PreparationError("local portable manifest differs from the committed remote object")
    selected_parent_ids = portable.get("selected_parent_ids")
    if not isinstance(selected_parent_ids, list):
        recordings = portable.get("recordings")
        selected_parent_ids = [item.get("recording_id") for item in recordings or () if isinstance(item, Mapping)]
    if len(selected_parent_ids) != len(set(selected_parent_ids)) or set(selected_parent_ids) != transform_parent_ids:
        raise PreparationError("consumed audio-parent inventory differs from the committed batch")
    if not parents <= transform_parent_ids:
        raise PreparationError("grouped consumed sources contain an unknown committed parent")
    if accepted_outputs is None:
        grouped_outputs = list(session.marker_records.values())
    else:
        grouped_outputs = list(accepted_outputs)
    output_keys: list[str] = []
    for item in grouped_outputs:
        identity = _object_identity(item, "accepted output")
        key = str(identity["key"])
        if key in output_keys:
            raise PreparationError("accepted output receipts contain duplicate keys", {"key": key})
        marker_record = session.marker_records.get(key)
        if marker_record is None:
            raise PreparationError("accepted output is absent from the committed marker", {"key": key})
        _compare_object_identity(item, marker_record, "grouped cleanup marker")
        if item.get("encryption") != marker_record.get("encryption"):
            raise PreparationError("accepted output encryption differs from its receipt", {"key": key})
        output_keys.append(key)
    if set(output_keys) != set(session.marker_records):
        raise PreparationError("accepted output inventory differs from the committed marker")
    if any(typed.path.expanduser().exists() for typed in typed_requests):
        # completed deletion journals may outlive the restored local copies
        _validate_restored_local_files(session)
    grouped_portable = session.manifest_payload if portable_manifest is None else portable_manifest
    return [
        _discard_consumed_source_with_session(
            typed,
            transform_receipt=transform_receipt,
            portable_manifest=grouped_portable,
            accepted_outputs=grouped_outputs,
            session=session,
        )
        for typed in typed_requests
    ]


def discard_consumed_source(
    request: ConsumedSource | Mapping[str, object],
    *,
    transform_receipt: Mapping[str, object] | Path | str,
    portable_manifest: Mapping[str, object] | Path | str,
    accepted_outputs: Sequence[Mapping[str, object]],
    restore_receipt: Mapping[str, object] | Path | str,
    backend: StorageBackend,
) -> dict[str, object]:
    """Delete one raw source file through the grouped cleanup owner."""

    if isinstance(restore_receipt, Mapping):
        raise ContractError("single consumed-source cleanup requires a durable cold restore receipt path")
    return discard_consumed_sources(
        [request],
        transform_receipt=transform_receipt,
        portable_manifest=portable_manifest,
        accepted_outputs=accepted_outputs,
        restore_receipt=restore_receipt,
        backend=backend,
    )[0]


def restore_object(
    backend: StorageBackend,
    receipt: Mapping[str, object],
    destination: Path,
    *,
    proof: RemoteRestoreProof | Mapping[str, object] | None = None,
    max_bytes: int | None = DEFAULT_MAX_OBJECT_BYTES,
) -> Path:
    """Cold-restore one object through its content hash, not a source path."""

    raw_key = receipt.get("key")
    if not isinstance(raw_key, str):
        raise ContractError("restore.key must be a string")
    key = _validate_key(raw_key, "restore.key")
    expected = require_content_hash(receipt.get("sha256"), "restore sha256")
    size_value = receipt.get("size")
    expected_size = None if size_value is None else _positive_int(size_value, "restore.size")
    if proof is not None:
        typed_proof = verify_remote_restore_proof(proof, backend=backend)
        if (
            typed_proof.object_key != key
            or typed_proof.object_sha256 != expected
            or (expected_size is not None and typed_proof.object_size != expected_size)
        ):
            raise PreparationError("restore proof does not match the requested object", {"key": key})
    marker = receipt.get("marker") or receipt.get("remote_marker")
    if isinstance(marker, Mapping):
        marker_payload = _read_marker(backend, marker)
        if not any(
            _object_identity(item)["key"] == key
            and _object_identity(item)["sha256"] == expected
            and (expected_size is None or _object_identity(item)["size"] == expected_size)
            for item in _marker_object_records(marker_payload)
        ):
            raise PreparationError("restore object is absent from the committed marker", {"key": key})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    digest = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as handle:
            for chunk in _iter_backend_bytes(backend, key, max_bytes=max_bytes):
                digest.update(chunk)
                size += len(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        actual = digest.hexdigest()
        if actual != expected or (expected_size is not None and size != expected_size):
            raise PreparationError(
                "restore hash or size mismatch",
                {
                    "key": key,
                    "actual": actual,
                    "expected": expected,
                    "actual_size": size,
                    "expected_size": expected_size,
                },
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if not destination.is_file() or sha256_file(destination) != expected:
        raise PreparationError("restore write mismatch", {"path": destination.as_posix()})
    return destination


def selection_objects_may_upload(membership: SourceMembership, selection: SelectionState) -> None:
    """Refuse to upload anything that is not an accepted training selection."""

    if membership is not SourceMembership.ACCEPTED or selection is not SelectionState.ACCEPTED:
        raise PreparationError(
            "only accepted train artifacts enter the minimal package",
            {"membership": membership.value, "selection": selection.value},
        )
