"""Content hashing helpers used by locks, releases, and receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of a byte string."""

    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 text."""

    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value with stable key order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON encoding."""

    return sha256_text(canonical_json(value))


def sha256_paths(paths: Iterable[Path], root: Path | None = None) -> dict[str, str]:
    """Return relative-path SHA-256 records for existing files."""

    records: dict[str, str] = {}
    for path in sorted(paths, key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        key = path.relative_to(root).as_posix() if root is not None else path.as_posix()
        records[key] = sha256_file(path)
    return records


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Return a mapping or raise a TypeError."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value
