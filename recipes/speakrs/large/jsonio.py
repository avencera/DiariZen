"""Atomic JSON publication used by locks, receipts, and CLI results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .hashing import canonical_json


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text through a temporary file, fsync, and replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    try:
        directory_fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def write_json(path: Path, value: Any) -> None:
    """Write pretty-printed JSON with a trailing newline."""

    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_canonical_json(path: Path, value: Any) -> None:
    """Write canonical JSON used as a hashed identity."""

    atomic_write_text(path, canonical_json(value))


def read_json(path: Path) -> Any:
    """Read a JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))
