"""External-disk probing and relocatable path helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .errors import PreparationError
from .jsonio import write_json


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
