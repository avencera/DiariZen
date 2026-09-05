"""Build-time identity and digest verification for the public GHCR image."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .errors import LargeError
from .jsonio import write_json


FORBIDDEN_LAYER_HINTS = (
    ".flac",
    ".wav",
    ".pt",
    ".ckpt",
    "id_rsa",
    ".env",
    "credentials.json",
    "HF_TOKEN",
    "VAST_API_KEY",
)


def verify_image(spec, image: str, output: Path | None) -> dict:
    """Pull anonymously, inspect layers, and record the digest lock."""

    if "@sha256:" not in image:
        raise LargeError("image", "image reference must be an immutable digest", {"image": image})
    digest = image.split("@", 1)[1]
    anonymous = _anonymous_manifest(image)
    inspect = _inspect_image(image)
    forbidden = [hint for hint in FORBIDDEN_LAYER_HINTS if hint.lower() in inspect.lower()]
    if forbidden:
        raise LargeError("image", "image appears to contain data or credentials", {"hints": forbidden})
    payload = {
        "ok": True,
        "reference": image,
        "digest": digest,
        "anonymous_manifest": anonymous,
        "inspect": inspect[-4000:],
    }
    lock_path = spec.artifacts_root / "image.lock.json"
    spec.artifacts_root.mkdir(parents=True, exist_ok=True)
    write_json(lock_path, payload)
    if output is not None:
        write_json(output, payload)
    return payload


def _anonymous_manifest(image: str) -> dict:
    env = os.environ.copy()
    for name in ("DOCKER_CONFIG", "REGISTRY_AUTH_FILE"):
        env.pop(name, None)
    env["DOCKER_CONFIG"] = str(Path("/nonexistent-docker-config"))
    completed = subprocess.run(
        ["docker", "manifest", "inspect", image],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise LargeError(
            "image",
            "anonymous manifest inspect failed",
            {"stderr": completed.stderr[-800:]},
        )
    return json.loads(completed.stdout)


def _inspect_image(image: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pull = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            check=False,
        )
        if pull.returncode != 0:
            raise LargeError("image", "digest pull failed", {"stderr": pull.stderr[-800:]})
        completed = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise LargeError("image", "image inspect failed", {"stderr": completed.stderr[-800:]})
    return completed.stdout
