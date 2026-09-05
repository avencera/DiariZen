"""Hashed pre-rental handoff package."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .contracts import DEFAULT_BUDGET, PREPARATION_KIND, spec_to_json
from .errors import LargeError
from .hashing import sha256_file
from .jsonio import atomic_write_text, write_json
from .prepare import verify_release


def package_handoff(spec, verification: dict, output: Path) -> dict:
    """Write the pre-rental package. Never fabricates qualification or launch locks."""

    output.mkdir(parents=True, exist_ok=True)
    source_commit = _git_commit()
    image_lock = spec.artifacts_root / "image.lock.json"
    image = json.loads(image_lock.read_text(encoding="utf-8")) if image_lock.is_file() else {}
    data_ok = False
    data_report = {}
    if (spec.release_root / "release.complete.json").is_file():
        try:
            data_report = verify_release(spec.release_root)
            data_ok = True
        except Exception as error:  # noqa: BLE001
            data_report = {"ok": False, "error": str(error)}
    work_complete = bool(verification.get("pre_rental_work_complete")) and data_ok and bool(image.get("digest"))
    external = (verification.get("checks") or {}).get("external-controls") or {}
    rental_gate = external.get("rental_gate_status") or "blocked_external_control"
    if work_complete and rental_gate == "ready_to_select_qualification_offer" and not external.get("missing"):
        rental_status = "ready_to_select_qualification_offer"
    else:
        rental_status = "blocked_external_control"
    budget = DEFAULT_BUDGET.identity()
    write_json(output / "budget.json", budget)
    write_json(
        output / "preparation.lock.json",
        {
            "kind": PREPARATION_KIND,
            "run_id": spec.run_id,
            "source_commit": source_commit,
            "spec": spec_to_json(spec),
            "data": data_report,
            "model_artifact": str(spec.artifacts_root / "wavlm-large-torchaudio.pt"),
            "image": image,
            "budget": budget,
        },
    )
    write_json(
        output / "qualification-plan.json",
        {
            "gpu_checks": ["G1-fresh-process", "G2-memory-throughput", "G3-numerical-recovery", "G4-canary"],
            "candidate_batches": {"4090": [2, 4, 8], "5090": [4, 8, 16]},
            "offer_fields": ["offer_id", "usd_per_hour", "disk_usd_per_hour", "gpu_profile", "hard_deadline"],
            "qualification_ceiling_usd": 5.0,
            "status": "not_run",
        },
    )
    readiness = {
        "pre_rental_work_complete": work_complete,
        "rental_gate_status": rental_status,
        "gpu_qualification_status": "not_run",
        "blockers": external.get("missing") or [],
        "image_digest": image.get("digest"),
        "source_commit": source_commit,
        "restore_commands": [
            f"cd {Path(__file__).resolve().parents[3]}",
            "python recipes/speakrs/large_run.py verify-data --release recipes/speakrs/data/large_cc_v1",
            "python recipes/speakrs/large_run.py status --launch launch.lock.json",
        ],
    }
    write_json(output / "readiness.json", readiness)
    write_json(output / "cpu-verification.json", verification)
    _write_sums(output)
    if (output / "qualification.json").exists() or (output / "launch.lock.json").exists():
        raise LargeError("handoff", "pre-rental package must not contain fabricated qualification or launch locks")
    return {
        "output": str(output),
        "pre_rental_work_complete": work_complete,
        "rental_gate_status": rental_status,
        "gpu_qualification_status": "not_run",
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def _write_sums(output: Path) -> None:
    lines = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(output).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    atomic_write_text(output / "SHA256SUMS", "\n".join(lines) + "\n")
