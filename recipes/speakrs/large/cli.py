"""Command implementations behind the thin large_run.py CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .budget import BudgetLedger
from .contracts import DEFAULT_BUDGET, LAUNCH_KIND, parse_kinded_lock
from .controller import Controller, FakeProvider
from .errors import LargeError
from .handoff import package_handoff
from .image import verify_image
from .jsonio import write_json
from .prepare import load_spec, prepare_release, verify_release
from .verify import verify_local


COMMANDS = (
    "prepare",
    "verify-data",
    "export-wavlm",
    "verify-local",
    "preflight",
    "qualify",
    "freeze-run",
    "control",
    "supervise",
    "select",
    "test",
    "archive",
    "status",
    "resume",
    "verify-image",
    "package-handoff",
)


def _print_json(payload: dict, *, stream=sys.stdout) -> None:
    stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fail(error: LargeError) -> int:
    _print_json(error.to_json(), stream=sys.stdout)
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the public command parser."""

    parser = argparse.ArgumentParser(prog="large_run.py", description="Speakrs WavLM Large pre-rental commands")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="plan or materialize the full CC release")
    prepare.add_argument("--spec", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--plan", action="store_true")

    verify_data = sub.add_parser("verify-data", help="verify a sealed release")
    verify_data.add_argument("--release", required=True, type=Path)

    export_cmd = sub.add_parser("export-wavlm", help="export a WavLM initializer")
    export_cmd.add_argument("--variant", choices=("large", "base_plus"), default="large")
    export_cmd.add_argument("--output", required=True, type=Path)

    verify_local_cmd = sub.add_parser("verify-local", help="run CPU pre-rental checks")
    verify_local_cmd.add_argument("--spec", required=True, type=Path)
    verify_local_cmd.add_argument("--stage", choices=("pre-rental",), default="pre-rental")
    verify_local_cmd.add_argument(
        "--check",
        action="append",
        dest="checks",
        choices=(
            "data",
            "model",
            "sampling",
            "runtime",
            "selection",
            "controller",
            "image",
            "backup",
            "external-controls",
        ),
    )
    verify_local_cmd.add_argument("--output", type=Path)

    preflight = sub.add_parser("preflight", help="worker preflight")
    preflight.add_argument("--spec", required=True, type=Path)
    preflight.add_argument("--require-cuda", action="store_true")

    qualify = sub.add_parser("qualify", help="GPU qualification (not run in this goal)")
    qualify.add_argument("--spec", required=True, type=Path)
    qualify.add_argument("--gpu-profile", required=True)
    qualify.add_argument("--spend-ceiling-usd", type=float, default=5.0)

    freeze = sub.add_parser("freeze-run", help="freeze a launch lock after qualification")
    freeze.add_argument("--spec", required=True, type=Path)
    freeze.add_argument("--qualification", required=True, type=Path)
    freeze.add_argument("--offer", required=True, type=Path)
    freeze.add_argument("--budget", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)

    control = sub.add_parser("control", help="trusted controller")
    control.add_argument("--launch", type=Path)
    control.add_argument("--qualification-lease", type=Path)
    control.add_argument("--connection", type=Path)
    control.add_argument("--backup-root", type=Path)

    supervise = sub.add_parser("supervise", help="single worker supervisor")
    supervise.add_argument("--launch", required=True, type=Path)

    select = sub.add_parser("select", help="development selection")
    select.add_argument("--launch", required=True, type=Path)

    test = sub.add_parser("test", help="held-out tests after selection seal")
    test.add_argument("--launch", required=True, type=Path)

    archive = sub.add_parser("archive", help="complete or incomplete archive")
    archive.add_argument("--launch", required=True, type=Path)

    status = sub.add_parser("status", help="controller/worker status")
    status.add_argument("--launch", required=True, type=Path)

    resume = sub.add_parser("resume", help="resume from trusted state")
    resume.add_argument("--launch", required=True, type=Path)

    image = sub.add_parser("verify-image", help="verify a published image digest")
    image.add_argument("--spec", required=True, type=Path)
    image.add_argument("--image", required=True)
    image.add_argument("--output", type=Path)

    handoff = sub.add_parser("package-handoff", help="write the hashed pre-rental package")
    handoff.add_argument("--spec", required=True, type=Path)
    handoff.add_argument("--verification", required=True, type=Path)
    handoff.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one Large command and always emit JSON."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        if error.code == 0:
            return 0
        _print_json({"ok": False, "error": {"code": "usage", "message": "invalid arguments"}})
        return 2
    try:
        result = dispatch(args)
    except LargeError as error:
        return _fail(error)
    except Exception as error:  # noqa: BLE001 - CLI must not emit traceback-only failures
        _print_json(
            {
                "ok": False,
                "error": {
                    "code": "internal",
                    "message": str(error),
                    "type": type(error).__name__,
                },
            }
        )
        return 1
    payload = {"ok": True, **result}
    output = getattr(args, "output", None)
    if isinstance(output, Path) and args.command in {"verify-local", "verify-image"}:
        write_json(output, payload)
    _print_json(payload)
    return 0 if payload.get("ok") else 1


def dispatch(args: argparse.Namespace) -> dict:
    """Dispatch a parsed command."""

    command = args.command
    if command == "prepare":
        spec = load_spec(args.spec)
        return prepare_release(spec, args.output, plan_only=bool(args.plan))
    if command == "verify-data":
        return verify_release(args.release)
    if command == "export-wavlm":
        from .export_wavlm import export_variant

        return export_variant(args.variant, args.output)
    if command == "verify-local":
        spec = load_spec(args.spec)
        result = verify_local(spec, stage=args.stage, checks=tuple(args.checks) if args.checks else None)
        if args.output:
            write_json(args.output, {"ok": result.get("ok", False), **result})
        if not result.get("ok"):
            raise LargeError("verify", "pre-rental verification failed", result)
        return result
    if command == "preflight":
        spec = load_spec(args.spec)
        if args.require_cuda:
            raise LargeError("preflight", "CUDA is required and this host has no GPU", {"require_cuda": True})
        return {"spec": spec.run_id, "cuda": False}
    if command == "qualify":
        return {
            "ok": False,
            "gpu_qualification_status": "not_run",
            "message": "GPU qualification is outside this pre-rental goal",
            "gpu_profile": args.gpu_profile,
            "spend_ceiling_usd": args.spend_ceiling_usd,
        }
    if command == "freeze-run":
        qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
        if qualification.get("gpu_qualification_status") in (None, "not_run") or not qualification.get("ok"):
            raise LargeError("freeze-run", "cannot freeze a launch lock without GPU qualification")
        parse_kinded_lock(
            {
                "kind": LAUNCH_KIND,
                "launch_id": "rejected-without-qualification",
                "offer": json.loads(args.offer.read_text(encoding="utf-8")),
                "physical_batch": qualification.get("physical_batch"),
                "accumulation": qualification.get("accumulation"),
                "affordable_cycles": qualification.get("affordable_cycles"),
                "worker_deadline": qualification.get("worker_deadline"),
                "qualification_digest": "missing",
                "gpu_qualification_status": qualification.get("gpu_qualification_status"),
            },
            LAUNCH_KIND,
        )
        return {"ok": True}
    if command == "control":
        if bool(args.launch) == bool(args.qualification_lease):
            raise LargeError("control", "--launch and --qualification-lease are mutually exclusive")
        from .recovery import LocalTransport

        controller = Controller(
            ledger=BudgetLedger(DEFAULT_BUDGET),
            provider=FakeProvider(),
            transport=LocalTransport(),
        )
        if args.qualification_lease:
            lease = json.loads(args.qualification_lease.read_text(encoding="utf-8"))
            return controller.control_qualification(lease)
        parse_kinded_lock(json.loads(args.launch.read_text(encoding="utf-8")), LAUNCH_KIND)
        return {"ok": True, "mode": "launch"}
    if command in {"supervise", "select", "test", "archive", "status", "resume"}:
        payload = json.loads(args.launch.read_text(encoding="utf-8"))
        parse_kinded_lock(payload, LAUNCH_KIND)
        if command == "resume" and payload.get("state") == "terminal":
            raise LargeError("resume", "terminal run cannot restart training")
        return {"command": command, "launch_id": payload.get("launch_id")}
    if command == "verify-image":
        spec = load_spec(args.spec)
        return verify_image(spec, args.image, args.output)
    if command == "package-handoff":
        spec = load_spec(args.spec)
        verification = json.loads(args.verification.read_text(encoding="utf-8"))
        return package_handoff(spec, verification, args.output)
    raise LargeError("usage", f"unknown command {command}")
