"""Command implementations behind the thin large_run.py CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .budget import BudgetLedger
from .contracts import DEFAULT_BUDGET, LAUNCH_KIND, parse_kinded_lock, parse_qualification_binding
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
    "data",
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

DATA_COMMANDS = (
    "plan",
    "prepare",
    "verify",
    "upload",
    "verify-remote",
    "commit-release",
    "restore-check",
    "final-restore",
    "evict",
    "handoff",
    "qualification-bundle",
    "qualification-binding",
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

    qualify = sub.add_parser("qualify", help="qualify a rented RTX 4090 with the supplied trainer TOML")
    qualify.add_argument(
        "--trainer-config",
        dest="trainer_config",
        required=True,
        type=Path,
        help="trainer TOML containing the real model, audio, RTTM, and UEM paths",
    )
    qualify.add_argument("--qualification-spec", type=Path, help="optional JSON qualification bounds")
    qualify.add_argument(
        "--qualification-binding",
        required=True,
        dest="qualification_binding",
        type=Path,
        help="strict data-release binding created by data qualification-binding",
    )
    qualify.add_argument("--qualification-lease", required=True, type=Path)
    qualify.add_argument("--gpu-profile", required=True)
    qualify.add_argument("--warmup-optimizer-updates", "--warmup-updates", type=int)
    qualify.add_argument("--measured-optimizer-updates", "--measured-updates", type=int)
    qualify.add_argument("--effective-batch", type=int)
    qualify.add_argument("--physical-batch-candidate", action="append", type=int, dest="physical_batch_candidates")
    qualify.add_argument("--planned-updates-per-cycle", "--updates-per-cycle", type=int)
    qualify.add_argument("--planned-max-cycles", "--max-cycles", type=int)
    qualify.add_argument("--output", required=True, type=Path)

    freeze = sub.add_parser("freeze-run", help="freeze a launch lock after qualification")
    freeze.add_argument("--spec", required=True, type=Path)
    freeze.add_argument("--qualification", required=True, type=Path)
    freeze.add_argument("--offer", required=True, type=Path)
    freeze.add_argument("--budget", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)

    control = sub.add_parser("control", help="trusted controller")
    control.add_argument("--launch", type=Path)
    control.add_argument("--qualification-lease", type=Path)
    control.add_argument("--qualification-binding", type=Path)
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

    data = sub.add_parser("data", help="verify and store accepted training inputs")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_plan = data_sub.add_parser("plan", help="validate membership and emit required inputs")
    data_plan.add_argument("--config", required=True, type=Path)
    data_plan.add_argument("--output", required=True, type=Path)
    data_prepare = data_sub.add_parser("prepare", help="process eligible sources with bounded staging")
    data_prepare.add_argument("--config", required=True, type=Path)
    data_prepare.add_argument("--source", default="all")
    data_prepare.add_argument("--output", required=True, type=Path)
    data_verify = data_sub.add_parser("verify", help="verify content, QA, splits, hours, and capacity")
    data_verify.add_argument("--release", required=True, type=Path)
    data_verify.add_argument("--config", required=True, type=Path)
    data_verify.add_argument("--output", required=True, type=Path)
    data_upload = data_sub.add_parser("upload", help="upload accepted training artifacts")
    data_upload.add_argument("--release", required=True, type=Path)
    data_upload.add_argument("--config", required=True, type=Path)
    data_upload.add_argument("--output", required=True, type=Path)
    data_upload.add_argument("--artifacts", type=Path)
    data_verify_remote = data_sub.add_parser("verify-remote", help="full readback and batch commit")
    data_verify_remote.add_argument("--receipt", required=True, type=Path)
    data_verify_remote.add_argument("--config", required=True, type=Path)
    data_verify_remote.add_argument("--output", required=True, type=Path)
    data_commit = data_sub.add_parser("commit-release", help="commit the final remote release seal")
    data_commit.add_argument("--release", required=True, type=Path)
    data_commit.add_argument("--receipts", required=True, type=Path)
    data_commit.add_argument("--config", required=True, type=Path)
    data_commit.add_argument("--output", required=True, type=Path)
    data_restore = data_sub.add_parser("restore-check", help="cold restore through the production reader")
    data_restore.add_argument("--receipt", required=True, type=Path)
    data_restore.add_argument("--config", required=True, type=Path)
    data_restore.add_argument("--output", required=True, type=Path)
    data_final_restore = data_sub.add_parser(
        "final-restore", help="build a content-bound restore index for the committed release"
    )
    data_final_restore.add_argument("--release", required=True, type=Path)
    data_final_restore.add_argument("--seal", required=True, type=Path)
    data_final_restore.add_argument("--config", required=True, type=Path)
    data_final_restore.add_argument("--output", required=True, type=Path)
    data_final_restore.add_argument("--restores", type=Path)
    data_evict = data_sub.add_parser("evict", help="remove eligible task-owned local files")
    data_evict.add_argument("--receipt", required=True, type=Path)
    data_evict.add_argument("--config", required=True, type=Path)
    data_evict.add_argument("--output", required=True, type=Path)
    data_data_handoff = data_sub.add_parser("handoff", help="emit the private index after a final seal")
    data_data_handoff.add_argument("--release", required=True, type=Path)
    data_data_handoff.add_argument("--receipt", required=True, type=Path)
    data_data_handoff.add_argument("--config", required=True, type=Path)
    data_data_handoff.add_argument("--output", required=True, type=Path)
    data_bundle = data_sub.add_parser(
        "qualification-bundle",
        help="restore one committed batch per source and build the real qualification trainer bundle",
    )
    data_bundle.add_argument("--release", required=True, type=Path)
    data_bundle.add_argument("--seal", required=True, type=Path)
    data_bundle.add_argument("--receipt", required=True, type=Path)
    data_bundle.add_argument("--selection", required=True, type=Path)
    data_bundle.add_argument("--wav-prefix", required=True)
    data_bundle.add_argument("--config", required=True, type=Path)
    data_bundle.add_argument("--output", required=True, type=Path)
    data_binding = data_sub.add_parser(
        "qualification-binding",
        help="bind the committed release and final restore proof for diagnostic GPU qualification",
    )
    data_binding.add_argument("--release", required=True, type=Path)
    data_binding.add_argument("--seal", required=True, type=Path)
    data_binding.add_argument("--receipt", required=True, type=Path)
    data_binding.add_argument("--bundle-manifest", required=True, type=Path)
    data_binding.add_argument("--wavlm-initializer", required=True, type=Path)
    data_binding.add_argument("--authorization-reference", required=True)
    data_binding.add_argument("--config", required=True, type=Path)
    data_binding.add_argument("--output", required=True, type=Path)
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
        code = _fail(error)
        if getattr(error, "unresolved", False) or error.code == "unresolved":
            return 2
        return code
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
    if command == "data":
        from .data import dispatch_data

        return dispatch_data(args)
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
        from .qualification import execute_qualification, load_qualification_spec, write_failed_qualification
        from .recovery import LocalTransport

        try:
            binding = parse_qualification_binding(json.loads(args.qualification_binding.read_text(encoding="utf-8")))
            lease = json.loads(args.qualification_lease.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LargeError(
                "qualification", "cannot load qualification binding or lease", {"error": str(error)}
            ) from error

        controller = Controller(
            ledger=BudgetLedger(DEFAULT_BUDGET),
            provider=FakeProvider(),
            transport=LocalTransport(),
        )
        qualification_control = controller.control_qualification(lease, binding)
        offer = lease.get("offer") if isinstance(lease, dict) else None
        if not isinstance(offer, dict) or offer.get("gpu_profile") != args.gpu_profile:
            raise LargeError(
                "qualification",
                "qualification lease GPU profile differs from the requested worker profile",
            )

        overrides = {
            "warmup_optimizer_updates": getattr(args, "warmup_optimizer_updates", None),
            "measured_optimizer_updates": getattr(args, "measured_optimizer_updates", None),
            "effective_batch": getattr(args, "effective_batch", None),
            "physical_batch_candidates": getattr(args, "physical_batch_candidates", None),
            "planned_updates_per_cycle": getattr(args, "planned_updates_per_cycle", None),
            "planned_max_cycles": getattr(args, "planned_max_cycles", None),
        }
        trainer_config = getattr(args, "trainer_config", None)
        if trainer_config is None or getattr(args, "output", None) is None:
            raise LargeError("usage", "qualify requires a trainer TOML and output path")
        try:
            qualification_spec = load_qualification_spec(getattr(args, "qualification_spec", None), overrides)
        except LargeError as error:
            return write_failed_qualification(
                trainer_config,
                args.gpu_profile,
                args.output,
                error,
                qualification_binding=binding,
                qualification_control=qualification_control,
            )
        return execute_qualification(
            trainer_config,
            args.gpu_profile,
            qualification_spec,
            args.output,
            qualification_binding=binding,
            qualification_control=qualification_control,
        )
    if command == "freeze-run":
        qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
        if (
            qualification.get("gpu_qualification_status") != "qualified"
            or not qualification.get("ok")
            or not qualification.get("qualification_binding_sha256")
            or qualification.get("training_ready") is not True
        ):
            raise LargeError("freeze-run", "cannot freeze a launch lock without training qualification")
        if qualification.get("qualification_only") is True:
            raise LargeError("freeze-run", "qualification-only artifact cannot create a training lock")
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
            if args.qualification_binding is None:
                raise LargeError("control", "qualification control requires --qualification-binding")
            binding = parse_qualification_binding(json.loads(args.qualification_binding.read_text(encoding="utf-8")))
            return controller.control_qualification(lease, binding)
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
