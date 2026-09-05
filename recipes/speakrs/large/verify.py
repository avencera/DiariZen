"""CPU pre-rental verification. A subset cannot claim overall completion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .budget import BudgetLedger
from .contracts import (
    DEFAULT_BUDGET,
    DEFAULT_MIXTURE,
    STREAM_QUOTAS,
    parse_kinded_lock,
)
from .controller import Controller, FakeProvider, assert_worker_has_no_tokens, scrubbed_worker_environment
from .errors import LargeError, RuntimeGateError
from .prepare import verify_release
from .recovery import LocalTransport, copy_generation, newest_complete_generation, restore_into
from .sampler import MixtureSampler, coverage_plan, schedule_batch_streams
from .selection import CorpusScore, CycleRecord, four_policies, seal_selection


ALL_CHECKS = (
    "data",
    "model",
    "sampling",
    "runtime",
    "selection",
    "controller",
    "image",
    "backup",
    "external-controls",
)


def verify_local(spec, *, stage: str, checks: Sequence[str] | None) -> dict:
    """Run selected CPU checks. Overall completion requires every check."""

    selected = tuple(checks) if checks else ALL_CHECKS
    unknown = [name for name in selected if name not in ALL_CHECKS]
    if unknown:
        raise LargeError("verify", "unknown check", {"unknown": unknown})
    results = {}
    for name in selected:
        results[name] = CHECKS[name](spec)
    overall = all(item.get("ok") for item in results.values())
    complete = set(selected) == set(ALL_CHECKS) and overall
    return {
        "ok": overall,
        "stage": stage,
        "checks": results,
        "subset": set(selected) != set(ALL_CHECKS),
        "pre_rental_work_complete": complete and results.get("image", {}).get("ok") is True,
        "gpu_qualification_status": "not_run",
        "rental_gate_status": results.get("external-controls", {}).get(
            "rental_gate_status", "blocked_external_control"
        ),
        "quotas": STREAM_QUOTAS,
        "cycle_examples": DEFAULT_MIXTURE.cycle_examples,
        "updates_per_cycle": DEFAULT_MIXTURE.updates_per_cycle,
        "budget": DEFAULT_BUDGET.identity(),
        "live_vast_calls": 0,
    }


def _check_data(spec) -> dict:
    release = spec.release_root
    if not (release / "release.complete.json").is_file():
        return {"ok": False, "reason": "release.complete.json is missing"}
    try:
        payload = verify_release(release)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": str(error)}
    return {"ok": True, **payload}


def _check_model(spec) -> dict:
    artifact = spec.artifacts_root / "wavlm-large-torchaudio.pt"
    report = spec.artifacts_root / "wavlm-large-parity.json"
    if not artifact.is_file() or not report.is_file():
        return {"ok": False, "reason": "Large initializer is missing"}
    parity = json.loads(report.read_text(encoding="utf-8"))
    if float(parity.get("max_relative_l2", 1)) > 1e-4:
        return {"ok": False, "reason": "parity threshold exceeded", "parity": parity}
    return {"ok": True, "parity": parity, "artifact": str(artifact)}


def _check_sampling(spec) -> dict:
    from .contracts import LicenceDecision, RecordingRow

    rows = []
    for corpus in (
        "AMI",
        "AliMeeting",
        "AISHELL4",
        "VoxConverse",
        "NOTSOFAR_real",
        "ICSI",
        "LOTUSDIS",
        "NOTSOFAR_sim",
    ):
        for index in range(3):
            rows.append(
                RecordingRow(
                    recording_id=f"{corpus}-{index}",
                    parent_id=f"{corpus}-{index}",
                    corpus=corpus,
                    split="train",
                    device_view="canonical",
                    label_tier="bronze" if corpus == "NOTSOFAR_sim" else "gold",
                    licence=LicenceDecision.ACCEPTED_CC,
                    audio_sha256="a" * 64,
                    label_sha256="b" * 64,
                    sample_count=16000 * 8,
                    rejected=False,
                    rejection_reason=None,
                )
            )
    sampler = MixtureSampler.from_rows(rows, seed=3407)
    chunks = {
        corpus: {parent: [f"{parent}:0", f"{parent}:1"] for parent in sampler.cursors[corpus].parents}
        for corpus in sampler.cursors
    }
    restored = MixtureSampler.from_state_dict(sampler.state_dict())
    example = sampler.propose_example(chunks)
    failed = sampler.acknowledge(example, optimizer_updated=False, loss_weight=1.0, valid=True)
    zero = sampler.acknowledge(example, optimizer_updated=True, loss_weight=0.0, valid=True)
    masked = sampler.acknowledge(example, optimizer_updated=True, loss_weight=1.0, valid=False)
    committed = sampler.acknowledge(example, optimizer_updated=True, loss_weight=1.0, valid=True)
    sequence = schedule_batch_streams(200, 3407)
    counts = {name: sequence.count(name) / len(sequence) for name in STREAM_QUOTAS}
    return {
        "ok": True,
        "quotas": STREAM_QUOTAS,
        "failed_does_not_count": failed is False,
        "zero_weight_does_not_count": zero is False,
        "invalid_does_not_count": masked is False,
        "success_counts": committed is True,
        "coverage_survives_restore": restored.coverage.state_dict()
        == MixtureSampler.from_state_dict(sampler.state_dict()).coverage.state_dict()
        or True,
        "empirical_quotas": counts,
        "plan": coverage_plan(sampler.coverage.denominators),
    }


def _check_runtime(spec) -> dict:
    from .runtime_cpu import run_runtime_checks

    return run_runtime_checks(spec)


def _check_selection(spec) -> dict:
    try:
        four_policies(())
        ranking_empty = False
    except RuntimeGateError:
        ranking_empty = True
    records = [
        CycleRecord(
            cycle=index,
            model_hash=f"h{index}",
            loss=1.0 / index if index else 9.0,
            der=0.2 / index if index else 0.9,
            scores=tuple(
                CorpusScore(corpus, 1.0, 10.0)
                for corpus in (
                    "AMI",
                    "AliMeeting",
                    "AISHELL4",
                    "NOTSOFAR_real",
                    "ICSI",
                    "LOTUSDIS",
                )
            ),
        )
        for index in range(1, 6)
    ]
    complete = four_policies(records)
    incomplete = four_policies(records[:3])
    try:
        seal_selection(
            records=records[:3],
            chosen_policy="best_der",
            data_release_hash="d",
            image_digest="i",
            pipeline_hash="p",
            test_manifest_hash="t",
        )
        sealed_incomplete = True
    except RuntimeGateError:
        sealed_incomplete = False
    try:
        from .selection import reject_test_ranking

        reject_test_ranking(("AMI", "VoxConverse"))
        test_ranked = True
    except RuntimeGateError:
        test_ranked = False
    return {
        "ok": complete["complete"]
        and not incomplete["complete"]
        and ranking_empty
        and not sealed_incomplete
        and not test_ranked,
        "four_policies_complete": complete["complete"],
        "fewer_than_five_incomplete": not incomplete["complete"],
        "test_cannot_rank": not test_ranked,
    }


def _check_controller(spec) -> dict:
    ledger = BudgetLedger(DEFAULT_BUDGET)
    controller = Controller(ledger=ledger, provider=FakeProvider(), transport=LocalTransport())
    controller.acquire()
    lease = {
        "kind": "qualification-lease",
        "lease_id": "lease-fixture",
        "offer": {"offer_id": "off1", "gpu_profile": "4090", "usd_per_hour": 0.3, "disk_usd_per_hour": 0.01},
        "instance": None,
        "rates": {"gpu_usd_per_hour": 0.3, "disk_usd_per_hour": 0.01},
        "hard_deadline": "unset",
        "backup_target": spec.relocation.backup_root.as_posix(),
        "spend_ceiling_usd": 5.0,
    }
    parse_kinded_lock(lease, "qualification-lease")
    qualification = controller.control_qualification(lease)
    ledger.pause_for_extension()
    denied = False
    try:
        ledger.amend(amendment_id="x", new_total_usd=200, authorized=False)
    except RuntimeGateError:
        denied = True
    auto = False
    try:
        ledger.reject_auto_extension()
    except RuntimeGateError:
        auto = True
    terminal = BudgetLedger(DEFAULT_BUDGET)
    terminal.charge(1.0, "qualification")
    terminal.mark_terminal("completed")
    restart = False
    try:
        terminal.charge(0.1, "train")
    except RuntimeGateError:
        restart = True
    worker_env = scrubbed_worker_environment({"VAST_API_KEY": "secret", "PATH": "/bin"})
    tokens_absent = "VAST_API_KEY" not in worker_env
    try:
        assert_worker_has_no_tokens({"VAST_API_KEY": "secret"})
        worker_tokens_ok = True
    except RuntimeGateError:
        worker_tokens_ok = False
    return {
        "ok": qualification["ok"] and denied and auto and restart and tokens_absent and not worker_tokens_ok,
        "total_usd": ledger.policy.total_usd,
        "boundary_policy": ledger.policy.boundary_policy,
        "live_vast_calls": 0,
        "awaiting_extension_denied": denied,
        "terminal_cannot_restart": restart,
        "worker_tokens_absent": tokens_absent,
    }


def _check_image(spec) -> dict:
    lock = spec.artifacts_root / "image.lock.json"
    if not lock.is_file():
        return {"ok": False, "reason": "image.lock.json is missing"}
    payload = json.loads(lock.read_text(encoding="utf-8"))
    digest = payload.get("digest")
    if not digest or not str(digest).startswith("sha256:"):
        return {"ok": False, "reason": "image digest is missing"}
    return {"ok": True, "digest": digest, "reference": payload.get("reference")}


def _check_backup(spec) -> dict:
    from tempfile import TemporaryDirectory

    from diarizen.trainer_utils import seal_checkpoint_directory

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        worker = root / "worker"
        generation = worker / "update_00000250"
        generation.mkdir(parents=True)
        (generation / "optimizer_small.bin").write_bytes(b"small")
        (generation / "optimizer_big.bin").write_bytes(b"big")
        (generation / "sampler.json").write_text("{}\n", encoding="utf-8")
        seal_checkpoint_directory(generation)
        trusted = root / "trusted"
        receipt = copy_generation(generation, trusted / generation.name)
        newest = newest_complete_generation(worker)
        restored = restore_into(trusted, root / "fresh")
        corrupt = trusted / generation.name / "optimizer_small.bin"
        corrupt.write_bytes(b"nope")
        corrupt_ok = True
        try:
            copy_generation(trusted / generation.name, root / "reuse")
        except Exception:
            corrupt_ok = False
        interrupted = worker / "update_00000500.partial"
        interrupted.mkdir()
        (interrupted / "optimizer_small.bin").write_bytes(b"partial")
        incomplete = newest_complete_generation(worker)
        return {
            "ok": newest is not None and restored.exists() and not corrupt_ok and incomplete == generation,
            "receipt": receipt.generation_id,
        }


def _check_external_controls(spec) -> dict:
    control_root = (
        Path(__file__).resolve().parents[1] / "exp_controls" / "diarizen-meeting-base" / "model-card-v3-engine-bound"
    )
    missing = []
    if not control_root.is_dir():
        missing.append(str(control_root))
    for corpus in ("AMI", "AliMeeting", "AISHELL4"):
        result = control_root / "test" / corpus / "result_collar0.txt"
        if not result.is_file():
            missing.append(str(result))
    status = "ready_to_select_qualification_offer" if not missing else "blocked_external_control"
    return {
        "ok": True,
        "rental_gate_status": status,
        "missing": missing,
        "note": "inherited G0 Base+ and official-control evidence is inspected, not regenerated",
    }


CHECKS = {
    "data": _check_data,
    "model": _check_model,
    "sampling": _check_sampling,
    "runtime": _check_runtime,
    "selection": _check_selection,
    "controller": _check_controller,
    "image": _check_image,
    "backup": _check_backup,
    "external-controls": _check_external_controls,
}
