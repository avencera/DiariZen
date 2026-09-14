"""Versioned human-annotation export, QA measurement, and completed-review validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .acceptance import load_qa_policy
from .errors import ContractError
from .hashing import sha256_file, sha256_json
from .jsonio import atomic_write_text, read_json, write_json
from .review_models import (
    FRAME_SECONDS,
    GENESIS_HASH,
    ActivityInterval,
    Clear,
    ClockClear,
    ClockMeasurementEntered,
    GridProposal,
    NotReviewed,
    SignedOff,
    SpeakerRole,
    UnresolvedDefect,
    occupied_frames,
    seconds_from_frame,
)
from .review_overlay import load_review_session, validate_review_overlay
from .review_packet import validate_review_packet
from .review_store import ReviewEvent, ReviewStore, open_review_store


EXPORT_SCHEMA = "speakrs-open-yap-review-export"
EXPORT_SCHEMA_VERSION = 1
QA_SCHEMA = "speakrs-open-yap-qa-measurement"
QA_SCHEMA_VERSION = 1
EXPORT_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "packet_manifest_sha256",
        "overlay_sha256",
        "policy_sha256",
        "event_count",
        "event_head_hash",
        "exported_windows",
        "excluded_windows",
        "qa_report",
        "events_jsonl",
        "progress_report",
        "export_sha256",
    }
)


def _load_policy(session: Mapping[str, Any], packet_root: Path) -> dict[str, Any]:
    packet_manifest = read_json(packet_root / "manifest.json")
    policy_path = Path(str(packet_manifest["policy"]["global_policy_path"]))
    policy = load_qa_policy(policy_path)
    digest = sha256_file(policy_path)
    if digest != session.get("policy_sha256") or digest != packet_manifest["policy"]["global_policy_sha256"]:
        raise ContractError("QA policy identity does not match the frozen packet")
    return policy


def _defect_export(assessment: object) -> dict[str, object]:
    if isinstance(assessment, Clear):
        return {"status": "clear", "notes": None}
    if isinstance(assessment, UnresolvedDefect):
        return {"status": "unresolved", "notes": assessment.reason}
    if isinstance(assessment, NotReviewed):
        return {"status": "not_reviewed", "notes": None}
    raise ContractError("defect assessment cannot be exported")


def _clock_export(assessment: object) -> dict[str, object]:
    if isinstance(assessment, ClockClear):
        payload: dict[str, object] = {
            "status": "clear",
            "offset_seconds": None,
            "drift_seconds_per_second": None,
            "notes": None,
        }
        if isinstance(assessment.measurement, ClockMeasurementEntered):
            payload["offset_seconds"] = assessment.measurement.offset_seconds
            payload["drift_seconds_per_second"] = assessment.measurement.drift_seconds_per_second
        return payload
    raise ContractError("clock assessment cannot be exported")


def _activity_export(intervals: tuple[ActivityInterval, ...]) -> list[dict[str, object]]:
    return [
        {
            "speaker": interval.speaker.value,
            "start_seconds": seconds_from_frame(interval.start_frame),
            "end_seconds": seconds_from_frame(interval.end_frame),
        }
        for interval in intervals
    ]


def _load_proposal(overlay_root: Path, record: Mapping[str, Any]) -> GridProposal:
    path = overlay_root / str(record["proposal"]["path"])
    return GridProposal.from_dict(read_json(path))


def _window_exclusion(state, *, signed_activity_ok: bool) -> str | None:
    kind = state.decision.kind
    if kind == "pending":
        return "pending"
    if kind == "uncertain":
        return "uncertain"
    if kind == "needs_follow_up":
        return "needs_follow_up"
    if kind == "returned":
        return "returned"
    if kind != "signed_off":
        return "unsigned"
    if state.defects.has_unresolved():
        return "unresolved_defect"
    if not signed_activity_ok:
        return "unsigned"
    return None


def _speaker_frame_counts(
    human: tuple[ActivityInterval, ...],
    proposal: tuple[ActivityInterval, ...],
) -> dict[str, int]:
    human_frames = occupied_frames(human)
    proposal_frames = occupied_frames(proposal)
    human_set: set[tuple[str, int]] = set()
    proposal_set: set[tuple[str, int]] = set()
    for role in SpeakerRole:
        for frame in human_frames[role]:
            human_set.add((role.value, frame))
        for frame in proposal_frames[role]:
            proposal_set.add((role.value, frame))
    miss = human_set - proposal_set
    false_alarm = proposal_set - human_set
    return {
        "human_activity": len(human_set),
        "miss": len(miss),
        "false_alarm": len(false_alarm),
        "proposal_activity": len(proposal_set),
    }


def _ratio(miss: int, false_alarm: int, human_activity: int) -> dict[str, object]:
    if human_activity == 0:
        return {
            "status": "unavailable",
            "human_activity": 0,
            "miss": miss,
            "false_alarm": false_alarm,
            "error": None,
        }
    return {
        "status": "measured",
        "human_activity": human_activity,
        "miss": miss,
        "false_alarm": false_alarm,
        "error": (miss + false_alarm) / human_activity,
    }


def measure_qa(
    store: ReviewStore,
    *,
    policy: Mapping[str, Any],
    eligible: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Measure candidate miss plus false alarm against signed human activity."""

    overall = {"human_activity": 0, "miss": 0, "false_alarm": 0}
    strata = {stratum: {"human_activity": 0, "miss": 0, "false_alarm": 0} for stratum in policy["difficult_strata"]}
    window_rows = []
    overlay_root = store.overlay.root
    for record in store.overlay.manifest["windows"]:
        window_id = str(record["window_id"])
        if window_id not in eligible:
            continue
        proposal = _load_proposal(overlay_root, record)
        human = tuple(ActivityInterval.from_dict(item) for item in eligible[window_id]["intervals"])
        counts = _speaker_frame_counts(human, proposal.intervals)
        overall["human_activity"] += counts["human_activity"]
        overall["miss"] += counts["miss"]
        overall["false_alarm"] += counts["false_alarm"]
        stratum = record.get("stratum")
        if stratum in strata:
            strata[stratum]["human_activity"] += counts["human_activity"]
            strata[stratum]["miss"] += counts["miss"]
            strata[stratum]["false_alarm"] += counts["false_alarm"]
        window_rows.append({"window_id": window_id, "stratum": stratum, **counts})
    overall_ratio = _ratio(overall["miss"], overall["false_alarm"], overall["human_activity"])
    stratum_reports = {
        name: _ratio(values["miss"], values["false_alarm"], values["human_activity"])
        for name, values in strata.items()
    }
    required = list(policy["difficult_strata"])
    missing_strata = [name for name, report in stratum_reports.items() if report["status"] != "measured"]
    all_eligible = len(eligible) == len(store.overlay.window_ids())
    overall_ok = overall_ratio["status"] == "measured" and float(overall_ratio["error"]) <= float(
        policy["overall_miss_fa_limit"]
    )
    stratum_ok = all(
        report["status"] == "measured" and float(report["error"]) <= float(policy["stratum_miss_fa_limit"])
        for report in stratum_reports.values()
    )
    passed = all_eligible and not missing_strata and overall_ok and stratum_ok
    return {
        "schema": QA_SCHEMA,
        "schema_version": QA_SCHEMA_VERSION,
        "policy_id": policy["policy_id"],
        "policy_sha256": store.session["policy_sha256"],
        "frame_seconds": FRAME_SECONDS,
        "collar_seconds": policy["collar_seconds"],
        "speaker_time_denominator": policy["speaker_time_denominator"],
        "limits": {
            "overall_miss_fa_limit": policy["overall_miss_fa_limit"],
            "stratum_miss_fa_limit": policy["stratum_miss_fa_limit"],
        },
        "window_count": len(store.overlay.window_ids()),
        "eligible_window_count": len(eligible),
        "all_windows_eligible": all_eligible,
        "overall": overall_ratio,
        "strata": stratum_reports,
        "required_strata": required,
        "missing_stratum_measurements": missing_strata,
        "windows": window_rows,
        "pass": passed,
    }


def _human_annotation(
    *,
    packet_window: Mapping[str, Any],
    state,
    activity: tuple[ActivityInterval, ...],
    signer: str,
    review_event_hash: str,
    disposition: str,
) -> dict[str, object]:
    if disposition not in {"speech", "no_speech"}:
        raise ContractError("window disposition is invalid")
    return {
        "schema": "speakrs-open-yap-human-annotation-template",
        "schema_version": 1,
        "window_id": packet_window["window_id"],
        "parent_id": packet_window["parent_id"],
        "review_status": "signed_off",
        "reviewer_id": state.last_review_actor,
        "independent_signoff": {
            "signer_id": signer,
            "event_hash": review_event_hash,
            "status": "accepted",
        },
        "speaker_activity": _activity_export(activity),
        "window_disposition": disposition,
        "speaker_identity_defect": _defect_export(state.defects.identity),
        "clock_alignment": _clock_export(state.defects.clock),
        "synchronization_defect": _defect_export(state.defects.synchronization),
        "redaction_defect": _defect_export(state.defects.redaction),
        "notes": None,
    }


def _next_export_dir(session_root: Path) -> Path:
    root = session_root / "exports"
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("export-v"):
            suffix = path.name.removeprefix("export-v")
            if suffix.isdigit():
                existing.append(int(suffix))
    version = max(existing, default=0) + 1
    return root / f"export-v{version}"


def collect_eligibility(store: ReviewStore) -> tuple[dict[str, dict[str, Any]], list[dict[str, object]]]:
    """Return eligible signed windows and exclusion records."""

    eligible: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, object]] = []
    overlay_root = store.overlay.root
    for record in store.overlay.manifest["windows"]:
        window_id = str(record["window_id"])
        state = store.window_state(window_id)
        signed_ok = False
        activity: tuple[ActivityInterval, ...] = ()
        disposition = None
        if isinstance(state.decision, SignedOff) and not state.defects.has_unresolved():
            proposal = _load_proposal(overlay_root, record)
            try:
                activity = store.signed_activity(window_id, proposal)
                disposition = store.signed_disposition(window_id)
                signed_ok = True
            except ContractError:
                signed_ok = False
        reason = _window_exclusion(state, signed_activity_ok=signed_ok)
        if reason is not None:
            exclusions.append({"window_id": window_id, "reason": reason, "decision": state.decision.to_dict()})
            continue
        eligible[window_id] = {
            "state": state,
            "activity": activity,
            "intervals": [interval.to_dict() for interval in activity],
            "record": record,
            "disposition": disposition,
        }
    return eligible, exclusions


def export_review(store: ReviewStore) -> dict[str, object]:
    """Materialize a versioned export beside the session."""

    packet_root = store.packet_root
    validate_review_packet(packet_root)
    validate_review_overlay(store.overlay.root, packet_root=packet_root)
    policy = _load_policy(store.session, packet_root)
    packet_manifest = read_json(packet_root / "manifest.json")
    packet_windows = {item["window"]["window_id"]: item["window"] for item in packet_manifest["windows"]}
    eligible, exclusions = collect_eligibility(store)
    qa = measure_qa(store, policy=policy, eligible=eligible)
    export_dir = _next_export_dir(store.root)
    export_dir.mkdir(parents=False)
    for window_id, item in sorted(eligible.items()):
        state = item["state"]
        if not isinstance(state.decision, SignedOff):
            raise ContractError("eligible window lost its signed-off state")
        annotation = _human_annotation(
            packet_window=packet_windows[window_id],
            state=state,
            activity=tuple(ActivityInterval.from_dict(raw) for raw in item["intervals"]),
            signer=state.decision.signer,
            review_event_hash=state.decision.review_event_hash,
            disposition=str(item["disposition"]),
        )
        relative = Path("windows") / window_id / "human-annotation.json"
        path = export_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, annotation)
        item["relative"] = relative.as_posix()
    events_path = export_dir / "events.jsonl"
    lines = [json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")) for event in store.events()]
    atomic_write_text(events_path, "\n".join(lines) + ("\n" if lines else ""))
    progress_payload = {
        "progress": store.progress().as_dict(),
        "excluded": exclusions,
        "exported_windows": sorted(window_id for window_id in eligible),
    }
    write_json(export_dir / "progress.json", progress_payload)
    write_json(export_dir / "qa-report.json", qa)
    exported_files = {window_id: str(item["relative"]) for window_id, item in eligible.items()}
    manifest = {
        "schema": EXPORT_SCHEMA,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "packet_manifest_sha256": store.packet_hash,
        "overlay_sha256": store.overlay_hash,
        "policy_sha256": store.session["policy_sha256"],
        "event_count": len(store.events()),
        "event_head_hash": store.events()[-1].event_hash if store.events() else None,
        "exported_windows": exported_files,
        "excluded_windows": exclusions,
        "qa_report": "qa-report.json",
        "events_jsonl": "events.jsonl",
        "progress_report": "progress.json",
    }
    manifest["export_sha256"] = sha256_json({key: value for key, value in manifest.items() if key != "export_sha256"})
    write_json(export_dir / "manifest.json", manifest)
    return {
        "ok": True,
        "export_path": export_dir.as_posix(),
        "exported_window_count": len(exported_files),
        "excluded_window_count": len(exclusions),
        "qa_pass": qa["pass"],
        "export_sha256": manifest["export_sha256"],
    }


def require_export_manifest(export_root: Path) -> dict[str, Any]:
    """Load an export manifest and reject unknown fields."""

    manifest = read_json(export_root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != EXPORT_SCHEMA:
        raise ContractError("export manifest schema is invalid")
    unknown = sorted(set(manifest) - EXPORT_MANIFEST_FIELDS)
    if unknown:
        raise ContractError("export manifest has unknown fields", {"unknown": unknown})
    return manifest


def validate_completed_review(session_root: Path, export_path: Path | None = None) -> dict[str, object]:
    """Recompute identities, event chains, export files, and QA measurements."""

    loaded = load_review_session(session_root)
    packet_root = loaded["packet_root"]
    packet_validation = validate_review_packet(packet_root)
    overlay_validation = validate_review_overlay(loaded["overlay"].root, packet_root=packet_root)
    store = open_review_store(session_root)
    if store.quarantined:
        raise ContractError("event store has quarantined temporary files", {"files": store.quarantined})
    prior = GENESIS_HASH
    seen_request: set[str] = set()
    sequences: list[int] = []
    for event in store.events():
        if event.prior_hash != prior:
            raise ContractError("completed review has a broken event chain")
        if event.request_id in seen_request:
            raise ContractError("completed review has a duplicate request_id")
        if event.sequence in sequences:
            raise ContractError("completed review has a duplicate sequence")
        seen_request.add(event.request_id)
        sequences.append(event.sequence)
        ReviewEvent.from_dict(event.to_dict())
        prior = event.event_hash
    eligible, exclusions = collect_eligibility(store)
    policy = _load_policy(store.session, packet_root)
    qa = measure_qa(store, policy=policy, eligible=eligible)
    export_report = None
    if export_path is not None:
        export_root = Path(export_path).expanduser().resolve()
        export_manifest = require_export_manifest(export_root)
        if export_manifest.get("packet_manifest_sha256") != store.packet_hash:
            raise ContractError("export packet hash is stale")
        if export_manifest.get("overlay_sha256") != store.overlay_hash:
            raise ContractError("export overlay hash is stale")
        if export_manifest.get("event_count") != len(store.events()):
            raise ContractError("export event set is stale")
        exported = export_manifest.get("exported_windows")
        if not isinstance(exported, dict):
            raise ContractError("export window map is invalid")
        if set(exported) != set(eligible):
            raise ContractError("export membership does not match eligible signed windows")
        for window_id, relative in exported.items():
            path = (export_root / str(relative)).resolve()
            if export_root not in path.parents:
                raise ContractError("export path is unsafe")
            annotation = read_json(path)
            signer = (annotation.get("independent_signoff") or {}).get("signer_id")
            if annotation.get("reviewer_id") == signer:
                raise ContractError("exported annotation has same-actor sign-off")
            if annotation.get("window_disposition") not in {"speech", "no_speech"}:
                raise ContractError("exported annotation has an invalid window_disposition")
            if window_id not in eligible:
                raise ContractError("export contains an ineligible window")
        recomputed = sha256_json({key: value for key, value in export_manifest.items() if key != "export_sha256"})
        if recomputed != export_manifest.get("export_sha256"):
            raise ContractError("export identity hash does not match")
        export_report = {"path": export_root.as_posix(), "export_sha256": export_manifest["export_sha256"]}
    return {
        "ok": True,
        "packet": packet_validation,
        "overlay": overlay_validation,
        "event_count": len(store.events()),
        "eligible_window_count": len(eligible),
        "excluded_window_count": len(exclusions),
        "qa": qa,
        "export": export_report,
        "complete": bool(qa["pass"] and export_report is not None),
    }


__all__ = [
    "collect_eligibility",
    "export_review",
    "measure_qa",
    "validate_completed_review",
]
