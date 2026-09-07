"""Data-command implementations. Owners stay in contracts, prepare, and storage."""

from __future__ import annotations

import importlib.util
import json
import secrets
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .acceptance import (
    QaPath,
    assert_policy_frozen_before_measurement,
    assert_split_isolation,
    audit_selection_hours_capacity,
    build_minimal_package,
    evaluate_existing_human_activity_labels,
    evaluate_human_reference_pilot,
    load_qa_policy,
    parse_rttm,
    verify_decoded_identity,
    verify_expected_rttm,
)
from .contracts import (
    DATA_PREPARATION_SCHEMA,
    BatchState,
    DataPreparationSpec,
    EvidenceReference,
    LocalCopyState,
    ObjectState,
    RemoteReleaseState,
    SelectionState,
    SourceMembership,
    VerifiedSelectionBatch,
    data_spec_to_json,
    parse_data_preparation_spec,
)
from .errors import ContractError, LargeError, PreparationError, UnresolvedInputError
from .hashing import sha256_bytes, sha256_file, sha256_json
from .jsonio import read_json, write_json
from .prepare import prepare_parent
from .release_closure import (
    CapacityClosure,
    aggregate_capacity,
    common_admitted_profiles,
    parse_capacity_manifest,
)
from .source_disposition import validate_source_dispositions
from .storage import (
    ConsumedSource,
    LocalCopy,
    RemoteRestoreProof,
    StorageBackend,
    WranglerR2Backend,
    assert_private_access,
    backend_from_destination,
    commit_batch,
    commit_release,
    directory_bytes,
    discard_consumed_source,
    enforce_cap,
    enforce_free_space_reserve,
    evict_copy,
    mark_eviction_eligible,
    mark_readback_verified,
    object_key,
    recover_deletion_journal,
    restore_object,
)
from .target_capacity import _load_powerset_class


FORBIDDEN_DATA_ACTIONS = frozenset({"train", "qualify", "rent", "accept-agreement", "submit-form"})


def load_data_spec(path: Path) -> DataPreparationSpec:
    """Load the data-only specification."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_data_preparation_spec(payload)


def _reject_forbidden_action(action: str | None) -> None:
    if action in FORBIDDEN_DATA_ACTIONS:
        raise ContractError("data commands cannot launch training, rent compute, or accept an agreement")


def _backend_from_spec(spec: DataPreparationSpec, override: StorageBackend | None = None) -> StorageBackend:
    if override is not None:
        return override
    backend = backend_from_destination(spec.r2)
    if isinstance(backend, WranglerR2Backend):
        backend.temporary_root = spec.disk.cache_root
    return backend


def _source_status(spec: DataPreparationSpec) -> list[dict[str, object]]:
    rows = []
    for source in spec.sources:
        permission = spec.permissions[source.permission_id]
        rows.append(
            {
                "name": source.name,
                "version": source.version,
                "membership": source.membership.value,
                "permission_state": source.permission_state.value,
                "missing_action": source.missing_action,
                "uses": {
                    name: {"decision": use.decision.value, "clause_ref": use.clause_ref}
                    for name, use in permission.uses.items()
                },
                "access_state": permission.access_state,
            }
        )
    return rows


def plan_data(spec: DataPreparationSpec, output: Path) -> dict[str, object]:
    """Validate membership, access, destination, and disk limits. Do not transfer."""

    probe_staging = directory_bytes(spec.disk.staging_root)
    probe_cache = directory_bytes(spec.disk.cache_root)
    enforce_cap(probe_staging, spec.disk.max_staging_bytes, "staging")
    enforce_cap(probe_cache, spec.disk.max_cache_bytes, "cache")
    pending = [row for row in _source_status(spec) if row["membership"] == SourceMembership.PENDING.value]
    payload = {
        "ok": True,
        "command": "plan",
        "release_id": spec.release_id,
        "schema": DATA_PREPARATION_SCHEMA,
        "source_membership": _source_status(spec),
        "access_states": {row["name"]: row["access_state"] for row in _source_status(spec)},
        "destination": {
            "provider": spec.r2.provider,
            "endpoint": spec.r2.endpoint,
            "bucket": spec.r2.bucket,
            "prefix": spec.r2.prefix,
            "credential_reference": spec.r2.credential_reference,
        },
        "disk_limits": {
            "max_staging_bytes": spec.disk.max_staging_bytes,
            "max_cache_bytes": spec.disk.max_cache_bytes,
            "free_space_reserve_bytes": spec.disk.free_space_reserve_bytes,
            "concurrency": spec.disk.concurrency,
            "staging_used_bytes": probe_staging,
            "cache_used_bytes": probe_cache,
        },
        "required_inputs": [row["missing_action"] for row in pending if row["missing_action"]],
        "pending_sources": [row["name"] for row in pending],
        "downloads": False,
        "bulk_transfer": False,
    }
    write_json(output, payload)
    return payload


def prepare_data(
    spec: DataPreparationSpec,
    output: Path,
    *,
    source: str,
    inventory: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """Process eligible sources with bounded staging. Never invent manual QA."""

    output.mkdir(parents=True, exist_ok=True)
    selected = [item for item in spec.sources if source == "all" or item.name == source]
    if not selected:
        raise PreparationError("unknown source", {"source": source})
    policy = load_qa_policy(spec.qa_policy_path)
    prepared = []
    pending = []
    for item in selected:
        if item.membership is SourceMembership.EXCLUDED:
            continue
        if not spec.permissions[item.permission_id].permitted_for_training_storage():
            pending.append({"name": item.name, "missing_action": item.missing_action, "kind": "permission"})
            continue
        input_path = spec.private_evidence_root / item.name / "selection-input.json"
        if not input_path.is_file():
            pending.append(
                {
                    "name": item.name,
                    "missing_action": "provide inspected source parent and evidence records",
                    "kind": "source-preparation",
                }
            )
            continue
        supplied = read_json(input_path)
        if supplied.get("source") != item.name or supplied.get("version") != item.version:
            raise PreparationError("source preparation input has the wrong source/version")
        batch_root = spec.disk.staging_root / "selections" / item.name / sha256_file(input_path)[:20]
        batch_root.mkdir(parents=True, exist_ok=True)
        recordings = []
        transforms = []
        for row in supplied.get("recordings") or []:
            source_audio, digest = _verified_reference(row["source_audio"], "source-audio")
            rttm_path, _ = _verified_reference(row["rttm"], "source-rttm")
            uem_path, _ = _verified_reference(row["uem"], "source-uem")
            intervals = parse_rttm(rttm_path.read_text(encoding="utf-8"))
            spans = _load_diarization_dataset().load_uem(str(uem_path))
            parent_id = str(row["recording_id"])
            facts = prepare_parent(
                source_audio,
                row.get("selected_channel"),
                digest,
                intervals,
                [{"start": start, "end": end} for start, end in spans.get(parent_id, ())],
                batch_root / f"{parent_id}.flac",
                spec.disk,
                parent_id=parent_id,
            )
            recording = facts["recording"]
            recording["known_speakers"] = list(row["known_speakers"])
            recordings.append(recording)
            transforms.append(facts["receipt"])
        if not recordings:
            raise PreparationError("source preparation cannot produce an empty batch")
        transform_path = batch_root / "time-transform.json"
        write_json(
            transform_path, {"schema": "speakrs-source-transforms-v1", "source": item.name, "parents": transforms}
        )
        manifest = {key: value for key, value in supplied.items() if key != "recordings"}
        manifest["schema"] = "speakrs-selection-v1"
        manifest["recordings"] = recordings
        manifest["evidence"]["time_transform"] = {"path": str(transform_path), "sha256": sha256_file(transform_path)}
        write_json(batch_root / "selection.json", manifest)
        prepared.append(
            {
                "name": item.name,
                "version": item.version,
                "state": SelectionState.CONTENT_VERIFIED.value,
                "release": str(batch_root),
                "parents": len(recordings),
                "accepted": False,
            }
        )
    payload = {
        "ok": True,
        "command": "prepare",
        "release_id": spec.release_id,
        "source": source,
        "qa_policy_id": policy["policy_id"],
        "qa_policy_sha256": sha256_json(policy),
        "prepared": prepared,
        "pending": pending,
        "inventory": list(inventory or []),
        "invented_manual_qa": False,
        "accepted": False,
    }
    write_json(output / "prepare.json", payload)
    write_json(output / "spec.resolved.json", data_spec_to_json(spec))
    return payload


def _implementation_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "recipes/speakrs/large/prepare.py",
        "recipes/speakrs/large/acceptance.py",
        "recipes/speakrs/large/target_capacity.py",
        "recipes/diar_ssl/dataset.py",
        "pyannote-audio/pyannote/audio/utils/powerset.py",
    )
    return {name: sha256_file(root / name) for name in paths}


def _closure_validator_sha256() -> str:
    """Hash the final-release closure validators independently of batch code."""

    return sha256_json(
        {
            "data.py": sha256_file(Path(__file__)),
            "release_closure.py": sha256_file(Path(__file__).with_name("release_closure.py")),
            "source_disposition.py": sha256_file(Path(__file__).with_name("source_disposition.py")),
        }
    )


def _selection_spec_digest(spec: DataPreparationSpec, source_name: str) -> str:
    source = next(item for item in spec.sources if item.name == source_name)
    serialized = data_spec_to_json(spec)
    permission = next(item for item in serialized["permissions"] if item["record_id"] == source.permission_id)
    return sha256_json(
        {
            "schema": "speakrs-selection-dependencies-v1",
            "release_id": spec.release_id,
            "source": source.name,
            "version": source.version,
            "permission": permission,
            "excluded": source.membership is SourceMembership.EXCLUDED,
            "frozen_splits": serialized["frozen_splits"].get(source_name),
            "profiles": serialized["profiles"],
            "implementation_hashes": _implementation_identity(),
            "destination": {key: value for key, value in serialized["r2"].items() if key != "credential_reference"},
        }
    )


def _verified_reference(payload: Mapping[str, Any], name: str) -> tuple[Path, str]:
    reference = EvidenceReference.parse(payload, name)
    if not reference.path.is_file() or sha256_file(reference.path) != reference.sha256:
        raise PreparationError("evidence is missing or changed", {"check": name})
    return reference.path, reference.sha256


def verify_data(
    spec: DataPreparationSpec,
    release: Path,
    output: Path | None,
) -> dict[str, object]:
    """Recompute one complete source selection and bind every acceptance dependency"""

    manifest_path = release / "selection.json"
    if not manifest_path.is_file():
        raise UnresolvedInputError("selection.json is required; source membership is not content proof")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "speakrs-selection-v1":
        raise ContractError("unknown source selection schema")
    source = next((item for item in spec.sources if item.name == manifest.get("source")), None)
    if source is None or source.membership is SourceMembership.EXCLUDED:
        raise PreparationError("selection source is absent or excluded")
    permission = spec.permissions[source.permission_id]
    if not permission.permitted_for_training_storage():
        raise UnresolvedInputError("selection permission is unresolved", {"source": source.name})
    if manifest.get("version") != source.version:
        raise PreparationError("selection source version differs from permission")
    policy = load_qa_policy(spec.qa_policy_path)
    evidence = manifest.get("evidence") or {}
    required_evidence = {
        "terms",
        "annotation_provenance",
        "channel_mapping",
        "time_transform",
        "coverage",
        "split_provenance",
    }
    if set(evidence) != required_evidence:
        raise PreparationError("selection evidence closure is incomplete", {"required": sorted(required_evidence)})
    evidence_hashes = {}
    for name, reference in evidence.items():
        _, evidence_hashes[name] = _verified_reference(reference, name)
    if evidence_hashes["terms"] != permission.terms_sha256:
        raise PreparationError("operative terms bytes do not match the permission record")
    qa = manifest.get("qa") or {}
    if qa.get("source_version") != source.version:
        raise PreparationError("QA evidence does not match the selected source version")
    assert_policy_frozen_before_measurement(policy, qa)
    if qa.get("qa_path") == QaPath.EXISTING_HUMAN_ACTIVITY:
        evaluate_existing_human_activity_labels(qa)
    else:
        evaluate_human_reference_pilot(qa, policy)
    splits = manifest.get("splits") or {}
    frozen = spec.frozen_splits.get(source.name)
    if frozen is None or {key: list(value) for key, value in frozen.items()} != splits:
        raise PreparationError("selection must preserve the exact frozen source split")
    speaker_graph = manifest.get("speaker_graph") or {}
    all_parents = {parent for ids in splits.values() for parent in ids}
    if set(speaker_graph) != all_parents or any(not speakers for speakers in speaker_graph.values()):
        raise PreparationError("source speaker graph must cover every frozen parent")
    assert_split_isolation({source.name: splits}, speaker_graph=speaker_graph)
    required_parents = set(splits.get("train") or ())
    declared = manifest.get("selected_parent_ids", list(splits.get("train") or ()))
    if not isinstance(declared, list) or len(set(declared)) != len(declared) or not set(declared) <= required_parents:
        raise PreparationError("batch parents must be a unique subset of the frozen train inventory")
    expected = set(declared)
    recordings = manifest.get("recordings") or []
    actual = [str(row.get("recording_id")) for row in recordings]
    if not expected or set(actual) != expected or len(actual) != len(expected):
        raise PreparationError("selected parent inventory differs from frozen training membership")
    artifact_hashes = {}
    intervals = []
    durations = {}
    uem_by_recording = {}
    for row in recordings:
        recording_id = str(row["recording_id"])
        audio_path, audio_hash = _verified_reference(row["audio"], "audio")
        rttm_path, rttm_hash = _verified_reference(row["rttm"], "rttm")
        if not row.get("uem"):
            raise PreparationError("every selected parent requires an explicit UEM object")
        uem_path, uem_hash = _verified_reference(row["uem"], "uem")
        regions = _load_diarization_dataset().load_uem(str(uem_path))
        if not regions or set(regions) != {recording_id}:
            raise PreparationError("UEM object must cover exactly the selected parent")
        uem_by_recording[recording_id] = regions[recording_id]
        identity = verify_decoded_identity(
            audio_path,
            expected_sha256=audio_hash,
            expected_sample_count=int(row["sample_count"]),
            expected_sample_rate=16000,
            expected_channels=1,
        )
        if any(end > identity.duration + 1e-6 for _, end in regions[recording_id]):
            raise PreparationError("UEM extends beyond decoded source audio")
        if audio_path.suffix.lower() != ".flac":
            raise PreparationError("minimal training audio must be the selected lossless FLAC")
        parsed = parse_rttm(rttm_path.read_text(encoding="utf-8"))
        if {item.recording_id for item in parsed} != {recording_id}:
            raise PreparationError("parent label object contains unexpected recordings")
        if not row.get("known_speakers") or set(row["known_speakers"]) != {item.speaker for item in parsed}:
            raise PreparationError("complete annotation speaker identity mapping is required")
        if not set(row["known_speakers"]) <= set(speaker_graph[recording_id]):
            raise PreparationError("annotation speaker identities differ from the frozen split speaker graph")
        verify_expected_rttm(
            parsed,
            duration=identity.duration,
            recording_id=recording_id,
            known_speakers=row.get("known_speakers"),
            redacted_regions=row.get("redacted_regions") or (),
        )
        for interval in parsed:
            covered = sum(
                max(0.0, min(interval.end, end) - max(interval.start, start)) for start, end in regions[recording_id]
            )
            if interval.end - interval.start - covered > 1e-6:
                raise PreparationError("valid activity falls outside the declared known UEM")
        intervals.extend(parsed)
        durations[recording_id] = identity.duration
        artifact_hashes[str(audio_path)] = audio_hash
        artifact_hashes[str(rttm_path)] = rttm_hash
        artifact_hashes[str(uem_path)] = uem_hash
    audit = audit_selection_hours_capacity(
        source=source.name,
        splits=splits,
        duration_by_recording=durations,
        intervals=intervals,
        uem_by_recording=uem_by_recording,
    )
    admitted = [item for item in audit["capacity"] if item["admitted"]]
    payload = {
        "schema": "speakrs-batch-verification-v2",
        "state": SelectionState.VERIFIED.value,
        "training_ready": False,
        "capacity_scope": "batch",
        "ok": True,
        "command": "verify",
        "source": source.name,
        "version": source.version,
        "manifest_sha256": sha256_file(manifest_path),
        "spec_sha256": _selection_spec_digest(spec, source.name),
        "selected_parent_ids": sorted(expected),
        "required_parent_ids": sorted(required_parents),
        "policy_sha256": sha256_json(policy),
        "implementation_hashes": _implementation_identity(),
        "evidence_hashes": evidence_hashes,
        "artifact_hashes": artifact_hashes,
        "provisional_profiles": admitted,
        "capacity": audit["capacity"],
        "hours": audit["hours"],
        "qa_path": qa["qa_path"],
        "complete_release": False,
    }
    VerifiedSelectionBatch.parse(payload)
    if output is not None:
        write_json(output, payload)
    return payload


def _load_verified_selection(
    spec: DataPreparationSpec, release: Path
) -> tuple[VerifiedSelectionBatch, dict[str, Any]]:
    receipt_path = release / "acceptance.json"
    if not receipt_path.is_file():
        raise PreparationError("upload requires the verified selection acceptance receipt")
    payload = read_json(receipt_path)
    receipt = VerifiedSelectionBatch.parse(payload)
    manifest = read_json(release / "selection.json")
    if receipt.spec_sha256 != _selection_spec_digest(spec, receipt.source) or receipt.manifest_sha256 != sha256_file(
        release / "selection.json"
    ):
        raise PreparationError("acceptance was invalidated by configuration or selection changes")
    if receipt.policy_sha256 != sha256_json(load_qa_policy(spec.qa_policy_path)):
        raise PreparationError("acceptance was invalidated by QA policy changes")
    for name, reference in manifest["evidence"].items():
        _, digest = _verified_reference(reference, name)
        if receipt.evidence_hashes.get(name) != digest:
            raise PreparationError("acceptance evidence changed")
    for path, digest in receipt.artifact_hashes.items():
        if not Path(path).is_file() or sha256_file(Path(path)) != digest:
            raise PreparationError("accepted training artifact is missing or changed")
    recomputed = verify_data(spec, release, None)
    if payload["schema"] == "speakrs-selection-acceptance-v1":
        # old immutable receipts required at least one batch-local profile and remain stronger storage proof
        recomputed["schema"] = payload["schema"]
        recomputed["state"] = SelectionState.ACCEPTED.value
        recomputed["admitted_profiles"] = recomputed.pop("provisional_profiles")
        recomputed.pop("training_ready")
        recomputed.pop("capacity_scope")
    if sha256_json(recomputed) != sha256_json(payload):
        raise PreparationError("acceptance receipt differs from recomputed evidence")
    return receipt, manifest


def _package_selection(
    spec: DataPreparationSpec, release: Path, acceptance: VerifiedSelectionBatch, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    artifacts = []
    parents = []
    for row in manifest["recordings"]:
        parent = {
            "recording_id": row["recording_id"],
            "sample_count": row["sample_count"],
            "known_speakers": row["known_speakers"],
            "uem_regions": _load_diarization_dataset().load_uem(row["uem"]["path"])[row["recording_id"]],
        }
        for field, purpose in (("audio", "train-audio"), ("rttm", "train-label"), ("uem", "train-label")):
            path = Path(row[field]["path"])
            digest = row[field]["sha256"]
            key = f"{spec.r2.prefix}/{object_key(acceptance.source, acceptance.version, 'train-' + acceptance.manifest_sha256[:20], digest, path.suffix)}"
            item = {
                "path": str(path),
                "sha256": digest,
                "key": key,
                "size": path.stat().st_size,
                "purpose": purpose,
                "source": acceptance.source,
                "version": acceptance.version,
                "parent_id": row["recording_id"],
                "codec": "flac" if field == "audio" else field,
            }
            artifacts.append(item)
            parent[field] = {name: item[name] for name in ("key", "sha256", "size", "codec")}
        parents.append(parent)
    source = next(item for item in spec.sources if item.name == acceptance.source)
    permission = next(
        item for item in data_spec_to_json(spec)["permissions"] if item["record_id"] == source.permission_id
    )
    accepted = read_json(release / "acceptance.json")

    # retain source and clock facts without making restored training depend on local paths
    def portable_facts(value):
        if isinstance(value, dict):
            return {
                key: portable_facts(item)
                for key, item in value.items()
                if key not in {"path", "source_audio", "rttm_path", "uem_path", "bounds"}
            }
        if isinstance(value, list):
            return [portable_facts(item) for item in value]
        return value

    provenance = {
        name: portable_facts(read_json(Path(reference["path"])))
        for name, reference in manifest["evidence"].items()
        if name != "terms"
    }
    portable = {
        "schema": "speakrs-portable-training-selection-v1",
        "source": acceptance.source,
        "version": acceptance.version,
        "release_id": spec.release_id,
        "acceptance_sha256": sha256_file(release / "acceptance.json"),
        "selection_sha256": acceptance.manifest_sha256,
        "selected_parent_ids": list(acceptance.selected_parent_ids),
        "required_parent_ids": list(acceptance.required_parent_ids),
        "configuration_sha256": acceptance.spec_sha256,
        "implementation_hashes": accepted["implementation_hashes"],
        "permission": permission,
        "qa": manifest["qa"],
        "qa_policy": load_qa_policy(spec.qa_policy_path),
        "evidence_hashes": acceptance.evidence_hashes,
        "provenance": provenance,
        "profiles": data_spec_to_json(spec)["profiles"],
        "splits": manifest["splits"],
        "speaker_graph": manifest["speaker_graph"],
        "hours": accepted["hours"],
        "capacity": accepted["capacity"],
        "recordings": parents,
        "admitted_profiles": accepted.get("provisional_profiles", accepted.get("admitted_profiles", [])),
        "complete_release": False,
    }
    if accepted["schema"] == "speakrs-batch-verification-v2":
        portable["admission_scope"] = "batch-provisional"
        portable["training_ready"] = False
    proof_path = release / "minimal-package.json"
    write_json(proof_path, portable)
    digest = sha256_file(proof_path)
    artifacts.append(
        {
            "path": str(proof_path),
            "sha256": digest,
            "key": f"{spec.r2.prefix}/{object_key(acceptance.source, acceptance.version, 'train-' + acceptance.manifest_sha256[:20], digest, '.json')}",
            "size": proof_path.stat().st_size,
            "purpose": "manifest",
            "source": acceptance.source,
            "version": acceptance.version,
            "parent_id": None,
            "codec": "json",
        }
    )
    return build_minimal_package(artifacts)


def upload_data(
    spec: DataPreparationSpec,
    release: Path,
    output: Path,
    *,
    artifacts: Sequence[Mapping[str, Any]] = (),
    backend: StorageBackend | None = None,
) -> dict[str, object]:
    """Upload the exact verified selection and its portable training proof"""

    acceptance, manifest = _load_verified_selection(spec, release)
    if artifacts:
        supplied = {
            str(Path(str(item["path"]))): sha256_file(Path(str(item["path"])))
            for item in build_minimal_package(artifacts)
        }
        if supplied != acceptance.artifact_hashes or len(artifacts) != len(supplied):
            raise PreparationError("upload inventory must exactly match the accepted selection")
    packaged = _package_selection(spec, release, acceptance, manifest)
    store = _backend_from_spec(spec, backend)
    result = {
        "ok": False,
        "command": "upload",
        "uploaded": [],
        "release": str(release),
        "acceptance_sha256": sha256_file(release / "acceptance.json"),
        "manifest_sha256": acceptance.manifest_sha256,
        "expected_objects": packaged,
    }
    write_json(output, result)
    for item in packaged:
        enforce_cap(directory_bytes(spec.disk.staging_root), spec.disk.max_staging_bytes, "staging")
        path = Path(item["path"])
        spec.disk.cache_root.mkdir(parents=True, exist_ok=True)
        enforce_cap(
            directory_bytes(spec.disk.cache_root),
            spec.disk.max_cache_bytes,
            "cache",
            additional_bytes=path.stat().st_size,
        )
        enforce_free_space_reserve(spec.disk.cache_root, spec.disk.free_space_reserve_bytes + path.stat().st_size)
        # backend operations must retain immutable content across interruption and retry
        store.put_bytes(item["key"], path.read_bytes())
        result["uploaded"].append({**item, "state": ObjectState.UPLOADED.value})
        write_json(output, result)
    result["ok"] = True
    write_json(output, result)
    return result


def verify_remote(
    spec: DataPreparationSpec,
    receipt_path: Path,
    output: Path,
    *,
    backend: StorageBackend | None = None,
    label_policy_id: str,
    split_id: str,
) -> dict[str, object]:
    """Full readback, inventory, privacy checks, and immutable batch commit."""

    store = _backend_from_spec(spec, backend)
    receipt = read_json(receipt_path)
    policy = load_qa_policy(spec.qa_policy_path)
    release = Path(str(receipt.get("release", "")))
    acceptance, manifest = _load_verified_selection(spec, release)
    expected = _package_selection(spec, release, acceptance, manifest)
    expected_by_key = {item["key"]: item for item in expected}
    uploaded = receipt.get("uploaded") or []
    if (
        not receipt.get("ok")
        or len(uploaded) != len(expected_by_key)
        or {item["key"] for item in uploaded} != set(expected_by_key)
    ):
        raise PreparationError("remote verification requires the complete independently accepted inventory")
    for item in uploaded:
        original = expected_by_key[item["key"]]
        if any(
            item.get(field) != original.get(field) for field in ("sha256", "size", "purpose", "source", "parent_id")
        ):
            raise PreparationError("uploaded inventory differs from accepted selection")
    if receipt.get("acceptance_sha256") != sha256_file(release / "acceptance.json"):
        raise PreparationError("upload receipt does not bind the current acceptance")
    verified = []
    for item in receipt.get("uploaded") or []:
        marked = mark_readback_verified(item, backend=store, expected_sha256=str(item["sha256"]))
        privacy = assert_private_access(store, marked["key"], spec.r2.prefix)
        marked["privacy"] = privacy
        verified.append(marked)
    batch = commit_batch(
        verified,
        state=BatchState.DRAFT,
        expected_objects=expected,
        acceptance_sha256=receipt["acceptance_sha256"],
        backend=store,
        inventory_prefix=expected[0]["key"].split("/objects/")[0],
        label_policy_id=label_policy_id,
        split_id=split_id,
        qa_policy_sha256=sha256_json(policy),
    )
    payload = {
        "ok": True,
        "command": "verify-remote",
        "batch": batch,
        "complete_release": False,
        "local_copies": [
            {key: item.get(key) for key in ("path", "key", "sha256", "size", "purpose", "source")} for item in expected
        ],
    }
    transform_reference = manifest["evidence"]["time_transform"]
    if read_json(Path(transform_reference["path"])).get("schema") == "speakrs-source-transforms-v1":
        payload["source_transform"] = transform_reference
    write_json(output, payload)
    return payload


def commit_remote_release(
    spec: DataPreparationSpec,
    release: Path,
    receipts_path: Path,
    output: Path,
    *,
    backend: StorageBackend | None = None,
) -> dict[str, object]:
    """Commit the final seal only when remote selections close the current spec."""

    from .contracts import require_content_hash
    from .storage import _object_identity, _read_marker

    store = _backend_from_spec(spec, backend)
    prefix = spec.r2.prefix
    serialized_spec = data_spec_to_json(spec)
    current_profiles = serialized_spec["profiles"]
    current_policy = load_qa_policy(spec.qa_policy_path)
    current_policy_sha256 = sha256_json(current_policy)
    current_implementation = _implementation_identity()
    current_closure_validator_sha256 = _closure_validator_sha256()
    source_dispositions = validate_source_dispositions(spec)

    source_by_name = {item.name: item for item in spec.sources}
    required_sources = []
    required_train_parents: dict[str, list[str]] = {}
    permission_by_source: dict[str, dict[str, object]] = {}
    permission_sha256: dict[str, str] = {}
    configuration_sha256_by_source: dict[str, str] = {}
    unresolved_permissions = []
    for source in spec.sources:
        if source.membership is SourceMembership.EXCLUDED:
            continue
        required_sources.append(source.name)
        splits = spec.frozen_splits.get(source.name)
        train = list((splits or {}).get("train") or ())
        if not splits or not train or len(set(train)) != len(train):
            raise PreparationError(
                "non-excluded source has no exact frozen train parent inventory",
                {"source": source.name},
            )
        required_train_parents[source.name] = sorted(str(parent) for parent in train)
        permission = spec.permissions[source.permission_id]
        permission_payload = next(
            item for item in serialized_spec["permissions"] if item["record_id"] == source.permission_id
        )
        permission_by_source[source.name] = permission_payload
        permission_sha256[source.name] = sha256_json(permission_payload)
        configuration_sha256_by_source[source.name] = _selection_spec_digest(spec, source.name)
        if source.permission_state.value != "permitted" or not permission.permitted_for_training_storage():
            unresolved_permissions.append(source.name)
    if unresolved_permissions:
        raise UnresolvedInputError(
            "source permission is unresolved for the complete release",
            {"sources": sorted(unresolved_permissions)},
        )

    def _receipt_batches(value: Any) -> list[Mapping[str, object]]:
        """Extract batch receipts from a versioned index or one verify receipt."""

        if isinstance(value, list):
            batches: list[Mapping[str, object]] = []
            for item in value:
                batches.extend(_receipt_batches(item))
            return batches
        if not isinstance(value, Mapping):
            raise PreparationError("remote batch receipts must be an object or array")
        if value.get("schema") == "speakrs-remote-batch-v1" and "batch_sha256" in value:
            return [value]
        if "batch" in value:
            return _receipt_batches(value["batch"])
        for name in ("batches", "receipts", "remote_receipts", "items"):
            if name in value:
                return _receipt_batches(value[name])
        raise PreparationError("remote batch receipts contain no committed batch")

    raw_receipts = read_json(receipts_path)
    batches = _receipt_batches(raw_receipts)
    if not batches:
        raise PreparationError("complete release requires at least one committed batch")

    coverage: set[tuple[str, str]] = set()
    union_parent_labels: set[tuple[str, str]] = set()
    required_batches: dict[str, str] = {}
    batch_marker_keys: list[str] = []
    capacity_by_batch: dict[str, tuple[str, CapacityClosure]] = {}

    def _manifest_object(
        marker_objects: list[Mapping[str, object]],
        batch_number: int,
    ) -> tuple[Mapping[str, object], dict[str, object], dict[str, Mapping[str, object]]]:
        identities: dict[str, Mapping[str, object]] = {}
        manifests: list[tuple[Mapping[str, object], dict[str, object]]] = []
        for object_number, item in enumerate(marker_objects):
            try:
                identity = _object_identity(item, f"batch[{batch_number}].objects[{object_number}]")
            except ContractError as error:
                raise PreparationError("committed batch contains an invalid object identity") from error
            key = str(identity["key"])
            if key in identities:
                raise PreparationError("committed batch contains duplicate object keys", {"key": key})
            identities[key] = item
            if item.get("purpose") == "manifest":
                manifests.append((item, identity))
        if len(manifests) != 1:
            raise PreparationError("committed batch requires exactly one portable training manifest")
        return manifests[0][0], manifests[0][1], identities

    def _portable_for_batch(
        batch: Mapping[str, object], marker_payload: Mapping[str, object], batch_number: int
    ) -> tuple[str, set[str], CapacityClosure]:
        marker_objects = marker_payload.get("objects")
        if not isinstance(marker_objects, list):
            raise PreparationError("committed batch marker is missing its object inventory")
        manifest_object, manifest_identity, marker_by_key = _manifest_object(marker_objects, batch_number)
        manifest_key = str(manifest_identity["key"])
        raw_manifest = store.get_bytes(manifest_key)
        if len(raw_manifest) != int(manifest_identity["size"]):
            raise PreparationError("portable manifest size differs from its committed identity")
        if sha256_bytes(raw_manifest) != str(manifest_identity["sha256"]):
            raise PreparationError("portable manifest bytes differ from its committed identity")
        try:
            portable = json.loads(raw_manifest)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PreparationError("portable training manifest is not valid JSON") from error
        if not isinstance(portable, Mapping) or portable.get("schema") != "speakrs-portable-training-selection-v1":
            raise PreparationError("unknown portable training manifest schema")
        if portable.get("complete_release") is True:
            raise PreparationError("a portable source manifest cannot claim complete release")
        batch_acceptance = require_content_hash(batch.get("acceptance_sha256"), "batch acceptance sha256")
        if portable.get("acceptance_sha256") != batch_acceptance:
            raise PreparationError("portable manifest acceptance differs from its committed batch")
        source_name = portable.get("source")
        if not isinstance(source_name, str) or source_name not in source_by_name:
            raise PreparationError("portable manifest names an unknown source")
        source = source_by_name[source_name]
        if source.membership is SourceMembership.EXCLUDED:
            raise PreparationError("portable manifest names an excluded source", {"source": source_name})
        if portable.get("version") != source.version or portable.get("release_id") != spec.release_id:
            raise PreparationError("portable manifest source version or release differs from the current spec")
        if portable.get("profiles") != current_profiles:
            raise PreparationError("portable manifest profiles differ from the current spec")
        if portable.get("implementation_hashes") != current_implementation:
            raise PreparationError("portable manifest implementation identity is stale")
        if portable.get("configuration_sha256") != configuration_sha256_by_source[source_name]:
            raise PreparationError("portable manifest configuration identity is stale")
        permission_payload = permission_by_source[source_name]
        if portable.get("permission") != permission_payload:
            raise PreparationError("portable manifest permission identity is stale", {"source": source_name})
        portable_policy = portable.get("qa_policy")
        if portable_policy != current_policy or sha256_json(portable_policy) != current_policy_sha256:
            raise PreparationError("portable manifest QA policy identity is stale")
        if batch.get("qa_policy_sha256") != current_policy_sha256:
            raise PreparationError("committed batch QA policy identity is stale")
        expected_splits = {split: list(ids) for split, ids in spec.frozen_splits[source_name].items()}
        portable_splits = portable.get("splits")
        if portable_splits != expected_splits:
            raise PreparationError("portable manifest does not preserve the exact frozen source split")
        split_members: list[str] = []
        for split, ids in expected_splits.items():
            if not isinstance(ids, list) or len(set(ids)) != len(ids):
                raise PreparationError("frozen source split contains duplicate parents", {"source": source_name})
            split_members.extend(str(parent) for parent in ids)
        if len(set(split_members)) != len(split_members):
            raise PreparationError("frozen source splits overlap", {"source": source_name})
        required_train = set(expected_splits.get("train") or ())
        required_parents = portable.get("required_parent_ids")
        if (
            not isinstance(required_parents, list)
            or len(required_parents) != len(set(required_parents))
            or set(required_parents) != required_train
        ):
            raise PreparationError("portable manifest required parents differ from frozen train parents")
        selected = portable.get("selected_parent_ids")
        if (
            not isinstance(selected, list)
            or not selected
            or len(selected) != len(set(selected))
            or not set(selected).issubset(required_train)
        ):
            raise PreparationError("portable manifest selected parents contain unknown or duplicate parents")
        recordings = portable.get("recordings")
        if not isinstance(recordings, list) or len(recordings) != len(selected):
            raise PreparationError("portable manifest recording inventory differs from selected parents")
        actual_recordings: list[str] = []
        referenced_keys: set[str] = set()
        for recording_number, row in enumerate(recordings):
            if not isinstance(row, Mapping) or not isinstance(row.get("recording_id"), str):
                raise PreparationError("portable manifest contains an invalid recording identity")
            recording_id = str(row["recording_id"])
            actual_recordings.append(recording_id)
            if recording_id not in set(selected) or actual_recordings.count(recording_id) != 1:
                raise PreparationError("portable manifest recording inventory contains a duplicate or unknown parent")
            for field, purpose in (("audio", "train-audio"), ("rttm", "train-label"), ("uem", "train-label")):
                reference = row.get(field)
                if not isinstance(reference, Mapping):
                    raise PreparationError("portable manifest recording is missing an object reference")
                try:
                    identity = _object_identity(reference, f"recordings[{recording_number}].{field}")
                except ContractError as error:
                    raise PreparationError("portable manifest contains an invalid object reference") from error
                key = str(identity["key"])
                if key in referenced_keys:
                    raise PreparationError("portable manifest references a duplicate object", {"key": key})
                referenced_keys.add(key)
                marker_item = marker_by_key.get(key)
                if marker_item is None:
                    raise PreparationError("portable manifest references an object outside its committed batch")
                marker_identity = _object_identity(marker_item, "committed batch object")
                if {key: identity[key] for key in ("key", "sha256", "size")} != {
                    key: marker_identity[key] for key in ("key", "sha256", "size")
                }:
                    raise PreparationError("portable manifest object identity differs from its batch marker")
                if (
                    marker_item.get("purpose") != purpose
                    or marker_item.get("parent_id") != recording_id
                    or marker_item.get("source") != source_name
                    or marker_item.get("version") != source.version
                ):
                    raise PreparationError("portable manifest object metadata differs from its batch marker")
        if set(actual_recordings) != set(selected):
            raise PreparationError("portable manifest recording inventory differs from selected parents")
        expected_object_keys = referenced_keys | {manifest_key}
        if set(marker_by_key) != expected_object_keys:
            raise PreparationError("committed batch contains objects outside its portable parent inventory")
        if (
            manifest_object.get("parent_id") is not None
            or manifest_object.get("source") != source_name
            or manifest_object.get("version") != source.version
        ):
            raise PreparationError("portable manifest marker metadata is inconsistent")
        if any(item.get("source") != source_name for item in marker_by_key.values()):
            raise PreparationError("committed batch mixes source object identities")
        if any(item.get("state") != ObjectState.READBACK_VERIFIED.value for item in marker_by_key.values()):
            raise PreparationError("committed batch includes an object without full readback verification")
        return (
            source_name,
            {str(item) for item in selected},
            parse_capacity_manifest(portable, label=f"batch[{batch_number}]"),
        )

    for batch_number, batch in enumerate(batches):
        if batch.get("schema") != "speakrs-remote-batch-v1" or batch.get("state") != BatchState.COMMITTED.value:
            raise PreparationError("release receipts must contain committed remote batches")
        marker_payload = _read_marker(store, batch.get("marker") if isinstance(batch.get("marker"), Mapping) else {})
        marker_ref = batch["marker"]
        if not isinstance(marker_ref, Mapping):
            raise PreparationError("committed batch is missing its immutable marker")
        marker_key = marker_ref.get("key")
        if not isinstance(marker_key, str) or not marker_key.startswith(prefix + "/"):
            raise PreparationError("committed batch marker is outside the authorized task prefix")
        assert_private_access(store, marker_key, prefix)
        batch_hash = require_content_hash(batch.get("batch_sha256"), "batch sha256")
        acceptance_hash = require_content_hash(batch.get("acceptance_sha256"), "batch acceptance sha256")
        if batch_hash in required_batches:
            raise PreparationError("release contains a duplicate committed batch")
        if (
            marker_payload.get("batch_sha256") != batch_hash
            or marker_payload.get("acceptance_sha256") != acceptance_hash
        ):
            raise PreparationError("committed batch marker identity differs from its receipt")
        source_name, selected_parents, capacity = _portable_for_batch(batch, marker_payload, batch_number)
        for parent in selected_parents:
            pair = (source_name, parent)
            if pair in coverage:
                raise PreparationError(
                    "release contains a duplicate selected parent", {"source": source_name, "parent_id": parent}
                )
            coverage.add(pair)
            union_parent_labels.add(pair)
        required_batches[batch_hash] = acceptance_hash
        batch_marker_keys.append(marker_key)
        capacity_by_batch[batch_hash] = (source_name, capacity)

    expected_coverage = {
        (source_name, parent) for source_name, parents in required_train_parents.items() for parent in parents
    }
    if coverage != expected_coverage:
        raise UnresolvedInputError(
            "committed batches do not cover the exact frozen train parent union",
            {
                "missing": sorted(expected_coverage - coverage),
                "extra": sorted(coverage - expected_coverage),
            },
        )
    source_capacity_closures: dict[str, CapacityClosure] = {}
    for source_name in required_sources:
        records = [
            capacity
            for batch_hash, (batch_source, capacity) in sorted(capacity_by_batch.items())
            if batch_source == source_name
        ]
        source_capacity_closures[source_name] = aggregate_capacity(records, label=f"source[{source_name}]")
    source_capacity = {source_name: closure.as_list() for source_name, closure in source_capacity_closures.items()}
    source_admitted_profiles = {
        source_name: [item for item in source_capacity[source_name] if item["admitted"]]
        for source_name in required_sources
    }
    common_admitted = common_admitted_profiles(source_capacity_closures)
    configuration_sha256 = sha256_json(
        {source_name: configuration_sha256_by_source[source_name] for source_name in sorted(required_sources)}
    )
    parent_union_sha256 = sha256_json(
        [f"{source_name}:{parent}" for source_name, parent in sorted(union_parent_labels)]
    )
    release_identity = {
        "release_id": spec.release_id,
        "configuration_sha256": configuration_sha256,
        "implementation_hashes": current_implementation,
        "closure_validator_sha256": current_closure_validator_sha256,
        "source_dispositions": source_dispositions,
        "qa_policy_sha256": current_policy_sha256,
        "required_sources": sorted(required_sources),
        "required_train_parents": {
            source_name: required_train_parents[source_name] for source_name in sorted(required_sources)
        },
        "parent_union_sha256": parent_union_sha256,
        "source_permission_sha256": {
            source_name: permission_sha256[source_name] for source_name in sorted(required_sources)
        },
        "source_capacity": {source_name: source_capacity[source_name] for source_name in sorted(required_sources)},
        "admitted_profiles": {
            source_name: source_admitted_profiles[source_name] for source_name in sorted(required_sources)
        },
        "common_admitted_profiles": common_admitted,
        "required_batches": dict(sorted(required_batches.items())),
    }
    sealed = commit_release(
        batches,
        required_sources=sorted(required_sources),
        state=RemoteReleaseState.DRAFT,
        backend=store,
        inventory_prefix=prefix,
        release_identity=release_identity,
        allowed_inventory_keys=batch_marker_keys,
    )
    payload = {
        "ok": True,
        "command": "commit-release",
        "release": str(release),
        "seal": sealed,
        "admitted_profiles": common_admitted,
        "training_ready": bool(common_admitted),
        "complete_release": True,
    }
    write_json(output, payload)
    return payload


def _load_diarization_dataset():
    dataset_path = Path(__file__).resolve().parents[1].parent / "diar_ssl" / "dataset.py"
    spec = importlib.util.spec_from_file_location("diar_ssl_dataset_restore", dataset_path)
    if spec is None or spec.loader is None:
        raise PreparationError("cannot load the production data reader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_remote_batch(spec: DataPreparationSpec, receipt: Mapping[str, Any], store: StorageBackend):
    batch = receipt.get("batch") or {}
    marker = batch.get("marker") or {}
    if not str(marker.get("key", "")).startswith(spec.r2.prefix + "/"):
        raise PreparationError("restore marker is outside the authorized task prefix")
    raw = store.get_bytes(marker["key"])
    if len(raw) != marker.get("size") or sha256_bytes(raw) != marker.get("sha256"):
        raise PreparationError("remote batch marker failed full content verification")
    committed = json.loads(raw)
    if committed.get("state") != BatchState.COMMITTED.value or committed.get("batch_sha256") != batch.get(
        "batch_sha256"
    ):
        raise PreparationError("remote marker does not commit this batch")
    return committed, marker


_RESTORE_ATTEMPT_SCHEMA = "speakrs-cold-restore-attempt-v1"
_RESTORE_ATTEMPT_STATE = "in-progress"


def _restore_attempt_path(output: Path) -> Path:
    """Return the durable sidecar that owns one cold-restore attempt."""

    return output.with_name(f"{output.stem}.attempt{output.suffix}")


def _restore_cache_path(
    spec: DataPreparationSpec,
    output: Path,
    batch_sha256: str,
    incarnation_id: str,
) -> Path:
    """Return the stable task-owned cache path for one restore incarnation."""

    attempt_id = sha256_json(
        {
            "output": output.expanduser().resolve(strict=False).as_posix(),
            "batch_sha256": batch_sha256,
            "incarnation_id": incarnation_id,
        }
    )[:32]
    return spec.disk.cache_root / f"cold-{attempt_id}"


def _restore_object_records(objects: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    """Return the exact committed object identities used by a restore attempt."""

    from .storage import _object_identity

    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(objects):
        try:
            identity = _object_identity(item, f"restore object[{index}]")
        except ContractError as error:
            raise PreparationError("remote batch contains an invalid restore object") from error
        key = str(identity["key"])
        if key in seen:
            raise PreparationError("remote batch contains duplicate restore objects", {"key": key})
        seen.add(key)
        records.append(dict(identity))
    return sorted(records, key=lambda item: str(item["key"]))


def _restore_attempt_identity(
    spec: DataPreparationSpec,
    committed: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> dict[str, object]:
    """Build the batch, source, code, and object identity for a resumable restore."""

    from .contracts import require_content_hash

    objects = committed.get("objects")
    if not isinstance(objects, list) or not objects:
        raise PreparationError("remote batch contains no restore objects")
    object_records = _restore_object_records(objects)
    manifest_records = [item for item in object_records if item.get("purpose") == "manifest"]
    if len(manifest_records) != 1:
        raise PreparationError("batch requires exactly one portable training manifest")
    sources = {item.get("source") for item in object_records}
    versions = {item.get("version") for item in object_records}
    if len(sources) != 1 or len(versions) != 1:
        raise PreparationError("remote batch mixes source identities")
    source_name = next(iter(sources))
    version = next(iter(versions))
    if not isinstance(source_name, str) or not source_name:
        raise PreparationError("remote batch is missing its source identity")
    if not isinstance(version, str) or not version:
        raise PreparationError("remote batch is missing its source version")
    source = next((item for item in spec.sources if item.name == source_name), None)
    if source is None or source.membership is SourceMembership.EXCLUDED or source.version != version:
        raise PreparationError("remote batch source identity differs from the current spec")
    batch_sha256 = committed.get("batch_sha256")
    acceptance_sha256 = committed.get("acceptance_sha256")
    marker_identity = {key: marker.get(key) for key in ("key", "sha256", "size")}
    if not isinstance(batch_sha256, str) or not isinstance(acceptance_sha256, str):
        raise PreparationError("remote batch is missing its content identities")
    try:
        batch_sha256 = require_content_hash(batch_sha256, "restore batch sha256")
        acceptance_sha256 = require_content_hash(acceptance_sha256, "restore acceptance sha256")
        require_content_hash(marker_identity["sha256"], "restore marker sha256")
    except ContractError as error:
        raise PreparationError("remote batch has invalid restore identities") from error
    marker_key = marker_identity["key"]
    marker_size = marker_identity["size"]
    if (
        not isinstance(marker_key, str)
        or not marker_key
        or isinstance(marker_size, bool)
        or not isinstance(marker_size, int)
        or marker_size <= 0
    ):
        raise PreparationError("remote batch marker identity is invalid")
    return {
        "batch_sha256": batch_sha256,
        "acceptance_sha256": acceptance_sha256,
        "marker": marker_identity,
        "objects": object_records,
        "source": source_name,
        "version": version,
        "release_id": spec.release_id,
        "configuration_sha256": _selection_spec_digest(spec, source_name),
        "profiles": data_spec_to_json(spec)["profiles"],
        "implementation_hashes": _implementation_identity(),
        "closure_validator_sha256": _closure_validator_sha256(),
    }


def _restore_attempt_cache(
    spec: DataPreparationSpec,
    output: Path,
    identity: Mapping[str, object],
) -> tuple[Path, Path, dict[str, object]]:
    """Load or create the durable restore-attempt owner before any transfer."""

    from .contracts import require_content_hash

    attempt_path = _restore_attempt_path(output)
    if attempt_path.exists():
        if attempt_path.is_symlink() or not attempt_path.is_file():
            raise PreparationError("restore attempt owner is not one regular file")
        try:
            attempt = read_json(attempt_path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PreparationError("restore attempt owner is not valid JSON") from error
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("schema") != _RESTORE_ATTEMPT_SCHEMA
            or attempt.get("state") != _RESTORE_ATTEMPT_STATE
            or attempt.get("output_path") != output.expanduser().resolve(strict=False).as_posix()
            or attempt.get("identity") != dict(identity)
        ):
            raise PreparationError("restore attempt identity differs from the current batch")
        completed = attempt.get("completed_objects")
        expected_keys = (
            {str(item["key"]) for item in identity["objects"]} if isinstance(identity.get("objects"), list) else set()
        )
        if not isinstance(completed, list) or any(not isinstance(item, str) for item in completed):
            raise PreparationError("restore attempt progress is invalid")
        if len(completed) != len(set(completed)) or any(item not in expected_keys for item in completed):
            raise PreparationError("restore attempt progress is invalid")
        try:
            incarnation_id = require_content_hash(attempt.get("incarnation_id"), "restore incarnation id")
        except ContractError as error:
            raise PreparationError("restore attempt incarnation identity is invalid") from error
        expected_cache = _restore_cache_path(spec, output, str(identity["batch_sha256"]), incarnation_id)
        cache_value = attempt.get("cache_path")
        if not isinstance(cache_value, str) or Path(cache_value).expanduser().resolve(
            strict=False
        ) != expected_cache.expanduser().resolve(strict=False):
            raise PreparationError("restore attempt cache path differs from its owner")
        cache = expected_cache
        if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
            raise PreparationError("restore attempt cache path is not one task-owned directory")
        cache.mkdir(parents=True, exist_ok=True)
        return attempt_path, cache, dict(attempt)

    incarnation_id = secrets.token_hex(32)
    expected_cache = _restore_cache_path(spec, output, str(identity["batch_sha256"]), incarnation_id)
    cache = expected_cache
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise PreparationError("restore attempt cache path is not one task-owned directory")
    cache.mkdir(parents=True, exist_ok=True)
    attempt = {
        "schema": _RESTORE_ATTEMPT_SCHEMA,
        "state": _RESTORE_ATTEMPT_STATE,
        "incarnation_id": incarnation_id,
        "output_path": output.expanduser().resolve(strict=False).as_posix(),
        "cache_path": cache.expanduser().resolve(strict=False).as_posix(),
        "identity": dict(identity),
        "completed_objects": [],
    }
    write_json(attempt_path, attempt)
    return attempt_path, cache, attempt


def _validate_restore_cache_file(path: Path, expected: Mapping[str, object]) -> None:
    """Recheck an existing attempt file before it is used as a restored object."""

    if path.is_symlink() or not path.is_file():
        raise PreparationError("restore attempt cache object is not one regular file", {"path": str(path)})
    expected_size = int(expected["size"])
    expected_sha256 = str(expected["sha256"])
    if path.stat().st_size != expected_size or sha256_file(path) != expected_sha256:
        raise PreparationError("restore attempt cache object changed", {"path": str(path)})


def _restore_destination(cache: Path, key: str) -> Path:
    """Map one committed object key to its task-owned cache filename."""

    return cache / Path(key).name


def _write_restore_attempt_progress(
    attempt_path: Path,
    attempt: Mapping[str, object],
    completed: Sequence[str],
) -> dict[str, object]:
    """Persist object ownership progress without publishing a success receipt."""

    updated = {**attempt, "completed_objects": sorted(set(completed))}
    write_json(attempt_path, updated)
    return updated


def _archive_restore_receipt(output: Path) -> Path | None:
    """Preserve a completed receipt before replacing its restore incarnation."""

    if not output.exists():
        return None
    if output.is_symlink() or not output.is_file():
        raise PreparationError("restore output is not one regular file")
    digest = sha256_file(output)
    archived = output.with_name(f"{output.stem}-{digest}{output.suffix}")
    if archived.exists():
        if archived.is_symlink() or not archived.is_file() or sha256_file(archived) != digest:
            raise PreparationError("restore receipt archive has a conflicting identity")
        output.unlink()
        return archived
    output.replace(archived)
    return archived


def restore_check(
    spec: DataPreparationSpec,
    receipt_path: Path,
    output: Path,
    *,
    backend: StorageBackend | None = None,
    samples: Sequence[Mapping[str, Any]] = (),
) -> dict[str, object]:
    """Cold restore portable objects and run the actual CPU reader, slots, and targets"""

    if samples:
        raise PreparationError("restore samples come from the committed portable manifest, not caller source paths")
    store = _backend_from_spec(spec, backend)
    receipt = read_json(receipt_path)
    committed, marker = _load_remote_batch(spec, receipt, store)
    objects = committed.get("objects")
    if not isinstance(objects, list) or not objects:
        raise PreparationError("remote batch contains no restore objects")
    manifests = [item for item in objects if isinstance(item, Mapping) and item.get("purpose") == "manifest"]
    if len(manifests) != 1:
        raise PreparationError("batch requires exactly one portable training manifest")
    spec.disk.cache_root.mkdir(parents=True, exist_ok=True)
    identity = _restore_attempt_identity(spec, committed, marker)
    attempt_path, cache, attempt = _restore_attempt_cache(spec, output, identity)
    incarnation_id = str(attempt["incarnation_id"])
    _archive_restore_receipt(output)
    restored: dict[str, Path] = {}
    identities: list[dict[str, object]] = []
    destinations = [_restore_destination(cache, str(item["key"])) for item in objects]
    if len(destinations) != len(set(destinations)):
        raise PreparationError("remote batch objects map to duplicate cache paths")
    completed = [str(item) for item in attempt.get("completed_objects", []) if isinstance(item, str)]
    missing_sizes = []
    for item in objects:
        key = str(item["key"])
        path = _restore_destination(cache, key)
        partial = path.with_name(path.name + ".partial")
        if partial.is_symlink() or (partial.exists() and not partial.is_file()):
            raise PreparationError("restore attempt partial object is not one regular file", {"path": str(partial)})
        partial.unlink(missing_ok=True)
        if path.exists():
            _validate_restore_cache_file(path, item)
            restored[key] = path
            completed.append(key)
        else:
            missing_sizes.append(int(item["size"]))
    remaining = sum(missing_sizes)
    transfer_peak = max(missing_sizes, default=0)
    enforce_cap(
        directory_bytes(spec.disk.cache_root),
        spec.disk.max_cache_bytes,
        "cache",
        additional_bytes=remaining + transfer_peak + 1024 * 1024,
    )
    enforce_free_space_reserve(spec.disk.cache_root, spec.disk.free_space_reserve_bytes + remaining + transfer_peak)
    attempt = _write_restore_attempt_progress(attempt_path, attempt, completed)
    for item in objects:
        key = str(item["key"])
        path = _restore_destination(cache, key)
        if key not in restored:
            path = restore_object(
                store,
                {**item, "marker": marker},
                path,
                max_bytes=min(spec.disk.max_cache_bytes, int(item["size"])),
            )
            _validate_restore_cache_file(path, item)
            restored[key] = path
            completed.append(key)
            attempt = _write_restore_attempt_progress(attempt_path, attempt, completed)
        identities.append(
            {"key": key, "sha256": sha256_file(path), "size": path.stat().st_size, "restored_path": str(path)}
        )
    portable = read_json(restored[manifests[0]["key"]])
    if (
        not isinstance(portable, Mapping)
        or portable.get("schema") != "speakrs-portable-training-selection-v1"
        or portable.get("acceptance_sha256") != committed["acceptance_sha256"]
    ):
        raise PreparationError("portable manifest differs from the committed acceptance")
    if portable.get("implementation_hashes") != _implementation_identity():
        raise PreparationError("restored input implementation differs from its accepted reader/encoder")
    if (
        portable.get("release_id") != identity["release_id"]
        or portable.get("source") != identity["source"]
        or portable.get("version") != identity["version"]
        or portable.get("profiles") != data_spec_to_json(spec)["profiles"]
        or portable.get("configuration_sha256") != _selection_spec_digest(spec, portable["source"])
    ):
        raise PreparationError("restored input differs from its accepted source/profile configuration")
    actual_keys = {item["key"] for item in objects if item["purpose"] != "manifest"}
    expected_keys = {row[field]["key"] for row in portable["recordings"] for field in ("audio", "rttm", "uem")}
    if actual_keys != expected_keys:
        raise PreparationError("portable parent inventory differs from committed objects")
    module = _load_diarization_dataset()
    powerset_class = _load_powerset_class()
    results = []
    for row in portable["recordings"]:
        rec = row["recording_id"]
        audio_path, label_path, uem_path = (restored[row[field]["key"]] for field in ("audio", "rttm", "uem"))
        identity = verify_decoded_identity(
            audio_path, expected_sha256=row["audio"]["sha256"], expected_sample_count=row["sample_count"]
        )
        intervals = parse_rttm(label_path.read_text(encoding="utf-8"))
        verify_expected_rttm(
            intervals, duration=identity.duration, recording_id=rec, known_speakers=row["known_speakers"]
        )
        scp = cache / f"{rec}.scp"
        scp.write_text(f"{rec} {audio_path}\n", encoding="utf-8")
        for chunk_seconds, shift, frames in ((8, 6, 399), (16, 12, 799)):
            dataset = module.DiarizationDataset(
                str(scp),
                str(label_path),
                str(uem_path),
                model_num_frames=frames,
                model_rf_duration=spec.profiles.rf_duration,
                model_rf_step=spec.profiles.rf_step,
                chunk_size=chunk_seconds,
                chunk_shift=shift,
                sample_rate=spec.profiles.sample_rate,
            )
            # scan real targets so overlap and both boundary windows are included without using a source path
            overlap_index = max(
                range(len(dataset)), key=lambda index: int(dataset[index][1].sum(axis=1).max(initial=0))
            )
            for index in sorted({0, len(dataset) - 1, overlap_index}):
                waveform, target, name = dataset[index]
                collated = module._collate_fn([(waveform, target, name)], max_speakers_per_chunk=4)
                if tuple(collated["xs"].shape) != (1, 1, chunk_seconds * 16000) or tuple(collated["ts"].shape) != (
                    1,
                    frames,
                    4,
                ):
                    raise PreparationError("restored production input has an unexpected tensor shape")
                if not torch.isfinite(collated["xs"]).all() or not torch.isfinite(collated["ts"]).all():
                    raise PreparationError("restored production input has non-finite values")
                encodings = []
                for overlap in (2, 4):
                    encoder = powerset_class(4, overlap)
                    encoded = encoder.to_powerset(collated["ts"].float())
                    decoded = encoder.to_multilabel(encoded)
                    if not torch.isfinite(encoded).all() or tuple(decoded.shape) != (1, frames, 4):
                        raise PreparationError("restored target encoder produced invalid values")
                    encodings.append({"max_overlap": overlap, "powerset_shape": list(encoded.shape)})
                results.append(
                    {
                        "recording_id": rec,
                        "chunk_seconds": chunk_seconds,
                        "chunk_shift": shift,
                        "chunk_index": index,
                        "waveform_shape": list(collated["xs"].shape),
                        "target_shape": list(collated["ts"].shape),
                        "finite": True,
                        "uem_regions": module.load_uem(str(uem_path))[rec],
                        "encodings": encodings,
                        "source_path_used": str(audio_path),
                        "speaker_mask": (collated["ts"].sum(dim=1) > 0).tolist(),
                    }
                )
    payload = {
        "schema": "speakrs-cold-restore-v1",
        "ok": True,
        "command": "restore-check",
        "incarnation_id": incarnation_id,
        "marker": marker,
        "batch_sha256": committed["batch_sha256"],
        "acceptance_sha256": committed["acceptance_sha256"],
        "objects": identities,
        "samples": results,
        "cache_root": str(cache),
        "local_copies": receipt.get("local_copies") or [],
        "source_transform": receipt.get("source_transform"),
        "complete_release": False,
    }
    if not results:
        raise PreparationError("cold restore produced no real CPU input checks")
    write_json(output, payload)
    attempt_path.unlink(missing_ok=True)
    return payload


def discard_data_sources(
    spec: DataPreparationSpec,
    receipt_path: Path,
    transform_path: Path,
    output: Path,
    *,
    backend: StorageBackend | None = None,
) -> dict[str, object]:
    """Dispose of consumed task-owned audio parents through their distinct transform proof"""

    restore = read_json(receipt_path)
    if restore.get("schema") != "speakrs-cold-restore-v1" or restore.get("ok") is not True:
        raise PreparationError("consumed-source cleanup requires the actual cold restore receipt")
    store = _backend_from_spec(spec, backend)
    batch, _ = _load_remote_batch(
        spec,
        {"batch": {"marker": restore["marker"], "batch_sha256": restore["batch_sha256"]}},
        store,
    )
    manifests = [item for item in batch["objects"] if item["purpose"] == "manifest"]
    if len(manifests) != 1:
        raise PreparationError("consumed-source cleanup requires one committed portable manifest")
    manifest_ref = manifests[0]
    raw = store.get_bytes(manifest_ref["key"])
    if len(raw) != manifest_ref["size"] or sha256_bytes(raw) != manifest_ref["sha256"]:
        raise PreparationError("consumed-source portable manifest identity changed")
    portable = json.loads(raw)
    transform = read_json(transform_path)
    parents = transform.get("parents") or []
    if (
        transform.get("schema") != "speakrs-source-transforms-v1"
        or {row["parent_id"] for row in parents} != set(portable["selected_parent_ids"])
        or len(parents) != len(portable["selected_parent_ids"])
        or len({row["source_audio"] for row in parents}) != len(parents)
    ):
        raise PreparationError("consumed audio-parent inventory differs from the committed batch")
    restore_digest = sha256_file(receipt_path)
    previous = read_json(output) if output.is_file() else {}
    if previous and previous.get("restore_receipt_sha256") != restore_digest:
        raise PreparationError("consumed-source journal belongs to another cold restore")
    discarded = list(previous.get("discarded") or [])
    retained = []
    for parent in parents:
        path = Path(parent["source_audio"])
        if not path.resolve(strict=False).is_relative_to(spec.disk.staging_root.resolve()):
            retained.append({"parent_id": parent["parent_id"], "reason": "original is outside task-owned staging"})
            continue
        if not path.exists():
            matches = [
                row
                for row in discarded
                if row.get("path") == str(path)
                and row.get("parent_id") == parent["parent_id"]
                and row.get("source_sha256") == parent["source_sha256"]
                and row.get("restore_receipt_sha256") == restore_digest
                and row.get("receipt_sha256")
                == sha256_json({key: value for key, value in row.items() if key != "receipt_sha256"})
            ]
            if len(matches) > 1:
                raise PreparationError("missing consumed source has duplicate disposal receipts")
            if matches:
                continue
            recovered = recover_deletion_journal(
                receipt_path,
                operation="consumed-source",
                path=path,
                expected_identity={
                    "source_sha256": parent["source_sha256"],
                    "parent_id": parent["parent_id"],
                    "acceptance_sha256": restore["acceptance_sha256"],
                    "restore_marker": {key: restore["marker"][key] for key in ("key", "sha256")},
                    "restore_receipt_sha256": restore_digest,
                },
            )
            if recovered is None:
                raise PreparationError("missing consumed source has no matching disposal receipt")
        discarded.append(
            discard_consumed_source(
                ConsumedSource(path, parent["parent_id"], parent["source_sha256"], spec.disk.staging_root),
                transform_receipt=transform_path,
                portable_manifest=portable,
                accepted_outputs=batch["objects"],
                restore_receipt=receipt_path,
                backend=store,
            )
        )
        write_json(
            output,
            {"ok": False, "discarded": discarded, "retained": retained, "restore_receipt_sha256": restore_digest},
        )
    payload = {
        "ok": True,
        "command": "discard-consumed-sources",
        "discarded": discarded,
        "retained": retained,
        "restore_receipt_sha256": restore_digest,
    }
    write_json(output, payload)
    return payload


def _find_release_seal(value: Any) -> Mapping[str, object] | None:
    """Find a final-release seal in a receipt wrapper."""

    if not isinstance(value, Mapping):
        return None
    if value.get("schema") == "speakrs-remote-release-v1":
        return value
    for name in ("seal", "release_seal", "final_release"):
        candidate = value.get(name)
        if isinstance(candidate, Mapping):
            found = _find_release_seal(candidate)
            if found is not None:
                return found
    return None


def _load_final_release_context(
    spec: DataPreparationSpec,
    seal_path: Path,
    store: StorageBackend,
) -> dict[str, Any]:
    """Validate the current final seal, source identities, and remote object union."""

    from .contracts import require_content_hash
    from .storage import _object_identity, _readback, _validate_batch_marker

    prefix = spec.r2.prefix
    seal = _find_release_seal(read_json(seal_path))
    if seal is None:
        raise PreparationError("final restore requires a final-release seal")
    if seal.get("state") != RemoteReleaseState.COMMITTED.value:
        raise PreparationError("final restore requires a committed final-release seal")
    marker = seal.get("marker")
    if not isinstance(marker, Mapping):
        raise PreparationError("final-release seal is missing its immutable marker")
    marker_key = marker.get("key")
    marker_sha256 = require_content_hash(marker.get("sha256"), "release marker sha256")
    marker_size = marker.get("size")
    if not isinstance(marker_key, str) or not marker_key.startswith(prefix + "/"):
        raise PreparationError("final-release marker is outside the authorized task prefix")
    if isinstance(marker_size, bool) or not isinstance(marker_size, int) or marker_size <= 0:
        raise PreparationError("final-release marker size is invalid")
    raw_marker = store.get_bytes(marker_key)
    if len(raw_marker) != marker_size or sha256_bytes(raw_marker) != marker_sha256:
        raise PreparationError("final-release marker failed full content verification")
    try:
        marker_payload = json.loads(raw_marker)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PreparationError("final-release marker is not valid JSON") from error
    if not isinstance(marker_payload, Mapping) or marker_payload.get("schema") != "speakrs-remote-release-v1":
        raise PreparationError("remote marker is not a final-release marker")
    if marker_payload.get("state") != RemoteReleaseState.COMMITTED.value:
        raise PreparationError("remote final-release marker is not committed")
    release_sha256 = require_content_hash(marker_payload.get("release_sha256"), "release sha256")
    marker_base = {key: value for key, value in marker_payload.items() if key != "release_sha256"}
    if sha256_json(marker_base) != release_sha256:
        raise PreparationError("final-release marker release hash is invalid")
    if seal.get("release_sha256") != release_sha256:
        raise PreparationError("local final-release seal differs from the remote marker")
    if {key: value for key, value in seal.items() if key != "marker"} != dict(marker_payload):
        raise PreparationError("local final-release seal content differs from its immutable marker")
    assert_private_access(store, marker_key, prefix)

    serialized_spec = data_spec_to_json(spec)
    current_implementation = _implementation_identity()
    current_closure_validator = _closure_validator_sha256()
    source_dispositions = validate_source_dispositions(spec)
    current_policy_sha256 = sha256_json(load_qa_policy(spec.qa_policy_path))
    source_by_name = {item.name: item for item in spec.sources}
    required_sources = sorted(item.name for item in spec.sources if item.membership is not SourceMembership.EXCLUDED)
    required_train_parents: dict[str, list[str]] = {}
    permission_sha256: dict[str, str] = {}
    configuration_sha256: dict[str, str] = {}
    for source in spec.sources:
        if source.membership is SourceMembership.EXCLUDED:
            continue
        splits = spec.frozen_splits.get(source.name)
        train = list((splits or {}).get("train") or ())
        if not splits or not train or len(set(train)) != len(train):
            raise PreparationError("non-excluded source has no exact frozen train parent inventory")
        required_train_parents[source.name] = sorted(str(parent) for parent in train)
        permission_payload = next(
            item for item in serialized_spec["permissions"] if item["record_id"] == source.permission_id
        )
        permission_sha256[source.name] = sha256_json(permission_payload)
        configuration_sha256[source.name] = _selection_spec_digest(spec, source.name)
    identity = marker_payload.get("release_identity")
    if not isinstance(identity, Mapping):
        raise PreparationError("final-release marker is missing its release identity")
    expected_identity = {
        "release_id": spec.release_id,
        "implementation_hashes": current_implementation,
        "closure_validator_sha256": current_closure_validator,
        "source_dispositions": source_dispositions,
        "qa_policy_sha256": current_policy_sha256,
        "required_sources": required_sources,
        "required_train_parents": required_train_parents,
        "configuration_sha256": sha256_json(configuration_sha256),
        "source_permission_sha256": permission_sha256,
    }
    if any(identity.get(name) != value for name, value in expected_identity.items()):
        raise PreparationError("final-release source or implementation identity is stale")
    expected_parent_union = sha256_json(
        [
            f"{source_name}:{parent}"
            for source_name in required_sources
            for parent in required_train_parents[source_name]
        ]
    )
    if identity.get("parent_union_sha256") != expected_parent_union:
        raise PreparationError("final-release parent union identity is stale")

    batch_payloads = marker_payload.get("batches")
    release_objects = marker_payload.get("objects")
    if not isinstance(batch_payloads, list) or not batch_payloads:
        raise PreparationError("final-release marker is missing committed batches")
    if not isinstance(release_objects, list) or not release_objects:
        raise PreparationError("final-release marker is missing its object inventory")
    release_by_key: dict[str, Mapping[str, object]] = {}
    for item in release_objects:
        try:
            identity_item = _object_identity(item, "final-release object")
        except ContractError as error:
            raise PreparationError("final-release marker contains an invalid object") from error
        key = str(identity_item["key"])
        if key in release_by_key:
            raise PreparationError("final-release marker contains duplicate object keys")
        release_by_key[key] = item
    batch_by_hash: dict[str, Mapping[str, object]] = {}
    batch_marker_by_hash: dict[str, Mapping[str, object]] = {}
    portable_by_batch: dict[str, Mapping[str, object]] = {}
    capacity_by_batch: dict[str, tuple[str, CapacityClosure]] = {}
    batch_union: dict[str, Mapping[str, object]] = {}
    for batch in batch_payloads:
        if not isinstance(batch, Mapping):
            raise PreparationError("final-release marker contains an invalid batch")
        try:
            batch_hash = require_content_hash(batch.get("batch_sha256"), "batch sha256")
            acceptance_hash = require_content_hash(batch.get("acceptance_sha256"), "batch acceptance sha256")
            marker_batch = _validate_batch_marker(store, batch)
        except ContractError as error:
            raise PreparationError("final-release marker contains an invalid batch") from error
        if batch_hash in batch_by_hash:
            raise PreparationError("final-release marker contains a duplicate batch")
        if marker_batch.get("batch_sha256") != batch_hash or marker_batch.get("acceptance_sha256") != acceptance_hash:
            raise PreparationError("final-release batch identity is inconsistent")
        marker_objects = marker_batch.get("objects")
        if not isinstance(marker_objects, list):
            raise PreparationError("final-release batch is missing its object inventory")
        batch_by_hash[batch_hash] = batch
        batch_marker_by_hash[batch_hash] = marker_batch
        for item in marker_objects:
            if not isinstance(item, Mapping):
                raise PreparationError("final-release batch contains an invalid object")
            try:
                identity_item = _object_identity(item, "final-release batch object")
            except ContractError as error:
                raise PreparationError("final-release batch contains an invalid object") from error
            key = str(identity_item["key"])
            if key in batch_union:
                raise PreparationError("final-release batches contain duplicate object keys")
            batch_union[key] = item
        manifests = [item for item in marker_objects if item.get("purpose") == "manifest"]
        if len(manifests) != 1:
            raise PreparationError("final-release batch requires exactly one portable manifest")
        manifest_identity = _object_identity(manifests[0], "final-release portable manifest")
        raw_manifest = store.get_bytes(str(manifest_identity["key"]))
        if len(raw_manifest) != int(manifest_identity["size"]) or sha256_bytes(raw_manifest) != str(
            manifest_identity["sha256"]
        ):
            raise PreparationError("final-release portable manifest failed full content verification")
        try:
            portable = json.loads(raw_manifest)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PreparationError("final-release portable manifest is not valid JSON") from error
        source_name = portable.get("source") if isinstance(portable, Mapping) else None
        if not isinstance(source_name, str) or source_name not in source_by_name:
            raise PreparationError("final-release portable manifest names an unknown source")
        source = source_by_name[source_name]
        permission_payload = next(
            item for item in serialized_spec["permissions"] if item["record_id"] == source.permission_id
        )
        if (
            portable.get("schema") != "speakrs-portable-training-selection-v1"
            or portable.get("release_id") != spec.release_id
            or portable.get("version") != source.version
            or portable.get("acceptance_sha256") != acceptance_hash
            or portable.get("profiles") != serialized_spec["profiles"]
            or portable.get("implementation_hashes") != current_implementation
            or portable.get("configuration_sha256") != configuration_sha256[source_name]
            or portable.get("permission") != permission_payload
        ):
            raise PreparationError("final-release portable source identity is stale")
        portable_by_batch[batch_hash] = portable
        capacity_by_batch[batch_hash] = (
            source_name,
            parse_capacity_manifest(portable, label=f"batch[{len(batch_by_hash) - 1}]"),
        )
    if set(batch_union) != set(release_by_key):
        raise PreparationError("final-release object union differs from its committed batches")
    for key, item in batch_union.items():
        if any(
            _object_identity(item)[name] != _object_identity(release_by_key[key])[name]
            for name in ("key", "sha256", "size")
        ):
            raise PreparationError("final-release object identity differs from its committed batches", {"key": key})
        _readback(
            store,
            key,
            expected_sha256=str(_object_identity(release_by_key[key])["sha256"]),
            expected_size=int(_object_identity(release_by_key[key])["size"]),
        )
        assert_private_access(store, key, prefix)
    required_batches = identity.get("required_batches")
    expected_batches = {
        batch_hash: require_content_hash(batch.get("acceptance_sha256"), "batch acceptance sha256")
        for batch_hash, batch in batch_by_hash.items()
    }
    if required_batches != dict(sorted(expected_batches.items())):
        raise PreparationError("final-release batch identity is stale or incomplete")
    if marker_payload.get("required_sources") != required_sources:
        raise PreparationError("final-release source membership differs from the current spec")
    return {
        "seal": seal,
        "marker_payload": marker_payload,
        "release_sha256": release_sha256,
        "batch_payloads": batch_payloads,
        "batch_by_hash": batch_by_hash,
        "batch_marker_by_hash": batch_marker_by_hash,
        "portable_by_batch": portable_by_batch,
        "capacity_by_batch": capacity_by_batch,
        "release_by_key": release_by_key,
        "required_sources": required_sources,
        "implementation_hashes": current_implementation,
        "closure_validator_sha256": current_closure_validator,
        "source_dispositions": source_dispositions,
    }


def _restore_samples_match_portable(
    samples: Any,
    portable: Mapping[str, object],
) -> bool:
    """Return whether CPU checks cover the committed portable parents and profiles."""

    recordings = portable.get("recordings")
    capacity = portable.get("capacity")
    if not isinstance(recordings, list) or not recordings or not isinstance(capacity, list) or not capacity:
        return False
    expected_recordings: set[str] = set()
    for row in recordings:
        if not isinstance(row, Mapping) or not isinstance(row.get("recording_id"), str):
            return False
        recording_id = str(row["recording_id"])
        if recording_id in expected_recordings:
            return False
        expected_recordings.add(recording_id)

    expected_profiles: set[tuple[int, int]] = set()
    for profile in capacity:
        if not isinstance(profile, Mapping):
            return False
        chunk_seconds = profile.get("chunk_seconds")
        max_overlap = profile.get("max_overlap")
        if (
            isinstance(chunk_seconds, bool)
            or not isinstance(chunk_seconds, int)
            or isinstance(max_overlap, bool)
            or not isinstance(max_overlap, int)
        ):
            return False
        expected_profiles.add((chunk_seconds, max_overlap))
    if not expected_profiles or not isinstance(samples, list) or not samples:
        return False

    frame_counts = {8: 399, 16: 799}
    powerset_channels = {2: 11, 4: 16}
    shifts = {8: 6, 16: 12}
    covered: set[tuple[str, int, int]] = set()
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("finite") is not True:
            return False
        recording_id = sample.get("recording_id")
        chunk_seconds = sample.get("chunk_seconds")
        if (
            not isinstance(recording_id, str)
            or recording_id not in expected_recordings
            or isinstance(chunk_seconds, bool)
            or not isinstance(chunk_seconds, int)
            or chunk_seconds not in {chunk for chunk, _ in expected_profiles}
        ):
            return False
        if sample.get("chunk_shift") != shifts.get(chunk_seconds):
            return False
        chunk_index = sample.get("chunk_index")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            return False
        frames = frame_counts.get(chunk_seconds)
        if sample.get("waveform_shape") != [1, 1, chunk_seconds * 16000] or sample.get("target_shape") != [
            1,
            frames,
            4,
        ]:
            return False
        if not isinstance(sample.get("source_path_used"), str):
            return False
        if not isinstance(sample.get("uem_regions"), list):
            return False
        speaker_mask = sample.get("speaker_mask")
        if (
            not isinstance(speaker_mask, list)
            or len(speaker_mask) != 1
            or not isinstance(speaker_mask[0], list)
            or len(speaker_mask[0]) != 4
            or any(not isinstance(item, bool) for item in speaker_mask[0])
        ):
            return False
        encodings = sample.get("encodings")
        if not isinstance(encodings, list):
            return False
        sample_profiles: set[tuple[int, int]] = set()
        for encoding in encodings:
            if not isinstance(encoding, Mapping):
                return False
            overlap = encoding.get("max_overlap")
            shape = encoding.get("powerset_shape")
            if (
                isinstance(overlap, bool)
                or not isinstance(overlap, int)
                or (chunk_seconds, overlap) not in expected_profiles
                or not isinstance(shape, list)
                or len(shape) != 3
                or shape[0] != 1
                or shape[1] != frames
                or isinstance(shape[2], bool)
                or not isinstance(shape[2], int)
                or shape[2] != powerset_channels.get(overlap)
                or (chunk_seconds, overlap) in sample_profiles
            ):
                return False
            sample_profiles.add((chunk_seconds, overlap))
        if sample_profiles != {profile for profile in expected_profiles if profile[0] == chunk_seconds}:
            return False
        covered.update((recording_id, chunk, overlap) for chunk, overlap in sample_profiles)

    return covered == {
        (recording_id, chunk_seconds, max_overlap)
        for recording_id in expected_recordings
        for chunk_seconds, max_overlap in expected_profiles
    }


def _batch_restore_matches(
    value: Any,
    batch: Mapping[str, object],
    marker_batch: Mapping[str, object],
    *,
    portable: Mapping[str, object] | None = None,
) -> bool:
    """Return whether one cold receipt still proves its committed batch."""

    from .contracts import require_content_hash
    from .storage import _object_identity

    if not isinstance(value, Mapping):
        return False
    if value.get("schema") != "speakrs-cold-restore-v1" or value.get("command") != "restore-check":
        return False
    if value.get("ok") is not True or value.get("complete_release") is not False:
        return False
    try:
        # original v1 receipts predate attempt incarnations and remain valid read-only proof
        if "incarnation_id" in value:
            require_content_hash(value["incarnation_id"], "restore incarnation id")
    except ContractError:
        return False
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        return False
    if portable is not None and not _restore_samples_match_portable(samples, portable):
        return False
    try:
        if require_content_hash(value.get("batch_sha256"), "restore batch sha256") != require_content_hash(
            batch.get("batch_sha256"), "batch sha256"
        ):
            return False
        if require_content_hash(value.get("acceptance_sha256"), "restore acceptance sha256") != require_content_hash(
            batch.get("acceptance_sha256"), "batch acceptance sha256"
        ):
            return False
    except ContractError:
        return False
    expected_marker = batch.get("marker")
    actual_marker = value.get("marker")
    if not isinstance(expected_marker, Mapping) or not isinstance(actual_marker, Mapping):
        return False
    if {key: actual_marker.get(key) for key in ("key", "sha256", "size")} != {
        key: expected_marker.get(key) for key in ("key", "sha256", "size")
    }:
        return False
    expected_objects = marker_batch.get("objects")
    actual_objects = value.get("objects")
    if not isinstance(expected_objects, list) or not isinstance(actual_objects, list):
        return False
    if len(expected_objects) != len(actual_objects):
        return False
    expected_by_key: dict[str, Mapping[str, object]] = {}
    for item in expected_objects:
        if not isinstance(item, Mapping):
            return False
        try:
            key = str(_object_identity(item)["key"])
        except ContractError:
            return False
        if key in expected_by_key:
            return False
        expected_by_key[key] = item
    actual_by_key: dict[str, Mapping[str, object]] = {}
    for item in actual_objects:
        try:
            identity = _object_identity(item)
        except ContractError:
            return False
        key = str(identity["key"])
        if key in actual_by_key or key not in expected_by_key:
            return False
        actual_by_key[key] = item
        expected = _object_identity(expected_by_key[key])
        if any(identity[name] != expected[name] for name in ("key", "sha256", "size")):
            return False
    if set(actual_by_key) != set(expected_by_key):
        return False
    if portable is None:
        return True
    recordings = portable.get("recordings")
    if not isinstance(recordings, list):
        return False
    audio_by_recording: dict[str, str] = {}
    for row in recordings:
        if not isinstance(row, Mapping) or not isinstance(row.get("recording_id"), str):
            return False
        recording_id = str(row["recording_id"])
        audio = row.get("audio")
        if not isinstance(audio, Mapping) or not isinstance(audio.get("key"), str):
            return False
        audio_key = str(audio["key"])
        expected_audio = expected_by_key.get(audio_key)
        actual_audio = actual_by_key.get(audio_key)
        if (
            expected_audio is None
            or actual_audio is None
            or expected_audio.get("purpose") != "train-audio"
            or expected_audio.get("parent_id") != recording_id
            or not isinstance(actual_audio.get("restored_path"), str)
        ):
            return False
        if recording_id in audio_by_recording:
            return False
        audio_by_recording[recording_id] = audio_key
    for sample in samples:
        recording_id = sample.get("recording_id")
        if not isinstance(recording_id, str):
            return False
        audio_key = audio_by_recording.get(recording_id)
        if audio_key is None or sample.get("source_path_used") != actual_by_key[audio_key].get("restored_path"):
            return False
    return True


def _validate_final_restore_aggregate(
    value: Mapping[str, object],
    *,
    release_sha256: str,
    seal: Mapping[str, object],
    implementation_hashes: Mapping[str, str],
    closure_validator_sha256: str,
    source_dispositions: Mapping[str, Mapping[str, object]],
    expected_batches: Mapping[str, Mapping[str, object]],
    expected_batch_markers: Mapping[str, Mapping[str, object]],
    expected_portable: Mapping[str, Mapping[str, object]],
    expected_objects: Mapping[str, Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Validate the content hash and metadata of a generated final restore index."""

    from .contracts import require_content_hash
    from .storage import _object_identity

    if value.get("schema") != "speakrs-final-restore-v1" or value.get("command") != "final-restore":
        raise PreparationError("unknown final restore aggregate schema")
    if value.get("state") != RemoteReleaseState.COMMITTED.value or value.get("ok") is not True:
        raise PreparationError("final restore aggregate is not committed")
    if value.get("complete_release") is not True or value.get("release_sha256") != release_sha256:
        raise PreparationError("final restore aggregate is not bound to the final seal")
    if value.get("seal") != seal:
        raise PreparationError("final restore aggregate seal differs from the current final seal")
    if value.get("implementation_hashes") != dict(implementation_hashes):
        raise PreparationError("final restore aggregate implementation identity is stale")
    if value.get("closure_validator_sha256") != closure_validator_sha256:
        raise PreparationError("final restore aggregate closure identity is stale")
    if value.get("source_dispositions") != dict(source_dispositions):
        raise PreparationError("final restore aggregate source disposition identity is stale")
    aggregate_sha256 = require_content_hash(value.get("aggregate_sha256"), "aggregate restore sha256")
    base = {key: item for key, item in value.items() if key != "aggregate_sha256"}
    if sha256_json(base) != aggregate_sha256:
        raise PreparationError("final restore aggregate hash is invalid")
    aggregate_objects = value.get("objects")
    if not isinstance(aggregate_objects, list) or len(aggregate_objects) != len(expected_objects):
        raise PreparationError("final restore aggregate object inventory is incomplete")
    aggregate_by_key: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(aggregate_objects):
        if not isinstance(item, Mapping):
            raise PreparationError("final restore aggregate contains an invalid object", {"index": index})
        try:
            identity = _object_identity(item, f"final restore aggregate object[{index}]")
        except ContractError as error:
            raise PreparationError("final restore aggregate contains an invalid object", {"index": index}) from error
        key = str(identity["key"])
        if key in aggregate_by_key:
            raise PreparationError("final restore aggregate contains a duplicate object", {"key": key})
        aggregate_by_key[key] = item
        expected = expected_objects.get(key)
        if expected is None:
            raise PreparationError("final restore aggregate object is outside the final release", {"key": key})
        expected_identity = _object_identity(expected, "final release object")
        if any(identity[name] != expected_identity[name] for name in ("key", "sha256", "size")):
            raise PreparationError("final restore aggregate object identity is stale", {"key": key})
    if set(aggregate_by_key) != set(expected_objects):
        raise PreparationError("final restore aggregate object union differs from the final release")
    receipts = value.get("restore_receipts")
    batches = value.get("batches")
    if (
        not isinstance(receipts, list)
        or not receipts
        or not isinstance(batches, list)
        or len(batches) != len(receipts)
    ):
        raise PreparationError("final restore aggregate receipt inventory is incomplete")
    expected_batch_hashes = set(expected_batches)
    seen: set[str] = set()
    for index, (batch, receipt) in enumerate(zip(batches, receipts, strict=True)):
        if not isinstance(batch, Mapping) or not isinstance(receipt, Mapping):
            raise PreparationError("final restore aggregate contains an invalid batch", {"index": index})
        batch_hash = require_content_hash(batch.get("batch_sha256"), "aggregate batch sha256")
        acceptance_hash = require_content_hash(batch.get("acceptance_sha256"), "aggregate acceptance sha256")
        if batch_hash in seen:
            raise PreparationError("final restore aggregate contains a duplicate batch")
        expected_batch = expected_batches.get(batch_hash)
        expected_marker = expected_batch_markers.get(batch_hash)
        expected_portable_value = expected_portable.get(batch_hash)
        if expected_batch is None or expected_marker is None or expected_portable_value is None:
            raise PreparationError("final restore aggregate batch is outside the final release", {"index": index})
        expected_acceptance = require_content_hash(expected_batch.get("acceptance_sha256"), "batch acceptance sha256")
        if acceptance_hash != expected_acceptance:
            raise PreparationError("final restore aggregate batch acceptance identity is stale", {"index": index})
        seen.add(batch_hash)
        if (
            receipt.get("schema") != "speakrs-cold-restore-v1"
            or receipt.get("command") != "restore-check"
            or receipt.get("batch_sha256") != batch_hash
            or receipt.get("acceptance_sha256") != acceptance_hash
        ):
            raise PreparationError("final restore aggregate batch and receipt identities differ", {"index": index})
        if not _batch_restore_matches(receipt, expected_batch, expected_marker, portable=expected_portable_value):
            raise PreparationError("final restore aggregate contains invalid batch restore evidence", {"index": index})
        if batch.get("restore_receipt_sha256") != sha256_json(receipt):
            raise PreparationError("final restore aggregate receipt hash is invalid", {"index": index})
    if seen != expected_batch_hashes:
        raise PreparationError("final restore aggregate batch union differs from the final release")
    return [receipt for receipt in receipts if isinstance(receipt, Mapping)]


def final_restore_data(
    spec: DataPreparationSpec,
    release: Path,
    seal_path: Path,
    output: Path,
    *,
    restore_receipts_path: Path | None = None,
    backend: StorageBackend | None = None,
) -> dict[str, object]:
    """Build a content-bound final restore index from valid batch proofs."""

    from .contracts import require_content_hash

    store = _backend_from_spec(spec, backend)
    context = _load_final_release_context(spec, seal_path, store)
    release_sha256 = str(context["release_sha256"])
    expected_batches: dict[str, Mapping[str, object]] = {
        str(batch_hash): batch for batch_hash, batch in context["batch_by_hash"].items()
    }
    marker_by_hash: dict[str, Mapping[str, object]] = context["batch_marker_by_hash"]
    candidates: list[tuple[Mapping[str, object] | None, str | None]] = []

    def collect(value: Any, declared_batch: str | None = None) -> None:
        if isinstance(value, list):
            for item in value:
                collect(item, declared_batch)
            return
        if isinstance(value, Mapping):
            schema = value.get("schema")
            if schema == "speakrs-final-restore-v1":
                receipts = _validate_final_restore_aggregate(
                    value,
                    release_sha256=release_sha256,
                    seal=context["seal"],
                    implementation_hashes=context["implementation_hashes"],
                    closure_validator_sha256=context["closure_validator_sha256"],
                    source_dispositions=context["source_dispositions"],
                    expected_batches=expected_batches,
                    expected_batch_markers=marker_by_hash,
                    expected_portable=context["portable_by_batch"],
                    expected_objects=context["release_by_key"],
                )
                for item in receipts:
                    collect(item)
                return
            if schema == "speakrs-cold-restore-v1":
                candidates.append((value, declared_batch))
                return
            if schema == "speakrs-restore-index-v1":
                if value.get("release_sha256") != release_sha256:
                    raise PreparationError("restore index belongs to a different final release")
                collect(value.get("receipts"), declared_batch)
                return
            if "restore_receipts" in value or "restores" in value:
                collect(value.get("restore_receipts", value.get("restores")), declared_batch)
                return
            path_value = value.get("path")
            if isinstance(path_value, str):
                declared = value.get("batch_sha256", declared_batch)
                if declared is not None:
                    declared = require_content_hash(declared, "restore index batch sha256")
                path = Path(path_value)
                if not path.is_file():
                    candidates.append((None, declared))
                    return
                try:
                    collect(read_json(path), declared)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    candidates.append((None, declared))
                return
            raise PreparationError("restore index contains an unknown receipt schema")
        if isinstance(value, str):
            path = Path(value)
            if not path.is_file():
                candidates.append((None, declared_batch))
                return
            try:
                collect(read_json(path), declared_batch)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                candidates.append((None, declared_batch))
            return
        raise PreparationError("restore index contains an invalid receipt")

    if restore_receipts_path is not None and restore_receipts_path.is_file():
        try:
            collect(read_json(restore_receipts_path))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            candidates = []

    expected_hashes = set(expected_batches)
    by_batch: dict[str, Mapping[str, object] | None] = {}
    for payload, declared_batch in candidates:
        batch_hash: str | None = declared_batch
        if payload is not None:
            raw_batch_hash = payload.get("batch_sha256")
            if not isinstance(raw_batch_hash, str):
                if len(expected_batches) == 1:
                    batch_hash = next(iter(expected_batches))
                else:
                    raise PreparationError("restore receipt has no batch identity")
            else:
                batch_hash = require_content_hash(raw_batch_hash, "restore batch sha256")
        if batch_hash is None:
            if len(expected_batches) != 1:
                raise PreparationError("restore receipt cannot be assigned to a final-release batch")
            batch_hash = next(iter(expected_batches))
        if batch_hash not in expected_hashes:
            raise PreparationError("restore receipt belongs to a different final release")
        if batch_hash in by_batch:
            raise PreparationError("restore index contains duplicate batch receipts")
        by_batch[batch_hash] = payload

    durable_root = output.parent / "final-restore-batches"
    for batch_hash in expected_hashes:
        existing = by_batch.get(batch_hash)
        if existing is not None and _batch_restore_matches(
            existing,
            expected_batches[batch_hash],
            marker_by_hash[batch_hash],
            portable=context["portable_by_batch"][batch_hash],
        ):
            continue
        durable_path = durable_root / batch_hash / "restore.json"
        if not durable_path.is_file():
            continue
        try:
            durable = read_json(durable_path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not _batch_restore_matches(
            durable,
            expected_batches[batch_hash],
            marker_by_hash[batch_hash],
            portable=context["portable_by_batch"][batch_hash],
        ):
            continue
        if restore_receipts_path is None:
            by_batch[batch_hash] = durable
            continue
        restore_digest = sha256_file(durable_path)
        eviction_path = durable_path.with_name("eviction.json")
        cleanup_complete = False
        if eviction_path.is_file():
            try:
                previous = read_json(eviction_path)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                previous = None
            cleanup_complete = (
                isinstance(previous, Mapping)
                and previous.get("restore_receipt_sha256") == restore_digest
                and previous.get("ok") is True
            )
        if not cleanup_complete:
            by_batch[batch_hash] = durable

    def _cleanup_durable_restore(durable_path: Path, eviction_path: Path) -> Path:
        """Finish a durable restore cleanup, including an interrupted eviction journal."""

        if not durable_path.is_file():
            return eviction_path
        restore_digest = sha256_file(durable_path)
        cleanup_path = eviction_path
        if cleanup_path.is_file():
            try:
                previous = read_json(cleanup_path)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                previous = None
            if isinstance(previous, Mapping) and previous.get("restore_receipt_sha256") == restore_digest:
                if previous.get("ok") is True:
                    return cleanup_path
            else:
                cleanup_path = durable_path.with_name(f"eviction-{restore_digest}.json")
        evict_data(spec, durable_path, cleanup_path, backend=store)
        return cleanup_path

    output.parent.mkdir(parents=True, exist_ok=True)
    restored: list[Mapping[str, object]] = []
    batch_results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="speakrs-final-restore-", dir=str(output.parent)) as temporary:
        work = Path(temporary)
        for index, batch_hash in enumerate(sorted(expected_batches)):
            batch = expected_batches[batch_hash]
            marker_batch = marker_by_hash[batch_hash]
            existing = by_batch.get(batch_hash)
            reused = existing is not None and _batch_restore_matches(
                existing,
                batch,
                marker_batch,
                portable=context["portable_by_batch"][batch_hash],
            )
            durable_path = durable_root / batch_hash / "restore.json"
            eviction_path = durable_root / batch_hash / "eviction.json"
            durable_restore = None
            if durable_path.is_file():
                try:
                    candidate = read_json(durable_path)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    candidate = None
                if _batch_restore_matches(
                    candidate,
                    batch,
                    marker_batch,
                    portable=context["portable_by_batch"][batch_hash],
                ):
                    durable_restore = candidate
            if reused:
                restore = existing
            else:
                batch_input = work / f"batch-{index:04d}.json"
                write_json(batch_input, {"batch": dict(batch)})
                restore = restore_check(spec, batch_input, durable_path, backend=store)
                durable_restore = restore
            if not isinstance(restore, Mapping) or not _batch_restore_matches(
                restore,
                batch,
                marker_batch,
                portable=context["portable_by_batch"][batch_hash],
            ):
                raise PreparationError("fresh cold restore did not match its committed batch")
            if durable_restore is not None:
                eviction_path = _cleanup_durable_restore(durable_path, eviction_path)
            restored.append(restore)
            batch_result = {
                "batch_sha256": batch_hash,
                "acceptance_sha256": require_content_hash(batch.get("acceptance_sha256"), "batch acceptance sha256"),
                "restore_receipt_sha256": sha256_json(restore),
                "reused": reused,
            }
            if durable_path.is_file() and eviction_path.is_file():
                batch_result.update(
                    {
                        "restore_receipt_path": str(durable_path),
                        "eviction_receipt_path": str(eviction_path),
                        "eviction_receipt_sha256": sha256_file(eviction_path),
                    }
                )
            batch_results.append(batch_result)

        base: dict[str, object] = {
            "schema": "speakrs-final-restore-v1",
            "state": RemoteReleaseState.COMMITTED.value,
            "command": "final-restore",
            "ok": True,
            "release": str(release),
            "seal": dict(context["seal"]),
            "release_sha256": release_sha256,
            "implementation_hashes": dict(context["implementation_hashes"]),
            "closure_validator_sha256": context["closure_validator_sha256"],
            "source_dispositions": dict(context["source_dispositions"]),
            "batches": batch_results,
            "restore_receipts": restored,
            "objects": [dict(context["release_by_key"][key]) for key in sorted(context["release_by_key"])],
            "complete_release": True,
        }
        aggregate = {**base, "aggregate_sha256": sha256_json(base)}
        _validate_final_restore_aggregate(
            aggregate,
            release_sha256=release_sha256,
            seal=context["seal"],
            implementation_hashes=context["implementation_hashes"],
            closure_validator_sha256=context["closure_validator_sha256"],
            source_dispositions=context["source_dispositions"],
            expected_batches=expected_batches,
            expected_batch_markers=marker_by_hash,
            expected_portable=context["portable_by_batch"],
            expected_objects=context["release_by_key"],
        )
    write_json(output, aggregate)
    return aggregate


restore_release_data = final_restore_data


def evict_data(
    spec: DataPreparationSpec,
    receipt_path: Path,
    output: Path,
    copies: Sequence[LocalCopy] = (),
    *,
    backend: StorageBackend | None = None,
) -> dict[str, object]:
    """Evict only managed copies bound to a committed batch and real cold restore"""

    if copies:
        raise PreparationError("eviction copies must come from the verified restore receipt")
    receipt = read_json(receipt_path)
    if (
        receipt.get("schema") != "speakrs-cold-restore-v1"
        or receipt.get("command") != "restore-check"
        or not receipt.get("ok")
    ):
        raise PreparationError("eviction requires the actual cold restore receipt")
    store = _backend_from_spec(spec, backend)
    restored_by_key = {item["key"]: item for item in receipt["objects"]}
    candidates = list(receipt.get("local_copies") or [])
    candidates += [{**item, "path": item["restored_path"], "source": "restored-cache"} for item in receipt["objects"]]
    restore_digest = sha256_file(receipt_path)
    previous = read_json(output) if output.is_file() else {}
    if previous and previous.get("restore_receipt_sha256") != restore_digest:
        raise PreparationError("eviction journal belongs to a different restore receipt")
    evicted = list(previous.get("evicted") or [])
    recovering: set[Path] = set()
    for item in candidates:
        path = Path(item["path"])
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise PreparationError("eviction candidate changed from a regular file")
            continue
        matches = [
            entry
            for entry in evicted
            if entry.get("path") == str(path)
            and entry.get("sha256") == item["sha256"]
            and entry.get("object_key") == item["key"]
            and entry.get("restore_receipt_sha256") == restore_digest
            and entry.get("receipt_sha256")
            == sha256_json({key: value for key, value in entry.items() if key != "receipt_sha256"})
        ]
        if len(matches) > 1:
            raise PreparationError("missing local copy has duplicate deletion receipts")
        if matches:
            continue
        recovered = recover_deletion_journal(
            receipt_path,
            operation="eviction",
            path=path,
            expected_identity={
                "copy_sha256": item["sha256"],
                "object_key": item["key"],
                "object_sha256": item["sha256"],
                "object_size": int(item["size"]),
                "acceptance_sha256": receipt["acceptance_sha256"],
                "marker_key": receipt["marker"]["key"],
                "marker_sha256": receipt["marker"]["sha256"],
                "restore_receipt_sha256": restore_digest,
            },
        )
        if recovered is None:
            raise PreparationError("missing local copy has no matching completed deletion receipt")
        recovering.add(path)
    source_cleanup = None
    if receipt.get("source_transform"):
        transform_path, _ = _verified_reference(receipt["source_transform"], "source transform")
        cleanup_path = output.with_name(output.stem + "-sources" + output.suffix)
        # raw disposal still checks the live cold copies before canonical-cache eviction removes them
        discard_data_sources(spec, receipt_path, transform_path, cleanup_path, backend=store)
        source_cleanup = {"path": str(cleanup_path), "sha256": sha256_file(cleanup_path)}
    for item in candidates:
        path = Path(item["path"])
        if not path.is_file() and path not in recovering:
            continue
        if path.is_symlink() or not any(
            path.resolve().is_relative_to(root.resolve()) for root in (spec.disk.staging_root, spec.disk.cache_root)
        ):
            raise PreparationError("eviction refuses a path outside the managed task roots")
        restored = restored_by_key.get(item["key"])
        if restored is None:
            raise PreparationError("eviction object has no matching cold restore identity")
        proof = RemoteRestoreProof(
            object_key=item["key"],
            object_sha256=item["sha256"],
            object_size=int(item["size"]),
            acceptance_sha256=receipt["acceptance_sha256"],
            marker_key=receipt["marker"]["key"],
            marker_sha256=receipt["marker"]["sha256"],
            restore_receipt_sha256=restore_digest,
            restored_sha256=restored["sha256"],
            restored_size=int(restored["size"]),
            restore_receipt_path=receipt_path,
        )
        copy = LocalCopy(
            path=path,
            # a saved preverified intent preserves eligibility across an interrupted unlink
            state=LocalCopyState.EVICTION_ELIGIBLE if path in recovering else LocalCopyState.RETAINED,
            source=item.get("source", ""),
            sha256=item["sha256"],
            labels_accepted=True,
            remote_proof=proof,
        )
        eligible = copy if path in recovering else mark_eviction_eligible(copy, backend=store)
        gone = evict_copy(eligible, backend=store)
        evicted.append(dict(gone.deletion_receipt))
        write_json(
            output,
            {"ok": False, "command": "evict", "evicted": evicted, "restore_receipt_sha256": restore_digest},
        )
    payload = {"ok": True, "command": "evict", "evicted": evicted, "restore_receipt_sha256": restore_digest}
    if source_cleanup is not None:
        payload["source_cleanup"] = source_cleanup
    write_json(output, payload)
    return payload


def handoff_data(
    spec: DataPreparationSpec,
    release: Path,
    receipt_path: Path,
    output: Path,
    *,
    backend: StorageBackend | None = None,
) -> dict[str, object]:
    """Require the generated content-bound final restore aggregate."""

    from .contracts import require_content_hash

    store = _backend_from_spec(spec, backend)
    receipt = read_json(receipt_path)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != "speakrs-final-restore-v1"
        or receipt.get("command") != "final-restore"
    ):
        raise PreparationError("handoff requires the generated final restore aggregate")
    if receipt.get("release") != str(release):
        raise PreparationError("final restore aggregate belongs to a different release path")

    context = _load_final_release_context(spec, receipt_path, store)
    release_sha256 = str(context["release_sha256"])
    seal = context["seal"]
    current_implementation = context["implementation_hashes"]
    current_closure_validator_sha256 = context["closure_validator_sha256"]
    source_dispositions = context["source_dispositions"]
    batch_by_hash = context["batch_by_hash"]
    marker_by_hash = context["batch_marker_by_hash"]
    portable_by_batch = context["portable_by_batch"]
    release_by_key = context["release_by_key"]
    _validate_final_restore_aggregate(
        receipt,
        release_sha256=release_sha256,
        seal=seal,
        implementation_hashes=current_implementation,
        closure_validator_sha256=current_closure_validator_sha256,
        source_dispositions=source_dispositions,
        expected_batches=batch_by_hash,
        expected_batch_markers=marker_by_hash,
        expected_portable=portable_by_batch,
        expected_objects=release_by_key,
    )

    unresolved_permissions = [
        source.name
        for source in spec.sources
        if source.membership is not SourceMembership.EXCLUDED
        and (
            source.permission_state.value != "permitted"
            or not spec.permissions[source.permission_id].permitted_for_training_storage()
        )
    ]
    if unresolved_permissions:
        raise UnresolvedInputError(
            "source permission is unresolved for the final handoff",
            {"sources": sorted(unresolved_permissions)},
        )

    required_sources = context["required_sources"]
    identity = context["marker_payload"]["release_identity"]
    if not isinstance(identity, Mapping):
        raise PreparationError("final-release marker is missing its release identity")
    source_capacity_closures: dict[str, CapacityClosure] = {}
    for source_name in required_sources:
        records = [
            capacity
            for batch_hash, (batch_source, capacity) in sorted(context["capacity_by_batch"].items())
            if batch_source == source_name
        ]
        source_capacity_closures[source_name] = aggregate_capacity(records, label=f"source[{source_name}]")
    source_capacity = {source_name: closure.as_list() for source_name, closure in source_capacity_closures.items()}
    source_admitted_profiles = {
        source_name: [item for item in source_capacity[source_name] if item["admitted"]]
        for source_name in required_sources
    }
    common_admitted = common_admitted_profiles(source_capacity_closures)
    if not common_admitted:
        raise PreparationError("no common admitted target capacity profile across required sources")
    expected_batches = {
        batch_hash: require_content_hash(batch.get("acceptance_sha256"), "batch acceptance sha256")
        for batch_hash, batch in batch_by_hash.items()
    }
    expected_capacity_identity = {
        "source_capacity": {name: source_capacity[name] for name in required_sources},
        "admitted_profiles": {name: source_admitted_profiles[name] for name in required_sources},
        "common_admitted_profiles": common_admitted,
        "required_batches": dict(sorted(expected_batches.items())),
    }
    if any(identity.get(name) != value for name, value in expected_capacity_identity.items()):
        raise PreparationError("final-release identity is stale for the current data spec")

    payload = {
        "ok": True,
        "command": "handoff",
        "release": str(release),
        "sources": _source_status(spec),
        "admitted_profiles": common_admitted,
        "training_ready": True,
        "seal_sha256": release_sha256,
        "restore_receipt_sha256": sha256_file(receipt_path),
        "complete_release": True,
    }
    write_json(output, payload)
    return payload


def dispatch_data(args, *, backend: StorageBackend | None = None) -> dict[str, object]:
    """Dispatch one data subcommand."""

    _reject_forbidden_action(getattr(args, "action", None))
    command = args.data_command
    spec = load_data_spec(args.config)
    if command == "plan":
        return plan_data(spec, args.output)
    if command == "prepare":
        return prepare_data(spec, args.output, source=args.source)
    if command == "verify":
        return verify_data(spec, args.release, args.output)
    if command == "upload":
        artifacts = read_json(args.artifacts) if getattr(args, "artifacts", None) else []
        return upload_data(spec, args.release, args.output, artifacts=artifacts, backend=backend)
    if command == "verify-remote":
        return verify_remote(
            spec,
            args.receipt,
            args.output,
            backend=backend,
            label_policy_id=getattr(args, "label_policy_id", "label-qa-policy"),
            split_id=getattr(args, "split_id", "frozen-splits"),
        )
    if command == "commit-release":
        return commit_remote_release(spec, args.release, args.receipts, args.output, backend=backend)
    if command == "restore-check":
        return restore_check(spec, args.receipt, args.output, backend=backend)
    if command == "final-restore":
        return final_restore_data(
            spec,
            args.release,
            args.seal,
            args.output,
            restore_receipts_path=getattr(args, "restores", None),
            backend=backend,
        )
    if command == "evict":
        return evict_data(spec, args.receipt, args.output, backend=backend)
    if command == "handoff":
        return handoff_data(spec, args.release, args.receipt, args.output, backend=backend)
    raise LargeError("usage", f"unknown data command {command}")
