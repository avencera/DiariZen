"""Validate evidence that removes a complete source from release admission"""

from __future__ import annotations

import json
from typing import Any

from .contracts import DataPreparationSpec, EvidenceReference, SourceMembership, data_spec_to_json
from .errors import PreparationError
from .hashing import sha256_file, sha256_json


DISPOSITION_SCHEMA = "speakrs-source-disposition-v1"


def _verify_reference(reference: EvidenceReference, label: str) -> None:
    if not reference.path.is_file() or sha256_file(reference.path) != reference.sha256:
        raise PreparationError("source disposition evidence is missing or changed", {"check": label})


def _read_reference(reference: EvidenceReference, label: str) -> Any:
    _verify_reference(reference, label)
    try:
        return json.loads(reference.path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError("source disposition evidence is not readable JSON", {"check": label}) from error


def validate_source_dispositions(spec: DataPreparationSpec) -> dict[str, dict[str, Any]]:
    """Check excluded source closure and return the path-free final-seal identity

    The decision record identifies the complete frozen selection. Supporting
    records remain private, but their content identities are in the final seal
    """

    dispositions = {}
    permissions = {row["record_id"]: row for row in data_spec_to_json(spec)["permissions"]}
    for source in spec.sources:
        if source.membership is not SourceMembership.EXCLUDED:
            continue
        reference = source.disposition_evidence
        if reference is None:
            raise PreparationError("excluded source has no content-bound disposition")
        record = _read_reference(reference, source.name)
        frozen = {name: list(parents) for name, parents in spec.frozen_splits.get(source.name, {}).items()}
        if (
            not isinstance(record, dict)
            or record.get("schema") != DISPOSITION_SCHEMA
            or record.get("source") != source.name
            or record.get("version") != source.version
            or record.get("state") != "excluded"
            or record.get("frozen_splits_sha256") != sha256_json(frozen)
            or record.get("permission_sha256") != sha256_json(permissions[source.permission_id])
            or not isinstance(record.get("reason"), str)
            or not record["reason"].strip()
        ):
            raise PreparationError("source disposition does not bind the current source and frozen selection")
        excluded = record.get("excluded_train_parent_ids")
        expected = list(frozen.get("train", []))
        if (
            not isinstance(excluded, list)
            or any(not isinstance(parent, str) or not parent for parent in excluded)
            or len(set(excluded)) != len(excluded)
            or set(excluded) != set(expected)
        ):
            raise PreparationError("source disposition does not cover the exact excluded train parent inventory")
        evidence = record.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            raise PreparationError("source disposition requires supporting evidence")
        identities = {}
        for label, payload in sorted(evidence.items()):
            proof = EvidenceReference.parse(payload, f"source disposition {label}")
            _verify_reference(proof, label)
            identities[label] = proof.sha256
        dispositions[source.name] = {
            "source": source.name,
            "version": source.version,
            "state": "excluded",
            "reason": record["reason"],
            "frozen_splits_sha256": record["frozen_splits_sha256"],
            "permission_sha256": record["permission_sha256"],
            "excluded_train_parent_ids": sorted(excluded),
            "record_sha256": reference.sha256,
            "evidence_sha256": identities,
        }
    return dispositions
