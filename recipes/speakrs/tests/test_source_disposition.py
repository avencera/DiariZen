"""Protect source exclusions from missing or stale evidence"""

import pytest

from recipes.speakrs.large.contracts import data_spec_to_json, parse_data_preparation_spec
from recipes.speakrs.large.errors import ContractError, PreparationError
from recipes.speakrs.large.hashing import sha256_file, sha256_json
from recipes.speakrs.large.jsonio import read_json, write_json
from recipes.speakrs.large.source_disposition import validate_source_dispositions
from recipes.speakrs.tests.test_selection_verification import selection as selection_fixture


@pytest.fixture
def exclusion(tmp_path):
    spec, raw, release, _ = selection_fixture.__wrapped__(tmp_path)
    source = raw["sources"][0]
    proof = release / "exclusion-proof.json"
    write_json(proof, {"source": "AMI", "all_train_parents_have_heldout_speakers": True})
    record = release / "source-disposition.json"
    write_json(
        record,
        {
            "schema": "speakrs-source-disposition-v1",
            "source": source["name"],
            "version": source["version"],
            "state": "excluded",
            "reason": "The complete frozen training selection shares heldout speakers",
            "frozen_splits_sha256": sha256_json(raw["frozen_splits"]["AMI"]),
            "permission_sha256": sha256_json(data_spec_to_json(spec)["permissions"][0]),
            "excluded_train_parent_ids": raw["frozen_splits"]["AMI"]["train"],
            "evidence": {"split_check": {"path": str(proof), "sha256": sha256_file(proof)}},
        },
    )
    source["membership"] = "excluded"
    source["missing_action"] = None
    source["disposition_evidence"] = {"path": str(record), "sha256": sha256_file(record)}
    return raw, record, proof


def test_exclusion_requires_a_content_reference(exclusion):
    raw, _, _ = exclusion
    del raw["sources"][0]["disposition_evidence"]
    with pytest.raises(ContractError, match="disposition reference"):
        parse_data_preparation_spec(raw)


def test_verified_exclusion_identity_has_no_local_paths(exclusion):
    raw, record, _ = exclusion
    result = validate_source_dispositions(parse_data_preparation_spec(raw))["AMI"]
    assert result["record_sha256"] == sha256_file(record)
    assert result["excluded_train_parent_ids"] == raw["frozen_splits"]["AMI"]["train"]
    assert str(record.parent) not in str(result)


@pytest.mark.parametrize("change", ["record", "proof", "missing_proof", "split", "permission", "version"])
def test_changed_exclusion_dependencies_fail_closed(exclusion, change):
    raw, record, proof = exclusion
    if change == "record":
        record.write_text("{}")
    elif change == "proof":
        proof.write_text("{}")
    elif change == "missing_proof":
        proof.unlink()
    elif change == "split":
        raw["frozen_splits"]["AMI"]["train"].append("new-parent")
    elif change == "permission":
        raw["permissions"][0]["reviewer"] = "changed-reviewer"
    else:
        raw["sources"][0]["version"] = "new-version"
        raw["permissions"][0]["version"] = "new-version"
    with pytest.raises(PreparationError, match="source disposition"):
        validate_source_dispositions(parse_data_preparation_spec(raw))


def test_exclusion_cannot_omit_a_frozen_parent(exclusion):
    raw, record, _ = exclusion
    value = read_json(record)
    value["excluded_train_parent_ids"] = []
    write_json(record, value)
    raw["sources"][0]["disposition_evidence"]["sha256"] = sha256_file(record)
    with pytest.raises(PreparationError, match="exact excluded train parent"):
        validate_source_dispositions(parse_data_preparation_spec(raw))
