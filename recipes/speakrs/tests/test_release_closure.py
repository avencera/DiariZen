"""Protect the final remote release and handoff closure."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import recipes.speakrs.large.data as data_module
from recipes.speakrs.large.contracts import data_spec_to_json, parse_data_preparation_spec
from recipes.speakrs.large.data import (
    commit_remote_release,
    dispatch_data,
    handoff_data,
    restore_check,
    upload_data,
    verify_data,
    verify_remote,
)
from recipes.speakrs.large.errors import PreparationError, UnresolvedInputError
from recipes.speakrs.large.hashing import sha256_file
from recipes.speakrs.large.jsonio import write_json
from recipes.speakrs.large.release_closure import (
    aggregate_capacity,
    common_admitted_profiles,
    parse_capacity_manifest,
)
from recipes.speakrs.large.storage import MemoryBackend
from recipes.speakrs.tests.test_selection_verification import selection as selection_fixture


@pytest.fixture
def committed_release(tmp_path):
    _, raw, release, _ = selection_fixture.__wrapped__(tmp_path)
    raw["disk"]["max_cache_bytes"] = 16 * 1024 * 1024
    raw["disk"]["max_staging_bytes"] = 16 * 1024 * 1024
    raw["disk"]["staging_root"] = str(release)
    spec = parse_data_preparation_spec(raw)
    backend = MemoryBackend()
    verify_data(spec, release, release / "acceptance.json")
    upload_data(spec, release, release / "upload.json", backend=backend)
    verify_remote(
        spec,
        release / "upload.json",
        release / "remote.json",
        backend=backend,
        label_policy_id="human-gold-v1",
        split_id="published-v1",
    )
    config = release / "data-preparation.json"
    write_json(config, data_spec_to_json(spec))
    seal_receipt = dispatch_data(
        SimpleNamespace(
            data_command="commit-release",
            config=config,
            release=release,
            receipts=release / "remote.json",
            output=release / "remote-release.json",
        ),
        backend=backend,
    )
    batch_restore = restore_check(spec, release / "remote.json", release / "batch-restore.json", backend=backend)
    return spec, raw, release, backend, config, seal_receipt["seal"], batch_restore


def test_pending_membership_does_not_override_a_verified_remote_selection(committed_release):
    _, _, _, _, _, seal, _ = committed_release
    assert seal["state"] == "committed"


def test_batch_restore_cannot_reach_handoff(committed_release):
    spec, _, release, backend, _, _, _ = committed_release
    with pytest.raises(PreparationError, match="generated final restore aggregate"):
        handoff_data(spec, release, release / "remote.json", release / "handoff.json", backend=backend)


def test_handoff_rejects_unbound_restore_bundle(committed_release):
    spec, _, release, backend, config, seal, batch_restore = committed_release
    write_json(
        release / "restore.json",
        {"seal": seal, "restore_receipts": [batch_restore]},
    )
    with pytest.raises(PreparationError, match="generated final restore aggregate"):
        dispatch_data(
            SimpleNamespace(
                data_command="handoff",
                config=config,
                release=release,
                receipt=release / "restore.json",
                output=release / "handoff.json",
            ),
            backend=backend,
        )


def _final_restore_args(config, release, seal, restores, output):
    return SimpleNamespace(
        data_command="final-restore",
        config=config,
        release=release,
        seal=seal,
        restores=restores,
        output=output,
    )


def test_final_restore_dispatch_reuses_actual_batch_proof_and_handoff_accepts_index(committed_release):
    spec, _, release, backend, config, seal, batch_restore = committed_release
    aggregate_path = release / "final-restore.json"
    result = dispatch_data(
        _final_restore_args(
            config, release, release / "remote-release.json", release / "batch-restore.json", aggregate_path
        ),
        backend=backend,
    )

    assert result["schema"] == "speakrs-final-restore-v1"
    assert result["complete_release"] is True
    assert result["release_sha256"] == seal["release_sha256"]
    assert result["batches"][0]["reused"] is True
    assert result["restore_receipts"][0]["batch_sha256"] == batch_restore["batch_sha256"]
    assert result["restore_receipts"][0]["objects"] == batch_restore["objects"]
    assert "eviction_receipt_sha256" not in result["batches"][0]

    handoff = handoff_data(spec, release, aggregate_path, release / "final-handoff.json", backend=backend)
    assert handoff["complete_release"] is True


def test_final_restore_validates_each_release_object_once(committed_release):
    from collections import Counter
    from unittest import mock

    _, _, release, backend, config, seal, _ = committed_release
    object_keys = {item["key"] for item in seal["objects"]} | {seal["marker"]["key"]}
    with (
        mock.patch.object(backend, "iter_bytes", wraps=backend.iter_bytes) as iter_bytes,
        mock.patch.object(backend, "anonymous_list", wraps=backend.anonymous_list) as anonymous_list,
    ):
        dispatch_data(
            _final_restore_args(
                config,
                release,
                release / "remote-release.json",
                release / "batch-restore.json",
                release / "single-read-final-restore.json",
            ),
            backend=backend,
        )

    reads = Counter(call.args[0] for call in iter_bytes.call_args_list if call.args[0] in object_keys)
    assert reads == Counter(dict.fromkeys(object_keys, 1))
    anonymous_list.assert_called_once()


def test_final_restore_reuses_legacy_v1_proof_without_an_incarnation(committed_release):
    spec, _, release, backend, config, _, batch_restore = committed_release
    legacy = {key: value for key, value in batch_restore.items() if key != "incarnation_id"}
    legacy_path = release / "legacy-v1-restore.json"
    write_json(legacy_path, legacy)
    original_bytes = legacy_path.read_bytes()
    aggregate_path = release / "legacy-final-restore.json"
    result = dispatch_data(
        _final_restore_args(config, release, release / "remote-release.json", legacy_path, aggregate_path),
        backend=backend,
    )
    assert result["batches"][0]["reused"] is True
    assert result["restore_receipts"][0] == json.loads(original_bytes)
    assert legacy_path.read_bytes() == original_bytes
    assert handoff_data(spec, release, aggregate_path, release / "legacy-handoff.json", backend=backend)[
        "training_ready"
    ]


def test_final_restore_refreshes_missing_and_stale_batch_proofs(committed_release):
    _, _, release, backend, config, _, batch_restore = committed_release

    stale = {**batch_restore, "samples": [{"finite": True}]}
    stale_path = release / "stale-restore.json"
    write_json(stale_path, stale)
    stale_result = dispatch_data(
        _final_restore_args(
            config, release, release / "remote-release.json", stale_path, release / "fresh-stale.json"
        ),
        backend=backend,
    )
    assert stale_result["batches"][0]["reused"] is False
    assert all(sample["finite"] is True for sample in stale_result["restore_receipts"][0]["samples"])
    durable_restore = release / "final-restore-batches" / batch_restore["batch_sha256"] / "restore.json"
    durable_eviction = durable_restore.with_name("eviction.json")
    assert durable_restore.is_file()
    assert durable_eviction.is_file()
    assert (
        stale_result["batches"][0]["eviction_receipt_sha256"]
        == hashlib.sha256(durable_eviction.read_bytes()).hexdigest()
    )
    assert all(not Path(item["restored_path"]).exists() for item in stale_result["restore_receipts"][0]["objects"])

    missing_result = dispatch_data(
        _final_restore_args(
            config,
            release,
            release / "remote-release.json",
            release / "does-not-exist.json",
            release / "fresh-missing.json",
        ),
        backend=backend,
    )
    assert missing_result["batches"][0]["reused"] is False


def test_final_restore_rejects_a_receipt_from_another_release(committed_release):
    _, _, release, backend, config, _, batch_restore = committed_release
    mixed = {**batch_restore, "batch_sha256": hashlib.sha256(b"other-release").hexdigest()}
    mixed_path = release / "mixed-restore.json"
    write_json(mixed_path, mixed)

    with pytest.raises(PreparationError, match="different final release"):
        dispatch_data(
            _final_restore_args(config, release, release / "remote-release.json", mixed_path, release / "mixed.json"),
            backend=backend,
        )


def test_final_validator_identity_is_separate_from_batch_implementation(committed_release, monkeypatch):
    spec, _, release, backend, config, seal, _ = committed_release
    implementation = seal["release_identity"]["implementation_hashes"]
    assert "recipes/speakrs/large/release_closure.py" not in implementation
    assert seal["release_identity"]["closure_validator_sha256"] == data_module._closure_validator_sha256()
    aggregate_path = release / "final-restore.json"
    dispatch_data(
        _final_restore_args(
            config, release, release / "remote-release.json", release / "batch-restore.json", aggregate_path
        ),
        backend=backend,
    )
    monkeypatch.setattr(data_module, "_closure_validator_sha256", lambda: "f" * 64)
    with pytest.raises(PreparationError, match="identity"):
        handoff_data(
            spec,
            release,
            aggregate_path,
            release / "stale-validator-handoff.json",
            backend=backend,
        )


def test_handoff_rejects_a_malformed_final_restore_aggregate(committed_release):
    spec, _, release, backend, config, _, _ = committed_release
    aggregate_path = release / "final-restore.json"
    dispatch_data(
        _final_restore_args(
            config, release, release / "remote-release.json", release / "batch-restore.json", aggregate_path
        ),
        backend=backend,
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["objects"] = aggregate["objects"][1:]
    aggregate["aggregate_sha256"] = data_module.sha256_json(
        {key: value for key, value in aggregate.items() if key != "aggregate_sha256"}
    )
    write_json(release / "malformed-final-restore.json", aggregate)
    with pytest.raises(PreparationError, match="object inventory"):
        handoff_data(
            spec,
            release,
            release / "malformed-final-restore.json",
            release / "malformed-handoff.json",
            backend=backend,
        )


def test_final_restore_resumes_durable_cleanup_after_interruption(committed_release, monkeypatch):
    spec, _, release, backend, config, _, batch_restore = committed_release
    stale = {**batch_restore, "samples": [{"finite": True}]}
    stale_path = release / "interrupted-stale.json"
    write_json(stale_path, stale)
    real_evict = data_module.evict_data
    interrupted = {"value": False}

    def fail_once(*args, **kwargs):
        if not interrupted["value"]:
            interrupted["value"] = True
            raise PreparationError("simulated final restore interruption")
        return real_evict(*args, **kwargs)

    monkeypatch.setattr(data_module, "evict_data", fail_once)
    with pytest.raises(PreparationError, match="simulated final restore interruption"):
        dispatch_data(
            _final_restore_args(
                config,
                release,
                release / "remote-release.json",
                stale_path,
                release / "interrupted.json",
            ),
            backend=backend,
        )

    durable_restore = release / "final-restore-batches" / batch_restore["batch_sha256"] / "restore.json"
    durable_eviction = durable_restore.with_name("eviction.json")
    assert durable_restore.is_file()
    assert not durable_eviction.exists()

    resumed = data_module.final_restore_data(
        spec,
        release,
        release / "remote-release.json",
        release / "resumed.json",
        restore_receipts_path=stale_path,
        backend=backend,
    )
    assert resumed["batches"][0]["reused"] is True
    assert durable_eviction.is_file()
    assert all(not Path(item["restored_path"]).exists() for item in resumed["restore_receipts"][0]["objects"])


def test_final_restore_new_incarnation_preserves_old_receipt_and_eviction(committed_release):
    spec, _, release, backend, config, _, _ = committed_release
    aggregate_path = release / "same-output.json"
    missing_receipts = release / "missing-batch-restore.json"
    first = dispatch_data(
        _final_restore_args(config, release, release / "remote-release.json", missing_receipts, aggregate_path),
        backend=backend,
    )
    batch_hash = first["batches"][0]["batch_sha256"]
    durable_root = release / "final-restore-batches" / batch_hash
    durable_restore = durable_root / "restore.json"
    old_digest = sha256_file(durable_restore)
    old_receipt = json.loads(durable_restore.read_text(encoding="utf-8"))
    old_eviction = durable_root / "eviction.json"
    old_eviction_receipt = json.loads(old_eviction.read_text(encoding="utf-8"))
    old_cache_paths = {Path(item["restored_path"]) for item in old_receipt["objects"]}
    assert old_receipt["incarnation_id"] == first["restore_receipts"][0]["incarnation_id"]
    assert all(not path.exists() for path in old_cache_paths)

    second = dispatch_data(
        _final_restore_args(config, release, release / "remote-release.json", missing_receipts, aggregate_path),
        backend=backend,
    )
    new_receipt = second["restore_receipts"][0]
    new_eviction_path = Path(second["batches"][0]["eviction_receipt_path"])
    archived_restore = durable_root / f"restore-{old_digest}.json"

    assert second["batches"][0]["reused"] is False
    assert new_receipt["incarnation_id"] != old_receipt["incarnation_id"]
    assert archived_restore.is_file()
    assert sha256_file(archived_restore) == old_digest
    assert json.loads(archived_restore.read_text(encoding="utf-8")) == old_receipt
    assert json.loads(old_eviction.read_text(encoding="utf-8")) == old_eviction_receipt
    assert old_eviction_receipt["restore_receipt_sha256"] == old_digest
    assert new_eviction_path != old_eviction
    assert new_eviction_path.is_file()
    assert all(not Path(item["restored_path"]).exists() for item in new_receipt["objects"])
    assert all(not path.exists() for path in old_cache_paths)


def _capacity_manifest(*, speaker_seconds=100.0, lost_seconds=0.0, failed=()):
    failed = set(failed)
    profiles = []
    for chunk_seconds, max_overlap, local_slots in ((8, 2, 4), (8, 4, 4), (16, 2, 4), (16, 4, 4)):
        key = (chunk_seconds, max_overlap, local_slots)
        loss = lost_seconds if key in failed else 0.0
        fraction = loss / speaker_seconds
        profiles.append(
            {
                "chunk_seconds": chunk_seconds,
                "max_overlap": max_overlap,
                "local_slots": local_slots,
                "speaker_seconds": speaker_seconds,
                "lost_seconds": loss,
                "loss_fraction": fraction,
                "admitted": fraction <= 0.005,
            }
        )
    return {
        "capacity": profiles,
        "admitted_profiles": [item for item in profiles if item["admitted"]],
    }


def test_capacity_closure_uses_weighted_totals_and_keeps_failed_profiles():
    large = parse_capacity_manifest(_capacity_manifest(speaker_seconds=1000.0), label="large")
    small = parse_capacity_manifest(
        _capacity_manifest(speaker_seconds=1.0, lost_seconds=0.006, failed=((8, 2, 4),)),
        label="small",
    )
    aggregate = aggregate_capacity((large, small), label="AMI")
    row = aggregate.as_list()[0]
    assert row["speaker_seconds"] == 1001.0
    assert row["lost_seconds"] == 0.006
    assert row["loss_fraction"] < 0.005
    assert row["admitted"] is True
    assert len(aggregate.as_list()) == 4


def test_common_capacity_is_an_intersection_not_a_union():
    all_profiles = parse_capacity_manifest(_capacity_manifest(), label="all")
    one_profile = parse_capacity_manifest(
        _capacity_manifest(lost_seconds=0.6, failed=((8, 2, 4), (16, 2, 4), (16, 4, 4))), label="one"
    )
    common = common_admitted_profiles({"AMI": all_profiles, "ICSI": one_profile})
    assert common == [{"chunk_seconds": 8, "max_overlap": 4, "local_slots": 4}]


def test_capacity_counters_must_be_integers():
    manifest = _capacity_manifest()
    manifest["capacity"][0]["chunks"] = 1.5
    with pytest.raises(PreparationError, match="metadata"):
        parse_capacity_manifest(manifest, label="fractional-counter")


def test_capacity_aggregate_rejects_nonfinite_totals():
    left = parse_capacity_manifest(_capacity_manifest(speaker_seconds=1e308), label="left")
    right = parse_capacity_manifest(_capacity_manifest(speaker_seconds=1e308), label="right")
    with pytest.raises(PreparationError, match="aggregated capacity totals"):
        aggregate_capacity((left, right), label="overflow")


def test_duplicate_batch_cannot_close_parent_union(committed_release):
    spec, _, release, backend, _, _, _ = committed_release
    write_json(
        release / "duplicate-receipts.json",
        [
            json.loads((release / "remote.json").read_text()),
            json.loads((release / "remote.json").read_text()),
        ],
    )
    with pytest.raises(PreparationError, match="duplicate committed batch"):
        commit_remote_release(
            spec,
            release,
            release / "duplicate-receipts.json",
            release / "duplicate.json",
            backend=backend,
        )


def test_missing_source_cannot_close_parent_union(committed_release):
    spec, raw, release, backend, _, _, _ = committed_release
    raw = {
        **raw,
        "permissions": [
            *raw["permissions"],
            {
                **raw["permissions"][0],
                "record_id": "OTHER-perm",
                "source": "OTHER",
                "terms_sha256": raw["permissions"][0]["terms_sha256"],
            },
        ],
        "sources": [
            *raw["sources"],
            {
                "name": "OTHER",
                "version": "official",
                "membership": "pending",
                "permission_id": "OTHER-perm",
                "permission_state": "permitted",
                "missing_action": "provide the source batch",
                "private_evidence_id": "ev-other",
            },
        ],
        "frozen_splits": {
            **raw["frozen_splits"],
            "OTHER": {
                "test": ["other-test"],
                "dev": ["other-dev"],
                "train": ["other-train"],
            },
        },
    }
    changed = parse_data_preparation_spec(raw)
    with pytest.raises(UnresolvedInputError, match="exact frozen train parent union"):
        commit_remote_release(changed, release, release / "remote.json", release / "missing.json", backend=backend)


@pytest.mark.parametrize("change", ["release_id", "permission"])
def test_stale_release_dependencies_cannot_close(committed_release, change):
    spec, raw, release, backend, _, _, _ = committed_release
    changed_raw = {**raw}
    if change == "release_id":
        changed_raw["release_id"] = "changed-release"
    else:
        changed_raw["permissions"] = [{**raw["permissions"][0], "terms_url": "https://example.invalid/changed-terms"}]
    changed = parse_data_preparation_spec(changed_raw)
    with pytest.raises(PreparationError, match="stale|differs"):
        commit_remote_release(changed, release, release / "remote.json", release / "stale.json", backend=backend)
