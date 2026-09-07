"""Protect selection closure and evidence invalidation at the upload boundary"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recipes.speakrs.large.contracts import parse_data_preparation_spec
from recipes.speakrs.large.data import _load_verified_selection, upload_data, verify_data
from recipes.speakrs.large.errors import PreparationError, UnresolvedInputError
from recipes.speakrs.large.hashing import sha256_file, sha256_json
from recipes.speakrs.large.jsonio import write_json
from recipes.speakrs.large.storage import MemoryBackend
from recipes.speakrs.tests.test_data_acceptance import _data_spec_payload


def _reference(path):
    return {"path": str(path), "sha256": sha256_file(path)}


@pytest.fixture
def selection(tmp_path):
    raw = _data_spec_payload(tmp_path)
    release = tmp_path / "release"
    release.mkdir()
    evidence = {}
    for name in (
        "terms",
        "annotation_provenance",
        "channel_mapping",
        "time_transform",
        "coverage",
        "split_provenance",
    ):
        path = release / f"{name}.json"
        write_json(path, {"check": name, "fixture": True})
        evidence[name] = _reference(path)
    raw["permissions"][0]["terms_sha256"] = evidence["terms"]["sha256"]
    raw["sources"][0]["permission_state"] = "permitted"
    spec = parse_data_preparation_spec(raw)
    audio = release / "ES2002a.flac"
    sf.write(audio, np.sin(np.arange(640000) * 0.02) * 0.1, 16000, subtype="PCM_16")
    label = release / "ES2002a.rttm"
    label.write_text("SPEAKER ES2002a 1 1.0 38.0 <NA> <NA> speaker-1 <NA> <NA>\n")
    uem = release / "ES2002a.uem"
    uem.write_text("ES2002a 1 0.0 40.0\n")
    policy = json.loads(spec.qa_policy_path.read_text())
    manifest = {
        "schema": "speakrs-selection-v1",
        "source": "AMI",
        "version": "official",
        "evidence": evidence,
        "splits": raw["frozen_splits"]["AMI"],
        "speaker_graph": {"ES2002a": ["speaker-1"], "ES2011a": ["speaker-2"], "IS1009a": ["speaker-3"]},
        "qa": {
            "qa_path": "existing-human-activity",
            "label_method": "human-gold",
            "source_version": "official",
            "annotation_provenance": "published activity",
            "channel_mapping_verified": True,
            "time_transform_verified": True,
            "bounds_complete": True,
            "unknown_regions_handled": True,
            "split_isolated": True,
            "policy_sha256": sha256_json(policy),
            "measured_at": "2026-09-07T20:00:00Z",
        },
        "recordings": [
            {
                "recording_id": "ES2002a",
                "audio": _reference(audio),
                "rttm": _reference(label),
                "uem": _reference(uem),
                "sample_count": 640000,
                "known_speakers": ["speaker-1"],
            }
        ],
    }
    write_json(release / "selection.json", manifest)
    return spec, raw, release, manifest


def test_membership_alone_cannot_verify(tmp_path):
    spec = parse_data_preparation_spec(_data_spec_payload(tmp_path, "accepted"))
    with pytest.raises(UnresolvedInputError, match="selection.json"):
        verify_data(spec, tmp_path, tmp_path / "acceptance.json")


def test_verified_existing_activity_needs_no_new_reviewer(selection):
    spec, _, release, _ = selection
    result = verify_data(spec, release, release / "acceptance.json")
    assert result["state"] == "qa-split-verified-capacity-measured"
    assert result["training_ready"] is False
    assert result["complete_release"] is False
    _load_verified_selection(spec, release)


def test_selected_parent_closure_rejects_omission(selection):
    spec, _, release, manifest = selection
    manifest["recordings"] = []
    write_json(release / "selection.json", manifest)
    with pytest.raises(PreparationError, match="parent inventory"):
        verify_data(spec, release, None)


def test_declared_batch_retains_full_required_parent_inventory(selection):
    _, raw, release, manifest = selection
    raw["frozen_splits"]["AMI"]["train"].append("ES2002b")
    manifest["splits"] = raw["frozen_splits"]["AMI"]
    manifest["speaker_graph"]["ES2002b"] = ["speaker-1"]
    manifest["selected_parent_ids"] = ["ES2002a"]
    write_json(release / "selection.json", manifest)
    result = verify_data(parse_data_preparation_spec(raw), release, release / "acceptance.json")
    assert result["selected_parent_ids"] == ["ES2002a"]
    assert result["required_parent_ids"] == ["ES2002a", "ES2002b"]
    assert result["complete_release"] is False


def test_disk_budget_change_does_not_invalidate_accepted_content(selection):
    spec, raw, release, _ = selection
    verify_data(spec, release, release / "acceptance.json")
    raw["disk"]["max_cache_bytes"] *= 2
    _load_verified_selection(parse_data_preparation_spec(raw), release)


def test_hard_parent_is_storable_without_provisional_profile_admission(selection):
    _, raw, release, manifest = selection
    speakers = [f"hard-{number}" for number in range(5)]
    row = manifest["recordings"][0]
    labels = Path(row["rttm"]["path"])
    labels.write_text("".join(f"SPEAKER ES2002a 1 1 38 <NA> <NA> {speaker} <NA> <NA>\n" for speaker in speakers))
    row["rttm"] = _reference(labels)
    row["known_speakers"] = speakers
    manifest["speaker_graph"]["ES2002a"] = speakers
    write_json(release / "selection.json", manifest)
    raw["disk"]["max_cache_bytes"] = 16 * 1024 * 1024
    spec = parse_data_preparation_spec(raw)
    verified = verify_data(spec, release, release / "acceptance.json")
    assert verified["provisional_profiles"] == []
    assert verified["training_ready"] is False
    assert len(verified["capacity"]) == 4
    assert all(profile["loss_fraction"] > 0.005 for profile in verified["capacity"])
    uploaded = upload_data(spec, release, release / "upload.json", backend=MemoryBackend())
    assert uploaded["ok"] is True


@pytest.mark.parametrize("changed", ["configuration", "manifest", "evidence", "audio", "receipt"])
def test_acceptance_invalidation(selection, changed):
    spec, raw, release, manifest = selection
    verify_data(spec, release, release / "acceptance.json")
    if changed == "configuration":
        raw["release_id"] = "changed-release"
        spec = parse_data_preparation_spec(raw)
    elif changed == "manifest":
        manifest["qa"]["annotation_provenance"] = "changed provenance"
        write_json(release / "selection.json", manifest)
    elif changed == "evidence":
        Path(manifest["evidence"]["coverage"]["path"]).write_text("changed coverage")
    elif changed == "audio":
        Path(manifest["recordings"][0]["audio"]["path"]).write_bytes(b"changed audio")
    else:
        receipt = json.loads((release / "acceptance.json").read_text())
        receipt["hours"]["timeline_hours"] = 999.0
        write_json(release / "acceptance.json", receipt)
    with pytest.raises(PreparationError):
        _load_verified_selection(spec, release)
