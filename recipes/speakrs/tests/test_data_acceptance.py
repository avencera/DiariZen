#!/usr/bin/env python3

"""Phase-4 acceptance failure tests for the data owners."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from recipes.speakrs.large.acceptance import (  # noqa: E402
    QaPath,
    admit_profiles,
    admit_source_after_automatic_qa,
    assert_policy_frozen_before_measurement,
    assert_split_isolation,
    audit_selection_hours_capacity,
    build_minimal_package,
    count_hours,
    evaluate_existing_human_activity_labels,
    evaluate_human_reference_pilot,
    load_qa_policy,
    parse_rttm,
    reject_automatic_self_agreement,
    reject_duplicate_channel_accounting,
    reject_target_clipping,
    select_qa_path,
    verify_decoded_identity,
    verify_expected_rttm,
)
from recipes.speakrs.large.contracts import (  # noqa: E402
    DATA_PREPARATION_SCHEMA,
    LabelMethod,
    SourceMembership,
    is_placeholder_hash,
    licence_from_permission,
    parse_data_preparation_spec,
    parse_old_release_identity,
    parse_permission_record,
    parse_spec,
    require_content_hash,
)
from recipes.speakrs.large.errors import ContractError, PreparationError, UnresolvedInputError  # noqa: E402
from recipes.speakrs.large.prepare import _icsi_split, seal_release  # noqa: E402


SPEC_PATH = REPO / "recipes" / "speakrs" / "conf" / "large_cc_v1.json"
QA_POLICY = REPO / "recipes" / "speakrs" / "conf" / "label-qa-policy.json"
LARGE_RUN = REPO / "recipes" / "speakrs" / "large_run.py"


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _cc_uses() -> dict[str, dict[str, str]]:
    return {
        name: {"decision": "permitted", "clause_ref": "CC-BY-4.0 §2(a)(1)"}
        for name in ("commercial_training", "model_distribution", "derived_labels", "private_tigris")
    }


def _permission_payload(source: str = "AMI", terms_class: str = "cc-by") -> dict:
    return {
        "record_id": f"{source}-perm",
        "source": source,
        "version": "official",
        "terms_url": "https://example.invalid/terms",
        "terms_sha256": _h(f"{source}-terms"),
        "terms_class": terms_class,
        "recipient": "Speakrs",
        "access_state": "obtained",
        "uses": _cc_uses(),
        "reviewer": "contracts",
    }


def _source_payload(source: str = "AMI", membership: str = "pending") -> dict:
    return {
        "name": source,
        "version": "official",
        "membership": membership,
        "permission_id": f"{source}-perm",
        "permission_state": "permitted" if membership == "accepted" else "unresolved",
        "missing_action": None if membership != "pending" else "authorize a human reviewer",
        "private_evidence_id": f"ev-{source}",
    }


def _data_spec_payload(tmp: Path, membership: str = "pending") -> dict:
    return {
        "schema": DATA_PREPARATION_SCHEMA,
        "schema_version": 1,
        "release_id": "speakrs-train-v1",
        "sources": [_source_payload("AMI", membership)],
        "permissions": [_permission_payload("AMI")],
        "qa_policy_path": str(QA_POLICY),
        "frozen_splits": {"AMI": {"test": ["IS1009a"], "dev": ["ES2011a"], "train": ["ES2002a"]}},
        "profiles": {
            "sample_rate": 16000,
            "local_slots": 4,
            "chunk_seconds": [8, 16],
            "max_overlap": [2, 4],
            "output_frames_8": 399,
            "rf_duration": 0.025,
            "rf_step": 0.020,
        },
        "disk": {
            "staging_root": str(tmp / "staging"),
            "cache_root": str(tmp / "cache"),
            "max_staging_bytes": 1024 * 1024,
            "max_cache_bytes": 1024 * 1024,
            "free_space_reserve_bytes": 1,
            "concurrency": 1,
        },
        "r2": {
            "provider": "r2",
            "endpoint": "https://example.invalid.r2.cloudflarestorage.com",
            "bucket": "praveen",
            "prefix": "datasets/diarization-data-verification",
            "credential_reference": "wrangler",
        },
        "permission_records_path": str(tmp / "permissions.json"),
        "private_evidence_root": str(tmp / "evidence"),
    }


def _rttm(recording: str, *turns: tuple[str, float, float]) -> str:
    lines = []
    for speaker, start, dur in turns:
        lines.append(f"SPEAKER {recording} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA>")
    return "\n".join(lines) + "\n"


class PlaceholderAndLabelTest(unittest.TestCase):
    def test_placeholder_hashes_cannot_seal(self):
        self.assertTrue(is_placeholder_hash("a" * 64))
        self.assertTrue(is_placeholder_hash("0" * 64))
        self.assertFalse(is_placeholder_hash(_h("real")))
        with self.assertRaises(ContractError):
            require_content_hash("a" * 64, "audio")
        spec = parse_spec(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
        recordings = []
        splits = {}
        for corpus in spec.required_corpora:
            splits[corpus] = {"train": [f"{corpus}-0"], "dev": [f"{corpus}-1"], "test": [f"{corpus}-2"]}
            for split, parent in (("train", f"{corpus}-0"), ("dev", f"{corpus}-1"), ("test", f"{corpus}-2")):
                recordings.append(
                    {
                        "recording_id": parent,
                        "parent_id": parent,
                        "corpus": corpus,
                        "split": split,
                        "device_view": "canonical",
                        "label_tier": "gold",
                        "licence": "accepted_cc",
                        "audio_sha256": "a" * 64,
                        "label_sha256": "b" * 64,
                        "sample_count": 16000,
                        "rejected": False,
                        "rejection_reason": None,
                        "language": "en",
                        "transformations": [],
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(PreparationError) as raised:
                seal_release(spec, Path(temporary) / "release", recordings, splits, [])
        self.assertIn("placeholder hashes", str(raised.exception))

    def test_empty_expected_rttm_cannot_seal(self):
        with self.assertRaises(PreparationError) as raised:
            verify_expected_rttm([], duration=8.0, recording_id="rec")
        self.assertIn("empty expected RTTM", str(raised.exception))

    def test_wrong_decoded_duration_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.wav"
            sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
            identity_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(PreparationError) as raised:
                verify_decoded_identity(
                    path,
                    expected_sha256=identity_hash,
                    expected_sample_count=32000,
                )
            self.assertIn("wrong decoded duration", str(raised.exception))
            with self.assertRaises(PreparationError):
                verify_decoded_identity(
                    path,
                    expected_sha256=_h("other"),
                    expected_sample_count=16000,
                )

    def test_speaker_identity_time_bounds_and_redaction(self):
        with self.assertRaises(PreparationError) as raised:
            verify_expected_rttm(
                parse_rttm(_rttm("rec", ("<NA>", 0.0, 1.0))),
                duration=2.0,
                recording_id="rec",
            )
        self.assertIn("wrong speaker identity", str(raised.exception))
        with self.assertRaises(PreparationError) as raised:
            verify_expected_rttm(
                parse_rttm(_rttm("rec", ("spk1", 0.0, 9.0))),
                duration=2.0,
                recording_id="rec",
            )
        self.assertIn("time bounds", str(raised.exception))
        with self.assertRaises(PreparationError) as raised:
            verify_expected_rttm(
                parse_rttm(_rttm("rec", ("spk1", 0.5, 1.0))),
                duration=2.0,
                recording_id="rec",
                redacted_regions=((0.4, 0.8),),
            )
        self.assertIn("redaction", str(raised.exception))


class PermissionTest(unittest.TestCase):
    def test_custom_agreement_cannot_be_relabeled_accepted_cc(self):
        payload = _permission_payload("SSSD", "custom")
        payload["uses"]["commercial_training"] = {
            "decision": "permitted",
            "clause_ref": "accepted_cc via adapter",
        }
        with self.assertRaises(ContractError) as raised:
            parse_permission_record(payload)
        self.assertIn("accepted CC", str(raised.exception))

    def test_adapter_cannot_self_approve(self):
        payload = _permission_payload()
        payload["self_approved"] = True
        with self.assertRaises(ContractError):
            parse_permission_record(payload)
        source = _source_payload("AMI", "accepted")
        source["licence"] = "accepted_cc"
        with tempfile.TemporaryDirectory() as temporary:
            spec_payload = _data_spec_payload(Path(temporary), "accepted")
            spec_payload["sources"] = [source]
            with self.assertRaises(ContractError):
                parse_data_preparation_spec(spec_payload)

    def test_data_spec_rejects_training_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _data_spec_payload(Path(temporary))
            payload["budget"] = {"total_usd": 150}
            with self.assertRaises(ContractError):
                parse_data_preparation_spec(payload)

    def test_data_spec_rejects_retired_tigris_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = _data_spec_payload(Path(temporary))
            payload["tigris"] = payload.pop("r2")
            with self.assertRaises(ContractError) as raised:
                parse_data_preparation_spec(payload)
            self.assertIn("r2", str(raised.exception))

    def test_old_identities_remain_valid_only_under_original_schema(self):
        old = parse_spec(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
        self.assertEqual(old.run_id, "large_cc_v1")
        with self.assertRaises(ContractError):
            parse_old_release_identity({"schema": DATA_PREPARATION_SCHEMA, "schema_version": 1})
        restored = parse_old_release_identity(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
        self.assertEqual(restored["run_id"], "large_cc_v1")
        custom = parse_permission_record(_permission_payload("SSSD", "custom"))
        self.assertNotEqual(licence_from_permission(custom).value, "accepted_cc")


class SplitHourCapacityTest(unittest.TestCase):
    def test_source_speaker_time_leakage_fails(self):
        with self.assertRaises(PreparationError) as raised:
            assert_split_isolation({"AMI": {"train": ["a"], "dev": [], "test": ["a"]}})
        self.assertIn("leakage", str(raised.exception))
        with self.assertRaises(PreparationError):
            assert_split_isolation(
                {"AMI": {"train": ["a"], "dev": ["b"], "test": ["c"]}},
                speaker_graph={"a": "spk", "c": "spk"},
            )

    def test_duplicate_channel_version_accounting_fails(self):
        with self.assertRaises(PreparationError) as raised:
            reject_duplicate_channel_accounting(
                [
                    {"source": "AMI", "parent_id": "ES2002a", "device_view": "array1"},
                    {"source": "AMI", "parent_id": "ES2002a", "device_view": "headset"},
                ]
            )
        self.assertIn("duplicate channel", str(raised.exception))

    def test_overlap4_does_not_qualify_overlap2(self):
        rttm = _rttm("rec", ("a", 0.0, 40.0), ("b", 0.0, 40.0), ("c", 0.0, 40.0))
        reports = admit_profiles(parse_rttm(rttm), duration=40.0)
        by_key = {(item.chunk_seconds, item.max_overlap): item for item in reports}
        self.assertTrue(by_key[(8, 4)].admitted)
        self.assertFalse(by_key[(8, 2)].admitted)
        self.assertTrue(by_key[(16, 4)].admitted)
        self.assertFalse(by_key[(16, 2)].admitted)

    def test_invalid_target_clipping_fails(self):
        original = parse_rttm(_rttm("rec", ("a", 0.0, 1.0), ("b", 0.0, 1.0)))
        clipped = parse_rttm(_rttm("rec", ("a", 0.0, 1.0)))
        with self.assertRaises(PreparationError) as raised:
            reject_target_clipping(original, clipped)
        self.assertIn("clipping", str(raised.exception))

    def test_audit_hours_capacity_does_not_accept(self):
        rttm = _rttm("train1", ("a", 0.0, 40.0), ("b", 0.0, 40.0), ("c", 0.0, 40.0)) + _rttm("test1", ("a", 0.0, 4.0))
        report = audit_selection_hours_capacity(
            source="AMI",
            splits={"train": ["train1"], "dev": [], "test": ["test1"]},
            duration_by_recording={"train1": 40.0, "test1": 4.0},
            intervals=parse_rttm(rttm),
        )
        self.assertFalse(report["accepted"])
        self.assertEqual(report["membership"], "pending")
        self.assertAlmostEqual(report["hours"]["timeline_hours"], 40.0 / 3600.0)
        self.assertAlmostEqual(report["hours"]["heldout_hours"], 4.0 / 3600.0)
        overlap2 = [item for item in report["capacity"] if item["max_overlap"] == 2][0]
        overlap4 = [item for item in report["capacity"] if item["max_overlap"] == 4][0]
        self.assertTrue(overlap4["admitted"])
        self.assertFalse(overlap2["admitted"])

    def test_hours_are_counted_separately(self):
        intervals = parse_rttm(_rttm("rec", ("a", 0.0, 1800.0), ("b", 0.0, 1800.0)))
        counts = count_hours(intervals, duration_by_recording={"rec": 1800.0, "held": 3600.0}, heldout_ids=["held"])
        self.assertAlmostEqual(counts.timeline_hours, 0.5)
        self.assertAlmostEqual(counts.speaker_hours, 1.0)
        self.assertAlmostEqual(counts.heldout_hours, 1.0)


class HumanQaTest(unittest.TestCase):
    def test_self_agreement_cannot_satisfy_qa(self):
        with self.assertRaises(PreparationError):
            reject_automatic_self_agreement(LabelMethod.TRANSCRIPT_VAD_SELF_AGREEMENT)
        policy = load_qa_policy(QA_POLICY)
        with self.assertRaises(PreparationError) as raised:
            evaluate_human_reference_pilot(
                {
                    "label_method": "transcript-vad-self-agreement",
                    "reviewer_id": "bot",
                    "sign_off": True,
                    "windows": [],
                    "miss_fa_overall": 0.0,
                    "miss_fa_by_stratum": {},
                },
                policy,
            )
        self.assertIn("self-agreement", str(raised.exception))

    def test_existing_human_labels_do_not_need_a_reviewer(self):
        self.assertEqual(
            select_qa_path(label_method=LabelMethod.HUMAN_GOLD),
            QaPath.EXISTING_HUMAN_ACTIVITY,
        )
        self.assertEqual(
            select_qa_path(label_method=LabelMethod.CHANNEL_DERIVED),
            QaPath.HUMAN_REFERENCE_PILOT,
        )
        self.assertEqual(
            select_qa_path(label_method=LabelMethod.HUMAN_GOLD, unresolved_semantic_defect=True),
            QaPath.HUMAN_REFERENCE_PILOT,
        )
        qa = evaluate_existing_human_activity_labels(
            {
                "qa_path": QaPath.EXISTING_HUMAN_ACTIVITY,
                "label_method": LabelMethod.HUMAN_GOLD.value,
                "source_version": "official",
                "annotation_provenance": "published-rttm",
                "channel_mapping_verified": True,
                "time_transform_verified": True,
                "bounds_complete": True,
                "unknown_regions_handled": True,
                "split_isolated": True,
                "admitted_profiles": [{"chunk_seconds": 8, "max_overlap": 4, "admitted": True}],
            }
        )
        self.assertTrue(qa["admitted"])
        self.assertFalse(qa["reviewer_required"])
        with self.assertRaises(PreparationError):
            evaluate_existing_human_activity_labels(
                {
                    "qa_path": QaPath.EXISTING_HUMAN_ACTIVITY,
                    "label_method": LabelMethod.HUMAN_GOLD.value,
                    "reviewer_id": "fabricated",
                    "source_version": "official",
                    "annotation_provenance": "published-rttm",
                    "channel_mapping_verified": True,
                    "time_transform_verified": True,
                    "bounds_complete": True,
                    "unknown_regions_handled": True,
                    "split_isolated": True,
                }
            )
        with self.assertRaises(PreparationError):
            evaluate_existing_human_activity_labels(
                {
                    "qa_path": QaPath.EXISTING_HUMAN_ACTIVITY,
                    "label_method": LabelMethod.CHANNEL_DERIVED.value,
                    "source_version": "x",
                    "annotation_provenance": "channels",
                    "channel_mapping_verified": True,
                    "time_transform_verified": True,
                    "bounds_complete": True,
                    "unknown_regions_handled": True,
                    "split_isolated": True,
                }
            )
        decision = admit_source_after_automatic_qa(
            permission_permitted=True,
            qa=qa,
            capacity=[
                {"chunk_seconds": 8, "max_overlap": 2, "admitted": False, "loss_fraction": 0.02},
                {"chunk_seconds": 8, "max_overlap": 4, "admitted": True, "loss_fraction": 0.0},
            ],
        )
        self.assertEqual(decision["membership"], "accepted")
        self.assertFalse(decision["fully_training_ready"])
        self.assertEqual(len(decision["admitted_profiles"]), 1)

    def test_missing_reviewer_is_unresolved(self):
        policy = load_qa_policy(QA_POLICY)
        with self.assertRaises(UnresolvedInputError):
            evaluate_human_reference_pilot(
                {
                    "label_method": "human-gold",
                    "reviewer_id": "",
                    "sign_off": False,
                    "windows": [],
                    "miss_fa_overall": 0.0,
                    "miss_fa_by_stratum": {},
                },
                policy,
            )

    def test_policy_must_be_frozen_before_measurement(self):
        policy = load_qa_policy(QA_POLICY)
        with self.assertRaises(PreparationError):
            assert_policy_frozen_before_measurement(policy, {"policy_sha256": "nope", "measured_at": "2026-09-08"})

    def test_only_accepted_train_artifacts_enter_minimal_package(self):
        with self.assertRaises(PreparationError):
            build_minimal_package(
                [{"purpose": "test-audio", "split": "test", "membership": SourceMembership.ACCEPTED.value}]
            )
        with self.assertRaises(PreparationError):
            build_minimal_package(
                [{"purpose": "train-audio", "split": "train", "membership": SourceMembership.PENDING.value}]
            )
        kept = build_minimal_package(
            [{"purpose": "train-audio", "split": "train", "membership": SourceMembership.ACCEPTED.value}]
        )
        self.assertEqual(len(kept), 1)

    def test_icsi_empty_frozen_test_fails(self):
        with self.assertRaises(PreparationError) as raised:
            _icsi_split(["Bmr001", "Bmr002"], frozen_test=())
        self.assertIn("frozen-test", str(raised.exception))


class DataPlanCliTest(unittest.TestCase):
    def test_data_plan_twice_agrees_and_does_not_transfer(self):
        import subprocess

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "staging").mkdir()
            (root / "cache").mkdir()
            config = root / "data-preparation.json"
            config.write_text(json.dumps(_data_spec_payload(root), indent=2) + "\n", encoding="utf-8")
            outputs = []
            for name in ("plan1.json", "plan2.json"):
                output = root / name
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(LARGE_RUN),
                        "data",
                        "plan",
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=str(REPO),
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertFalse(payload["bulk_transfer"])
                self.assertFalse(payload["downloads"])
                self.assertIn("source_membership", payload)
                self.assertIn("access_states", payload)
                self.assertIn("destination", payload)
                self.assertIn("disk_limits", payload)
                self.assertIn("required_inputs", payload)
                self.assertNotIn("AWS_SECRET", output.read_text(encoding="utf-8"))
                outputs.append(payload)
            self.assertEqual(outputs[0], outputs[1])
            verify = subprocess.run(
                [
                    sys.executable,
                    str(LARGE_RUN),
                    "data",
                    "verify",
                    "--release",
                    str(root / "release"),
                    "--config",
                    str(config),
                    "--output",
                    str(root / "acceptance.json"),
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=str(REPO),
            )
            self.assertEqual(verify.returncode, 2, verify.stdout + verify.stderr)


if __name__ == "__main__":
    unittest.main()
