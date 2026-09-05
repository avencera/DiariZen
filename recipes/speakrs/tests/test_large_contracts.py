#!/usr/bin/env python3

"""Contract, sampler, budget, selection, and CLI tests for the Large recipe."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from recipes.speakrs.large.budget import BudgetLedger  # noqa: E402
from recipes.speakrs.large.contracts import (  # noqa: E402
    DEFAULT_BUDGET,
    LAUNCH_KIND,
    PREPARATION_KIND,
    QUALIFICATION_LEASE_KIND,
    STREAM_QUOTAS,
    LicenceDecision,
    RecordingRow,
    parse_kinded_lock,
    parse_spec,
)
from recipes.speakrs.large.controller import (  # noqa: E402
    LiveVastGuard,
    assert_worker_has_no_tokens,
    lease_from_offer,
    scrubbed_worker_environment,
)
from recipes.speakrs.large.errors import ContractError, PreparationError, RuntimeGateError  # noqa: E402
from recipes.speakrs.large.handoff import package_handoff  # noqa: E402
from recipes.speakrs.large.prepare import seal_release  # noqa: E402
from recipes.speakrs.large.sampler import MixtureSampler  # noqa: E402
from recipes.speakrs.large.selection import (  # noqa: E402
    CorpusScore,
    CycleRecord,
    four_policies,
    open_test_path,
    seal_selection,
)


SPEC_PATH = REPO / "recipes" / "speakrs" / "conf" / "large_cc_v1.json"
LARGE_RUN = REPO / "recipes" / "speakrs" / "large_run.py"


def _row(corpus: str, parent_id: str, split: str = "train") -> RecordingRow:
    return RecordingRow(
        recording_id=parent_id,
        parent_id=parent_id,
        corpus=corpus,
        split=split,
        device_view="canonical",
        label_tier="bronze" if corpus == "NOTSOFAR_sim" else "gold",
        licence=LicenceDecision.ACCEPTED_CC,
        audio_sha256="a" * 64,
        label_sha256="b" * 64,
        sample_count=16000 * 8,
        rejected=False,
        rejection_reason=None,
    )


def _all_rows() -> list[RecordingRow]:
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
        for index in range(4):
            rows.append(_row(corpus, f"{corpus}-{index}"))
    return rows


class SpecContractTest(unittest.TestCase):
    def test_parses_sealed_spec(self):
        spec = parse_spec(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
        self.assertEqual(spec.budget.total_usd, 150)
        self.assertEqual(spec.budget.boundary_policy, "pause_for_extension")
        self.assertEqual(spec.mixture.cycle_examples, 128000)
        self.assertEqual(sum(STREAM_QUOTAS.values()), 1.0)

    def test_rejects_unknown_spec_key(self):
        payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        payload["token"] = "secret"
        with self.assertRaises(ContractError):
            parse_spec(payload)

    def test_preparation_lock_cannot_parse_as_launch(self):
        parse_kinded_lock({"kind": PREPARATION_KIND, "run_id": "large_cc_v1"}, PREPARATION_KIND)
        with self.assertRaises(ContractError):
            parse_kinded_lock(
                {"kind": PREPARATION_KIND, "run_id": "x", "launch_id": "nope", "offer": {}},
                PREPARATION_KIND,
            )
        with self.assertRaises(ContractError):
            parse_kinded_lock({"kind": LAUNCH_KIND}, LAUNCH_KIND)


class PublishedCorpusLoaderTest(unittest.TestCase):
    def test_load_module_registers_dataclasses(self):
        from recipes.speakrs.large.prepare import _load_published_three_corpus

        published = _load_published_three_corpus()
        self.assertEqual(len(published["AMI"]["train"]), 134)
        self.assertEqual(len(published["AliMeeting"]["train"]), 209)
        self.assertEqual(len(published["AISHELL4"]["train"]), 173)


class LotusdisParentTest(unittest.TestCase):
    def test_chunk_path_maps_to_parent_session(self):
        from recipes.speakrs.large.prepare import _lotusdis_parent_id, _parse_lotusdis_csv

        self.assertEqual(
            _lotusdis_parent_id("lotus_dis_ult/audio/jbl/Hijack_S001_T057_Jbl_chunk1.wav"),
            "Hijack_S001_T057",
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "train.csv").write_text(
                "path,sentence\nlotus_dis_ult/audio/jbl/Hijack_S001_T057_Jbl_chunk1.wav,x\n",
                encoding="utf-8",
            )
            (directory / "dev.csv").write_text(
                "path,sentence\nlotus_dis_ult/audio/jbl/Hijack_S081_T069_Jbl_chunk1.wav,x\n",
                encoding="utf-8",
            )
            (directory / "test.csv").write_text(
                "path,sentence\nlotus_dis_ult/audio/jbl/Hijack_S010_T038_Jbl_chunk1.wav,x\n",
                encoding="utf-8",
            )
            rows = _parse_lotusdis_csv(directory)
        self.assertEqual(
            {(row["parent_id"], row["split"]) for row in rows},
            {("Hijack_S001_T057", "train"), ("Hijack_S081_T069", "dev"), ("Hijack_S010_T038", "test")},
        )


class NotsofarSimMapTest(unittest.TestCase):
    def test_parse_utterances_map_and_cached_listing(self):
        import io
        import tarfile

        from recipes.speakrs.large.prepare import (
            _file_starts_with_html,
            _huggingface_sim_list,
            _parse_tar_utterances_map,
        )

        buffer = io.BytesIO()
        payload = json.dumps({"utt-b": 2, "utt-a": 1}).encode("utf-8")
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            info = tarfile.TarInfo("utterances.map")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        self.assertEqual(_parse_tar_utterances_map(buffer.getvalue()), ["utt-a", "utt-b"])
        with self.assertRaises(PreparationError):
            _parse_tar_utterances_map(b"short")
        with tempfile.TemporaryDirectory() as temporary:
            html = Path(temporary) / "quota.html"
            html.write_bytes(b"<!DOCTYPE html><html>quota")
            self.assertTrue(_file_starts_with_html(html))
            cache = Path(temporary) / "sim"
            cache.mkdir()
            (cache / "hf-train-maps.jsonl").write_text(
                json.dumps({"ids": ["u1", "u2"], "count": 2})
                + "\n"
                + json.dumps({"ids": ["u2", "u3"], "count": 2})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_huggingface_sim_list(cache), ["u1", "u2", "u3"])


class PrepareRejectionTest(unittest.TestCase):
    def test_incomplete_release_is_rejected(self):
        spec = parse_spec(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release"
            recordings = [
                {
                    "recording_id": "AMI-0",
                    "parent_id": "AMI-0",
                    "corpus": "AMI",
                    "split": "train",
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
            ]
            with self.assertRaises(PreparationError):
                seal_release(spec, output, recordings, {"AMI": {"train": ["AMI-0"], "dev": [], "test": []}}, [])

    def test_test_leakage_is_rejected(self):
        spec = parse_spec(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release"
            recordings = []
            splits = {}
            for corpus in spec.required_corpora:
                splits[corpus] = {"train": [f"{corpus}-0"], "dev": [f"{corpus}-1"], "test": [f"{corpus}-0"]}
                for split, parent_id in (("train", f"{corpus}-0"), ("dev", f"{corpus}-1"), ("test", f"{corpus}-0")):
                    recordings.append(
                        {
                            "recording_id": f"{parent_id}-{split}",
                            "parent_id": parent_id,
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
            with self.assertRaises(PreparationError):
                seal_release(spec, output, recordings, splits, [])


class SamplerCoverageTest(unittest.TestCase):
    def test_failed_prefetch_and_zero_weight_do_not_advance_coverage(self):
        sampler = MixtureSampler.from_rows(_all_rows(), seed=3407)
        chunks = {
            corpus: {parent: [f"{parent}:0"] for parent in sampler.cursors[corpus].parents}
            for corpus in sampler.cursors
        }
        example = sampler.propose_example(chunks)
        self.assertFalse(sampler.acknowledge(example, optimizer_updated=False, loss_weight=1.0, valid=True))
        self.assertFalse(sampler.acknowledge(example, optimizer_updated=True, loss_weight=0.0, valid=True))
        self.assertFalse(sampler.acknowledge(example, optimizer_updated=True, loss_weight=1.0, valid=False))
        before = {corpus: set(parents) for corpus, parents in sampler.coverage.seen.items()}
        self.assertTrue(all(len(parents) == 0 for parents in before.values()))
        self.assertTrue(sampler.acknowledge(example, optimizer_updated=True, loss_weight=1.0, valid=True))
        restored = MixtureSampler.from_state_dict(sampler.state_dict())
        self.assertEqual(restored.coverage.state_dict(), sampler.coverage.state_dict())

    def test_missing_corpus_is_rejected(self):
        with self.assertRaises(ContractError):
            MixtureSampler.from_rows([_row("AMI", "AMI-0")], seed=3407)

    def test_parent_is_scheduled_before_repeats(self):
        sampler = MixtureSampler.from_rows(_all_rows(), seed=3407)
        cursor = sampler.cursors["AMI"]
        first = list(cursor.unseen_parents)
        chunks = {"AMI": {parent: [f"{parent}:0"] for parent in first}}
        seen = []
        while cursor.unseen_parents:
            parent = cursor.propose_parent(chunks["AMI"], sampler._rng)
            cursor.consume_parent(parent, chunks["AMI"], sampler._rng)
            seen.append(parent)
        self.assertEqual(sorted(seen), sorted(first))
        self.assertEqual(len(seen), len(set(seen)))


class BudgetControllerTest(unittest.TestCase):
    def test_awaiting_extension_cannot_resume_without_amendment(self):
        ledger = BudgetLedger(DEFAULT_BUDGET)
        ledger.pause_for_extension()
        self.assertFalse(ledger.can_resume_training())
        with self.assertRaises(RuntimeGateError):
            ledger.amend(amendment_id="a1", new_total_usd=200, authorized=False)
        ledger.amend(amendment_id="a1", new_total_usd=200, authorized=True)
        self.assertEqual(ledger.spent_usd, 0.0)
        self.assertTrue(ledger.can_resume_training())
        with self.assertRaises(RuntimeGateError):
            ledger.reject_auto_extension()

    def test_terminal_cannot_restart_or_reset_spend(self):
        ledger = BudgetLedger(DEFAULT_BUDGET)
        ledger.charge(3.0, "qualification")
        ledger.mark_terminal("completed")
        with self.assertRaises(RuntimeGateError):
            ledger.charge(1.0, "train")
        self.assertEqual(ledger.spent_usd, 3.0)
        self.assertFalse(ledger.can_resume_training())

    def test_live_vast_guard_blocks_mutations(self):
        guard = LiveVastGuard()
        with self.assertRaises(RuntimeGateError):
            guard.create("offer")
        with self.assertRaises(RuntimeGateError):
            guard.destroy("id")

    def test_worker_environment_has_no_tokens(self):
        cleaned = scrubbed_worker_environment({"VAST_API_KEY": "x", "GHCR_TOKEN": "y", "PATH": "/bin"})
        self.assertNotIn("VAST_API_KEY", cleaned)
        self.assertNotIn("GHCR_TOKEN", cleaned)
        with self.assertRaises(RuntimeGateError):
            assert_worker_has_no_tokens({"HF_TOKEN": "z"})

    def test_qualification_lease_is_not_a_launch_lock(self):
        lease = lease_from_offer(
            {"offer_id": "o1", "gpu_profile": "4090", "usd_per_hour": 0.3, "disk_usd_per_hour": 0.01},
            "prep",
            "/tmp/backup",
        )
        parse_kinded_lock(lease, QUALIFICATION_LEASE_KIND)
        with self.assertRaises(ContractError):
            parse_kinded_lock(lease, LAUNCH_KIND)


class HandoffPackageTest(unittest.TestCase):
    def test_package_handoff_does_not_fabricate_qualification_or_launch(self):
        spec = parse_spec(json.loads(SPEC_PATH.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = package_handoff(
                spec,
                {
                    "pre_rental_work_complete": False,
                    "gpu_qualification_status": "not_run",
                    "checks": {
                        "external-controls": {
                            "rental_gate_status": "blocked_external_control",
                            "missing": ["corrected Base+ G0"],
                        }
                    },
                },
                output,
            )
            self.assertFalse(result["pre_rental_work_complete"])
            self.assertEqual(result["gpu_qualification_status"], "not_run")
            self.assertEqual(result["rental_gate_status"], "blocked_external_control")
            self.assertFalse((output / "launch.lock.json").exists())
            self.assertFalse((output / "qualification.json").exists())
            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["gpu_qualification_status"], "not_run")
            self.assertTrue((output / "SHA256SUMS").is_file())
            self.assertTrue((output / "preparation.lock.json").is_file())


class SelectionTest(unittest.TestCase):
    def _records(self, count: int) -> list[CycleRecord]:
        corpora = ("AMI", "AliMeeting", "AISHELL4", "NOTSOFAR_real", "ICSI", "LOTUSDIS")
        return [
            CycleRecord(
                cycle=index,
                model_hash=f"h{index}",
                loss=1.0 / index,
                der=0.2 / index,
                scores=tuple(CorpusScore(corpus, 1.0, 10.0) for corpus in corpora),
            )
            for index in range(1, count + 1)
        ]

    def test_fewer_than_five_models_cannot_claim_four_policies(self):
        incomplete = four_policies(self._records(3))
        self.assertFalse(incomplete["complete"])
        complete = four_policies(self._records(5))
        self.assertTrue(complete["complete"])
        with self.assertRaises(RuntimeGateError):
            seal_selection(
                records=self._records(3),
                chosen_policy="best_der",
                data_release_hash="d",
                image_digest="i",
                pipeline_hash="p",
                test_manifest_hash="t",
            )

    def test_test_data_cannot_enter_ranking(self):
        records = self._records(5)
        poisoned = [
            CycleRecord(
                cycle=record.cycle,
                model_hash=record.model_hash,
                loss=record.loss,
                der=record.der,
                scores=record.scores + (CorpusScore("VoxConverse", 1.0, 10.0),),
            )
            for record in records
        ]
        with self.assertRaises(RuntimeGateError):
            four_policies(poisoned)

    def test_test_path_requires_selection_seal(self):
        with self.assertRaises(RuntimeGateError):
            open_test_path({})
        sealed = seal_selection(
            records=self._records(5),
            chosen_policy="best_der",
            data_release_hash="d",
            image_digest="i",
            pipeline_hash="p",
            test_manifest_hash="t",
        )
        opened = open_test_path(sealed)
        self.assertTrue(opened["test_access"])
        self.assertEqual(len(opened["test_corpora"]), 7)


class CliSurfaceTest(unittest.TestCase):
    def test_help_and_strict_input_failure(self):
        help_run = subprocess.run(
            [sys.executable, str(LARGE_RUN), "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_run.returncode, 0)
        first = subprocess.run(
            [sys.executable, str(LARGE_RUN), "prepare"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        second = subprocess.run(
            [sys.executable, str(LARGE_RUN), "prepare"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(first.returncode, 0)
        self.assertNotEqual(second.returncode, 0)
        self.assertTrue(first.stdout.strip().startswith("{"))
        self.assertIn("error", first.stdout)
        self.assertEqual(json.loads(first.stdout)["ok"], False)
        self.assertEqual(json.loads(second.stdout)["ok"], False)

    def test_prepare_plan_does_not_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
            spec["relocation"]["audio_root"] = str(Path(temporary) / "audio")
            spec["relocation"]["backup_root"] = str(Path(temporary) / "backup")
            spec["relocation"]["source_cache"] = str(Path(temporary) / "source")
            spec["relocation"]["evidence_root"] = str(Path(temporary) / "evidence")
            spec_path = Path(temporary) / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            output = Path(temporary) / "release"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(LARGE_RUN),
                    "prepare",
                    "--plan",
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(output),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertFalse(payload["downloaded"])
            plan = json.loads((output / "resource-plan.json").read_text(encoding="utf-8"))
            self.assertFalse(plan["downloads"])


if __name__ == "__main__":
    unittest.main()
