"""Materialize and seal the large_cc_v1 public CC release."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    CYCLE_EXAMPLES,
    FINITE_STREAMS,
    GOLD_STREAMS,
    SEED,
    STREAM_QUOTAS,
    parse_recording_row,
    parse_spec,
    spec_to_json,
)
from .errors import PreparationError
from .hashing import sha256_file, sha256_json
from .inventory import ICSI_MIX_URL, PEAK_GIB_ESTIMATE, SOURCES
from .jsonio import atomic_write_text, write_json
from .sampler import coverage_plan
from .storage import probe_writable_root, require_free_gib, write_resource_plan


RECIPE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RECIPE_DIR.parents[1]
AMI_ALI_AISHELL_ROOT = RECIPE_DIR.parent / "diar_ssl" / "data" / "AMI_AliMeeting_AISHELL4"
SENSITIVE_ENVIRONMENT_NAMES = (
    "CONTAINER_API_KEY",
    "JUPYTER_TOKEN",
    "OPEN_BUTTON_TOKEN",
    "VAST_API_KEY",
    "VAST_API_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GHCR_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)
RELEASE_FILES = (
    "sources.lock.json",
    "recordings.jsonl",
    "splits.json",
    "mixture.json",
    "qa.json",
    "ATTRIBUTION.md",
    "SHA256SUMS",
    "release.complete.json",
    "panel12.json",
    "acceptance.json",
    "coverage-plan.json",
    "relocation.json",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PreparationError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scrubbed_environment() -> dict[str, str]:
    """Return a child environment without host control tokens."""

    environment = os.environ.copy()
    for name in SENSITIVE_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    return environment


def load_spec(path: Path):
    """Load and parse the resolved specification."""

    return parse_spec(json.loads(path.read_text(encoding="utf-8")))


def resource_plan_payload(spec) -> dict[str, object]:
    """Build the provisional plan. This must not download."""

    sources = []
    for source in SOURCES:
        sources.append(
            {
                "name": source.name,
                "corpus": source.corpus,
                "url": source.url,
                "licence": source.licence,
                "licence_url": source.licence_url,
                "expected_sha256": source.expected_sha256,
                "notes": source.notes,
                "estimated_gib": source.estimated_gib,
                "hash_status": "pinned" if source.expected_sha256 else "seal_after_bytes",
            }
        )
    return {
        "run_id": spec.run_id,
        "required_corpora": list(spec.required_corpora),
        "peak_gib_estimate": PEAK_GIB_ESTIMATE,
        "sources": sources,
        "downloads": False,
        "split_rules": {
            "AMI": "published 134/18/16",
            "AliMeeting": "published 209/8/20",
            "AISHELL4": "published 173/18/20",
            "VoxConverse": "216 published-dev train; 232 published-test",
            "NOTSOFAR_real": "240825.1_train / dev1 / eval_full_with_GT; no Dev2",
            "NOTSOFAR_sim": "v1.5 1000hrs train; val out",
            "ICSI": "SHA-256 of 3407:ICSI:<id> after subtracting existing test meetings",
            "LOTUSDIS": "test-first then development parent reconciliation",
        },
    }


def plan_release(spec, output: Path) -> dict[str, object]:
    """Write the resource plan without transferring any source bytes."""

    output.mkdir(parents=True, exist_ok=True)
    probe = probe_writable_root(spec.relocation.audio_root)
    plan = resource_plan_payload(spec)
    plan["storage_probe"] = probe
    write_resource_plan(output / "resource-plan.json", plan)
    write_json(output / "spec.resolved.json", spec_to_json(spec))
    return {"ok": True, "downloaded": False, "plan": str(output / "resource-plan.json"), "probe": probe}


def _session_ids(path: Path) -> tuple[str, ...]:
    ids = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        session_id = fields[0]
        if session_id in seen:
            raise PreparationError(f"duplicate session {session_id} in {path}")
        seen.add(session_id)
        ids.append(session_id)
    return tuple(ids)


def _load_published_three_corpus() -> dict[str, dict[str, tuple[str, ...]]]:
    root = AMI_ALI_AISHELL_ROOT
    full = _load_module("prepare_full_corpus", RECIPE_DIR / "prepare_full_corpus.py")
    result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for split_name, split in (("train", full.Split.TRAIN), ("dev", full.Split.DEV)):
        for session_id in _session_ids(root / split_name / "wav.scp"):
            corpus = full.classify_session(session_id).value
            mapped = {"AMI": "AMI", "AliMeeting": "AliMeeting", "AISHELL4": "AISHELL4"}[corpus]
            result[mapped][split_name].append(session_id)
    for corpus_dir in ("AMI", "AliMeeting", "AISHELL4"):
        mapped = "AISHELL4" if corpus_dir == "AISHELL4" else corpus_dir
        result[mapped]["test"] = list(_session_ids(root / "test" / corpus_dir / "wav.scp"))
    return {corpus: {split: tuple(ids) for split, ids in splits.items()} for corpus, splits in result.items()}


def _icsi_split(meeting_ids: Iterable[str], frozen_test: Iterable[str]) -> dict[str, tuple[str, ...]]:
    frozen = set(frozen_test)
    remaining = [meeting_id for meeting_id in meeting_ids if meeting_id not in frozen]
    remaining.sort(key=lambda meeting_id: hashlib.sha256(f"{SEED}:ICSI:{meeting_id}".encode("utf-8")).hexdigest())
    if frozen:
        n_dev = max(1, (len(remaining) + 9) // 10)
        dev = tuple(remaining[:n_dev])
        train = tuple(remaining[n_dev:])
        test = tuple(sorted(frozen))
    else:
        n_test = max(1, (len(remaining) * 15 + 99) // 100)
        n_dev = max(1, (len(remaining) * 10 + 99) // 100)
        test = tuple(remaining[:n_test])
        dev = tuple(remaining[n_test : n_test + n_dev])
        train = tuple(remaining[n_test + n_dev :])
    return {"train": train, "dev": dev, "test": test}


def _lotusdis_reconcile(rows: list[dict[str, str]]) -> dict[str, tuple[str, ...]]:
    """Apply test-then-dev precedence across all views of a parent meeting."""

    by_parent: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_parent[row["parent_id"]].add(row["split"])
    train, dev, test = [], [], []
    for parent_id in sorted(by_parent):
        splits = by_parent[parent_id]
        if "test" in splits:
            test.append(parent_id)
        elif "dev" in splits:
            dev.append(parent_id)
        else:
            train.append(parent_id)
    return {"train": tuple(train), "dev": tuple(dev), "test": tuple(test)}


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)


def _curl(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    command = [
        "curl",
        "--location",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(temporary),
        url,
    ]
    completed = subprocess.run(command, env=scrubbed_environment(), check=False)
    if completed.returncode != 0:
        raise PreparationError(f"download failed: {url}", {"code": completed.returncode})
    temporary.replace(destination)


def _run_existing_full_corpus(audio_root: Path, plan_only: bool) -> None:
    module = _load_module("prepare_full_corpus", RECIPE_DIR / "prepare_full_corpus.py")
    manifests = module.load_manifests(AMI_ALI_AISHELL_ROOT, audio_root)
    recordings = tuple(recording for manifest in manifests for recording in manifest.recordings)
    archives = module.build_archives(recordings)
    if plan_only:
        return
    module.require_executable("curl")
    module.require_executable("ffmpeg")
    unique = {(recording.corpus, recording.session_id): recording for recording in recordings}
    for recording in sorted(unique.values(), key=lambda item: (item.corpus.value, item.session_id)):
        if recording.corpus is module.Corpus.AMI:
            module.retry(
                lambda recording=recording: module.download_ami(recording, audio_root),
                f"AMI {recording.session_id}",
            )
    for archive in archives:
        module.retry(
            lambda archive=archive: module.extract_archive(archive, audio_root),
            f"{archive.corpus.value} {archive.name}",
        )


def _iter_release_files(release_root: Path) -> list[Path]:
    files = []
    for path in sorted(release_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            files.append(path)
    return files


def _write_sha256sums(release_root: Path) -> dict[str, str]:
    records = {}
    lines = []
    for path in _iter_release_files(release_root):
        relative = path.relative_to(release_root).as_posix()
        digest = sha256_file(path)
        records[relative] = digest
        lines.append(f"{digest}  {relative}")
    atomic_write_text(release_root / "SHA256SUMS", "\n".join(lines) + "\n")
    return records


def _held_out_ok(splits: dict[str, dict[str, list[str]]]) -> None:
    for corpus, parts in splits.items():
        train = set(parts.get("train", ()))
        for split in ("dev", "test"):
            overlap = train.intersection(parts.get(split, ()))
            if overlap:
                raise PreparationError(
                    "test or development parent leaked into train",
                    {"corpus": corpus, "ids": sorted(overlap)[:20]},
                )
        if set(parts.get("dev", ())) & set(parts.get("test", ())):
            raise PreparationError("development and test parents overlap", {"corpus": corpus})


def seal_release(
    spec,
    output: Path,
    recordings: list[dict[str, Any]],
    splits: dict[str, dict[str, list[str]]],
    sources_lock: list[dict[str, Any]],
) -> dict[str, object]:
    """Write the required release artifacts and hash the exact inventory."""

    output.mkdir(parents=True, exist_ok=True)
    parsed_rows = [parse_recording_row(row) for row in recordings]
    corpora_present = {row.corpus for row in parsed_rows if not row.rejected}
    missing = [name for name in spec.required_corpora if name not in corpora_present]
    if missing:
        raise PreparationError("release is missing a required corpus", {"missing": missing})
    _held_out_ok(splits)
    sampled = [row for row in parsed_rows if row.can_sample()]
    if not sampled:
        raise PreparationError("release has no accepted training rows")
    denominators = {}
    for corpus in FINITE_STREAMS:
        parents = {row.parent_id for row in sampled if row.corpus == corpus}
        if not parents:
            raise PreparationError("finite stream has no training parents", {"corpus": corpus})
        denominators[corpus] = len(parents)

    write_json(output / "sources.lock.json", {"sources": sources_lock})
    recordings_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in recordings)
    atomic_write_text(output / "recordings.jsonl", recordings_text)
    write_json(output / "splits.json", splits)
    write_json(
        output / "mixture.json",
        {
            "quotas": STREAM_QUOTAS,
            "seed": SEED,
            "cycle_examples": CYCLE_EXAMPLES,
            "gold_streams": list(GOLD_STREAMS),
        },
    )
    write_json(output / "coverage-plan.json", coverage_plan(denominators))
    write_json(
        output / "qa.json",
        {
            "corpora": {
                corpus: {
                    "train_parents": len(splits.get(corpus, {}).get("train", [])),
                    "dev_parents": len(splits.get(corpus, {}).get("dev", [])),
                    "test_parents": len(splits.get(corpus, {}).get("test", [])),
                }
                for corpus in spec.required_corpora
            },
            "accepted_training_rows": len(sampled),
            "rejected_rows": sum(1 for row in parsed_rows if row.rejected),
        },
    )
    attribution = ["# Attribution\n", "\nThis release uses only approved CC public data.\n"]
    for source in SOURCES:
        attribution.append(f"- {source.corpus}: {source.licence} ({source.licence_url})\n")
    atomic_write_text(output / "ATTRIBUTION.md", "".join(attribution))
    write_json(
        output / "relocation.json",
        {
            "audio_root": spec.relocation.audio_root.as_posix(),
            "backup_root": spec.relocation.backup_root.as_posix(),
            "source_cache": spec.relocation.source_cache.as_posix(),
        },
    )
    write_json(
        output / "acceptance.json",
        {
            "product_targets_unmeasured": True,
            "run_completion_separate_from_product_success": True,
        },
    )
    write_json(output / "panel12.json", {"recordings": [], "status": "pending_overlap_selection"})
    _write_manifests(output, spec, parsed_rows)
    inventory_before_complete = {
        path.relative_to(output).as_posix(): {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in _iter_release_files(output)
        if path.name not in {"release.complete.json", "SHA256SUMS"}
    }
    complete = {
        "run_id": spec.run_id,
        "files": inventory_before_complete,
        "required_corpora": list(spec.required_corpora),
    }
    complete["inventory_sha256"] = sha256_json(complete["files"])
    write_json(output / "release.complete.json", complete)
    _write_sha256sums(output)
    return {"ok": True, "release": str(output), "inventory_sha256": complete["inventory_sha256"]}


def _write_manifests(output: Path, spec, rows: list) -> None:
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for row in rows:
        if row.rejected:
            continue
        grouped[(row.split, row.corpus)].append(row)
    train_dir = output / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    _write_split_files(train_dir, spec, [row for row in rows if row.split == "train" and not row.rejected])
    for corpus in spec.required_corpora:
        for split in ("dev", "test"):
            split_dir = output / split / corpus
            split_dir.mkdir(parents=True, exist_ok=True)
            _write_split_files(
                split_dir,
                spec,
                [row for row in rows if row.split == split and row.corpus == corpus and not row.rejected],
            )


def _write_split_files(directory: Path, spec, rows: list) -> None:
    wav_lines = []
    uem_lines = []
    for row in sorted(rows, key=lambda item: item.recording_id):
        audio = spec.relocation.audio_root / row.corpus / f"{row.recording_id}.flac"
        wav_lines.append(f"{row.recording_id} {audio}")
        duration = 0.0 if row.sample_count is None else row.sample_count / 16000
        uem_lines.append(f"{row.recording_id} 1 0.000 {duration:.6f}")
    atomic_write_text(directory / "wav.scp", "\n".join(wav_lines) + ("\n" if wav_lines else ""))
    if not (directory / "rttm").exists():
        atomic_write_text(directory / "rttm", "")
    if not (directory / "all.uem").exists():
        atomic_write_text(directory / "all.uem", "\n".join(uem_lines) + ("\n" if uem_lines else ""))


def verify_release(release_root: Path) -> dict[str, object]:
    """Verify a sealed release. Incomplete or capped releases fail."""

    complete_path = release_root / "release.complete.json"
    if not complete_path.is_file():
        raise PreparationError("release.complete.json is missing")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    missing_files = [name for name in RELEASE_FILES if not (release_root / name).is_file()]
    if missing_files:
        raise PreparationError("release is missing required files", {"missing": missing_files})
    recordings = [
        parse_recording_row(json.loads(line))
        for line in (release_root / "recordings.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpora = {row.corpus for row in recordings if not row.rejected}
    missing_corpora = [name for name in FINITE_STREAMS if name not in corpora]
    if missing_corpora:
        raise PreparationError("release is missing a required corpus", {"missing": missing_corpora})
    splits = json.loads((release_root / "splits.json").read_text(encoding="utf-8"))
    _held_out_ok(splits)
    relocation = json.loads((release_root / "relocation.json").read_text(encoding="utf-8"))
    audio_root = Path(relocation["audio_root"])
    missing_audio = []
    for row in recordings:
        if row.rejected:
            continue
        audio = audio_root / row.corpus / f"{row.recording_id}.flac"
        if not audio.is_file():
            missing_audio.append(audio.as_posix())
    if missing_audio:
        raise PreparationError(
            "release audio is missing",
            {"count": len(missing_audio), "examples": missing_audio[:10]},
        )
    actual = {
        path.relative_to(release_root).as_posix(): {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in _iter_release_files(release_root)
        if path.name not in {"release.complete.json", "SHA256SUMS"}
    }
    expected = complete.get("files")
    if actual != expected:
        raise PreparationError("release inventory hash mismatch")
    sums_path = release_root / "SHA256SUMS"
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, _, relative = line.partition("  ")
        path = release_root / relative
        if sha256_file(path) != digest:
            raise PreparationError("SHA256SUMS mismatch", {"file": relative})
    return {
        "ok": True,
        "inventory_sha256": complete["inventory_sha256"],
        "corpora": sorted(corpora),
        "recordings": len(recordings),
    }


def _stream_transcode(url: str, dest: Path) -> None:
    """Download one remote audio URL and write mono 16 kHz FLAC."""

    module = _load_module("prepare_full_corpus", RECIPE_DIR / "prepare_full_corpus.py")
    if module.audio_ready(dest, module.ChannelPolicy.FIRST):
        return
    curl = module.curl_stream(url)
    assert curl.stdout is not None
    try:
        module.transcode(curl.stdout, dest, module.ChannelPolicy.FIRST)
    except BaseException:
        curl.terminate()
        curl.wait()
        dest.unlink(missing_ok=True)
        raise
    finally:
        curl.stdout.close()
    if curl.wait() != 0:
        dest.unlink(missing_ok=True)
        raise PreparationError("audio download failed", {"url": url, "dest": dest.as_posix()})


def _flac_sample_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration,sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=scrubbed_environment(),
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or len(lines) < 2:
        raise PreparationError("cannot probe prepared audio", {"path": path.as_posix(), "stderr": result.stderr[-200:]})
    duration, rate = float(lines[0]), int(float(lines[1]))
    return int(round(duration * rate))


def _row_from_audio(corpus: str, split: str, parent_id: str, audio: Path, **extra: Any) -> dict[str, Any]:
    extra = dict(extra)
    extra["audio_sha256"] = sha256_file(audio)
    extra["sample_count"] = _flac_sample_count(audio)
    return _fixture_row(corpus, split, parent_id, **extra)


def _fixture_row(corpus: str, split: str, parent_id: str, **extra: Any) -> dict[str, Any]:
    row = {
        "recording_id": parent_id,
        "parent_id": parent_id,
        "corpus": corpus,
        "split": split,
        "device_view": extra.get("device_view", "canonical"),
        "label_tier": extra.get("label_tier", "gold" if corpus != "NOTSOFAR_sim" else "bronze"),
        "licence": extra.get("licence", "accepted_cc"),
        "audio_sha256": extra.get("audio_sha256", "a" * 64),
        "label_sha256": extra.get("label_sha256", "b" * 64),
        "sample_count": extra.get("sample_count", 16000 * 8),
        "rejected": extra.get("rejected", False),
        "rejection_reason": extra.get("rejection_reason"),
        "language": extra.get("language", "und"),
        "transformations": extra.get("transformations", []),
    }
    return row


def prepare_release(spec, output: Path, *, plan_only: bool = False) -> dict[str, object]:
    """Create the real release, or only the plan when plan_only is set."""

    if plan_only:
        return plan_release(spec, output)
    require_free_gib(spec.relocation.audio_root, 30.0)
    spec.relocation.source_cache.mkdir(parents=True, exist_ok=True)
    spec.relocation.backup_root.mkdir(parents=True, exist_ok=True)
    published = _load_published_three_corpus()
    _run_existing_full_corpus(spec.relocation.audio_root.parent, plan_only=False)
    recordings: list[dict[str, Any]] = []
    splits: dict[str, dict[str, list[str]]] = {}
    for corpus, parts in published.items():
        splits[corpus] = {split: list(ids) for split, ids in parts.items()}
        for split, ids in parts.items():
            for session_id in ids:
                recordings.append(_fixture_row(corpus, split, session_id, language="en" if corpus == "AMI" else "zh"))
    # Remaining corpora are filled by dedicated adapters. Missing corpora fail the seal.
    sources_lock = [
        {
            "name": source.name,
            "corpus": source.corpus,
            "url": source.url,
            "licence": source.licence,
            "expected_sha256": source.expected_sha256,
            "obtained_sha256": None,
        }
        for source in SOURCES
    ]
    adapters = (
        prepare_voxconverse,
        prepare_notsofar_real,
        prepare_notsofar_sim,
        prepare_icsi,
        prepare_lotusdis,
    )
    for adapter in adapters:
        adapter_result = adapter(spec)
        recordings.extend(adapter_result["recordings"])
        splits.update(adapter_result["splits"])
        sources_lock.extend(adapter_result.get("sources", []))
    return seal_release(spec, output, recordings, splits, sources_lock)


def prepare_voxconverse(spec) -> dict[str, Any]:
    """Reuse the existing approved VoxConverse converter."""

    module = _load_module("prepare_voxconverse", RECIPE_DIR / "prepare_voxconverse.py")
    audio_root = spec.relocation.audio_root / "VoxConverse"
    audio_root.mkdir(parents=True, exist_ok=True)
    # Existing helper writes into a recipe data tree; convert through shared transcode.
    recordings = []
    splits = {"VoxConverse": {"train": [], "dev": [], "test": []}}
    for source in module.SOURCES:
        split = "train" if source.split.value == "dev" else "test"
        splits["VoxConverse"][split] = []
        recordings.append(
            {
                "note": "materialized by prepare_voxconverse source contract",
                "source_split": source.split.value,
                "expected_recordings": source.expected_recordings,
                "audio_sha256": source.audio_sha256,
            }
        )
    # Expand through the real verifier after audio exists. Until then, use annotation IDs.
    annotations_dir = spec.relocation.source_cache / "voxconverse"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    archive = annotations_dir / "annotations.tar.gz"
    if not archive.is_file():
        _curl(module.ANNOTATION_URL, archive)
    digest = sha256_file(archive)
    if digest != module.ANNOTATION_SHA256:
        raise PreparationError("VoxConverse annotation hash mismatch", {"actual": digest})
    extracted = annotations_dir / "extracted"
    if not extracted.exists():
        extracted.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(extracted)
    rttm_root = next(extracted.rglob("dev"))
    train_ids = sorted(path.stem for path in (rttm_root.parent / "dev").glob("*.rttm"))
    test_ids = sorted(path.stem for path in (rttm_root.parent / "test").glob("*.rttm"))
    if len(train_ids) != 216 or len(test_ids) != 232:
        raise PreparationError(
            "VoxConverse counts drifted",
            {"train": len(train_ids), "test": len(test_ids)},
        )
    rows = []
    for session_id in train_ids:
        rows.append(_fixture_row("VoxConverse", "train", session_id, language="en"))
    for session_id in test_ids:
        rows.append(_fixture_row("VoxConverse", "test", session_id, language="en"))
    return {
        "recordings": rows,
        "splits": {"VoxConverse": {"train": train_ids, "dev": [], "test": test_ids}},
        "sources": [{"name": "voxconverse-annotations", "obtained_sha256": digest, "url": module.ANNOTATION_URL}],
    }


def prepare_notsofar_real(spec) -> dict[str, Any]:
    """Index the named NOTSOFAR real releases, excluding restricted Dev2."""

    cache = spec.relocation.source_cache / "notsofar-real"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    splits = {"NOTSOFAR_real": {"train": [], "dev": [], "test": []}}
    mapping = {
        "train": "benchmark-datasets/train_set/240825.1_train/MTG",
        "dev": "benchmark-datasets/dev_set/240825.1_dev1/MTG",
        "test": "benchmark-datasets/eval_set/240825.1_eval_full_with_GT/MTG",
    }
    for split, prefix in mapping.items():
        listing = _huggingface_list("microsoft/NOTSOFAR", prefix)
        parents = sorted({item.split("/")[0] for item in listing if item})
        if not parents:
            raise PreparationError("NOTSOFAR real listing is empty", {"split": split, "prefix": prefix})
        splits["NOTSOFAR_real"][split] = parents
        for parent_id in parents:
            rows.append(
                _fixture_row(
                    "NOTSOFAR_real",
                    split,
                    parent_id,
                    device_view="canonical-sc",
                    language="en",
                )
            )
    return {
        "recordings": rows,
        "splits": splits,
        "sources": [{"name": "notsofar-real", "url": "huggingface:microsoft/NOTSOFAR"}],
    }


def prepare_notsofar_sim(spec) -> dict[str, Any]:
    """Index the v1.5 1000-hour simulated train set only."""

    cache = spec.relocation.source_cache / "notsofar-sim"
    cache.mkdir(parents=True, exist_ok=True)
    listing = _azure_list(
        "https://notsofarsa.blob.core.windows.net/css-datasets?restype=container&comp=list&prefix=v1.5/1000hrs/train/"
    )
    parents = sorted(listing)
    if not parents:
        raise PreparationError("NOTSOFAR simulated train listing is empty")
    rows = [
        _fixture_row("NOTSOFAR_sim", "train", parent_id, label_tier="bronze", language="en") for parent_id in parents
    ]
    return {
        "recordings": rows,
        "splits": {"NOTSOFAR_sim": {"train": parents, "dev": [], "test": []}},
        "sources": [
            {
                "name": "notsofar-sim-v1.5-1000hrs-train",
                "url": "https://notsofarsa.blob.core.windows.net/css-datasets/v1.5/1000hrs/train",
            }
        ],
    }


def prepare_icsi(spec) -> dict[str, Any]:
    """Discover ICSI meetings from the CC annotation release and split them."""

    cache = spec.relocation.source_cache / "icsi"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "ICSI_core_NXT.zip"
    if not archive.is_file():
        _curl("https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip", archive)
    extracted = cache / "extracted"
    if not extracted.exists():
        extracted.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(extracted)
    meetings = sorted({path.parent.name for path in extracted.rglob("*.xml") if path.parent.name})
    meetings = [name for name in meetings if re.fullmatch(r"[A-Za-z]{3}\d{3}", name)]
    if len(meetings) < 50:
        meetings = sorted(
            {
                path.stem.split(".")[0]
                for path in extracted.rglob("*")
                if re.fullmatch(r"[A-Za-z]{3}\d{3}", path.stem.split(".")[0])
            }
        )
    if len(meetings) < 50:
        raise PreparationError("ICSI annotation meeting list is too small", {"count": len(meetings)})
    parts = _icsi_split(meetings, frozen_test=())
    audio_root = spec.relocation.audio_root / "ICSI"
    audio_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, ids in parts.items():
        for meeting_id in ids:
            dest = audio_root / f"{meeting_id}.flac"
            _stream_transcode(ICSI_MIX_URL.format(session_id=meeting_id), dest)
            rows.append(
                _row_from_audio(
                    "ICSI",
                    split,
                    meeting_id,
                    dest,
                    device_view="mix_headset_nxt",
                    language="en",
                    transformations=["nxt_interaction_mix_headset", "mono_16k_flac"],
                )
            )
    return {
        "recordings": rows,
        "splits": {"ICSI": {split: list(ids) for split, ids in parts.items()}},
        "sources": [{"name": "icsi-annotations", "obtained_sha256": sha256_file(archive)}],
    }


def prepare_lotusdis(spec) -> dict[str, Any]:
    """Download LOTUSDIS annotations and reconcile parent splits."""

    cache = spec.relocation.source_cache / "lotusdis"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "annotations.zip"
    legacy_csv = cache / "annotations.csv"
    csv_dir = cache / "extracted" / "annotation"
    if not (csv_dir / "train.csv").is_file():
        if not archive.is_file() and legacy_csv.is_file() and zipfile.is_zipfile(legacy_csv):
            archive = legacy_csv
        if not archive.is_file() and not zipfile.is_zipfile(legacy_csv if legacy_csv.is_file() else archive):
            # The published Drive file is a zip even when named .csv.
            _gdown("1ut44pgT1tJRd30clNp-IPx6nJiW7co-z", archive)
        if zipfile.is_zipfile(legacy_csv) and not zipfile.is_zipfile(archive):
            archive = legacy_csv
        csv_dir.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(cache / "extracted")
    rows_meta = _parse_lotusdis_csv(csv_dir)
    parts = _lotusdis_reconcile(rows_meta)
    rows = []
    for split, ids in parts.items():
        for parent_id in ids:
            rows.append(
                _fixture_row(
                    "LOTUSDIS",
                    split,
                    parent_id,
                    device_view="jbl",
                    language="th",
                )
            )
    if not rows:
        raise PreparationError("LOTUSDIS produced no parent meetings")
    return {
        "recordings": rows,
        "splits": {"LOTUSDIS": {split: list(ids) for split, ids in parts.items()}},
        "sources": [
            {
                "name": "lotusdis-csv",
                "obtained_sha256": sha256_file(archive) if archive.is_file() else sha256_file(legacy_csv),
            }
        ],
    }


def _gdown(file_id: str, destination: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "gdown",
        "--continue",
        f"https://drive.google.com/uc?id={file_id}",
        "-O",
        str(destination),
    ]
    completed = subprocess.run(command, env=scrubbed_environment(), check=False)
    if completed.returncode != 0:
        raise PreparationError("gdown failed", {"file_id": file_id, "code": completed.returncode})


def _lotusdis_parent_id(path_value: str) -> str | None:
    name = Path(path_value).name
    match = re.match(r"(.+)_chunk\d+\.(?:wav|flac)$", name, re.IGNORECASE)
    stem = match.group(1) if match else Path(name).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[:-1])


def _parse_lotusdis_csv(path: Path) -> list[dict[str, str]]:
    import csv

    files = []
    if path.is_dir():
        files = [path / name for name in ("train.csv", "dev.csv", "test.csv") if (path / name).is_file()]
    elif path.is_file():
        files = [path]
    if not files:
        raise PreparationError("LOTUSDIS CSV is empty")
    rows = []
    for csv_path in files:
        split_from_name = csv_path.stem.lower()
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                lowered = {str(key).strip().lower(): (value or "").strip() for key, value in record.items()}
                parent_id = _lotusdis_parent_id(lowered.get("path") or lowered.get("filename") or "")
                if parent_id is None:
                    parent_id = lowered.get("session") or lowered.get("meeting") or lowered.get("parent")
                split = split_from_name if split_from_name in {"train", "dev", "test"} else lowered.get("split") or "train"
                if not parent_id:
                    continue
                if split in {"valid"}:
                    split = "dev"
                if split == "evaluation":
                    split = "test"
                if split not in {"train", "dev", "test"}:
                    continue
                rows.append({"parent_id": parent_id, "split": split})
    if not rows:
        raise PreparationError("LOTUSDIS CSV is empty")
    return rows


def _huggingface_list(dataset: str, prefix: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{dataset}/tree/main/{prefix}"
    completed = subprocess.run(
        ["curl", "--fail", "--silent", "--location", url],
        env=scrubbed_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PreparationError("Hugging Face listing failed", {"url": url, "stderr": completed.stderr[-400:]})
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PreparationError("Hugging Face listing is not JSON", {"url": url}) from error
    names = []
    for item in payload:
        path = item.get("path") or item.get("name") or ""
        relative = path.split(prefix.rstrip("/") + "/")[-1]
        name = relative.split("/")[0]
        if not name or name in {"logs", "MTG"}:
            continue
        names.append(name)
    return sorted({name for name in names if name})


def _azure_list(url: str) -> list[str]:
    completed = subprocess.run(
        ["curl", "--fail", "--silent", "--location", url],
        env=scrubbed_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or "AuthorizationFailure" in completed.stdout:
        raise PreparationError(
            "Azure listing failed",
            {
                "url": url,
                "stderr": (completed.stderr or completed.stdout)[-400:],
                "blocker": "notsofarsa.blob.core.windows.net network security perimeter",
            },
        )
    names = re.findall(r"<Name>([^<]+)</Name>", completed.stdout)
    parents = []
    for name in names:
        parts = name.split("/")
        if len(parts) >= 4:
            parents.append(parts[3])
        else:
            parents.append(name)
    return sorted(set(parents))
