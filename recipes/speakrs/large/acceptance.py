"""Content, QA, split, hour, and capacity owners used by data preparation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

from .contracts import (
    CAPACITY_LOSS_LIMIT,
    CAPACITY_PROFILES,
    QA_OVERALL_LIMIT,
    QA_STRATUM_LIMIT,
    LabelMethod,
    SelectionState,
    SourceMembership,
    is_placeholder_hash,
    require_content_hash,
)
from .errors import ContractError, PreparationError, UnresolvedInputError
from .hashing import sha256_file, sha256_json
from .target_capacity import CapacityReport as CapacityReport
from .target_capacity import admit_profiles as admit_profiles
from .target_capacity import measure_capacity_loss


ALLOWED_MINIMAL_PURPOSES = frozenset({"train-audio", "train-label", "manifest", "proof"})
FORBIDDEN_MINIMAL_PURPOSES = frozenset(
    {
        "raw-archive",
        "alternate-channel",
        "test-audio",
        "dev-audio",
        "benchmark-audio",
        "model-weight",
        "unused-text",
        "demographics",
        "source-video",
    }
)


@dataclass(frozen=True)
class RttmInterval:
    """One speaker interval on the training signal."""

    recording_id: str
    start: float
    end: float
    speaker: str
    redacted: bool = False


@dataclass(frozen=True)
class DecodedIdentity:
    """Measured audio identity."""

    path: Path
    sha256: str
    sample_count: int
    sample_rate: int
    channels: int
    duration: float


@dataclass(frozen=True)
class HourCounts:
    """Separate timeline, speaker, and device hour totals."""

    timeline_hours: float
    speaker_hours: float
    device_hours: float
    unknown_hours: float
    excluded_hours: float
    heldout_hours: float
    rejected_hours: float


def load_qa_policy(path: Path) -> dict[str, Any]:
    """Load a frozen label-QA policy. Missing required keys fail."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("label-qa-policy must be an object")
    required = {
        "policy_id",
        "seed",
        "speech_convention",
        "frame_seconds",
        "collar_seconds",
        "speaker_time_denominator",
        "window_seconds",
        "uniform_windows",
        "targeted_windows",
        "min_parents",
        "reviewer_procedure",
        "overall_miss_fa_limit",
        "stratum_miss_fa_limit",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ContractError("label-qa-policy is missing required keys", {"missing": missing})
    if float(payload["collar_seconds"]) != 0.0:
        raise ContractError("label-qa-policy collar_seconds must be 0")
    if float(payload["overall_miss_fa_limit"]) != QA_OVERALL_LIMIT:
        raise ContractError("overall miss+FA limit must remain 0.05")
    if float(payload["stratum_miss_fa_limit"]) != QA_STRATUM_LIMIT:
        raise ContractError("stratum miss+FA limit must remain 0.10")
    return payload


def assert_policy_frozen_before_measurement(policy: Mapping[str, Any], measurement: Mapping[str, Any]) -> None:
    """Reject measurements that predate or ignore the frozen policy."""

    policy_hash = sha256_json(dict(policy))
    recorded = measurement.get("policy_sha256")
    if recorded != policy_hash:
        raise PreparationError(
            "QA policy must be frozen before measurement",
            {"expected": policy_hash, "actual": recorded},
        )
    measured_at = measurement.get("measured_at")
    frozen_at = policy.get("frozen_at")
    if not frozen_at or not measured_at or str(measured_at) < str(frozen_at):
        raise PreparationError("measurement timestamp does not follow policy freeze")


class QaPath(str, Enum):
    """Which QA path a source must follow."""

    EXISTING_HUMAN_ACTIVITY = "existing-human-activity"
    HUMAN_REFERENCE_PILOT = "human-reference-pilot"


def select_qa_path(
    *,
    label_method: LabelMethod | str,
    newly_derived: bool = False,
    unresolved_semantic_defect: bool = False,
) -> str:
    """Choose QA path. Existing human activity labels do not need a new reviewer."""

    method = label_method.value if isinstance(label_method, LabelMethod) else str(label_method)
    if newly_derived or unresolved_semantic_defect:
        return QaPath.HUMAN_REFERENCE_PILOT
    if method in {LabelMethod.HUMAN_GOLD.value, "human-activity", "human-gold"}:
        return QaPath.EXISTING_HUMAN_ACTIVITY
    return QaPath.HUMAN_REFERENCE_PILOT


def evaluate_existing_human_activity_labels(record: Mapping[str, Any]) -> dict[str, Any]:
    """Automatic QA for published human activity labels. No reviewer field required."""

    if record.get("qa_path") != QaPath.EXISTING_HUMAN_ACTIVITY:
        raise PreparationError("existing-human-activity path required", {"qa_path": record.get("qa_path")})
    method = str(record.get("label_method", ""))
    if method != LabelMethod.HUMAN_GOLD.value or record.get("newly_derived"):
        raise PreparationError("only unchanged reliable human activity labels can use this QA path")
    if record.get("reviewer_id") in {"fabricated", "automatic", "none", "n/a"}:
        raise PreparationError("existing-label acceptance must not use a fabricated reviewer field")
    if record.get("unresolved_semantic_defect"):
        raise UnresolvedInputError(
            "unresolved semantic defect requires the human-reference path",
            {"defect": record.get("unresolved_semantic_defect")},
        )
    required = (
        "source_version",
        "annotation_provenance",
        "channel_mapping_verified",
        "time_transform_verified",
        "bounds_complete",
        "unknown_regions_handled",
        "split_isolated",
    )
    missing = [key for key in required if not record.get(key)]
    missing.extend(key for key in required[2:] if record.get(key) is not True and key not in missing)
    if missing:
        raise PreparationError("existing-human-activity automatic checks are incomplete", {"missing": missing})
    reject_automatic_self_agreement(method)
    return {
        "ok": True,
        "admitted": True,
        "qa_path": QaPath.EXISTING_HUMAN_ACTIVITY,
        "reviewer_required": False,
        "admitted_profiles": list(record.get("admitted_profiles") or []),
    }


def admit_source_after_automatic_qa(
    *,
    permission_permitted: bool,
    qa: Mapping[str, Any],
    capacity: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Accept existing-human-label sources when automatic checks pass. Capacity is per profile."""

    if not permission_permitted:
        raise PreparationError("source permission is not permitted")
    if not qa.get("admitted"):
        raise PreparationError("QA path did not admit this source")
    if not capacity:
        raise PreparationError("measured capacity profiles are required")
    for item in capacity:
        loss = float(item.get("loss_fraction", float("nan")))
        if not math.isfinite(loss) or loss < 0 or bool(item.get("admitted")) != (loss <= CAPACITY_LOSS_LIMIT):
            raise PreparationError("capacity admission disagrees with measured loss")
    admitted_profiles = [item for item in capacity if item.get("admitted")]
    if not admitted_profiles:
        raise PreparationError("no profile is available for this selection")
    unavailable = [item for item in capacity if not item.get("admitted")]
    return {
        "membership": SourceMembership.ACCEPTED.value,
        "qa_path": qa.get("qa_path"),
        "reviewer_required": False,
        "admitted_profiles": admitted_profiles,
        "unavailable_profiles": unavailable,
        "fully_training_ready": bool(admitted_profiles) and not unavailable,
    }


def reject_automatic_self_agreement(method: LabelMethod | str) -> None:
    """Reject ASR/VAD/diarization self-agreement as human-reference QA."""

    value = method.value if isinstance(method, LabelMethod) else str(method)
    if value in {
        LabelMethod.TRANSCRIPT_VAD_SELF_AGREEMENT.value,
        LabelMethod.MACHINE_UNREVIEWED.value,
        "asr-self-agreement",
        "diarization-self-agreement",
        "vad-self-agreement",
    }:
        raise PreparationError(
            "transcript/VAD self-agreement cannot satisfy human-reference QA",
            {"method": value},
        )


def evaluate_human_reference_pilot(result: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Admit or reject a human-reference pilot. Automatic labels cannot sign this."""

    reject_automatic_self_agreement(str(result.get("label_method", "")))
    if not result.get("reviewer_id"):
        raise UnresolvedInputError(
            "human-reference QA requires an authorized reviewer",
            {"missing_action": "name an authorized human label reviewer and complete the pilot"},
        )
    if result.get("automatic_only"):
        raise PreparationError("transcript/VAD self-agreement cannot satisfy human-reference QA")
    if not result.get("sign_off"):
        raise PreparationError("human-reference pilot requires independent sign-off")
    windows = result.get("windows")
    if not isinstance(windows, list):
        raise PreparationError("pilot windows are missing")
    expected = int(policy["uniform_windows"]) + int(policy["targeted_windows"])
    if len(windows) < expected and not result.get("exhausted_small_release"):
        raise PreparationError(
            "pilot did not inspect the required number of windows",
            {"actual": len(windows), "expected": expected},
        )
    if len({window.get("id") for window in windows}) != len(windows):
        raise PreparationError("pilot windows must be nonduplicated")
    parents = {window.get("parent_id") for window in windows}
    min_parents = min(int(policy["min_parents"]), int(result.get("available_parents", policy["min_parents"])))
    if len(parents) < min_parents:
        raise PreparationError("pilot must cover distinct parents", {"parents": len(parents)})
    overall = float(result["miss_fa_overall"])
    if overall > QA_OVERALL_LIMIT:
        raise PreparationError(
            "pilot combined miss+FA exceeds 5%",
            {"miss_fa_overall": overall},
        )
    strata = result.get("miss_fa_by_stratum") or {}
    if not isinstance(strata, Mapping):
        raise PreparationError("pilot strata are missing")
    for name, value in strata.items():
        if float(value) > QA_STRATUM_LIMIT:
            raise PreparationError(
                "pilot combined miss+FA exceeds 10% in a difficult stratum",
                {"stratum": name, "miss_fa": float(value)},
            )
    for defect in ("speaker_identity_defect", "synchronization_defect", "redaction_defect"):
        if result.get(defect):
            raise PreparationError("unresolved speaker-identity, sync, or redaction defect", {"defect": defect})
    return {"ok": True, "admitted": True, "miss_fa_overall": overall}


def measure_decoded_identity(path: Path) -> DecodedIdentity:
    """Hash and decode one audio file."""

    if not path.is_file():
        raise PreparationError("audio file is missing", {"path": path.as_posix()})
    digest = sha256_file(path)
    if is_placeholder_hash(digest):
        raise PreparationError("placeholder hashes cannot seal a real release", {"path": path.as_posix()})
    info = sf.info(str(path))
    decoded_count = 0
    for block in sf.blocks(str(path), blocksize=65536, always_2d=True):
        if not np.isfinite(block).all():
            raise PreparationError("decoded audio contains non-finite samples")
        decoded_count += len(block)
    if decoded_count != info.frames or decoded_count <= 0:
        raise PreparationError("decoded audio is empty or incomplete")
    sample_count = decoded_count
    channels = int(info.channels)
    rate = int(info.samplerate)
    return DecodedIdentity(
        path=path,
        sha256=digest,
        sample_count=sample_count,
        sample_rate=rate,
        channels=channels,
        duration=sample_count / rate if rate else 0.0,
    )


def verify_decoded_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_sample_count: int,
    expected_sample_rate: int = 16000,
    expected_channels: int = 1,
) -> DecodedIdentity:
    """Fail on wrong bytes, duration, rate, or channel count."""

    expected = require_content_hash(expected_sha256, "expected_sha256")
    identity = measure_decoded_identity(path)
    if identity.sha256 != expected:
        raise PreparationError(
            "audio sha256 does not match the sealed identity",
            {"path": path.as_posix()},
        )
    if identity.sample_rate != expected_sample_rate:
        raise PreparationError(
            "decoded sample rate is wrong",
            {"actual": identity.sample_rate, "expected": expected_sample_rate},
        )
    if identity.channels != expected_channels:
        raise PreparationError(
            "decoded channel count is wrong",
            {"actual": identity.channels, "expected": expected_channels},
        )
    if identity.sample_count != expected_sample_count:
        raise PreparationError(
            "wrong decoded duration",
            {"actual": identity.sample_count, "expected": expected_sample_count},
        )
    return identity


def parse_rttm(text: str) -> list[RttmInterval]:
    """Parse RTTM speaker intervals."""

    intervals: list[RttmInterval] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if not fields or fields[0].upper() != "SPEAKER":
            continue
        if len(fields) < 8:
            raise PreparationError("RTTM line is malformed", {"line": line})
        start = float(fields[3])
        duration = float(fields[4])
        speaker = fields[7]
        if speaker in {"<NA>", "NA", ""}:
            speaker = fields[8] if len(fields) > 8 else ""
        intervals.append(
            RttmInterval(
                recording_id=fields[1],
                start=start,
                end=start + duration,
                speaker=speaker,
            )
        )
    return intervals


def verify_expected_rttm(
    intervals: Sequence[RttmInterval],
    *,
    duration: float,
    recording_id: str,
    redacted_regions: Sequence[tuple[float, float]] = (),
    known_speakers: Sequence[str] | None = None,
) -> None:
    """Reject empty, out-of-bounds, identity, or redaction defects."""

    relevant = [item for item in intervals if item.recording_id == recording_id]
    if not relevant:
        raise PreparationError(
            "empty expected RTTM cannot seal a real release",
            {"recording_id": recording_id},
        )
    for item in relevant:
        if not math.isfinite(item.start) or not math.isfinite(item.end):
            raise PreparationError("label time bounds must be finite")
        if item.start < -1e-6 or item.end - duration > 1e-3:
            raise PreparationError(
                "label time bounds exceed the decoded audio",
                {"recording_id": recording_id, "start": item.start, "end": item.end, "duration": duration},
            )
        if item.end <= item.start:
            raise PreparationError("label duration must be positive", {"recording_id": recording_id})
        if not item.speaker or item.speaker in {"<NA>", "unknown", "UNK"}:
            raise PreparationError(
                "wrong speaker identity",
                {"recording_id": recording_id, "speaker": item.speaker},
            )
        if known_speakers is not None and item.speaker not in set(known_speakers):
            raise PreparationError(
                "wrong speaker identity",
                {"recording_id": recording_id, "speaker": item.speaker},
            )
        for red_start, red_end in redacted_regions:
            overlap = min(item.end, red_end) - max(item.start, red_start)
            if overlap > 1e-6 and not item.redacted:
                raise PreparationError(
                    "redaction handling failed",
                    {"recording_id": recording_id, "speaker": item.speaker},
                )


def assert_split_isolation(
    splits: Mapping[str, Mapping[str, Iterable[str]]],
    *,
    speaker_graph: Mapping[str, str | Sequence[str]] | None = None,
    time_index: Mapping[tuple[str, str], tuple[float, float]] | None = None,
) -> None:
    """Reject parent, speaker, and time leakage into train."""

    for corpus, parts in splits.items():
        train = set(parts.get("train", ()))
        for split in ("dev", "test"):
            overlap = train.intersection(set(parts.get(split, ())))
            if overlap:
                raise PreparationError(
                    "source/speaker/time leakage",
                    {"corpus": corpus, "kind": "parent", "ids": sorted(overlap)[:20]},
                )
        if set(parts.get("dev", ())) & set(parts.get("test", ())):
            raise PreparationError("development and test parents overlap", {"corpus": corpus})
        frozen_test = set(parts.get("test", ()))
        if speaker_graph:

            def speakers(parents):
                values = [speaker_graph[parent] for parent in parents if parent in speaker_graph]
                return {speaker for value in values for speaker in ([value] if isinstance(value, str) else value)}

            train_speakers = speakers(train)
            held_speakers = speakers(set(parts.get("dev", ())) | frozen_test)
            leaked = sorted(train_speakers & held_speakers)
            if leaked:
                raise PreparationError(
                    "source/speaker/time leakage",
                    {"corpus": corpus, "kind": "speaker", "ids": leaked[:20]},
                )
        if time_index:
            for parent in train:
                train_span = time_index.get((corpus, parent))
                if train_span is None:
                    continue
                for held in set(parts.get("dev", ())) | frozen_test:
                    held_span = time_index.get((corpus, held))
                    if held_span is None or parent != held:
                        continue
                    overlap = min(train_span[1], held_span[1]) - max(train_span[0], held_span[0])
                    if overlap > 0:
                        raise PreparationError(
                            "source/speaker/time leakage",
                            {"corpus": corpus, "kind": "time", "parent": parent},
                        )


def reject_duplicate_channel_accounting(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject counting two channels or versions of the same parent as extra hours."""

    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        if row.get("rejected"):
            continue
        key = (str(row["source"]), str(row["parent_id"]))
        view = str(row.get("device_view") or row.get("version") or "")
        if key in seen and seen[key] != view:
            raise PreparationError(
                "duplicate channel/version accounting",
                {"source": key[0], "parent_id": key[1], "views": [seen[key], view]},
            )
        seen[key] = view


def count_hours(
    intervals: Sequence[RttmInterval],
    *,
    duration_by_recording: Mapping[str, float],
    device_hours_by_recording: Mapping[str, float] | None = None,
    unknown_by_recording: Mapping[str, float] | None = None,
    excluded_hours: float = 0.0,
    heldout_ids: Iterable[str] = (),
    rejected_ids: Iterable[str] = (),
) -> HourCounts:
    """Count unique timeline, speaker time, and device time separately."""

    heldout = set(heldout_ids)
    rejected = set(rejected_ids)
    timeline = 0.0
    speaker = 0.0
    device = 0.0
    unknown = 0.0
    heldout_hours = 0.0
    rejected_hours = 0.0
    by_rec: dict[str, list[RttmInterval]] = defaultdict(list)
    for item in intervals:
        by_rec[item.recording_id].append(item)
    for recording_id, duration in duration_by_recording.items():
        if recording_id in rejected:
            rejected_hours += duration / 3600.0
            continue
        if recording_id in heldout:
            heldout_hours += duration / 3600.0
            continue
        timeline += duration / 3600.0
        device += (device_hours_by_recording or {}).get(recording_id, duration) / 3600.0
        unknown += (unknown_by_recording or {}).get(recording_id, 0.0) / 3600.0
        by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for item in by_rec.get(recording_id, []):
            by_speaker[item.speaker].append((max(0.0, item.start), min(duration, item.end)))
        for spans in by_speaker.values():
            cursor = 0.0
            for start, end in sorted(spans):
                speaker += max(0.0, end - max(start, cursor)) / 3600.0
                cursor = max(cursor, end)
    return HourCounts(
        timeline_hours=timeline,
        speaker_hours=speaker,
        device_hours=device,
        unknown_hours=unknown,
        excluded_hours=excluded_hours,
        heldout_hours=heldout_hours,
        rejected_hours=rejected_hours,
    )


def load_uem_durations(text: str) -> dict[str, float]:
    """Parse Kaldi UEM into recording durations in seconds."""

    durations: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            raise PreparationError("UEM line is malformed", {"line": line})
        start = float(fields[-2])
        end = float(fields[-1])
        durations[fields[0]] = max(0.0, end - start)
    return durations


def audit_selection_hours_capacity(
    *,
    source: str,
    splits: Mapping[str, Iterable[str]],
    duration_by_recording: Mapping[str, float],
    intervals: Sequence[RttmInterval],
    uem_by_recording: Mapping[str, Sequence[tuple[float, float]]] | None = None,
) -> dict[str, Any]:
    """Count hours and capacity on known UEM regions without accepting the source"""

    split_map = {name: tuple(ids) for name, ids in splits.items()}
    assert_split_isolation({source: split_map})
    train_ids = set(split_map.get("train", ()))
    heldout = set(split_map.get("dev", ())) | set(split_map.get("test", ()))
    unknown = {
        recording: duration
        - sum(end - start for start, end in (uem_by_recording or {}).get(recording, [(0.0, duration)]))
        for recording, duration in duration_by_recording.items()
    }
    hours = count_hours(
        intervals,
        duration_by_recording=duration_by_recording,
        heldout_ids=heldout,
        rejected_ids=(),
        unknown_by_recording=unknown,
    )
    by_recording: dict[str, list[RttmInterval]] = defaultdict(list)
    for item in intervals:
        if item.recording_id in train_ids:
            by_recording[item.recording_id].append(item)
    capacity = []
    for profile in CAPACITY_PROFILES:
        speaker_seconds = 0.0
        lost_seconds = 0.0
        slot_lost_seconds = 0.0
        encoded_lost_seconds = 0.0
        overlap_lost_seconds = 0.0
        chunks = 0
        for recording_id in sorted(train_ids):
            duration = float(duration_by_recording.get(recording_id, 0.0))
            if duration <= 0:
                continue
            for region in (uem_by_recording or {}).get(recording_id, [(0.0, duration)]):
                report = measure_capacity_loss(
                    by_recording.get(recording_id, ()),
                    duration=duration,
                    uem=region,
                    chunk_seconds=int(profile["chunk_seconds"]),
                    max_overlap=int(profile["max_overlap"]),
                    local_slots=int(profile["local_slots"]),
                    chunk_shift={8: 6, 16: 12}[int(profile["chunk_seconds"])],
                    recording_id=recording_id,
                )
                speaker_seconds += report.speaker_seconds
                lost_seconds += report.lost_seconds
                slot_lost_seconds += report.slot_lost_seconds
                encoded_lost_seconds += report.encoded_lost_seconds
                overlap_lost_seconds += report.overlap_lost_seconds
                chunks += report.chunks
        loss_fraction = 0.0 if speaker_seconds <= 0 else lost_seconds / speaker_seconds
        capacity.append(
            {
                "chunk_seconds": int(profile["chunk_seconds"]),
                "max_overlap": int(profile["max_overlap"]),
                "local_slots": int(profile["local_slots"]),
                "speaker_seconds": speaker_seconds,
                "lost_seconds": lost_seconds,
                "slot_lost_seconds": slot_lost_seconds,
                "encoded_lost_seconds": encoded_lost_seconds,
                "overlap_limit_excess_seconds": overlap_lost_seconds,
                "chunks": chunks,
                "chunk_shift": {8: 6, 16: 12}[int(profile["chunk_seconds"])],
                "model_num_frames": {8: 399, 16: 799}[int(profile["chunk_seconds"])],
                "loss_fraction": loss_fraction,
                "admitted": loss_fraction <= CAPACITY_LOSS_LIMIT,
            }
        )
    return {
        "source": source,
        "accepted": False,
        "membership": SourceMembership.PENDING.value,
        "train_parents": len(train_ids),
        "dev_parents": len(split_map.get("dev", ())),
        "test_parents": len(split_map.get("test", ())),
        "hours": {
            "timeline_hours": hours.timeline_hours,
            "speaker_hours": hours.speaker_hours,
            "device_hours": hours.device_hours,
            "unknown_hours": hours.unknown_hours,
            "excluded_hours": hours.excluded_hours,
            "heldout_hours": hours.heldout_hours,
            "rejected_hours": hours.rejected_hours,
        },
        "capacity": capacity,
        "note": "discovery accounting only; not an acceptance seal",
    }


def reject_target_clipping(original: Sequence[RttmInterval], admitted: Sequence[RttmInterval]) -> None:
    """Fail when labels were truncated to pass a profile."""

    original_key = {(item.recording_id, item.start, item.end, item.speaker) for item in original}
    admitted_key = {(item.recording_id, item.start, item.end, item.speaker) for item in admitted}
    if admitted_key != original_key:
        raise PreparationError("invalid target clipping fails", {"dropped": len(original_key - admitted_key)})


def build_minimal_package(files: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only accepted train audio, labels, and required manifests/proof."""

    accepted: list[dict[str, Any]] = []
    for item in files:
        purpose = str(item.get("purpose", ""))
        split = str(item.get("split", "train"))
        membership = str(item.get("membership", SourceMembership.ACCEPTED.value))
        if purpose in FORBIDDEN_MINIMAL_PURPOSES or split in {"dev", "test"}:
            raise PreparationError(
                "only accepted train artifacts enter the minimal package",
                {"purpose": purpose, "split": split},
            )
        if membership != SourceMembership.ACCEPTED.value:
            raise PreparationError(
                "only accepted train artifacts enter the minimal package",
                {"membership": membership},
            )
        if purpose not in ALLOWED_MINIMAL_PURPOSES:
            raise PreparationError(
                "only accepted train artifacts enter the minimal package",
                {"purpose": purpose},
            )
        accepted.append(dict(item))
    return accepted


def selection_may_upload(state: SelectionState, membership: SourceMembership) -> bool:
    """Remote upload is legal only after QA/split/capacity acceptance."""

    return state is SelectionState.ACCEPTED and membership is SourceMembership.ACCEPTED
