"""Build a bounded private human-review packet for the Open Yap source.

The packet is an evidence artifact, not a training release.  This module keeps
the source archive streamed, derives a declared full-source sampling frame from
metadata only, and uses :func:`prepare_parent` for every emitted review signal.
It never treats machine transcript timestamps as human-reference labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

from .acceptance import load_qa_policy
from .contracts import DiskLimits
from .errors import PreparationError
from .hashing import sha256_file
from .jsonio import write_json
from .prepare import prepare_parent


REVIEW_PACKET_SCHEMA = "speakrs-open-yap-human-review-packet"
REVIEW_PACKET_SCHEMA_VERSION = 1
ARCHIVE_INVENTORY_SCHEMA = "speakrs-open-yap-archive-inventory"
ARCHIVE_INVENTORY_SCHEMA_VERSION = 1
WINDOW_SCHEMA = "speakrs-open-yap-review-window"
WINDOW_SCHEMA_VERSION = 1
PROCEDURE_SCHEMA = "speakrs-open-yap-review-procedure"
PROCEDURE_SCHEMA_VERSION = 1

OPEN_YAP_SAMPLE_RATE = 48_000
REVIEW_SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.02
OPEN_YAP_FRAME_SAMPLES = int(OPEN_YAP_SAMPLE_RATE * FRAME_SECONDS)
REVIEW_FRAME_SAMPLES = int(REVIEW_SAMPLE_RATE * FRAME_SECONDS)
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_UNIFORM_WINDOWS = 50
DEFAULT_TARGETED_WINDOWS = 50
DEFAULT_MIN_PARENTS = 10
DEFAULT_MAX_PARENT_INGRESS_BYTES = 1 << 30
DEFAULT_MAX_PACKET_BYTES = 1 << 30
MAX_JSON_MEMBER_BYTES = 64 * 1024 * 1024
MAX_CANDIDATES_PER_STRATUM = 64
ARCHIVE_MEMBER_RE = re.compile(r"^conversations/([^/]+)/([^/]+)$")
SPEAKER_ROLES = ("speaker_a", "speaker_b")
TARGETED_STRATA = ("overlap", "quiet-speech", "short-turns", "channel-defects")
REQUIRED_JSON_MEMBERS = frozenset(
    {
        "meta.json",
        "speaker_a_meta.json",
        "speaker_b_meta.json",
        "speaker_a_transcript.json",
        "speaker_b_transcript.json",
    }
)
REQUIRED_AUDIO_MEMBERS = frozenset({"speaker_a.flac", "speaker_b.flac"})
# the full archive places this aggregate manifest after the conversation
# directories; keep the allow-list narrow so an unknown top-level member does
# not silently become part of the inventory
GLOBAL_METADATA_MEMBERS = frozenset({"conversations/manifest.json"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_identity(value: str) -> str:
    """Hash a source identity before it enters a report."""

    return _sha256_bytes(f"open-yap:speaker-id:{value}".encode("utf-8"))


def _hash_file_with_progress(path: Path, progress: Callable[[Mapping[str, object]], None] | None) -> str:
    digest = hashlib.sha256()
    scanned = 0
    with path.open("rb") as source:
        while True:
            block = source.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            scanned += len(block)
            if progress is not None and scanned % (256 * 1024 * 1024) < len(block):
                progress({"stage": "archive_hash", "archive_bytes_scanned": scanned})
    if progress is not None:
        progress({"stage": "archive_hash_complete", "archive_bytes_scanned": scanned})
    return digest.hexdigest()


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreparationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PreparationError(f"{label} must be finite and positive")
    return result


def _nonnegative_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreparationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PreparationError(f"{label} must be finite and non-negative")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreparationError(f"{label} must be a positive integer")
    return value


def _member_parts(name: str) -> tuple[str, str]:
    """Validate one archive member name without exposing its contents."""

    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PreparationError("Open Yap archive contains an unsafe member path")
    match = ARCHIVE_MEMBER_RE.fullmatch(name)
    if match is None:
        raise PreparationError("Open Yap archive contains an unexpected member path")
    return match.group(1), match.group(2)


def _read_member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[bytes, str]:
    if member.size < 0 or member.size > MAX_JSON_MEMBER_BYTES:
        raise PreparationError(
            "Open Yap JSON member exceeds the bounded metadata limit",
            {"size_bytes": member.size, "limit_bytes": MAX_JSON_MEMBER_BYTES},
        )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise PreparationError("Open Yap JSON member cannot be read")
    parts: list[bytes] = []
    remaining = member.size
    while remaining:
        block = extracted.read(min(1024 * 1024, remaining))
        if not block:
            raise PreparationError("Open Yap JSON member ended before its declared size")
        parts.append(block)
        remaining -= len(block)
    if extracted.read(1):
        raise PreparationError("Open Yap JSON member is longer than its declared size")
    payload = b"".join(parts)
    return payload, _sha256_bytes(payload)


def _read_member_digest(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    """Hash one bounded metadata member without retaining its payload."""

    if member.size < 0 or member.size > MAX_JSON_MEMBER_BYTES:
        raise PreparationError(
            "Open Yap metadata member exceeds the bounded metadata limit",
            {"size_bytes": member.size, "limit_bytes": MAX_JSON_MEMBER_BYTES},
        )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise PreparationError("Open Yap metadata member cannot be read")
    digest = hashlib.sha256()
    remaining = member.size
    while remaining:
        block = extracted.read(min(1024 * 1024, remaining))
        if not block:
            raise PreparationError("Open Yap metadata member ended before its declared size")
        digest.update(block)
        remaining -= len(block)
    if extracted.read(1):
        raise PreparationError("Open Yap metadata member is longer than its declared size")
    return digest.hexdigest()


def _read_json_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[Any, str]:
    payload, digest = _read_member_bytes(archive, member)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError("Open Yap JSON member is not valid UTF-8 JSON") from error
    return value, digest


def _numeric_intervals(value: object, *, duration: float) -> tuple[tuple[float, float], ...]:
    """Extract only bounded numeric word intervals, never transcript text."""

    intervals: list[tuple[float, float]] = []

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            start = node.get("start")
            end = node.get("end")
            if (
                isinstance(start, (int, float))
                and not isinstance(start, bool)
                and isinstance(end, (int, float))
                and not isinstance(end, bool)
            ):
                start_value = float(start)
                end_value = float(end)
                if math.isfinite(start_value) and math.isfinite(end_value):
                    if end_value > start_value and end_value > 0.0 and start_value < duration:
                        intervals.append((max(0.0, start_value), min(duration, end_value)))
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    intervals.sort()
    return tuple(intervals)


def _recording_metadata(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PreparationError("Open Yap speaker metadata must be an object")
    recording = value.get("recording")
    if not isinstance(recording, Mapping):
        raise PreparationError("Open Yap speaker metadata has no recording facts")
    return recording


def _audio_quality_penalty(value: object) -> float:
    """Convert source quality metadata to a bounded candidate score."""

    if not isinstance(value, Mapping):
        return 0.0
    penalty = 0.0
    for key, raw in value.items():
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or not math.isfinite(float(raw)):
            continue
        number = float(raw)
        name = str(key).lower()
        if "silent_while_partner_spoke" in name:
            penalty = max(penalty, min(1.0, max(0.0, number) / 100.0))
        elif "dnsmos_ovr" in name:
            penalty = max(penalty, min(1.0, max(0.0, 4.0 - number) / 4.0))
        elif "dnsmos_sig" in name or "dnsmos_bak" in name:
            penalty = max(penalty, min(1.0, max(0.0, 3.5 - number) / 3.5))
        elif "noise_floor" in name:
            penalty = max(penalty, min(1.0, max(0.0, -number - 35.0) / 45.0))
    return penalty


def _interval_activity(intervals: Sequence[tuple[float, float]], frame_count: int) -> np.ndarray:
    difference = np.zeros(frame_count + 1, dtype=np.int32)
    for start, end in intervals:
        start_frame = max(0, min(frame_count, math.floor(start / FRAME_SECONDS)))
        end_frame = max(start_frame + 1, min(frame_count, math.ceil(end / FRAME_SECONDS)))
        if start_frame >= frame_count or end_frame <= 0:
            continue
        difference[start_frame] += 1
        difference[end_frame] -= 1
    return np.cumsum(difference[:-1]) > 0


def _rolling_sum(values: np.ndarray, width: int) -> np.ndarray:
    if len(values) < width:
        return np.zeros(0, dtype=np.int64)
    cumulative = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(values, dtype=np.int64)))
    return cumulative[width:] - cumulative[:-width]


def _candidate_windows(
    parent_id: str,
    duration: float,
    intervals_by_speaker: Mapping[str, Sequence[tuple[float, float]]],
    quality_penalty: float,
    duration_mismatch: float,
) -> tuple[int, tuple["WindowCandidate", ...]]:
    window_frames = int(round(DEFAULT_WINDOW_SECONDS / FRAME_SECONDS))
    frame_count = max(1, math.ceil(duration / FRAME_SECONDS))
    window_count = max(0, math.floor((duration - DEFAULT_WINDOW_SECONDS) / FRAME_SECONDS + 1e-9) + 1)
    if window_count == 0:
        return 0, ()
    active_a = _interval_activity(intervals_by_speaker.get("speaker_a", ()), frame_count)
    active_b = _interval_activity(intervals_by_speaker.get("speaker_b", ()), frame_count)
    union = active_a | active_b
    overlap = active_a & active_b
    union_seconds = _rolling_sum(union, window_frames) * FRAME_SECONDS
    overlap_seconds = _rolling_sum(overlap, window_frames) * FRAME_SECONDS
    starts = np.zeros(frame_count, dtype=np.int32)
    short_starts = np.zeros(frame_count, dtype=np.int32)
    for intervals in intervals_by_speaker.values():
        for start, end in intervals:
            start_frame = max(0, min(frame_count - 1, math.floor(start / FRAME_SECONDS)))
            starts[start_frame] += 1
            if end - start <= 0.5:
                short_starts[start_frame] += 1
    word_counts = _rolling_sum(starts, window_frames)
    short_counts = _rolling_sum(short_starts, window_frames)
    overlap_values = np.asarray(overlap_seconds, dtype=np.float64)
    active_values = np.asarray(union_seconds, dtype=np.float64)
    quiet_values = np.where(active_values > 0.0, 1.0 / np.maximum(active_values, FRAME_SECONDS), 0.0)
    short_values = np.asarray(short_counts, dtype=np.float64) + (np.asarray(word_counts) == 0)
    channel_value = min(1.0, max(0.0, quality_penalty + duration_mismatch))
    scores = {
        "overlap": overlap_values,
        "quiet-speech": quiet_values,
        "short-turns": short_values,
        "channel-defects": np.full(window_count, channel_value, dtype=np.float64),
    }
    candidate_indices: set[int] = set()
    for values in scores.values():
        # Keep a small per-parent candidate reservoir. The full population is
        # still represented by population_window_count; only ranking inputs are
        # retained after this parent has been reduced.
        count = min(MAX_CANDIDATES_PER_STRATUM, window_count)
        if count == window_count:
            candidate_indices.update(range(window_count))
            continue
        candidate_indices.update(int(index) for index in np.argsort(-values, kind="stable")[:count])
    candidates = []
    for start_frame in sorted(candidate_indices):
        candidates.append(
            WindowCandidate(
                parent_id=parent_id,
                start_frame=start_frame,
                overlap_score=float(overlap_values[start_frame]),
                quiet_score=float(quiet_values[start_frame]),
                short_turn_score=float(short_values[start_frame]),
                channel_defect_score=channel_value,
                active_seconds=float(active_values[start_frame]),
            )
        )
    return window_count, tuple(candidates)


@dataclass(frozen=True)
class WindowCandidate:
    """A metadata-derived candidate score, never a human label."""

    parent_id: str
    start_frame: int
    overlap_score: float
    quiet_score: float
    short_turn_score: float
    channel_defect_score: float
    active_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_id": self.parent_id,
            "start_frame": self.start_frame,
            "overlap_score": self.overlap_score,
            "quiet_score": self.quiet_score,
            "short_turn_score": self.short_turn_score,
            "channel_defect_score": self.channel_defect_score,
            "active_seconds": self.active_seconds,
        }

    def score_for(self, stratum: str) -> float:
        if stratum == "overlap":
            return self.overlap_score
        if stratum == "quiet-speech":
            return self.quiet_score if self.active_seconds > 0.0 else 0.0
        if stratum == "short-turns":
            return self.short_turn_score
        if stratum == "channel-defects":
            return self.channel_defect_score
        raise PreparationError("unknown Open Yap review stratum", {"stratum": stratum})


@dataclass(frozen=True)
class ParentInventory:
    """Bounded metadata facts for one source parent."""

    parent_id: str
    duration_seconds: float
    speaker_id_sha256: Mapping[str, str]
    member_sizes_bytes: Mapping[str, int]
    member_sha256: Mapping[str, str]
    speaker_durations_seconds: Mapping[str, float]
    speaker_sample_rates: Mapping[str, int]
    speaker_channels: Mapping[str, int]
    quality_penalties: Mapping[str, float]
    duration_mismatch: float
    word_counts: Mapping[str, int]
    invalid_word_intervals: Mapping[str, int]
    population_window_count: int
    candidates: tuple[WindowCandidate, ...]
    transcript_members: Mapping[str, str]

    @property
    def audio_member_names(self) -> tuple[str, str]:
        return tuple(f"conversations/{self.parent_id}/{role}.flac" for role in SPEAKER_ROLES)  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_id": self.parent_id,
            "duration_seconds": self.duration_seconds,
            "speaker_id_sha256": dict(self.speaker_id_sha256),
            "member_sizes_bytes": dict(self.member_sizes_bytes),
            "member_sha256": dict(self.member_sha256),
            "speaker_durations_seconds": dict(self.speaker_durations_seconds),
            "speaker_sample_rates": dict(self.speaker_sample_rates),
            "speaker_channels": dict(self.speaker_channels),
            "quality_penalties": dict(self.quality_penalties),
            "duration_mismatch": self.duration_mismatch,
            "word_counts": dict(self.word_counts),
            "invalid_word_intervals": dict(self.invalid_word_intervals),
            "population_window_count": self.population_window_count,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "transcript_members": dict(self.transcript_members),
        }


@dataclass(frozen=True)
class ArchiveInventory:
    """Full-source metadata sampling frame."""

    archive_path: Path
    archive_size_bytes: int
    archive_sha256: str
    parents: tuple[ParentInventory, ...]
    member_count: int
    json_member_count: int
    audio_member_count: int
    global_metadata_members: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    @property
    def available_parents(self) -> int:
        return sum(parent.population_window_count > 0 for parent in self.parents)

    @property
    def population_window_count(self) -> int:
        return sum(parent.population_window_count for parent in self.parents)

    @property
    def speaker_graph(self) -> dict[str, dict[str, str]]:
        return {parent.parent_id: dict(parent.speaker_id_sha256) for parent in self.parents}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ARCHIVE_INVENTORY_SCHEMA,
            "schema_version": ARCHIVE_INVENTORY_SCHEMA_VERSION,
            "archive": {
                "path": self.archive_path.as_posix(),
                "size_bytes": self.archive_size_bytes,
                "sha256": self.archive_sha256,
            },
            "scan": {
                "member_count": self.member_count,
                "json_member_count": self.json_member_count,
                "audio_member_count": self.audio_member_count,
                "global_metadata_members": {name: dict(facts) for name, facts in self.global_metadata_members.items()},
                "parent_count": len(self.parents),
                "available_parent_count": self.available_parents,
                "population_window_count": self.population_window_count,
                "population_definition": "all 30-second windows on a 20 ms grid with end <= source duration",
                "metadata_only": True,
            },
            "speaker_graph": self.speaker_graph,
            "parents": [parent.to_dict() for parent in self.parents],
        }


@dataclass(frozen=True)
class ReviewWindow:
    """One selected window with no asserted human label."""

    window_id: str
    parent_id: str
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    selection_kind: str
    stratum: str | None
    candidate_score: float | None
    candidate_basis: str

    def key(self) -> tuple[str, int]:
        return self.parent_id, self.start_frame

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": WINDOW_SCHEMA,
            "schema_version": WINDOW_SCHEMA_VERSION,
            "window_id": self.window_id,
            "parent_id": self.parent_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "selection_kind": self.selection_kind,
            "stratum": self.stratum,
            "candidate_score": self.candidate_score,
            "candidate_basis": self.candidate_basis,
            "human_annotation_status": "pending_authorized_human_review",
        }


@dataclass(frozen=True)
class ReviewSelection:
    """Policy-frozen deterministic selection from an archive inventory."""

    windows: tuple[ReviewWindow, ...]
    seed: int
    window_seconds: float
    frame_seconds: float
    uniform_requested: int
    targeted_requested: int
    available_parents: int
    population_window_count: int
    exhausted_small_release: bool

    def __iter__(self):
        return iter(self.windows)

    @property
    def uniform_windows(self) -> tuple[ReviewWindow, ...]:
        return tuple(window for window in self.windows if window.selection_kind == "uniform")

    @property
    def targeted_windows(self) -> tuple[ReviewWindow, ...]:
        return tuple(window for window in self.windows if window.selection_kind == "targeted")

    @property
    def parent_count(self) -> int:
        return len({window.parent_id for window in self.windows})

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "window_seconds": self.window_seconds,
            "frame_seconds": self.frame_seconds,
            "uniform_requested": self.uniform_requested,
            "targeted_requested": self.targeted_requested,
            "uniform_selected": len(self.uniform_windows),
            "targeted_selected": len(self.targeted_windows),
            "available_parents": self.available_parents,
            "selected_parents": self.parent_count,
            "population_window_count": self.population_window_count,
            "exhausted_small_release": self.exhausted_small_release,
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass
class _ParentAccumulator:
    parent_id: str
    member_sizes: dict[str, int] = field(default_factory=dict)
    member_sha256: dict[str, str] = field(default_factory=dict)
    json_members_seen: set[str] = field(default_factory=set)
    meta_facts: dict[str, object] = field(default_factory=dict)
    speaker_durations: dict[str, float] = field(default_factory=dict)
    speaker_rates: dict[str, int] = field(default_factory=dict)
    speaker_channels: dict[str, int] = field(default_factory=dict)
    quality_penalties: dict[str, float] = field(default_factory=dict)
    intervals: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    invalid_word_intervals: dict[str, int] = field(default_factory=dict)
    word_counts: dict[str, int] = field(default_factory=dict)


def _new_accumulator(parent_id: str) -> _ParentAccumulator:
    return _ParentAccumulator(parent_id=parent_id)


def _parse_parent_json(accumulator: _ParentAccumulator, basename: str, value: object) -> None:
    if basename == "meta.json":
        if not isinstance(value, Mapping):
            raise PreparationError("Open Yap meta.json must be an object")
        accumulator.meta_facts = {
            key: value.get(key) for key in ("duration_seconds", "speaker_a_id", "speaker_b_id") if key in value
        }
        return
    role_match = re.fullmatch(r"(speaker_[ab])_(meta|transcript)\.json", basename)
    if role_match is None:
        return
    role, kind = role_match.groups()
    if kind == "meta":
        recording = _recording_metadata(value)
        duration = _finite_positive(recording.get("duration_seconds"), f"{basename}.recording.duration_seconds")
        rate = _positive_int(recording.get("sample_rate"), f"{basename}.recording.sample_rate")
        channels = _positive_int(recording.get("channels"), f"{basename}.recording.channels")
        accumulator.speaker_durations[role] = duration
        accumulator.speaker_rates[role] = rate
        accumulator.speaker_channels[role] = channels
        metrics = value.get("audio_metrics") if isinstance(value, Mapping) else None
        accumulator.quality_penalties[role] = _audio_quality_penalty(metrics)
        return
    intervals = _numeric_intervals(value, duration=float("inf"))
    accumulator.intervals[role] = intervals
    accumulator.word_counts[role] = len(intervals)
    accumulator.invalid_word_intervals[role] = 0


def _finalise_parent(accumulator: _ParentAccumulator) -> ParentInventory:
    meta = accumulator.meta_facts
    if not isinstance(meta, Mapping):
        raise PreparationError("Open Yap parent is missing meta.json")
    for basename in REQUIRED_JSON_MEMBERS:
        if basename not in accumulator.json_members_seen:
            raise PreparationError("Open Yap parent is missing required JSON metadata", {"member": basename})
    for basename in REQUIRED_AUDIO_MEMBERS:
        if basename not in accumulator.member_sizes:
            raise PreparationError("Open Yap parent is missing required audio", {"member": basename})
    duration_value = meta.get("duration_seconds")
    duration = _finite_positive(duration_value, "meta.json.duration_seconds")
    ids: dict[str, str] = {}
    for role in SPEAKER_ROLES:
        identity = meta.get(f"{role}_id")
        if not isinstance(identity, str) or not identity:
            raise PreparationError("Open Yap parent is missing a speaker identity field")
        ids[role] = _sha256_identity(identity)
        if role not in accumulator.speaker_durations:
            raise PreparationError("Open Yap parent is missing speaker recording metadata", {"role": role})
        if role not in accumulator.intervals:
            raise PreparationError("Open Yap parent is missing speaker transcript metadata", {"role": role})
    speaker_durations = dict(accumulator.speaker_durations)
    mismatch = abs(speaker_durations["speaker_a"] - speaker_durations["speaker_b"]) / duration
    quality = dict(accumulator.quality_penalties)
    quality["pair"] = max(quality.values(), default=0.0)
    population_count, candidates = _candidate_windows(
        accumulator.parent_id,
        duration,
        accumulator.intervals,
        quality["pair"],
        min(1.0, mismatch),
    )
    transcript_members = {
        role: f"conversations/{accumulator.parent_id}/{role}_transcript.json" for role in SPEAKER_ROLES
    }
    return ParentInventory(
        parent_id=accumulator.parent_id,
        duration_seconds=duration,
        speaker_id_sha256=ids,
        member_sizes_bytes=dict(accumulator.member_sizes),
        member_sha256=dict(accumulator.member_sha256),
        speaker_durations_seconds=speaker_durations,
        speaker_sample_rates=dict(accumulator.speaker_rates),
        speaker_channels=dict(accumulator.speaker_channels),
        quality_penalties=quality,
        duration_mismatch=min(1.0, mismatch),
        word_counts=dict(accumulator.word_counts),
        invalid_word_intervals=dict(accumulator.invalid_word_intervals),
        population_window_count=population_count,
        candidates=candidates,
        transcript_members=transcript_members,
    )


def scan_archive_metadata(
    archive_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    sealed_sha256: str | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> ArchiveInventory:
    """Scan the complete archive with bounded metadata memory.

    Audio members are skipped during this pass.  Only numeric timing, source
    member sizes, metadata hashes, and salted speaker-ID hashes are retained.
    The archive is never extracted to a bulk cache.
    """

    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise PreparationError("Open Yap archive is missing", {"path": archive.as_posix()})
    size_bytes = archive.stat().st_size
    if expected_size_bytes is not None and size_bytes != expected_size_bytes:
        raise PreparationError(
            "Open Yap archive size does not match the sealed identity",
            {"expected": expected_size_bytes, "actual": size_bytes},
        )
    if sealed_sha256 is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", sealed_sha256):
            raise PreparationError("sealed Open Yap archive SHA-256 is invalid")
        archive_sha256 = sealed_sha256.lower()
    else:
        archive_sha256 = _hash_file_with_progress(archive, progress)
    if expected_sha256 is not None and archive_sha256 != expected_sha256.lower():
        raise PreparationError("Open Yap archive SHA-256 does not match the sealed identity")
    parents: list[ParentInventory] = []
    seen_parent_ids: set[str] = set()
    current_parent_id: str | None = None
    current_accumulator: _ParentAccumulator | None = None
    member_count = 0
    json_member_count = 0
    audio_member_count = 0
    scanned_member_bytes = 0
    global_metadata_members: dict[str, dict[str, object]] = {}
    try:
        with tarfile.open(archive.as_posix(), mode="r|gz") as source:
            for member in source:
                member_count += 1
                scanned_member_bytes += int(member.size)
                if progress is not None and (member_count % 100 == 0):
                    progress(
                        {
                            "stage": "metadata_scan",
                            "members_scanned": member_count,
                            "parents_seen": len(parents) + (current_accumulator is not None),
                            "member_bytes_scanned": scanned_member_bytes,
                        }
                    )
                if not member.isfile():
                    raise PreparationError("Open Yap archive contains a non-regular member")
                if member.name in GLOBAL_METADATA_MEMBERS:
                    if not member.isfile():
                        raise PreparationError("Open Yap global metadata member is not a regular file")
                    if member.name in global_metadata_members:
                        raise PreparationError("Open Yap archive contains a duplicate global metadata member")
                    digest = _read_member_digest(source, member)
                    global_metadata_members[member.name] = {
                        "size_bytes": int(member.size),
                        "sha256": digest,
                        "content_retained": False,
                    }
                    json_member_count += 1
                    continue
                parent_id, basename = _member_parts(member.name)
                if parent_id != current_parent_id:
                    if current_accumulator is not None:
                        parents.append(_finalise_parent(current_accumulator))
                    if parent_id in seen_parent_ids:
                        raise PreparationError("Open Yap parent members are not contiguous")
                    seen_parent_ids.add(parent_id)
                    current_parent_id = parent_id
                    current_accumulator = _new_accumulator(parent_id)
                assert current_accumulator is not None
                accumulator = current_accumulator
                if basename in accumulator.member_sizes:
                    raise PreparationError("Open Yap archive contains a duplicate parent member")
                accumulator.member_sizes[basename] = int(member.size)
                if basename.endswith(".flac"):
                    if basename not in REQUIRED_AUDIO_MEMBERS:
                        raise PreparationError("Open Yap archive contains an unexpected audio member")
                    audio_member_count += 1
                    continue
                if basename.endswith(".json"):
                    json_member_count += 1
                    value, digest = _read_json_member(source, member)
                    accumulator.member_sha256[basename] = digest
                    accumulator.json_members_seen.add(basename)
                    _parse_parent_json(accumulator, basename, value)
                    continue
                if basename.endswith("_dnsmos.json"):
                    # Quality files are optional for the sampling frame.  Their
                    # bytes are not needed for labels or reviewer identity.
                    continue
                raise PreparationError("Open Yap archive contains an unexpected member")
    except (OSError, tarfile.TarError) as error:
        raise PreparationError("Open Yap archive cannot be streamed") from error
    if current_accumulator is not None:
        parents.append(_finalise_parent(current_accumulator))
    parents = tuple(sorted(parents, key=lambda parent: parent.parent_id))
    if not parents:
        raise PreparationError("Open Yap archive contains no conversation parents")
    if progress is not None:
        progress(
            {
                "stage": "metadata_scan_complete",
                "members_scanned": member_count,
                "parents_seen": len(parents),
                "member_bytes_scanned": scanned_member_bytes,
            }
        )
    return ArchiveInventory(
        archive_path=archive,
        archive_size_bytes=size_bytes,
        archive_sha256=archive_sha256,
        parents=parents,
        member_count=member_count,
        json_member_count=json_member_count,
        audio_member_count=audio_member_count,
        global_metadata_members=global_metadata_members,
    )


def load_archive_inventory(
    inventory_path: Path,
    *,
    archive_path: Path | None = None,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> ArchiveInventory:
    """Load a completed metadata inventory without scanning the archive again."""

    path = Path(inventory_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError("Open Yap archive inventory is unreadable") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != ARCHIVE_INVENTORY_SCHEMA:
        raise PreparationError("Open Yap archive inventory schema is invalid")
    archive_record = payload.get("archive")
    scan_record = payload.get("scan")
    parent_records = payload.get("parents")
    if (
        not isinstance(archive_record, Mapping)
        or not isinstance(scan_record, Mapping)
        or not isinstance(parent_records, list)
    ):
        raise PreparationError("Open Yap archive inventory is incomplete")
    recorded_archive = Path(str(archive_record.get("path", ""))).expanduser().resolve()
    archive = Path(archive_path).expanduser().resolve() if archive_path is not None else recorded_archive
    if recorded_archive != archive or not archive.is_file():
        raise PreparationError("Open Yap archive inventory does not identify the current archive")
    try:
        archive_size = int(archive_record["size_bytes"])
        archive_sha256 = str(archive_record["sha256"]).lower()
    except (KeyError, TypeError, ValueError) as error:
        raise PreparationError("Open Yap archive inventory has an invalid archive identity") from error
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha256) or archive.stat().st_size != archive_size:
        raise PreparationError("Open Yap archive inventory has an invalid archive identity")
    if expected_size_bytes is not None and archive_size != expected_size_bytes:
        raise PreparationError("Open Yap archive inventory size does not match the sealed identity")
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) or archive_sha256 != expected_sha256.lower():
            raise PreparationError("Open Yap archive inventory SHA-256 does not match the sealed identity")

    parents: list[ParentInventory] = []
    for record in parent_records:
        if not isinstance(record, Mapping):
            raise PreparationError("Open Yap archive inventory parent record is malformed")
        try:
            parent_id = str(record["parent_id"])
            duration = float(record["duration_seconds"])
            speaker_ids = dict(record["speaker_id_sha256"])
            member_sizes = {str(key): int(value) for key, value in dict(record["member_sizes_bytes"]).items()}
            member_hashes = {str(key): str(value) for key, value in dict(record["member_sha256"]).items()}
            speaker_durations = {
                str(key): float(value) for key, value in dict(record["speaker_durations_seconds"]).items()
            }
            speaker_rates = {str(key): int(value) for key, value in dict(record["speaker_sample_rates"]).items()}
            speaker_channels = {str(key): int(value) for key, value in dict(record["speaker_channels"]).items()}
            quality = {str(key): float(value) for key, value in dict(record["quality_penalties"]).items()}
            mismatch = float(record["duration_mismatch"])
            word_counts = {str(key): int(value) for key, value in dict(record["word_counts"]).items()}
            invalid_counts = {str(key): int(value) for key, value in dict(record["invalid_word_intervals"]).items()}
            population_count = int(record["population_window_count"])
            transcript_members = {str(key): str(value) for key, value in dict(record["transcript_members"]).items()}
            candidate_records = record["candidates"]
        except (KeyError, TypeError, ValueError) as error:
            raise PreparationError("Open Yap archive inventory parent record is invalid") from error
        if not isinstance(parent_id, str) or not parent_id or not isinstance(candidate_records, list):
            raise PreparationError("Open Yap archive inventory parent record is invalid")
        candidates: list[WindowCandidate] = []
        for candidate in candidate_records:
            if not isinstance(candidate, Mapping):
                raise PreparationError("Open Yap archive inventory candidate record is malformed")
            try:
                candidate_parent = str(candidate["parent_id"])
                candidate_item = WindowCandidate(
                    parent_id=candidate_parent,
                    start_frame=int(candidate["start_frame"]),
                    overlap_score=float(candidate["overlap_score"]),
                    quiet_score=float(candidate["quiet_score"]),
                    short_turn_score=float(candidate["short_turn_score"]),
                    channel_defect_score=float(candidate["channel_defect_score"]),
                    active_seconds=float(candidate["active_seconds"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise PreparationError("Open Yap archive inventory candidate record is invalid") from error
            if candidate_parent != parent_id:
                raise PreparationError("Open Yap archive inventory candidate has the wrong parent")
            candidates.append(candidate_item)
        if int(record.get("candidate_count", -1)) != len(candidates):
            raise PreparationError("Open Yap archive inventory candidate count is inconsistent")
        parents.append(
            ParentInventory(
                parent_id=parent_id,
                duration_seconds=duration,
                speaker_id_sha256=speaker_ids,
                member_sizes_bytes=member_sizes,
                member_sha256=member_hashes,
                speaker_durations_seconds=speaker_durations,
                speaker_sample_rates=speaker_rates,
                speaker_channels=speaker_channels,
                quality_penalties=quality,
                duration_mismatch=mismatch,
                word_counts=word_counts,
                invalid_word_intervals=invalid_counts,
                population_window_count=population_count,
                candidates=tuple(candidates),
                transcript_members=transcript_members,
            )
        )
    parents.sort(key=lambda parent: parent.parent_id)
    try:
        member_count = int(scan_record["member_count"])
        json_member_count = int(scan_record["json_member_count"])
        audio_member_count = int(scan_record["audio_member_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise PreparationError("Open Yap archive inventory scan counts are invalid") from error
    global_metadata = scan_record.get("global_metadata_members", {})
    if not isinstance(global_metadata, Mapping):
        raise PreparationError("Open Yap archive inventory global metadata is invalid")
    return ArchiveInventory(
        archive_path=archive,
        archive_size_bytes=archive_size,
        archive_sha256=archive_sha256,
        parents=tuple(parents),
        member_count=member_count,
        json_member_count=json_member_count,
        audio_member_count=audio_member_count,
        global_metadata_members={str(key): dict(value) for key, value in global_metadata.items()},
    )


def _window_id(parent_id: str, start_frame: int, kind: str) -> str:
    token = _sha256_bytes(parent_id.encode("utf-8"))[:16]
    return f"oy-{token}-{kind}-{start_frame:09d}"


def _make_window(
    parent_id: str,
    start_frame: int,
    *,
    kind: str,
    stratum: str | None,
    score: float | None,
    candidate_basis: str,
    window_seconds: float,
) -> ReviewWindow:
    window_frames = int(round(window_seconds / FRAME_SECONDS))
    return ReviewWindow(
        window_id=_window_id(parent_id, start_frame, kind),
        parent_id=parent_id,
        start_frame=start_frame,
        end_frame=start_frame + window_frames,
        start_seconds=start_frame * FRAME_SECONDS,
        end_seconds=(start_frame + window_frames) * FRAME_SECONDS,
        selection_kind=kind,
        stratum=stratum,
        candidate_score=score,
        candidate_basis=candidate_basis,
    )


def _candidate_sort_key(candidate: WindowCandidate, stratum: str, seed: int) -> tuple[float, str]:
    tie = _sha256_bytes(f"{seed}:{stratum}:{candidate.parent_id}:{candidate.start_frame}".encode("utf-8"))
    return (-candidate.score_for(stratum), tie)


def select_review_windows(
    inventory: ArchiveInventory,
    *,
    seed: int = 3407,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    uniform_windows: int = DEFAULT_UNIFORM_WINDOWS,
    targeted_windows: int = DEFAULT_TARGETED_WINDOWS,
    min_parents: int = DEFAULT_MIN_PARENTS,
) -> ReviewSelection:
    """Select policy-frozen uniform and targeted windows deterministically."""

    if window_seconds != DEFAULT_WINDOW_SECONDS:
        raise PreparationError("Open Yap review windows must be 30 seconds")
    for value, label in ((seed, "seed"), (uniform_windows, "uniform_windows"), (targeted_windows, "targeted_windows")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PreparationError(f"{label} must be a non-negative integer")
    if isinstance(min_parents, bool) or not isinstance(min_parents, int) or min_parents < 1:
        raise PreparationError("min_parents must be a positive integer")
    population = inventory.population_window_count
    requested_total = uniform_windows + targeted_windows
    rng = random.Random(seed)
    uniform_count = min(uniform_windows, population)
    uniform_indices = sorted(rng.sample(range(population), uniform_count)) if uniform_count else []
    parent_by_index: list[tuple[ParentInventory, int]] = []
    cursor = 0
    for parent in inventory.parents:
        if parent.population_window_count:
            parent_by_index.append((parent, cursor))
            cursor += parent.population_window_count

    def locate(index: int) -> tuple[ParentInventory, int]:
        for parent, first in parent_by_index:
            if first <= index < first + parent.population_window_count:
                return parent, index - first
        raise PreparationError("uniform population index cannot be located")

    selected: list[ReviewWindow] = []
    used: set[tuple[str, int]] = set()
    for index in uniform_indices:
        parent, start_frame = locate(index)
        selected.append(
            _make_window(
                parent.parent_id,
                start_frame,
                kind="uniform",
                stratum=None,
                score=None,
                candidate_basis="seeded-uniform-full-source-window-population",
                window_seconds=window_seconds,
            )
        )
        used.add((parent.parent_id, start_frame))

    quotas = {stratum: targeted_windows // len(TARGETED_STRATA) for stratum in TARGETED_STRATA}
    for stratum in TARGETED_STRATA[: targeted_windows % len(TARGETED_STRATA)]:
        quotas[stratum] += 1
    candidate_by_stratum: dict[str, list[WindowCandidate]] = {}
    for stratum in TARGETED_STRATA:
        candidates = [candidate for parent in inventory.parents for candidate in parent.candidates]
        candidates.sort(key=lambda candidate: _candidate_sort_key(candidate, stratum, seed))
        candidate_by_stratum[stratum] = candidates

    targeted: list[ReviewWindow] = []
    targeted_parents: set[str] = set()
    # First spread the target set across new parents.  This is a sampling
    # coverage rule, not evidence that any candidate defect is real.
    while len(targeted_parents | {window.parent_id for window in selected}) < min(
        min_parents, inventory.available_parents
    ) and any(quotas.values()):
        progressed = False
        for stratum in TARGETED_STRATA:
            if quotas[stratum] <= 0:
                continue
            candidate = None
            targeted_keys = {window.key() for window in targeted}
            for item in candidate_by_stratum[stratum]:
                key = (item.parent_id, item.start_frame)
                if key in used or key in targeted_keys or item.parent_id in targeted_parents:
                    continue
                candidate = item
                break
            if candidate is None:
                continue
            score = candidate.score_for(stratum)
            targeted.append(
                _make_window(
                    candidate.parent_id,
                    candidate.start_frame,
                    kind="targeted",
                    stratum=stratum,
                    score=score,
                    candidate_basis="metadata-derived-candidate-ranking-pending-human-review",
                    window_seconds=window_seconds,
                )
            )
            used.add((candidate.parent_id, candidate.start_frame))
            targeted_parents.add(candidate.parent_id)
            quotas[stratum] -= 1
            progressed = True
            if len(targeted_parents | {window.parent_id for window in selected}) >= min(
                min_parents, inventory.available_parents
            ):
                break
        if not progressed:
            break

    for stratum in TARGETED_STRATA:
        remaining = quotas[stratum]
        for candidate in candidate_by_stratum[stratum]:
            if remaining <= 0:
                break
            key = (candidate.parent_id, candidate.start_frame)
            if key in used:
                continue
            targeted.append(
                _make_window(
                    candidate.parent_id,
                    candidate.start_frame,
                    kind="targeted",
                    stratum=stratum,
                    score=candidate.score_for(stratum),
                    candidate_basis="metadata-derived-candidate-ranking-pending-human-review",
                    window_seconds=window_seconds,
                )
            )
            used.add(key)
            remaining -= 1

    selected.extend(targeted)
    exhausted = len(selected) < requested_total
    return ReviewSelection(
        windows=tuple(selected),
        seed=seed,
        window_seconds=window_seconds,
        frame_seconds=FRAME_SECONDS,
        uniform_requested=uniform_windows,
        targeted_requested=targeted_windows,
        available_parents=inventory.available_parents,
        population_window_count=population,
        exhausted_small_release=exhausted,
    )


def _directory_size(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def _packet_size(root: Path) -> int:
    """Measure persistent packet files without one-parent temporary ingress."""

    total = 0
    if not root.exists():
        return 0
    for path in root.iterdir():
        if path.name == ".working":
            continue
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
        elif path.is_dir() and not path.is_symlink():
            total += _directory_size(path)
    return total


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise PreparationError("review packet output escaped its root") from error


def _copy_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise PreparationError("Open Yap audio member cannot be read")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with destination.open("wb") as output:
        while True:
            block = extracted.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            output.write(block)
            count += len(block)
    if count != member.size:
        raise PreparationError("Open Yap audio member ended before its declared size")
    return digest.hexdigest()


def _extract_json_for_parent(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> str:
    payload, digest = _read_member_bytes(archive, member)
    destination.write_bytes(payload)
    return digest


def _read_source_intervals(path: Path, *, duration: float) -> tuple[tuple[float, float], ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError("Open Yap transcript metadata cannot be read") from error
    return _numeric_intervals(value, duration=duration)


def _candidate_annotation(
    window: ReviewWindow,
    parent: ParentInventory,
    extracted_json: Mapping[str, Path],
) -> dict[str, object]:
    annotations: list[dict[str, object]] = []
    for role in SPEAKER_ROLES:
        intervals = _read_source_intervals(extracted_json[role], duration=parent.duration_seconds)
        role_intervals = []
        for start, end in intervals:
            clipped_start = max(start, window.start_seconds)
            clipped_end = min(end, window.end_seconds)
            if clipped_end <= clipped_start:
                continue
            role_intervals.append(
                {
                    "start_seconds": round(clipped_start - window.start_seconds, 9),
                    "end_seconds": round(clipped_end - window.start_seconds, 9),
                }
            )
        annotations.append({"speaker_role": role, "intervals": role_intervals})
    return {
        "schema": "speakrs-open-yap-machine-candidate-annotation",
        "schema_version": 1,
        "status": "candidate-only-not-human-reference",
        "window_id": window.window_id,
        "parent_id": window.parent_id,
        "source_timeline": {"start_seconds": window.start_seconds, "end_seconds": window.end_seconds},
        "speaker_activity": annotations,
        "transcript_text_retained": False,
        "human_reference": None,
    }


def _human_template(window: ReviewWindow) -> dict[str, object]:
    return {
        "schema": "speakrs-open-yap-human-annotation-template",
        "schema_version": 1,
        "window_id": window.window_id,
        "parent_id": window.parent_id,
        "review_status": "pending_authorized_human_review",
        "reviewer_id": None,
        "independent_signoff": None,
        "speaker_activity": None,
        "window_disposition": None,
        "speaker_identity_defect": None,
        "clock_alignment": None,
        "synchronization_defect": None,
        "redaction_defect": None,
        "notes": None,
    }


def _procedure(policy_path: Path) -> dict[str, object]:
    try:
        policy = load_qa_policy(policy_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError("Open Yap review policy cannot be read") from error
    return {
        "schema": PROCEDURE_SCHEMA,
        "schema_version": PROCEDURE_SCHEMA_VERSION,
        "source": "Open Yap 1k",
        "global_policy_path": policy_path.as_posix(),
        "global_policy_sha256": sha256_file(policy_path),
        "policy_id": policy["policy_id"],
        "speech_convention": {
            "mark": [
                "audible lexical speech, including unintelligible speech attempts",
                "filled pauses and short spoken responses",
                "the spoken portion of speech that continues during laughter",
            ],
            "exclude": [
                "breath, cough, sniff, or pure non-speech laughter",
                "silence and audible pauses without speech",
            ],
            "boundaries": "use the first and last audible speech sample on the 20 ms grid",
            "short_turns": "preserve every audible short turn; do not apply a minimum duration",
            "gaps": "leave pauses unlabelled and do not smooth across an audible gap",
            "collar_seconds": 0.0,
        },
        "review_signal": {
            "listen_first": "emitted mono 16 kHz review signal",
            "reference": "paired original speaker tracks may be used to resolve activity and identity",
            "candidate_annotations": "machine timestamp proposals are separate and must be corrected or rejected",
            "human_fields": "blank template fields are the only source of human-reference labels",
        },
        "human_annotation_format": {
            "pending_values": {
                "speaker_activity": None,
                "window_disposition": None,
                "speaker_identity_defect": None,
                "clock_alignment": None,
            },
            "window_disposition": {
                "allowed": ["speech", "no_speech"],
                "no_speech": "after listening, use no_speech with an empty speaker_activity array",
            },
            "speaker_activity": {
                "type": "array of interval objects",
                "speaker_allowed": list(SPEAKER_ROLES),
                "time_fields": "start_seconds and end_seconds are window-relative seconds on the 20 ms grid",
                "interval_shape": {
                    "speaker": "speaker_a or speaker_b",
                    "start_seconds": 0.0,
                    "end_seconds": 0.4,
                },
            },
            "defects": {
                "speaker_identity_defect": "use {status: clear|unresolved, notes: string|null}; unresolved requires notes",
                "clock_alignment": "use {status: clear|unresolved, offset_seconds: number|null, drift_seconds_per_second: number|null, notes: string|null}; unresolved requires notes and does not assume zero",
            },
            "format_example_not_a_reference": {
                "window_disposition": "speech",
                "speaker_activity": [{"speaker": "speaker_a", "start_seconds": 0.0, "end_seconds": 0.4}],
                "speaker_identity_defect": {"status": "unresolved", "notes": "FORMAT EXAMPLE ONLY"},
                "clock_alignment": {
                    "status": "unresolved",
                    "offset_seconds": None,
                    "drift_seconds_per_second": None,
                    "notes": "FORMAT EXAMPLE ONLY",
                },
            },
        },
        "pairing_and_clock": {
            "source_tracks": "separate mono speaker files",
            "zero_offset_assumed": False,
            "status": "unresolved_pending_human_review",
            "review_action": "record offset, drift, identity, and channel defects; do not infer zero from equal headers",
        },
        "privacy": {
            "transcript_text_retained": False,
            "demographic_values_retained": False,
            "speaker_identity_values_retained": False,
            "speaker_identity_hashes": "private source-scoped salted SHA-256 only",
        },
        "reviewer": {
            "authorized_human_required": True,
            "independent_signoff_required": True,
            "reviewer_id_storage": "private human-review record only",
        },
    }


def _write_review_guide(output_dir: Path, procedure: Mapping[str, object]) -> None:
    write_json(output_dir / "source-procedure.json", dict(procedure))
    guide = """# Open Yap review guide

This private packet is pending an authorized human review. Listen to the emitted
mono review signal first. Use the paired original speaker tracks as reference.

Mark audible spoken activity, including unintelligible speech attempts, filled
pauses, and short spoken responses. Do not mark a breath, cough, sniff, or pure
laughter. If laughter contains speech, mark only the spoken portion. Leave
audible pauses unmarked. Preserve short turns, use the 20 ms grid, use no collar,
and do not smooth across gaps.

The candidate annotation is a machine proposal. It is not a reference label.
Write corrections only in the blank human template. Keep speaker_activity null
until review. After review, set window_disposition to speech or no_speech. For
no_speech, use an empty speaker_activity array. For speech, use an array of
objects with speaker set to speaker_a or speaker_b and start_seconds and
end_seconds as window-relative seconds on the 20 ms grid. Times are numbers, not
strings, and each end must be greater than its start.

FORMAT EXAMPLE — NOT A FILLED REFERENCE:
{
  "window_disposition": "speech",
  "speaker_activity": [
    {"speaker": "speaker_a", "start_seconds": 0.0, "end_seconds": 0.4}
  ]
}

Record speaker identity, clock alignment or drift, redaction, and systematic
defects. Use {"status": "unresolved", "notes": "..."} for an unresolved
identity defect. For clock uncertainty, use status unresolved with notes and
leave offset_seconds and drift_seconds_per_second null; never infer zero from
equal sample-rate or duration headers. Mark a reviewed clear identity or clock
with status clear. Leave all defect fields null while review is pending.
"""
    (output_dir / "review-guide.md").write_text(guide, encoding="utf-8")


def _check_packet_size(output_dir: Path, maximum_bytes: int) -> None:
    if _packet_size(output_dir) > maximum_bytes:
        raise PreparationError(
            "Open Yap review packet exceeds its bounded size",
            {"path": output_dir.as_posix(), "limit_bytes": maximum_bytes},
        )


def _process_parent(
    *,
    parent: ParentInventory,
    windows: Sequence[ReviewWindow],
    source_paths: Mapping[str, Path],
    source_hashes: Mapping[str, str],
    transcript_paths: Mapping[str, Path],
    output_dir: Path,
    working_dir: Path,
    disk_limits: DiskLimits,
    maximum_packet_bytes: int,
    maximum_parent_ingress_bytes: int,
) -> list[dict[str, object]]:
    source_bytes = sum(parent.member_sizes_bytes.get(f"{role}.flac", 0) for role in SPEAKER_ROLES)
    if source_bytes > maximum_parent_ingress_bytes:
        raise PreparationError(
            "Open Yap selected parent exceeds the bounded working ingress",
            {"parent_id": parent.parent_id, "size_bytes": source_bytes, "limit_bytes": maximum_parent_ingress_bytes},
        )
    parent_work = working_dir / parent.parent_id
    parent_work.mkdir(parents=True, exist_ok=True)
    for role in SPEAKER_ROLES:
        if not source_paths[role].is_file() or not transcript_paths[role].is_file():
            raise PreparationError("selected Open Yap parent has incomplete extracted members")
    results: list[dict[str, object]] = []
    for window in windows:
        window_dir = output_dir / "windows" / window.window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        source_reference: dict[str, dict[str, object]] = {}
        canonical_arrays: dict[str, np.ndarray] = {}
        canonical_window_facts: dict[str, Mapping[str, object]] = {}
        for role in SPEAKER_ROLES:
            source_window = window_dir / f"reference-{role}.flac"
            with sf.SoundFile(str(source_paths[role]), mode="r") as source_handle:
                source_rate = int(source_handle.samplerate)
                start = window.start_frame * OPEN_YAP_FRAME_SAMPLES
                end = window.end_frame * OPEN_YAP_FRAME_SAMPLES
                if start < 0 or end <= start or end > int(source_handle.frames):
                    raise PreparationError("Open Yap source track cannot cover the selected review window")
                source_handle.seek(start)
                samples = source_handle.read(end - start, dtype="float64", always_2d=False)
            if source_rate != OPEN_YAP_SAMPLE_RATE or samples.ndim != 1:
                raise PreparationError("Open Yap source track is not mono 48 kHz")
            if len(samples) != end - start:
                raise PreparationError("Open Yap source track ended before the selected review window")
            sf.write(source_window, samples, source_rate, format="FLAC", subtype="PCM_16")
            source_window_sha256 = sha256_file(source_window)
            source_reference[role] = {
                "archive_member": f"conversations/{parent.parent_id}/{role}.flac",
                "archive_member_sha256": source_hashes[role],
                "window_path": _safe_relative(source_window, output_dir),
                "window_sha256": source_window_sha256,
                "sample_rate": source_rate,
                "channels": 1,
                "start_sample": start,
                "end_sample": end,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
            }
            canonical = parent_work / f"{window.window_id}-{role}-canonical.flac"
            canonical_window_facts[role] = prepare_parent(
                source_window,
                "mono",
                source_window_sha256,
                [],
                None,
                canonical,
                disk_limits,
                parent_id=f"{window.window_id}-{role}-review",
            )
            with sf.SoundFile(str(canonical), mode="r") as canonical_handle:
                canonical_arrays[role] = canonical_handle.read(
                    round((window.end_seconds - window.start_seconds) * REVIEW_SAMPLE_RATE),
                    dtype="float64",
                    always_2d=False,
                )
        frame_count = int(round((window.end_seconds - window.start_seconds) * REVIEW_SAMPLE_RATE))
        for role in SPEAKER_ROLES:
            if len(canonical_arrays[role]) != frame_count:
                raise PreparationError("canonical review window has an incomplete sample range")
        mixed = np.clip((canonical_arrays["speaker_a"] + canonical_arrays["speaker_b"]) * 0.5, -1.0, 1.0)
        mixed_input = window_dir / "emitted-input.wav"
        sf.write(mixed_input, mixed, REVIEW_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        mixed_hash = sha256_file(mixed_input)
        emitted = window_dir / "emitted.flac"
        emitted_facts = prepare_parent(
            mixed_input,
            "mono",
            mixed_hash,
            [],
            None,
            emitted,
            disk_limits,
            parent_id=f"{window.window_id}-emitted",
        )
        candidate = _candidate_annotation(window, parent, transcript_paths)
        candidate_path = window_dir / "candidate-annotation.json"
        human_path = window_dir / "human-annotation.json"
        write_json(candidate_path, candidate)
        write_json(human_path, _human_template(window))
        result = {
            "window": window.to_dict(),
            "files": {
                "reference_speaker_a": _safe_relative(window_dir / "reference-speaker_a.flac", output_dir),
                "reference_speaker_b": _safe_relative(window_dir / "reference-speaker_b.flac", output_dir),
                "emitted": _safe_relative(emitted, output_dir),
                "emitted_input": _safe_relative(mixed_input, output_dir),
                "emitted_transform": _safe_relative(Path(emitted_facts["receipt_path"]), output_dir),
                "candidate_annotation": _safe_relative(candidate_path, output_dir),
                "human_annotation_template": _safe_relative(human_path, output_dir),
            },
            "source_audio": source_reference,
            "emitted_audio": {
                "path": _safe_relative(emitted, output_dir),
                "sha256": emitted_facts["canonical_sha256"],
                "sample_rate": REVIEW_SAMPLE_RATE,
                "channels": 1,
                "sample_count": emitted_facts["sample_count"],
                "source_input_sha256": mixed_hash,
                "mix": "0.5 * speaker_a + 0.5 * speaker_b after shared 48 kHz to 16 kHz preparation",
                "transform_receipt": emitted_facts["receipt"],
            },
            "pairing_and_clock": {
                "status": "unresolved_pending_human_review",
                "zero_offset_assumed": False,
                "source_timeline": {"start_seconds": window.start_seconds, "end_seconds": window.end_seconds},
                "canonical_frame_count": frame_count,
            },
            "candidate_labels": {"path": _safe_relative(candidate_path, output_dir), "human_reference": False},
            "human_review": {"template_path": _safe_relative(human_path, output_dir), "status": "pending"},
            "canonical_window_transforms": {
                role: {
                    "source_member_sha256": source_hashes[role],
                    "source_window_sha256": source_reference[role]["window_sha256"],
                    "canonical_sha256": canonical_window_facts[role]["canonical_sha256"],
                    "transform_receipt": canonical_window_facts[role]["receipt"],
                }
                for role in SPEAKER_ROLES
            },
        }
        results.append(result)
        _check_packet_size(output_dir, maximum_packet_bytes)
    shutil.rmtree(parent_work, ignore_errors=True)
    return results


def build_review_packet(
    archive_path: Path,
    output_dir: Path,
    disk_limits: DiskLimits,
    *,
    inventory: ArchiveInventory | None = None,
    selection: ReviewSelection | None = None,
    policy_path: Path | None = None,
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    sealed_archive_sha256: str | None = None,
    maximum_parent_ingress_bytes: int = DEFAULT_MAX_PARENT_INGRESS_BYTES,
    maximum_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
) -> dict[str, object]:
    """Build the private packet while retaining at most one source parent."""

    if maximum_parent_ingress_bytes <= 0 or maximum_packet_bytes <= 0:
        raise PreparationError("review packet byte limits must be positive")
    archive = Path(archive_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if inventory is None:
        inventory = scan_archive_metadata(
            archive,
            expected_sha256=expected_archive_sha256,
            expected_size_bytes=expected_archive_size_bytes,
            sealed_sha256=sealed_archive_sha256,
        )
    else:
        if inventory.archive_path != archive or inventory.archive_size_bytes != archive.stat().st_size:
            raise PreparationError("review packet inventory does not match the current archive")
        trusted_sha256 = expected_archive_sha256 or sealed_archive_sha256
        if trusted_sha256 is not None:
            if not re.fullmatch(r"[0-9a-fA-F]{64}", trusted_sha256):
                raise PreparationError("sealed Open Yap archive SHA-256 is invalid")
            if inventory.archive_sha256 != trusted_sha256.lower():
                raise PreparationError("review packet inventory does not match the sealed archive identity")
        elif inventory.archive_sha256 != sha256_file(archive):
            raise PreparationError("review packet inventory does not match the current archive")
    if selection is None:
        selection = select_review_windows(inventory)
    if selection.population_window_count != inventory.population_window_count:
        raise PreparationError("review selection was produced from a different source population")
    if output.exists() and any(output.iterdir()):
        raise PreparationError("review packet output must be empty before construction")
    output.mkdir(parents=True, exist_ok=True)
    procedure_path = policy_path or Path(__file__).resolve().parents[1] / "conf" / "label-qa-policy.json"
    procedure = _procedure(procedure_path)
    _write_review_guide(output, procedure)
    write_json(output / "archive-inventory.json", inventory.to_dict())
    write_json(output / "selection.json", selection.to_dict())
    selected_by_parent: dict[str, list[ReviewWindow]] = defaultdict(list)
    for window in selection.windows:
        selected_by_parent[window.parent_id].append(window)
    parent_by_id = {parent.parent_id: parent for parent in inventory.parents}
    for parent_id in selected_by_parent:
        if parent_id not in parent_by_id:
            raise PreparationError("review selection contains an unknown parent")
    working_dir = output / ".working"
    working_dir.mkdir(parents=True, exist_ok=True)
    result_windows: list[dict[str, object]] = []
    seen_parents: set[str] = set()
    current_parent: str | None = None
    current_members: set[str] = set()
    current_source_paths: dict[str, Path] = {}
    current_source_hashes: dict[str, str] = {}
    current_transcript_paths: dict[str, Path] = {}

    def finish_current() -> None:
        nonlocal current_parent, current_members, current_source_paths, current_source_hashes, current_transcript_paths
        if current_parent is None or current_parent not in selected_by_parent:
            current_parent = None
            current_members = set()
            current_source_paths = {}
            current_source_hashes = {}
            current_transcript_paths = {}
            return
        if current_parent in seen_parents:
            raise PreparationError("selected Open Yap parent is not contiguous in the archive")
        parent = parent_by_id[current_parent]
        needed = {"speaker_a.flac", "speaker_b.flac", "speaker_a_transcript.json", "speaker_b_transcript.json"}
        missing = sorted(needed - set(current_members))
        if missing:
            raise PreparationError("selected Open Yap parent is missing packet members", {"missing": missing})
        result_windows.extend(
            _process_parent(
                parent=parent,
                windows=selected_by_parent[current_parent],
                source_paths=current_source_paths,
                source_hashes=current_source_hashes,
                transcript_paths=current_transcript_paths,
                output_dir=output,
                working_dir=working_dir,
                disk_limits=disk_limits,
                maximum_packet_bytes=maximum_packet_bytes,
                maximum_parent_ingress_bytes=maximum_parent_ingress_bytes,
            )
        )
        seen_parents.add(current_parent)
        current_parent = None
        current_members = set()
        current_source_paths = {}
        current_source_hashes = {}
        current_transcript_paths = {}

    try:
        with tarfile.open(archive.as_posix(), mode="r|gz") as source:
            for member in source:
                if member.name in GLOBAL_METADATA_MEMBERS:
                    finish_current()
                    continue
                parent_id, basename = _member_parts(member.name)
                if parent_id != current_parent:
                    finish_current()
                    current_parent = parent_id
                if parent_id not in selected_by_parent:
                    continue
                if basename in current_members:
                    raise PreparationError("selected Open Yap parent contains duplicate members")
                if basename in REQUIRED_AUDIO_MEMBERS:
                    parent_work = working_dir / parent_id
                    parent_work.mkdir(parents=True, exist_ok=True)
                    source_path = parent_work / basename
                    current_source_hashes[basename.removesuffix(".flac")] = _copy_member(source, member, source_path)
                    current_source_paths[basename.removesuffix(".flac")] = source_path
                    current_members.add(basename)
                    continue
                if basename in {"speaker_a_transcript.json", "speaker_b_transcript.json"}:
                    parent_work = working_dir / parent_id
                    parent_work.mkdir(parents=True, exist_ok=True)
                    transcript_path = parent_work / basename
                    _extract_json_for_parent(source, member, transcript_path)
                    current_transcript_paths[basename.removesuffix("_transcript.json")] = transcript_path
                    current_members.add(basename)
                    continue
            finish_current()
    except (OSError, tarfile.TarError) as error:
        raise PreparationError("Open Yap packet source cannot be streamed") from error
    if set(seen_parents) != set(selected_by_parent):
        missing = sorted(set(selected_by_parent) - seen_parents)
        raise PreparationError("selected Open Yap parents were not found", {"count": len(missing)})
    shutil.rmtree(working_dir, ignore_errors=True)
    result_windows.sort(key=lambda item: str(item["window"]["window_id"]))
    manifest = {
        "schema": REVIEW_PACKET_SCHEMA,
        "schema_version": REVIEW_PACKET_SCHEMA_VERSION,
        "status": "pending_human_reference_pilot",
        "upload_allowed": False,
        "candidate_labels_status": "machine-proposal-only",
        "human_reference_status": "not_performed",
        "clock_status": "unresolved_pending_human_review",
        "archive": {
            "path": archive.as_posix(),
            "size_bytes": inventory.archive_size_bytes,
            "sha256": inventory.archive_sha256,
        },
        "policy": procedure,
        "sampling": selection.to_dict(),
        "speaker_graph": inventory.speaker_graph,
        "windows": result_windows,
        "privacy": {
            "transcript_text_retained": False,
            "demographic_values_retained": False,
            "speaker_identity_values_retained": False,
        },
        "bounds": {
            "maximum_parent_ingress_bytes": maximum_parent_ingress_bytes,
            "maximum_packet_bytes": maximum_packet_bytes,
            "packet_bytes": _packet_size(output),
            "one_parent_extracted_at_a_time": True,
        },
    }
    write_json(output / "manifest.json", manifest)
    _check_packet_size(output, maximum_packet_bytes)
    manifest["bounds"] = {**manifest["bounds"], "packet_bytes": _packet_size(output)}
    write_json(output / "manifest.json", manifest)
    return {
        "ok": True,
        "packet_path": output.as_posix(),
        "manifest_path": (output / "manifest.json").as_posix(),
        "window_count": len(result_windows),
        "parent_count": len(seen_parents),
        "uniform_count": len(selection.uniform_windows),
        "targeted_count": len(selection.targeted_windows),
        "packet_bytes": _packet_size(output),
        "human_review_required": True,
        "clock_verification_required": True,
    }


def validate_review_packet(path: Path) -> dict[str, object]:
    """Validate packet structure, hashes, and unresolved human fields."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError("review packet manifest is unreadable") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema") != REVIEW_PACKET_SCHEMA:
        raise PreparationError("review packet manifest schema is invalid")
    if manifest.get("upload_allowed") is not False:
        raise PreparationError("review packet must not be uploadable")
    windows = manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise PreparationError("review packet has no windows")
    ids: set[str] = set()
    for record in windows:
        if not isinstance(record, Mapping):
            raise PreparationError("review packet window record is malformed")
        window = record.get("window")
        files = record.get("files")
        if not isinstance(window, Mapping) or not isinstance(files, Mapping):
            raise PreparationError("review packet window record is incomplete")
        window_id = window.get("window_id")
        if not isinstance(window_id, str) or not window_id or window_id in ids:
            raise PreparationError("review packet window IDs must be unique")
        ids.add(window_id)
        for key in (
            "reference_speaker_a",
            "reference_speaker_b",
            "emitted",
            "emitted_transform",
            "candidate_annotation",
            "human_annotation_template",
        ):
            relative = files.get(key)
            if not isinstance(relative, str):
                raise PreparationError("review packet file reference is missing")
            target = (root / relative).resolve()
            if not target.is_file() or root not in target.parents:
                raise PreparationError("review packet file reference is unsafe or missing")
        candidate = json.loads((root / str(files["candidate_annotation"])).read_text(encoding="utf-8"))
        human = json.loads((root / str(files["human_annotation_template"])).read_text(encoding="utf-8"))
        if candidate.get("human_reference") is not None:
            raise PreparationError("candidate annotation contains human-reference data")
        if any(
            human.get(key) is not None
            for key in ("reviewer_id", "independent_signoff", "speaker_activity", "window_disposition")
        ):
            raise PreparationError("human annotation template is not blank")
    return {
        "ok": True,
        "window_count": len(windows),
        "parent_count": len({record["window"]["parent_id"] for record in windows}),
        "upload_allowed": False,
        "human_review_required": True,
    }


prepare_review_packet = build_review_packet
inspect_archive_metadata = scan_archive_metadata


__all__ = [
    "ARCHIVE_INVENTORY_SCHEMA",
    "ArchiveInventory",
    "ParentInventory",
    "ReviewSelection",
    "ReviewWindow",
    "WindowCandidate",
    "build_review_packet",
    "inspect_archive_metadata",
    "load_archive_inventory",
    "prepare_review_packet",
    "scan_archive_metadata",
    "select_review_windows",
    "validate_review_packet",
]
