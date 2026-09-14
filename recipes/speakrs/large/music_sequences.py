"""Prepare deterministic long-gap speech sequences with instrumental segments.

This module owns the offline materialisation of long-gap examples.  It does
not mix music under speech; the runtime loader applies that augmentation.  A
sequence contains source speech excerpts, exact sample-count digital silence,
and contiguous no-vocals music excerpts.  Every source and output boundary is
represented in the derivation manifest on the 16 kHz sample clock.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import soundfile as sf

from diarizen.music_augmentation import MusicAugmentationError, load_music_manifest


if __package__ in {None, ""}:
    from recipes.speakrs.large.errors import PreparationError
    from recipes.speakrs.large.hashing import sha256_file, sha256_json
else:
    from .errors import PreparationError
    from .hashing import sha256_file, sha256_json


SAMPLE_RATE = 16_000
CHANNELS = 1
CODEC = "FLAC"
SUBTYPE = "PCM_16"
SCHEMA = "speakrs-music-long-gaps"
SCHEMA_VERSION = 1
MUSIC_MANIFEST_NAME = "training-manifest.json"
DATASET_VERSION = "music-long-gaps-v1"
DEFAULT_SEED = 2_026_0914
DEFAULT_MAX_PARENTS_PER_SOURCE = 10
DEFAULT_MAX_TASK_BYTES = 2 * 1024**3
DEFAULT_MIN_FREE_BYTES = 100 * 1024**3
TARGET_MUSIC_RMS_DBFS = -24.0
MUSIC_RMS_TOLERANCE_DB = 0.25
MUSIC_PEAK_LIMIT = 0.99
EXCERPT_MIN_FRAMES = 30 * SAMPLE_RATE
EXCERPT_TARGET_FRAMES = 45 * SAMPLE_RATE
EXCERPT_MAX_FRAMES = 60 * SAMPLE_RATE
GAP_DURATIONS_SECONDS = (10, 30, 60)
GAP_DURATIONS_FRAMES = tuple(value * SAMPLE_RATE for value in GAP_DURATIONS_SECONDS)
VARIANTS = ("leading_music", "silence_between", "music_between", "silence_music_between")

DEFAULT_BUNDLES = {
    "lotusdis": Path(
        "/Volumes/CacheDisk/dev-cache/diarization-data-verification/releases/lotusdis-con123-strict-v3/training-bundle"
    ),
    "simsamu": Path(
        "/Volumes/CacheDisk/dev-cache/diarization-data-verification/releases/simsamu-source-bound-v2/training-bundle"
    ),
    "chime6": Path(
        "/Volumes/CacheDisk/dev-cache/diarization-data-verification/releases/chime6-u06-ch1-v1/training-bundle"
    ),
    "notsofar": Path(
        "/Volumes/CacheDisk/dev-cache/diarization-data-verification/releases/"
        "notsofar-real-source-bound-v4/training-bundle"
    ),
    "aishell4": Path(
        "/Volumes/CacheDisk/dev-cache/diarization-data-verification/releases/aishell4-source-bound-v1/training-bundle"
    ),
}
DEFAULT_MUSIC_POOL = Path(
    "/Volumes/CacheDisk/dev-cache/diarization-data-verification/augmentation/musan-instrumental-v1"
)
DEFAULT_OUTPUT = Path("/Volumes/CacheDisk/dev-cache/diarization-data-verification/augmentation/music-long-gaps-v1")


Kind = Literal["speech", "silence", "music"]


class MusicSequenceError(PreparationError):
    """A bounded long-gap preparation failure."""

    def __init__(self, message: str, details: Mapping[str, object] | None = None) -> None:
        super().__init__(message, dict(details or {}))


@dataclass(frozen=True, slots=True)
class UemInterval:
    """One sample-exact source UEM region."""

    recording_id: str
    start_frame: int
    end_frame: int
    start_text: str
    end_text: str


@dataclass(frozen=True, slots=True)
class SourceLabel:
    """One source RTTM speaker row represented on the sample clock."""

    recording_id: str
    speaker: str
    start_frame: int
    end_frame: int
    start_text: str
    end_text: str


@dataclass(frozen=True, slots=True)
class ParentRecording:
    """Verified immutable source recording and its existing annotations."""

    source: str
    parent_id: str
    bundle_root: Path
    bundle_path: Path
    bundle_sha256: str
    bundle_manifest_sha256: str
    wav_scp_sha256: str
    rttm_manifest_sha256: str
    uem_manifest_sha256: str
    audio_path: Path
    audio_sha256: str
    audio_size: int
    rttm_sha256: str
    uem_sha256: str
    sample_rate: int
    channels: int
    frames: int
    uem: tuple[UemInterval, ...]
    labels: tuple[SourceLabel, ...]


@dataclass(frozen=True, slots=True)
class MusicTrack:
    """Verified no-vocals track from the instrumental pool."""

    track_id: str
    path: Path
    audio_sha256: str
    pcm_sha256: str
    source_sha256: str
    source_path: str | None
    split: str
    no_vocals: bool
    sample_rate: int
    channels: int
    frames: int
    manifest_record: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SpeechExcerpt:
    """One selected speech-bearing source interval."""

    start_frame: int
    end_frame: int
    speech_frames: int


@dataclass(frozen=True, slots=True)
class MusicSelection:
    """One contiguous music read and its deterministic gain."""

    track: MusicTrack
    start_frame: int
    end_frame: int
    gain_linear: float
    gain_db: float
    source_rms_dbfs: float
    output_rms_dbfs: float
    source_peak: float
    output_peak: float


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """One output segment in sequence order."""

    kind: Kind
    output_start_frame: int
    output_end_frame: int
    excerpt_index: int | None = None
    music: MusicSelection | None = None


@dataclass(frozen=True, slots=True)
class OutputLabel:
    """One clipped and shifted output speaker row."""

    recording_id: str
    speaker: str
    source_speaker: str
    source_start_frame: int
    source_end_frame: int
    output_start_frame: int
    output_end_frame: int


def _fail(message: str, **details: object) -> MusicSequenceError:
    """Build a typed preparation failure."""

    return MusicSequenceError(message, details)


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise _fail(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _safe_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{label} must be an integer >= {minimum}")
    return value


def _safe_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise _fail(f"{label} must be a non-empty relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise _fail(f"{label} must be a safe relative path", path=str(relative))
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise _fail(f"{label} escapes its owned root", path=str(relative)) from error
    return path


def _parse_decimal(value: object, label: str) -> tuple[Decimal, str]:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise _fail(f"{label} must be a finite decimal")
    text = str(value)
    try:
        decimal = Decimal(text)
    except (InvalidOperation, ValueError) as error:
        raise _fail(f"{label} is not a decimal", value=text) from error
    if not decimal.is_finite():
        raise _fail(f"{label} must be finite", value=text)
    return decimal, text


def _seconds_to_frame(value: object, label: str, *, rounding: str) -> tuple[int, str]:
    decimal, text = _parse_decimal(value, label)
    if decimal < 0:
        raise _fail(f"{label} must not be negative", value=text)
    scaled = decimal * SAMPLE_RATE
    mode = ROUND_FLOOR if rounding == "floor" else ROUND_CEILING
    frame = int(scaled.to_integral_value(rounding=mode))
    return frame, text


def _frame_seconds(frame: int) -> str:
    return f"{frame / SAMPLE_RATE:.9f}"


def _stable_int(seed: int, *parts: object) -> int:
    encoded = ":".join(str(part) for part in (seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return result or "item"


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail(f"{label} is unreadable", path=str(path)) from error
    if not isinstance(payload, Mapping):
        raise _fail(f"{label} must be a JSON object", path=str(path))
    return payload


def _read_manifest_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _fail(f"{label} is unreadable", path=str(path)) from error


def _parse_uem(path: Path) -> dict[str, tuple[UemInterval, ...]]:
    by_recording: dict[str, list[UemInterval]] = {}
    for line_number, line in enumerate(_read_manifest_text(path, "all.uem").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 4:
            raise _fail("all.uem row must have four fields", line=line_number)
        recording_id, _channel, start_text, end_text = fields
        start_frame, start_raw = _seconds_to_frame(start_text, "UEM start", rounding="floor")
        end_frame, end_raw = _seconds_to_frame(end_text, "UEM end", rounding="ceil")
        if not recording_id or end_frame <= start_frame:
            raise _fail("all.uem row has invalid bounds", line=line_number)
        by_recording.setdefault(recording_id, []).append(
            UemInterval(recording_id, start_frame, end_frame, start_raw, end_raw)
        )
    return {
        key: tuple(sorted(value, key=lambda item: (item.start_frame, item.end_frame)))
        for key, value in by_recording.items()
    }


def _parse_rttm(path: Path) -> dict[str, tuple[SourceLabel, ...]]:
    by_recording: dict[str, list[SourceLabel]] = {}
    for line_number, line in enumerate(_read_manifest_text(path, "all.rttm").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8 or fields[0].upper() != "SPEAKER":
            raise _fail("all.rttm row is not a SPEAKER row", line=line_number)
        recording_id, start_text, duration_text, speaker = fields[1], fields[3], fields[4], fields[7]
        start_decimal, start_raw = _parse_decimal(start_text, "RTTM start")
        duration_decimal, _duration_raw = _parse_decimal(duration_text, "RTTM duration")
        if start_decimal < 0 or duration_decimal <= 0 or not recording_id or not speaker or speaker == "<NA>":
            raise _fail("all.rttm row has invalid activity", line=line_number)
        end_decimal = start_decimal + duration_decimal
        start_frame, _ = _seconds_to_frame(start_raw, "RTTM start", rounding="floor")
        end_frame, _ = _seconds_to_frame(end_decimal, "RTTM end", rounding="ceil")
        if end_frame <= start_frame:
            raise _fail("all.rttm row has no sample-clock activity", line=line_number)
        by_recording.setdefault(recording_id, []).append(
            SourceLabel(recording_id, speaker, start_frame, end_frame, start_raw, _decimal_text(end_decimal))
        )
    return {
        key: tuple(sorted(value, key=lambda item: (item.start_frame, item.end_frame, item.speaker)))
        for key, value in by_recording.items()
    }


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _verify_audio(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    require_flac: bool = True,
) -> tuple[int, int, int]:
    if not path.is_file():
        raise _fail(f"{label} does not exist", path=str(path))
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise _fail(
            f"{label} SHA-256 does not match its manifest", path=str(path), expected=expected_sha256, actual=actual
        )
    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as error:
        raise _fail(f"{label} is not a readable audio file", path=str(path)) from error
    if (require_flac and info.format.upper() != CODEC) or info.subtype.upper() != SUBTYPE:
        raise _fail(
            f"{label} has an unsupported sample identity",
            format=info.format,
            subtype=info.subtype,
        )
    if info.samplerate != SAMPLE_RATE or info.channels != CHANNELS or info.frames <= 0:
        raise _fail(
            f"{label} must be mono 16 kHz with samples",
            sample_rate=info.samplerate,
            channels=info.channels,
            frames=info.frames,
        )
    return info.samplerate, info.channels, info.frames


def _recording_rows(bundle: Mapping[str, object]) -> list[Mapping[str, object]]:
    rows = bundle.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise _fail("training bundle has no recordings")
    result = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise _fail("training bundle recording is not an object")
        recording_id = row.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id or recording_id in seen:
            raise _fail("training bundle recording IDs are invalid or duplicated")
        seen.add(recording_id)
        result.append(row)
    return result


def _verify_bundle(bundle_root: Path) -> tuple[Mapping[str, object], list[ParentRecording]]:
    root = bundle_root.resolve()
    bundle_path = root / "bundle.json"
    bundle = _read_json(bundle_path, "bundle.json")
    if bundle.get("schema") != "speakrs-training-bundle-v1":
        raise _fail("unsupported training bundle schema", path=str(bundle_path))
    manifests = bundle.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != {"wav_scp", "rttm", "uem"}:
        raise _fail("training bundle manifest map is incomplete", path=str(bundle_path))
    manifest_paths = {"wav_scp": root / "wav.scp", "rttm": root / "all.rttm", "uem": root / "all.uem"}
    for key, path in manifest_paths.items():
        expected = _require_digest(manifests.get(key), f"bundle manifests.{key}")
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            raise _fail("training bundle manifest hash mismatch", file=path.name, expected=expected, actual=actual)
    uem_by_recording = _parse_uem(manifest_paths["uem"])
    labels_by_recording = _parse_rttm(manifest_paths["rttm"])
    bundle_ids = {str(row["recording_id"]) for row in _recording_rows(bundle)}
    unknown_uem = sorted(set(uem_by_recording) - bundle_ids)
    unknown_labels = sorted(set(labels_by_recording) - bundle_ids)
    if unknown_uem or unknown_labels:
        raise _fail(
            "training bundle annotations contain unknown recordings",
            unknown_uem=unknown_uem,
            unknown_rttm=unknown_labels,
        )
    wav_rows: dict[str, str] = {}
    for line_number, line in enumerate(_read_manifest_text(manifest_paths["wav_scp"], "wav.scp").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in wav_rows:
            raise _fail("wav.scp row is malformed or duplicated", line=line_number)
        wav_rows[fields[0]] = fields[1]

    bundle_sha256 = sha256_file(bundle_path)
    source_names: set[str] = set()
    recordings: list[ParentRecording] = []
    for row in _recording_rows(bundle):
        parent_id = str(row["recording_id"])
        source = row.get("source")
        if not isinstance(source, str) or not source:
            raise _fail("training bundle recording source is missing", parent_id=parent_id)
        source_names.add(source)
        audio_relative = row.get("audio_path")
        audio_path = _safe_path(root, audio_relative, "recording audio_path")
        expected_audio = _require_digest(row.get("audio_sha256"), "recording audio_sha256")
        sample_rate, channels, frames = _verify_audio(audio_path, expected_audio, f"recording {parent_id} audio")
        if parent_id not in wav_rows:
            raise _fail("recording is missing from wav.scp", parent_id=parent_id)
        scp_path = wav_rows[parent_id]
        scp_basename = Path(scp_path).name
        if scp_basename != audio_path.name:
            raise _fail("wav.scp path does not identify the bundle audio", parent_id=parent_id)
        uem = uem_by_recording.get(parent_id, ())
        labels = labels_by_recording.get(parent_id, ())
        if not uem or not labels:
            continue
        for region in uem:
            if region.end_frame > frames:
                raise _fail("UEM exceeds decoded source audio", parent_id=parent_id)
        for label in labels:
            if label.end_frame > frames:
                raise _fail("RTTM exceeds decoded source audio", parent_id=parent_id)
        rttm_sha256 = _require_digest(row.get("rttm_sha256"), "recording rttm_sha256")
        uem_sha256 = _require_digest(row.get("uem_sha256"), "recording uem_sha256")
        label_dir = root / "labels" / source
        rttm_file = label_dir / f"{rttm_sha256}.rttm"
        uem_file = label_dir / f"{uem_sha256}.uem"
        if rttm_file.is_file() and sha256_file(rttm_file) != rttm_sha256:
            raise _fail("recording RTTM object hash mismatch", parent_id=parent_id)
        if uem_file.is_file() and sha256_file(uem_file) != uem_sha256:
            raise _fail("recording UEM object hash mismatch", parent_id=parent_id)
        recordings.append(
            ParentRecording(
                source=source,
                parent_id=parent_id,
                bundle_root=root,
                bundle_path=bundle_path,
                bundle_sha256=bundle_sha256,
                bundle_manifest_sha256=bundle_sha256,
                wav_scp_sha256=_require_digest(manifests["wav_scp"], "bundle wav_scp hash"),
                rttm_manifest_sha256=_require_digest(manifests["rttm"], "bundle rttm hash"),
                uem_manifest_sha256=_require_digest(manifests["uem"], "bundle uem hash"),
                audio_path=audio_path,
                audio_sha256=expected_audio,
                audio_size=audio_path.stat().st_size,
                rttm_sha256=rttm_sha256,
                uem_sha256=uem_sha256,
                sample_rate=sample_rate,
                channels=channels,
                frames=frames,
                uem=uem,
                labels=labels,
            )
        )
    if len(source_names) != 1:
        raise _fail(
            "each source bundle must contain exactly one source", path=str(bundle_path), sources=sorted(source_names)
        )
    return bundle, recordings


def load_parent_recordings(bundle_paths: Mapping[str, Path] | Sequence[Path]) -> tuple[ParentRecording, ...]:
    """Verify supplied training bundles and return their labelled recordings."""

    paths = (
        list(bundle_paths.items())
        if isinstance(bundle_paths, Mapping)
        else [(str(index), path) for index, path in enumerate(bundle_paths)]
    )
    all_recordings: list[ParentRecording] = []
    seen: set[tuple[str, str]] = set()
    for _name, value in paths:
        _bundle, recordings = _verify_bundle(Path(value))
        for recording in recordings:
            identity = (recording.source, recording.parent_id)
            if identity in seen:
                raise _fail(
                    "duplicate source parent across bundles", source=recording.source, parent_id=recording.parent_id
                )
            seen.add(identity)
            all_recordings.append(recording)
    return tuple(sorted(all_recordings, key=lambda item: (item.source, item.parent_id)))


def _pool_manifest_path(pool_root: Path) -> Path:
    path = pool_root / MUSIC_MANIFEST_NAME
    if not path.is_file():
        raise _fail("instrumental pool manifest does not exist", path=str(path))
    return path


def load_music_tracks(pool_root: Path) -> tuple[MusicTrack, ...]:
    """Load and verify train/no-vocals music tracks and their source hashes."""

    root = Path(pool_root).resolve()
    manifest_path = _pool_manifest_path(root)
    try:
        manifest = load_music_manifest(manifest_path, expected_sample_rate=SAMPLE_RATE)
    except MusicAugmentationError as error:
        raise _fail("instrumental pool manifest failed its strict schema checks", path=str(manifest_path)) from error
    tracks: list[MusicTrack] = []
    for track in manifest.eligible_tracks:
        item = {
            "track_id": track.track_id,
            "path": track.relative_path,
            "sha256": track.sha256,
            "pcm_sha256": track.pcm_sha256,
            "sample_count": track.sample_count,
            "sample_rate": track.sample_rate,
            "channels": track.channels,
            "split": track.split,
            "artist": track.artist,
            "vocals": track.vocals,
            "source_member": track.source_member,
            "source_sha256": track.source_sha256,
            "license": dict(track.license),
        }
        tracks.append(
            MusicTrack(
                track_id=track.track_id,
                path=track.path,
                audio_sha256=track.sha256,
                pcm_sha256=track.pcm_sha256,
                source_sha256=track.source_sha256,
                source_path=track.source_member,
                split=track.split,
                no_vocals=True,
                sample_rate=track.sample_rate,
                channels=track.channels,
                frames=track.sample_count,
                manifest_record=dict(item),
            )
        )
    eligible = [track for track in tracks if track.frames >= GAP_DURATIONS_FRAMES[-1]]
    if not eligible:
        raise _fail("instrumental pool has no track at least 60 seconds long")
    return tuple(sorted(tracks, key=lambda item: item.track_id))


def _intersects(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _speech_frames(
    parent: ParentRecording,
    start_frame: int,
    end_frame: int,
    label_starts: Sequence[int] | None = None,
) -> int:
    if label_starts is None:
        label_starts = tuple(label.start_frame for label in parent.labels)
    first = max(0, bisect.bisect_left(label_starts, start_frame) - 1)
    last = bisect.bisect_left(label_starts, end_frame)
    return sum(
        _intersects(start_frame, end_frame, label.start_frame, label.end_frame) for label in parent.labels[first:last]
    )


def _window_candidates(parent: ParentRecording) -> list[SpeechExcerpt]:
    candidates: set[tuple[int, int]] = set()
    label_starts = tuple(label.start_frame for label in parent.labels)
    for region in parent.uem:
        length = region.end_frame - region.start_frame
        if length < EXCERPT_MIN_FRAMES:
            continue
        target = min(EXCERPT_TARGET_FRAMES, EXCERPT_MAX_FRAMES, length)
        latest = region.end_frame - target
        starts = {
            region.start_frame,
            latest,
            region.start_frame + (latest - region.start_frame) // 2,
        }
        for quantile in (1, 3):
            starts.add(region.start_frame + ((latest - region.start_frame) * quantile) // 4)
        overlapping_labels = [
            label
            for label in parent.labels
            if _intersects(region.start_frame, region.end_frame, label.start_frame, label.end_frame) > 0
        ]
        stride = max(1, len(overlapping_labels) // 16)
        for label in overlapping_labels[::stride][:16]:
            if _intersects(region.start_frame, region.end_frame, label.start_frame, label.end_frame) <= 0:
                continue
            centered = label.start_frame - target // 2
            starts.add(max(region.start_frame, min(latest, centered)))
            centered_end = label.end_frame - target // 2
            starts.add(max(region.start_frame, min(latest, centered_end)))
        for start in starts:
            end = start + target
            if start < region.start_frame or end > region.end_frame:
                continue
            speech = _speech_frames(parent, start, end, label_starts)
            if speech > 0:
                candidates.add((start, end))
    return [
        SpeechExcerpt(start, end, _speech_frames(parent, start, end, label_starts))
        for start, end in sorted(candidates)
    ]


def select_speech_excerpts(parent: ParentRecording, *, count: int = 3) -> tuple[SpeechExcerpt, ...]:
    """Select distinct deterministic speech-bearing excerpts within UEM."""

    if isinstance(count, bool) or count <= 0:
        raise ValueError("count must be positive")
    candidates = _window_candidates(parent)
    if len(candidates) < count:
        raise _fail("parent has fewer distinct speech-bearing excerpts than required", parent_id=parent.parent_id)
    selected: list[SpeechExcerpt] = []
    remaining = list(candidates)
    while remaining and len(selected) < count:
        if not selected:
            choice = max(remaining, key=lambda item: (item.speech_frames, -item.start_frame))
        else:
            choice = max(
                remaining,
                key=lambda item: (
                    min(abs(item.start_frame - prior.start_frame) for prior in selected),
                    item.speech_frames,
                    -item.start_frame,
                ),
            )
        selected.append(choice)
        remaining.remove(choice)
    return tuple(sorted(selected, key=lambda item: (item.start_frame, item.end_frame)))


def clip_source_rttm(
    labels: Sequence[SourceLabel],
    *,
    source_start_frame: int,
    source_end_frame: int,
    output_recording_id: str,
    output_start_frame: int,
    parent_id: str,
) -> tuple[OutputLabel, ...]:
    """Clip existing RTTM rows to one excerpt and shift by sample offset."""

    if source_start_frame < 0 or source_end_frame <= source_start_frame or output_start_frame < 0:
        raise ValueError("source and output frame ranges must be positive")
    rows: list[OutputLabel] = []
    for label in labels:
        start = max(label.start_frame, source_start_frame)
        end = min(label.end_frame, source_end_frame)
        if end <= start:
            continue
        rows.append(
            OutputLabel(
                recording_id=output_recording_id,
                speaker=f"{parent_id}::{label.speaker}",
                source_speaker=label.speaker,
                source_start_frame=start,
                source_end_frame=end,
                output_start_frame=output_start_frame + (start - source_start_frame),
                output_end_frame=output_start_frame + (end - source_start_frame),
            )
        )
    return tuple(sorted(rows, key=lambda item: (item.output_start_frame, item.output_end_frame, item.speaker)))


def _music_gain(samples: np.ndarray, track_id: str) -> tuple[np.ndarray, float, float, float, float, float]:
    values = np.asarray(samples, dtype=np.int16).reshape(-1)
    if not len(values):
        raise _fail("music excerpt is empty", track_id=track_id)
    source = values.astype(np.float64) / 32768.0
    if not np.isfinite(source).all():
        raise _fail("music excerpt contains non-finite samples", track_id=track_id)
    source_rms = float(np.sqrt(np.mean(np.square(source))))
    source_peak = float(np.max(np.abs(source)))
    if source_rms <= 0 or source_peak > 1.0:
        raise _fail("music excerpt has invalid RMS or peak", track_id=track_id)
    target_rms = 10 ** (TARGET_MUSIC_RMS_DBFS / 20.0)
    gain_linear = target_rms / source_rms
    if source_peak * gain_linear > MUSIC_PEAK_LIMIT:
        raise _fail(
            "music excerpt cannot reach -24 dBFS without clipping",
            track_id=track_id,
            source_peak=source_peak,
        )
    rendered = np.rint(source * gain_linear * 32768.0).astype(np.int64)
    if np.any(rendered < -32768) or np.any(rendered > 32767):
        raise _fail("music gain would clip PCM16", track_id=track_id)
    output = rendered.astype(np.int16)
    output_values = output.astype(np.float64) / 32768.0
    output_rms = float(np.sqrt(np.mean(np.square(output_values))))
    output_peak = float(np.max(np.abs(output_values)))
    output_dbfs = 20 * math.log10(output_rms) if output_rms > 0 else float("-inf")
    if abs(output_dbfs - TARGET_MUSIC_RMS_DBFS) > MUSIC_RMS_TOLERANCE_DB or output_peak > MUSIC_PEAK_LIMIT:
        raise _fail(
            "music excerpt failed the normalized RMS/peak bound",
            track_id=track_id,
            output_rms_dbfs=output_dbfs,
            output_peak=output_peak,
        )
    gain_db = 20 * math.log10(gain_linear)
    return output, gain_linear, gain_db, 20 * math.log10(source_rms), output_dbfs, source_peak


def _select_music(
    tracks: Sequence[MusicTrack],
    *,
    duration_frames: int,
    seed: int,
    parent_id: str,
    variant: str,
    segment_index: int,
) -> MusicSelection:
    eligible = [track for track in tracks if track.split == "train" and track.frames >= duration_frames]
    if not eligible:
        raise _fail("instrumental pool has no track long enough", duration_frames=duration_frames)
    eligible.sort(key=lambda item: item.track_id)
    track = eligible[_stable_int(seed, parent_id, variant, segment_index) % len(eligible)]
    available = track.frames - duration_frames
    start = _stable_int(seed, "music-offset", parent_id, variant, segment_index, track.track_id) % (available + 1)
    end = start + duration_frames
    try:
        samples = sf.read(str(track.path), start=start, stop=end, dtype="int16", always_2d=False)[0]
    except (OSError, RuntimeError) as error:
        raise _fail("instrumental track excerpt cannot be decoded", track_id=track.track_id) from error
    if len(samples) != duration_frames:
        raise _fail("instrumental track excerpt has the wrong frame count", track_id=track.track_id)
    _rendered, gain_linear, gain_db, source_rms_dbfs, output_rms_dbfs, source_peak = _music_gain(
        samples, track.track_id
    )
    output_peak = float(np.max(np.abs(_rendered.astype(np.float64) / 32768.0)))
    return MusicSelection(
        track=track,
        start_frame=start,
        end_frame=end,
        gain_linear=gain_linear,
        gain_db=gain_db,
        source_rms_dbfs=source_rms_dbfs,
        output_rms_dbfs=output_rms_dbfs,
        source_peak=source_peak,
        output_peak=output_peak,
    )


def _read_music(selection: MusicSelection) -> np.ndarray:
    samples = sf.read(
        str(selection.track.path),
        start=selection.start_frame,
        stop=selection.end_frame,
        dtype="int16",
        always_2d=False,
    )[0]
    rendered, *_rest = _music_gain(samples, selection.track.track_id)
    return rendered


def _variant_segments(
    excerpts: Sequence[SpeechExcerpt],
    music: Sequence[MusicSelection],
    variant: str,
) -> tuple[SegmentPlan, ...]:
    if len(excerpts) != 3:
        raise ValueError("three excerpts are required")
    music_iter = iter(music)
    if variant == "leading_music":
        kinds: tuple[tuple[Kind, int | None], ...] = (
            ("music", None),
            ("speech", 0),
            ("silence", 30 * SAMPLE_RATE),
            ("speech", 1),
            ("speech", 2),
        )
    elif variant == "silence_between":
        kinds = (
            ("speech", 0),
            ("silence", 60 * SAMPLE_RATE),
            ("speech", 1),
            ("music", None),
            ("speech", 2),
        )
    elif variant == "music_between":
        kinds = (
            ("speech", 0),
            ("music", None),
            ("speech", 1),
            ("silence", 10 * SAMPLE_RATE),
            ("speech", 2),
        )
    elif variant == "silence_music_between":
        kinds = (
            ("speech", 0),
            ("silence", 30 * SAMPLE_RATE),
            ("music", None),
            ("speech", 1),
            ("speech", 2),
        )
    else:
        raise ValueError(f"unknown music sequence variant: {variant}")
    segments: list[SegmentPlan] = []
    cursor = 0
    for kind, value in kinds:
        if kind == "speech":
            assert value is not None
            length = excerpts[value].end_frame - excerpts[value].start_frame
            segment = SegmentPlan(kind, cursor, cursor + length, excerpt_index=value)
        elif kind == "silence":
            assert value is not None
            segment = SegmentPlan(kind, cursor, cursor + value)
        else:
            selection = next(music_iter)
            segment = SegmentPlan(kind, cursor, cursor + selection.end_frame - selection.start_frame, music=selection)
        segments.append(segment)
        cursor = segment.output_end_frame
    return tuple(segments)


def _rttm_line(label: OutputLabel) -> str:
    start = _frame_seconds(label.output_start_frame)
    duration = _frame_seconds(label.output_end_frame - label.output_start_frame)
    return f"SPEAKER {label.recording_id} 1 {start} {duration} <NA> <NA> {label.speaker} <NA> <NA>\n"


def _write_json(path: Path, payload: object, before_write: Callable[[], None] | None = None) -> None:
    if before_write is not None:
        before_write()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_audio(
    path: Path,
    segments: Sequence[SegmentPlan],
    parent: ParentRecording,
    excerpts: Sequence[SpeechExcerpt],
    before_write: Callable[[], None] | None = None,
) -> tuple[tuple[OutputLabel, ...], list[dict[str, object]]]:
    labels: list[OutputLabel] = []
    mappings: list[dict[str, object]] = []
    if before_write is not None:
        before_write()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sf.SoundFile(
            str(path),
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            format=CODEC,
            subtype=SUBTYPE,
        ) as destination:
            for segment in segments:
                if segment.kind == "silence":
                    samples = np.zeros(segment.output_end_frame - segment.output_start_frame, dtype=np.int16)
                elif segment.kind == "music":
                    if segment.music is None:
                        raise _fail("music segment has no selection")
                    samples = _read_music(segment.music)
                else:
                    if segment.excerpt_index is None:
                        raise _fail("speech segment has no excerpt")
                    excerpt = excerpts[segment.excerpt_index]
                    try:
                        samples = sf.read(
                            str(parent.audio_path),
                            start=excerpt.start_frame,
                            stop=excerpt.end_frame,
                            dtype="int16",
                            always_2d=False,
                        )[0]
                    except (OSError, RuntimeError) as error:
                        raise _fail("source excerpt cannot be decoded", parent_id=parent.parent_id) from error
                    if len(samples) != excerpt.end_frame - excerpt.start_frame:
                        raise _fail("source excerpt has the wrong frame count", parent_id=parent.parent_id)
                    clipped = clip_source_rttm(
                        parent.labels,
                        source_start_frame=excerpt.start_frame,
                        source_end_frame=excerpt.end_frame,
                        output_recording_id=path.stem,
                        output_start_frame=segment.output_start_frame,
                        parent_id=parent.parent_id,
                    )
                    labels.extend(clipped)
                    mappings.append(
                        {
                            "kind": "speech_excerpt",
                            "source_recording_id": parent.parent_id,
                            "source_audio_sha256": parent.audio_sha256,
                            "source_start_frame": excerpt.start_frame,
                            "source_end_frame": excerpt.end_frame,
                            "output_start_frame": segment.output_start_frame,
                            "output_end_frame": segment.output_end_frame,
                            "mapping": "output_frame = output_start_frame + source_frame - source_start_frame",
                            "source_start_seconds": _frame_seconds(excerpt.start_frame),
                            "source_end_seconds": _frame_seconds(excerpt.end_frame),
                        }
                    )
                    mappings.extend(
                        {
                            "kind": "rttm_interval",
                            "source_recording_id": parent.parent_id,
                            "source_speaker": label.source_speaker,
                            "scoped_speaker": label.speaker,
                            "source_start_frame": label.source_start_frame,
                            "source_end_frame": label.source_end_frame,
                            "output_start_frame": label.output_start_frame,
                            "output_end_frame": label.output_end_frame,
                            "mapping": "output_frame = output_start_frame + source_frame - source_start_frame",
                        }
                        for label in clipped
                    )
                if len(samples) != segment.output_end_frame - segment.output_start_frame:
                    raise _fail("rendered segment has the wrong frame count", kind=segment.kind)
                if before_write is not None:
                    before_write()
                destination.write(np.asarray(samples, dtype=np.int16))
    except (OSError, RuntimeError) as error:
        raise _fail("output audio could not be written", path=str(path)) from error
    return tuple(
        sorted(labels, key=lambda item: (item.output_start_frame, item.output_end_frame, item.speaker))
    ), mappings


def _segment_manifest(segment: SegmentPlan) -> dict[str, object]:
    result: dict[str, object] = {
        "kind": segment.kind,
        "output_start_frame": segment.output_start_frame,
        "output_end_frame": segment.output_end_frame,
        "duration_frames": segment.output_end_frame - segment.output_start_frame,
        "output_start_seconds": _frame_seconds(segment.output_start_frame),
        "output_end_seconds": _frame_seconds(segment.output_end_frame),
    }
    if segment.kind == "speech":
        result["excerpt_index"] = segment.excerpt_index
    if segment.kind == "music":
        assert segment.music is not None
        selection = segment.music
        result.update(
            {
                "track_id": selection.track.track_id,
                "audio_sha256": selection.track.audio_sha256,
                "pcm_sha256": selection.track.pcm_sha256,
                "source_sha256": selection.track.source_sha256,
                "source_path": selection.track.source_path,
                "source_start_frame": selection.start_frame,
                "source_end_frame": selection.end_frame,
                "gain_linear": selection.gain_linear,
                "gain_db": selection.gain_db,
                "source_rms_dbfs": selection.source_rms_dbfs,
                "output_rms_dbfs": selection.output_rms_dbfs,
                "source_peak": selection.source_peak,
                "output_peak": selection.output_peak,
                "looped": False,
            }
        )
    if segment.kind == "silence":
        result["digital_zero"] = True
    return result


def _parent_manifest(parent: ParentRecording) -> dict[str, object]:
    return {
        "source": parent.source,
        "parent_id": parent.parent_id,
        "bundle_path": str(parent.bundle_path),
        "bundle_sha256": parent.bundle_sha256,
        "bundle_manifest_sha256": parent.bundle_manifest_sha256,
        "bundle_manifests": {
            "wav_scp": parent.wav_scp_sha256,
            "all_rttm": parent.rttm_manifest_sha256,
            "all_uem": parent.uem_manifest_sha256,
        },
        "audio_path": str(parent.audio_path),
        "audio_sha256": parent.audio_sha256,
        "audio_size": parent.audio_size,
        "rttm_sha256": parent.rttm_sha256,
        "uem_sha256": parent.uem_sha256,
        "sample_rate": parent.sample_rate,
        "channels": parent.channels,
        "source_frames": parent.frames,
    }


def _identity_payload(
    parents: Sequence[ParentRecording],
    tracks: Sequence[MusicTrack],
    *,
    pool_manifest_sha256: str,
    seed: int,
    max_parents_per_source: int,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "codec": CODEC,
        "subtype": SUBTYPE,
        "seed": seed,
        "max_parents_per_source": max_parents_per_source,
        "gap_durations_seconds": list(GAP_DURATIONS_SECONDS),
        "variants": list(VARIANTS),
        "parents": [
            {
                "source": parent.source,
                "parent_id": parent.parent_id,
                "bundle_sha256": parent.bundle_sha256,
                "audio_sha256": parent.audio_sha256,
            }
            for parent in parents
        ],
        "music_pool_manifest_sha256": pool_manifest_sha256,
        "music_tracks": [
            {
                "track_id": track.track_id,
                "audio_sha256": track.audio_sha256,
                "pcm_sha256": track.pcm_sha256,
                "source_sha256": track.source_sha256,
            }
            for track in tracks
        ],
    }


def _select_parents(recordings: Sequence[ParentRecording], max_per_source: int) -> tuple[ParentRecording, ...]:
    by_source: dict[str, list[ParentRecording]] = {}
    for parent in recordings:
        try:
            select_speech_excerpts(parent)
        except MusicSequenceError:
            continue
        by_source.setdefault(parent.source, []).append(parent)
    selected: list[ParentRecording] = []
    for source in sorted(by_source):
        selected.extend(sorted(by_source[source], key=lambda item: item.parent_id)[:max_per_source])
    if not selected:
        raise _fail("no eligible speech parents were found")
    return tuple(selected)


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def combined_free_bytes(paths: Iterable[Path]) -> int:
    """Return free bytes across distinct filesystems containing the inputs."""

    seen_devices: set[int] = set()
    total = 0
    for path in paths:
        probe = _nearest_existing(Path(path).resolve())
        try:
            device = os.stat(probe).st_dev
            usage = shutil.disk_usage(probe)
        except OSError as error:
            raise _fail("cannot inspect free space", path=str(probe)) from error
        if device in seen_devices:
            continue
        seen_devices.add(device)
        total += usage.free
    return total


def _check_output_audio(path: Path, expected_frames: int) -> None:
    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as error:
        raise _fail("rendered output audio is unreadable", path=str(path)) from error
    if (
        info.format.upper() != CODEC
        or info.subtype.upper() != SUBTYPE
        or info.samplerate != SAMPLE_RATE
        or info.channels != CHANNELS
        or info.frames != expected_frames
    ):
        raise _fail("rendered output audio has the wrong sample identity", path=str(path))


def _write_manifest_files(
    stage: Path,
    outputs: Sequence[dict[str, object]],
    parents: Sequence[ParentRecording],
    tracks: Sequence[MusicTrack],
    identity: Mapping[str, object],
    before_write: Callable[[], None] | None = None,
) -> dict[str, object]:
    wav_rows = "".join(f"{item['recording_id']} {item['audio_path']}\n" for item in outputs)
    rttm_rows = "".join(item["rttm_text"] for item in outputs)
    uem_rows = "".join(item["uem_text"] for item in outputs)
    if before_write is not None:
        before_write()
    (stage / "wav.scp").write_text(wav_rows, encoding="utf-8")
    if before_write is not None:
        before_write()
    (stage / "all.rttm").write_text(rttm_rows, encoding="utf-8")
    if before_write is not None:
        before_write()
    (stage / "all.uem").write_text(uem_rows, encoding="utf-8")
    public_outputs = [
        {key: value for key, value in item.items() if key not in {"rttm_text", "uem_text"}} for item in outputs
    ]
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "state": "ready",
        "dataset_version": DATASET_VERSION,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "codec": CODEC,
        "subtype": SUBTYPE,
        "seed": identity["seed"],
        "max_parents_per_source": identity["max_parents_per_source"],
        "gap_durations_seconds": list(GAP_DURATIONS_SECONDS),
        "variants": list(VARIANTS),
        "parent_count": len(parents),
        "parents": [_parent_manifest(parent) for parent in parents],
        "music_pool": {
            "manifest_sha256": identity["music_pool_manifest_sha256"],
            "tracks": [
                {
                    "track_id": track.track_id,
                    "audio_sha256": track.audio_sha256,
                    "pcm_sha256": track.pcm_sha256,
                    "source_sha256": track.source_sha256,
                    "split": track.split,
                    "no_vocals": track.no_vocals,
                }
                for track in tracks
            ],
        },
        "input_identity": dict(identity),
        "input_identity_sha256": sha256_json(dict(identity)),
        "outputs": public_outputs,
        "manifests": {},
    }
    _write_json(stage / "derivation.json", manifest, before_write)
    manifest["manifests"] = {
        "wav_scp": sha256_file(stage / "wav.scp"),
        "rttm": sha256_file(stage / "all.rttm"),
        "uem": sha256_file(stage / "all.uem"),
    }
    _write_json(stage / "derivation.json", manifest, before_write)
    ready_digest = sha256_file(stage / "derivation.json")
    if before_write is not None:
        before_write()
    (stage / "READY").write_text(
        json.dumps(
            {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "derivation_sha256": ready_digest}, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _existing_ready(output: Path, identity: Mapping[str, object]) -> dict[str, object] | None:
    if not output.exists():
        return None
    manifest_path = output / "derivation.json"
    ready_path = output / "READY"
    if not manifest_path.is_file() or not ready_path.is_file():
        raise _fail("output exists but is incomplete", path=str(output))
    manifest = _read_json(manifest_path, "derivation.json")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("state") != "ready"
    ):
        raise _fail("output exists but is not a ready long-gap release", path=str(output))
    if manifest.get("input_identity_sha256") != sha256_json(dict(identity)):
        raise _fail("output exists for different inputs", path=str(output))
    ready = _read_json(ready_path, "READY")
    if ready.get("derivation_sha256") != sha256_file(manifest_path):
        raise _fail("output READY marker does not match its derivation manifest", path=str(output))
    manifests = manifest.get("manifests")
    if not isinstance(manifests, Mapping):
        raise _fail("ready output has no manifest hashes", path=str(output))
    for key, filename in (("wav_scp", "wav.scp"), ("rttm", "all.rttm"), ("uem", "all.uem")):
        expected = manifests.get(key)
        actual = sha256_file(output / filename)
        if expected != actual:
            raise _fail("ready output manifest hash mismatch", file=filename)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise _fail("ready output has no rendered recordings", path=str(output))
    for item in outputs:
        if not isinstance(item, Mapping):
            raise _fail("ready output recording is not an object", path=str(output))
        audio_path = _safe_path(output, item.get("audio_path"), "ready output audio_path")
        expected_audio = _require_digest(item.get("audio_sha256"), "ready output audio_sha256")
        if sha256_file(audio_path) != expected_audio:
            raise _fail("ready output audio hash mismatch", path=str(audio_path))
        _check_output_audio(audio_path, _safe_int(item.get("frames"), "ready output frames", minimum=1))
    return dict(manifest)


def prepare_music_sequences(
    output: Path = DEFAULT_OUTPUT,
    music_pool: Path = DEFAULT_MUSIC_POOL,
    bundle_paths: Mapping[str, Path] | Sequence[Path] | None = None,
    seed: int = DEFAULT_SEED,
    max_parents_per_source: int = DEFAULT_MAX_PARENTS_PER_SOURCE,
    max_task_bytes: int = DEFAULT_MAX_TASK_BYTES,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> dict[str, object]:
    """Materialise the bounded long-gap dataset and return its ready manifest."""

    if isinstance(max_parents_per_source, bool) or max_parents_per_source <= 0:
        raise ValueError("max_parents_per_source must be positive")
    if isinstance(max_task_bytes, bool) or max_task_bytes <= 0:
        raise ValueError("max_task_bytes must be positive")
    if isinstance(min_free_bytes, bool) or min_free_bytes < 0:
        raise ValueError("min_free_bytes must not be negative")
    output = Path(output).resolve()
    music_pool = Path(music_pool).resolve()
    if bundle_paths is None:
        bundle_paths = DEFAULT_BUNDLES
    parents = load_parent_recordings(bundle_paths)
    selected_parents = _select_parents(parents, max_parents_per_source)
    tracks = load_music_tracks(music_pool)
    pool_manifest_path = _pool_manifest_path(music_pool)
    pool_manifest_sha256 = sha256_file(pool_manifest_path)
    identity = _identity_payload(
        selected_parents,
        tracks,
        pool_manifest_sha256=pool_manifest_sha256,
        seed=seed,
        max_parents_per_source=max_parents_per_source,
    )
    existing = _existing_ready(output, identity)
    if existing is not None:
        return existing
    resource_roots = [output, music_pool, *(parent.bundle_root for parent in selected_parents)]

    def ensure_free_space() -> None:
        available = combined_free_bytes(resource_roots)
        if available < min_free_bytes:
            raise _fail(
                "combined free space is below the preparation bound", available=available, required=min_free_bytes
            )

    ensure_free_space()
    planned_bytes = 0
    plan_rows: list[tuple[ParentRecording, str, tuple[SpeechExcerpt, ...], tuple[SegmentPlan, ...]]] = []
    used_ids: set[str] = set()
    for parent in selected_parents:
        excerpts = select_speech_excerpts(parent)
        for variant in VARIANTS:
            music_durations = {
                "leading_music": (10 * SAMPLE_RATE,),
                "silence_between": (30 * SAMPLE_RATE,),
                "music_between": (60 * SAMPLE_RATE,),
                "silence_music_between": (60 * SAMPLE_RATE,),
            }[variant]
            selections = tuple(
                _select_music(
                    tracks,
                    duration_frames=duration,
                    seed=seed,
                    parent_id=parent.parent_id,
                    variant=variant,
                    segment_index=index,
                )
                for index, duration in enumerate(music_durations)
            )
            segments = _variant_segments(excerpts, selections, variant)
            output_id = f"{_slug(parent.source)}__{_slug(parent.parent_id)}__{variant}"
            if output_id in used_ids:
                output_id = (
                    f"{output_id}__{hashlib.sha256(f'{parent.source}:{parent.parent_id}'.encode()).hexdigest()[:8]}"
                )
            used_ids.add(output_id)
            plan_rows.append((parent, output_id, excerpts, segments))
            planned_bytes += segments[-1].output_end_frame * 2
    planned_bytes += len(plan_rows) * 64 * 1024
    if planned_bytes > max_task_bytes:
        raise _fail(
            "planned long-gap output exceeds the task byte bound",
            planned_bytes=planned_bytes,
            max_task_bytes=max_task_bytes,
        )
    if output.exists():
        raise _fail("output exists but is incomplete", path=str(output))
    stage = output.with_name(f".{output.name}.partial")
    if stage.exists():
        raise _fail("incomplete output staging directory already exists", path=str(stage))
    ensure_free_space()
    stage.mkdir(parents=True)
    stage_created = True
    output_rows: list[dict[str, object]] = []
    try:
        for parent, output_id, excerpts, segments in plan_rows:
            ensure_free_space()
            audio_relative = Path("audio") / f"{output_id}.flac"
            audio_path = stage / audio_relative
            labels, mappings = _write_audio(audio_path, segments, parent, excerpts, ensure_free_space)
            _check_output_audio(audio_path, segments[-1].output_end_frame)
            label_lines = "".join(_rttm_line(label) for label in labels)
            duration = segments[-1].output_end_frame
            output_rows.append(
                {
                    "recording_id": output_id,
                    "source": parent.source,
                    "parent_id": parent.parent_id,
                    "audio_path": audio_relative.as_posix(),
                    "audio_sha256": sha256_file(audio_path),
                    "audio_size": audio_path.stat().st_size,
                    "frames": duration,
                    "duration_seconds": duration / SAMPLE_RATE,
                    "segments": [_segment_manifest(segment) for segment in segments],
                    "source_interval_mappings": mappings,
                    "inserted_intervals": [
                        {
                            "kind": segment.kind,
                            "output_start_frame": segment.output_start_frame,
                            "output_end_frame": segment.output_end_frame,
                            "duration_seconds": (segment.output_end_frame - segment.output_start_frame) / SAMPLE_RATE,
                            "rttm_rows": 0,
                        }
                        for segment in segments
                        if segment.kind in {"silence", "music"}
                    ],
                    "rttm_text": label_lines,
                    "uem_text": f"{output_id} 1 0.000000000 {_frame_seconds(duration)}\n",
                }
            )
        ensure_free_space()
        manifest = _write_manifest_files(stage, output_rows, selected_parents, tracks, identity, ensure_free_space)
        ensure_free_space()
        stage.replace(output)
        return manifest
    except BaseException:
        if stage_created and stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _parse_bundle_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--bundle must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--bundle must use NAME=PATH")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    """Build the long-gap preparation CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--music-pool", type=Path, default=DEFAULT_MUSIC_POOL)
    parser.add_argument("--bundle", action="append", type=_parse_bundle_argument, metavar="NAME=PATH")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-parents-per-source", type=int, default=DEFAULT_MAX_PARENTS_PER_SOURCE)
    parser.add_argument("--max-task-bytes", type=int, default=DEFAULT_MAX_TASK_BYTES)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run bounded long-gap preparation and print the derivation summary."""

    args = build_parser().parse_args(argv)
    bundles: Mapping[str, Path] | None = None
    if args.bundle:
        bundles = dict(args.bundle)
    try:
        manifest = prepare_music_sequences(
            output=args.output,
            music_pool=args.music_pool,
            bundle_paths=bundles,
            seed=args.seed,
            max_parents_per_source=args.max_parents_per_source,
            max_task_bytes=args.max_task_bytes,
            min_free_bytes=args.min_free_bytes,
        )
    except MusicSequenceError as error:
        print(json.dumps({"ok": False, "error": str(error), "details": error.details}, indent=2, sort_keys=True))
        return 1
    result = {
        "ok": True,
        "output": str(args.output),
        "derivation_manifest": str(Path(args.output) / "derivation.json"),
        "parent_count": manifest.get("parent_count"),
        "output_count": len(manifest.get("outputs", [])),
        "duration_hours": sum(float(item["duration_seconds"]) for item in manifest.get("outputs", [])) / 3600,
        "state": manifest.get("state"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if __package__ in {None, ""}:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from recipes.speakrs.large import music_sequences as _self

        raise SystemExit(_self.main())
    raise SystemExit(main())
