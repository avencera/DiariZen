"""Deterministic, bounded instrumental-music augmentation for training audio.

The augmentation is deliberately a loader-time transform.  It keeps the
source recording and its RTTM labels as the authority, and derives every draw
from the recording identity and exact chunk sample bounds.  A worker, rank, or
access order therefore cannot change a chunk's realized transform.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf


MUSIC_AUGMENTATION_SCHEMA = "speakrs-music-augmentation-v1"
MUSIC_MANIFEST_SCHEMA = "speakrs-instrumental-music-v1"
PROPORTION_TOTAL = 1_000_000
EXPECTED_SAMPLE_RATE = 16_000
MAX_MUSIC_CROP_ATTEMPTS = 8
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_MODULUS = 1 << 64
MusicMode = Literal["clean", "mixed", "mixed_no_speech", "music_only"]


class MusicAugmentationError(ValueError):
    """Raised when an augmentation configuration, manifest, or sample is invalid."""


class UnusableMusicCropError(MusicAugmentationError):
    """Raised when one verified music crop cannot satisfy the requested policy."""


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise MusicAugmentationError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MusicAugmentationError(f"{field_name} must be a finite number") from error
    if not math.isfinite(result):
        raise MusicAugmentationError(f"{field_name} must be a finite number")
    return result


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MusicAugmentationError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MusicAugmentationError(f"{field_name} must be a nonnegative integer")
    return value


def _sha256(value: Path) -> str:
    digest = hashlib.sha256()
    try:
        with value.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise MusicAugmentationError(f"cannot read music file: {value}") from error
    return digest.hexdigest()


def _validated_hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MusicAugmentationError(f"{field_name} must be a SHA-256 hex digest")
    return value.lower()


@dataclass(frozen=True, slots=True)
class MusicAugmentationProportions:
    """Integer-million probabilities for clean, mixed, and music-only chunks."""

    clean: int = PROPORTION_TOTAL
    mixed: int = 0
    music_only: int = 0

    def __post_init__(self) -> None:
        values = (self.clean, self.mixed, self.music_only)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise MusicAugmentationError("augmentation proportions must be nonnegative integers")
        if sum(values) != PROPORTION_TOTAL:
            raise MusicAugmentationError(f"augmentation proportions must sum to {PROPORTION_TOTAL}")

    def as_dict(self) -> dict[str, int]:
        """Return the stable JSON names for the three probabilities."""

        return {"clean": self.clean, "mixed": self.mixed, "music_only": self.music_only}


@dataclass(frozen=True, slots=True)
class MusicAugmentationConfig:
    """Validated version-one configuration for runtime music augmentation."""

    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    proportions: MusicAugmentationProportions = field(default_factory=MusicAugmentationProportions)
    seed: int = 0
    split: str = "train"
    enabled: bool = True
    min_snr_db: float = 10.0
    max_snr_db: float = 20.0
    min_no_speech_rms_dbfs: float = -30.0
    max_no_speech_rms_dbfs: float = -18.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise MusicAugmentationError("enabled must be a boolean")
        if self.split != "train":
            raise MusicAugmentationError("music augmentation is only valid for split='train'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise MusicAugmentationError("seed must be an integer")
        min_snr = _finite_number(self.min_snr_db, "min_snr_db")
        max_snr = _finite_number(self.max_snr_db, "max_snr_db")
        min_level = _finite_number(self.min_no_speech_rms_dbfs, "min_no_speech_rms_dbfs")
        max_level = _finite_number(self.max_no_speech_rms_dbfs, "max_no_speech_rms_dbfs")
        if min_snr > max_snr:
            raise MusicAugmentationError("min_snr_db must not exceed max_snr_db")
        if min_level > max_level:
            raise MusicAugmentationError("min_no_speech_rms_dbfs must not exceed max_no_speech_rms_dbfs")
        if not isinstance(self.proportions, MusicAugmentationProportions):
            raise MusicAugmentationError("proportions must be MusicAugmentationProportions")

        if self.manifest_path is not None:
            path = Path(self.manifest_path).expanduser().resolve()
            object.__setattr__(self, "manifest_path", path)
        if self.manifest_sha256 is not None:
            object.__setattr__(self, "manifest_sha256", _validated_hash(self.manifest_sha256, "manifest_sha256"))
        if self.enabled:
            if self.manifest_path is None or self.manifest_sha256 is None:
                raise MusicAugmentationError("enabled music augmentation requires manifest_path and manifest_sha256")

    @property
    def base_seed(self) -> int:
        """Return the stable seed used to derive every chunk transform."""

        return self.seed

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible configuration values."""

        return {
            "schema": MUSIC_AUGMENTATION_SCHEMA,
            "enabled": self.enabled,
            "split": self.split,
            "seed": self.seed,
            "manifest_path": None if self.manifest_path is None else self.manifest_path.as_posix(),
            "manifest_sha256": self.manifest_sha256,
            "proportions": self.proportions.as_dict(),
            "mixed_snr_db": {"min": self.min_snr_db, "max": self.max_snr_db},
            "no_speech_rms_dbfs": {
                "min": self.min_no_speech_rms_dbfs,
                "max": self.max_no_speech_rms_dbfs,
            },
        }


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MusicAugmentationError(f"{field_name} must be an object")
    return value


def _range_value(raw: object, field_name: str, *, default: tuple[float, float]) -> tuple[float, float]:
    if raw is None:
        return default
    if isinstance(raw, Mapping):
        low = raw.get("min", raw.get("min_db", raw.get("min_dbfs")))
        high = raw.get("max", raw.get("max_db", raw.get("max_dbfs")))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)) and len(raw) == 2:
        low, high = raw
    else:
        raise MusicAugmentationError(f"{field_name} must contain exactly two bounds")
    if low is None or high is None:
        raise MusicAugmentationError(f"{field_name} must contain min and max")
    low_value = _finite_number(low, f"{field_name}.min")
    high_value = _finite_number(high, f"{field_name}.max")
    if low_value > high_value:
        raise MusicAugmentationError(f"{field_name}.min must not exceed max")
    return low_value, high_value


def load_music_augmentation_config(path: str | Path) -> MusicAugmentationConfig:
    """Load and validate a version-one augmentation configuration JSON file."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MusicAugmentationError(f"cannot load music augmentation config: {config_path}") from error
    raw = _mapping(raw, "music augmentation config")
    if raw.get("schema") != MUSIC_AUGMENTATION_SCHEMA:
        raise MusicAugmentationError("music augmentation config schema is unsupported")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise MusicAugmentationError("enabled must be a boolean")
    split = raw.get("split", "train")
    if not isinstance(split, str):
        raise MusicAugmentationError("split must be a string")
    seed = raw.get("seed", raw.get("base_seed", 0))

    proportions_raw = raw.get("proportions")
    if proportions_raw is None:
        proportions_raw = {
            "clean": raw.get(
                "clean", raw.get("clean_weight", raw.get("clean_ppm", PROPORTION_TOTAL if not enabled else None))
            ),
            "mixed": raw.get("mixed", raw.get("mixed_weight", raw.get("mixed_ppm", 0))),
            "music_only": raw.get("music_only", raw.get("music_only_weight", raw.get("music_only_ppm", 0))),
        }
    proportions_raw = _mapping(proportions_raw, "proportions")
    proportions = MusicAugmentationProportions(
        clean=_nonnegative_integer(
            proportions_raw.get("clean", proportions_raw.get("clean_weight", proportions_raw.get("clean_ppm"))),
            "proportions.clean",
        ),
        mixed=_nonnegative_integer(
            proportions_raw.get("mixed", proportions_raw.get("mixed_weight", proportions_raw.get("mixed_ppm"))),
            "proportions.mixed",
        ),
        music_only=_nonnegative_integer(
            proportions_raw.get(
                "music_only",
                proportions_raw.get("music_only_weight", proportions_raw.get("music_only_ppm")),
            ),
            "proportions.music_only",
        ),
    )

    manifest_value = raw.get("manifest_path", raw.get("manifest"))
    manifest_hash = raw.get("manifest_sha256", raw.get("manifest_hash"))
    if isinstance(manifest_value, Mapping):
        manifest_hash = manifest_value.get("sha256", manifest_hash)
        manifest_value = manifest_value.get("path")
    if manifest_value is None:
        manifest_path = None
    elif isinstance(manifest_value, str) and manifest_value:
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = config_path.parent / manifest_path
        manifest_path = manifest_path.resolve()
    else:
        raise MusicAugmentationError("manifest_path must be a non-empty path")

    snr_raw = raw.get("mixed_snr_db", raw.get("snr_db"))
    if snr_raw is None and ("min_snr_db" in raw or "max_snr_db" in raw):
        snr_raw = {"min": raw.get("min_snr_db"), "max": raw.get("max_snr_db")}
    snr_min, snr_max = _range_value(
        snr_raw,
        "mixed_snr_db",
        default=(10.0, 20.0),
    )
    level_raw = raw.get("no_speech_rms_dbfs", raw.get("silent_music_rms_dbfs"))
    if level_raw is None and ("min_no_speech_rms_dbfs" in raw or "max_no_speech_rms_dbfs" in raw):
        level_raw = {
            "min": raw.get("min_no_speech_rms_dbfs"),
            "max": raw.get("max_no_speech_rms_dbfs"),
        }
    level_min, level_max = _range_value(
        level_raw,
        "no_speech_rms_dbfs",
        default=(-30.0, -18.0),
    )
    return MusicAugmentationConfig(
        manifest_path=manifest_path,
        manifest_sha256=None if manifest_hash is None else _validated_hash(manifest_hash, "manifest_sha256"),
        proportions=proportions,
        seed=seed,
        split=split,
        enabled=enabled,
        min_snr_db=snr_min,
        max_snr_db=snr_max,
        min_no_speech_rms_dbfs=level_min,
        max_no_speech_rms_dbfs=level_max,
    )


@dataclass(frozen=True, slots=True)
class MusicTrack:
    """One verified mono instrumental track from the music manifest."""

    track_id: str
    path: Path
    relative_path: str
    sha256: str
    pcm_sha256: str
    sample_count: int
    sample_rate: int
    channels: int
    split: str
    artist: str
    vocals: str
    source_member: str
    source_sha256: str
    license_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.track_id or not isinstance(self.track_id, str):
            raise MusicAugmentationError("music track_id must be a non-empty string")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise MusicAugmentationError("music track path must be an absolute path")
        if not self.relative_path or Path(self.relative_path).is_absolute():
            raise MusicAugmentationError("music track path must be relative in the manifest")
        object.__setattr__(self, "sha256", _validated_hash(self.sha256, "track.sha256"))
        object.__setattr__(self, "pcm_sha256", _validated_hash(self.pcm_sha256, "track.pcm_sha256"))
        object.__setattr__(self, "source_sha256", _validated_hash(self.source_sha256, "track.source_sha256"))
        _positive_integer(self.sample_count, "track.sample_count")
        if self.sample_rate != EXPECTED_SAMPLE_RATE or self.channels != 1:
            raise MusicAugmentationError("music tracks must be mono 16 kHz audio")
        if self.split not in {"train", "validation"}:
            raise MusicAugmentationError("track.split must be train or validation")
        if not isinstance(self.artist, str) or not self.artist:
            raise MusicAugmentationError("track.artist must be a non-empty string")
        if self.vocals != "N":
            raise MusicAugmentationError("music manifest contains a track that is not marked vocals='N'")
        if not isinstance(self.source_member, str) or not self.source_member:
            raise MusicAugmentationError("track.source_member must be a non-empty string")
        if not isinstance(self.license_metadata, Mapping) or not self.license_metadata:
            raise MusicAugmentationError("track.license must contain metadata")
        object.__setattr__(self, "license_metadata", dict(self.license_metadata))

    @property
    def license(self) -> Mapping[str, Any]:
        """Return the source licence metadata supplied by the manifest."""

        return self.license_metadata


@dataclass(frozen=True, slots=True)
class MusicManifest:
    """Validated instrumental music manifest and its resolved track records."""

    path: Path
    sample_rate: int
    tracks: tuple[MusicTrack, ...]

    @property
    def eligible_tracks(self) -> tuple[MusicTrack, ...]:
        """Return train-split instrumental tracks in stable track-id order."""

        return tuple(
            sorted((track for track in self.tracks if track.split == "train"), key=lambda item: item.track_id)
        )


def _decoded_pcm_identity(path: Path) -> tuple[str, int, bool]:
    digest = hashlib.sha256()
    sample_count = 0
    has_energy = False
    try:
        with sf.SoundFile(str(path)) as audio:
            if audio.samplerate != EXPECTED_SAMPLE_RATE or audio.channels != 1:
                raise MusicAugmentationError("music track must decode as mono 16 kHz audio")
            while True:
                block = audio.read(65_536, dtype="int16", always_2d=True)
                if len(block) == 0:
                    break
                digest.update(np.ascontiguousarray(block).tobytes())
                sample_count += len(block)
                has_energy = has_energy or bool(np.any(block))
    except MusicAugmentationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise MusicAugmentationError(f"music track is unreadable: {path}") from error
    if sample_count <= 0:
        raise MusicAugmentationError(f"music track is empty: {path}")
    return digest.hexdigest(), sample_count, has_energy


def _manifest_track_path(root: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise MusicAugmentationError("track.path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise MusicAugmentationError("music track path traversal is not allowed")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise MusicAugmentationError("music track path escapes the manifest directory")
    return value, resolved


def _track_vocals(value: object) -> str:
    if value == "N" and isinstance(value, str):
        return "N"
    raise MusicAugmentationError("music manifest track vocals must be exactly 'N'")


def load_music_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_sample_rate: int = EXPECTED_SAMPLE_RATE,
) -> MusicManifest:
    """Load, hash, and validate every eligible track in a music manifest."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise MusicAugmentationError(f"music manifest does not exist: {manifest_path}")
    if expected_sha256 is not None:
        expected_sha256 = _validated_hash(expected_sha256, "manifest_sha256")
        actual_sha256 = _sha256(manifest_path)
        if actual_sha256 != expected_sha256:
            raise MusicAugmentationError("music manifest SHA-256 differs from the config")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MusicAugmentationError(f"cannot load music manifest: {manifest_path}") from error
    raw = _mapping(raw, "music manifest")
    if raw.get("schema") != MUSIC_MANIFEST_SCHEMA:
        raise MusicAugmentationError("music manifest schema is unsupported")
    sample_rate = raw.get("sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate != expected_sample_rate:
        raise MusicAugmentationError("music manifest sample_rate does not match the dataset")
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise MusicAugmentationError("music augmentation requires a 16 kHz dataset")
    tracks_raw = raw.get("tracks")
    if not isinstance(tracks_raw, list) or not tracks_raw:
        raise MusicAugmentationError("music manifest tracks must be a non-empty array")

    root = manifest_path.parent.resolve()
    tracks: list[MusicTrack] = []
    track_ids: set[str] = set()
    for index, item in enumerate(tracks_raw):
        item = _mapping(item, f"tracks[{index}]")
        track_id = item.get("track_id")
        if not isinstance(track_id, str) or not track_id or track_id in track_ids:
            raise MusicAugmentationError("music track IDs must be unique non-empty strings")
        track_ids.add(track_id)
        relative_path, resolved_path = _manifest_track_path(root, item.get("path"))
        track = MusicTrack(
            track_id=track_id,
            path=resolved_path,
            relative_path=relative_path,
            sha256=item.get("sha256"),
            pcm_sha256=item.get("pcm_sha256"),
            sample_count=_positive_integer(item.get("sample_count"), f"tracks[{index}].sample_count"),
            sample_rate=_positive_integer(item.get("sample_rate"), f"tracks[{index}].sample_rate"),
            channels=_positive_integer(item.get("channels"), f"tracks[{index}].channels"),
            split=item.get("split"),
            artist=item.get("artist"),
            vocals=_track_vocals(item.get("vocals")),
            source_member=item.get("source_member"),
            source_sha256=item.get("source_sha256"),
            license_metadata=item.get("license"),
        )
        if track.split == "train":
            if not track.path.is_file():
                raise MusicAugmentationError(f"music track does not exist: {track.path}")
            if _sha256(track.path) != track.sha256:
                raise MusicAugmentationError(f"music track file SHA-256 differs: {track.track_id}")
            pcm_sha256, actual_count, has_energy = _decoded_pcm_identity(track.path)
            if actual_count != track.sample_count:
                raise MusicAugmentationError(f"music track sample_count differs: {track.track_id}")
            if pcm_sha256 != track.pcm_sha256:
                raise MusicAugmentationError(f"music track decoded PCM SHA-256 differs: {track.track_id}")
            if not has_energy:
                raise MusicAugmentationError(f"music track has zero energy: {track.track_id}")
        tracks.append(track)
    manifest = MusicManifest(path=manifest_path, sample_rate=sample_rate, tracks=tuple(tracks))
    if not manifest.eligible_tracks:
        raise MusicAugmentationError("music manifest has no eligible train tracks")
    return manifest


def speech_reference_mask(
    chunked_annotations: object,
    *,
    chunk_start_sample: int,
    chunk_end_sample: int,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
) -> np.ndarray:
    """Return the exact sample union covered by chunked RTTM annotations."""

    if isinstance(chunk_start_sample, bool) or not isinstance(chunk_start_sample, int):
        raise MusicAugmentationError("chunk_start_sample must be an integer")
    if isinstance(chunk_end_sample, bool) or not isinstance(chunk_end_sample, int):
        raise MusicAugmentationError("chunk_end_sample must be an integer")
    if chunk_start_sample < 0 or chunk_end_sample <= chunk_start_sample:
        raise MusicAugmentationError("chunk sample bounds must be positive and ordered")
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise MusicAugmentationError("speech reference mask requires a 16 kHz sample rate")
    mask = np.zeros(chunk_end_sample - chunk_start_sample, dtype=bool)
    if chunked_annotations is None:
        return mask

    try:
        annotations = tuple(chunked_annotations)
    except TypeError as error:
        raise MusicAugmentationError("chunked_annotations must be iterable") from error
    for annotation in annotations:
        if isinstance(annotation, Mapping):
            start_value = annotation.get("start")
            end_value = annotation.get("end")
        elif isinstance(annotation, np.void) and annotation.dtype.names is not None:
            start_value = annotation["start"] if "start" in annotation.dtype.names else None
            end_value = annotation["end"] if "end" in annotation.dtype.names else None
        elif isinstance(annotation, Sequence) and not isinstance(annotation, (str, bytes, bytearray)):
            if len(annotation) == 2:
                start_value, end_value = annotation
            elif len(annotation) >= 4:
                start_value, end_value = annotation[1:3]
            elif len(annotation) == 3:
                start_value, end_value = annotation[:2]
            else:
                start_value = end_value = None
        else:
            start_value = getattr(annotation, "start", None)
            end_value = getattr(annotation, "end", None)
        if start_value is None or end_value is None:
            continue
        start = _finite_number(start_value, "annotation.start")
        end = _finite_number(end_value, "annotation.end")
        if end <= start:
            continue
        absolute_start = int(start * sample_rate)
        absolute_end = int(end * sample_rate)
        clipped_start = max(chunk_start_sample, absolute_start)
        clipped_end = min(chunk_end_sample, absolute_end)
        if clipped_end > clipped_start:
            mask[clipped_start - chunk_start_sample : clipped_end - chunk_start_sample] = True
    return mask


def _rms(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = values if mask is None else values[..., mask]
    if selected.size == 0:
        return 0.0
    result = float(np.sqrt(np.mean(np.square(selected, dtype=np.float64), dtype=np.float64)))
    if not math.isfinite(result):
        raise MusicAugmentationError("audio contains nonfinite samples")
    return result


def _dbfs(value: float) -> float | None:
    return None if value <= 0 else 20.0 * math.log10(value)


def _uniform(value: int, low: float, high: float) -> float:
    if low == high:
        return low
    fraction = (value + 0.5) / _MODULUS
    return low + ((high - low) * fraction)


@dataclass(frozen=True, slots=True)
class MusicAugmentationReceipt:
    """Auditable realization details for one deterministic chunk transform."""

    mode: MusicMode
    requested_mode: MusicMode
    recording_id: str
    chunk_start_sample: int
    chunk_end_sample: int
    music_track_id: str | None
    music_offset_sample: int | None
    speech_gain: float
    music_gain: float
    global_scale: float
    requested_snr_db: float | None
    measured_snr_db: float | None
    requested_music_rms_dbfs: float | None
    measured_music_rms_dbfs: float | None
    speech_rms: float
    music_rms: float
    peak: float
    attempt_index: int = 0
    attempt_count: int = 1

    def __post_init__(self) -> None:
        if not self.recording_id:
            raise MusicAugmentationError("receipt recording_id must be non-empty")
        if self.chunk_start_sample < 0 or self.chunk_end_sample <= self.chunk_start_sample:
            raise MusicAugmentationError("receipt chunk bounds are invalid")
        if (
            isinstance(self.attempt_index, bool)
            or not isinstance(self.attempt_index, int)
            or isinstance(self.attempt_count, bool)
            or not isinstance(self.attempt_count, int)
            or self.attempt_index < 0
            or self.attempt_count <= 0
            or self.attempt_index >= self.attempt_count
        ):
            raise MusicAugmentationError("receipt crop attempt index/count are invalid")
        for field_name in (
            "speech_gain",
            "music_gain",
            "global_scale",
            "speech_rms",
            "music_rms",
            "peak",
        ):
            value = _finite_number(getattr(self, field_name), f"receipt.{field_name}")
            if value < 0:
                raise MusicAugmentationError(f"receipt.{field_name} must be nonnegative")
        for field_name in (
            "requested_snr_db",
            "measured_snr_db",
            "requested_music_rms_dbfs",
            "measured_music_rms_dbfs",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _finite_number(value, f"receipt.{field_name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible receipt fields for QA and audit logs."""

        return {
            "mode": self.mode,
            "requested_mode": self.requested_mode,
            "recording_id": self.recording_id,
            "chunk_start_sample": self.chunk_start_sample,
            "chunk_end_sample": self.chunk_end_sample,
            "music_track_id": self.music_track_id,
            "music_offset_sample": self.music_offset_sample,
            "speech_gain": self.speech_gain,
            "music_gain": self.music_gain,
            "global_scale": self.global_scale,
            "requested_snr_db": self.requested_snr_db,
            "measured_snr_db": self.measured_snr_db,
            "requested_music_rms_dbfs": self.requested_music_rms_dbfs,
            "measured_music_rms_dbfs": self.measured_music_rms_dbfs,
            "speech_rms": self.speech_rms,
            "music_rms": self.music_rms,
            "peak": self.peak,
            "attempt_index": self.attempt_index,
            "attempt_count": self.attempt_count,
        }


class MusicAugmenter:
    """Apply one fixed, hash-derived music transform per training chunk."""

    def __init__(self, config: MusicAugmentationConfig, *, sample_rate: int = EXPECTED_SAMPLE_RATE):
        if not isinstance(config, MusicAugmentationConfig):
            raise MusicAugmentationError("config must be MusicAugmentationConfig")
        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise MusicAugmentationError("music augmentation requires a 16 kHz dataset")
        self.config = config
        self.sample_rate = sample_rate
        self.manifest = (
            None
            if not config.enabled
            else load_music_manifest(
                config.manifest_path,
                expected_sha256=config.manifest_sha256,
                expected_sample_rate=sample_rate,
            )
        )
        self._tracks = () if self.manifest is None else self.manifest.eligible_tracks

    @classmethod
    def from_config(cls, path: str | Path, *, sample_rate: int = EXPECTED_SAMPLE_RATE) -> MusicAugmenter:
        """Load a JSON config and construct its verified runtime augmenter."""

        return cls(load_music_augmentation_config(path), sample_rate=sample_rate)

    def _draw(self, purpose: str, recording_id: str, start: int, end: int, *, attempt: int = 0) -> int:
        payload = f"music-augmentation-v1\0{self.config.seed}\0{purpose}\0{recording_id}\0{start}\0{end}"
        if attempt:
            payload += f"\0attempt\0{attempt}"
        return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")

    def _mode(self, recording_id: str, start: int, end: int) -> MusicMode:
        draw = self._draw("mode", recording_id, start, end) % PROPORTION_TOTAL
        if draw < self.config.proportions.clean:
            return "clean"
        if draw < self.config.proportions.clean + self.config.proportions.mixed:
            return "mixed"
        return "music_only"

    def _track_and_offset(
        self,
        recording_id: str,
        start: int,
        end: int,
        *,
        attempt: int = 0,
    ) -> tuple[MusicTrack, int]:
        if not self._tracks:
            raise MusicAugmentationError("music manifest has no eligible train tracks")
        sample_count = end - start
        candidates = tuple(track for track in self._tracks if track.sample_count >= sample_count)
        if not candidates:
            raise MusicAugmentationError(
                f"no music track is long enough for the requested chunk ({sample_count} samples)"
            )
        track = candidates[self._draw("track", recording_id, start, end, attempt=attempt) % len(candidates)]
        max_offset = track.sample_count - sample_count
        offset = self._draw("offset", recording_id, start, end, attempt=attempt) % (max_offset + 1)
        return track, offset

    def _read_music(self, track: MusicTrack, offset: int, sample_count: int) -> np.ndarray:
        try:
            data, sample_rate = sf.read(
                str(track.path),
                start=offset,
                stop=offset + sample_count,
                dtype="float64",
                always_2d=True,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise MusicAugmentationError(f"cannot read music track: {track.track_id}") from error
        if sample_rate != self.sample_rate or data.shape != (sample_count, 1):
            raise MusicAugmentationError(f"music track metadata changed after verification: {track.track_id}")
        music = np.ascontiguousarray(data[:, 0])
        if not np.all(np.isfinite(music)):
            raise MusicAugmentationError(f"music track contains nonfinite samples: {track.track_id}")
        if _rms(music) <= 0:
            raise UnusableMusicCropError(f"music track chunk has zero energy: {track.track_id}")
        return music

    def apply(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        recording_id: str,
        chunk_start_sample: int,
        chunk_end_sample: int,
        chunked_annotations: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform one chunk while preserving the dataset's normal two-value API."""

        transformed_x, transformed_y, _ = self.apply_with_receipt(
            x,
            y,
            recording_id=recording_id,
            chunk_start_sample=chunk_start_sample,
            chunk_end_sample=chunk_end_sample,
            chunked_annotations=chunked_annotations,
        )
        return transformed_x, transformed_y

    def _apply_music_crop(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        recording_id: str,
        chunk_start_sample: int,
        chunk_end_sample: int,
        requested_mode: MusicMode,
        reference_mask: np.ndarray,
        speech_rms: float,
        requested_snr_db: float | None,
        requested_music_rms_dbfs: float | None,
        attempt: int,
        attempt_count: int,
    ) -> tuple[np.ndarray, np.ndarray, MusicAugmentationReceipt]:
        """Apply one verified crop; only typed crop-policy failures are retryable."""

        sample_count = chunk_end_sample - chunk_start_sample
        track, offset = self._track_and_offset(
            recording_id,
            chunk_start_sample,
            chunk_end_sample,
            attempt=attempt,
        )
        music = self._read_music(track, offset, sample_count)
        music_reference = reference_mask if requested_mode == "mixed" and reference_mask.any() else None
        music_rms = _rms(music, music_reference)
        if music_rms <= 0:
            raise UnusableMusicCropError(f"music track has zero reference energy: {track.track_id}")

        measured_snr_db = None
        measured_music_rms_dbfs = None
        speech_gain = 1.0
        if requested_mode == "mixed" and reference_mask.any():
            if requested_snr_db is None:
                raise MusicAugmentationError("mixed crop is missing its requested SNR")
            music_gain = speech_rms / (10.0 ** (requested_snr_db / 20.0) * music_rms)
            output_mode: MusicMode = "mixed"
        else:
            output_mode = "mixed_no_speech" if requested_mode == "mixed" else "music_only"
            speech_gain = 0.0 if output_mode == "music_only" else 1.0
            if requested_music_rms_dbfs is None:
                raise MusicAugmentationError("non-speech crop is missing its requested music level")
            music_gain = (10.0 ** (requested_music_rms_dbfs / 20.0)) / music_rms

        music_matrix = np.broadcast_to(music, x.shape)
        if output_mode == "music_only":
            output = np.array(music_matrix * music_gain, dtype=np.float64, copy=True)
        else:
            output = np.array(np.asarray(x, dtype=np.float64) + music_matrix * music_gain, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(output)):
            raise MusicAugmentationError("augmented audio contains nonfinite samples")
        peak_before_scale = float(np.max(np.abs(output))) if output.size else 0.0
        if not math.isfinite(peak_before_scale):
            raise MusicAugmentationError("augmented audio contains nonfinite samples")
        global_scale = 0.99 / peak_before_scale if peak_before_scale > 0.99 else 1.0
        if global_scale != 1.0:
            output *= global_scale
        peak = float(np.max(np.abs(output))) if output.size else 0.0

        if output_mode == "mixed":
            measured_speech_rms = speech_rms * speech_gain * global_scale
            measured_music_rms = music_rms * music_gain * global_scale
            if measured_speech_rms <= 0 or measured_music_rms <= 0:
                raise MusicAugmentationError("mixed audio has zero reference energy")
            measured_snr_db = 20.0 * math.log10(measured_speech_rms / measured_music_rms)
        else:
            measured_music_rms_dbfs = _dbfs(music_rms * music_gain * global_scale)
            if measured_music_rms_dbfs is None or not (
                self.config.min_no_speech_rms_dbfs - 1e-6
                <= measured_music_rms_dbfs
                <= self.config.max_no_speech_rms_dbfs + 1e-6
            ):
                raise UnusableMusicCropError("music RMS level is infeasible under the 0.99 peak bound")

        receipt = MusicAugmentationReceipt(
            mode=output_mode,
            requested_mode=requested_mode,
            recording_id=recording_id,
            chunk_start_sample=chunk_start_sample,
            chunk_end_sample=chunk_end_sample,
            music_track_id=track.track_id,
            music_offset_sample=offset,
            speech_gain=speech_gain,
            music_gain=music_gain,
            global_scale=global_scale,
            requested_snr_db=requested_snr_db,
            measured_snr_db=measured_snr_db,
            requested_music_rms_dbfs=requested_music_rms_dbfs,
            measured_music_rms_dbfs=measured_music_rms_dbfs,
            speech_rms=speech_rms,
            music_rms=music_rms,
            peak=peak,
            attempt_index=attempt,
            attempt_count=attempt_count,
        )
        return output, np.zeros_like(y) if output_mode == "music_only" else y, receipt

    def apply_with_receipt(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        recording_id: str,
        chunk_start_sample: int,
        chunk_end_sample: int,
        chunked_annotations: object,
    ) -> tuple[np.ndarray, np.ndarray, MusicAugmentationReceipt]:
        """Transform one chunk and return the deterministic QA receipt."""

        if not isinstance(recording_id, str) or not recording_id:
            raise MusicAugmentationError("recording_id must be a non-empty string")
        if isinstance(chunk_start_sample, bool) or not isinstance(chunk_start_sample, int):
            raise MusicAugmentationError("chunk_start_sample must be an integer")
        if isinstance(chunk_end_sample, bool) or not isinstance(chunk_end_sample, int):
            raise MusicAugmentationError("chunk_end_sample must be an integer")
        if chunk_start_sample < 0 or chunk_end_sample <= chunk_start_sample:
            raise MusicAugmentationError("chunk sample bounds must be positive and ordered")
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise MusicAugmentationError("x must be a [channel, sample] NumPy array")
        if not isinstance(y, np.ndarray):
            raise MusicAugmentationError("y must be a NumPy array")
        sample_count = chunk_end_sample - chunk_start_sample
        if x.shape[1] != sample_count:
            raise MusicAugmentationError("chunk sample bounds do not match x")
        if not np.all(np.isfinite(x)):
            raise MusicAugmentationError("source audio contains nonfinite samples")

        requested_mode = (
            self._mode(recording_id, chunk_start_sample, chunk_end_sample) if self.config.enabled else "clean"
        )
        source_peak = float(np.max(np.abs(x))) if x.size else 0.0
        if not math.isfinite(source_peak):
            raise MusicAugmentationError("source audio contains nonfinite samples")
        if requested_mode == "clean":
            receipt = MusicAugmentationReceipt(
                mode="clean",
                requested_mode="clean",
                recording_id=recording_id,
                chunk_start_sample=chunk_start_sample,
                chunk_end_sample=chunk_end_sample,
                music_track_id=None,
                music_offset_sample=None,
                speech_gain=1.0,
                music_gain=0.0,
                global_scale=1.0,
                requested_snr_db=None,
                measured_snr_db=None,
                requested_music_rms_dbfs=None,
                measured_music_rms_dbfs=None,
                speech_rms=_rms(x),
                music_rms=0.0,
                peak=source_peak,
                attempt_index=0,
                attempt_count=1,
            )
            return x, y, receipt

        reference_mask = speech_reference_mask(
            chunked_annotations,
            chunk_start_sample=chunk_start_sample,
            chunk_end_sample=chunk_end_sample,
            sample_rate=self.sample_rate,
        )
        if requested_mode == "mixed" and not reference_mask.any() and np.any(y != 0):
            raise MusicAugmentationError("positive targets require a non-empty speech reference union")
        speech_rms = _rms(x, reference_mask) if reference_mask.any() else 0.0
        requested_snr_db = None
        requested_music_rms_dbfs = None
        if requested_mode == "mixed" and reference_mask.any():
            if speech_rms <= 0:
                raise MusicAugmentationError("labelled speech has zero source energy")
            requested_snr_db = _uniform(
                self._draw("snr", recording_id, chunk_start_sample, chunk_end_sample),
                self.config.min_snr_db,
                self.config.max_snr_db,
            )
        else:
            level_purpose = "silent-level" if requested_mode == "mixed" else "music-only-level"
            requested_music_rms_dbfs = _uniform(
                self._draw(level_purpose, recording_id, chunk_start_sample, chunk_end_sample),
                self.config.min_no_speech_rms_dbfs,
                self.config.max_no_speech_rms_dbfs,
            )
        last_error: UnusableMusicCropError | None = None
        for attempt in range(MAX_MUSIC_CROP_ATTEMPTS):
            try:
                return self._apply_music_crop(
                    x,
                    y,
                    recording_id=recording_id,
                    chunk_start_sample=chunk_start_sample,
                    chunk_end_sample=chunk_end_sample,
                    requested_mode=requested_mode,
                    reference_mask=reference_mask,
                    speech_rms=speech_rms,
                    requested_snr_db=requested_snr_db,
                    requested_music_rms_dbfs=requested_music_rms_dbfs,
                    attempt=attempt,
                    attempt_count=attempt + 1,
                )
            except UnusableMusicCropError as error:
                last_error = error
        detail = "" if last_error is None else f"; last crop was unusable: {last_error}"
        raise MusicAugmentationError(
            f"no viable music crop after {MAX_MUSIC_CROP_ATTEMPTS} attempts{detail}"
        ) from last_error

    def receipt_for(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        recording_id: str,
        chunk_start_sample: int,
        chunk_end_sample: int,
        chunked_annotations: object,
    ) -> MusicAugmentationReceipt:
        """Return the QA receipt for a chunk without changing the dataset tuple API."""

        _, _, receipt = self.apply_with_receipt(
            x,
            y,
            recording_id=recording_id,
            chunk_start_sample=chunk_start_sample,
            chunk_end_sample=chunk_end_sample,
            chunked_annotations=chunked_annotations,
        )
        return receipt


__all__ = [
    "EXPECTED_SAMPLE_RATE",
    "MAX_MUSIC_CROP_ATTEMPTS",
    "MUSIC_AUGMENTATION_SCHEMA",
    "MUSIC_MANIFEST_SCHEMA",
    "MusicAugmentationConfig",
    "MusicAugmentationError",
    "MusicAugmentationProportions",
    "MusicAugmentationReceipt",
    "MusicAugmenter",
    "MusicManifest",
    "MusicMode",
    "MusicTrack",
    "UnusableMusicCropError",
    "load_music_augmentation_config",
    "load_music_manifest",
    "speech_reference_mask",
]
