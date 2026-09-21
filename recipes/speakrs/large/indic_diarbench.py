"""Typed Indic DiarBench binding, validation, and canonical conversion."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import soundfile as sf

from .acceptance import RttmInterval
from .contracts import DiskLimits
from .errors import PreparationError
from .hashing import sha256_file, sha256_json
from .jsonio import write_json
from .prepare import prepare_parent


INDIC_SOURCE = "Indic DiarBench"
INDIC_REPOSITORY = "sarvamai/indic-diarbench"
INDIC_REVISION = "92877bad8aab6e598167d91c6ee02aa8ca6ede09"
INDIC_VERSION = f"hf-{INDIC_REVISION}"
INDIC_SPLIT = "test"
INDIC_LICENSE = "CC BY 4.0"
INDIC_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
INDIC_CARD_URL = f"https://huggingface.co/datasets/{INDIC_REPOSITORY}"
INDIC_REVISION_URL = f"{INDIC_CARD_URL}/tree/{INDIC_REVISION}"
INDIC_EXPECTED_FILE_COUNT = 24
INDIC_SAMPLE_RATE = 16_000
INDIC_CHANNELS = 1
INDIC_AUDIO_BLOCK_FRAMES = 64 * 1024
INDIC_FINGERPRINT_FRAME_SIZE = 160
INDIC_METADATA_DURATION_TOLERANCE_SECONDS = 0.005 + 1.0 / INDIC_SAMPLE_RATE
INDIC_CAPACITY_PROFILES: tuple[dict[str, int], ...] = (
    {"chunk_seconds": 8, "chunk_shift": 6, "local_slots": 4, "max_overlap": 4, "model_num_frames": 399},
    {"chunk_seconds": 8, "chunk_shift": 6, "local_slots": 6, "max_overlap": 6, "model_num_frames": 399},
    {"chunk_seconds": 8, "chunk_shift": 6, "local_slots": 8, "max_overlap": 8, "model_num_frames": 399},
)
INDIC_LANGUAGES = frozenset(
    {
        "Assamese",
        "Bengali",
        "Bodo",
        "Dogri",
        "Gujarati",
        "Hindi",
        "Kannada",
        "Kashmiri",
        "Konkani",
        "Maithili",
        "Malayalam",
        "Manipuri",
        "Marathi",
        "Nepali",
        "Odia",
        "Punjabi",
        "Sanskrit",
        "Santali",
        "Sindhi",
        "Tamil",
        "Telugu",
        "Urdu",
    }
)
_SAMPLE_ID = re.compile(r"^[a-z]+_[0-9]{3}$")
_RECORDING_ID = re.compile(r"^[a-z]+_(nf|ff|itw)_[0-9]{3}$")


class IndicPreparationState(str, Enum):
    """One immutable state in the Indic source preparation lifecycle."""

    DOWNLOADED = "downloaded"
    INVENTORIED = "inventoried"
    VALIDATED = "validated"
    CONVERTED = "converted"
    ACCEPTED = "accepted"
    SEALED = "sealed"
    RESTORED = "restored"


class IndicDatasetType(str, Enum):
    """Publisher acoustic condition."""

    NEAR_FIELD = "Near field"
    FAR_FIELD = "Far field"
    IN_THE_WILD = "In the wild"


class IndicRejectionReason(str, Enum):
    """Stable reason codes for rows excluded from the training component."""

    DUPLICATE_AUDIO = "duplicate-audio"
    PRACTICAL_DUPLICATE_AUDIO = "practical-duplicate-audio"
    EMPTY_ANNOTATION = "empty-annotation"
    INVALID_SCHEMA = "invalid-schema"
    INVALID_AUDIO = "invalid-audio"
    LABEL_OUT_OF_AUDIO = "label-out-of-audio"
    METADATA_DURATION_MISMATCH = "metadata-duration-mismatch"
    CAPACITY_WINDOW_UNAVAILABLE = "capacity-window-unavailable"
    WEB_PARENT_IDENTITY_UNAVAILABLE = "web-parent-identity-unavailable"
    DUPLICATE_SAMPLE_ID = "duplicate-sample-id"


class IndicRowError(PreparationError):
    """Validation failure with a stable Indic row reason code."""

    def __init__(self, reason: IndicRejectionReason, message: str, details: Mapping[str, object] | None = None):
        self.reason = reason
        super().__init__(message, {"reason_code": reason.value, **dict(details or {})})


@dataclass(frozen=True, slots=True)
class PublisherFile:
    """One file in the pinned Hugging Face revision."""

    path: str
    size: int
    publisher_oid: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PublisherBinding:
    """Publisher card, terms, and source-file identity bound to one revision."""

    repository: str
    revision: str
    split: str
    license: str
    license_url: str
    card_url: str
    revision_url: str
    card_sha256: str
    license_record_sha256: str
    files: tuple[PublisherFile, ...]

    @property
    def identity(self) -> str:
        """Return the stable source version used by the data contract."""

        return f"{self.repository}@{self.revision}:{self.split}"


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One human-corrected speaker-attributed transcript segment."""

    speaker_id: str
    transcript: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class AudioFacts:
    """Decoded audio and duplicate-check facts for one publisher WAV."""

    sample_rate_hz: int
    channels: int
    frames: int
    duration_seconds: float
    subtype: str
    raw_sha256: str
    decoded_pcm_sha256: str
    practical_fingerprint: str
    finite: bool
    clipped_samples: int
    zero_samples: int
    dropout_runs: int


@dataclass(frozen=True, slots=True)
class ValidatedRow:
    """A publisher row whose audio, schema, and labels share one sample clock."""

    source_file: str
    row_index: int
    sample_id: str
    recording_id: str
    language: str
    dataset_type: IndicDatasetType
    audio_path: str
    metadata_duration_seconds: float
    num_speakers: int
    num_segments: int
    segments: tuple[TranscriptSegment, ...]
    audio: AudioFacts
    speaker_seconds: float
    speech_seconds: float
    overlap_seconds: float
    max_simultaneous_speakers: int

    @property
    def parent_identity(self) -> str:
        """Return the publisher parent group, without inferring people."""

        return self.recording_id

    @property
    def canonical_id(self) -> str:
        """Return the globally unique trainer recording identity."""

        return f"indic-diarbench::{self.sample_id}"

    @property
    def parent_id(self) -> str:
        """Return the qualified parent identity used by the component evidence."""

        return f"indic-diarbench-parent::{self.recording_id}"

    @property
    def canonical_speakers(self) -> tuple[str, ...]:
        """Return source-local speaker labels encoded as RTTM-safe tokens."""

        return tuple(sorted({canonical_speaker_id(segment.speaker_id) for segment in self.segments}))

    @property
    def speaker_id_map(self) -> dict[str, str]:
        """Return the reversible local mapping retained with source metadata."""

        return {canonical_speaker_id(segment.speaker_id): segment.speaker_id for segment in self.segments}

    def metadata(self) -> dict[str, object]:
        """Return source metadata retained in the component manifest."""

        return {
            "source": INDIC_SOURCE,
            "publisher_revision": INDIC_REVISION,
            "publisher_split": INDIC_SPLIT,
            "sample_id": self.sample_id,
            "recording_id": self.recording_id,
            "parent_identity": self.parent_identity,
            "language": self.language,
            "dataset_type": self.dataset_type.value,
            "audio_path": self.audio_path,
            "metadata_duration_seconds": self.metadata_duration_seconds,
            "decoded_duration_seconds": self.audio.duration_seconds,
            "duration_difference_seconds": self.audio.duration_seconds - self.metadata_duration_seconds,
            "num_speakers": self.num_speakers,
            "num_segments": self.num_segments,
            "canonical_speakers": list(self.canonical_speakers),
            "speaker_id_map": self.speaker_id_map,
            "speaker_seconds": self.speaker_seconds,
            "speech_seconds": self.speech_seconds,
            "overlap_seconds": self.overlap_seconds,
            "overlap_ratio": self.overlap_seconds / self.speech_seconds if self.speech_seconds else 0.0,
            "max_simultaneous_speakers": self.max_simultaneous_speakers,
            "audio": {
                "raw_sha256": self.audio.raw_sha256,
                "decoded_pcm_sha256": self.audio.decoded_pcm_sha256,
                "practical_fingerprint": self.audio.practical_fingerprint,
                "sample_rate_hz": self.audio.sample_rate_hz,
                "channels": self.audio.channels,
                "frames": self.audio.frames,
                "subtype": self.audio.subtype,
                "finite": self.audio.finite,
                "clipped_samples": self.audio.clipped_samples,
                "zero_samples": self.audio.zero_samples,
                "dropout_runs": self.audio.dropout_runs,
            },
        }


@dataclass(frozen=True, slots=True)
class RejectedRow:
    """One row that could not enter the validated inventory."""

    source_file: str
    row_index: int
    sample_id: str | None
    recording_id: str | None
    reason_code: IndicRejectionReason
    details: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RawPublisherRow:
    """One Parquet row with embedded audio bytes and source location."""

    source_file: str
    row_index: int
    payload: object
    audio_bytes: bytes


@dataclass(frozen=True, slots=True)
class DuplicateReference:
    """One existing training or held-out audio identity."""

    source: str
    recording_id: str
    parent_id: str
    raw_sha256: str | None
    decoded_pcm_sha256: str
    practical_fingerprint: str
    frames: int


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    """One exact or practical duplicate match."""

    sample_id: str
    against_source: str
    against_recording_id: str
    reason_code: IndicRejectionReason
    match_kind: str


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    """All validated and rejected rows from the pinned publisher release."""

    binding: PublisherBinding
    rows: tuple[ValidatedRow, ...]
    rejected_rows: tuple[RejectedRow, ...]

    def summary(self) -> dict[str, object]:
        """Return exact row, parent, language, and acoustic statistics."""

        by_language: dict[str, dict[str, float | int]] = {}
        by_condition: dict[str, dict[str, float | int]] = {}
        for row in self.rows:
            language = by_language.setdefault(row.language, {"rows": 0, "parents": set(), "seconds": 0.0})
            language["rows"] = int(language["rows"]) + 1
            parents = language["parents"]
            assert isinstance(parents, set)
            parents.add(row.parent_identity)
            language["seconds"] = float(language["seconds"]) + row.audio.duration_seconds
            condition = by_condition.setdefault(row.dataset_type.value, {"rows": 0, "parents": set(), "seconds": 0.0})
            condition["rows"] = int(condition["rows"]) + 1
            condition_parents = condition["parents"]
            assert isinstance(condition_parents, set)
            condition_parents.add(row.parent_identity)
            condition["seconds"] = float(condition["seconds"]) + row.audio.duration_seconds

        def clean(value: Mapping[str, object]) -> dict[str, object]:
            return {
                "rows": int(value["rows"]),
                "parents": len(value["parents"]),
                "seconds": float(value["seconds"]),
                "hours": float(value["seconds"]) / 3600.0,
            }

        return {
            "source": INDIC_SOURCE,
            "repository": self.binding.repository,
            "revision": self.binding.revision,
            "split": self.binding.split,
            "published_file_count": len(self.binding.files),
            "validated_rows": len(self.rows),
            "rejected_rows": len(self.rejected_rows),
            "validated_parent_count": len({row.parent_identity for row in self.rows}),
            "languages": sorted({row.language for row in self.rows}),
            "language_count": len({row.language for row in self.rows}),
            "duration_seconds": sum(row.audio.duration_seconds for row in self.rows),
            "duration_hours": sum(row.audio.duration_seconds for row in self.rows) / 3600.0,
            "speaker_seconds": sum(row.speaker_seconds for row in self.rows),
            "speech_seconds": sum(row.speech_seconds for row in self.rows),
            "overlap_seconds": sum(row.overlap_seconds for row in self.rows),
            "by_language": {key: clean(by_language[key]) for key in sorted(by_language)},
            "by_dataset_type": {key: clean(by_condition[key]) for key in sorted(by_condition)},
        }


@dataclass(frozen=True, slots=True)
class TrainingAdmission:
    """Immutable row membership and duplicate exclusion decisions."""

    accepted: tuple[ValidatedRow, ...]
    excluded: tuple[RejectedRow, ...]
    duplicate_matches: tuple[DuplicateMatch, ...]
    state: IndicPreparationState = IndicPreparationState.ACCEPTED

    def manifest(self) -> dict[str, object]:
        """Return exact membership with stable reason codes."""

        return {
            "schema": "indic-diarbench-training-admission-v1",
            "source": INDIC_SOURCE,
            "revision": INDIC_REVISION,
            "publisher_split": INDIC_SPLIT,
            "training_only": True,
            "accepted_sample_ids": [row.sample_id for row in self.accepted],
            "accepted_parent_ids": sorted({row.parent_identity for row in self.accepted}),
            "excluded": [
                {
                    "sample_id": item.sample_id,
                    "recording_id": item.recording_id,
                    "source_file": item.source_file,
                    "row_index": item.row_index,
                    "reason_code": item.reason_code.value,
                    "details": dict(item.details),
                }
                for item in self.excluded
            ],
            "duplicate_matches": [
                {
                    "sample_id": item.sample_id,
                    "against_source": item.against_source,
                    "against_recording_id": item.against_recording_id,
                    "reason_code": item.reason_code.value,
                    "match_kind": item.match_kind,
                }
                for item in self.duplicate_matches
            ],
        }


def advance_state(current: IndicPreparationState, target: IndicPreparationState) -> IndicPreparationState:
    """Allow only the source lifecycle transitions used by release evidence."""

    allowed = {
        IndicPreparationState.DOWNLOADED: {IndicPreparationState.INVENTORIED},
        IndicPreparationState.INVENTORIED: {IndicPreparationState.VALIDATED},
        IndicPreparationState.VALIDATED: {IndicPreparationState.CONVERTED},
        IndicPreparationState.CONVERTED: {IndicPreparationState.ACCEPTED},
        IndicPreparationState.ACCEPTED: {IndicPreparationState.SEALED},
        IndicPreparationState.SEALED: {IndicPreparationState.RESTORED},
        IndicPreparationState.RESTORED: set(),
    }
    if target not in allowed[current]:
        raise PreparationError(
            "invalid Indic DiarBench preparation state transition",
            {"current": current.value, "target": target.value},
        )
    return target


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_speaker_id(source_id: str) -> str:
    """Return a collision-resistant RTTM token for one source-local speaker label."""

    return f"spk-{_sha256_bytes(source_id.encode('utf-8'))[:16]}"


def bind_publisher_release(source_root: Path, receipt_path: Path, license_record_path: Path) -> PublisherBinding:
    """Bind downloaded bytes, the dataset card, and the CC BY 4.0 record."""

    source_root = Path(source_root)
    payload = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "indic-diarbench-publisher-receipt-v1":
        raise PreparationError("Indic publisher receipt schema is unknown")
    if payload.get("repository") != INDIC_REPOSITORY or payload.get("revision") != INDIC_REVISION:
        raise PreparationError("Indic publisher receipt is not pinned to the accepted revision")
    files_payload = payload.get("files")
    if not isinstance(files_payload, list) or len(files_payload) != INDIC_EXPECTED_FILE_COUNT:
        raise PreparationError("Indic publisher file inventory is incomplete")
    files: list[PublisherFile] = []
    seen: set[str] = set()
    for item in files_payload:
        if not isinstance(item, Mapping):
            raise PreparationError("Indic publisher file receipt is malformed")
        path = item.get("path")
        size = item.get("size")
        publisher_oid = item.get("publisher_oid")
        sha256 = item.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not isinstance(publisher_oid, str)
            or not isinstance(sha256, str)
            or path in seen
        ):
            raise PreparationError("Indic publisher file receipt has invalid identity")
        seen.add(path)
        path_on_disk = source_root / path
        if not path_on_disk.is_file() or path_on_disk.stat().st_size != size or sha256_file(path_on_disk) != sha256:
            raise PreparationError("Indic publisher file bytes differ from their receipt", {"path": path})
        files.append(PublisherFile(path, size, publisher_oid, sha256))

    card_path = source_root / "README.md"
    card_text = card_path.read_text(encoding="utf-8")
    if "license: cc-by-4.0" not in card_text.lower():
        raise PreparationError("Indic dataset card does not record CC BY 4.0")
    license_path = Path(license_record_path)
    if not license_path.is_file() or license_path.stat().st_size == 0:
        raise PreparationError("CC BY 4.0 license record is missing")
    return PublisherBinding(
        repository=INDIC_REPOSITORY,
        revision=INDIC_REVISION,
        split=INDIC_SPLIT,
        license=INDIC_LICENSE,
        license_url=INDIC_LICENSE_URL,
        card_url=INDIC_CARD_URL,
        revision_url=INDIC_REVISION_URL,
        card_sha256=_sha256_bytes(card_text.encode("utf-8")),
        license_record_sha256=sha256_file(license_path),
        files=tuple(sorted(files, key=lambda item: item.path)),
    )


def _require_pyarrow() -> Any:
    """Load the Parquet reader only when a publisher inventory is requested."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise PreparationError("Indic Parquet validation requires pyarrow") from error
    return parquet


def iter_publisher_rows(
    source_root: Path, binding: PublisherBinding, *, batch_size: int = 8
) -> Iterator[RawPublisherRow]:
    """Stream every row from every pinned language Parquet file."""

    yield from _iter_publisher_rows_with_index(source_root, binding, batch_size=batch_size)


def _iter_publisher_rows_with_index(
    source_root: Path, binding: PublisherBinding, *, batch_size: int = 8
) -> Iterator[RawPublisherRow]:
    """Stream Parquet rows while preserving their stable row index."""

    parquet = _require_pyarrow()
    for publisher_file in binding.files:
        if not publisher_file.path.endswith(".parquet"):
            continue
        path = Path(source_root) / publisher_file.path
        parquet_file = parquet.ParquetFile(path)
        row_index = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for payload in batch.to_pylist():
                audio = payload.get("audio") if isinstance(payload, Mapping) else None
                audio_bytes = audio.get("bytes") if isinstance(audio, Mapping) else None
                if isinstance(audio_bytes, memoryview):
                    audio_bytes = audio_bytes.tobytes()
                if not isinstance(audio_bytes, bytes):
                    audio_bytes = b""
                yield RawPublisherRow(publisher_file.path, row_index, payload, audio_bytes)
                row_index += 1


def _invalid(reason: IndicRejectionReason, message: str, **details: object) -> IndicRowError:
    return IndicRowError(reason, message, details)


def _audio_facts_from_stream(source: Any, raw_sha256: str) -> AudioFacts:
    """Decode one seekable audio stream in bounded blocks."""

    pcm_hash = hashlib.sha256()
    envelopes: list[float] = []
    frames = 0
    zero_samples = 0
    clipped_samples = 0
    dropout_runs = 0
    zero_run = 0
    try:
        sample_rate = int(source.samplerate)
        channels = int(source.channels)
        subtype = str(source.subtype or "")
        while True:
            block = source.read(INDIC_AUDIO_BLOCK_FRAMES, dtype="int16", always_2d=True)
            if block.size == 0:
                break
            frames += int(block.shape[0])
            pcm_hash.update(np.asarray(block, dtype="<i2").tobytes(order="C"))
            zero = np.all(block == 0, axis=1)
            zero_samples += int(np.count_nonzero(zero))
            clipped_samples += int(np.count_nonzero(np.any(np.abs(block) >= 32767, axis=1)))
            padded = np.concatenate((np.array([False]), zero, np.array([False])))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            for start, end in zip(edges[::2], edges[1::2]):
                run_length = int(end - start)
                if start == 0 and zero_run:
                    run_length += zero_run
                zero_run = run_length if end == len(zero) and bool(zero[-1]) else 0
                if run_length >= INDIC_SAMPLE_RATE // 10 and not (end == len(zero) and bool(zero[-1])):
                    dropout_runs += 1
            mono = block[:, 0].astype(np.float64) / 32768.0
            frame_count = len(mono) // INDIC_FINGERPRINT_FRAME_SIZE
            if frame_count:
                frames_2d = mono[: frame_count * INDIC_FINGERPRINT_FRAME_SIZE].reshape(
                    frame_count, INDIC_FINGERPRINT_FRAME_SIZE
                )
                envelopes.extend(np.sqrt(np.mean(np.square(frames_2d), axis=1)).tolist())
            remainder = mono[frame_count * INDIC_FINGERPRINT_FRAME_SIZE :]
            if remainder.size:
                envelopes.append(float(np.sqrt(np.mean(np.square(remainder)))))
        if zero_run >= INDIC_SAMPLE_RATE // 10:
            dropout_runs += 1
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise _invalid(
            IndicRejectionReason.INVALID_AUDIO, "publisher audio cannot be decoded", error=str(error)
        ) from error
    if frames <= 0 or sample_rate <= 0 or channels <= 0:
        raise _invalid(IndicRejectionReason.INVALID_AUDIO, "publisher audio has no decoded samples")
    envelope = np.asarray(envelopes, dtype=np.float64)
    scale = float(np.percentile(envelope, 95)) if envelope.size else 0.0
    if scale <= 0.0:
        quantized = np.zeros(envelope.shape, dtype=np.uint8)
    else:
        quantized = np.clip(np.rint(envelope / scale * 255.0), 0, 255).astype(np.uint8)
    fingerprint = hashlib.sha256(
        f"{sample_rate}:{channels}:{frames}:{INDIC_FINGERPRINT_FRAME_SIZE}:".encode("ascii") + quantized.tobytes()
    ).hexdigest()
    return AudioFacts(
        sample_rate_hz=sample_rate,
        channels=channels,
        frames=frames,
        duration_seconds=frames / sample_rate,
        subtype=subtype,
        raw_sha256=raw_sha256,
        decoded_pcm_sha256=pcm_hash.hexdigest(),
        practical_fingerprint=fingerprint,
        finite=True,
        clipped_samples=clipped_samples,
        zero_samples=zero_samples,
        dropout_runs=dropout_runs,
    )


def _audio_facts(audio_bytes: bytes) -> AudioFacts:
    """Decode embedded publisher audio bytes and calculate duplicate identities."""

    if not audio_bytes:
        raise _invalid(IndicRejectionReason.INVALID_AUDIO, "publisher audio bytes are empty")
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes), mode="r") as source:
            return _audio_facts_from_stream(source, _sha256_bytes(audio_bytes))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, IndicRowError):
            raise
        raise _invalid(
            IndicRejectionReason.INVALID_AUDIO, "publisher audio cannot be decoded", error=str(error)
        ) from error


def audio_file_facts(path: Path) -> AudioFacts:
    """Decode an existing training or held-out audio file for duplicate checks."""

    path = Path(path)
    if not path.is_file():
        raise PreparationError("duplicate reference audio is missing", {"path": str(path)})
    try:
        with sf.SoundFile(path, mode="r") as source:
            return _audio_facts_from_stream(source, sha256_file(path))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PreparationError(
            "duplicate reference audio cannot be decoded", {"path": str(path), "error": str(error)}
        ) from error


def _merge_segments(segments: Sequence[TranscriptSegment]) -> dict[str, list[tuple[float, float]]]:
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        by_speaker.setdefault(segment.speaker_id, []).append((segment.start, segment.end))
    merged: dict[str, list[tuple[float, float]]] = {}
    for speaker, values in by_speaker.items():
        combined: list[tuple[float, float]] = []
        for start, end in sorted(values):
            if not combined or start > combined[-1][1]:
                combined.append((start, end))
            else:
                combined[-1] = (combined[-1][0], max(combined[-1][1], end))
        merged[speaker] = combined
    return merged


def _speech_statistics(segments: Sequence[TranscriptSegment]) -> tuple[float, float, int]:
    merged = _merge_segments(segments)
    speaker_seconds = sum(end - start for spans in merged.values() for start, end in spans)
    events: list[tuple[float, int]] = []
    for spans in merged.values():
        for start, end in spans:
            events.extend(((start, 1), (end, -1)))
    active = 0
    speech_seconds = 0.0
    max_active = 0
    previous: float | None = None
    for position, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        if previous is not None and active:
            speech_seconds += position - previous
        active += delta
        max_active = max(max_active, active)
        previous = position
    return speaker_seconds, speech_seconds, max_active


def validate_publisher_row(raw: RawPublisherRow) -> ValidatedRow:
    """Validate schema, audio clock, labels, and publisher counters for one row."""

    payload = raw.payload
    if not isinstance(payload, Mapping):
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic publisher row is not a mapping")
    required = {
        "audio",
        "recording_id",
        "language",
        "annotated_transcript",
        "dataset_type",
        "sample_id",
        "num_speakers",
        "num_segments",
        "duration_seconds",
    }
    if set(payload) != required:
        raise _invalid(
            IndicRejectionReason.INVALID_SCHEMA,
            "Indic row fields differ from the pinned dataset schema",
            fields=sorted(payload),
        )
    sample_id = payload["sample_id"]
    recording_id = payload["recording_id"]
    language = payload["language"]
    dataset_type = payload["dataset_type"]
    if not isinstance(sample_id, str) or not _SAMPLE_ID.fullmatch(sample_id):
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic sample_id has the wrong form")
    if not isinstance(recording_id, str) or not _RECORDING_ID.fullmatch(recording_id):
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic recording_id has the wrong form")
    if not isinstance(language, str) or language not in INDIC_LANGUAGES:
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic language is not one of the 22 pinned languages")
    if Path(raw.source_file).parts[0] != language:
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic Parquet path language differs from row language")
    try:
        condition = IndicDatasetType(str(dataset_type))
    except ValueError as error:
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic dataset_type is unknown") from error
    expected_condition = {
        IndicDatasetType.NEAR_FIELD: "nf",
        IndicDatasetType.FAR_FIELD: "ff",
        IndicDatasetType.IN_THE_WILD: "itw",
    }[condition]
    if f"_{expected_condition}_" not in recording_id:
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "recording_id condition disagrees with dataset_type")
    audio = payload["audio"]
    if not isinstance(audio, Mapping) or set(audio) != {"bytes", "path"}:
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic audio field is malformed")
    audio_path = audio["path"]
    if not isinstance(audio_path, str) or Path(audio_path).name != f"{sample_id}.wav":
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic audio path is not bound to sample_id")
    metadata_duration = payload["duration_seconds"]
    if isinstance(metadata_duration, bool) or not isinstance(metadata_duration, (int, float)):
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic duration_seconds is not numeric")
    metadata_duration = float(metadata_duration)
    if not math.isfinite(metadata_duration) or metadata_duration <= 0:
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic duration_seconds is not positive and finite")
    num_speakers = payload["num_speakers"]
    num_segments = payload["num_segments"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (num_speakers, num_segments)):
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic counters are not integers")
    transcript = payload["annotated_transcript"]
    if not isinstance(transcript, list):
        raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic annotated_transcript is not a list")
    if not transcript:
        raise _invalid(IndicRejectionReason.EMPTY_ANNOTATION, "Indic row has no speaker annotation")
    segments: list[TranscriptSegment] = []
    for item in transcript:
        if not isinstance(item, Mapping) or set(item) != {"speaker_id", "transcript", "start_time", "end_time"}:
            raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic transcript segment fields are malformed")
        speaker = item["speaker_id"]
        text = item["transcript"]
        start = item["start_time"]
        end = item["end_time"]
        if (
            not isinstance(speaker, str)
            or not speaker
            or not isinstance(text, str)
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
        ):
            raise _invalid(IndicRejectionReason.INVALID_SCHEMA, "Indic transcript segment types are malformed")
        start = float(start)
        end = float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise _invalid(
                IndicRejectionReason.INVALID_SCHEMA,
                "Indic transcript segment bounds are invalid",
                speaker_id=speaker,
                start=start,
                end=end,
            )
        segments.append(TranscriptSegment(speaker, text, start, end))
    facts = _audio_facts(raw.audio_bytes)
    if facts.sample_rate_hz != INDIC_SAMPLE_RATE or facts.channels != INDIC_CHANNELS:
        raise _invalid(
            IndicRejectionReason.INVALID_AUDIO,
            "Indic audio is not the declared 16 kHz mono signal",
            sample_rate_hz=facts.sample_rate_hz,
            channels=facts.channels,
        )
    duration_tolerance = INDIC_METADATA_DURATION_TOLERANCE_SECONDS
    if abs(facts.duration_seconds - metadata_duration) > duration_tolerance:
        raise _invalid(
            IndicRejectionReason.METADATA_DURATION_MISMATCH,
            "Indic metadata duration differs from decoded duration",
            metadata_duration_seconds=metadata_duration,
            decoded_duration_seconds=facts.duration_seconds,
        )
    for segment in segments:
        if segment.end > facts.duration_seconds + 1e-9:
            raise _invalid(
                IndicRejectionReason.LABEL_OUT_OF_AUDIO,
                "Indic annotation extends beyond decoded audio",
                speaker_id=segment.speaker_id,
                start=segment.start,
                end=segment.end,
                decoded_duration_seconds=facts.duration_seconds,
            )
    if int(num_segments) != len(segments) or int(num_speakers) != len({item.speaker_id for item in segments}):
        raise _invalid(
            IndicRejectionReason.INVALID_SCHEMA,
            "Indic publisher counters disagree with annotations",
            declared_segments=num_segments,
            actual_segments=len(segments),
            declared_speakers=num_speakers,
            actual_speakers=len({item.speaker_id for item in segments}),
        )
    speaker_seconds, speech_seconds, max_active = _speech_statistics(segments)
    return ValidatedRow(
        source_file=raw.source_file,
        row_index=raw.row_index,
        sample_id=sample_id,
        recording_id=recording_id,
        language=language,
        dataset_type=condition,
        audio_path=audio_path,
        metadata_duration_seconds=metadata_duration,
        num_speakers=int(num_speakers),
        num_segments=int(num_segments),
        segments=tuple(segments),
        audio=facts,
        speaker_seconds=speaker_seconds,
        speech_seconds=speech_seconds,
        overlap_seconds=max(0.0, speaker_seconds - speech_seconds),
        max_simultaneous_speakers=max_active,
    )


def scan_publisher(source_root: Path, binding: PublisherBinding) -> DatasetInventory:
    """Validate all published rows without changing source labels or clocks."""

    rows: list[ValidatedRow] = []
    rejected: list[RejectedRow] = []
    seen_samples: set[str] = set()
    for raw in _iter_publisher_rows_with_index(source_root, binding):
        sample_id_value = raw.payload.get("sample_id") if isinstance(raw.payload, Mapping) else None
        recording_id_value = raw.payload.get("recording_id") if isinstance(raw.payload, Mapping) else None
        sample_id = sample_id_value if isinstance(sample_id_value, str) else None
        recording_id = recording_id_value if isinstance(recording_id_value, str) else None
        if sample_id in seen_samples:
            rejected.append(
                RejectedRow(
                    raw.source_file,
                    raw.row_index,
                    sample_id,
                    recording_id,
                    IndicRejectionReason.DUPLICATE_SAMPLE_ID,
                    {"sample_id": sample_id},
                )
            )
            continue
        try:
            row = validate_publisher_row(raw)
        except IndicRowError as error:
            rejected.append(
                RejectedRow(raw.source_file, raw.row_index, sample_id, recording_id, error.reason, error.details)
            )
            continue
        seen_samples.add(row.sample_id)
        rows.append(row)
    if len(rows) + len(rejected) != 1164:
        raise PreparationError(
            "Indic published row count differs from the dataset card",
            {"validated": len(rows), "rejected": len(rejected), "expected": 1164},
        )
    return DatasetInventory(binding, tuple(sorted(rows, key=lambda item: item.sample_id)), tuple(rejected))


def audit_duplicates(
    rows: Sequence[ValidatedRow], references: Sequence[DuplicateReference]
) -> tuple[DuplicateMatch, ...]:
    """Find exact decoded and practical audio duplicates against existing inputs."""

    raw_index: dict[str, list[DuplicateReference]] = {}
    pcm_index: dict[str, list[DuplicateReference]] = {}
    practical_index: dict[tuple[str, int], list[DuplicateReference]] = {}
    for reference in references:
        if reference.raw_sha256:
            raw_index.setdefault(reference.raw_sha256, []).append(reference)
        pcm_index.setdefault(reference.decoded_pcm_sha256, []).append(reference)
        practical_index.setdefault((reference.practical_fingerprint, reference.frames), []).append(reference)
    matches: list[DuplicateMatch] = []
    for row in rows:
        candidates = raw_index.get(row.audio.raw_sha256, [])
        if candidates:
            matches.extend(
                DuplicateMatch(
                    row.sample_id, item.source, item.recording_id, IndicRejectionReason.DUPLICATE_AUDIO, "raw-bytes"
                )
                for item in candidates
            )
            continue
        candidates = pcm_index.get(row.audio.decoded_pcm_sha256, [])
        if candidates:
            matches.extend(
                DuplicateMatch(
                    row.sample_id, item.source, item.recording_id, IndicRejectionReason.DUPLICATE_AUDIO, "decoded-pcm"
                )
                for item in candidates
            )
            continue
        candidates = practical_index.get((row.audio.practical_fingerprint, row.audio.frames), [])
        matches.extend(
            DuplicateMatch(
                row.sample_id,
                item.source,
                item.recording_id,
                IndicRejectionReason.PRACTICAL_DUPLICATE_AUDIO,
                "duration-normalized-envelope",
            )
            for item in candidates
        )
    return tuple(sorted(matches, key=lambda item: (item.sample_id, item.against_source, item.against_recording_id)))


def admit_training_rows(
    inventory: DatasetInventory,
    duplicate_matches: Sequence[DuplicateMatch],
) -> TrainingAdmission:
    """Admit only validated meeting rows and retain every exclusion reason."""

    matches_by_sample: dict[str, list[DuplicateMatch]] = {}
    for match in duplicate_matches:
        matches_by_sample.setdefault(match.sample_id, []).append(match)
    excluded = [
        RejectedRow(
            item.source_file, item.row_index, item.sample_id, item.recording_id, item.reason_code, item.details
        )
        for item in inventory.rejected_rows
    ]
    accepted: list[ValidatedRow] = []
    for row in inventory.rows:
        if row.sample_id in matches_by_sample:
            for match in matches_by_sample[row.sample_id]:
                excluded.append(
                    RejectedRow(
                        row.source_file,
                        row.row_index,
                        row.sample_id,
                        row.recording_id,
                        match.reason_code,
                        {
                            "against_source": match.against_source,
                            "against_recording_id": match.against_recording_id,
                            "match_kind": match.match_kind,
                        },
                    )
                )
            continue
        if row.dataset_type is IndicDatasetType.IN_THE_WILD:
            excluded.append(
                RejectedRow(
                    row.source_file,
                    row.row_index,
                    row.sample_id,
                    row.recording_id,
                    IndicRejectionReason.WEB_PARENT_IDENTITY_UNAVAILABLE,
                    {
                        "publisher_parent_identity": row.parent_identity,
                        "publisher_provenance": "recording_id is deliberately anonymized and source medium is unavailable",
                    },
                )
            )
            continue
        if row.audio.duration_seconds < 8.0:
            excluded.append(
                RejectedRow(
                    row.source_file,
                    row.row_index,
                    row.sample_id,
                    row.recording_id,
                    IndicRejectionReason.CAPACITY_WINDOW_UNAVAILABLE,
                    {
                        "decoded_duration_seconds": row.audio.duration_seconds,
                        "required_chunk_seconds": 8,
                        "chunk_shift_seconds": 6,
                    },
                )
            )
            continue
        accepted.append(row)
    return TrainingAdmission(
        accepted=tuple(sorted(accepted, key=lambda item: item.sample_id)),
        excluded=tuple(
            sorted(excluded, key=lambda item: (item.sample_id or "", item.row_index, item.reason_code.value))
        ),
        duplicate_matches=tuple(duplicate_matches),
    )


def materialize_source_audio(audio_bytes: bytes, expected_sha256: str, destination: Path) -> Path:
    """Write one embedded publisher WAV through a hash-bound atomic path."""

    destination = Path(destination)
    actual = _sha256_bytes(audio_bytes)
    if actual != expected_sha256:
        raise PreparationError("Indic embedded audio hash differs before materialization")
    if destination.is_file():
        if sha256_file(destination) != expected_sha256:
            raise PreparationError("Indic materialized audio differs from its source identity")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise PreparationError("Indic materialization has a stale partial file", {"path": str(partial)})
    partial.write_bytes(audio_bytes)
    if sha256_file(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise PreparationError("Indic materialized audio failed its hash check")
    partial.replace(destination)
    return destination


def canonicalize_row(
    row: ValidatedRow,
    audio_path: Path,
    output_root: Path,
    disk_limits: DiskLimits,
) -> dict[str, Any]:
    """Convert one validated publisher row using the shared parent owner."""

    destination = Path(output_root) / "audio" / row.language / f"{row.audio.decoded_pcm_sha256}.flac"
    rttm = Path(output_root) / "labels" / row.language / f"{row.audio.decoded_pcm_sha256}.rttm"
    uem = Path(output_root) / "labels" / row.language / f"{row.audio.decoded_pcm_sha256}.uem"
    intervals = tuple(
        RttmInterval(row.canonical_id, item.start, item.end, canonical_speaker_id(item.speaker_id))
        for item in row.segments
    )
    result = prepare_parent(
        audio_path,
        None,
        row.audio.raw_sha256,
        intervals,
        ((0.0, row.audio.duration_seconds),),
        destination,
        disk_limits,
        parent_id=row.canonical_id,
        label_destination=rttm,
        uem_destination=uem,
    )
    result["indic_metadata"] = row.metadata()
    result["parent_identity"] = row.parent_identity
    result["source_file"] = row.source_file
    result["source_row_index"] = row.row_index
    return result


def capacity_profiles() -> tuple[dict[str, int], ...]:
    """Return copies of the required 4/4, 6/6, and 8/8 profiles."""

    return tuple(dict(profile) for profile in INDIC_CAPACITY_PROFILES)


def write_inventory(path: Path, inventory: DatasetInventory) -> None:
    """Write the immutable publisher inventory evidence."""

    payload = {
        "schema": "indic-diarbench-publisher-inventory-v1",
        "binding": {
            "repository": inventory.binding.repository,
            "revision": inventory.binding.revision,
            "split": inventory.binding.split,
            "license": inventory.binding.license,
            "license_url": inventory.binding.license_url,
            "card_url": inventory.binding.card_url,
            "revision_url": inventory.binding.revision_url,
            "card_sha256": inventory.binding.card_sha256,
            "license_record_sha256": inventory.binding.license_record_sha256,
            "files": [
                {
                    "path": item.path,
                    "size": item.size,
                    "publisher_oid": item.publisher_oid,
                    "sha256": item.sha256,
                }
                for item in inventory.binding.files
            ],
        },
        "summary": inventory.summary(),
        "rows": [
            row.metadata() | {"source_file": row.source_file, "source_row_index": row.row_index}
            for row in inventory.rows
        ],
        "rejected_rows": [
            {
                "source_file": item.source_file,
                "source_row_index": item.row_index,
                "sample_id": item.sample_id,
                "recording_id": item.recording_id,
                "reason_code": item.reason_code.value,
                "details": dict(item.details),
            }
            for item in inventory.rejected_rows
        ],
    }
    write_json(Path(path), payload)


def component_identity(manifest: Mapping[str, object]) -> str:
    """Return a content identity for source metadata and exact membership."""

    return sha256_json(dict(manifest))


__all__ = [
    "AudioFacts",
    "DuplicateMatch",
    "DuplicateReference",
    "DatasetInventory",
    "INDIC_CARD_URL",
    "INDIC_CAPACITY_PROFILES",
    "INDIC_LANGUAGES",
    "INDIC_LICENSE",
    "INDIC_LICENSE_URL",
    "INDIC_REPOSITORY",
    "INDIC_REVISION",
    "INDIC_REVISION_URL",
    "INDIC_SOURCE",
    "INDIC_SPLIT",
    "INDIC_VERSION",
    "IndicDatasetType",
    "IndicPreparationState",
    "IndicRejectionReason",
    "IndicRowError",
    "PublisherBinding",
    "PublisherFile",
    "RawPublisherRow",
    "RejectedRow",
    "TrainingAdmission",
    "TranscriptSegment",
    "ValidatedRow",
    "admit_training_rows",
    "advance_state",
    "audit_duplicates",
    "audio_file_facts",
    "bind_publisher_release",
    "canonical_speaker_id",
    "canonicalize_row",
    "capacity_profiles",
    "component_identity",
    "iter_publisher_rows",
    "materialize_source_audio",
    "scan_publisher",
    "validate_publisher_row",
    "write_inventory",
]
