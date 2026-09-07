"""Materialize and seal the large_cc_v1 public CC release."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

from .acceptance import RttmInterval, audit_selection_hours_capacity, load_uem_durations, parse_rttm
from .contracts import (
    CYCLE_EXAMPLES,
    FINITE_STREAMS,
    GOLD_STREAMS,
    SEED,
    STREAM_QUOTAS,
    DiskLimits,
    is_placeholder_hash,
    parse_recording_row,
    parse_spec,
    spec_to_json,
)
from .errors import PreparationError
from .hashing import sha256_file, sha256_json
from .inventory import (
    ICSI_MIX_URL,
    LOTUSDIS_FULL_MEETING_BYTES,
    LOTUSDIS_FULL_MEETING_ID,
    NOTSOFAR_HF_DATASET,
    NOTSOFAR_SIM_HF_PREFIX,
    NOTSOFAR_SIM_PREFIX,
    PEAK_GIB_ESTIMATE,
    SOURCES,
)
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
LOTUSDIS_VIEW_PREFERENCE = ("jbl", "bt3m", "bt10m", "con123")
NOTSOFAR_SIM_MIXTURE_CHANNELS = 7
NOTSOFAR_SIM_SAMPLE_RATE = 16000
GDRIVE_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

PARENT_PREPARATION_SCHEMA = "speakrs-parent-preparation"
PARENT_PREPARATION_SCHEMA_VERSION = 2
PARENT_PREPARATION_SAMPLE_RATE = 16000
PARENT_PREPARATION_BLOCK_FRAMES = 64 * 1024
PARENT_PREPARATION_DROPOUT_FRAMES = 160
PARENT_PREPARATION_RECEIPT_SUFFIX = ".transform.json"
PARENT_PREPARATION_CLIP_THRESHOLD = 1.0 - 1.0 / (1 << 15)
PARENT_PREPARATION_SWR_FILTER_SIZE = 64
PARENT_PREPARATION_SWR_PHASE_SHIFT = 10
PARENT_PREPARATION_SWR_CUTOFF = 0.97
PARENT_PREPARATION_SWR_KAISER_BETA = 9.0
SOURCE_DOWNLOAD_SCHEMA = "speakrs-source-download"
SOURCE_DOWNLOAD_SCHEMA_VERSION = 1
SOURCE_DOWNLOAD_BLOCK_BYTES = 1024 * 1024


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PreparationError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses in the loaded file look up the module in sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scrubbed_environment() -> dict[str, str]:
    """Return a child environment without host control tokens."""

    environment = os.environ.copy()
    for name in SENSITIVE_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    return environment


@dataclass
class _AudioStatistics:
    """Accumulate bounded-block audio quality facts."""

    sample_count: int = 0
    finite_samples: int = 0
    non_finite_samples: int = 0
    clipped_samples: int = 0
    zero_samples: int = 0
    dropout_samples: int = 0
    dropout_runs: int = 0
    longest_dropout_samples: int = 0
    minimum: float | None = None
    maximum: float | None = None
    peak_abs: float = 0.0
    sum_squares: float = 0.0
    _zero_run: int = 0

    def update(self, block: np.ndarray, *, clip_threshold: float = PARENT_PREPARATION_CLIP_THRESHOLD) -> None:
        """Accumulate one decoded mono block without retaining it."""

        samples = np.asarray(block, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            return
        self.sample_count += int(samples.size)
        finite = np.isfinite(samples)
        self.finite_samples += int(finite.sum())
        self.non_finite_samples += int((~finite).sum())
        if finite.any():
            values = samples[finite]
            block_min = float(values.min())
            block_max = float(values.max())
            self.minimum = block_min if self.minimum is None else min(self.minimum, block_min)
            self.maximum = block_max if self.maximum is None else max(self.maximum, block_max)
            self.peak_abs = max(self.peak_abs, float(np.max(np.abs(values))))
            self.clipped_samples += int(np.count_nonzero(np.abs(values) >= clip_threshold))
            self.sum_squares += float(np.dot(values, values))

        zero = finite & (samples == 0.0)
        self.zero_samples += int(zero.sum())
        self._update_zero_runs(zero)

    def _update_zero_runs(self, zero: np.ndarray) -> None:
        """Track exact-zero runs, including runs crossing block boundaries."""

        if zero.size == 0:
            return
        padded = np.concatenate((np.array([False]), zero, np.array([False])))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        cursor = 0
        for start, end in zip(edges[::2], edges[1::2]):
            if start > cursor and self._zero_run:
                self._finish_zero_run()
            if start == 0 and self._zero_run:
                self._zero_run += int(end - start)
            else:
                self._zero_run = int(end - start)
            cursor = int(end)
            if end < zero.size:
                self._finish_zero_run()
        if not bool(zero[-1]) and self._zero_run:
            self._finish_zero_run()

    def _finish_zero_run(self) -> None:
        """Record one completed exact-zero run."""

        run = self._zero_run
        self._zero_run = 0
        if run < PARENT_PREPARATION_DROPOUT_FRAMES:
            return
        self.dropout_runs += 1
        self.dropout_samples += run
        self.longest_dropout_samples = max(self.longest_dropout_samples, run)

    def finish(self) -> None:
        """Close a run that ended at end of stream."""

        if self._zero_run:
            self._finish_zero_run()

    def as_dict(self, *, sample_rate: int, channels: int = 1) -> dict[str, object]:
        """Return JSON-safe facts, including non-semantic dropout indicators."""

        self.finish()
        finite = self.non_finite_samples == 0
        return {
            "sample_count": self.sample_count,
            "sample_rate": sample_rate,
            "channels": channels,
            "finite": finite,
            "finite_samples": self.finite_samples,
            "non_finite_samples": self.non_finite_samples,
            "clipped_samples": self.clipped_samples,
            "zero_samples": self.zero_samples,
            "dropout_definition": f"exact zero runs >= {PARENT_PREPARATION_DROPOUT_FRAMES} samples",
            "dropout_samples": self.dropout_samples,
            "dropout_runs": self.dropout_runs,
            "longest_dropout_samples": self.longest_dropout_samples,
            "longest_dropout_seconds": self.longest_dropout_samples / sample_rate,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "peak_abs": self.peak_abs,
            "rms": math.sqrt(self.sum_squares / self.finite_samples) if self.finite_samples else None,
        }


def _parent_path_is_within(path: Path, root: Path) -> bool:
    """Return whether a path resolves below a task-owned root."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _existing_storage_root(path: Path) -> Path:
    """Return an existing ancestor suitable for a free-space probe."""

    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _free_bytes(path: Path) -> int:
    """Return available bytes for the filesystem containing a path."""

    root = _existing_storage_root(path)
    try:
        usage = os.statvfs(root)
    except OSError as error:
        raise PreparationError("cannot probe free space", {"path": str(root), "error": str(error)}) from error
    return int(usage.f_frsize * usage.f_bavail)


def _directory_bytes_bounded(root: Path) -> int:
    """Return bytes below a directory without following directory symlinks."""

    if not root.exists():
        return 0
    total = 0
    try:
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    except OSError as error:
        raise PreparationError("cannot inspect bounded storage", {"root": str(root), "error": str(error)}) from error
    return total


def _require_sha256(value: object, label: str) -> str:
    """Return a non-placeholder SHA-256 digest."""

    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise PreparationError(f"{label} must be a SHA-256 hex digest")
    if is_placeholder_hash(value):
        raise PreparationError(f"{label} must be a real SHA-256 digest")
    return value.lower()


def _resampler_executable_facts(executable: str) -> dict[str, str]:
    """Return exact identity facts for the executable that performs resampling."""

    path = Path(executable).resolve()
    if not path.is_file():
        raise PreparationError("resampler executable is missing")
    try:
        digest = sha256_file(path)
    except OSError as error:
        raise PreparationError("resampler executable cannot be hashed") from error
    try:
        version = subprocess.run(
            [path.as_posix(), "-version"],
            check=False,
            capture_output=True,
            text=True,
            env=scrubbed_environment(),
        )
    except OSError as error:
        raise PreparationError("resampler executable version cannot be read") from error
    version_line = next((line.strip() for line in version.stdout.splitlines() if line.strip()), "")
    if version.returncode != 0 or not version_line:
        raise PreparationError("resampler executable version cannot be read")
    return {"path": path.as_posix(), "sha256": digest, "version": version_line}


def _source_audio_info(path: Path) -> tuple[int, int, int, str]:
    """Read source audio metadata without decoding the full file."""

    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError, TypeError) as error:
        raise PreparationError("source audio cannot be decoded", {"path": str(path), "error": str(error)}) from error
    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    frames = int(info.frames)
    if sample_rate <= 0 or channels <= 0 or frames <= 0:
        raise PreparationError(
            "source audio has no usable decoded samples",
            {"path": str(path), "sample_rate": sample_rate, "channels": channels, "frames": frames},
        )
    return sample_rate, channels, frames, str(info.subtype or "")


def _select_channel(selected_channel: int | str | None, source_channels: int) -> tuple[int, str]:
    """Validate an explicit channel index or the mono identity assertion."""

    if isinstance(selected_channel, bool):
        raise PreparationError("selected_channel must be an integer index or validated mono identity")
    if isinstance(selected_channel, int):
        if selected_channel < 0 or selected_channel >= source_channels:
            raise PreparationError(
                "selected channel is outside the decoded source",
                {"selected_channel": selected_channel, "channels": source_channels},
            )
        return selected_channel, "index"
    if selected_channel is None or str(selected_channel).lower() in {
        "mono",
        "identity_mono",
        "validated_identity_mono",
        "validated-mono",
    }:
        if source_channels != 1:
            raise PreparationError(
                "multi-channel source requires an explicit selected channel",
                {"channels": source_channels},
            )
        return 0, "validated_identity_mono"
    raise PreparationError(
        "selected_channel must be an integer index or validated mono identity",
        {"selected_channel": str(selected_channel)},
    )


def _flac_output_subtype(source_subtype: str) -> tuple[str, str]:
    """Choose a deterministic FLAC subtype and describe any sample conversion."""

    normalized = source_subtype.upper()
    if "PCM_24" in normalized:
        return "PCM_24", "decoded PCM_24 preserved"
    if "PCM_16" in normalized:
        return "PCM_16", "decoded PCM_16 preserved"
    if "PCM_S8" in normalized or "PCM_U8" in normalized:
        return "PCM_16", "decoded integer samples written as PCM_16"
    if "FLOAT" in normalized or "DOUBLE" in normalized:
        return "PCM_24", "decoded floating samples written as deterministic PCM_24"
    return "PCM_16", "decoded samples written as deterministic PCM_16"


def _normalise_interval(item: Any, parent_id: str) -> dict[str, object]:
    """Normalize one speaker interval without judging its semantic quality."""

    if isinstance(item, RttmInterval):
        recording_id, start, end, speaker, redacted = (
            item.recording_id,
            item.start,
            item.end,
            item.speaker,
            item.redacted,
        )
    elif isinstance(item, Mapping):
        recording_id = item.get("recording_id", parent_id)
        start = item.get("start")
        end = item.get("end")
        speaker = item.get("speaker")
        redacted = item.get("redacted", False)
    elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        if len(item) == 3:
            recording_id, start, end, speaker = parent_id, item[0], item[1], item[2]
        elif len(item) == 4:
            recording_id, start, end, speaker = item
        else:
            raise PreparationError("speaker interval tuple must have three or four fields")
        redacted = False
    else:
        raise PreparationError("speaker interval has an unsupported shape")
    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError) as error:
        raise PreparationError("speaker interval times must be numeric") from error
    if not math.isfinite(start_value) or not math.isfinite(end_value) or end_value <= start_value:
        raise PreparationError("speaker interval must have finite positive duration")
    if not isinstance(recording_id, str) or not recording_id:
        raise PreparationError("speaker interval recording_id must be a non-empty string")
    if not isinstance(speaker, str) or not speaker:
        raise PreparationError("speaker interval speaker must be a non-empty string")
    if not isinstance(redacted, bool):
        raise PreparationError("speaker interval redacted must be a boolean")
    if recording_id != parent_id:
        raise PreparationError(
            "speaker interval belongs to a different parent",
            {"interval_recording_id": recording_id, "parent_id": parent_id},
        )
    return {
        "recording_id": parent_id,
        "start": start_value,
        "end": end_value,
        "speaker": speaker,
        "redacted": redacted,
    }


def _normalise_intervals(intervals: Iterable[Any], parent_id: str, source_duration: float) -> list[dict[str, object]]:
    """Normalize and bounds-check intervals while preserving all supplied labels."""

    if intervals is None:
        raise PreparationError("intervals must be an iterable, not None")
    normalized = [_normalise_interval(item, parent_id) for item in intervals]
    for item in normalized:
        if float(item["start"]) < 0.0 or float(item["end"]) > source_duration + 1e-9:
            raise PreparationError(
                "speaker interval is outside the decoded source timeline",
                {"parent_id": parent_id, "start": item["start"], "end": item["end"], "duration": source_duration},
            )
    return sorted(normalized, key=lambda item: (item["start"], item["end"], item["speaker"], item["redacted"]))


def _normalise_uem(uem: Iterable[Any] | None, parent_id: str, source_duration: float) -> list[dict[str, object]]:
    """Normalize UEM regions and retain gaps as unknown rather than silence labels."""

    if uem is None:
        regions: list[Any] = [(0.0, source_duration)]
    elif isinstance(uem, str):
        regions = []
        for line in uem.splitlines():
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 4:
                raise PreparationError("UEM line is malformed", {"line": line})
            if fields[0] != parent_id:
                raise PreparationError("UEM region belongs to a different parent", {"recording_id": fields[0]})
            try:
                start = float(fields[2])
                end = float(fields[3])
            except ValueError as error:
                raise PreparationError("UEM times must be numeric", {"line": line}) from error
            regions.append((start, end))
    elif isinstance(uem, Mapping):
        regions = [uem]
    elif isinstance(uem, Sequence) and not isinstance(uem, (bytes, bytearray)):
        if len(uem) == 2 and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in uem):
            regions = [uem]
        else:
            regions = list(uem)
    else:
        regions = list(uem)

    normalized: list[dict[str, object]] = []
    for item in regions:
        if isinstance(item, Mapping):
            recording_id = item.get("recording_id", parent_id)
            start = item.get("start")
            end = item.get("end")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) == 2:
            recording_id, start, end = parent_id, item[0], item[1]
        else:
            raise PreparationError("UEM region must be a (start, end) pair or mapping")
        if recording_id != parent_id:
            raise PreparationError("UEM region belongs to a different parent", {"recording_id": recording_id})
        try:
            start_value = float(start)
            end_value = float(end)
        except (TypeError, ValueError) as error:
            raise PreparationError("UEM times must be numeric") from error
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or start_value < 0.0
            or end_value <= start_value
            or end_value > source_duration + 1e-9
        ):
            raise PreparationError(
                "UEM region is outside the decoded source timeline",
                {"parent_id": parent_id, "start": start_value, "end": end_value, "duration": source_duration},
            )
        normalized.append({"recording_id": parent_id, "start": start_value, "end": end_value})
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    for previous, current in zip(normalized, normalized[1:]):
        if float(current["start"]) < float(previous["end"]) - 1e-9:
            raise PreparationError("UEM regions overlap", {"parent_id": parent_id})
    return normalized


def _sample_boundary(seconds: float, sample_rate: int, *, upper: bool) -> int:
    """Map a decimal-second boundary to a deterministic sample index."""

    value = Fraction(str(seconds)) * sample_rate
    return math.ceil(value) if upper else math.floor(value)


def _timing_mapping(
    *,
    parent_id: str,
    source_rate: int,
    source_frames: int,
    output_frames: int,
    intervals: Sequence[Mapping[str, object]],
    uem: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build exact sample-boundary mappings for labels and UEM regions."""

    def map_region(item: Mapping[str, object]) -> dict[str, object]:
        start = float(item["start"])
        end = float(item["end"])
        return {
            **dict(item),
            "source_start_sample": _sample_boundary(start, source_rate, upper=False),
            "source_end_sample": _sample_boundary(end, source_rate, upper=True),
            "canonical_start_sample": _sample_boundary(start, PARENT_PREPARATION_SAMPLE_RATE, upper=False),
            "canonical_end_sample": _sample_boundary(end, PARENT_PREPARATION_SAMPLE_RATE, upper=True),
        }

    return {
        "parent_id": parent_id,
        "source": {
            "sample_rate": source_rate,
            "sample_count": source_frames,
            "duration_seconds": source_frames / source_rate,
        },
        "canonical": {
            "sample_rate": PARENT_PREPARATION_SAMPLE_RATE,
            "sample_count": output_frames,
            "duration_seconds": output_frames / PARENT_PREPARATION_SAMPLE_RATE,
        },
        "sample_rate_ratio": {
            "numerator": PARENT_PREPARATION_SAMPLE_RATE,
            "denominator": source_rate,
        },
        "mapping_rule": "timeline seconds are preserved; sample boundaries use floor(start) and ceil(end)",
        "sample_index_rule": "canonical_index = source_index * 16000 / source_sample_rate",
        "intervals": [map_region(item) for item in intervals],
        "uem": [map_region(item) for item in uem],
    }


def _label_texts(
    parent_id: str, intervals: Sequence[Mapping[str, object]], uem: Sequence[Mapping[str, object]]
) -> tuple[str, str]:
    """Render deterministic RTTM and UEM text for the selected parent."""

    rttm_lines = [
        "SPEAKER {parent} 1 {start:.9f} {duration:.9f} <NA> <NA> {speaker} <NA>".format(
            parent=parent_id,
            start=float(item["start"]),
            duration=float(item["end"]) - float(item["start"]),
            speaker=str(item["speaker"]),
        )
        for item in intervals
    ]
    uem_lines = [f"{parent_id} 1 {float(item['start']):.9f} {float(item['end']):.9f}" for item in uem]
    return "\n".join(rttm_lines) + ("\n" if rttm_lines else ""), "\n".join(uem_lines) + ("\n" if uem_lines else "")


def _fsync_directory(path: Path) -> None:
    """Flush a directory after publishing a canonical parent group."""

    try:
        descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_atomic_temp(path: Path, text: str) -> Path:
    """Write one receipt or label to its reserved temporary path."""

    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise PreparationError("stale partial canonical file exists", {"path": str(temporary)})
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _inspect_canonical_audio(path: Path) -> tuple[int, dict[str, object], str]:
    """Decode one canonical FLAC in blocks and return count, stats, and hash."""

    digest = sha256_file(path)
    try:
        info = sf.info(str(path))
        if int(info.samplerate) != PARENT_PREPARATION_SAMPLE_RATE or int(info.channels) != 1:
            raise PreparationError(
                "canonical audio has wrong decoded identity",
                {"path": str(path), "sample_rate": int(info.samplerate), "channels": int(info.channels)},
            )
        stats = _AudioStatistics()
        clip_threshold = (
            1.0 - 1.0 / (1 << 23) if "PCM_24" in str(info.subtype or "").upper() else PARENT_PREPARATION_CLIP_THRESHOLD
        )
        with sf.SoundFile(str(path), mode="r") as source:
            while True:
                block = source.read(PARENT_PREPARATION_BLOCK_FRAMES, dtype="float64", always_2d=True)
                if block.size == 0:
                    break
                stats.update(block[:, 0], clip_threshold=clip_threshold)
        facts = stats.as_dict(sample_rate=PARENT_PREPARATION_SAMPLE_RATE)
    except (OSError, RuntimeError, TypeError) as error:
        raise PreparationError(
            "canonical audio cannot be decoded", {"path": str(path), "error": str(error)}
        ) from error
    if not facts["finite"] or int(facts["sample_count"]) <= 0:
        raise PreparationError("canonical audio contains no finite samples", {"path": str(path), "stats": facts})
    return int(facts["sample_count"]), facts, digest


def _facts_from_receipt(
    receipt: Mapping[str, object], destination: Path, receipt_path: Path, *, idempotent: bool
) -> dict[str, Any]:
    """Expose a stable parent recording row and the preparation receipts."""

    audio = receipt.get("audio")
    if not isinstance(audio, Mapping):
        raise PreparationError("canonical transform receipt has no audio facts")
    parent_id = str(receipt.get("parent_id") or destination.stem)
    label_sha256 = str(receipt.get("label_sha256") or "")
    rttm_path = Path(str(receipt.get("rttm_path") or destination.with_suffix(".rttm")))
    uem_path = Path(str(receipt.get("uem_path") or destination.with_suffix(".uem")))
    row = {
        "recording_id": parent_id,
        "parent_id": parent_id,
        "audio": {"path": destination.as_posix(), "sha256": str(audio.get("sha256"))},
        "rttm": {"path": rttm_path.as_posix(), "sha256": label_sha256},
        "uem": {"path": uem_path.as_posix(), "sha256": str(receipt.get("uem_sha256"))},
        "sample_count": int(audio.get("sample_count", 0)),
        "sample_rate": int(audio.get("sample_rate", PARENT_PREPARATION_SAMPLE_RATE)),
    }
    result = {
        "ok": True,
        "parent_id": parent_id,
        "canonical_path": destination.as_posix(),
        "audio_path": destination.as_posix(),
        "receipt_path": receipt_path.as_posix(),
        "source_sha256": receipt.get("source_sha256"),
        "canonical_sha256": audio.get("sha256"),
        "output_sha256": audio.get("sha256"),
        "label_sha256": label_sha256,
        "label_hash": label_sha256,
        "uem_sha256": receipt.get("uem_sha256"),
        "sample_count": int(audio.get("sample_count", 0)),
        "sample_rate": int(audio.get("sample_rate", PARENT_PREPARATION_SAMPLE_RATE)),
        "channels": int(audio.get("channels", 1)),
        "audio_statistics": receipt.get("audio_statistics"),
        "source_audio_statistics": receipt.get("source_audio_statistics"),
        "timing_mapping": receipt.get("timing_mapping"),
        "transform_receipt": receipt.get("transform"),
        "transform_sha256": receipt.get("transform_sha256"),
        "receipt": dict(receipt),
        "recording": row,
        "idempotent": idempotent,
    }
    return result


def prepare_parent(
    source_audio: Path,
    selected_channel: int | str | None,
    expected_source_sha256: str,
    intervals: Iterable[Any],
    uem: Iterable[Any] | None,
    destination: Path,
    disk_limits: DiskLimits,
    *,
    parent_id: str | None = None,
    label_destination: Path | None = None,
    uem_destination: Path | None = None,
) -> dict[str, Any]:
    """Prepare one real parent as a bounded, lossless mono FLAC and receipt group.

    The source is decoded in blocks at 16 kHz. Other rates use the installed
    ffmpeg ``libswresample`` Kaiser-windowed-sinc resampler with fixed options.
    This function records mechanical content facts only; permission, label
    quality, and acceptance remain owners of the caller's contracts and
    acceptance modules.
    """

    source = Path(source_audio).expanduser()
    destination = Path(destination).expanduser()
    if not source.is_file():
        raise PreparationError("source audio file is missing", {"path": str(source)})
    if destination.suffix.lower() != ".flac":
        raise PreparationError("canonical destination must use the .flac suffix", {"path": str(destination)})
    staging_root = Path(disk_limits.staging_root).expanduser().resolve()
    cache_root = Path(disk_limits.cache_root).expanduser().resolve()
    destination_resolved = destination.resolve(strict=False)
    if not _parent_path_is_within(destination_resolved, staging_root):
        raise PreparationError(
            "canonical destination must be under disk_limits.staging_root",
            {"destination": str(destination_resolved), "staging_root": str(staging_root)},
        )
    source_resolved = source.resolve()
    if destination_resolved == source_resolved:
        raise PreparationError("canonical destination must not overwrite source audio", {"path": str(source_resolved)})
    if (
        int(disk_limits.max_staging_bytes) <= 0
        or int(disk_limits.max_cache_bytes) <= 0
        or int(disk_limits.free_space_reserve_bytes) < 0
        or int(disk_limits.concurrency) <= 0
    ):
        raise PreparationError("disk limits must contain positive staging/cache/concurrency bounds")

    label_path = (
        Path(label_destination).expanduser() if label_destination is not None else destination.with_suffix(".rttm")
    )
    uem_path = Path(uem_destination).expanduser() if uem_destination is not None else destination.with_suffix(".uem")
    receipt_path = destination.with_name(destination.name + PARENT_PREPARATION_RECEIPT_SUFFIX)
    for output_path in (label_path, uem_path, receipt_path):
        if not _parent_path_is_within(output_path.resolve(strict=False), staging_root):
            raise PreparationError(
                "canonical labels and receipt must be under disk_limits.staging_root",
                {"path": str(output_path), "staging_root": str(staging_root)},
            )
    if (
        len(
            {
                destination_resolved,
                label_path.resolve(strict=False),
                uem_path.resolve(strict=False),
                receipt_path.resolve(strict=False),
            }
        )
        != 4
    ):
        raise PreparationError("canonical output paths must be distinct")

    expected_source_sha256 = _require_sha256(expected_source_sha256, "expected_source_sha256")
    source_sha256 = sha256_file(source)
    if source_sha256 != expected_source_sha256:
        raise PreparationError(
            "source audio SHA-256 does not match expected identity",
            {"path": str(source), "expected": expected_source_sha256, "actual": source_sha256},
        )
    source_rate, source_channels, source_frames, source_subtype = _source_audio_info(source)
    channel_index, channel_kind = _select_channel(selected_channel, source_channels)
    parent = parent_id or destination.stem
    if not isinstance(parent, str) or not parent:
        raise PreparationError("parent_id must be a non-empty string")
    source_duration = source_frames / source_rate
    normalized_intervals = _normalise_intervals(intervals, parent, source_duration)
    normalized_uem = _normalise_uem(uem, parent, source_duration)
    rttm_text, uem_text = _label_texts(parent, normalized_intervals, normalized_uem)
    label_sha256 = hashlib.sha256(rttm_text.encode("utf-8")).hexdigest()
    uem_sha256 = hashlib.sha256(uem_text.encode("utf-8")).hexdigest()
    label_manifest_sha256 = sha256_json(
        {
            "schema": "speakrs-parent-label-v1",
            "parent_id": parent,
            "intervals": normalized_intervals,
            "uem": normalized_uem,
        }
    )
    existing = [path for path in (destination, label_path, uem_path, receipt_path) if path.exists()]
    temporary_paths = tuple(
        path.with_name(path.name + ".partial") for path in (destination, label_path, uem_path, receipt_path)
    )
    if any(path.exists() for path in temporary_paths):
        raise PreparationError(
            "stale partial canonical file exists",
            {"paths": [str(path) for path in temporary_paths if path.exists()]},
        )
    if existing:
        if len(existing) != 4:
            raise PreparationError(
                "partial canonical parent group exists; refusing to overwrite it",
                {"existing": [str(path) for path in existing]},
            )
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise PreparationError("existing canonical transform receipt is unreadable") from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != PARENT_PREPARATION_SCHEMA
            or receipt.get("schema_version") != PARENT_PREPARATION_SCHEMA_VERSION
            or receipt.get("source_sha256") != source_sha256
            or receipt.get("label_sha256") != label_sha256
            or receipt.get("uem_sha256") != uem_sha256
            or receipt.get("parent_id") != parent
        ):
            raise PreparationError("existing canonical parent has a different transform, source, or label identity")
        count, _, digest = _inspect_canonical_audio(destination)
        audio = receipt.get("audio")
        if (
            not isinstance(audio, Mapping)
            or audio.get("sha256") != digest
            or int(audio.get("sample_count", -1)) != count
        ):
            raise PreparationError("existing canonical parent receipt does not match decoded audio")
        if (
            hashlib.sha256(label_path.read_bytes()).hexdigest() != label_sha256
            or hashlib.sha256(uem_path.read_bytes()).hexdigest() != uem_sha256
        ):
            raise PreparationError("existing canonical parent labels do not match their receipt")
        return _facts_from_receipt(receipt, destination, receipt_path, idempotent=True)

    ffmpeg_executable = None
    ffmpeg_facts = None
    if source_rate != PARENT_PREPARATION_SAMPLE_RATE:
        ffmpeg_executable = shutil.which("ffmpeg")
        if ffmpeg_executable is None:
            raise PreparationError("resampling requires an installed ffmpeg executable")
        ffmpeg_facts = _resampler_executable_facts(ffmpeg_executable)

    output_frames = max(1, math.ceil(source_frames * PARENT_PREPARATION_SAMPLE_RATE / source_rate))
    output_subtype, sample_conversion = _flac_output_subtype(source_subtype)
    bytes_per_sample = 3 if output_subtype == "PCM_24" else 2
    predicted_audio_bytes = output_frames * bytes_per_sample + 64 * 1024
    predicted_label_bytes = len(rttm_text.encode("utf-8")) + len(uem_text.encode("utf-8"))
    predicted_receipt_bytes = 32 * 1024
    predicted_bytes = predicted_audio_bytes + predicted_label_bytes + predicted_receipt_bytes
    used_staging = _directory_bytes_bounded(staging_root)
    used_cache = _directory_bytes_bounded(cache_root)
    if used_cache > int(disk_limits.max_cache_bytes):
        raise PreparationError(
            "cache cap exhausted before parent preparation",
            {"used_bytes": used_cache, "cap_bytes": int(disk_limits.max_cache_bytes)},
        )
    if used_staging + predicted_bytes > int(disk_limits.max_staging_bytes):
        raise PreparationError(
            "staging cap exhausted before parent preparation",
            {
                "used_bytes": used_staging,
                "predicted_bytes": predicted_bytes,
                "cap_bytes": int(disk_limits.max_staging_bytes),
            },
        )
    free_bytes = _free_bytes(staging_root)
    if free_bytes < int(disk_limits.free_space_reserve_bytes) + predicted_bytes:
        raise PreparationError(
            "free-space reserve would be crossed before parent preparation",
            {
                "free_bytes": free_bytes,
                "reserve_bytes": int(disk_limits.free_space_reserve_bytes),
                "predicted_bytes": predicted_bytes,
            },
        )

    temporary_audio, temporary_label, temporary_uem, temporary_receipt = temporary_paths
    for output_path in (destination, label_path, uem_path, receipt_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    source_statistics = _AudioStatistics()
    try:
        if source_rate == PARENT_PREPARATION_SAMPLE_RATE:
            with (
                sf.SoundFile(str(source), mode="r") as source_handle,
                sf.SoundFile(
                    str(temporary_audio),
                    mode="w",
                    samplerate=PARENT_PREPARATION_SAMPLE_RATE,
                    channels=1,
                    format="FLAC",
                    subtype=output_subtype,
                ) as output_handle,
            ):
                decoded_frames = 0
                while True:
                    block = source_handle.read(PARENT_PREPARATION_BLOCK_FRAMES, dtype="float64", always_2d=True)
                    if block.size == 0:
                        break
                    selected = block[:, channel_index]
                    source_statistics.update(selected)
                    if not np.isfinite(selected).all():
                        raise PreparationError(
                            "source audio contains non-finite decoded samples", {"path": str(source)}
                        )
                    output_handle.write(selected)
                    decoded_frames += int(selected.shape[0])
                if decoded_frames != source_frames:
                    raise PreparationError(
                        "decoded source frame count changed while preparing parent",
                        {"actual": decoded_frames, "expected": source_frames},
                    )
        else:
            assert ffmpeg_executable is not None
            assert ffmpeg_facts is not None
            output_sample_format = "s32" if output_subtype == "PCM_24" else "s16"
            filter_graph = (
                f"pan=mono|c0=c{channel_index},"
                "aresample="
                "resampler=swr:"
                f"filter_type=kaiser:filter_size={PARENT_PREPARATION_SWR_FILTER_SIZE}:"
                f"phase_shift={PARENT_PREPARATION_SWR_PHASE_SHIFT}:"
                "linear_interp=0:exact_rational=1:"
                f"cutoff={PARENT_PREPARATION_SWR_CUTOFF}:"
                f"kaiser_beta={PARENT_PREPARATION_SWR_KAISER_BETA}:"
                f"dither_method=none:osr={PARENT_PREPARATION_SAMPLE_RATE}"
            )
            command = [
                ffmpeg_executable,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-map_metadata",
                "-1",
                "-vn",
                "-sn",
                "-dn",
                "-af",
                filter_graph,
                "-ar",
                str(PARENT_PREPARATION_SAMPLE_RATE),
                "-ac",
                "1",
                "-sample_fmt",
                output_sample_format,
                "-c:a",
                "flac",
                "-compression_level",
                "5",
                "-flags",
                "+bitexact",
                "-fflags",
                "+bitexact",
                "-f",
                "flac",
                "-y",
                str(temporary_audio),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=scrubbed_environment(),
            )
            if completed.returncode != 0:
                raise PreparationError(
                    "ffmpeg libswresample conversion failed",
                    {"path": str(source), "stderr": completed.stderr[-500:]},
                )
            with sf.SoundFile(str(source), mode="r") as source_handle:
                while True:
                    block = source_handle.read(PARENT_PREPARATION_BLOCK_FRAMES, dtype="float64", always_2d=True)
                    if block.size == 0:
                        break
                    selected = block[:, channel_index]
                    source_statistics.update(selected)
                    if not np.isfinite(selected).all():
                        raise PreparationError(
                            "source audio contains non-finite decoded samples", {"path": str(source)}
                        )

        output_count, audio_statistics, output_sha256 = _inspect_canonical_audio(temporary_audio)
        if output_count <= 0:
            raise PreparationError("canonical audio has no decoded samples", {"path": str(temporary_audio)})
        source_stats_dict = source_statistics.as_dict(sample_rate=source_rate)
        timing_mapping = _timing_mapping(
            parent_id=parent,
            source_rate=source_rate,
            source_frames=source_frames,
            output_frames=output_count,
            intervals=normalized_intervals,
            uem=normalized_uem,
        )
        transform = {
            "schema": PARENT_PREPARATION_SCHEMA,
            "schema_version": PARENT_PREPARATION_SCHEMA_VERSION,
            "target": "lossless FLAC, 16 kHz, mono",
            "selected_channel": channel_kind,
            "selected_channel_index": channel_index,
            "source_sample_rate": source_rate,
            "source_channels": source_channels,
            "source_subtype": source_subtype,
            "output_subtype": output_subtype,
            "sample_conversion": sample_conversion,
            "decoder": "soundfile block streaming" if source_rate == PARENT_PREPARATION_SAMPLE_RATE else "ffmpeg",
            "decoder_executable": ffmpeg_facts,
            "resampler": None
            if source_rate == PARENT_PREPARATION_SAMPLE_RATE
            else {
                "name": "libswresample",
                "engine": "swr",
                "filter_type": "kaiser",
                "filter_size": PARENT_PREPARATION_SWR_FILTER_SIZE,
                "phase_shift": PARENT_PREPARATION_SWR_PHASE_SHIFT,
                "linear_interp": False,
                "exact_rational": True,
                "cutoff": PARENT_PREPARATION_SWR_CUTOFF,
                "kaiser_beta": PARENT_PREPARATION_SWR_KAISER_BETA,
                "dither_method": "none",
                "target_sample_rate": PARENT_PREPARATION_SAMPLE_RATE,
                "output_sample_format": "s32" if output_subtype == "PCM_24" else "s16",
                "filter_graph": filter_graph,
            },
            "metadata": "all source container metadata omitted",
            "flac_compression_level": 5,
        }
        receipt = {
            "schema": PARENT_PREPARATION_SCHEMA,
            "schema_version": PARENT_PREPARATION_SCHEMA_VERSION,
            "parent_id": parent,
            "source_audio": str(source),
            "source_sha256": source_sha256,
            "rttm_path": label_path.as_posix(),
            "uem_path": uem_path.as_posix(),
            "selected_channel": {"kind": channel_kind, "index": channel_index, "source_channels": source_channels},
            "audio": {
                "path": destination.as_posix(),
                "sha256": output_sha256,
                "sample_count": output_count,
                "sample_rate": PARENT_PREPARATION_SAMPLE_RATE,
                "channels": 1,
                "codec": "FLAC",
                "subtype": output_subtype,
            },
            "label_sha256": label_sha256,
            "uem_sha256": uem_sha256,
            "label_manifest_sha256": label_manifest_sha256,
            "timing_mapping": timing_mapping,
            "audio_statistics": audio_statistics,
            "source_audio_statistics": source_stats_dict,
            "transform": transform,
            "transform_sha256": sha256_json(transform),
            "bounds": {
                "staging_root": staging_root.as_posix(),
                "cache_root": cache_root.as_posix(),
                "staging_used_before_bytes": used_staging,
                "cache_used_before_bytes": used_cache,
                "predicted_output_bytes": predicted_bytes,
                "free_bytes_before": free_bytes,
                "free_space_reserve_bytes": int(disk_limits.free_space_reserve_bytes),
            },
        }
        temporary_label = _write_atomic_temp(label_path, rttm_text)
        temporary_uem = _write_atomic_temp(uem_path, uem_text)
        temporary_receipt = _write_atomic_temp(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if _directory_bytes_bounded(staging_root) > int(disk_limits.max_staging_bytes):
            raise PreparationError("staging cap exhausted while publishing canonical parent")
        temporary_audio.replace(destination)
        temporary_label.replace(label_path)
        temporary_uem.replace(uem_path)
        temporary_receipt.replace(receipt_path)
        _fsync_directory(destination.parent)
        return _facts_from_receipt(receipt, destination, receipt_path, idempotent=False)
    except BaseException:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        for path in (destination, label_path, uem_path, receipt_path):
            if path not in existing:
                path.unlink(missing_ok=True)
        raise


# Keep descriptive aliases for callers that use the preparation vocabulary.
canonicalize_parent = prepare_parent
prepare_parent_audio = prepare_parent
canonicalize_parent_audio = prepare_parent


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
    if not frozen:
        raise PreparationError("ICSI frozen-test exclusions cannot be empty")
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


class _SourceDownloadInterrupted(Exception):
    """A source response ended before its advertised bytes arrived."""


def _source_download_header(response: Any, name: str) -> str | None:
    """Read one response header without depending on a concrete HTTP type."""

    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    if hasattr(headers, "items"):
        for key, value in headers.items():
            if str(key).lower() == name.lower():
                return str(value)
    getter = getattr(headers, "get", None)
    if getter is not None:
        value = getter(name)
        if value is not None:
            return str(value)
    return None


def _source_download_status(response: Any) -> int:
    """Read the HTTP status from a real or test response."""

    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "code", None)
    if status is None:
        status = response.getcode()
    return int(status)


def _source_download_receipt(
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    attempts: int,
    resumed_from_bytes: int,
    *,
    idempotent: bool,
) -> dict[str, Any]:
    """Build the private, URL-free source download receipt."""

    receipt = {
        "schema": SOURCE_DOWNLOAD_SCHEMA,
        "schema_version": SOURCE_DOWNLOAD_SCHEMA_VERSION,
        "destination": destination.as_posix(),
        "size": expected_size,
        "sha256": expected_sha256,
        "attempts": attempts,
        "resumed_from_bytes": resumed_from_bytes,
        "completed": True,
    }
    return {
        "path": destination.as_posix(),
        "size": expected_size,
        "sha256": expected_sha256,
        "attempts": attempts,
        "resumed_from_bytes": resumed_from_bytes,
        "idempotent": idempotent,
        "receipt": receipt,
    }


def download_source_file(
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    disk_limits: DiskLimits,
    *,
    max_attempts: int = 3,
    block_size: int = SOURCE_DOWNLOAD_BLOCK_BYTES,
) -> dict[str, Any]:
    """Download one source into a bounded, resumable, content-addressed path.

    The sibling ``.part`` file is the only mutable download state. Each HTTP
    attempt starts at its current size, and only the final size and SHA-256
    permit atomic publication. The URL is intentionally excluded from all
    receipts and raised errors because it may contain private query tokens.
    """

    if not isinstance(url, str) or not url:
        raise PreparationError("source download URL must be a non-empty string")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise PreparationError("expected source size must be a positive integer")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise PreparationError("max_attempts must be a positive integer")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise PreparationError("block_size must be a positive integer")
    expected_sha256 = _require_sha256(expected_sha256, "expected_sha256")

    try:
        staging_root = Path(disk_limits.staging_root).expanduser().resolve()
        cache_root = Path(disk_limits.cache_root).expanduser().resolve()
        max_staging_bytes = int(disk_limits.max_staging_bytes)
        max_cache_bytes = int(disk_limits.max_cache_bytes)
        reserve_bytes = int(disk_limits.free_space_reserve_bytes)
        concurrency = int(disk_limits.concurrency)
    except (AttributeError, TypeError, ValueError) as error:
        raise PreparationError("disk limits are invalid") from error
    if max_staging_bytes <= 0 or max_cache_bytes <= 0 or reserve_bytes < 0 or concurrency <= 0:
        raise PreparationError("disk limits must contain positive staging/cache/concurrency bounds")

    destination = Path(destination).expanduser().resolve(strict=False)
    roots = ((staging_root, max_staging_bytes, "staging"), (cache_root, max_cache_bytes, "cache"))
    matching_roots = [item for item in roots if _parent_path_is_within(destination, item[0])]
    if not matching_roots:
        raise PreparationError("source destination must be under staging or cache storage")
    owner_root, owner_cap, owner_label = max(matching_roots, key=lambda item: len(item[0].parts))
    part = destination.with_name(destination.name + ".part")

    if destination.exists():
        if not destination.is_file():
            raise PreparationError("source destination exists but is not a file")
        if destination.stat().st_size != expected_size:
            raise PreparationError("existing source destination has an unexpected size")
        actual_sha256 = sha256_file(destination)
        if actual_sha256 != expected_sha256:
            raise PreparationError("existing source destination has an unexpected SHA-256")
        return _source_download_receipt(
            destination,
            expected_size,
            expected_sha256,
            0,
            0,
            idempotent=True,
        )

    if part.exists():
        if part.is_symlink() or not part.is_file():
            raise PreparationError("source partial path exists but is not a regular file")
        initial_size = part.stat().st_size
    else:
        initial_size = 0
    if initial_size > expected_size:
        raise PreparationError("source partial file is larger than expected")
    if initial_size == expected_size:
        actual_sha256 = sha256_file(part)
        if actual_sha256 != expected_sha256:
            raise PreparationError("source partial file has an unexpected SHA-256")
        if destination.exists():
            raise PreparationError("source destination appeared while publishing")
        part.replace(destination)
        _fsync_directory(destination.parent)
        return _source_download_receipt(
            destination,
            expected_size,
            expected_sha256,
            0,
            initial_size,
            idempotent=False,
        )

    storage_usage = {}
    for root, cap, label in roots:
        used = _directory_bytes_bounded(root)
        storage_usage[label] = used
        if used > cap:
            raise PreparationError(
                f"{label} cap exhausted before source download",
                {"used_bytes": used, "cap_bytes": cap},
            )
    remaining = expected_size - initial_size
    owner_used = storage_usage[owner_label]
    if owner_used + remaining > owner_cap:
        raise PreparationError(
            f"{owner_label} cap exhausted before source download",
            {"used_bytes": owner_used, "remaining_bytes": remaining, "cap_bytes": owner_cap},
        )
    free_bytes = _free_bytes(owner_root)
    if free_bytes < reserve_bytes + remaining:
        raise PreparationError(
            "free-space reserve would be crossed before source download",
            {"free_bytes": free_bytes, "reserve_bytes": reserve_bytes, "remaining_bytes": remaining},
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    resumed_from_bytes = initial_size
    attempts = 0
    last_size = initial_size

    def protocol_error(message: str, *, status: int | None = None, offset: int | None = None) -> None:
        details: dict[str, int] = {}
        if status is not None:
            details["status"] = status
        if offset is not None:
            details["offset"] = offset
        raise PreparationError(message, details)

    while True:
        current_size = part.stat().st_size if part.exists() else 0
        if current_size < last_size:
            raise PreparationError("source partial file shrank during download", {"previous_bytes": last_size})
        if current_size > expected_size:
            raise PreparationError("source partial file grew beyond expected size")
        last_size = current_size
        if current_size == expected_size:
            actual_sha256 = sha256_file(part)
            if actual_sha256 != expected_sha256:
                raise PreparationError("source partial file has an unexpected SHA-256")
            if destination.exists():
                raise PreparationError("source destination appeared while publishing")
            part.replace(destination)
            _fsync_directory(destination.parent)
            return _source_download_receipt(
                destination,
                expected_size,
                expected_sha256,
                attempts,
                resumed_from_bytes,
                idempotent=False,
            )
        if attempts >= max_attempts:
            raise PreparationError(
                "source download did not complete within the attempt limit",
                {"attempts": attempts, "received_bytes": current_size},
            )

        offset = current_size
        # The range is sent even for the first request so every attempt has an explicit byte origin
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"}, method="GET")
        except Exception:
            raise PreparationError("source download request is invalid") from None
        attempts += 1
        response = None
        try:
            response = urllib.request.urlopen(request)
            status = _source_download_status(response)
            if status not in (200, 206):
                protocol_error("source server returned an unsupported HTTP status", status=status, offset=offset)

            content_range = _source_download_header(response, "Content-Range")
            content_length = _source_download_header(response, "Content-Length")
            advertised_length: int | None = None
            if status == 206:
                if content_range is None:
                    protocol_error("source server omitted Content-Range", status=status, offset=offset)
                match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", content_range.strip(), flags=re.IGNORECASE)
                if match is None:
                    protocol_error("source server returned an invalid Content-Range", status=status, offset=offset)
                range_start, range_end, range_total = (int(value) for value in match.groups())
                if (
                    range_start != offset
                    or range_end < range_start
                    or range_total != expected_size
                    or range_end >= expected_size
                ):
                    protocol_error(
                        "source Content-Range does not match the expected source", status=status, offset=offset
                    )
                advertised_length = range_end - range_start + 1
            elif offset != 0:
                protocol_error("source server ignored the resume range", status=status, offset=offset)

            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    protocol_error("source server returned an invalid Content-Length", status=status, offset=offset)
                if (
                    declared_length < 0
                    or (advertised_length is not None and declared_length != advertised_length)
                    or (status == 200 and declared_length != expected_size)
                ):
                    protocol_error(
                        "source Content-Length does not match the expected source", status=status, offset=offset
                    )
                if advertised_length is None:
                    advertised_length = declared_length

            received = 0
            try:
                with part.open("ab", buffering=0) as output:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        if not isinstance(chunk, (bytes, bytearray, memoryview)):
                            protocol_error(
                                "source server returned a non-byte response body", status=status, offset=offset
                            )
                        chunk = bytes(chunk)
                        if received + len(chunk) > expected_size - offset:
                            protocol_error("source response exceeded the expected size", status=status, offset=offset)
                        if advertised_length is not None and received + len(chunk) > advertised_length:
                            protocol_error(
                                "source response exceeded its advertised length", status=status, offset=offset
                            )
                        output.write(chunk)
                        received += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            except PreparationError:
                raise
            except Exception:
                raise _SourceDownloadInterrupted
            if advertised_length is not None and received != advertised_length:
                raise _SourceDownloadInterrupted
        except PreparationError:
            raise
        except urllib.error.HTTPError as error:
            protocol_error("source server rejected the download request", status=int(error.code), offset=offset)
        except Exception:
            # Transport failures are retried without exposing exception text, which can contain the URL
            if attempts >= max_attempts:
                raise PreparationError(
                    "source download was interrupted",
                    {"attempts": attempts, "received_bytes": part.stat().st_size if part.exists() else 0},
                ) from None
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass


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
    for row in parsed_rows:
        if row.rejected:
            continue
        if is_placeholder_hash(row.audio_sha256) or is_placeholder_hash(row.label_sha256):
            raise PreparationError(
                "placeholder hashes cannot seal a real release",
                {"recording_id": row.recording_id},
            )
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
    rttm_path = directory / "rttm"
    if rows:
        if not rttm_path.exists():
            raise PreparationError("empty expected RTTM cannot seal a real release", {"path": str(rttm_path)})
        rttm_text = rttm_path.read_text(encoding="utf-8")
        if not any(line.startswith("SPEAKER") for line in rttm_text.splitlines()):
            raise PreparationError("empty expected RTTM cannot seal a real release", {"path": str(rttm_path)})
    elif not rttm_path.exists():
        atomic_write_text(rttm_path, "")
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
    for row in recordings:
        if row.rejected:
            continue
        if is_placeholder_hash(row.audio_sha256) or is_placeholder_hash(row.label_sha256):
            raise PreparationError(
                "placeholder hashes cannot seal a real release",
                {"recording_id": row.recording_id},
            )
        audio = audio_root / row.corpus / f"{row.recording_id}.flac"
        actual = sha256_file(audio)
        if actual != row.audio_sha256:
            raise PreparationError(
                "audio sha256 does not match the sealed identity",
                {"recording_id": row.recording_id},
            )
        decoded = _flac_sample_count(audio)
        if row.sample_count is None or decoded != row.sample_count:
            raise PreparationError(
                "wrong decoded duration",
                {"recording_id": row.recording_id, "actual": decoded, "expected": row.sample_count},
            )
    for split_name in ("train",):
        rttm_path = release_root / split_name / "rttm"
        if not rttm_path.is_file() or not any(
            line.startswith("SPEAKER") for line in rttm_path.read_text(encoding="utf-8").splitlines()
        ):
            raise PreparationError("empty expected RTTM cannot seal a real release", {"path": str(rttm_path)})
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


def _transcode_stream(stream, dest: Path) -> None:
    """Write one audio stream as mono 16 kHz FLAC through the shared transcoder."""

    module = _load_module("prepare_full_corpus", RECIPE_DIR / "prepare_full_corpus.py")
    if module.audio_ready(dest, module.ChannelPolicy.FIRST):
        return
    module.transcode(stream, dest, module.ChannelPolicy.FIRST)


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


def _pcm_s16le_wav(pcm: bytes, sample_rate: int = NOTSOFAR_SIM_SAMPLE_RATE) -> bytes:
    """Wrap mono little-endian PCM16 in a WAV header."""

    if len(pcm) % 2:
        raise PreparationError("pcm16 length is odd", {"bytes": len(pcm)})
    data_bytes = len(pcm)
    return (
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_bytes,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            data_bytes,
        )
        + pcm
    )


def _interleaved_s16le_channel0(pcm: bytes, channels: int) -> bytes:
    """Keep the first mixture microphone from interleaved int16 frames."""

    if channels < 1:
        raise PreparationError("mixture channel count is invalid", {"channels": channels})
    frame = 2 * channels
    if len(pcm) % frame:
        raise PreparationError(
            "mixture pcm length is not aligned",
            {"bytes": len(pcm), "channels": channels},
        )
    if channels == 1:
        return pcm
    frames = len(pcm) // frame
    out = bytearray(frames * 2)
    for index in range(frames):
        src = index * frame
        out[index * 2 : index * 2 + 2] = pcm[src : src + 2]
    return bytes(out)


def _lotusdis_member_identity(name: str) -> tuple[str, str] | None:
    path = Path(name)
    if path.suffix.lower() != ".wav":
        return None
    match = re.match(r"(Hijack_S\d+_T\d+)_(.+)$", path.stem, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1), match.group(2).lower()


def _lotusdis_select_view(members: Iterable[str], parent_id: str) -> str:
    """Pick jbl, then bt3m, bt10m, con123, then the first non-lavalier view."""

    by_device: dict[str, str] = {}
    for member in members:
        parsed = _lotusdis_member_identity(member)
        if parsed is None:
            continue
        meeting, device = parsed
        if meeting != parent_id:
            continue
        by_device[device] = member
    if not by_device:
        raise PreparationError("LOTUSDIS parent has no audio", {"parent_id": parent_id})
    for preferred in LOTUSDIS_VIEW_PREFERENCE:
        if preferred in by_device:
            return by_device[preferred]
    eligible = sorted(device for device in by_device if not device.startswith("lav"))
    if not eligible:
        raise PreparationError("LOTUSDIS parent has only lavalier views", {"parent_id": parent_id})
    return by_device[eligible[0]]


def _extract_sim_tar(archive: Path, audio_root: Path) -> list[str]:
    """Write mixture-channel-0 FLACs from one official CSS train tar."""

    audio_root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    pending_id = None
    pending_channels = NOTSOFAR_SIM_MIXTURE_CHANNELS
    with tarfile.open(archive) as tar:
        for member in tar:
            name = member.name
            if member.isfile() and name.endswith(".json") and not name.endswith("utterances.map"):
                handle = tar.extractfile(member)
                if handle is None:
                    raise PreparationError("simulated json member is unreadable", {"name": name})
                payload = json.loads(handle.read().decode("utf-8"))
                pending_id = Path(name).stem
                shape = payload.get("columns", {}).get("mixture", {}).get("shape") or [
                    0,
                    NOTSOFAR_SIM_MIXTURE_CHANNELS,
                ]
                pending_channels = int(shape[1]) if len(shape) > 1 else NOTSOFAR_SIM_MIXTURE_CHANNELS
                continue
            if not (member.isfile() and name.endswith(".mixture")):
                continue
            utterance_id = Path(name).stem
            dest = audio_root / f"{utterance_id}.flac"
            if dest.is_file():
                written.append(utterance_id)
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise PreparationError("simulated mixture member is unreadable", {"name": name})
            channels = pending_channels if pending_id == utterance_id else NOTSOFAR_SIM_MIXTURE_CHANNELS
            pcm = _interleaved_s16le_channel0(handle.read(), channels)
            _transcode_stream(io.BytesIO(_pcm_s16le_wav(pcm)), dest)
            written.append(utterance_id)
    if not written:
        raise PreparationError("simulated tar produced no mixture audio", {"archive": archive.as_posix()})
    return written


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
            "default=noprint_wrappers=1",
            path.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=scrubbed_environment(),
    )
    fields = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    if result.returncode != 0 or "duration" not in fields or "sample_rate" not in fields:
        raise PreparationError(
            "cannot probe prepared audio", {"path": path.as_posix(), "stderr": result.stderr[-200:]}
        )
    duration, rate = float(fields["duration"]), int(float(fields["sample_rate"]))
    return int(round(duration * rate))


def _row_from_audio(corpus: str, split: str, parent_id: str, audio: Path, **extra: Any) -> dict[str, Any]:
    extra = dict(extra)
    if not audio.is_file():
        raise PreparationError("real recording audio is missing", {"path": str(audio), "parent_id": parent_id})
    extra["audio_sha256"] = sha256_file(audio)
    extra["sample_count"] = _flac_sample_count(audio)
    if not extra.get("label_sha256"):
        extra["label_sha256"] = None
        extra["licence"] = extra.get("licence", "unresolved")
        extra["label_tier"] = extra.get("label_tier", "bronze")
        extra["rejected"] = True
        extra["rejection_reason"] = "measured label facts are not available for this parent"
    return _recording_row(corpus, split, parent_id, **extra)


def _recording_row(corpus: str, split: str, parent_id: str, **extra: Any) -> dict[str, Any]:
    """Build a row only from measured audio, label, and permission facts."""

    required = ("audio_sha256", "sample_count", "licence", "label_tier")
    missing = [key for key in required if key not in extra or extra[key] in (None, "")]
    rejected = bool(extra.get("rejected", False))
    if not rejected and extra.get("label_sha256") in (None, ""):
        missing.append("label_sha256")
    if missing:
        raise PreparationError(
            "real recording rows require measured audio, label, and permission facts",
            {"corpus": corpus, "parent_id": parent_id, "missing": missing},
        )
    if not rejected and is_placeholder_hash(extra["label_sha256"]):
        raise PreparationError(
            "real recording rows cannot use placeholder hashes",
            {"corpus": corpus, "parent_id": parent_id},
        )
    if is_placeholder_hash(extra["audio_sha256"]):
        raise PreparationError(
            "real recording rows cannot use placeholder hashes",
            {"corpus": corpus, "parent_id": parent_id},
        )
    if rejected and not extra.get("rejection_reason"):
        raise PreparationError(
            "rejected recording rows require an explicit reason",
            {"corpus": corpus, "parent_id": parent_id},
        )
    row = {
        "recording_id": parent_id,
        "parent_id": parent_id,
        "corpus": corpus,
        "split": split,
        "device_view": extra.get("device_view", "canonical"),
        "label_tier": extra["label_tier"],
        "licence": extra["licence"],
        "audio_sha256": extra["audio_sha256"],
        "label_sha256": extra.get("label_sha256"),
        "sample_count": extra["sample_count"],
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
                recordings.append(
                    _recording_row(corpus, split, session_id, language="en" if corpus == "AMI" else "zh")
                )
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
        rows.append(_recording_row("VoxConverse", "train", session_id, language="en"))
    for session_id in test_ids:
        rows.append(_recording_row("VoxConverse", "test", session_id, language="en"))
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
                _recording_row(
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
    source_url = NOTSOFAR_SIM_PREFIX
    try:
        listing = _azure_list(
            "https://notsofarsa.blob.core.windows.net/css-datasets"
            "?restype=container&comp=list&prefix=v1.5/1000hrs/train/"
        )
    except PreparationError:
        # Official Microsoft helper now pulls the same v1.5/1000hrs/train tree from Hugging Face.
        listing = _huggingface_sim_list(cache)
        source_url = f"https://huggingface.co/datasets/{NOTSOFAR_HF_DATASET}/{NOTSOFAR_SIM_HF_PREFIX}"
    parents = sorted(listing)
    if not parents:
        raise PreparationError("NOTSOFAR simulated train listing is empty")
    rows = _materialize_notsofar_sim_audio(spec, cache, parents)
    return {
        "recordings": rows,
        "splits": {"NOTSOFAR_sim": {"train": parents, "dev": [], "test": []}},
        "sources": [
            {
                "name": "notsofar-sim-v1.5-1000hrs-train",
                "url": source_url,
            }
        ],
    }


def _materialize_notsofar_sim_audio(spec, cache: Path, parents: list[str]) -> list[dict[str, Any]]:
    """Download official tars one at a time and keep mixture channel 0 only."""

    audio_root = spec.relocation.audio_root / "NOTSOFAR_sim"
    audio_root.mkdir(parents=True, exist_ok=True)
    resolve = f"https://huggingface.co/datasets/{NOTSOFAR_HF_DATASET}/resolve/main/{NOTSOFAR_SIM_HF_PREFIX}"
    index_path = cache / "hf-train-maps.jsonl"
    tar_items = []
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tar_items.append(json.loads(line))
    if not tar_items:
        raise PreparationError("simulated tar map index is missing", {"path": index_path.as_posix()})
    rows_by_id: dict[str, dict[str, Any]] = {}
    for item in sorted(tar_items, key=lambda row: int(row.get("index", 0))):
        ids = list(item.get("ids") or [])
        tar_name = item.get("name") or f"dataset-{int(item['index']):06d}.tar"
        missing = [utterance_id for utterance_id in ids if not (audio_root / f"{utterance_id}.flac").is_file()]
        if missing:
            tar_path = cache / tar_name
            if not tar_path.is_file():
                _curl(f"{resolve}/{tar_name}", tar_path)
            _extract_sim_tar(tar_path, audio_root)
            tar_path.unlink(missing_ok=True)
        for utterance_id in ids:
            dest = audio_root / f"{utterance_id}.flac"
            if not dest.is_file():
                raise PreparationError(
                    "simulated utterance audio is missing after extract",
                    {"id": utterance_id, "tar": tar_name},
                )
            rows_by_id[utterance_id] = _row_from_audio(
                "NOTSOFAR_sim",
                "train",
                utterance_id,
                dest,
                label_tier="bronze",
                language="en",
                device_view="mixture_ch0",
                transformations=["css_mixture_channel0", "mono_16k_flac"],
            )
    missing_parents = [parent_id for parent_id in parents if parent_id not in rows_by_id]
    if missing_parents:
        raise PreparationError(
            "simulated train listing is missing prepared audio",
            {"count": len(missing_parents), "examples": missing_parents[:10]},
        )
    return [rows_by_id[parent_id] for parent_id in parents]


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
    frozen_test: tuple[str, ...] = ()
    config_frozen = spec.relocation.evidence_root / "icsi-frozen-test.txt"
    if config_frozen.is_file():
        frozen_test = tuple(
            line.strip() for line in config_frozen.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if not frozen_test:
        raise PreparationError("ICSI frozen-test exclusions cannot be empty")
    parts = _icsi_split(meetings, frozen_test=frozen_test)
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
    if not any(parts.values()):
        raise PreparationError("LOTUSDIS produced no parent meetings")
    meeting_zip = cache / "wav.zip"
    if not meeting_zip.is_file() or meeting_zip.stat().st_size != LOTUSDIS_FULL_MEETING_BYTES:
        _gdrive_range_download(
            LOTUSDIS_FULL_MEETING_ID,
            meeting_zip,
            LOTUSDIS_FULL_MEETING_BYTES,
            cookies=cache / "gdrive.cookies",
        )
    rows = _materialize_lotusdis_audio(spec, meeting_zip, parts)
    sources = [
        {
            "name": "lotusdis-csv",
            "obtained_sha256": sha256_file(archive) if archive.is_file() else sha256_file(legacy_csv),
        },
        {
            "name": "lotusdis-full-meeting",
            "obtained_sha256": sha256_file(meeting_zip),
            "bytes": meeting_zip.stat().st_size,
        },
    ]
    return {
        "recordings": rows,
        "splits": {"LOTUSDIS": {split: list(ids) for split, ids in parts.items()}},
        "sources": sources,
    }


def _materialize_lotusdis_audio(spec, meeting_zip: Path, parts: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    """Transcode the preferred device view for every parent meeting."""

    audio_root = spec.relocation.audio_root / "LOTUSDIS"
    audio_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(meeting_zip) as zipped:
        members = zipped.namelist()
        rows = []
        for split, ids in parts.items():
            for parent_id in ids:
                member = _lotusdis_select_view(members, parent_id)
                identity = _lotusdis_member_identity(member)
                if identity is None:
                    raise PreparationError("LOTUSDIS member is not a parent view", {"member": member})
                dest = audio_root / f"{parent_id}.flac"
                if not dest.is_file():
                    with zipped.open(member) as handle:
                        _transcode_stream(handle, dest)
                rows.append(
                    _row_from_audio(
                        "LOTUSDIS",
                        split,
                        parent_id,
                        dest,
                        device_view=identity[1],
                        language="th",
                        transformations=["preferred_device_view", "mono_16k_flac"],
                    )
                )
    return rows


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


def _file_starts_with_html(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        head = handle.read(20).lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


def _parse_tar_utterances_map(data: bytes) -> list[str]:
    """Read utterance IDs from the first member of a CSS dataset tar."""

    if len(data) < 512:
        raise PreparationError("utterances.map tar header is truncated")
    name = data[0:100].split(b"\x00", 1)[0]
    if name != b"utterances.map":
        raise PreparationError(
            "expected utterances.map as first tar member",
            {"name": name.decode("utf-8", "replace")},
        )
    size_field = data[124:136].split(b"\x00", 1)[0].decode("ascii", "replace").strip()
    try:
        size = int(size_field, 8)
    except ValueError as error:
        raise PreparationError("utterances.map size is not octal") from error
    payload = data[512 : 512 + size]
    if len(payload) < size:
        raise PreparationError(
            "utterances.map is truncated",
            {"got": len(payload), "expected": size},
        )
    try:
        mapping = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise PreparationError("utterances.map is not JSON") from error
    if not isinstance(mapping, dict) or not mapping:
        raise PreparationError("utterances.map has no utterance ids")
    return sorted(mapping)


def _huggingface_sim_list(cache: Path) -> list[str]:
    """List v1.5/1000hrs/train utterance IDs from official Hugging Face tars."""

    index_path = cache / "hf-train-maps.jsonl"
    if index_path.is_file():
        ids: list[str] = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.extend(json.loads(line)["ids"])
        if ids:
            return sorted(set(ids))
    listing = _huggingface_list(NOTSOFAR_HF_DATASET, NOTSOFAR_SIM_HF_PREFIX)
    tar_names = [name for name in listing if name.endswith(".tar")]
    if len(tar_names) < 800:
        raise PreparationError(
            "Hugging Face simulated train listing is incomplete",
            {"count": len(tar_names)},
        )
    resolve = f"https://huggingface.co/datasets/{NOTSOFAR_HF_DATASET}/resolve/main/{NOTSOFAR_SIM_HF_PREFIX}"
    rows = []
    for name in tar_names:
        dest = cache / f"{name}.map.bin"
        completed = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--location",
                "--max-time",
                "90",
                "-r",
                "0-65535",
                "--output",
                str(dest),
                f"{resolve}/{name}",
            ],
            env=scrubbed_environment(),
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0 or not dest.is_file():
            dest.unlink(missing_ok=True)
            raise PreparationError("Hugging Face simulated tar map download failed", {"name": name})
        data = dest.read_bytes()
        dest.unlink(missing_ok=True)
        ids = _parse_tar_utterances_map(data)
        rows.append({"name": name, "count": len(ids), "ids": ids})
    atomic_write_text(
        index_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    parents = sorted({utterance_id for row in rows for utterance_id in row["ids"]})
    if not parents:
        raise PreparationError("Hugging Face simulated train listing is empty")
    return parents


def _gdrive_confirm_uuid(file_id: str, cookies: Path) -> str:
    cookies.parent.mkdir(parents=True, exist_ok=True)
    if not cookies.is_file():
        cookies.write_text("", encoding="utf-8")
    html_path = cookies.with_name("gdrive-confirm.html")
    completed = subprocess.run(
        [
            "curl",
            "--silent",
            "--location",
            "-A",
            GDRIVE_USER_AGENT,
            "-b",
            str(cookies),
            "-c",
            str(cookies),
            "--max-time",
            "30",
            "--output",
            str(html_path),
            f"https://drive.google.com/uc?export=download&id={file_id}",
        ],
        env=scrubbed_environment(),
        check=False,
    )
    text = html_path.read_text(errors="replace") if html_path.is_file() else ""
    if "Too many users" in text and "uuid" not in text:
        raise PreparationError("Google Drive quota blocked the confirm page", {"file_id": file_id})
    match = re.search(r'name="uuid" value="([^"]+)"', text)
    if completed.returncode != 0 or match is None:
        raise PreparationError("Google Drive confirm page is missing uuid", {"file_id": file_id})
    return match.group(1)


def _gdrive_range_download(file_id: str, destination: Path, expected_bytes: int, *, cookies: Path) -> None:
    """Download a large Drive file with Range chunks. A Range-less GET hits quota HTML."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    cookies.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        if _file_starts_with_html(destination):
            destination.unlink()
        else:
            return
    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset and _file_starts_with_html(partial):
        partial.unlink()
        offset = 0
    chunk = 8 * 1024 * 1024
    uuid = _gdrive_confirm_uuid(file_id, cookies)
    failures = 0
    while offset < expected_bytes:
        end = min(offset + chunk, expected_bytes) - 1
        tmp = partial.with_name(f"{partial.name}.chunk-{offset}")
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid}"
        completed = subprocess.run(
            [
                "curl",
                "--silent",
                "--location",
                "-A",
                GDRIVE_USER_AGENT,
                "-b",
                str(cookies),
                "-c",
                str(cookies),
                "--max-time",
                "180",
                "-r",
                f"{offset}-{end}",
                "--output",
                str(tmp),
                url,
            ],
            env=scrubbed_environment(),
            check=False,
        )
        expected = end - offset + 1
        html = tmp.is_file() and _file_starts_with_html(tmp)
        bad = completed.returncode != 0 or not tmp.is_file() or tmp.stat().st_size != expected or html
        if bad:
            tmp.unlink(missing_ok=True)
            if html:
                # Quota HTML is transient. Do not burn the hard-failure budget on it.
                chunk = max(1024 * 1024, chunk // 2)
                time.sleep(120)
            else:
                failures += 1
                if failures >= 80:
                    raise PreparationError(
                        "Google Drive range download failed",
                        {"file_id": file_id, "offset": offset, "chunk": chunk},
                    )
                time.sleep(min(30, 2 ** min(failures, 5)))
            try:
                uuid = _gdrive_confirm_uuid(file_id, cookies)
            except PreparationError:
                continue
            continue
        failures = 0
        if chunk < 8 * 1024 * 1024:
            chunk = min(8 * 1024 * 1024, chunk * 2)
        if offset == 0 and tmp.read_bytes()[:2] != b"PK":
            tmp.unlink(missing_ok=True)
            raise PreparationError("Google Drive range download is not a zip", {"file_id": file_id})
        with partial.open("ab") as dest, tmp.open("rb") as src:
            dest.write(src.read())
            dest.flush()
        tmp.unlink(missing_ok=True)
        offset = partial.stat().st_size
    if partial.stat().st_size != expected_bytes:
        raise PreparationError(
            "Google Drive range download size mismatch",
            {"got": partial.stat().st_size, "expected": expected_bytes},
        )
    partial.replace(destination)


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
                split = (
                    split_from_name if split_from_name in {"train", "dev", "test"} else lowered.get("split") or "train"
                )
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


def _flac_duration_seconds(path: Path) -> float:
    count = _flac_sample_count(path)
    return count / 16000.0


def _icsi_intervals_from_zip(archive: Path) -> list[RttmInterval]:
    from xml.etree import ElementTree

    intervals: list[RttmInterval] = []
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if "/Segments/" not in name or not name.endswith(".segs.xml"):
                continue
            stem = Path(name).name
            meeting_id = stem.split(".")[0]
            root = ElementTree.fromstring(zipped.read(name))
            for node in root.iter():
                start = node.attrib.get("starttime")
                end = node.attrib.get("endtime")
                speaker = node.attrib.get("participant")
                if not start or not end or not speaker:
                    continue
                try:
                    start_value = float(start)
                    end_value = float(end)
                except ValueError:
                    continue
                if end_value <= start_value:
                    continue
                intervals.append(
                    RttmInterval(
                        recording_id=meeting_id,
                        start=start_value,
                        end=end_value,
                        speaker=speaker,
                    )
                )
    unique: dict[tuple[str, float, float, str], RttmInterval] = {}
    for item in intervals:
        unique[(item.recording_id, item.start, item.end, item.speaker)] = item
    return list(unique.values())


def _notsofar_intervals(split_root: Path) -> tuple[list[RttmInterval], dict[str, float]]:
    intervals: list[RttmInterval] = []
    durations: dict[str, float] = {}
    if not split_root.is_dir():
        return intervals, durations
    for meeting_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        trans_path = meeting_dir / "gt_transcription.json"
        if not trans_path.is_file():
            continue
        turns = json.loads(trans_path.read_text(encoding="utf-8"))
        ends = 0.0
        for turn in turns:
            start = float(turn["start_time"])
            end = float(turn["end_time"])
            ends = max(ends, end)
            intervals.append(
                RttmInterval(
                    recording_id=meeting_dir.name,
                    start=start,
                    end=end,
                    speaker=str(turn.get("speaker_id") or "unk"),
                )
            )
        durations[meeting_dir.name] = ends
    return intervals, durations


def _textgrid_xmax_and_intervals(path: Path, recording_id: str) -> tuple[float, list[RttmInterval]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    xmax_match = re.search(r"xmax\s*=\s*([0-9.]+)", text)
    duration = float(xmax_match.group(1)) if xmax_match else 0.0
    intervals: list[RttmInterval] = []
    for name, body in re.findall(
        r'name\s*=\s*"([^"]+)"(.*?)(?:item \[\d+\]:|\Z)',
        text,
        flags=re.S,
    ):
        lowered = name.lower()
        if lowered in {"utterance", "transcript", "text", "ortho"}:
            continue
        for xmin, xmax, mark in re.findall(
            r"xmin\s*=\s*([0-9.]+)\s*xmax\s*=\s*([0-9.]+)\s*text\s*=\s*\"([^\"]*)\"",
            body,
        ):
            label = mark.strip()
            if not label or label in {"", "<NA>", "sil", "sp", "xxx"}:
                continue
            intervals.append(
                RttmInterval(
                    recording_id=recording_id,
                    start=float(xmin),
                    end=float(xmax),
                    speaker=name if lowered in {"speaker", "speakers"} else label,
                )
            )
    return duration, intervals


def audit_local_core_sources(
    *,
    audio_root: Path,
    source_cache: Path,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Account splits, hours, and capacity for local ungated cores. Does not accept."""

    reports: dict[str, Any] = {}
    published = _load_published_three_corpus()
    root = AMI_ALI_AISHELL_ROOT
    intervals = parse_rttm((root / "train" / "rttm").read_text(encoding="utf-8"))
    intervals.extend(parse_rttm((root / "dev" / "rttm").read_text(encoding="utf-8")))
    durations = load_uem_durations((root / "train" / "all.uem").read_text(encoding="utf-8"))
    durations.update(load_uem_durations((root / "dev" / "all.uem").read_text(encoding="utf-8")))
    for corpus_dir, mapped in (("AMI", "AMI"), ("AliMeeting", "AliMeeting"), ("AISHELL4", "AISHELL4")):
        test_root = root / "test" / corpus_dir
        intervals.extend(parse_rttm((test_root / "rttm").read_text(encoding="utf-8")))
        durations.update(load_uem_durations((test_root / "all.uem").read_text(encoding="utf-8")))
        names = set(published[mapped]["train"] + published[mapped]["dev"] + published[mapped]["test"])
        corpus_intervals = [item for item in intervals if item.recording_id in names]
        corpus_durations = {key: value for key, value in durations.items() if key in names}
        reports[mapped] = audit_selection_hours_capacity(
            source=mapped,
            splits=published[mapped],
            duration_by_recording=corpus_durations,
            intervals=corpus_intervals,
        )

    icsi_audio = audio_root / "ICSI"
    icsi_archive = source_cache / "icsi" / "ICSI_core_NXT.zip"
    frozen_path = (evidence_root or Path()) / "icsi-frozen-test.txt"
    if icsi_audio.is_dir() and icsi_archive.is_file():
        meetings = sorted(path.stem for path in icsi_audio.glob("*.flac"))
        frozen = ()
        if frozen_path.is_file():
            frozen = tuple(
                line.strip() for line in frozen_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        if frozen:
            parts = _icsi_split(meetings, frozen)
            icsi_durations = {meeting: _flac_duration_seconds(icsi_audio / f"{meeting}.flac") for meeting in meetings}
            reports["ICSI"] = audit_selection_hours_capacity(
                source="ICSI",
                splits=parts,
                duration_by_recording=icsi_durations,
                intervals=_icsi_intervals_from_zip(icsi_archive),
            )

    notsofar_cache = source_cache / "notsofar-real"
    if notsofar_cache.is_dir():
        splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
        notsofar_intervals: list[RttmInterval] = []
        notsofar_durations: dict[str, float] = {}
        for split in ("train", "dev", "test"):
            found, found_durations = _notsofar_intervals(notsofar_cache / split)
            splits[split] = sorted({item.recording_id for item in found})
            notsofar_intervals.extend(found)
            notsofar_durations.update(found_durations)
            audio_split = audio_root / "NOTSOFAR_real"
            for recording_id in splits[split]:
                flac = audio_split / f"{recording_id}.flac"
                if flac.is_file():
                    notsofar_durations[recording_id] = _flac_duration_seconds(flac)
        if splits["train"]:
            reports["NOTSOFAR_real"] = audit_selection_hours_capacity(
                source="NOTSOFAR_real",
                splits=splits,
                duration_by_recording=notsofar_durations,
                intervals=notsofar_intervals,
            )

    lotus_csv = source_cache / "lotusdis" / "extracted" / "annotation"
    lotus_tg = source_cache / "lotusdis" / "extracted" / "textgrid"
    if (lotus_csv / "train.csv").is_file() and lotus_tg.is_dir():
        parts = _lotusdis_reconcile(_parse_lotusdis_csv(lotus_csv))
        lotus_intervals: list[RttmInterval] = []
        lotus_durations: dict[str, float] = {}
        by_parent: dict[str, list[Path]] = defaultdict(list)
        for path in lotus_tg.glob("*.TextGrid"):
            parent = _lotusdis_parent_id(path.name)
            by_parent[parent].append(path)
        for parent, paths in by_parent.items():
            chosen = paths[0]
            duration, found = _textgrid_xmax_and_intervals(chosen, parent)
            lotus_durations[parent] = duration
            lotus_intervals.extend(found)
        reports["LOTUSDIS"] = audit_selection_hours_capacity(
            source="LOTUSDIS",
            splits=parts,
            duration_by_recording=lotus_durations,
            intervals=lotus_intervals,
        )
    return reports
