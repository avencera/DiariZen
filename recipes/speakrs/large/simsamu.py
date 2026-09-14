"""Strict SIMSAMU metadata, RTTM, and canonical-audio contracts."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import soundfile as sf

from .acceptance import RttmInterval
from .errors import PreparationError
from .hashing import sha256_file
from .prepare import scrubbed_environment


SIMSAMU_REVISION = "6b639431f658d636c30dbf342074d723bf303e0f"
SIMSAMU_SAMPLE_RATE = 8_000
SIMSAMU_CANONICAL_SAMPLE_RATE = 16_000
SIMSAMU_PUBLISHED_TURN_COUNT = 2_997

_RECORDING_ID_RE = re.compile(r"^[a-z0-9_]+$")
_PERSON_ID_RE = re.compile(r"^speaker_[1-9][0-9]*$")
_LOCAL_ROLE_RE = re.compile(r"^(medecin|patient(?:_[1-9][0-9]*)?)$")


@dataclass(frozen=True, slots=True)
class SimsamuMetadata:
    """Global people assigned to the local roles in one recording."""

    recording_id: str
    doctor: str
    patients: tuple[str, ...]

    def __post_init__(self) -> None:
        if _RECORDING_ID_RE.fullmatch(self.recording_id) is None:
            raise ValueError("recording_id must be a canonical SIMSAMU identifier")
        people = (self.doctor, *self.patients)
        if not self.patients or any(_PERSON_ID_RE.fullmatch(person) is None for person in people):
            raise ValueError("SIMSAMU people must use canonical speaker identifiers")
        if len(set(people)) != len(people):
            raise ValueError("one SIMSAMU recording cannot assign one person to two roles")

    @property
    def people(self) -> frozenset[str]:
        """Return every recurring person in the recording."""

        return frozenset((self.doctor, *self.patients))

    def person_for_role(self, role: str) -> str:
        """Map one publisher-local role to its recurring person."""

        if role == "medecin":
            return self.doctor
        if len(self.patients) == 1 and role == "patient":
            return self.patients[0]
        if len(self.patients) > 1 and role.startswith("patient_"):
            index_text = role.removeprefix("patient_")
            if index_text.isdigit():
                index = int(index_text) - 1
                if 0 <= index < len(self.patients):
                    return self.patients[index]

        raise PreparationError(
            "SIMSAMU RTTM role does not match metadata",
            {"recording_id": self.recording_id, "role": role},
        )


@dataclass(frozen=True, slots=True)
class SimsamuAnnotation:
    """Validated and globally mapped activity for one recording."""

    recording_id: str
    intervals: tuple[RttmInterval, ...]
    local_roles: frozenset[str]
    people: frozenset[str]
    overlap_seconds: float


@dataclass(frozen=True, slots=True)
class SimsamuCanonicalAudio:
    """Content identity and decoded clock for one canonical recording."""

    recording_id: str
    source_path: Path
    source_sha256: str
    path: Path
    sha256: str
    sample_count: int
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class SimsamuSourceAudio:
    """Publisher container facts measured at the media boundary."""

    path: Path
    sha256: str
    sample_rate: int
    channels: int
    duration: float


def parse_simsamu_metadata(path: Path) -> tuple[SimsamuMetadata, ...]:
    """Parse the complete publisher person map without inferred identities."""

    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["rec_name", "medecin", "patient"]:
                raise PreparationError("SIMSAMU metadata has an unexpected header")
            rows = []
            for row in reader:
                patients = tuple(row["patient"].split("-"))
                rows.append(SimsamuMetadata(row["rec_name"], row["medecin"], patients))
    except (OSError, UnicodeError, csv.Error, KeyError, ValueError) as error:
        raise PreparationError("SIMSAMU metadata is invalid") from error

    recording_ids = tuple(row.recording_id for row in rows)
    if len(recording_ids) != 61 or len(set(recording_ids)) != 61:
        raise PreparationError("SIMSAMU metadata must contain 61 unique recordings")
    people = frozenset(person for row in rows for person in row.people)
    if people != frozenset(f"speaker_{index}" for index in range(1, 15)):
        raise PreparationError("SIMSAMU metadata must contain the exact 14-person set")
    if sum(len(row.patients) == 2 for row in rows) != 1:
        raise PreparationError("SIMSAMU metadata must contain one three-person recording")

    return tuple(rows)


def parse_simsamu_rttm(path: Path, metadata: SimsamuMetadata, duration: float) -> SimsamuAnnotation:
    """Parse, map, and validate one complete publisher RTTM."""

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    intervals = []
    local_roles = set()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise PreparationError("SIMSAMU RTTM is unreadable", {"recording_id": metadata.recording_id}) from error
    if not lines:
        raise PreparationError("SIMSAMU RTTM is empty", {"recording_id": metadata.recording_id})

    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER" or fields[1:3] != ["<NA>", "<NA>"]:
            raise PreparationError(
                "SIMSAMU RTTM row has an unexpected form",
                {"recording_id": metadata.recording_id, "line": line_number},
            )
        role = fields[7]
        if _LOCAL_ROLE_RE.fullmatch(role) is None:
            raise PreparationError(
                "SIMSAMU RTTM has an invalid local role",
                {"recording_id": metadata.recording_id, "line": line_number},
            )
        try:
            start = float(fields[3])
            segment_duration = float(fields[4])
        except ValueError as error:
            raise PreparationError(
                "SIMSAMU RTTM has an invalid interval",
                {"recording_id": metadata.recording_id, "line": line_number},
            ) from error
        end = start + segment_duration
        if (
            not math.isfinite(start)
            or not math.isfinite(segment_duration)
            or start < 0
            or segment_duration <= 0
            or end > duration + 1e-9
        ):
            raise PreparationError(
                "SIMSAMU RTTM interval is outside the decoded audio clock",
                {"recording_id": metadata.recording_id, "line": line_number},
            )
        local_roles.add(role)
        intervals.append(RttmInterval(metadata.recording_id, start, end, metadata.person_for_role(role)))

    expected_roles = {"medecin"}
    expected_roles.add("patient" if len(metadata.patients) == 1 else "patient_1")
    if len(metadata.patients) == 2:
        expected_roles.add("patient_2")
    if local_roles != expected_roles:
        raise PreparationError(
            "SIMSAMU RTTM roles do not cover the metadata roles",
            {"recording_id": metadata.recording_id, "roles": sorted(local_roles)},
        )

    overlap_seconds = _overlap_seconds(intervals)
    if overlap_seconds > 1e-9:
        raise PreparationError(
            "SIMSAMU publisher labels contain overlap",
            {"recording_id": metadata.recording_id, "overlap_seconds": overlap_seconds},
        )

    return SimsamuAnnotation(
        recording_id=metadata.recording_id,
        intervals=tuple(intervals),
        local_roles=frozenset(local_roles),
        people=metadata.people,
        overlap_seconds=overlap_seconds,
    )


def _overlap_seconds(intervals: Iterable[RttmInterval]) -> float:
    events = []
    for interval in intervals:
        events.append((interval.start, 1))
        events.append((interval.end, -1))
    active = 0
    previous = 0.0
    overlap = 0.0
    for time, delta in sorted(events, key=lambda item: (item[0], item[1])):
        if active > 1:
            overlap += time - previous
        active += delta
        previous = time

    return overlap


def probe_simsamu_source(path: Path) -> SimsamuSourceAudio:
    """Read and validate the single audio stream in one publisher M4A object."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise PreparationError("SIMSAMU source inspection requires ffprobe")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=scrubbed_environment(),
    )
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        stream = streams[0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration = float(stream["duration"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise PreparationError("SIMSAMU source audio probe is invalid", {"path": str(path)}) from error
    if completed.returncode != 0 or len(streams) != 1:
        raise PreparationError("SIMSAMU source audio must contain one audio stream", {"path": str(path)})
    if sample_rate != SIMSAMU_SAMPLE_RATE or channels != 1 or not math.isfinite(duration) or duration <= 0:
        raise PreparationError("SIMSAMU source audio has the wrong format", {"path": str(path)})

    return SimsamuSourceAudio(path, sha256_file(path), sample_rate, channels, duration)


def prepare_simsamu_audio(source: Path, destination: Path, recording_id: str) -> SimsamuCanonicalAudio:
    """Decode one 8 kHz mono publisher object to deterministic 16 kHz mono FLAC."""

    if not source.is_file():
        raise PreparationError("SIMSAMU source audio is missing", {"recording_id": recording_id})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise PreparationError("stale SIMSAMU canonical audio partial exists", {"recording_id": recording_id})

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PreparationError("SIMSAMU conversion requires ffmpeg")
    if not destination.is_file():
        command = [
            ffmpeg,
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
            "aresample=resampler=swr:filter_type=kaiser:filter_size=32:phase_shift=10:linear_interp=0:exact_rational=1:cutoff=0.97:kaiser_beta=9:dither_method=none:osr=16000",
            "-ar",
            str(SIMSAMU_CANONICAL_SAMPLE_RATE),
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
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
            str(temporary),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=scrubbed_environment(),
        )
        if completed.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise PreparationError(
                "SIMSAMU ffmpeg conversion failed",
                {"recording_id": recording_id, "stderr": completed.stderr[-500:]},
            )
        temporary.replace(destination)

    try:
        info = sf.info(str(destination))
    except (OSError, RuntimeError) as error:
        raise PreparationError("SIMSAMU canonical audio is unreadable", {"recording_id": recording_id}) from error
    if info.samplerate != SIMSAMU_CANONICAL_SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
        raise PreparationError("SIMSAMU canonical audio has the wrong format", {"recording_id": recording_id})

    return SimsamuCanonicalAudio(
        recording_id=recording_id,
        source_path=source,
        source_sha256=sha256_file(source),
        path=destination,
        sha256=sha256_file(destination),
        sample_count=info.frames,
        sample_rate=info.samplerate,
        channels=info.channels,
    )
