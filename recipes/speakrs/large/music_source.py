"""Prepare the instrumental portion of the official MUSAN music source.

The source archive is deliberately consumed as one compressed stream.  Audio
members are normalized into bounded local candidates while the stream runs;
the candidates are not admitted to the release until the complete compressed
stream has reached EOF and its publisher checksum matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, TextIO

import numpy as np
import soundfile as sf

from .errors import PreparationError
from .hashing import sha256_bytes, sha256_file, sha256_json
from .jsonio import atomic_write_text


SCHEMA = "speakrs-instrumental-music-v1"
SCHEMA_VERSION = 1
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_CHECKSUM_URL = "https://www.openslr.org/resources/17/checksum.md5"
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
GIB = 1024**3
MAX_STAGING_BYTES = 8 * GIB
DEFAULT_STAGING_BYTES = 7 * GIB
DEFAULT_FREE_RESERVE_BYTES = 100 * GIB
DEFAULT_BLOCK_BYTES = 1024 * 1024
DEFAULT_MAX_MEMBER_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_METADATA_BYTES = 16 * 1024 * 1024
DEFAULT_HTTP_TIMEOUT_SECONDS = 60
READY_FILE = ".complete.json"
STATE_FILE = ".state.json"
CONFIG_FILE = ".config.json"
METADATA_NAMES = frozenset({"ANNOTATIONS", "LICENSE", "README"})
TRACK_ID_RE = re.compile(r"^music-[A-Za-z0-9][A-Za-z0-9_-]*$")
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"^\s*={3,}\s*$", re.MULTILINE)
LICENSE_BLOCKED_RE = re.compile(r"\b(?:NC|ND|NON[- ]?COMMERCIAL|NO[- ]?DERIV(?:S|ATIVES)?)\b", re.IGNORECASE)
LICENSE_CC_RE = re.compile(r"\bCC\s*BY(?:\s*[- ]\s*(SA|NC|ND))?(?:\s+(\d+(?:\.\d+)?))?\b", re.IGNORECASE)
LICENSE_CC0_RE = re.compile(r"\bCC\s*0(?:\s+(\d+(?:\.\d+)?))?\b", re.IGNORECASE)
LICENSE_PD_RE = re.compile(r"\bPUBLIC\s+DOMAIN(?:\s+MARK)?(?:\s+(\d+(?:\.\d+)?))?\b", re.IGNORECASE)


def _fail(message: str, details: Mapping[str, Any] | None = None) -> PreparationError:
    """Build one consistent preparation failure."""

    return PreparationError(message, dict(details or {}))


def _safe_component(value: str, label: str) -> str:
    """Validate one path component that this recipe owns."""

    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise _fail(f"{label} must be a non-empty path component")
    if "\\" in value or "/" in value or "\x00" in value:
        raise _fail(f"{label} contains an unsafe path component", {"value": value})
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise _fail(f"{label} contains unsupported characters", {"value": value})
    return value


def safe_archive_path(name: str) -> PurePosixPath:
    """Return a safe POSIX archive path or reject traversal and odd roots."""

    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise _fail("MUSAN archive member path is unsafe", {"path": repr(name)})
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise _fail("MUSAN archive member path contains traversal", {"path": name})
    # Tar directory names conventionally have a trailing slash, which
    # PurePosixPath removes.  Empty components elsewhere are not accepted.
    if any(not part for part in path.parts):
        raise _fail("MUSAN archive member path contains an empty component", {"path": name})
    if name.endswith("/"):
        return path
    if path.as_posix() != name:
        raise _fail("MUSAN archive member path is not canonical", {"path": name})
    return path


def validate_archive_member(member: tarfile.TarInfo) -> PurePosixPath:
    """Validate a tar member before any member content is read."""

    path = safe_archive_path(member.name)
    if member.issym() or member.islnk():
        raise _fail("MUSAN archive contains a link member", {"path": member.name})
    if not member.isfile() and not member.isdir():
        raise _fail("MUSAN archive contains an unsupported member type", {"path": member.name})
    if member.size < 0:
        raise _fail("MUSAN archive member has a negative size", {"path": member.name})
    return path


def _existing_root(path: Path) -> Path:
    """Find an existing ancestor for a filesystem free-space probe."""

    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _free_bytes(path: Path) -> int:
    """Return currently available bytes without following a data path."""

    try:
        usage = os.statvfs(_existing_root(path))
    except OSError as error:
        raise _fail("cannot probe free space", {"path": str(path), "error": str(error)}) from error
    return int(usage.f_frsize * usage.f_bavail)


def _directory_bytes(root: Path) -> int:
    """Count owned regular-file bytes without following symlinks."""

    if not root.exists():
        return 0
    if root.is_symlink() or not root.is_dir():
        raise _fail("music staging root is not a regular directory", {"path": str(root)})
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise _fail("music staging contains a symlink", {"path": str(path)})
        if path.is_file():
            total += path.stat().st_size
    return total


def _fsync_directory(path: Path) -> None:
    """Synchronize one directory when the platform supports directory fsync."""

    try:
        descriptor = os.open(path.as_posix(), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Write bytes atomically and synchronize the containing directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise _fail("owned partial path is not a regular file", {"path": str(temporary)})
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, value: Any) -> None:
    """Write deterministic pretty JSON through an atomic replacement."""

    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise _fail("owned JSON path is not a regular file", {"path": str(path)})
        if path.read_text(encoding="utf-8") != payload:
            raise _fail("existing owned JSON conflicts with the current preparation", {"path": str(path)})
        return
    atomic_write_text(path, payload)


def _replace_json(path: Path, value: Any) -> None:
    """Atomically replace one mutable runtime-state document."""

    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise _fail("owned state path is not a regular file", {"path": str(path)})
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    atomic_write_text(path, payload)


class _Log:
    """Small dual stdout/file progress logger."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.handle: TextIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a", encoding="utf-8")

    def close(self) -> None:
        """Close the optional runtime log."""

        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def write(self, event: str, **fields: Any) -> None:
        """Write one compact JSON event to stdout and the inspectable log."""

        payload = {"event": event, "time": time.time(), **fields}
        line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        print(line, flush=True)
        if self.handle is not None:
            self.handle.write(line + "\n")
            self.handle.flush()


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    """Finite, validated boundaries for one MUSAN preparation run."""

    output: Path
    source_url: str = MUSAN_URL
    checksum_url: str = MUSAN_CHECKSUM_URL
    max_staging_bytes: int = DEFAULT_STAGING_BYTES
    minimum_free_bytes: int = DEFAULT_FREE_RESERVE_BYTES
    block_bytes: int = DEFAULT_BLOCK_BYTES
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES
    max_metadata_bytes: int = DEFAULT_MAX_METADATA_BYTES
    http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS
    split_seed: str = "musan-instrumental-music-v1"
    log_path: Path | None = None

    def __post_init__(self) -> None:
        """Reject unbounded or unsafe preparation configuration."""

        output = Path(self.output).expanduser()
        if not output.is_absolute():
            raise _fail("music output must be an absolute path")
        if output.name in {"", ".", ".."} or output.name.startswith("."):
            raise _fail("music output must use a visible directory name", {"output": str(output)})
        if output.exists() and output.is_symlink():
            raise _fail("music output must not be a symlink", {"output": str(output)})
        if not isinstance(self.source_url, str) or not self.source_url.startswith("https://"):
            raise _fail("MUSAN source URL must use HTTPS")
        if not isinstance(self.checksum_url, str) or not self.checksum_url.startswith("https://"):
            raise _fail("MUSAN checksum URL must use HTTPS")
        if not isinstance(self.split_seed, str) or not self.split_seed:
            raise _fail("split seed must be non-empty")
        bounded = (
            ("max_staging_bytes", self.max_staging_bytes, 1, MAX_STAGING_BYTES),
            ("minimum_free_bytes", self.minimum_free_bytes, 0, 10 * 1024**4),
            ("block_bytes", self.block_bytes, 4 * 1024, 16 * 1024 * 1024),
            ("max_member_bytes", self.max_member_bytes, 1, 2 * GIB),
            ("max_metadata_bytes", self.max_metadata_bytes, 1, 256 * 1024 * 1024),
            ("http_timeout_seconds", self.http_timeout_seconds, 1, 3600),
        )
        for label, value, minimum, maximum in bounded:
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise _fail(f"{label} is outside its finite bound", {"value": value})

    @property
    def output_root(self) -> Path:
        """Return the normalized release output root."""

        return Path(self.output).expanduser().resolve(strict=False)

    @property
    def staging_root(self) -> Path:
        """Return the incomplete staging root kept distinct from ready data."""

        return self.output_root / ".staging"

    def identity(self, expected_md5: str) -> dict[str, Any]:
        """Return the source and transformation identity used for resume."""

        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source_url": self.source_url,
            "checksum_url": self.checksum_url,
            "expected_archive_md5": expected_md5,
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": TARGET_CHANNELS,
            "pcm_encoding": "s16le",
            "split_seed": self.split_seed,
        }


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    """Validated publisher annotation for one music track."""

    source: str
    track_id: str
    genres: tuple[str, ...]
    vocals: str
    artist: str
    annotation_member: str

    @property
    def artist_group(self) -> str:
        """Return a deterministic artist identity with the known source typo fixed."""

        artist = self.artist.strip()
        artist = {"Kevin_MacLoad": "Kevin_MacLeod"}.get(artist, artist)
        return " ".join(artist.replace("_", " ").split()).casefold()


@dataclass(frozen=True, slots=True)
class LicenseRecord:
    """One explicit publisher licence block bound to one track."""

    name: str
    url: str | None
    attribution: str
    publisher_member: str


@dataclass(frozen=True, slots=True)
class PcmFacts:
    """Decoded integer PCM identity for one canonical audio file."""

    sample_count: int
    sample_rate: int
    channels: int
    pcm_sha256: str
    rms: float
    peak_abs: int


@dataclass(frozen=True, slots=True)
class StagedTrack:
    """One validated normalized candidate and its source member identity."""

    source: str
    track_id: str
    source_member: str
    source_sha256: str
    source_bytes: int
    audio_path: Path
    sha256: str
    pcm_sha256: str
    sample_count: int
    sample_rate: int
    channels: int
    rms: float
    peak_abs: int

    def as_sidecar(self) -> dict[str, Any]:
        """Return the sidecar used to verify resumable staged candidates."""

        return {
            "schema": SCHEMA,
            "source": self.source,
            "track_id": self.track_id,
            "source_member": self.source_member,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "sha256": self.sha256,
            "pcm_sha256": self.pcm_sha256,
            "sample_count": self.sample_count,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "rms": self.rms,
            "peak_abs": self.peak_abs,
        }


def _parse_md5(text: str) -> str:
    """Extract the publisher MD5 for the named MUSAN archive."""

    for line in text.splitlines():
        match = re.match(r"^\s*([0-9a-fA-F]{32})\s+(?:\*?)([^\s]+)\s*$", line)
        if match and PurePosixPath(match.group(2)).name == "musan.tar.gz":
            return match.group(1).lower()
    raise _fail("publisher checksum file has no musan.tar.gz entry")


def fetch_publisher_checksum(config: PreparationConfig) -> tuple[str, bytes]:
    """Fetch the small publisher checksum file with a strict byte bound."""

    request = urllib.request.Request(config.checksum_url, headers={"Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(request, timeout=config.http_timeout_seconds) as response:
            payload = response.read(config.max_metadata_bytes + 1)
    except OSError as error:
        raise _fail("publisher checksum download failed", {"url": config.checksum_url, "error": str(error)}) from error
    if len(payload) > config.max_metadata_bytes:
        raise _fail("publisher checksum file exceeds the metadata bound")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _fail("publisher checksum file is not UTF-8") from error
    return _parse_md5(text), payload


class _HashingReader:
    """Hash every compressed source byte returned by an upstream stream."""

    def __init__(self, stream: BinaryIO, *, block_bytes: int, log: _Log, expected_bytes: int | None = None) -> None:
        self.stream = stream
        self.block_bytes = block_bytes
        self.log = log
        self.expected_bytes = expected_bytes
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()
        self.bytes_read = 0
        self.eof = False
        self._last_report = 0
        self._started = time.monotonic()

    def read(self, size: int = -1) -> bytes:
        """Read and hash one compressed block."""

        if size is None or size < 0:
            size = self.block_bytes
        chunk = self.stream.read(size)
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise _fail("source stream returned a non-byte block")
        payload = bytes(chunk)
        if payload:
            self.sha256.update(payload)
            self.md5.update(payload)
            self.bytes_read += len(payload)
            if self.bytes_read - self._last_report >= 256 * 1024 * 1024:
                self._report()
        else:
            self.eof = True
            self._report()
        return payload

    def drain(self) -> None:
        """Consume all remaining compressed bytes after tar EOF."""

        while self.read(self.block_bytes):
            pass

    def _report(self) -> None:
        """Emit bounded source byte progress."""

        if self.bytes_read == self._last_report and not self.eof:
            return
        self._last_report = self.bytes_read
        elapsed = max(time.monotonic() - self._started, 1e-6)
        self.log.write(
            "source_progress",
            bytes=self.bytes_read,
            expected_bytes=self.expected_bytes,
            mebibytes_per_second=round(self.bytes_read / elapsed / 1024**2, 2),
            eof=self.eof,
        )


def _read_member(stream: BinaryIO, expected_size: int, block_bytes: int) -> tuple[bytes, str]:
    """Read one bounded metadata member and return bytes plus SHA-256."""

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = stream.read(block_bytes)
        if not chunk:
            break
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise _fail("archive member returned a non-byte block")
        payload = bytes(chunk)
        received += len(payload)
        if received > expected_size:
            raise _fail("archive member exceeded its tar header size", {"expected": expected_size})
        digest.update(payload)
        chunks.append(payload)
    if received != expected_size:
        raise _fail("archive member ended before its tar header size", {"expected": expected_size, "actual": received})
    return b"".join(chunks), digest.hexdigest()


def _copy_member_to_file(stream: BinaryIO, destination: Path, expected_size: int, block_bytes: int) -> str:
    """Copy one audio member to a private partial WAV while hashing it."""

    digest = hashlib.sha256()
    received = 0
    with destination.open("wb") as handle:
        while True:
            chunk = stream.read(block_bytes)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise _fail("archive member returned a non-byte block")
            payload = bytes(chunk)
            received += len(payload)
            if received > expected_size:
                raise _fail("archive audio member exceeded its tar header size", {"expected": expected_size})
            digest.update(payload)
            handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if received != expected_size:
        raise _fail(
            "archive audio member ended before its tar header size", {"expected": expected_size, "actual": received}
        )
    return digest.hexdigest()


def _inspect_and_encode(source: Path, destination: Path, block_frames: int = 64 * 1024) -> tuple[PcmFacts, PcmFacts]:
    """Validate a 16 kHz mono PCM-16 WAV and encode identical PCM as FLAC."""

    try:
        source_info = sf.info(source.as_posix())
    except (OSError, RuntimeError, TypeError) as error:
        raise _fail("source WAV cannot be decoded", {"path": str(source), "error": str(error)}) from error
    if (
        source_info.format.upper() != "WAV"
        or int(source_info.samplerate) != TARGET_SAMPLE_RATE
        or int(source_info.channels) != TARGET_CHANNELS
        or int(source_info.frames) <= 0
        or str(source_info.subtype).upper() != "PCM_16"
    ):
        raise _fail(
            "source audio does not meet the 16 kHz mono PCM-16 WAV contract",
            {
                "path": str(source),
                "format": source_info.format,
                "sample_rate": source_info.samplerate,
                "channels": source_info.channels,
                "frames": source_info.frames,
                "subtype": source_info.subtype,
            },
        )

    source_digest = hashlib.sha256()
    sample_count = 0
    sum_squares = 0.0
    peak_abs = 0
    try:
        with (
            sf.SoundFile(source.as_posix(), mode="r") as source_file,
            sf.SoundFile(
                destination.as_posix(),
                mode="w",
                samplerate=TARGET_SAMPLE_RATE,
                channels=TARGET_CHANNELS,
                format="FLAC",
                subtype="PCM_16",
            ) as output_file,
        ):
            while True:
                samples = source_file.read(block_frames, dtype="int16", always_2d=True)
                if samples.size == 0:
                    break
                if samples.shape[1] != TARGET_CHANNELS or not np.isfinite(samples.astype(np.float64)).all():
                    raise _fail("source audio contains invalid decoded samples", {"path": str(source)})
                source_digest.update(np.asarray(samples, dtype="<i2").tobytes(order="C"))
                values = samples.astype(np.float64, copy=False)
                sample_count += int(samples.shape[0])
                sum_squares += float(np.square(values).sum())
                peak_abs = max(peak_abs, int(np.abs(samples).max()))
                output_file.write(samples)
    except PreparationError:
        raise
    except (OSError, RuntimeError, TypeError) as error:
        raise _fail("source audio failed during PCM decoding", {"path": str(source), "error": str(error)}) from error

    if sample_count != int(source_info.frames):
        raise _fail(
            "source WAV decoded frame count differs from its header",
            {"expected": source_info.frames, "actual": sample_count},
        )
    source_facts = PcmFacts(
        sample_count=sample_count,
        sample_rate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
        pcm_sha256=source_digest.hexdigest(),
        rms=math.sqrt(sum_squares / sample_count) / 32768.0,
        peak_abs=peak_abs,
    )
    output_facts = _inspect_flac(destination, block_frames)
    if output_facts != source_facts:
        raise _fail(
            "canonical FLAC does not preserve source PCM identity",
            {"source_pcm_sha256": source_facts.pcm_sha256, "output_pcm_sha256": output_facts.pcm_sha256},
        )
    return source_facts, output_facts


def _inspect_flac(path: Path, block_frames: int = 64 * 1024) -> PcmFacts:
    """Decode one FLAC and compute its exact PCM identity."""

    try:
        info = sf.info(path.as_posix())
    except (OSError, RuntimeError, TypeError) as error:
        raise _fail("canonical FLAC cannot be decoded", {"path": str(path), "error": str(error)}) from error
    if (
        info.format.upper() != "FLAC"
        or int(info.samplerate) != TARGET_SAMPLE_RATE
        or int(info.channels) != TARGET_CHANNELS
        or int(info.frames) <= 0
        or str(info.subtype).upper() != "PCM_16"
    ):
        raise _fail(
            "canonical FLAC has invalid format facts",
            {
                "path": str(path),
                "format": info.format,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "subtype": info.subtype,
            },
        )
    digest = hashlib.sha256()
    count = 0
    sum_squares = 0.0
    peak_abs = 0
    try:
        with sf.SoundFile(path.as_posix(), mode="r") as audio:
            while True:
                samples = audio.read(block_frames, dtype="int16", always_2d=True)
                if samples.size == 0:
                    break
                if not np.isfinite(samples.astype(np.float64)).all():
                    raise _fail("canonical FLAC contains non-finite samples", {"path": str(path)})
                digest.update(np.asarray(samples, dtype="<i2").tobytes(order="C"))
                values = samples.astype(np.float64, copy=False)
                count += int(samples.shape[0])
                sum_squares += float(np.square(values).sum())
                peak_abs = max(peak_abs, int(np.abs(samples).max()))
    except PreparationError:
        raise
    except (OSError, RuntimeError, TypeError) as error:
        raise _fail("canonical FLAC failed during decode", {"path": str(path), "error": str(error)}) from error
    if count != int(info.frames):
        raise _fail(
            "canonical FLAC decoded frame count differs from its header",
            {"path": str(path), "expected": info.frames, "actual": count},
        )
    if sum_squares <= 0.0:
        raise _fail("music track has zero decoded energy", {"path": str(path)})
    return PcmFacts(
        sample_count=count,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        pcm_sha256=digest.hexdigest(),
        rms=math.sqrt(sum_squares / count) / 32768.0,
        peak_abs=peak_abs,
    )


def _stage_sidecar_path(audio_path: Path) -> Path:
    """Return the sidecar path for one staged candidate."""

    return audio_path.with_suffix(".json")


def _load_staged(path: Path, expected: Mapping[str, Any], *, sidecar_path: Path | None = None) -> StagedTrack:
    """Verify a staged candidate and return its typed facts."""

    sidecar_path = sidecar_path or _stage_sidecar_path(path)
    if path.is_symlink() or sidecar_path.is_symlink() or not path.is_file() or not sidecar_path.is_file():
        raise _fail("staged candidate is incomplete or unsafe", {"track_id": expected.get("track_id")})
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail("staged candidate sidecar is unreadable", {"path": str(sidecar_path)}) from error
    if not isinstance(sidecar, dict) or sidecar.get("schema") != SCHEMA:
        raise _fail("staged candidate sidecar has an invalid schema", {"path": str(sidecar_path)})
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise _fail(
                "staged candidate conflicts with the current source member",
                {"key": key, "track_id": expected.get("track_id")},
            )
    pcm = _inspect_flac(path)
    output_digest = sha256_file(path)
    if sidecar.get("sha256") != output_digest or sidecar.get("pcm_sha256") != pcm.pcm_sha256:
        raise _fail("staged candidate hash or PCM identity is invalid", {"path": str(path)})
    if int(sidecar.get("sample_count", -1)) != pcm.sample_count:
        raise _fail("staged candidate sample count is invalid", {"path": str(path)})
    return StagedTrack(
        source=str(sidecar["source"]),
        track_id=str(sidecar["track_id"]),
        source_member=str(sidecar["source_member"]),
        source_sha256=str(sidecar["source_sha256"]),
        source_bytes=int(sidecar["source_bytes"]),
        audio_path=path,
        sha256=output_digest,
        pcm_sha256=pcm.pcm_sha256,
        sample_count=pcm.sample_count,
        sample_rate=pcm.sample_rate,
        channels=pcm.channels,
        rms=pcm.rms,
        peak_abs=pcm.peak_abs,
    )


def _ensure_capacity(config: PreparationConfig, required_bytes: int, *, label: str) -> None:
    """Stop before a write can cross the staging cap or free-space reserve."""

    if required_bytes < 0:
        raise _fail("required write size cannot be negative", {"label": label})
    used = _directory_bytes(config.output_root)
    if used + required_bytes > config.max_staging_bytes:
        raise _fail(
            "music task-data cap exhausted",
            {
                "label": label,
                "used_bytes": used,
                "required_bytes": required_bytes,
                "cap_bytes": config.max_staging_bytes,
            },
        )
    free = _free_bytes(config.output_root)
    if free < config.minimum_free_bytes + required_bytes:
        raise _fail(
            "free-space reserve would be crossed before music write",
            {
                "label": label,
                "free_bytes": free,
                "reserve_bytes": config.minimum_free_bytes,
                "required_bytes": required_bytes,
            },
        )


def _candidate_path(config: PreparationConfig, track_id: str) -> Path:
    """Return a safe candidate path for one validated track ID."""

    _safe_component(track_id, "track ID")
    return config.staging_root / "candidates" / f"{track_id}.flac"


def _stage_audio_member(
    config: PreparationConfig,
    source: str,
    track_id: str,
    member: tarfile.TarInfo,
    member_stream: BinaryIO,
    log: _Log,
) -> StagedTrack | None:
    """Hash, validate, normalize, and atomically stage one audio member."""

    if member.size > config.max_member_bytes:
        raise _fail("audio member exceeds the per-member bound", {"path": member.name, "bytes": member.size})
    candidate = _candidate_path(config, track_id)
    sidecar = _stage_sidecar_path(candidate)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate.exists() or sidecar.exists():
        if candidate.is_symlink() or sidecar.is_symlink():
            raise _fail("staged candidate path is a symlink", {"track_id": track_id})
        # Existing candidates are resumable only when the incoming member
        # hash proves the exact same source bytes.
        incoming_digest = hashlib.sha256()
        received = 0
        while True:
            chunk = member_stream.read(config.block_bytes)
            if not chunk:
                break
            payload = bytes(chunk)
            received += len(payload)
            if received > member.size:
                raise _fail("archive audio member exceeded its tar header size", {"path": member.name})
            incoming_digest.update(payload)
        if received != member.size:
            raise _fail("archive audio member ended before its tar header size", {"path": member.name})
        expected = {
            "source": source,
            "track_id": track_id,
            "source_member": member.name,
            "source_sha256": incoming_digest.hexdigest(),
            "source_bytes": member.size,
        }
        if not sidecar.is_file():
            raise _fail("staged candidate sidecar is missing", {"track_id": track_id})
        audio_path = candidate
        if not candidate.is_file():
            published = [
                path
                for split in ("train", "validation")
                if (path := config.output_root / "audio" / split / f"{track_id}.flac").is_file()
            ]
            if len(published) != 1:
                raise _fail("staged candidate audio is missing or ambiguous", {"track_id": track_id})
            audio_path = published[0]
        staged = _load_staged(audio_path, expected, sidecar_path=sidecar)
        log.write("audio_reused", source=source, track_id=track_id, bytes=member.size)
        return staged

    _ensure_capacity(config, member.size * 2 + 128 * 1024, label=f"source audio {track_id}")
    raw = config.staging_root / "raw" / f"{track_id}.wav.partial"
    if raw.exists() or raw.is_symlink():
        if raw.is_symlink() or not raw.is_file():
            raise _fail("owned raw partial path is unsafe", {"path": str(raw)})
        raw.unlink()
    raw.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_digest = _copy_member_to_file(member_stream, raw, member.size, config.block_bytes)
        temporary = candidate.with_name(candidate.name + ".partial")
        if temporary.exists() or temporary.is_symlink():
            if temporary.is_symlink() or not temporary.is_file():
                raise _fail("owned FLAC partial path is unsafe", {"path": str(temporary)})
            temporary.unlink()
        source_facts, _ = _inspect_and_encode(raw, temporary)
        output_digest = sha256_file(temporary)
        temporary.replace(candidate)
        staged = StagedTrack(
            source=source,
            track_id=track_id,
            source_member=member.name,
            source_sha256=source_digest,
            source_bytes=member.size,
            audio_path=candidate,
            sha256=output_digest,
            pcm_sha256=source_facts.pcm_sha256,
            sample_count=source_facts.sample_count,
            sample_rate=source_facts.sample_rate,
            channels=source_facts.channels,
            rms=source_facts.rms,
            peak_abs=source_facts.peak_abs,
        )
        _atomic_json(sidecar, staged.as_sidecar())
        log.write(
            "audio_staged",
            source=source,
            track_id=track_id,
            source_bytes=member.size,
            staging_bytes=_directory_bytes(config.staging_root),
        )
        return staged
    except PreparationError:
        candidate.with_name(candidate.name + ".partial").unlink(missing_ok=True)
        candidate.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, ValueError) as error:
        candidate.with_name(candidate.name + ".partial").unlink(missing_ok=True)
        candidate.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise _fail("audio candidate preparation failed", {"track_id": track_id, "error": str(error)}) from error
    finally:
        raw.unlink(missing_ok=True)


def parse_annotations(text: str, *, source: str, member: str) -> dict[str, TrackMetadata]:
    """Parse strict ``id genres vocals artist`` publisher rows."""

    records: dict[str, TrackMetadata] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split(maxsplit=3)
        if len(fields) != 4 or not TRACK_ID_RE.fullmatch(fields[0]):
            raise _fail(
                "MUSAN ANNOTATIONS row is ambiguous", {"source": source, "member": member, "line": line_number}
            )
        track_id, genres_text, vocals, artist = fields
        if vocals not in {"Y", "N"} or not genres_text or not artist.strip():
            raise _fail(
                "MUSAN ANNOTATIONS row has invalid fields",
                {"source": source, "line": line_number, "track_id": track_id},
            )
        if track_id in records:
            raise _fail("MUSAN ANNOTATIONS contains a duplicate track ID", {"source": source, "track_id": track_id})
        genres = tuple(part for part in genres_text.split(",") if part)
        if not genres:
            raise _fail("MUSAN ANNOTATIONS row has no genre", {"track_id": track_id})
        records[track_id] = TrackMetadata(source, track_id, genres, vocals, artist.strip(), member)
    if not records:
        raise _fail("MUSAN ANNOTATIONS contains no track rows", {"source": source, "member": member})
    return records


def _license_name(line: str) -> tuple[str | None, bool]:
    """Return a normalized accepted licence name and blocked marker."""

    if LICENSE_BLOCKED_RE.search(line):
        return None, True
    match = LICENSE_CC0_RE.search(line)
    if match:
        version = match.group(1)
        return f"CC0 {version}" if version else "CC0", False
    match = LICENSE_PD_RE.search(line)
    if match:
        version = match.group(1)
        return f"Public Domain Mark {version}" if "MARK" in line.upper() and version else "Public Domain", False
    match = LICENSE_CC_RE.search(line)
    if match:
        variant, version = (match.group(1) or "").upper(), match.group(2)
        name = "CC BY-SA" if variant == "SA" else "CC BY"
        return f"{name} {version}" if version else name, False
    upper = line.upper()
    if "ATTRIBUTION" in upper and ("LICENSE" in upper or "LICENCE" in upper):
        version_match = re.search(r"\b\d+(?:\.\d+)?\b", line)
        version = version_match.group(0) if version_match else None
        name = "CC BY-SA" if "SHARE" in upper and "ALIKE" in upper else "CC BY"
        return f"{name} {version}" if version else name, False
    return None, False


@dataclass(frozen=True, slots=True)
class _LicenseSection:
    """Intermediate licence section before track assignment."""

    ids: tuple[str, ...]
    name: str | None
    blocked: bool
    url: str | None
    text: str


def parse_license_records(
    text: str, *, source: str, member: str, track_ids: Iterable[str]
) -> dict[str, LicenseRecord | None]:
    """Bind explicit licence blocks to IDs without inferring ambiguous grants."""

    allowed = set(track_ids)
    sections = SEPARATOR_RE.split(text)
    parsed: list[_LicenseSection] = []
    for section in sections:
        raw = section.strip()
        if not raw:
            continue
        ids = tuple(dict.fromkeys(line.strip() for line in raw.splitlines() if line.strip() in allowed))
        names: set[str] = set()
        blocked = False
        for line in raw.splitlines():
            name, line_blocked = _license_name(line)
            if name is not None:
                names.add(name)
            blocked = blocked or line_blocked
        urls = URL_RE.findall(raw)
        parsed.append(
            _LicenseSection(
                ids,
                next(iter(names)) if len(names) == 1 else None,
                blocked or len(names) > 1,
                urls[0].rstrip(".,;") if urls else None,
                raw,
            )
        )

    result: dict[str, LicenseRecord | None] = {}
    pending: _LicenseSection | None = None
    for index, section in enumerate(parsed):
        if not section.ids:
            if index == 0 and section.name is not None and not section.blocked:
                pending = section
            continue
        assigned = section
        if assigned.name is None and not assigned.blocked and pending is not None:
            assigned = _LicenseSection(
                section.ids, pending.name, pending.blocked, pending.url, f"{pending.text}\n{section.text}"
            )
        for track_id in assigned.ids:
            if track_id in result:
                result[track_id] = None
                continue
            if assigned.name is None or assigned.blocked:
                result[track_id] = None
            else:
                result[track_id] = LicenseRecord(assigned.name, assigned.url, assigned.text, member)
    # IDs omitted from the publisher file remain explicitly unresolved.
    for track_id in allowed:
        result.setdefault(track_id, None)
    return result


def _license_is_usable(record: LicenseRecord | None) -> bool:
    """Allow only grants compatible with the derivative FLAC release."""

    if record is None:
        return False
    name = record.name.upper()
    return name.startswith("CC0") or name.startswith("PUBLIC DOMAIN") or name.startswith("CC BY")


def _manifest_license(record: LicenseRecord) -> dict[str, Any]:
    """Return one explicit per-track licence binding."""

    return {
        "name": record.name,
        "url": record.url,
        "publisher_member": record.publisher_member,
        "attribution": record.attribution,
    }


def _split_tracks(
    tracks: list[tuple[TrackMetadata, LicenseRecord, StagedTrack]], seed: str
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign whole artist/PCM components to an approximately 90/10 split."""

    if not tracks:
        raise _fail("no valid no-vocal music tracks remain after metadata and audio checks")
    parent: dict[str, str] = {}
    groups: dict[str, set[str]] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for metadata, _, staged in tracks:
        artist_key = "artist:" + metadata.artist_group
        pcm_key = "pcm:" + staged.pcm_sha256
        union(artist_key, pcm_key)
    for metadata, _, staged in tracks:
        root = find("artist:" + metadata.artist_group)
        groups.setdefault(root, set()).add(staged.track_id)
    components = sorted(groups.values(), key=lambda values: sha256_text(seed, sorted(values)))
    total = len(tracks)
    target = max(1, round(total * 0.10)) if len(components) > 1 else 0
    states: dict[int, tuple[str, ...]] = {0: ()}
    for component in components:
        size = len(component)
        key = tuple(sorted(component))
        for count, selected in list(states.items()):
            new_count = count + size
            if new_count <= target and new_count not in states:
                states[new_count] = selected + key
    validation_count = min(states, key=lambda count: (abs(count - target), count))
    validation_ids = set(states[validation_count])
    split = {
        track_id: ("validation" if track_id in validation_ids else "train")
        for _, _, staged in tracks
        for track_id in (staged.track_id,)
    }
    validation_artists = {
        metadata.artist_group for metadata, _, staged in tracks if split[staged.track_id] == "validation"
    }
    train_artists = {metadata.artist_group for metadata, _, staged in tracks if split[staged.track_id] == "train"}
    if validation_artists & train_artists:
        raise _fail("artist-disjoint split construction failed")
    identity_payload = {
        "algorithm": "artist-and-pcm-component-sha256-knapsack-v1",
        "seed": seed,
        "validation_fraction": 0.10,
        "assignments": [[track_id, split[track_id]] for track_id in sorted(split)],
        "artist_groups": sorted((metadata.artist_group, split[staged.track_id]) for metadata, _, staged in tracks),
    }
    split_record = {
        **identity_payload,
        "identity_sha256": sha256_json(identity_payload),
        "track_count": total,
        "train_count": sum(value == "train" for value in split.values()),
        "validation_count": sum(value == "validation" for value in split.values()),
        "train_artist_groups": sorted(train_artists),
        "validation_artist_groups": sorted(validation_artists),
    }
    return split, split_record


def sha256_text(seed: str, values: Iterable[str]) -> str:
    """Hash a split seed and sorted component identity."""

    return hashlib.sha256((seed + "\n" + "\n".join(values)).encode("utf-8")).hexdigest()


def _track_record(metadata: TrackMetadata, licence: LicenseRecord, staged: StagedTrack, split: str) -> dict[str, Any]:
    """Build one complete manifest track record."""

    relative_audio = PurePosixPath("audio", split, f"{staged.track_id}.flac").as_posix()
    return {
        "track_id": staged.track_id,
        "path": relative_audio,
        "sha256": staged.sha256,
        "pcm_sha256": staged.pcm_sha256,
        "sample_count": staged.sample_count,
        "sample_rate": staged.sample_rate,
        "channels": staged.channels,
        "split": split,
        "artist": metadata.artist,
        "artist_group": metadata.artist_group,
        "genres": list(metadata.genres),
        "vocals": metadata.vocals,
        "source_member": staged.source_member,
        "source_sha256": staged.source_sha256,
        "source_bytes": staged.source_bytes,
        "rms": staged.rms,
        "peak_abs": staged.peak_abs,
        "license": _manifest_license(licence),
        "annotation_member": metadata.annotation_member,
    }


def _source_metadata_records(config: PreparationConfig) -> list[dict[str, Any]]:
    """Return hashes for retained source metadata files."""

    root = config.staging_root / "metadata"
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def _reconcile_final_audio(destination: Path, staged: StagedTrack) -> None:
    """Move one candidate to its final path or verify a prior partial move."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or sha256_file(destination) != staged.sha256:
            raise _fail(
                "final audio path conflicts with the selected candidate",
                {"path": str(destination), "track_id": staged.track_id},
            )
        facts = _inspect_flac(destination)
        if facts.pcm_sha256 != staged.pcm_sha256 or facts.sample_count != staged.sample_count:
            raise _fail("final audio path has a conflicting PCM identity", {"path": str(destination)})
        return
    if staged.audio_path.is_symlink() or not staged.audio_path.is_file():
        raise _fail("selected staged audio is absent or unsafe", {"track_id": staged.track_id})
    staged.audio_path.replace(destination)


def _write_view(
    path: Path, base: Mapping[str, Any], view: str, tracks: list[dict[str, Any]], full_digest: str
) -> None:
    """Write one train or validation manifest view."""

    payload = dict(base)
    payload["view"] = view
    payload["full_manifest_sha256"] = full_digest
    payload["tracks"] = tracks
    payload["track_count"] = len(tracks)
    _atomic_json(path, payload)


def _verify_split_records(full: Mapping[str, Any]) -> None:
    """Verify the full manifest's split and artist boundaries."""

    tracks = full.get("tracks")
    split = full.get("split")
    if not isinstance(tracks, list) or not isinstance(split, dict):
        raise _fail("music manifest lacks track or split records")
    ids = [row.get("track_id") for row in tracks if isinstance(row, dict)]
    if len(ids) != len(tracks) or len(set(ids)) != len(ids):
        raise _fail("music manifest track IDs are not unique")
    train_ids = {row["track_id"] for row in tracks if row.get("split") == "train"}
    validation_ids = {row["track_id"] for row in tracks if row.get("split") == "validation"}
    if train_ids & validation_ids or train_ids | validation_ids != set(ids):
        raise _fail("music manifest split membership is inconsistent")
    train_artists = {row.get("artist_group") for row in tracks if row.get("split") == "train"}
    validation_artists = {row.get("artist_group") for row in tracks if row.get("split") == "validation"}
    if train_artists & validation_artists:
        raise _fail("music manifest artist groups cross the split boundary")
    if split.get("identity_sha256") != sha256_json(
        {key: split[key] for key in ("algorithm", "seed", "validation_fraction", "assignments", "artist_groups")}
    ):
        raise _fail("music split identity does not match its assignments")


def verify_release(output: Path) -> dict[str, Any]:
    """Verify a published release without network access or mutation."""

    output = Path(output).expanduser().resolve()
    marker_path = output / READY_FILE
    manifest_path = output / "manifest.json"
    if output.is_symlink() or not output.is_dir() or not marker_path.is_file() or not manifest_path.is_file():
        raise _fail("music release is not marked ready", {"output": str(output)})
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        full = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail("published music manifest is unreadable") from error
    if not isinstance(marker, dict) or marker.get("state") != "ready" or marker.get("schema") != SCHEMA:
        raise _fail("published music completion marker is invalid")
    if not isinstance(full, dict) or full.get("schema") != SCHEMA or full.get("state") != "ready":
        raise _fail("published music manifest schema or state is invalid")
    if marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise _fail("published music manifest hash differs from its completion marker")
    _verify_split_records(full)
    tracks = full["tracks"]
    for row in tracks:
        if not isinstance(row, dict):
            raise _fail("music manifest contains a non-object track")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            raise _fail("music manifest contains an unsafe audio path", {"path": relative})
        path = output / PurePosixPath(relative)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise _fail("published music audio hash does not match its manifest", {"track_id": row.get("track_id")})
        facts = _inspect_flac(path)
        for key, value in (
            ("pcm_sha256", facts.pcm_sha256),
            ("sample_count", facts.sample_count),
            ("sample_rate", facts.sample_rate),
            ("channels", facts.channels),
        ):
            if row.get(key) != value:
                raise _fail(
                    "published music audio facts do not match its manifest",
                    {"track_id": row.get("track_id"), "field": key},
                )
    for filename, view in (("training-manifest.json", "train"), ("validation-manifest.json", "validation")):
        path = output / filename
        if not path.is_file():
            raise _fail("published music view is missing", {"path": str(path)})
        view_payload = json.loads(path.read_text(encoding="utf-8"))
        if view_payload.get("view") != view or view_payload.get("full_manifest_sha256") != sha256_file(manifest_path):
            raise _fail("published music view is not bound to the full manifest", {"path": str(path)})
        expected_ids = {row["track_id"] for row in tracks if row.get("split") == view}
        actual_ids = {row.get("track_id") for row in view_payload.get("tracks", [])}
        if expected_ids != actual_ids:
            raise _fail("published music view has inconsistent membership", {"view": view})
    return full


def _build_manifests(
    config: PreparationConfig,
    expected_md5: str,
    checksum_bytes: bytes,
    archive_hash: _HashingReader,
    tracks: list[tuple[TrackMetadata, LicenseRecord, StagedTrack]],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construct the complete manifest after source and selection validation."""

    split, split_record = _split_tracks(tracks, config.split_seed)
    records = [
        _track_record(metadata, licence, staged, split[staged.track_id])
        for metadata, licence, staged in sorted(tracks, key=lambda item: item[2].track_id)
    ]
    collection_sources = sorted({metadata.source for metadata, _, _ in tracks})
    collections = []
    for source in collection_sources:
        source_tracks = [(metadata, licence) for metadata, licence, _ in tracks if metadata.source == source]
        licenses = {
            (licence.name, licence.url, licence.publisher_member)
            for _, licence in source_tracks
        }
        collections.append(
            {
                "source": source,
                "selected_track_count": len(source_tracks),
                "annotation_member": source_tracks[0][0].annotation_member,
                "license_mode": "per_track",
                "licenses": [
                    {"name": name, "url": url, "publisher_member": publisher_member}
                    for name, url, publisher_member in sorted(licenses, key=lambda value: tuple(item or "" for item in value))
                ],
            }
        )
    source_metadata = _source_metadata_records(config)
    base = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "sample_rate": TARGET_SAMPLE_RATE,
        "state": "ready",
        "view": "full",
        "collections": collections,
        "source": {
            "url": config.source_url,
            "checksum_url": config.checksum_url,
            "publisher_checksum": {"algorithm": "md5", "value": expected_md5},
            "archive_bytes": archive_hash.bytes_read,
            "archive_md5": archive_hash.md5.hexdigest(),
            "archive_sha256": archive_hash.sha256.hexdigest(),
            "checksum_file_sha256": sha256_bytes(checksum_bytes),
            "metadata": source_metadata,
        },
        "audio": {
            "format": "FLAC",
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": TARGET_CHANNELS,
            "pcm_encoding": "s16le",
            "lossless": True,
        },
        "split": split_record,
        "split_identity_sha256": split_record["identity_sha256"],
        "track_count": len(records),
        "rejected": sorted(
            rejected,
            key=lambda row: (str(row.get("source", "")), str(row.get("track_id", "")), str(row.get("reason", ""))),
        ),
        "tracks": records,
    }
    return base


def _publish(
    config: PreparationConfig,
    full: dict[str, Any],
    tracks: list[tuple[TrackMetadata, LicenseRecord, StagedTrack]],
) -> dict[str, Any]:
    """Atomically publish selected audio, metadata, manifests, and readiness."""

    output = config.output_root
    output.mkdir(parents=True, exist_ok=True)
    for metadata, _, staged in tracks:
        split = next(row["split"] for row in full["tracks"] if row["track_id"] == staged.track_id)
        _reconcile_final_audio(output / "audio" / split / f"{staged.track_id}.flac", staged)
    metadata_destination = output / "source" / "metadata"
    for record in full["source"]["metadata"]:
        source_path = config.staging_root / "metadata" / PurePosixPath(record["path"])
        destination = metadata_destination / PurePosixPath(record["path"])
        if destination.exists():
            if destination.is_symlink() or not destination.is_file() or sha256_file(destination) != record["sha256"]:
                raise _fail("published source metadata conflicts with the selected source", {"path": str(destination)})
        else:
            _atomic_bytes(destination, source_path.read_bytes())
    # Build the full manifest before either view.  The file itself is the
    # binding identity used by the two split-scoped consumers.
    full_path = output / "manifest.json"
    _atomic_json(full_path, full)
    full_digest = sha256_file(full_path)
    train = [row for row in full["tracks"] if row["split"] == "train"]
    validation = [row for row in full["tracks"] if row["split"] == "validation"]
    _write_view(output / "training-manifest.json", full, "train", train, full_digest)
    _write_view(output / "validation-manifest.json", full, "validation", validation, full_digest)
    attribution_lines = [
        "# MUSAN instrumental source attribution",
        "",
        f"Source archive: {full['source']['url']}",
        f"Archive SHA-256: {full['source']['archive_sha256']}",
        "",
    ]
    for row in full["tracks"]:
        license_record = row["license"]
        attribution_lines.extend(
            [
                f"## {row['track_id']} — {row['artist']}",
                "",
                f"Licence: {license_record['name']}",
                f"Licence URL: {license_record['url'] or 'not stated in publisher file'}",
                "",
                license_record["attribution"],
                "",
            ]
        )
    _atomic_bytes(output / "ATTRIBUTION.md", ("\n".join(attribution_lines).rstrip() + "\n").encode("utf-8"))
    marker = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "state": "ready",
        "manifest_sha256": full_digest,
        "training_manifest_sha256": sha256_file(output / "training-manifest.json"),
        "validation_manifest_sha256": sha256_file(output / "validation-manifest.json"),
        "archive_sha256": full["source"]["archive_sha256"],
        "track_count": full["track_count"],
    }
    _atomic_json(output / READY_FILE, marker)
    return verify_release(output)


def prepare_stream(
    stream: BinaryIO,
    config: PreparationConfig,
    expected_md5: str,
    *,
    expected_bytes: int | None = None,
    checksum_bytes: bytes = b"",
    source_label: str = "stream",
    logger: _Log | None = None,
) -> dict[str, Any]:
    """Prepare from one compressed stream, primarily for tests and local fixtures."""

    expected_md5 = expected_md5.lower()
    if not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
        raise _fail("expected publisher MD5 must be 32 lowercase hexadecimal characters")
    output = config.output_root
    if output.exists() and (output / READY_FILE).is_file():
        return verify_release(output)
    output.mkdir(parents=True, exist_ok=True)
    staging = config.staging_root
    staging.mkdir(parents=True, exist_ok=True)
    identity = config.identity(expected_md5)
    config_path = staging / CONFIG_FILE
    if config_path.exists():
        try:
            previous_identity = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise _fail("music staging configuration is unreadable", {"path": str(config_path)}) from error
        if previous_identity != identity:
            raise _fail("music staging belongs to a different source or transformation")
    else:
        _atomic_json(config_path, identity)
    log = logger or _Log(config.log_path)
    own_logger = logger is None
    _replace_json(staging / STATE_FILE, {**identity, "state": "incomplete", "source_label": source_label})
    archive_hash = _HashingReader(stream, block_bytes=config.block_bytes, log=log, expected_bytes=expected_bytes)
    audio_seen: set[str] = set()
    metadata_seen: set[str] = set()
    staged: dict[str, StagedTrack] = {}
    rejected: list[dict[str, Any]] = []
    metadata_sources: dict[str, dict[str, str]] = {}
    metadata_text: dict[str, dict[str, str]] = {}
    eligible_by_source: dict[str, set[str]] = {}
    started = time.monotonic()
    try:
        with tarfile.open(fileobj=archive_hash, mode="r|gz", bufsize=config.block_bytes) as archive:
            for member in archive:
                path = validate_archive_member(member)
                if not path.parts[:2] == ("musan", "music"):
                    continue
                if len(path.parts) < 4:
                    continue
                source = _safe_component(path.parts[2], "music source")
                leaf = path.name
                if member.isdir():
                    continue
                if leaf in METADATA_NAMES and len(path.parts) == 4:
                    if member.size > config.max_metadata_bytes:
                        raise _fail("source metadata member exceeds the metadata bound", {"path": member.name})
                    if member.name in metadata_seen:
                        raise _fail("archive contains duplicate source metadata", {"path": member.name})
                    metadata_seen.add(member.name)
                    member_stream = archive.extractfile(member)
                    if member_stream is None:
                        raise _fail("cannot read source metadata member", {"path": member.name})
                    with member_stream:
                        payload, digest = _read_member(member_stream, member.size, config.block_bytes)
                    destination = staging / "metadata" / source / leaf
                    if destination.exists():
                        if (
                            destination.is_symlink()
                            or not destination.is_file()
                            or sha256_file(destination) != digest
                            or destination.read_bytes() != payload
                        ):
                            raise _fail("source metadata conflicts with resumable staging", {"path": member.name})
                    else:
                        _ensure_capacity(config, len(payload) + 4096, label=f"metadata {member.name}")
                        _atomic_bytes(destination, payload)
                    metadata_sources.setdefault(source, {})[leaf] = member.name
                    try:
                        metadata_text.setdefault(source, {})[leaf] = payload.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise _fail("source metadata member is not UTF-8", {"path": member.name}) from error
                    source_text = metadata_text[source]
                    if "ANNOTATIONS" in source_text and "LICENSE" in source_text:
                        annotations = parse_annotations(
                            source_text["ANNOTATIONS"], source=source, member=metadata_sources[source]["ANNOTATIONS"]
                        )
                        licenses = parse_license_records(
                            source_text["LICENSE"],
                            source=source,
                            member=metadata_sources[source]["LICENSE"],
                            track_ids=annotations,
                        )
                        eligible_by_source[source] = {
                            track_id
                            for track_id, annotation in annotations.items()
                            if annotation.vocals == "N" and _license_is_usable(licenses.get(track_id))
                        }
                    log.write("metadata_staged", source=source, member=member.name, bytes=len(payload))
                    continue
                if path.suffix.lower() != ".wav" or not TRACK_ID_RE.fullmatch(path.stem):
                    continue
                track_id = path.stem
                if track_id in audio_seen:
                    raise _fail("archive contains duplicate music track IDs", {"track_id": track_id})
                audio_seen.add(track_id)
                if source not in eligible_by_source:
                    raise _fail(
                        "source ANNOTATIONS and LICENSE must precede source audio",
                        {"source": source, "path": member.name},
                    )
                if track_id not in eligible_by_source[source]:
                    continue
                member_stream = archive.extractfile(member)
                if member_stream is None:
                    raise _fail("cannot read source audio member", {"path": member.name})
                with member_stream:
                    try:
                        result = _stage_audio_member(config, source, track_id, member, member_stream, log)
                    except PreparationError as error:
                        rejected.append(
                            {
                                "source": source,
                                "track_id": track_id,
                                "source_member": member.name,
                                "reason": error.message,
                            }
                        )
                        log.write("audio_rejected", source=source, track_id=track_id, reason=error.message)
                        result = None
                if result is not None:
                    staged[track_id] = result
        archive_hash.drain()
        if not archive_hash.eof:
            raise _fail("compressed source did not reach EOF")
        if expected_bytes is not None and archive_hash.bytes_read != expected_bytes:
            raise _fail(
                "compressed source byte count differs from its HTTP declaration",
                {"expected": expected_bytes, "actual": archive_hash.bytes_read},
            )
        actual_md5 = archive_hash.md5.hexdigest()
        if actual_md5 != expected_md5:
            raise _fail(
                "publisher archive checksum mismatch",
                {"expected_md5": expected_md5, "actual_md5": actual_md5, "bytes": archive_hash.bytes_read},
            )
        if not checksum_bytes:
            checksum_bytes = f"{expected_md5}  musan.tar.gz\n".encode("ascii")
        _atomic_bytes(staging / "checksum.md5", checksum_bytes)
        selected: list[tuple[TrackMetadata, LicenseRecord, StagedTrack]] = []
        for source, metadata_map in sorted(metadata_sources.items()):
            if "ANNOTATIONS" not in metadata_map or "LICENSE" not in metadata_map:
                raise _fail("music source lacks required ANNOTATIONS or LICENSE metadata", {"source": source})
            annotation_path = staging / "metadata" / source / "ANNOTATIONS"
            license_path = staging / "metadata" / source / "LICENSE"
            annotations = parse_annotations(
                annotation_path.read_text(encoding="utf-8"), source=source, member=metadata_map["ANNOTATIONS"]
            )
            license_records = parse_license_records(
                license_path.read_text(encoding="utf-8"),
                source=source,
                member=metadata_map["LICENSE"],
                track_ids=annotations,
            )
            source_audio = {track.track_id: track for track in staged.values() if track.source == source}
            for track_id, annotation in sorted(annotations.items()):
                candidate = source_audio.get(track_id)
                if annotation.vocals != "N":
                    continue
                if candidate is None:
                    rejected.append({"source": source, "track_id": track_id, "reason": "no valid source audio member"})
                    continue
                licence = license_records.get(track_id)
                if not _license_is_usable(licence):
                    rejected.append(
                        {
                            "source": source,
                            "track_id": track_id,
                            "reason": "missing, blocked, or ambiguous explicit licence",
                        }
                    )
                    continue
                assert licence is not None
                selected.append((annotation, licence, candidate))
        if not selected:
            raise _fail("no valid no-vocal tracks remain after metadata and audio checks")
        full = _build_manifests(config, expected_md5, checksum_bytes, archive_hash, selected, rejected)
        split_map = {row["track_id"]: row["split"] for row in full["tracks"]}
        _atomic_json(
            staging / "selection.json",
            {"track_ids": sorted(split_map), "split": split_map, "manifest_sha256": sha256_json(full)},
        )
        result = _publish(config, full, selected)
        _replace_json(
            staging / STATE_FILE,
            {
                **identity,
                "state": "ready",
                "track_count": result["track_count"],
                "archive_sha256": result["source"]["archive_sha256"],
            },
        )
        log.write(
            "ready",
            tracks=result["track_count"],
            train_tracks=result["split"]["train_count"],
            validation_tracks=result["split"]["validation_count"],
            archive_bytes=result["source"]["archive_bytes"],
            archive_md5=result["source"]["archive_md5"],
            archive_sha256=result["source"]["archive_sha256"],
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
        return result
    except BaseException as error:
        state = {
            **identity,
            "state": "failed",
            "source_label": source_label,
            "error": str(error),
            "archive_bytes": archive_hash.bytes_read,
        }
        try:
            _replace_json(staging / STATE_FILE, state)
        except BaseException:
            pass
        log.write("failed", error=str(error), archive_bytes=archive_hash.bytes_read)
        raise
    finally:
        if own_logger:
            log.close()


def prepare(config: PreparationConfig) -> dict[str, Any]:
    """Fetch checksums and stream the official MUSAN archive once."""

    output = config.output_root
    if output.exists() and (output / READY_FILE).is_file():
        return verify_release(output)
    expected_md5, checksum_bytes = fetch_publisher_checksum(config)
    request = urllib.request.Request(
        config.source_url, headers={"Accept-Encoding": "identity", "User-Agent": "speakrs-musan-preparer/1"}
    )
    response = None
    try:
        response = urllib.request.urlopen(request, timeout=config.http_timeout_seconds)
        expected_bytes = None
        headers = getattr(response, "headers", None)
        if headers is not None:
            value = headers.get("Content-Length")
            if value is not None:
                try:
                    expected_bytes = int(value)
                except (TypeError, ValueError):
                    raise _fail("source HTTP Content-Length is invalid", {"value": value})
        return prepare_stream(
            response,
            config,
            expected_md5,
            expected_bytes=expected_bytes,
            checksum_bytes=checksum_bytes,
            source_label=config.source_url,
        )
    except OSError as error:
        raise _fail("MUSAN archive download failed", {"url": config.source_url, "error": str(error)}) from error
    finally:
        if response is not None:
            response.close()


def _default_log_path() -> Path:
    """Return the repository-local inspectable runtime log path."""

    return Path.cwd() / "_scratch" / "music-source" / "musan-instrumental-v1.log"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded preparation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="stream and prepare official MUSAN instrumental audio")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--source-url", default=MUSAN_URL)
    prepare_parser.add_argument("--checksum-url", default=MUSAN_CHECKSUM_URL)
    prepare_parser.add_argument("--max-staging-gib", type=float, default=7.0)
    prepare_parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    prepare_parser.add_argument("--block-size-mib", type=float, default=1.0)
    prepare_parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the preparation CLI and print the final manifest summary."""

    args = _parse_args(argv)
    if args.command != "prepare":
        raise _fail("unsupported music source command")
    max_staging = int(args.max_staging_gib * GIB)
    minimum_free = int(args.minimum_free_gib * GIB)
    block_bytes = int(args.block_size_mib * 1024**2)
    config = PreparationConfig(
        output=args.output,
        source_url=args.source_url,
        checksum_url=args.checksum_url,
        max_staging_bytes=max_staging,
        minimum_free_bytes=minimum_free,
        block_bytes=block_bytes,
        log_path=args.log_file or _default_log_path(),
    )
    result = prepare(config)
    print(
        json.dumps(
            {
                "manifest": str(config.output_root / "manifest.json"),
                "training_manifest": str(config.output_root / "training-manifest.json"),
                "validation_manifest": str(config.output_root / "validation-manifest.json"),
                "track_count": result["track_count"],
                "train_count": result["split"]["train_count"],
                "validation_count": result["split"]["validation_count"],
                "archive_bytes": result["source"]["archive_bytes"],
                "archive_md5": result["source"]["archive_md5"],
                "archive_sha256": result["source"]["archive_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparationError as error:
        print(json.dumps(error.to_json(), sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(2)
