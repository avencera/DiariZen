"""Build and verify the portable frozen development bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping
from urllib.parse import quote

import soundfile as sf

from .contracts import DataPreparationSpec
from .errors import PreparationError
from .hashing import sha256_file
from .validation.contracts import Sha256Digest


ESTABLISHED_SOURCES = ("AMI", "AliMeeting")
AISHELL5_SOURCE = "AISHELL-5"
DEV_SOURCES = (*ESTABLISHED_SOURCES, AISHELL5_SOURCE)
EXPECTED_DEV_RECORDINGS = 44
DEV_BUNDLE_SCHEMA = "speakrs-frozen-dev-bundle-v2"
DEV_BUNDLE_PATHS_SCHEMA = "speakrs-frozen-dev-bundle-paths-v1"
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class DevBundleFileRole(str, Enum):
    """The single role a frozen development file fulfils."""

    AUDIO = "audio"
    RTTM = "rttm"
    UEM = "uem"


@dataclass(frozen=True, slots=True)
class DevBundleFile:
    """One content-addressed file in a frozen recording."""

    logical_id: str
    path: str
    sha256: Sha256Digest
    byte_length: int
    role: DevBundleFileRole

    def __post_init__(self) -> None:
        """Reject invalid file identities even when constructed without JSON."""

        role = self.role
        try:
            role = DevBundleFileRole(role)
        except (TypeError, ValueError) as error:
            raise PreparationError("bundle file.role is invalid") from error
        object.__setattr__(self, "role", role)
        digest = _require_digest(self.sha256, "bundle file.sha256")
        object.__setattr__(self, "sha256", digest)
        _require_identity(self.logical_id, "bundle file.logical_id")
        _safe_relative_path(self.path, "bundle file.path")
        _require_length(self.byte_length, "bundle file.byte_length")

    @classmethod
    def from_json(cls, payload: Any, *, label: str = "bundle file") -> "DevBundleFile":
        """Parse one strict file identity."""

        data = _require_object(payload, label)
        _reject_unknown(data, {"logical_id", "path", "sha256", "byte_length", "role"}, label)
        logical_id = _require_identity(data.get("logical_id"), f"{label}.logical_id")
        path = _safe_relative_path(data.get("path"), f"{label}.path")
        digest = _require_digest(data.get("sha256"), f"{label}.sha256")
        byte_length = _require_length(data.get("byte_length"), f"{label}.byte_length")
        role_value = data.get("role")
        try:
            role = DevBundleFileRole(role_value)
        except (TypeError, ValueError) as error:
            raise PreparationError(
                f"{label}.role must be one of {[role.value for role in DevBundleFileRole]}"
            ) from error
        return cls(logical_id, path, digest, byte_length, role)

    def to_json(self) -> dict[str, object]:
        """Return the exact logical file identity."""

        return {
            "logical_id": self.logical_id,
            "path": self.path,
            "sha256": self.sha256.value,
            "byte_length": self.byte_length,
            "role": self.role.value,
        }


@dataclass(frozen=True, slots=True)
class DevBundleRecording:
    """One complete frozen recording and its audio, RTTM, and UEM files."""

    logical_id: str
    recording_id: str
    source: str
    files: tuple[DevBundleFile, ...]

    def __post_init__(self) -> None:
        """Reject incomplete recording identities even outside JSON parsing."""

        files = tuple(self.files)
        object.__setattr__(self, "files", files)
        _require_identity(self.logical_id, "bundle recording.logical_id")
        recording_id = _require_logical_text(self.recording_id, "bundle recording.recording_id")
        source = _require_logical_text(self.source, "bundle recording.source")
        if source not in DEV_SOURCES:
            raise PreparationError("bundle recording.source is not a frozen development source", {"source": source})
        if self.logical_id != _recording_logical_id(source, recording_id):
            raise PreparationError("bundle recording.logical_id does not match source and recording_id")
        if len(files) != len({file.role for file in files}) or {file.role for file in files} != set(DevBundleFileRole):
            raise PreparationError(
                "bundle recording must contain exactly one audio, RTTM, and UEM file",
                {"recording_id": recording_id},
            )
        expected_ids = {_file_logical_id(self.logical_id, role) for role in DevBundleFileRole}
        if {file.logical_id for file in files} != expected_ids:
            raise PreparationError(
                "bundle recording file logical identities are incomplete", {"recording_id": recording_id}
            )

    @classmethod
    def from_json(cls, payload: Any, *, label: str = "bundle recording") -> "DevBundleRecording":
        """Parse one recording and require its complete file pair."""

        data = _require_object(payload, label)
        _reject_unknown(data, {"logical_id", "recording_id", "source", "files"}, label)
        logical_id = _require_identity(data.get("logical_id"), f"{label}.logical_id")
        recording_id = _require_logical_text(data.get("recording_id"), f"{label}.recording_id")
        source = _require_logical_text(data.get("source"), f"{label}.source")
        files_value = data.get("files")
        if not isinstance(files_value, list):
            raise PreparationError(f"{label}.files must be a list")
        files = tuple(
            DevBundleFile.from_json(file_value, label=f"{label}.files[{index}]")
            for index, file_value in enumerate(files_value)
        )
        if logical_id != _recording_logical_id(source, recording_id):
            raise PreparationError(f"{label}.logical_id does not match source and recording_id")
        roles = {file.role for file in files}
        required_roles = set(DevBundleFileRole)
        if roles != required_roles or len(files) != len(required_roles):
            raise PreparationError(
                "bundle recording must contain exactly one audio, RTTM, and UEM file",
                {"recording_id": recording_id, "roles": sorted(role.value for role in roles)},
            )
        expected_file_ids = {_file_logical_id(logical_id, role) for role in DevBundleFileRole}
        if {file.logical_id for file in files} != expected_file_ids:
            raise PreparationError(
                "bundle recording file logical identities are incomplete", {"recording_id": recording_id}
            )
        return cls(logical_id, recording_id, source, tuple(sorted(files, key=lambda file: file.role.value)))

    def to_json(self) -> dict[str, object]:
        """Return the exact logical recording identity."""

        return {
            "logical_id": self.logical_id,
            "recording_id": self.recording_id,
            "source": self.source,
            "files": [file.to_json() for file in sorted(self.files, key=lambda item: item.role.value)],
        }

    def file(self, role: DevBundleFileRole) -> DevBundleFile:
        """Return the file for one required recording role."""

        for file in self.files:
            if file.role is role:
                return file
        raise KeyError(role)


@dataclass(frozen=True, slots=True)
class FrozenDevBundle:
    """Location-neutral identity for one complete frozen development bundle."""

    recordings: tuple[DevBundleRecording, ...]
    schema: str = DEV_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        """Reject duplicate recording or file identities before publication."""

        if self.schema != DEV_BUNDLE_SCHEMA:
            raise PreparationError("bundle schema is not the frozen development schema")
        if not self.recordings:
            raise PreparationError("frozen development bundle contains no recordings")
        recording_ids: set[str] = set()
        recording_logical_ids: set[str] = set()
        file_logical_ids: set[str] = set()
        file_paths: set[str] = set()
        for recording in self.recordings:
            if not isinstance(recording, DevBundleRecording):
                raise PreparationError("frozen development bundle recordings must be typed recording models")
            if recording.recording_id in recording_ids:
                raise PreparationError("bundle has duplicate recording ids", {"recording_id": recording.recording_id})
            if recording.logical_id in recording_logical_ids:
                raise PreparationError(
                    "bundle has duplicate recording logical identities", {"logical_id": recording.logical_id}
                )
            recording_ids.add(recording.recording_id)
            recording_logical_ids.add(recording.logical_id)
            for file in recording.files:
                if file.logical_id in file_logical_ids:
                    raise PreparationError(
                        "bundle has duplicate file logical identities", {"logical_id": file.logical_id}
                    )
                if file.path in file_paths:
                    raise PreparationError("bundle has duplicate file paths", {"path": file.path})
                file_logical_ids.add(file.logical_id)
                file_paths.add(file.path)

    @classmethod
    def from_json(cls, payload: Any, *, expected_recordings: int | None = None) -> "FrozenDevBundle":
        """Parse a canonical manifest and reject all unknown or incomplete data."""

        data = _require_object(payload, "frozen development bundle")
        _reject_unknown(data, {"schema", "recordings"}, "frozen development bundle")
        if data.get("schema") != DEV_BUNDLE_SCHEMA:
            raise PreparationError("frozen development bundle has an unsupported schema")
        recordings_value = data.get("recordings")
        if not isinstance(recordings_value, list):
            raise PreparationError("frozen development bundle.recordings must be a list")
        if expected_recordings is not None and len(recordings_value) != expected_recordings:
            raise PreparationError(
                f"frozen development bundle must contain exactly {expected_recordings} recordings",
                {"actual": len(recordings_value)},
            )
        recordings = tuple(
            DevBundleRecording.from_json(value, label=f"bundle recordings[{index}]")
            for index, value in enumerate(recordings_value)
        )
        if expected_recordings == EXPECTED_DEV_RECORDINGS and {recording.source for recording in recordings} != set(
            DEV_SOURCES
        ):
            raise PreparationError("44-recording frozen development bundle must cover all frozen sources")
        ordered = tuple(sorted(recordings, key=_recording_sort_key))
        return cls(ordered)

    def to_json(self) -> dict[str, object]:
        """Return the canonical logical manifest without local path-view data."""

        return {
            "schema": self.schema,
            "recordings": [recording.to_json() for recording in sorted(self.recordings, key=_recording_sort_key)],
        }

    def canonical_json(self) -> str:
        """Serialize this identity with compact, deterministic JSON."""

        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def canonical_bytes(self) -> bytes:
        """Return the bytes bound by the bundle identity."""

        return self.canonical_json().encode("utf-8")

    def identity_sha256(self) -> str:
        """Return the location-neutral SHA-256 identity."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def files(self) -> tuple[DevBundleFile, ...]:
        """Return all files in deterministic recording and role order."""

        return tuple(
            file
            for recording in sorted(self.recordings, key=_recording_sort_key)
            for file in sorted(recording.files, key=lambda item: item.role.value)
        )


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreparationError(f"{label} must be an object")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise PreparationError(f"{label} has unknown fields", {"unknown": unknown})


def _require_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or _CONTROL_RE.search(value):
        raise PreparationError(f"{label} must be a non-empty logical identity")
    return value


def _require_logical_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or _CONTROL_RE.search(value):
        raise PreparationError(f"{label} must be a non-empty logical text value")
    if "/" in value or "\\" in value:
        raise PreparationError(f"{label} must not contain path separators")
    return value


def _require_digest(value: Any, label: str) -> Sha256Digest:
    if isinstance(value, Sha256Digest):
        return value
    try:
        return Sha256Digest.parse(value, label)
    except (TypeError, ValueError) as error:
        raise PreparationError(f"{label} must be a lowercase SHA-256 digest") from error


def _require_length(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreparationError(f"{label} must be a non-negative integer")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or _CONTROL_RE.search(value):
        raise PreparationError(f"{label} must be a safe relative path")
    if "\\" in value or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise PreparationError(f"{label} must be a safe relative path", {"path": value})
    if len(value) >= 2 and value[1] == ":":
        raise PreparationError(f"{label} must not contain a drive prefix", {"path": value})
    parts = value.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise PreparationError(f"{label} must not contain traversal components", {"path": value})
    if PurePosixPath(value).as_posix() != value:
        raise PreparationError(f"{label} is not canonical", {"path": value})
    return value


def _recording_logical_id(source: str, recording_id: str) -> str:
    return f"{source}:{recording_id}"


def _file_logical_id(recording_logical_id: str, role: DevBundleFileRole) -> str:
    return f"{recording_logical_id}:{role.value}"


def _recording_sort_key(recording: DevBundleRecording) -> tuple[int, str, str]:
    try:
        source_index = DEV_SOURCES.index(recording.source)
    except ValueError:
        source_index = len(DEV_SOURCES)
    return source_index, recording.source, recording.recording_id


def _frozen_dev_ids(spec: DataPreparationSpec) -> dict[str, tuple[str, ...]]:
    """Return disjoint frozen dev ids and reject an ambiguous split contract."""

    dev: dict[str, tuple[str, ...]] = {}
    all_dev_ids: set[str] = set()
    for source, splits in spec.frozen_splits.items():
        source_dev = tuple(splits.get("dev", ()))
        train = set(splits.get("train", ()))
        test = set(splits.get("test", ()))
        if len(set(source_dev)) != len(source_dev):
            raise PreparationError("frozen development split has duplicate ids", {"source": source})
        overlap = set(source_dev) & (train | test)
        if overlap:
            raise PreparationError(
                "frozen development split overlaps train or test", {"source": source, "ids": sorted(overlap)}
            )
        global_overlap = all_dev_ids & set(source_dev)
        if global_overlap:
            raise PreparationError(
                "development recording ids are not globally unique", {"ids": sorted(global_overlap)}
            )
        all_dev_ids.update(source_dev)
        if source_dev:
            dev[source] = source_dev
    required = set(ESTABLISHED_SOURCES) | {AISHELL5_SOURCE}
    if set(dev) != required:
        raise PreparationError("development sources do not match the four-source run", {"actual": sorted(dev)})
    return {source: dev[source] for source in DEV_SOURCES}


def _established_labels(root: Path, expected: set[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load the established AMI and AliMeeting label rows for exact frozen ids."""

    rttm_path = root / "rttm"
    uem_path = root / "all.uem"
    if not rttm_path.is_file() or not uem_path.is_file():
        raise PreparationError("established development labels are incomplete", {"root": str(root)})
    rttm: dict[str, list[str]] = {recording_id: [] for recording_id in expected}
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise PreparationError("established development RTTM row is invalid")
        if fields[1] in rttm:
            rttm[fields[1]].append(line + "\n")
    missing_rttm = sorted(recording_id for recording_id, rows in rttm.items() if not rows)
    if missing_rttm:
        raise PreparationError("established development RTTM is missing frozen ids", {"ids": missing_rttm})
    uem: dict[str, str] = {}
    for line in uem_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise PreparationError("established development UEM row is invalid")
        if fields[0] in expected:
            if fields[0] in uem:
                raise PreparationError("established development UEM has duplicate ids", {"id": fields[0]})
            uem[fields[0]] = line + "\n"
    missing_uem = sorted(expected - set(uem))
    if missing_uem:
        raise PreparationError("established development UEM is missing frozen ids", {"ids": missing_uem})
    return rttm, uem


def _aishell5_intervals(path: Path, recording_id: str) -> tuple[float, list[str]]:
    """Convert speaker-tier TextGrid intervals into RTTM rows."""

    text = path.read_text(encoding="utf-8", errors="strict").replace("\r\n", "\n")
    xmax_match = re.search(r"^xmax\s*=\s*([0-9.]+)\s*$", text, flags=re.M)
    if xmax_match is None:
        raise PreparationError("AISHELL-5 TextGrid has no duration", {"path": str(path)})
    duration = float(xmax_match.group(1))
    rows: list[str] = []
    for body in re.findall(r"^\s*item \[\d+\]:\s*(.*?)(?=^\s*item \[\d+\]:|\Z)", text, flags=re.M | re.S):
        name_match = re.search(r'^\s*name\s*=\s*"([^\"]+)"\s*$', body, flags=re.M)
        if name_match is None:
            continue
        speaker = name_match.group(1).strip()
        if not speaker:
            raise PreparationError("AISHELL-5 TextGrid has an unnamed speaker tier", {"path": str(path)})
        for start_text, end_text, mark in re.findall(
            r'^\s*xmin\s*=\s*([0-9.]+)\s*\n\s*xmax\s*=\s*([0-9.]+)\s*\n\s*text\s*=\s*"([^\"]*)"',
            body,
            flags=re.M,
        ):
            if not mark.strip():
                continue
            start = float(start_text)
            end = float(end_text)
            if start < 0 or end <= start or end > duration + 1e-6:
                raise PreparationError("AISHELL-5 TextGrid interval is outside its recording", {"path": str(path)})
            rows.append(f"SPEAKER {recording_id} 1 {start:.6f} {end - start:.6f} <NA> <NA> {speaker} <NA> <NA>\n")
    if not rows:
        raise PreparationError("AISHELL-5 TextGrid contains no speech", {"path": str(path)})
    return duration, rows


def _copy_audio(source: Path, destination: Path, expected_digest: str, expected_size: int | None = None) -> None:
    """Publish one verified audio object through an atomic local path."""

    source = Path(source)
    destination = Path(destination)
    if expected_size is None:
        expected_size = source.stat().st_size
    source_size = source.stat().st_size
    source_digest = sha256_file(source)
    if source_size != expected_size or source_digest != expected_digest:
        raise PreparationError("development audio differs from its declared identity", {"path": str(source)})
    if (
        destination.is_file()
        and not destination.is_symlink()
        and destination.stat().st_size == expected_size
        and sha256_file(destination) == expected_digest
    ):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as destination_handle:
            source_digest = hashlib.sha256()
            source_size = 0
            while block := source_handle.read(1024 * 1024):
                source_digest.update(block)
                source_size += len(block)
                destination_handle.write(block)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        if source_size != expected_size or source_digest.hexdigest() != expected_digest:
            raise PreparationError("development audio changed while it was copied", {"path": str(source)})
        if temporary.stat().st_size != expected_size or sha256_file(temporary) != expected_digest:
            raise PreparationError("development audio copy failed content verification", {"path": str(source)})
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_verified_bytes(destination: Path, payload: bytes, *, digest: str | None = None) -> tuple[str, int]:
    """Write generated bytes and verify the exact bytes published to disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_digest = digest or hashlib.sha256(payload).hexdigest()
    if (
        destination.is_file()
        and destination.stat().st_size == len(payload)
        and sha256_file(destination) == expected_digest
    ):
        return expected_digest, len(payload)
    destination.write_bytes(payload)
    actual_size = destination.stat().st_size
    actual_digest = sha256_file(destination)
    if actual_size != len(payload) or actual_digest != expected_digest:
        raise PreparationError(
            "generated development bundle file failed content verification", {"path": str(destination)}
        )
    return actual_digest, actual_size


def _path_safe_component(value: str) -> str:
    return quote(value, safe="-_.~")


def _file_identity(
    recording_logical_id: str,
    role: DevBundleFileRole,
    path: Path,
    digest: str | Sha256Digest,
    byte_length: int,
) -> DevBundleFile:
    """Construct one verified file identity with a bundle-relative path."""

    return DevBundleFile(
        logical_id=_file_logical_id(recording_logical_id, role),
        path=path.as_posix(),
        sha256=_require_digest(digest, "file sha256"),
        byte_length=_require_length(byte_length, "file byte length"),
        role=role,
    )


def parse_dev_bundle_manifest(payload: Any, *, expected_recordings: int | None = None) -> FrozenDevBundle:
    """Parse a canonical frozen development manifest."""

    return FrozenDevBundle.from_json(payload, expected_recordings=expected_recordings)


def load_dev_bundle(path: Path, *, expected_recordings: int | None = None) -> FrozenDevBundle:
    """Read and parse a canonical frozen development manifest from disk."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError("frozen development manifest cannot be read", {"path": str(manifest_path)}) from error
    return parse_dev_bundle_manifest(payload, expected_recordings=expected_recordings)


def bundle_identity(bundle: FrozenDevBundle | Mapping[str, Any]) -> str:
    """Return the location-neutral SHA-256 for a canonical bundle."""

    parsed = bundle if isinstance(bundle, FrozenDevBundle) else parse_dev_bundle_manifest(bundle)
    return parsed.identity_sha256()


def _verify_derived_paths_view(root: Path, bundle: FrozenDevBundle) -> None:
    """Verify optional trainer paths and aggregate annotation views."""

    paths_manifest = root / "bundle.paths.json"
    if not paths_manifest.is_file():
        return
    try:
        payload = json.loads(paths_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError("frozen development path view cannot be read") from error
    data = _require_object(payload, "frozen development path view")
    _reject_unknown(
        data, {"schema", "identity_sha256", "wav_prefix", "recordings", "manifests"}, "frozen development path view"
    )
    if data.get("schema") != DEV_BUNDLE_PATHS_SCHEMA or data.get("identity_sha256") != bundle.identity_sha256():
        raise PreparationError("frozen development path view is bound to a different bundle")
    wav_prefix = data.get("wav_prefix")
    if not isinstance(wav_prefix, str) or not wav_prefix.strip():
        raise PreparationError("frozen development path view wav_prefix is invalid")
    path_recordings = data.get("recordings")
    if not isinstance(path_recordings, list) or len(path_recordings) != len(bundle.recordings):
        raise PreparationError("frozen development path view recording coverage is incomplete")
    expected_path_recordings = [
        {
            "logical_id": recording.logical_id,
            "recording_id": recording.recording_id,
            "source": recording.source,
            "audio_path": recording.file(DevBundleFileRole.AUDIO).path,
        }
        for recording in bundle.recordings
    ]
    if path_recordings != expected_path_recordings:
        raise PreparationError("frozen development path view differs from bundle identities")
    manifests = data.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != {"wav_scp", "rttm", "uem"}:
        raise PreparationError("frozen development path view manifests are incomplete")
    manifest_names = {"wav_scp": "wav.scp", "rttm": "all.rttm", "uem": "all.uem"}
    for name, filename in manifest_names.items():
        identity = _require_object(manifests[name], f"frozen development path view.manifests.{name}")
        _reject_unknown(identity, {"path", "sha256", "byte_length"}, f"frozen development path view.manifests.{name}")
        if identity.get("path") != filename:
            raise PreparationError(
                "frozen development path view manifest path is invalid", {"path": identity.get("path")}
            )
        digest = _require_digest(identity.get("sha256"), f"frozen development path view.manifests.{name}.sha256").value
        size = _require_length(
            identity.get("byte_length"), f"frozen development path view.manifests.{name}.byte_length"
        )
        path = root / filename
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise PreparationError(
                "frozen development path view manifest differs from its identity", {"path": filename}
            )

    expected_rttm = b"".join(
        (root / PurePosixPath(recording.file(DevBundleFileRole.RTTM).path)).read_bytes()
        for recording in bundle.recordings
    )
    expected_uem = b"".join(
        (root / PurePosixPath(recording.file(DevBundleFileRole.UEM).path)).read_bytes()
        for recording in bundle.recordings
    )
    if (root / "all.rttm").read_bytes() != expected_rttm or (root / "all.uem").read_bytes() != expected_uem:
        raise PreparationError("frozen development annotation views differ from recording identities")
    expected_wav = "".join(
        f"{recording.recording_id} {(Path(wav_prefix) / recording.file(DevBundleFileRole.AUDIO).path).as_posix()}\n"
        for recording in bundle.recordings
    )
    if (root / "wav.scp").read_text(encoding="utf-8") != expected_wav:
        raise PreparationError("frozen development wav view differs from recording identities")


def verify_dev_bundle(
    root: Path,
    manifest: FrozenDevBundle | Mapping[str, Any] | Path | None = None,
    expected_recordings: int | None = EXPECTED_DEV_RECORDINGS,
) -> dict[str, object]:
    """Verify every expected bundle file before a caller allocates CUDA."""

    root = Path(root)
    if manifest is None:
        if root.is_file():
            manifest = root
            root = root.parent
        else:
            manifest = root / "bundle.json"
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        if manifest_path.is_dir():
            root = manifest_path
            manifest_path = root / "bundle.json"
        elif manifest_path.name == "bundle.json":
            root = manifest_path.parent
        if manifest_path.is_symlink():
            raise PreparationError("frozen development manifest must be a regular file", {"path": str(manifest_path)})
        parsed = load_dev_bundle(manifest_path, expected_recordings=expected_recordings)
        raw_manifest = manifest_path.read_bytes()
        if raw_manifest != parsed.canonical_bytes():
            raise PreparationError("frozen development manifest is not canonical", {"path": str(manifest_path)})
    else:
        parsed = (
            manifest
            if isinstance(manifest, FrozenDevBundle)
            else parse_dev_bundle_manifest(manifest, expected_recordings=expected_recordings)
        )
        if expected_recordings is not None and len(parsed.recordings) != expected_recordings:
            raise PreparationError(
                f"frozen development bundle must contain exactly {expected_recordings} recordings",
                {"actual": len(parsed.recordings)},
            )
        manifest_path = root / "bundle.json"
        if manifest_path.is_file() and not manifest_path.is_symlink():
            on_disk = load_dev_bundle(manifest_path, expected_recordings=expected_recordings)
            if (
                on_disk.identity_sha256() != parsed.identity_sha256()
                or manifest_path.read_bytes() != on_disk.canonical_bytes()
            ):
                raise PreparationError("frozen development manifest differs from the supplied identity")
    if not root.is_dir():
        raise PreparationError("frozen development bundle root is missing", {"path": str(root)})
    expected_paths = {file.path for file in parsed.files}
    actual_identity_paths = {
        path.relative_to(root).as_posix()
        for directory in (root / "audio", root / "annotations")
        if directory.is_dir()
        for path in directory.rglob("*")
        if (path.is_file() or path.is_symlink()) and not path.name.endswith(".part")
    }
    extra_paths = sorted(actual_identity_paths - expected_paths)
    if extra_paths:
        raise PreparationError("frozen development bundle has extra identity files", {"paths": extra_paths})
    missing_paths = sorted(expected_paths - actual_identity_paths)
    if missing_paths:
        raise PreparationError("frozen development bundle is missing identity files", {"paths": missing_paths})
    for file in parsed.files:
        path = root / PurePosixPath(file.path)
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise PreparationError("frozen development bundle file escapes its root", {"path": file.path}) from error
        if path.is_symlink() or not path.is_file():
            raise PreparationError("frozen development bundle file is missing", {"path": file.path})
        actual_size = path.stat().st_size
        actual_digest = sha256_file(path)
        if actual_size != file.byte_length or actual_digest != file.sha256.value:
            raise PreparationError(
                "frozen development bundle file differs from its identity",
                {
                    "logical_id": file.logical_id,
                    "path": file.path,
                    "expected_size": file.byte_length,
                    "actual_size": actual_size,
                    "expected_sha256": file.sha256.value,
                    "actual_sha256": actual_digest,
                },
            )
    _verify_derived_paths_view(root, parsed)
    return {
        "ok": True,
        "schema": parsed.schema,
        "identity_sha256": parsed.identity_sha256(),
        "recordings": len(parsed.recordings),
        "files": len(parsed.files),
    }


def build_dev_bundle(
    spec: DataPreparationSpec,
    audio_root: Path,
    established_dev: Path,
    aishell5_dev: Path,
    wav_prefix: str,
    output: Path,
) -> dict[str, object]:
    """Build and atomically publish trainer inputs for frozen development recordings."""

    if not isinstance(wav_prefix, str) or not wav_prefix.strip():
        raise PreparationError("development bundle requires a non-empty trainer-relative wav prefix")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise PreparationError("development bundle output already exists", {"path": str(output)})
    dev_ids = _frozen_dev_ids(spec)
    established_ids = set(dev_ids["AMI"]) | set(dev_ids["AliMeeting"])
    established_rttm, established_uem = _established_labels(Path(established_dev), established_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = tempfile.mkdtemp(prefix=f".{output.name}-", dir=str(output.parent))
    partial = Path(temporary_name)
    recordings: list[DevBundleRecording] = []
    wav_rows: list[str] = []
    rttm_rows: list[str] = []
    uem_rows: list[str] = []
    try:
        for source in DEV_SOURCES:
            for recording_id in sorted(dev_ids[source]):
                if any(recording.recording_id == recording_id for recording in recordings):
                    raise PreparationError("development recording ids are not globally unique", {"id": recording_id})
                if source in ESTABLISHED_SOURCES:
                    source_audio = Path(audio_root) / source / f"{recording_id}.flac"
                    rows = established_rttm[recording_id]
                    uem_row = established_uem[recording_id]
                else:
                    room = recording_id.removeprefix("dev-")
                    source_audio = Path(aishell5_dev) / room / "DX01C01.wav"
                    source_label = Path(aishell5_dev) / room / "DX01C01.TextGrid"
                    _, rows = _aishell5_intervals(source_label, recording_id)
                    uem_row = None
                if not source_audio.is_file():
                    raise PreparationError("frozen development audio is missing", {"path": str(source_audio)})
                audio_info = sf.info(source_audio)
                if audio_info.samplerate != 16_000 or audio_info.channels != 1 or audio_info.frames <= 0:
                    raise PreparationError(
                        "development audio must be non-empty mono 16 kHz", {"path": str(source_audio)}
                    )
                duration = audio_info.frames / audio_info.samplerate
                for row in rows:
                    fields = row.split()
                    if float(fields[3]) + float(fields[4]) > duration + 0.25:
                        raise PreparationError(
                            "development RTTM exceeds audio duration", {"recording_id": recording_id}
                        )
                if uem_row is not None:
                    uem_fields = uem_row.split()
                    if float(uem_fields[2]) < 0 or float(uem_fields[3]) > duration + 0.25:
                        raise PreparationError(
                            "established development UEM exceeds audio duration", {"recording_id": recording_id}
                        )

                audio_digest = sha256_file(source_audio)
                audio_size = source_audio.stat().st_size
                recording_logical_id = _recording_logical_id(source, recording_id)
                safe_id = _path_safe_component(recording_id)
                suffix = source_audio.suffix.lower() or ".audio"
                audio_relative = Path("audio") / source / f"{safe_id}-{audio_digest}{suffix}"
                _copy_audio(source_audio, partial / audio_relative, audio_digest, audio_size)

                rttm_bytes = "".join(rows).encode("utf-8")
                uem_bytes = (uem_row or f"{recording_id} 1 0.000000 {duration:.6f}\n").encode("utf-8")
                rttm_digest = hashlib.sha256(rttm_bytes).hexdigest()
                uem_digest = hashlib.sha256(uem_bytes).hexdigest()
                rttm_relative = Path("annotations") / source / f"{safe_id}-{rttm_digest}.rttm"
                uem_relative = Path("annotations") / source / f"{safe_id}-{uem_digest}.uem"
                _write_verified_bytes(partial / rttm_relative, rttm_bytes, digest=rttm_digest)
                _write_verified_bytes(partial / uem_relative, uem_bytes, digest=uem_digest)

                wav_rows.append(f"{recording_id} {(Path(wav_prefix) / audio_relative).as_posix()}\n")
                rttm_rows.extend(rows)
                uem_rows.append(uem_bytes.decode("utf-8"))
                recordings.append(
                    DevBundleRecording(
                        logical_id=recording_logical_id,
                        recording_id=recording_id,
                        source=source,
                        files=(
                            _file_identity(
                                recording_logical_id,
                                DevBundleFileRole.AUDIO,
                                audio_relative,
                                audio_digest,
                                audio_size,
                            ),
                            _file_identity(
                                recording_logical_id,
                                DevBundleFileRole.RTTM,
                                rttm_relative,
                                rttm_digest,
                                len(rttm_bytes),
                            ),
                            _file_identity(
                                recording_logical_id,
                                DevBundleFileRole.UEM,
                                uem_relative,
                                uem_digest,
                                len(uem_bytes),
                            ),
                        ),
                    )
                )

        parsed = FrozenDevBundle(tuple(sorted(recordings, key=_recording_sort_key)))
        manifest_path = partial / "bundle.json"
        manifest_path.write_bytes(parsed.canonical_bytes())
        if manifest_path.read_bytes() != parsed.canonical_bytes():
            raise PreparationError("frozen development manifest failed content verification")

        manifest_paths = {
            "wav_scp": partial / "wav.scp",
            "rttm": partial / "all.rttm",
            "uem": partial / "all.uem",
        }
        _write_verified_bytes(manifest_paths["wav_scp"], "".join(wav_rows).encode("utf-8"))
        _write_verified_bytes(manifest_paths["rttm"], "".join(rttm_rows).encode("utf-8"))
        _write_verified_bytes(manifest_paths["uem"], "".join(uem_rows).encode("utf-8"))
        paths_payload = {
            "schema": DEV_BUNDLE_PATHS_SCHEMA,
            "identity_sha256": parsed.identity_sha256(),
            "wav_prefix": wav_prefix,
            "recordings": [
                {
                    "logical_id": recording.logical_id,
                    "recording_id": recording.recording_id,
                    "source": recording.source,
                    "audio_path": recording.file(DevBundleFileRole.AUDIO).path,
                }
                for recording in parsed.recordings
            ],
            "manifests": {
                name: {"path": path.name, "sha256": sha256_file(path), "byte_length": path.stat().st_size}
                for name, path in manifest_paths.items()
            },
        }
        paths_manifest = partial / "bundle.paths.json"
        paths_manifest.write_text(json.dumps(paths_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, output)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    return {
        "ok": True,
        "command": "dev-bundle",
        "bundle": str(output),
        "bundle_manifest": str(output / "bundle.json"),
        "bundle_manifest_sha256": sha256_file(output / "bundle.json"),
        "bundle_identity_sha256": parsed.identity_sha256(),
        "recordings": len(parsed.recordings),
        "sources": list(dev_ids),
    }
