#!/usr/bin/env python3

"""Add VoxConverse development audio to training and keep test audio held out."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from prepare_full_corpus import (
    ChannelPolicy,
    atomic_write,
    probe_audio,
    require_executable,
    scrubbed_environment,
    transcode,
)


RECIPE_DIR = Path(__file__).resolve().parent
VOXCONVERSE_COMMIT = "24bf60be297701cd7e4ef18550c6d390c1b87365"
ANNOTATION_URL = f"https://github.com/joonson/voxconverse/archive/{VOXCONVERSE_COMMIT}.tar.gz"
ANNOTATION_SHA256 = "e8c25c91b014657d7e4ad86f9bef4a7eb399929d8d4fab910d8e6c6ab63d1197"
DEV_AUDIO_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
TEST_AUDIO_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
SESSION_PATTERN = re.compile(r"^[a-z]{5}$")


class Split(str, Enum):
    """A VoxConverse source split and its destination role."""

    DEV = "dev"
    TEST = "test"


@dataclass(frozen=True)
class SourceSplit:
    """One pinned VoxConverse audio and annotation source."""

    split: Split
    audio_url: str
    audio_sha256: str
    expected_recordings: int


@dataclass(frozen=True)
class Annotation:
    """Validated RTTM content for one recording."""

    session_id: str
    rttm: str


@dataclass(frozen=True)
class PreparedSplit:
    """Prepared annotations and the source audio digest for one split."""

    source: SourceSplit
    annotations: tuple[Annotation, ...]
    audio_sha256: str


SOURCES = (
    SourceSplit(Split.DEV, DEV_AUDIO_URL, "e83a68b5df3bc945a3cf4544102038792ae79972753c585769e58ea677c523a8", 216),
    SourceSplit(
        Split.TEST,
        TEST_AUDIO_URL,
        "472ebf1eaeb1dcb5c311b07a8b5c31bcedcccbf98f386d90a88cde2452da8c68",
        232,
    ),
)
TRAIN_RECORDINGS = 732
MANIFEST_PROVENANCE_VERSION = 1
MANIFEST_NAMES = ("wav.scp", "rttm", "all.uem")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def manifest_record_ids(path: Path, id_column: int) -> set[str]:
    """Return unique recording IDs from one materialized manifest."""

    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) <= id_column:
            raise RuntimeError(f"Manifest row lacks column {id_column}: {path}:{line_number}")
        recording_id = fields[id_column]
        if recording_id in ids and path.name != "rttm":
            raise RuntimeError(f"Duplicate recording in manifest: {path}:{line_number}: {recording_id}")
        ids.add(recording_id)

    return ids


def manifest_paths(output_root: Path) -> tuple[Path, ...]:
    """Return every training, development, and evaluation manifest path."""

    split_dirs = (
        output_root / "train",
        output_root / "dev",
        *(output_root / "test" / corpus for corpus in ("AMI", "AliMeeting", "AISHELL4", "VoxConverse")),
    )
    return tuple(split_dir / name for split_dir in split_dirs for name in MANIFEST_NAMES)


def describe_manifests(output_root: Path) -> dict[str, dict[str, object]]:
    """Describe exact materialized manifests for later read-only verification."""

    descriptions = {}
    for path in manifest_paths(output_root):
        if not path.is_file():
            raise RuntimeError(f"Required manifest is absent: {path}")
        id_column = 1 if path.name == "rttm" else 0
        relative_path = path.relative_to(output_root).as_posix()
        descriptions[relative_path] = {
            "sha256": sha256(path),
            "recordings": len(manifest_record_ids(path, id_column)),
        }

    return descriptions


def validate_provenance_identity(provenance: object) -> dict[str, object]:
    """Return validated VoxConverse provenance for the pinned source release."""

    if not isinstance(provenance, dict):
        raise RuntimeError("VoxConverse provenance must be a JSON object")
    if (
        provenance.get("annotation_commit") != VOXCONVERSE_COMMIT
        or provenance.get("annotation_url") != ANNOTATION_URL
        or provenance.get("annotation_sha256") != ANNOTATION_SHA256
    ):
        raise RuntimeError("VoxConverse provenance does not match the pinned annotation release")

    audio = provenance.get("audio")
    if not isinstance(audio, dict):
        raise RuntimeError("VoxConverse provenance lacks audio source records")
    for source in SOURCES:
        record = audio.get(source.split.value)
        if (
            not isinstance(record, dict)
            or record.get("url") != source.audio_url
            or record.get("sha256") != source.audio_sha256
            or record.get("recordings") != source.expected_recordings
        ):
            raise RuntimeError(f"VoxConverse {source.split.value} provenance is invalid")

    return provenance


def read_provenance(path: Path) -> dict[str, object]:
    """Read and validate the pinned VoxConverse provenance file."""

    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read VoxConverse provenance: {path}: {error}") from error

    return validate_provenance_identity(provenance)


def verify_materialized_data(output_root: Path) -> None:
    """Verify exact output manifests, split membership, and referenced audio."""

    provenance = read_provenance(output_root / "provenance.voxconverse.json")
    if provenance.get("manifest_provenance_version") != MANIFEST_PROVENANCE_VERSION:
        raise RuntimeError("VoxConverse provenance does not seal the current output manifests")

    expected_descriptions = provenance.get("manifests")
    actual_descriptions = describe_manifests(output_root)
    if expected_descriptions != actual_descriptions:
        raise RuntimeError("Prepared data manifests do not match VoxConverse provenance")

    train_dir = output_root / "train"
    train_ids = {name: manifest_record_ids(train_dir / name, 1 if name == "rttm" else 0) for name in MANIFEST_NAMES}
    if any(ids != train_ids["wav.scp"] for ids in train_ids.values()):
        raise RuntimeError("Training wav.scp, RTTM, and UEM have different recording membership")
    if len(train_ids["wav.scp"]) != TRAIN_RECORDINGS:
        raise RuntimeError(f"Expected {TRAIN_RECORDINGS} training recordings, found {len(train_ids['wav.scp'])}")

    expected_splits = provenance.get("recording_ids")
    if not isinstance(expected_splits, dict):
        raise RuntimeError("VoxConverse provenance lacks recording membership")
    dev_ids = set(expected_splits.get(Split.DEV.value, ()))
    test_ids = set(expected_splits.get(Split.TEST.value, ()))
    if len(dev_ids) != SOURCES[0].expected_recordings or len(test_ids) != SOURCES[1].expected_recordings:
        raise RuntimeError("VoxConverse provenance has invalid split membership")
    if not all(SESSION_PATTERN.fullmatch(session_id) for session_id in dev_ids | test_ids):
        raise RuntimeError("VoxConverse provenance contains an invalid recording ID")
    if dev_ids & test_ids or not dev_ids.issubset(train_ids["wav.scp"]):
        raise RuntimeError("VoxConverse split membership is inconsistent with the training set")

    test_dir = output_root / "test" / "VoxConverse"
    for name in MANIFEST_NAMES:
        ids = manifest_record_ids(test_dir / name, 1 if name == "rttm" else 0)
        if ids != test_ids:
            raise RuntimeError(f"VoxConverse test {name} has incorrect recording membership")

    for session_id in dev_ids | test_ids:
        path = output_root / "audio" / "VoxConverse" / f"{session_id}.flac"
        if not probe_audio(path):
            raise RuntimeError(f"VoxConverse audio is absent or invalid: {path}")


def validate_session_id(session_id: str) -> str:
    """Return a safe VoxConverse session ID."""

    if not SESSION_PATTERN.fullmatch(session_id):
        raise ValueError(f"Invalid VoxConverse session ID: {session_id!r}")

    return session_id


def download(url: str, destination: Path) -> None:
    """Download one source to an atomic, resumable local file."""

    if destination.is_file() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {destination.name}", flush=True)
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--silent",
            "--show-error",
            "--output",
            partial.as_posix(),
            url,
        ],
        check=True,
        env=scrubbed_environment(),
    )
    partial.replace(destination)


def parse_rttm(session_id: str, content: str) -> Annotation:
    """Parse and validate the RTTM for one VoxConverse recording."""

    validate_session_id(session_id)
    rows = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"Invalid RTTM row for {session_id}:{line_number}")
        if fields[1] != session_id or fields[2] != "1":
            raise ValueError(f"Unexpected RTTM identity for {session_id}:{line_number}")
        start = float(fields[3])
        duration = float(fields[4])
        if not math.isfinite(start) or not math.isfinite(duration):
            raise ValueError(f"Non-finite RTTM time for {session_id}:{line_number}")
        if start < 0 or duration <= 0 or fields[7] == "<NA>":
            raise ValueError(f"Invalid RTTM activity for {session_id}:{line_number}")
        rows.append(" ".join(fields))

    if not rows:
        raise ValueError(f"No RTTM activity for {session_id}")

    return Annotation(session_id, "\n".join(rows) + "\n")


def load_annotations(archive_path: Path) -> dict[Split, tuple[Annotation, ...]]:
    """Load both annotation splits from the pinned source archive."""

    grouped: dict[Split, list[Annotation]] = {split: [] for split in Split}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".rttm":
                continue
            if len(path.parts) < 3 or path.parts[-2] not in {split.value for split in Split}:
                continue
            split = Split(path.parts[-2])
            session_id = validate_session_id(path.stem)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read annotation: {member.name}")
            with source:
                content = source.read().decode("utf-8")
            grouped[split].append(parse_rttm(session_id, content))

    result = {
        split: tuple(sorted(annotations, key=lambda item: item.session_id)) for split, annotations in grouped.items()
    }
    for source in SOURCES:
        actual = len(result[source.split])
        if actual != source.expected_recordings:
            raise RuntimeError(
                f"Expected {source.expected_recordings} {source.split.value} annotations, found {actual}"
            )

    return result


def archive_member_session(member: zipfile.ZipInfo) -> str | None:
    """Return a safe audio member session ID, or none for other members."""

    if member.is_dir():
        return None
    path = PurePosixPath(member.filename)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".wav":
        return None
    try:
        return validate_session_id(path.stem)
    except ValueError:
        return None


def prepare_audio(
    source: SourceSplit,
    annotations: tuple[Annotation, ...],
    archive_path: Path,
    output_root: Path,
) -> PreparedSplit:
    """Extract and normalize the required recordings from one ZIP source."""

    digest = sha256(archive_path)
    if digest != source.audio_sha256:
        raise RuntimeError(f"VoxConverse {source.split.value} archive SHA-256 is invalid")
    audio_dir = output_root / "audio" / "VoxConverse"
    pending = {
        annotation.session_id
        for annotation in annotations
        if not probe_audio(audio_dir / f"{annotation.session_id}.flac")
    }
    seen: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            session_id = archive_member_session(member)
            if session_id is None or session_id not in pending:
                continue
            if session_id in seen:
                raise RuntimeError(f"Duplicate audio member for {session_id}")
            seen.add(session_id)
            print(f"VoxConverse {source.split.value} {session_id}", flush=True)
            with archive.open(member) as audio:
                transcode(
                    audio,
                    audio_dir / f"{session_id}.flac",
                    ChannelPolicy.FIRST,
                )

    missing = pending - seen
    if missing:
        raise RuntimeError(f"VoxConverse {source.split.value} archive lacks: {', '.join(sorted(missing))}")

    return PreparedSplit(source, annotations, digest)


def reuse_prepared_audio(
    source: SourceSplit,
    annotations: tuple[Annotation, ...],
    output_root: Path,
    provenance: dict[str, object],
) -> PreparedSplit:
    """Validate prepared audio and reuse its pinned source archive digest."""

    audio = provenance["audio"]
    if not isinstance(audio, dict):
        raise RuntimeError("VoxConverse provenance lacks audio source records")
    record = audio[source.split.value]
    if not isinstance(record, dict):
        raise RuntimeError(f"VoxConverse {source.split.value} provenance is invalid")

    for annotation in annotations:
        path = output_root / "audio" / "VoxConverse" / f"{annotation.session_id}.flac"
        if not probe_audio(path):
            raise RuntimeError(f"VoxConverse audio is absent or invalid: {path}")

    return PreparedSplit(source, annotations, str(record["sha256"]))


def audio_duration(path: Path) -> float:
    """Return the validated audio duration in seconds."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=scrubbed_environment(),
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Invalid duration for {path}")

    return duration


def render_manifest(
    annotations: tuple[Annotation, ...],
    output_root: Path,
    manifest_audio_prefix: Path,
) -> tuple[str, str, str]:
    """Render wav.scp, RTTM, and full-recording UEM content."""

    wav_lines = []
    rttm_parts = []
    uem_lines = []
    for annotation in annotations:
        filename = f"{annotation.session_id}.flac"
        audio_path = output_root / "audio" / "VoxConverse" / filename
        if not probe_audio(audio_path):
            raise RuntimeError(f"Required audio is absent or invalid: {audio_path}")
        wav_lines.append(f"{annotation.session_id} {manifest_audio_prefix / 'VoxConverse' / filename}")
        rttm_parts.append(annotation.rttm)
        uem_lines.append(f"{annotation.session_id} 1 0.000 {audio_duration(audio_path):.3f}")

    return (
        "\n".join(wav_lines) + "\n",
        "".join(rttm_parts),
        "\n".join(uem_lines) + "\n",
    )


def manifest_ids(content: str, id_column: int = 0) -> set[str]:
    """Return the recording IDs in a Kaldi-style manifest."""

    ids = set()
    for line in content.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) <= id_column:
            raise ValueError(f"Manifest row lacks column {id_column}: {line!r}")
        ids.add(fields[id_column])

    return ids


def replace_training_manifest_records(
    destination: Path,
    content: str,
    id_column: int = 0,
) -> None:
    """Replace VoxConverse rows and preserve all other training records."""

    addition_ids = manifest_ids(content, id_column)
    retained_lines = []
    for line_number, line in enumerate(destination.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) <= id_column:
            raise RuntimeError(f"Manifest row lacks column {id_column}: {destination}:{line_number}")
        if fields[id_column] not in addition_ids:
            retained_lines.append(line)

    retained = "\n".join(retained_lines)
    if retained:
        retained += "\n"
    atomic_write(destination, retained + content)


def write_manifests(
    prepared: dict[Split, PreparedSplit],
    output_root: Path,
    manifest_audio_prefix: Path,
) -> None:
    """Add dev to training and write the held-out test set."""

    dev_wav, dev_rttm, dev_uem = render_manifest(
        prepared[Split.DEV].annotations,
        output_root,
        manifest_audio_prefix,
    )
    train_dir = output_root / "train"
    replace_training_manifest_records(train_dir / "wav.scp", dev_wav)
    replace_training_manifest_records(train_dir / "rttm", dev_rttm, id_column=1)
    replace_training_manifest_records(train_dir / "all.uem", dev_uem)

    test_wav, test_rttm, test_uem = render_manifest(
        prepared[Split.TEST].annotations,
        output_root,
        manifest_audio_prefix,
    )
    test_dir = output_root / "test" / "VoxConverse"
    atomic_write(test_dir / "wav.scp", test_wav)
    atomic_write(test_dir / "rttm", test_rttm)
    atomic_write(test_dir / "all.uem", test_uem)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=RECIPE_DIR / "data" / "full")
    parser.add_argument(
        "--manifest-audio-prefix",
        type=Path,
        default=Path("../speakrs/data/full/audio"),
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="prepare audio now and keep source archives for a later manifest pass",
    )
    parser.add_argument(
        "--reuse-prepared-audio",
        action="store_true",
        help="rebuild manifests from pinned annotations and previously provenanced audio",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify exact prepared manifests and audio without downloading or writing",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare VoxConverse training and held-out evaluation data."""

    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.audio_only and (args.reuse_prepared_audio or args.verify):
        raise ValueError("--audio-only cannot be combined with reuse or verification")
    if args.reuse_prepared_audio and args.verify:
        raise ValueError("--reuse-prepared-audio and --verify cannot be combined")
    if args.verify:
        require_executable("ffprobe")
        verify_materialized_data(output_root)
        print("VoxConverse and combined manifest verification passed", flush=True)
        return

    download_dir = output_root / ".downloads"
    require_executable("curl")
    require_executable("ffprobe")
    if not args.reuse_prepared_audio:
        require_executable("ffmpeg")

    annotation_archive = download_dir / f"voxconverse-{VOXCONVERSE_COMMIT}.tar.gz"
    download(ANNOTATION_URL, annotation_archive)
    annotation_sha256 = sha256(annotation_archive)
    if annotation_sha256 != ANNOTATION_SHA256:
        raise RuntimeError("VoxConverse annotation archive SHA-256 is invalid")
    annotations = load_annotations(annotation_archive)

    prepared = {}
    if args.reuse_prepared_audio:
        existing_provenance = read_provenance(output_root / "provenance.voxconverse.json")
        for source in SOURCES:
            prepared[source.split] = reuse_prepared_audio(
                source,
                annotations[source.split],
                output_root,
                existing_provenance,
            )
    else:
        for source in SOURCES:
            audio_archive = download_dir / f"voxconverse-{source.split.value}.zip"
            download(source.audio_url, audio_archive)
            prepared[source.split] = prepare_audio(
                source,
                annotations[source.split],
                audio_archive,
                output_root,
            )

    if args.audio_only:
        print("VoxConverse audio preparation is complete", flush=True)
        return

    write_manifests(prepared, output_root, args.manifest_audio_prefix)
    provenance = {
        "dataset": "VoxConverse",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "copyright_notice": "Copyright remains with the original video owners",
        "training_source_split": "dev",
        "held_out_evaluation_split": "test",
        "annotation_commit": VOXCONVERSE_COMMIT,
        "annotation_url": ANNOTATION_URL,
        "annotation_sha256": annotation_sha256,
        "audio": {
            split.value: {
                "url": item.source.audio_url,
                "sha256": item.audio_sha256,
                "recordings": len(item.annotations),
            }
            for split, item in prepared.items()
        },
        "manifest_provenance_version": MANIFEST_PROVENANCE_VERSION,
        "recording_ids": {
            split.value: [annotation.session_id for annotation in item.annotations] for split, item in prepared.items()
        },
        "manifests": describe_manifests(output_root),
    }
    atomic_write(
        output_root / "provenance.voxconverse.json",
        json.dumps(provenance, indent=2) + "\n",
    )
    for source in SOURCES:
        (download_dir / f"voxconverse-{source.split.value}.zip").unlink(missing_ok=True)
    annotation_archive.unlink()
    print("VoxConverse preparation is complete", flush=True)


if __name__ == "__main__":
    main()
