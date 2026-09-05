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
    SourceSplit(Split.DEV, DEV_AUDIO_URL, 216),
    SourceSplit(Split.TEST, TEST_AUDIO_URL, 232),
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


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


def append_training_manifest(
    destination: Path,
    content: str,
    id_column: int = 0,
) -> None:
    """Append a disjoint VoxConverse manifest to the commercial training set."""

    existing = destination.read_text()
    existing_ids = manifest_ids(existing, id_column)
    addition_ids = manifest_ids(content, id_column)
    duplicate_ids = existing_ids & addition_ids
    if duplicate_ids == addition_ids:
        return
    if duplicate_ids:
        raise RuntimeError(f"Duplicate training recordings in {destination}: {', '.join(sorted(duplicate_ids))}")
    separator = "" if existing.endswith("\n") else "\n"
    atomic_write(destination, existing + separator + content)


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
    append_training_manifest(train_dir / "wav.scp", dev_wav)
    append_training_manifest(train_dir / "rttm", dev_rttm, id_column=1)
    append_training_manifest(train_dir / "all.uem", dev_uem)

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
    return parser.parse_args()


def main() -> None:
    """Prepare VoxConverse training and held-out evaluation data."""

    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    download_dir = output_root / ".downloads"
    require_executable("curl")
    require_executable("ffmpeg")
    require_executable("ffprobe")

    annotation_archive = download_dir / f"voxconverse-{VOXCONVERSE_COMMIT}.tar.gz"
    download(ANNOTATION_URL, annotation_archive)
    annotation_sha256 = sha256(annotation_archive)
    annotations = load_annotations(annotation_archive)

    prepared = {}
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
    }
    atomic_write(
        output_root / "provenance.voxconverse.json",
        json.dumps(provenance, indent=2) + "\n",
    )
    for source in SOURCES:
        (download_dir / f"voxconverse-{source.split.value}.zip").unlink()
    annotation_archive.unlink()
    print("VoxConverse preparation is complete", flush=True)


if __name__ == "__main__":
    main()
