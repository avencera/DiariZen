#!/usr/bin/env python3
"""Prepare a deterministic AMI single distant microphone training pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import wave
from dataclasses import asdict, dataclass
from pathlib import Path


AMI_PREFIXES = ("EN", "ES", "IB", "IN", "IS", "TS")
AMI_MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"


@dataclass(frozen=True)
class Recording:
    """A validated recording interval from an AMI UEM file."""

    session: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        """Return the selected interval duration in seconds."""
        return self.end - self.start


def parse_uem(path: Path) -> list[Recording]:
    """Parse AMI recording intervals from a UEM file."""
    recordings = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or not fields[0].startswith(AMI_PREFIXES):
            continue
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: expected four UEM fields")

        recording = Recording(fields[0], float(fields[2]), float(fields[3]))
        if recording.start < 0 or recording.end <= recording.start:
            raise ValueError(f"{path}:{line_number}: invalid recording interval")
        recordings.append(recording)

    if not recordings:
        raise ValueError(f"No AMI recordings found in {path}")
    return recordings


def select_hours(recordings: list[Recording], target_hours: float) -> list[Recording]:
    """Select recordings in manifest order until the target duration is met."""
    if target_hours <= 0:
        raise ValueError("Target hours must be positive")

    selected = []
    duration = 0.0
    for recording in recordings:
        selected.append(recording)
        duration += recording.duration
        if duration >= target_hours * 3600:
            return selected

    raise ValueError(f"Requested {target_hours} hours, but only {duration / 3600:.3f} hours are available")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_wav(path: Path) -> None:
    """Validate the audio properties required by the recipe."""
    with wave.open(path.as_posix(), "rb") as source:
        if source.getframerate() != 16_000:
            raise ValueError(f"{path}: expected 16 kHz audio")
        if source.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono audio")
        if source.getnframes() == 0:
            raise ValueError(f"{path}: audio file has no frames")


def download_recording(recording: Recording, audio_dir: Path) -> tuple[Path, str]:
    """Download one AMI Array1 channel with an atomic final rename."""
    filename = f"{recording.session}.Array1-01.wav"
    destination = audio_dir / filename
    if destination.exists():
        validate_wav(destination)
        return destination, sha256(destination)

    url = f"{AMI_MIRROR}/{recording.session}/audio/{filename}"
    temporary = destination.with_suffix(".wav.part")
    temporary.unlink(missing_ok=True)
    print(f"Downloading {recording.session} from {url}", flush=True)
    with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)

    validate_wav(temporary)
    temporary.replace(destination)
    return destination, sha256(destination)


def filter_annotations(source: Path, destination: Path, sessions: set[str], session_field: int) -> None:
    """Copy annotation rows for the selected sessions."""
    selected_lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[session_field] in sessions:
            selected_lines.append(line)

    if not selected_lines:
        raise ValueError(f"No selected annotations found in {source}")
    destination.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")


def prepare_split(source_dir: Path, output_dir: Path, recordings: list[Recording]) -> dict[str, object]:
    """Prepare one dataset split and return its manifest data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    files = []
    scp_lines = []
    for recording in recordings:
        audio_path, digest = download_recording(recording, audio_dir)
        scp_lines.append(f"{recording.session} {audio_path.resolve()}")
        files.append({**asdict(recording), "duration": recording.duration, "sha256": digest})

    sessions = {recording.session for recording in recordings}
    (output_dir / "wav.scp").write_text("\n".join(scp_lines) + "\n", encoding="utf-8")
    filter_annotations(source_dir / "all.uem", output_dir / "all.uem", sessions, 0)
    filter_annotations(source_dir / "rttm", output_dir / "rttm", sessions, 1)

    return {
        "hours": sum(recording.duration for recording in recordings) / 3600,
        "recordings": files,
    }


def main() -> None:
    """Prepare the training subset and complete AMI development split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hours", type=float, default=14.4)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4"
    output_root = Path(__file__).resolve().parent / "data/ami_sdm_pilot"

    train = select_hours(parse_uem(source_root / "train/all.uem"), args.train_hours)
    dev = parse_uem(source_root / "dev/all.uem")
    manifest = {
        "condition": "AMI single distant microphone Array1-01",
        "license": "CC BY 4.0",
        "source": AMI_MIRROR,
        "train": prepare_split(source_root / "train", output_root / "train", train),
        "dev": prepare_split(source_root / "dev", output_root / "dev", dev),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": manifest_path.as_posix(),
                "train_hours": manifest["train"]["hours"],
                "dev_hours": manifest["dev"]["hours"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
