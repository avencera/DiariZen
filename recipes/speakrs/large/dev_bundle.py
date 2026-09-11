"""Build the exact frozen development bundle for the four-source run."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import soundfile as sf

from .contracts import DataPreparationSpec
from .errors import PreparationError
from .hashing import sha256_file
from .jsonio import write_json


ESTABLISHED_SOURCES = ("AMI", "AliMeeting")
AISHELL5_SOURCE = "AISHELL-5"


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
    return dev


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


def _copy_audio(source: Path, destination: Path, expected_digest: str) -> None:
    """Publish one verified audio object through an atomic local path."""

    if destination.is_file() and sha256_file(destination) == expected_digest:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    with source.open("rb") as source_handle, temporary.open("wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    if sha256_file(temporary) != expected_digest:
        temporary.unlink(missing_ok=True)
        raise PreparationError("development audio changed while it was copied", {"path": str(source)})
    temporary.replace(destination)


def build_dev_bundle(
    spec: DataPreparationSpec,
    audio_root: Path,
    established_dev: Path,
    aishell5_dev: Path,
    wav_prefix: str,
    output: Path,
) -> dict[str, object]:
    """Build trainer inputs for all frozen development recordings."""

    if not wav_prefix.strip():
        raise PreparationError("development bundle requires a non-empty trainer-relative wav prefix")
    output = Path(output)
    if output.exists():
        raise PreparationError("development bundle output already exists", {"path": str(output)})
    dev_ids = _frozen_dev_ids(spec)
    established_ids = set(dev_ids["AMI"]) | set(dev_ids["AliMeeting"])
    established_rttm, established_uem = _established_labels(Path(established_dev), established_ids)
    partial = output.with_name(f".{output.name}.partial")
    partial.mkdir(parents=True, exist_ok=True)
    recordings: list[dict[str, object]] = []
    wav_rows: list[str] = []
    rttm_rows: list[str] = []
    uem_rows: list[str] = []
    seen_ids: set[str] = set()
    for source, recording_ids in dev_ids.items():
        for recording_id in recording_ids:
            if recording_id in seen_ids:
                raise PreparationError("development recording ids are not globally unique", {"id": recording_id})
            seen_ids.add(recording_id)
            if source in ESTABLISHED_SOURCES:
                source_audio = Path(audio_root) / source / f"{recording_id}.flac"
                source_label = Path(established_dev) / "rttm"
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
                raise PreparationError("development audio must be non-empty mono 16 kHz", {"path": str(source_audio)})
            duration = audio_info.frames / audio_info.samplerate
            for row in rows:
                fields = row.split()
                if float(fields[3]) + float(fields[4]) > duration + 0.25:
                    raise PreparationError("development RTTM exceeds audio duration", {"recording_id": recording_id})
            if uem_row is not None:
                uem_fields = uem_row.split()
                if float(uem_fields[2]) < 0 or float(uem_fields[3]) > duration + 0.25:
                    raise PreparationError(
                        "established development UEM exceeds audio duration", {"recording_id": recording_id}
                    )
            audio_digest = sha256_file(source_audio)
            label_digest = sha256_file(source_label)
            suffix = source_audio.suffix.lower()
            destination = partial / "audio" / source / f"{audio_digest}{suffix}"
            _copy_audio(source_audio, destination, audio_digest)
            relative = destination.relative_to(partial)
            wav_rows.append(f"{recording_id} {(Path(wav_prefix) / relative).as_posix()}\n")
            rttm_rows.extend(rows)
            uem_rows.append(uem_row or f"{recording_id} 1 0.000000 {duration:.6f}\n")
            recordings.append(
                {
                    "recording_id": recording_id,
                    "source": source,
                    "audio_path": relative.as_posix(),
                    "audio_sha256": audio_digest,
                    "audio_size": source_audio.stat().st_size,
                    "frames": audio_info.frames,
                    "sample_rate": audio_info.samplerate,
                    "label_sha256": label_digest,
                }
            )
    manifest_paths = {"wav_scp": partial / "wav.scp", "rttm": partial / "all.rttm", "uem": partial / "all.uem"}
    manifest_paths["wav_scp"].write_text("".join(wav_rows), encoding="utf-8")
    manifest_paths["rttm"].write_text("".join(rttm_rows), encoding="utf-8")
    manifest_paths["uem"].write_text("".join(uem_rows), encoding="utf-8")
    manifest = {
        "schema": "speakrs-frozen-dev-bundle-v1",
        "frozen_splits": {source: list(ids) for source, ids in dev_ids.items()},
        "wav_prefix": wav_prefix,
        "manifests": {name: sha256_file(path) for name, path in manifest_paths.items()},
        "inputs": {
            "established_rttm_sha256": sha256_file(Path(established_dev) / "rttm"),
            "established_uem_sha256": sha256_file(Path(established_dev) / "all.uem"),
        },
        "recordings": recordings,
    }
    write_json(partial / "bundle.json", manifest)
    partial.replace(output)
    return {
        "ok": True,
        "command": "dev-bundle",
        "bundle": str(output),
        "bundle_manifest": str(output / "bundle.json"),
        "bundle_manifest_sha256": sha256_file(output / "bundle.json"),
        "recordings": len(recordings),
        "sources": list(dev_ids),
    }
