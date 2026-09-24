"""Restartable close-talk and aligner-model acquisition for tight labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Mapping, Sequence

from .errors import PreparationError
from .hashing import sha256_file
from .jsonio import read_json, write_json
from .notsofar_real import NOTSOFAR_REAL_REVISION
from .tight_close_talk import (
    CLOSE_TALK_SOURCES,
    CloseTalkMember,
    aishell5_missing_headsets,
    close_talk_training_ids,
    duration_seconds_by_recording,
    duration_within_one_second,
    load_icsi_meeting_pairs_from_zip,
    parse_aishell5_da_member,
    parse_chime6_worn_member,
    select_aishell5_headset_members,
    select_chime6_worn_members,
    select_icsi_headset_channels,
    select_lotusdis_lavalier_members,
    select_notsofar_close_talk,
    split_recording_id,
)


DEFAULT_DV_ROOT = Path("/Volumes/CacheDisk/dev-cache/diarization-data-verification")
DEFAULT_STAGING_ROOT = DEFAULT_DV_ROOT / "staging" / "tight-labels-v1"
MINIMUM_FREE_BYTES = 120 * 1024**3
STREAM_BLOCK_BYTES = 1024 * 1024

CHIME6_TRAIN_URL = "https://www.openslr.org/resources/150/CHiME6_train.tar.gz"
CHIME6_TRAIN_BYTES = 97_238_876_482
CHIME6_TRAIN_SHA256 = "18c75141123233837b1fce20ede76e13737edfac30c5b31f86d4330b67c1827d"

AISHELL5_TRAIN_URL = "https://openslr.trmal.net/resources/159/train.tar.gz"
AISHELL5_TRAIN_URLS = (
    "https://openslr.trmal.net/resources/159/train.tar.gz",
    "https://www.openslr.org/resources/159/train.tar.gz",
    "https://openslr.elda.org/resources/159/train.tar.gz",
)
AISHELL5_TRAIN_BYTES = 54_598_541_447
AISHELL5_TRAIN_SHA256 = "3861f9422a6d4c91ea797a3eb3f720a33921566b40c68e32f68ba2aa240e5e3a"

ICSI_CHANNEL_URL = "https://groups.inf.ed.ac.uk/ami/ICSIsignals/SPH/{meeting}/{channel}"
ICSI_NXT_ZIP = Path("/Volumes/CacheDisk/dev-cache/wavlm-large-pre-rental/source/icsi/ICSI_core_NXT.zip")

NOTSOFAR_HF_DATASET = "microsoft/NOTSOFAR"
NOTSOFAR_TRAIN_PREFIX = "benchmark-datasets/train_set/240825.1_train/MTG"

LOTUSDIS_WAV_ZIP = Path("/Volumes/CacheDisk/dev-cache/wavlm-large-pre-rental/source/lotusdis/wav.zip")

MFA_MODELS: tuple[dict[str, str], ...] = (
    {
        "id": "acoustic-english_mfa-v3.1.0",
        "kind": "acoustic",
        "language": "english",
        "version": "3.1.0",
        "filename": "english_mfa.zip",
        "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_mfa-v3.1.0/english_mfa.zip",
        "license": "CC BY 4.0",
        "revision": "acoustic-english_mfa-v3.1.0",
    },
    {
        "id": "acoustic-mandarin_mfa-v3.0.0",
        "kind": "acoustic",
        "language": "mandarin",
        "version": "3.0.0",
        "filename": "mandarin_mfa.zip",
        "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-mandarin_mfa-v3.0.0/mandarin_mfa.zip",
        "license": "CC BY 4.0",
        "revision": "acoustic-mandarin_mfa-v3.0.0",
    },
    {
        "id": "acoustic-french_mfa-v3.0.0",
        "kind": "acoustic",
        "language": "french",
        "version": "3.0.0",
        "filename": "french_mfa.zip",
        "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-french_mfa-v3.0.0/french_mfa.zip",
        "license": "CC BY 4.0",
        "revision": "acoustic-french_mfa-v3.0.0",
    },
    {
        "id": "dictionary-english_us_mfa-v3.1.0",
        "kind": "dictionary",
        "language": "english_us",
        "version": "3.1.0",
        "filename": "english_us_mfa.dict",
        "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_us_mfa-v3.1.0/english_us_mfa.dict",
        "license": "CC BY 4.0",
        "revision": "dictionary-english_us_mfa-v3.1.0",
    },
    {
        "id": "dictionary-mandarin_china_mfa-v3.0.0",
        "kind": "dictionary",
        "language": "mandarin_china",
        "version": "3.0.0",
        "filename": "mandarin_china_mfa.dict",
        "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-mandarin_china_mfa-v3.0.0/mandarin_china_mfa.dict",
        "license": "CC BY 4.0",
        "revision": "dictionary-mandarin_china_mfa-v3.0.0",
    },
    {
        "id": "dictionary-french_mfa-v3.0.0",
        "kind": "dictionary",
        "language": "french",
        "version": "3.0.0",
        "filename": "french_mfa.dict",
        "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-french_mfa-v3.0.0/french_mfa.dict",
        "license": "CC BY 4.0",
        "revision": "dictionary-french_mfa-v3.0.0",
    },
    {
        "id": "g2p-english_us_mfa-v3.0.0",
        "kind": "g2p",
        "language": "english_us",
        "version": "3.0.0",
        "filename": "english_us_mfa.zip",
        "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/g2p-english_us_mfa-v3.0.0/english_us_mfa.zip",
        "license": "CC BY 4.0",
        "revision": "g2p-english_us_mfa-v3.0.0",
    },
    {
        "id": "g2p-mandarin_china_mfa-v3.0.0",
        "kind": "g2p",
        "language": "mandarin_china",
        "version": "3.0.0",
        "filename": "mandarin_china_mfa.zip",
        "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/g2p-mandarin_china_mfa-v3.0.0/mandarin_china_mfa.zip",
        "license": "CC BY 4.0",
        "revision": "g2p-mandarin_china_mfa-v3.0.0",
    },
    {
        "id": "g2p-french_mfa-v3.0.0",
        "kind": "g2p",
        "language": "french",
        "version": "3.0.0",
        "filename": "french_mfa.zip",
        "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/g2p-french_mfa-v3.0.0/french_mfa.zip",
        "license": "CC BY 4.0",
        "revision": "g2p-french_mfa-v3.0.0",
    },
)

MMS_FA_URL = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
MMS_FA_LICENSE = "CC-BY-NC 4.0"
PARAKEET_MLX_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
PARAKEET_MLX_REVISION = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
PARAKEET_LICENSE = "CC BY 4.0"


@dataclass(frozen=True, slots=True)
class KeptMember:
    """One extracted or downloaded close-talk file with content identity."""

    member: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class StreamExtractResult:
    """Outcome of one hashed tar.gz pass that kept selected members."""

    bytes_streamed: int
    stream_sha256: str
    kept: tuple[KeptMember, ...]


class HashingReader(io.RawIOBase):
    """Hash every compressed byte while a tar.gz stream is consumed."""

    def __init__(self, source: BinaryIO) -> None:
        super().__init__()
        self._source = source
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:  # noqa: A003 - IO API
        data = self._source.read() if size is None or size < 0 else self._source.read(size)
        if data:
            self._digest.update(data)
            self.bytes_read += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        return n

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def drain(self) -> None:
        while self.read(STREAM_BLOCK_BYTES):
            pass


def require_free_space(path: Path, extra_bytes: int = 0, minimum_free_bytes: int = MINIMUM_FREE_BYTES) -> None:
    """Refuse a write that would leave CacheDisk below the reserve."""

    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free - extra_bytes < minimum_free_bytes:
        raise PreparationError(
            "write would cross the free-space reserve",
            {"path": str(path), "free_bytes": free, "extra_bytes": extra_bytes, "reserve": minimum_free_bytes},
        )


def _match_wanted_member(member_name: str, wanted: Mapping[str, Path]) -> Path | None:
    normalized = member_name.replace("\\", "/")
    for suffix, destination in wanted.items():
        if normalized == suffix or normalized.endswith("/" + suffix) or normalized.endswith(suffix):
            return destination
    return None


def _copy_hashed(source: BinaryIO, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        while True:
            block = source.read(STREAM_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            output.write(block)
            size += len(block)
        output.flush()
        os.fsync(output.fileno())
    return size, digest.hexdigest()


def extract_selected_tar_gz_members(
    stream: BinaryIO,
    destination: Path,
    wanted: Mapping[str, Path],
    *,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
    progress: Callable[[int], None] | None = None,
) -> StreamExtractResult:
    """Stream one gzip tar, hash the compressed bytes, and keep selected members."""

    destination.mkdir(parents=True, exist_ok=True)
    reader = HashingReader(stream)
    kept: dict[str, KeptMember] = {}
    last_report = 0
    try:
        with tarfile.open(fileobj=io.BufferedReader(reader), mode="r|gz") as archive:
            for member in archive:
                if progress is not None and reader.bytes_read - last_report >= 1024**3:
                    progress(reader.bytes_read)
                    last_report = reader.bytes_read
                target = _match_wanted_member(member.name, wanted)
                if target is None or not member.isfile() or member.size <= 0:
                    continue
                if target.exists() and target.stat().st_size == member.size:
                    kept[member.name] = KeptMember(member.name, target, member.size, sha256_file(target))
                    continue
                require_free_space(destination, member.size, minimum_free_bytes)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise PreparationError("archive member could not be read", {"member": member.name})
                partial = target.with_name(target.name + ".partial")
                partial.unlink(missing_ok=True)
                try:
                    with extracted:
                        size, digest = _copy_hashed(extracted, partial)
                    if size != member.size:
                        raise PreparationError(
                            "extracted member size does not match the tar header",
                            {"member": member.name, "header": member.size, "actual": size},
                        )
                    os.replace(partial, target)
                finally:
                    partial.unlink(missing_ok=True)
                kept[member.name] = KeptMember(member.name, target, size, digest)
    except (OSError, EOFError, tarfile.ReadError, tarfile.CompressionError) as error:
        raise PreparationError(
            "gzip tar stream is unreadable",
            {"error": str(error), "bytes_read": reader.bytes_read},
        ) from error
    reader.drain()
    return StreamExtractResult(reader.bytes_read, reader.hexdigest(), tuple(kept[name] for name in sorted(kept)))


def curl_stream(url: str) -> subprocess.Popen[bytes]:
    """Start one retrying download to stdout."""

    return subprocess.Popen(
        [
            "curl",
            "--http1.1",
            "--location",
            "--fail",
            "--retry",
            "8",
            "--retry-delay",
            "8",
            "--retry-all-errors",
            "--speed-limit",
            "10000",
            "--speed-time",
            "60",
            "--show-error",
            "--silent",
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def download_url_resumable(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> KeptMember:
    """Download one file with curl resume, then hash it."""

    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        return KeptMember(destination.name, destination, destination.stat().st_size, sha256_file(destination))
    require_free_space(destination.parent, expected_size or 0, minimum_free_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    command = [
        "curl",
        "--location",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "8",
        "--retry-all-errors",
        "--show-error",
        "--silent",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise PreparationError(
            "download failed",
            {"path": str(destination), "stderr": completed.stderr.decode("utf-8", errors="replace")[-2000:]},
        )
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise PreparationError(
            "downloaded size does not match Content-Length",
            {"path": str(destination), "expected": expected_size, "actual": partial.stat().st_size},
        )
    digest = sha256_file(partial)
    os.replace(partial, destination)
    return KeptMember(destination.name, destination, destination.stat().st_size, digest)


def audio_duration_seconds(path: Path) -> float:
    """Decode duration for WAV, FLAC, or SPHERE files."""

    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0:
            raise PreparationError("audio has no duration", {"path": str(path)})
        return info.frames / info.samplerate
    except (OSError, RuntimeError, ImportError):
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            raise PreparationError(
                "audio does not decode",
                {"path": str(path), "stderr": completed.stderr[-2000:]},
            ) from None
        try:
            duration = float(completed.stdout.strip())
        except ValueError as error:
            raise PreparationError("audio duration is not numeric", {"path": str(path)}) from error
        if duration <= 0:
            raise PreparationError("audio has no duration", {"path": str(path)})
        return duration


def _component_records(dv_root: Path, component_id: str) -> list[dict[str, object]]:
    path = (
        dv_root
        / "assets"
        / "commercial-data-preparation-v1"
        / "final-union-training-v1"
        / "components"
        / f"{component_id}.json"
    )
    payload = read_json(path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise PreparationError("component records are missing", {"path": str(path)})
    return records


def load_training_context(dv_root: Path) -> tuple[dict[str, frozenset[str]], dict[str, dict[str, float]]]:
    """Load training-union membership and mix durations for close-talk sources."""

    manifest_path = (
        dv_root / "assets" / "commercial-data-preparation-v1" / "final-union-training-v1" / "training-manifest.json"
    )
    union_path = dv_root / "next-run" / "indic-diarbench-union-v1" / "release.json"
    manifest = read_json(manifest_path)
    union = read_json(union_path)
    grouped = close_talk_training_ids(manifest)
    required = union.get("required_sources")
    if not isinstance(required, list):
        raise PreparationError("Indic union release is missing required_sources")
    expected_names = {
        "aishell-5": "AISHELL-5",
        "chime-6": "CHiME-6",
        "icsi": "ICSI",
        "lotusdis": "LOTUSDIS",
        "notsofar-real": "NOTSOFAR-real",
    }
    for source, label in expected_names.items():
        if label not in required:
            raise PreparationError("Indic union is missing a close-talk source", {"source": label})
        if source not in grouped:
            raise PreparationError("training view is missing a close-talk source", {"source": source})
    durations = {
        source: duration_seconds_by_recording(_component_records(dv_root, source)) for source in CLOSE_TALK_SOURCES
    }
    return grouped, durations


def _kept_payload(
    member: CloseTalkMember,
    kept: KeptMember,
    mix_duration: float | None,
) -> dict[str, object]:
    duration = audio_duration_seconds(kept.path)
    delta = None if mix_duration is None else duration - mix_duration
    return {
        "member": member.member,
        "path": str(kept.path),
        "size": kept.size,
        "sha256": kept.sha256,
        "recording_id": member.recording_id,
        "speaker": member.speaker,
        "duration_seconds": duration,
        "training_mix_duration_seconds": mix_duration,
        "duration_delta_seconds": delta,
        "duration_within_1s": None if mix_duration is None else duration_within_one_second(duration, mix_duration),
        "extra": dict(member.extra),
    }


def write_source_receipt(path: Path, payload: Mapping[str, object]) -> None:
    write_json(path, payload)


def _receipt_status(payload: Mapping[str, object]) -> str:
    if payload.get("stream_sha256_match") is False:
        return "failed"
    kept = payload.get("kept_members")
    if not isinstance(kept, list) or not kept:
        return "failed"
    decode_fail = any(not isinstance(item, Mapping) or not item.get("duration_seconds") for item in kept)
    if decode_fail:
        return "failed"
    duration_mismatch = any(item.get("duration_within_1s") is False for item in kept if isinstance(item, Mapping))
    if payload.get("missing_audio") or duration_mismatch:
        return "partial"
    return "ok"


def acquire_chime6(
    dv_root: Path,
    staging_root: Path,
    training_ids: Iterable[str],
    mix_durations: Mapping[str, float],
    *,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> dict[str, object]:
    """Stream the CHiME-6 train archive and keep worn binaural files."""

    started = time.time()
    selected = select_chime6_worn_members(training_ids)
    output = staging_root / "close-talk" / "chime-6"
    receipt_path = output / "receipt.json"
    wanted = {item.member: output / item.local_name for item in selected}
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        if (
            existing.get("stream_sha256") == CHIME6_TRAIN_SHA256
            and existing.get("stream_sha256_match") is True
            and all(path.is_file() for path in wanted.values())
        ):
            return existing
    require_free_space(output, 0, minimum_free_bytes)
    print(f"streaming CHiME-6 train archive ({CHIME6_TRAIN_BYTES} bytes)", flush=True)
    curl = curl_stream(CHIME6_TRAIN_URL)
    assert curl.stdout is not None
    try:
        result = extract_selected_tar_gz_members(
            curl.stdout,
            output,
            wanted,
            minimum_free_bytes=minimum_free_bytes,
            progress=lambda nbytes: print(f"  CHiME-6 streamed {nbytes / 1024**3:.2f} GiB", flush=True),
        )
    except BaseException:
        curl.terminate()
        curl.wait()
        raise
    finally:
        curl.stdout.close()
    stderr = curl.stderr.read().decode("utf-8", errors="replace") if curl.stderr is not None else ""
    if curl.wait() != 0:
        raise PreparationError("CHiME-6 stream failed", {"stderr": stderr[-2000:]})
    match = result.stream_sha256 == CHIME6_TRAIN_SHA256 and result.bytes_streamed == CHIME6_TRAIN_BYTES
    kept_by_suffix = {parse_chime6_worn_member(item.member): item for item in result.kept}
    kept_payload = []
    missing = []
    for member in selected:
        key = parse_chime6_worn_member(member.member)
        kept = kept_by_suffix.get(key)
        if kept is None:
            local = wanted[member.member]
            if local.is_file():
                kept = KeptMember(member.member, local, local.stat().st_size, sha256_file(local))
        if kept is None:
            missing.append({"recording_id": member.recording_id, "speaker": member.speaker, "member": member.member})
            continue
        kept_payload.append(_kept_payload(member, kept, mix_durations.get(member.recording_id)))
    payload = {
        "schema": "speakrs-tight-labels-close-talk-receipt-v1",
        "source": "chime-6",
        "source_url": CHIME6_TRAIN_URL,
        "bytes_streamed": result.bytes_streamed,
        "stream_sha256": result.stream_sha256,
        "expected_stream_sha256": CHIME6_TRAIN_SHA256,
        "stream_sha256_match": match,
        "kept_members": kept_payload,
        "missing_audio": missing,
        "elapsed_seconds": time.time() - started,
    }
    payload["status"] = _receipt_status(payload)
    write_source_receipt(receipt_path, payload)
    return payload


def acquire_aishell5(
    dv_root: Path,
    staging_root: Path,
    training_ids: Iterable[str],
    mix_durations: Mapping[str, float],
    *,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> dict[str, object]:
    """Stream the AISHELL-5 train archive and keep near-field headset WAVs."""

    started = time.time()
    pe = dv_root / "private-evidence" / "AISHELL-5-prefix"
    members = read_json(pe / "train-extract" / "members.json")
    zone_map = read_json(pe / "train-speaker-zone-map.json")
    selected = select_aishell5_headset_members(members, training_ids, zone_map)
    output = staging_root / "close-talk" / "aishell-5"
    receipt_path = output / "receipt.json"
    wanted = {item.member: output / item.local_name for item in selected}
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        if (
            existing.get("stream_sha256") == AISHELL5_TRAIN_SHA256
            and existing.get("stream_sha256_match") is True
            and all(path.is_file() for path in wanted.values())
        ):
            return existing
    require_free_space(output, 0, minimum_free_bytes)
    print(f"streaming AISHELL-5 train archive ({AISHELL5_TRAIN_BYTES} bytes)", flush=True)
    result = None
    last_error: Exception | None = None
    for attempt, url in enumerate(AISHELL5_TRAIN_URLS * 2, 1):
        print(f"  AISHELL-5 attempt {attempt} {url}", flush=True)
        curl = curl_stream(url)
        assert curl.stdout is not None
        try:
            result = extract_selected_tar_gz_members(
                curl.stdout,
                output,
                wanted,
                minimum_free_bytes=minimum_free_bytes,
                progress=lambda nbytes: print(f"  AISHELL-5 streamed {nbytes / 1024**3:.2f} GiB", flush=True),
            )
        except (PreparationError, OSError, tarfile.ReadError) as error:
            curl.terminate()
            curl.wait()
            last_error = error
            print(f"  AISHELL-5 attempt {attempt} failed: {error}", flush=True)
            continue
        finally:
            curl.stdout.close()
        stderr = curl.stderr.read().decode("utf-8", errors="replace") if curl.stderr is not None else ""
        if curl.wait() != 0:
            last_error = PreparationError("AISHELL-5 stream failed", {"stderr": stderr[-2000:], "url": url})
            print(f"  AISHELL-5 curl failed attempt {attempt}", flush=True)
            result = None
            continue
        break
    if result is None:
        if last_error is not None:
            raise last_error
        raise PreparationError("AISHELL-5 stream failed on every mirror")
    match = result.stream_sha256 == AISHELL5_TRAIN_SHA256 and result.bytes_streamed == AISHELL5_TRAIN_BYTES
    kept_by_member = {parse_aishell5_da_member(item.member): item for item in result.kept}
    kept_payload = []
    missing = list(aishell5_missing_headsets(selected, training_ids, zone_map))
    for member in selected:
        key = parse_aishell5_da_member(member.member)
        kept = kept_by_member.get(key)
        if kept is None:
            local = wanted[member.member]
            if local.is_file():
                kept = KeptMember(member.member, local, local.stat().st_size, sha256_file(local))
        if kept is None:
            missing.append({"recording_id": member.recording_id, "speaker": member.speaker, "member": member.member})
            continue
        kept_payload.append(_kept_payload(member, kept, mix_durations.get(member.recording_id)))
    mapping = {
        item["recording_id"] + "/" + str(item["extra"]["seat"]): {
            "folder": item["extra"]["folder"],
            "seat": item["extra"]["seat"],
            "speaker": item["speaker"],
            "member": item["member"],
        }
        for item in kept_payload
    }
    payload = {
        "schema": "speakrs-tight-labels-close-talk-receipt-v1",
        "source": "aishell-5",
        "source_url": AISHELL5_TRAIN_URL,
        "bytes_streamed": result.bytes_streamed,
        "stream_sha256": result.stream_sha256,
        "expected_stream_sha256": AISHELL5_TRAIN_SHA256,
        "stream_sha256_match": match,
        "folder_to_recording_id": {
            f"{int(split_recording_id(recording_id)[1].split('-')[1]):03d}": recording_id
            for recording_id in sorted(training_ids)
        },
        "seat_to_speaker": mapping,
        "kept_members": kept_payload,
        "missing_audio": missing,
        "elapsed_seconds": time.time() - started,
    }
    payload["status"] = _receipt_status(payload)
    write_source_receipt(receipt_path, payload)
    return payload


def acquire_icsi(
    dv_root: Path,
    staging_root: Path,
    training_ids: Iterable[str],
    mix_durations: Mapping[str, float],
    *,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> dict[str, object]:
    """Download ICSI headset SPH channels listed in NXT dialogue acts."""

    started = time.time()
    pairs = load_icsi_meeting_pairs_from_zip(ICSI_NXT_ZIP)
    selected = select_icsi_headset_channels(pairs, training_ids)
    output = staging_root / "close-talk" / "icsi"
    receipt_path = output / "receipt.json"
    kept_payload = []
    missing: list[dict[str, object]] = []
    bytes_streamed = 0
    for index, member in enumerate(selected, 1):
        destination = output / member.local_name
        url = ICSI_CHANNEL_URL.format(meeting=member.extra["meeting_id"], channel=member.extra["channel"])
        print(f"ICSI {index}/{len(selected)} {member.local_name}", flush=True)
        try:
            kept = download_url_resumable(url, destination, minimum_free_bytes=minimum_free_bytes)
        except PreparationError as error:
            missing.append(
                {
                    "recording_id": member.recording_id,
                    "speaker": member.speaker,
                    "member": member.member,
                    "error": error.message,
                }
            )
            continue
        bytes_streamed += kept.size
        kept_payload.append(_kept_payload(member, kept, mix_durations.get(member.recording_id)))
    present = {item["recording_id"] for item in kept_payload}
    for recording_id in sorted(training_ids):
        if recording_id not in present:
            missing.append({"recording_id": recording_id, "reason": "no headset channel in NXT metadata"})
    payload = {
        "schema": "speakrs-tight-labels-close-talk-receipt-v1",
        "source": "icsi",
        "source_url": ICSI_CHANNEL_URL,
        "bytes_streamed": bytes_streamed,
        "stream_sha256": None,
        "expected_stream_sha256": None,
        "stream_sha256_match": None,
        "nxt_zip": str(ICSI_NXT_ZIP),
        "kept_members": kept_payload,
        "missing_audio": missing,
        "elapsed_seconds": time.time() - started,
    }
    payload["status"] = _receipt_status(payload)
    write_source_receipt(receipt_path, payload)
    return payload


def acquire_notsofar(
    dv_root: Path,
    staging_root: Path,
    training_ids: Iterable[str],
    mix_durations: Mapping[str, float],
    *,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> dict[str, object]:
    """Fetch NOTSOFAR close-talk WAVs from the same Hugging Face revision."""

    started = time.time()
    source_root = dv_root / "staging" / "notsofar-real-source-v1" / NOTSOFAR_TRAIN_PREFIX
    selected: list[CloseTalkMember] = []
    for recording_id in sorted(training_ids):
        _source, meeting_id = split_recording_id(recording_id)
        meeting_dir = source_root / meeting_id
        devices = json.loads((meeting_dir / "devices.json").read_text(encoding="utf-8"))
        metadata = json.loads((meeting_dir / "gt_meeting_metadata.json").read_text(encoding="utf-8"))
        selected.extend(select_notsofar_close_talk(devices, metadata, recording_id))
    output = staging_root / "close-talk" / "notsofar-real"
    receipt_path = output / "receipt.json"
    kept_payload = []
    missing: list[dict[str, object]] = []
    bytes_streamed = 0
    for index, member in enumerate(selected, 1):
        destination = output / member.local_name
        repo_path = f"{NOTSOFAR_TRAIN_PREFIX}/{member.extra['meeting_id']}/{member.extra['repository_path']}"
        url = f"https://huggingface.co/datasets/{NOTSOFAR_HF_DATASET}/resolve/{NOTSOFAR_REAL_REVISION}/{repo_path}"
        print(f"NOTSOFAR {index}/{len(selected)} {member.local_name}", flush=True)
        try:
            kept = download_url_resumable(url, destination, minimum_free_bytes=minimum_free_bytes)
        except PreparationError as error:
            missing.append(
                {
                    "recording_id": member.recording_id,
                    "speaker": member.speaker,
                    "member": member.member,
                    "error": error.message,
                }
            )
            continue
        bytes_streamed += kept.size
        kept_payload.append(_kept_payload(member, kept, mix_durations.get(member.recording_id)))
    payload = {
        "schema": "speakrs-tight-labels-close-talk-receipt-v1",
        "source": "notsofar-real",
        "source_url": f"https://huggingface.co/datasets/{NOTSOFAR_HF_DATASET}",
        "revision": NOTSOFAR_REAL_REVISION,
        "version": "240825.1_train",
        "bytes_streamed": bytes_streamed,
        "stream_sha256": None,
        "expected_stream_sha256": None,
        "stream_sha256_match": None,
        "kept_members": kept_payload,
        "missing_audio": missing,
        "elapsed_seconds": time.time() - started,
    }
    payload["status"] = _receipt_status(payload)
    write_source_receipt(receipt_path, payload)
    return payload


def acquire_lotusdis(
    dv_root: Path,
    staging_root: Path,
    training_ids: Iterable[str],
    mix_durations: Mapping[str, float],
    *,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> dict[str, object]:
    """Extract Lav1–Lav3 WAVs from the local LOTUSDIS archive."""

    started = time.time()
    if not LOTUSDIS_WAV_ZIP.is_file():
        raise PreparationError("LOTUSDIS wav.zip is missing", {"path": str(LOTUSDIS_WAV_ZIP)})
    output = staging_root / "close-talk" / "lotusdis"
    receipt_path = output / "receipt.json"
    with zipfile.ZipFile(LOTUSDIS_WAV_ZIP) as archive:
        selected = select_lotusdis_lavalier_members(archive.namelist(), training_ids)
        wanted = {item.member: output / item.local_name for item in selected}
        kept_payload = []
        missing: list[dict[str, object]] = []
        bytes_streamed = 0
        for index, member in enumerate(selected, 1):
            destination = wanted[member.member]
            info = archive.getinfo(member.member)
            if destination.is_file() and destination.stat().st_size == info.file_size:
                kept = KeptMember(member.member, destination, info.file_size, sha256_file(destination))
            else:
                require_free_space(output, info.file_size, minimum_free_bytes)
                partial = destination.with_name(destination.name + ".partial")
                partial.unlink(missing_ok=True)
                print(f"LOTUSDIS {index}/{len(selected)} {member.local_name}", flush=True)
                with archive.open(member.member) as source:
                    size, digest = _copy_hashed(source, partial)
                os.replace(partial, destination)
                kept = KeptMember(member.member, destination, size, digest)
            bytes_streamed += kept.size
            kept_payload.append(_kept_payload(member, kept, mix_durations.get(member.recording_id)))
        present: dict[str, set[str]] = {}
        for item in kept_payload:
            present.setdefault(str(item["recording_id"]), set()).add(str(item["extra"]["channel"]))
        for recording_id in sorted(training_ids):
            have = present.get(recording_id, set())
            _source, parent_id = split_recording_id(recording_id)
            for channel in ("Lav1", "Lav2", "Lav3"):
                if channel not in have:
                    missing.append(
                        {
                            "recording_id": recording_id,
                            "channel": channel,
                            "member": f"wav/{parent_id}/{parent_id}_{channel}.wav",
                        }
                    )
    payload = {
        "schema": "speakrs-tight-labels-close-talk-receipt-v1",
        "source": "lotusdis",
        "source_url": str(LOTUSDIS_WAV_ZIP),
        "bytes_streamed": bytes_streamed,
        "stream_sha256": None,
        "expected_stream_sha256": None,
        "stream_sha256_match": None,
        "speaker_mapping": "unknown",
        "kept_members": kept_payload,
        "missing_audio": missing,
        "elapsed_seconds": time.time() - started,
    }
    payload["status"] = _receipt_status(payload)
    write_source_receipt(receipt_path, payload)
    return payload


def _huggingface_snapshot(repo_id: str, revision: str, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "hf",
            "download",
            repo_id,
            "--revision",
            revision,
            "--local-dir",
            str(destination),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise PreparationError(
            "Hugging Face snapshot failed",
            {"repo": repo_id, "stderr": (completed.stderr or "")[-2000:]},
        )
    return [path for path in destination.rglob("*") if path.is_file() and ".cache" not in path.parts]


def install_mfa(tools_root: Path) -> dict[str, object]:
    """Install MFA with kalpy from conda-forge into an isolated prefix."""

    tools_root.mkdir(parents=True, exist_ok=True)
    conda_prefix = tools_root / "mfa-conda"
    mfa_bin = conda_prefix / "bin" / "mfa"
    if mfa_bin.is_file():
        version_run = subprocess.run([str(mfa_bin), "version"], capture_output=True, check=False, text=True)
        if version_run.returncode == 0:
            return {
                "prefix": str(conda_prefix),
                "installer": "micromamba-conda-forge",
                "install_ok": True,
                "version_ok": True,
                "version": version_run.stdout.strip(),
            }
    micromamba_root = tools_root / "micromamba"
    micromamba = micromamba_root / "bin" / "micromamba"
    micromamba_root.mkdir(parents=True, exist_ok=True)
    if not micromamba.is_file():
        completed = subprocess.run(
            [
                "bash",
                "-lc",
                f"curl -L https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xjv -C {micromamba_root} bin/micromamba",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0 or not micromamba.is_file():
            raise PreparationError("micromamba download failed", {"stderr": completed.stderr[-2000:]})
    env = {**os.environ, "MAMBA_ROOT_PREFIX": str(micromamba_root)}
    create = subprocess.run(
        [
            str(micromamba),
            "create",
            "-y",
            "-p",
            str(conda_prefix),
            "-c",
            "conda-forge",
            "montreal-forced-aligner",
            "python=3.11",
        ],
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )
    if create.returncode != 0:
        raise PreparationError("MFA conda create failed", {"stderr": create.stderr[-4000:]})
    version_run = subprocess.run([str(mfa_bin), "version"], capture_output=True, check=False, text=True)
    return {
        "prefix": str(conda_prefix),
        "installer": "micromamba-conda-forge",
        "install_ok": create.returncode == 0,
        "install_stderr": create.stderr[-4000:],
        "version_ok": version_run.returncode == 0,
        "version": (version_run.stdout or version_run.stderr).strip(),
    }


def acquire_models(staging_root: Path, *, minimum_free_bytes: int = MINIMUM_FREE_BYTES) -> dict[str, object]:
    """Download pinned MFA, MMS, and Parakeet models and install MFA."""

    started = time.time()
    models_root = staging_root / "models"
    tools_root = staging_root / "tools"
    kept: list[dict[str, object]] = []
    bytes_streamed = 0
    mfa_root = models_root / "mfa"
    for spec in MFA_MODELS:
        destination = mfa_root / spec["kind"] / spec["filename"]
        print(f"model {spec['id']}", flush=True)
        downloaded = download_url_resumable(spec["url"], destination, minimum_free_bytes=minimum_free_bytes)
        bytes_streamed += downloaded.size
        kept.append(
            {
                "id": spec["id"],
                "kind": spec["kind"],
                "language": spec["language"],
                "version": spec["version"],
                "source_url": spec["url"],
                "revision": spec["revision"],
                "license": spec["license"],
                "path": str(downloaded.path),
                "size": downloaded.size,
                "sha256": downloaded.sha256,
            }
        )
    mms_path = models_root / "mms" / "ctc_alignment_mling_uroman.pt"
    print("model torchaudio MMS_FA", flush=True)
    mms = download_url_resumable(MMS_FA_URL, mms_path, minimum_free_bytes=minimum_free_bytes)
    bytes_streamed += mms.size
    kept.append(
        {
            "id": "torchaudio-mms-fa",
            "kind": "forced-alignment",
            "language": "multilingual-uroman",
            "version": "MMS_FA",
            "source_url": MMS_FA_URL,
            "revision": "torchaudio.pipelines.MMS_FA",
            "license": MMS_FA_LICENSE,
            "path": str(mms.path),
            "size": mms.size,
            "sha256": mms.sha256,
        }
    )
    parakeet_dir = models_root / "parakeet-tdt-0.6b-v3"
    print("model parakeet-mlx", flush=True)
    parakeet_files = _huggingface_snapshot(PARAKEET_MLX_REPO, PARAKEET_MLX_REVISION, parakeet_dir)
    parakeet_records = []
    for path in sorted(parakeet_files):
        digest = sha256_file(path)
        size = path.stat().st_size
        bytes_streamed += size
        parakeet_records.append(
            {
                "path": str(path),
                "relative": str(path.relative_to(parakeet_dir)),
                "size": size,
                "sha256": digest,
            }
        )
    kept.append(
        {
            "id": "parakeet-tdt-0.6b-v3-mlx",
            "kind": "asr",
            "language": "multilingual-european",
            "version": "0.6b-v3",
            "source_url": f"https://huggingface.co/{PARAKEET_MLX_REPO}",
            "revision": PARAKEET_MLX_REVISION,
            "license": PARAKEET_LICENSE,
            "files": parakeet_records,
        }
    )
    mfa_install = install_mfa(tools_root)
    payload = {
        "schema": "speakrs-tight-labels-models-receipt-v1",
        "source": "models",
        "bytes_streamed": bytes_streamed,
        "stream_sha256": None,
        "expected_stream_sha256": None,
        "stream_sha256_match": None,
        "kept_members": kept,
        "mfa_install": mfa_install,
        "missing_audio": [],
        "elapsed_seconds": time.time() - started,
    }
    payload["status"] = "ok" if mfa_install.get("version_ok") else "partial"
    write_source_receipt(models_root / "receipt.json", payload)
    return payload


def write_acquisition_report(
    staging_root: Path,
    receipts: Mapping[str, Mapping[str, object]],
    started: float,
) -> Path:
    """Write sizes, counts, gaps, elapsed time, and remaining free space."""

    lines = [
        "# Tight-label close-talk acquisition",
        "",
        f"Started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started))}",
        f"Finished: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"Elapsed seconds: {time.time() - started:.1f}",
        "",
        "| Source | Status | Kept files | Kept bytes | Missing | Stream SHA-256 match | Elapsed s |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for source, receipt in receipts.items():
        kept = receipt.get("kept_members") or []
        kept_bytes = 0
        if isinstance(kept, list):
            for item in kept:
                if isinstance(item, Mapping):
                    kept_bytes += int(item.get("size") or 0)
                    files = item.get("files")
                    if isinstance(files, list):
                        kept_bytes += sum(int(file.get("size") or 0) for file in files if isinstance(file, Mapping))
            kept_count = len(kept)
        else:
            kept_count = 0
        missing = receipt.get("missing_audio") or []
        missing_count = len(missing) if isinstance(missing, list) else 0
        match = receipt.get("stream_sha256_match")
        lines.append(
            f"| {source} | {receipt.get('status')} | {kept_count} | {kept_bytes} | {missing_count} | {match} | "
            f"{float(receipt.get('elapsed_seconds') or 0):.1f} |"
        )
    usage = shutil.disk_usage(staging_root)
    lines.extend(
        [
            "",
            "## Disk",
            "",
            f"CacheDisk free bytes: {usage.free}",
            f"CacheDisk free GiB: {usage.free / 1024**3:.2f}",
            f"Reserve GiB: {MINIMUM_FREE_BYTES / 1024**3:.0f}",
            "",
            "## Gaps",
            "",
        ]
    )
    for source, receipt in receipts.items():
        missing = receipt.get("missing_audio") or []
        if not missing:
            continue
        lines.append(f"### {source}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(missing, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    duration_fails = []
    for source, receipt in receipts.items():
        for item in receipt.get("kept_members") or []:
            if isinstance(item, Mapping) and item.get("duration_within_1s") is False:
                duration_fails.append(
                    {
                        "source": source,
                        **{
                            k: item.get(k)
                            for k in (
                                "recording_id",
                                "member",
                                "duration_seconds",
                                "training_mix_duration_seconds",
                                "duration_delta_seconds",
                            )
                        },
                    }
                )
    if duration_fails:
        lines.append("## Duration mismatches")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(duration_fails, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    report_path = staging_root / "ACQUISITION-REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


ACQUIRE_FNS = {
    "chime-6": acquire_chime6,
    "aishell-5": acquire_aishell5,
    "icsi": acquire_icsi,
    "notsofar-real": acquire_notsofar,
    "lotusdis": acquire_lotusdis,
}


def acquire(
    sources: Sequence[str], *, dv_root: Path = DEFAULT_DV_ROOT, staging_root: Path = DEFAULT_STAGING_ROOT
) -> dict[str, object]:
    """Run selected close-talk and model acquisitions."""

    started = time.time()
    require_free_space(staging_root, 0)
    training_ids, mix_durations = load_training_context(dv_root)
    selected = list(sources)
    if selected == ["all"]:
        selected = ["lotusdis", "models", "icsi", "notsofar-real", "chime-6", "aishell-5"]
    receipts: dict[str, dict[str, object]] = {}
    for source in selected:
        print(f"acquire {source}", flush=True)
        if source == "models":
            receipts[source] = acquire_models(staging_root)
            continue
        fn = ACQUIRE_FNS[source]
        receipts[source] = fn(dv_root, staging_root, training_ids[source], mix_durations[source])
        print(
            f"  status={receipts[source].get('status')} kept={len(receipts[source].get('kept_members') or [])}",
            flush=True,
        )
    report_path = write_acquisition_report(staging_root, receipts, started)
    return {
        "staging_root": str(staging_root),
        "report": str(report_path),
        "receipts": {
            source: str(staging_root / ("models" if source == "models" else f"close-talk/{source}") / "receipt.json")
            for source in receipts
        },
        "status": {source: receipt.get("status") for source, receipt in receipts.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire tight-label close-talk audio and aligner models")
    parser.add_argument(
        "--source",
        default="all",
        help="one of all, chime-6, aishell-5, icsi, notsofar-real, lotusdis, models",
    )
    parser.add_argument("--dv-root", type=Path, default=DEFAULT_DV_ROOT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sources = [part.strip() for part in str(args.source).split(",") if part.strip()]
    result = acquire(sources, dv_root=args.dv_root, staging_root=args.staging_root)
    print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
