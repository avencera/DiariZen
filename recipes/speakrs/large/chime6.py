"""Typed CHiME-6 transcript and canonical microphone contracts."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import soundfile as sf

from .acceptance import RttmInterval
from .errors import PreparationError
from .hashing import sha256_file
from .jsonio import atomic_write_text, read_json, write_json


class Chime6Role(str, Enum):
    """One immutable publisher session role."""

    TRAIN = "train"
    DEVELOPMENT = "dev"
    EVALUATION = "eval"


class Chime6PreparationState(str, Enum):
    """One explicit CHiME-6 source-release lifecycle state."""

    TRANSFERRING = "transferring"
    DOWNLOADED = "downloaded"
    CONVERTED = "converted"
    ACCEPTED = "accepted"
    SEALED = "sealed"
    RESTORED = "restored"


class Chime6ActivityKind(str, Enum):
    """The audited activity meaning of one publisher transcript row."""

    SPEECH = "speech"
    INAUDIBLE_SPEECH = "inaudible-speech"
    LAUGHTER = "laughter"
    NON_SPEECH_NOISE = "non-speech-noise"
    UNLABELED = "unlabeled"


@dataclass(frozen=True, slots=True)
class Chime6TagAudit:
    """Counts of transcript rows after explicit bracket-tag classification."""

    speech: int = 0
    inaudible_speech: int = 0
    laughter: int = 0
    non_speech_noise: int = 0
    unlabeled: int = 0

    @classmethod
    def from_kinds(cls, kinds: Sequence[Chime6ActivityKind]) -> Chime6TagAudit:
        """Count classified publisher rows without losing their semantics."""

        return cls(
            speech=sum(kind is Chime6ActivityKind.SPEECH for kind in kinds),
            inaudible_speech=sum(kind is Chime6ActivityKind.INAUDIBLE_SPEECH for kind in kinds),
            laughter=sum(kind is Chime6ActivityKind.LAUGHTER for kind in kinds),
            non_speech_noise=sum(kind is Chime6ActivityKind.NON_SPEECH_NOISE for kind in kinds),
            unlabeled=sum(kind is Chime6ActivityKind.UNLABELED for kind in kinds),
        )


@dataclass(frozen=True, slots=True)
class Chime6Transcript:
    """Validated speaker activity for one CHiME-6 recording and parent party."""

    session_id: str
    party_id: str
    role: Chime6Role
    intervals: tuple[RttmInterval, ...]
    speakers: frozenset[str]
    tag_audit: Chime6TagAudit


@dataclass(frozen=True, slots=True)
class Chime6CanonicalAudio:
    """One selected archive member and its canonical audio identity."""

    session_id: str
    archive_member: str
    archive_member_size: int
    publisher_md5: str
    path: Path
    sha256: str
    size: int
    sample_rate: int
    channels: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class Chime6CanonicalLabels:
    """Deterministic RTTM and sample-clock UEM for one train recording."""

    session_id: str
    party_id: str
    rttm_path: Path
    rttm_sha256: str
    uem_path: Path
    uem_sha256: str
    interval_count: int
    speaker_count: int
    sample_count: int
    sample_rate: int


CHIME6_TRAIN_SESSIONS = frozenset(
    {"S03", "S04", "S05", "S06", "S07", "S08", "S12", "S13", "S16", "S17", "S18", "S19", "S20", "S22", "S23", "S24"}
)
CHIME6_DEVELOPMENT_SESSIONS = frozenset({"S02", "S09"})
CHIME6_EVALUATION_SESSIONS = frozenset({"S01", "S21"})
CHIME6_CANONICAL_ARRAY = "U06"
CHIME6_CANONICAL_CHANNEL = "CH1"
CHIME6_CANONICAL_VIEW = f"{CHIME6_CANONICAL_ARRAY}.{CHIME6_CANONICAL_CHANNEL}"
CHIME6_TRAIN_ARCHIVE_SIZE = 97_238_876_482
CHIME6_TRAIN_U06_CH1_MD5 = {
    "S03": "586292fa6d390c5cfa9cead597ff831b",
    "S04": "380b98f1852b855dafae3615597898cf",
    "S05": "5fb5837b586203ad505f195bcbcf2b32",
    "S06": "dbb951c2d601f60c921fe8f2e7436c3f",
    "S07": "8e2ef1270232d0d217add595a0558c72",
    "S08": "0f36f963d383912adacf994175e718cc",
    "S12": "66a6e241fbab7d1877a59bbf0bea2399",
    "S13": "4f0a70717dae1cfc822585cd321fa9b7",
    "S16": "45ca51cc52b0ce03c7c2869ac013509c",
    "S17": "c03e1a3df921519866ae1f4cc95862f7",
    "S18": "f80f31de5c150a46fbedf19683150ecf",
    "S19": "6e96b8704ea7e1041ba0fc0a91a96fbd",
    "S20": "8fd47469e5597c079f3ed387da2267dc",
    "S22": "037bcb6ad90bd5fc7c6cc8ea6cf23986",
    "S23": "13ff91fc8fa1affa172c561b0f9216cb",
    "S24": "9c7efdf9788e052b78f7b622fdba67c6",
}
CHIME6_SESSION_SPEAKERS = {
    "S01": frozenset({"P01", "P02", "P03", "P04"}),
    "S02": frozenset({"P05", "P06", "P07", "P08"}),
    "S03": frozenset({"P09", "P10", "P11", "P12"}),
    "S04": frozenset({"P09", "P10", "P11", "P12"}),
    "S05": frozenset({"P13", "P14", "P15", "P16"}),
    "S06": frozenset({"P13", "P14", "P15", "P16"}),
    "S07": frozenset({"P17", "P18", "P19", "P20"}),
    "S08": frozenset({"P21", "P22", "P23", "P24"}),
    "S09": frozenset({"P25", "P26", "P27", "P28"}),
    "S12": frozenset({"P33", "P34", "P35", "P36"}),
    "S13": frozenset({"P33", "P34", "P35", "P36"}),
    "S16": frozenset({"P21", "P22", "P23", "P24"}),
    "S17": frozenset({"P17", "P18", "P19", "P20"}),
    "S18": frozenset({"P41", "P42", "P43", "P44"}),
    "S19": frozenset({"P49", "P50", "P51", "P52"}),
    "S20": frozenset({"P49", "P50", "P51", "P52"}),
    "S21": frozenset({"P45", "P46", "P47", "P48"}),
    "S22": frozenset({"P41", "P42", "P43", "P44"}),
    "S23": frozenset({"P53", "P54", "P55", "P56"}),
    "S24": frozenset({"P53", "P54", "P55", "P56"}),
}

_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)$")
_KNOWN_TAG_RE = re.compile(r"\[(noise|laughs|inaudible)(?:\s+[^\]]+)?\]", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"\[[^\]]+\]")
_SESSION_RE = re.compile(r"^S\d{2}$")
_SPEAKER_RE = re.compile(r"^P\d{2}$")
_TRAIN_ROW_FIELDS = frozenset({"end_time", "start_time", "words", "speaker", "session_id"})
_HELDOUT_ROW_FIELDS = _TRAIN_ROW_FIELDS | {"location", "ref"}


def chime6_role(session_id: str) -> Chime6Role:
    """Return the immutable publisher role for a known CHiME-6 session."""

    if session_id in CHIME6_TRAIN_SESSIONS:
        return Chime6Role.TRAIN
    if session_id in CHIME6_DEVELOPMENT_SESSIONS:
        return Chime6Role.DEVELOPMENT
    if session_id in CHIME6_EVALUATION_SESSIONS:
        return Chime6Role.EVALUATION

    raise PreparationError("unknown CHiME-6 publisher session", {"session_id": session_id})


def chime6_party_id(session_id: str) -> str:
    """Return one stable parent identity for sessions with the same people."""

    chime6_role(session_id)
    speakers = CHIME6_SESSION_SPEAKERS[session_id]

    return "party-" + "-".join(sorted(speakers))


def chime6_canonical_audio_member(session_id: str) -> str:
    """Return the selected synchronized single-channel archive member suffix."""

    if chime6_role(session_id) is not Chime6Role.TRAIN:
        raise PreparationError(
            "only CHiME-6 train sessions can enter the training release", {"session_id": session_id}
        )

    return f"audio/train/{session_id}_{CHIME6_CANONICAL_VIEW}.wav"


def parse_chime6_time(value: str) -> float:
    """Parse a publisher ``HH:MM:SS.ss`` timestamp as seconds."""

    if not isinstance(value, str):
        raise PreparationError("CHiME-6 timestamp must be a string")
    match = _TIME_RE.fullmatch(value)
    if match is None:
        raise PreparationError("CHiME-6 timestamp has an invalid form", {"timestamp": value})
    hours, minutes, seconds = (float(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise PreparationError("CHiME-6 timestamp is outside clock bounds", {"timestamp": value})

    return hours * 3600 + minutes * 60 + seconds


def classify_chime6_words(words: str) -> Chime6ActivityKind:
    """Classify speech, laughter, inaudible speech, and non-speech noise."""

    if not isinstance(words, str):
        raise PreparationError("CHiME-6 transcript words must be a string")
    if not words.strip():
        return Chime6ActivityKind.UNLABELED
    tags = tuple(match.group(1).lower() for match in _KNOWN_TAG_RE.finditer(words))
    residual = _KNOWN_TAG_RE.sub(" ", words)
    unknown = _ANY_TAG_RE.findall(residual)
    if unknown:
        raise PreparationError("CHiME-6 transcript contains an unknown bracket tag", {"tags": sorted(set(unknown))})
    if residual.strip():
        return Chime6ActivityKind.SPEECH
    if "laughs" in tags:
        return Chime6ActivityKind.LAUGHTER
    if "inaudible" in tags:
        return Chime6ActivityKind.INAUDIBLE_SPEECH
    if tags and set(tags) == {"noise"}:
        return Chime6ActivityKind.NON_SPEECH_NOISE

    raise PreparationError("CHiME-6 transcript row has no classified activity")


def parse_chime6_transcript(
    payload: Any,
    *,
    expected_session_id: str,
    expected_role: Chime6Role,
) -> Chime6Transcript:
    """Parse one transcript and enforce publisher role and activity semantics."""

    if chime6_role(expected_session_id) is not expected_role:
        raise PreparationError(
            "CHiME-6 transcript role does not match publisher membership",
            {"session_id": expected_session_id, "role": expected_role.value},
        )
    if not isinstance(payload, list) or not payload:
        raise PreparationError("CHiME-6 transcript must be a non-empty array")

    intervals = []
    kinds = []
    row_speakers = set()
    row_fields = _TRAIN_ROW_FIELDS if expected_role is Chime6Role.TRAIN else _HELDOUT_ROW_FIELDS
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping) or set(raw) != row_fields:
            raise PreparationError("CHiME-6 transcript row has invalid fields", {"row": index})
        session_id = raw["session_id"]
        speaker = raw["speaker"]
        if session_id != expected_session_id or _SESSION_RE.fullmatch(str(session_id)) is None:
            raise PreparationError("CHiME-6 transcript row has the wrong session", {"row": index})
        if not isinstance(speaker, str) or _SPEAKER_RE.fullmatch(speaker) is None:
            raise PreparationError("CHiME-6 transcript row has an invalid speaker", {"row": index})
        row_speakers.add(speaker)
        start = parse_chime6_time(raw["start_time"])
        end = parse_chime6_time(raw["end_time"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise PreparationError("CHiME-6 transcript row has invalid bounds", {"row": index})
        kind = classify_chime6_words(raw["words"])
        kinds.append(kind)
        if kind in {Chime6ActivityKind.NON_SPEECH_NOISE, Chime6ActivityKind.UNLABELED}:
            continue
        intervals.append(RttmInterval(expected_session_id, start, end, speaker))

    expected_speakers = CHIME6_SESSION_SPEAKERS[expected_session_id]
    if row_speakers != expected_speakers:
        raise PreparationError(
            "CHiME-6 transcript speaker set differs from its frozen party",
            {"session_id": expected_session_id},
        )

    return Chime6Transcript(
        session_id=expected_session_id,
        party_id=chime6_party_id(expected_session_id),
        role=expected_role,
        intervals=tuple(intervals),
        speakers=expected_speakers,
        tag_audit=Chime6TagAudit.from_kinds(kinds),
    )


def _copy_member_with_md5(source: Any, target: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with target.open("xb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            output.write(block)

    return digest.hexdigest()


def _convert_to_canonical_flac(source: Path, target: Path, minimum_free_bytes: int) -> tuple[str, int]:
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        "-compression_level",
        "5",
        "-n",
        str(target),
    ]
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode != 0:
        raise PreparationError(
            "CHiME-6 canonical conversion failed",
            {"stderr": process.stderr.decode("utf-8", errors="replace")[-2000:]},
        )
    if not target.is_file() or target.stat().st_size <= 0:
        raise PreparationError("CHiME-6 canonical conversion produced no output")
    if shutil.disk_usage(target.parent).free < minimum_free_bytes:
        raise PreparationError("CHiME-6 conversion crossed the free-space reserve")

    return sha256_file(target), target.stat().st_size


def _audio_receipt(identity: Chime6CanonicalAudio) -> dict[str, object]:
    return {
        "schema": "speakrs-chime6-canonical-audio-v1",
        "session_id": identity.session_id,
        "archive_member": identity.archive_member,
        "archive_member_size": identity.archive_member_size,
        "publisher_md5": identity.publisher_md5,
        "path": str(identity.path),
        "sha256": identity.sha256,
        "size": identity.size,
        "sample_rate": identity.sample_rate,
        "channels": identity.channels,
        "sample_count": identity.sample_count,
    }


def _load_existing_audio(
    destination: Path,
    *,
    session_id: str,
    archive_member: str,
    archive_member_size: int,
) -> Chime6CanonicalAudio | None:
    final_path = destination / f"{session_id}.flac"
    receipt_path = destination / "receipts" / f"{session_id}.json"
    if not receipt_path.exists():
        if final_path.exists():
            final_path.unlink()

        return None
    if not final_path.is_file():
        raise PreparationError("CHiME-6 receipt has no canonical audio", {"session_id": session_id})
    raw = read_json(receipt_path)
    expected_fields = {
        "schema",
        "session_id",
        "archive_member",
        "archive_member_size",
        "publisher_md5",
        "path",
        "sha256",
        "size",
        "sample_rate",
        "channels",
        "sample_count",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise PreparationError("CHiME-6 audio receipt has invalid fields", {"session_id": session_id})
    identity = Chime6CanonicalAudio(
        session_id=str(raw["session_id"]),
        archive_member=str(raw["archive_member"]),
        archive_member_size=int(raw["archive_member_size"]),
        publisher_md5=str(raw["publisher_md5"]),
        path=Path(str(raw["path"])),
        sha256=str(raw["sha256"]),
        size=int(raw["size"]),
        sample_rate=int(raw["sample_rate"]),
        channels=int(raw["channels"]),
        sample_count=int(raw["sample_count"]),
    )
    audio_info = sf.info(final_path)
    if (
        raw["schema"] != "speakrs-chime6-canonical-audio-v1"
        or identity.session_id != session_id
        or identity.archive_member != archive_member
        or identity.archive_member_size != archive_member_size
        or identity.publisher_md5 != CHIME6_TRAIN_U06_CH1_MD5[session_id]
        or identity.path != final_path
        or identity.size != final_path.stat().st_size
        or sha256_file(final_path) != identity.sha256
        or identity.sample_rate != 16_000
        or identity.channels != 1
        or identity.sample_count <= 0
        or identity.sample_count > identity.archive_member_size
        or audio_info.samplerate != identity.sample_rate
        or audio_info.channels != identity.channels
        or audio_info.frames != identity.sample_count
        or re.fullmatch(r"[0-9a-f]{64}", identity.sha256) is None
    ):
        raise PreparationError("CHiME-6 audio receipt does not match its output", {"session_id": session_id})

    return identity


def prepare_chime6_train_audio(
    archive: Path,
    destination: Path,
    *,
    minimum_free_bytes: int = 100 * 1024**3,
) -> tuple[Chime6CanonicalAudio, ...]:
    """Stream one archive pass and retain only verified U06.CH1 train audio.

    Each selected WAV is verified against the synchronization publisher MD5
    while it leaves the archive. The final FLAC SHA-256 is calculated in the
    conversion output stream. At most one raw selected WAV exists at a time.
    """

    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes must be non-negative")
    if not archive.is_file():
        raise PreparationError("CHiME-6 train archive is missing", {"path": str(archive)})
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "receipts").mkdir(exist_ok=True)
    selected: dict[str, Chime6CanonicalAudio] = {}

    with tarfile.open(archive, mode="r|gz") as source_archive:
        for member in source_archive:
            matching = [
                session_id
                for session_id in CHIME6_TRAIN_SESSIONS
                if member.name.endswith(chime6_canonical_audio_member(session_id))
            ]
            if not matching:
                continue
            session_id = matching[0]
            if session_id in selected:
                raise PreparationError("CHiME-6 archive repeats a canonical member", {"session_id": session_id})
            if not member.isfile() or member.size <= 0:
                raise PreparationError("CHiME-6 canonical member is not a non-empty file", {"member": member.name})
            existing = _load_existing_audio(
                destination,
                session_id=session_id,
                archive_member=member.name,
                archive_member_size=member.size,
            )
            if existing is not None:
                selected[session_id] = existing
                continue
            if shutil.disk_usage(destination).free - member.size < minimum_free_bytes:
                raise PreparationError(
                    "CHiME-6 extraction would cross the free-space reserve", {"member": member.name}
                )
            extracted = source_archive.extractfile(member)
            if extracted is None:
                raise PreparationError("CHiME-6 canonical member could not be read", {"member": member.name})

            raw_partial = destination / f".{session_id}.source.partial.wav"
            final_partial = destination / f".{session_id}.canonical.partial.flac"
            final_path = destination / f"{session_id}.flac"
            raw_partial.unlink(missing_ok=True)
            final_partial.unlink(missing_ok=True)
            try:
                with extracted:
                    actual_md5 = _copy_member_with_md5(extracted, raw_partial)
                expected_md5 = CHIME6_TRAIN_U06_CH1_MD5[session_id]
                if actual_md5 != expected_md5:
                    raise PreparationError(
                        "CHiME-6 source member MD5 does not match the publisher identity",
                        {"session_id": session_id, "expected": expected_md5, "actual": actual_md5},
                    )
                sha256, size = _convert_to_canonical_flac(raw_partial, final_partial, minimum_free_bytes)
                os.replace(final_partial, final_path)
            finally:
                raw_partial.unlink(missing_ok=True)
                final_partial.unlink(missing_ok=True)

            audio = sf.info(final_path)
            if audio.samplerate != 16000 or audio.channels != 1 or audio.frames <= 0 or audio.frames > member.size:
                raise PreparationError(
                    "CHiME-6 canonical audio has the wrong decoded shape", {"session_id": session_id}
                )
            identity = Chime6CanonicalAudio(
                session_id=session_id,
                archive_member=member.name,
                archive_member_size=member.size,
                publisher_md5=actual_md5,
                path=final_path,
                sha256=sha256,
                size=size,
                sample_rate=audio.samplerate,
                channels=audio.channels,
                sample_count=audio.frames,
            )
            write_json(destination / "receipts" / f"{session_id}.json", _audio_receipt(identity))
            selected[session_id] = identity

    missing = sorted(CHIME6_TRAIN_SESSIONS - set(selected))
    if missing:
        raise PreparationError("CHiME-6 archive is missing canonical train members", {"missing": missing})

    return tuple(selected[session_id] for session_id in sorted(selected))


def _rttm_number(value: float) -> str:
    text = f"{value:.9f}".rstrip("0").rstrip(".")

    return text or "0"


def _merge_same_speaker_intervals(
    transcript: Chime6Transcript,
    *,
    duration: float,
) -> tuple[RttmInterval, ...]:
    by_speaker: dict[str, list[RttmInterval]] = {}
    for interval in transcript.intervals:
        if interval.start < 0 or interval.end > duration or interval.end <= interval.start:
            raise PreparationError(
                "CHiME-6 activity is outside the canonical sample clock",
                {"session_id": transcript.session_id, "speaker": interval.speaker},
            )
        by_speaker.setdefault(interval.speaker, []).append(interval)

    merged = []
    for speaker, intervals in sorted(by_speaker.items()):
        current: RttmInterval | None = None
        for interval in sorted(intervals, key=lambda item: (item.start, item.end)):
            if current is None:
                current = interval
                continue
            if interval.start <= current.end:
                current = RttmInterval(
                    transcript.session_id,
                    current.start,
                    max(current.end, interval.end),
                    speaker,
                )
                continue
            merged.append(current)
            current = interval
        if current is not None:
            merged.append(current)

    return tuple(sorted(merged, key=lambda item: (item.start, item.end, item.speaker)))


def prepare_chime6_train_labels(
    transcripts: Sequence[Chime6Transcript],
    audio: Sequence[Chime6CanonicalAudio],
    destination: Path,
) -> tuple[Chime6CanonicalLabels, ...]:
    """Write train RTTM and UEM labels bound to decoded canonical audio."""

    transcript_by_session = {item.session_id: item for item in transcripts}
    audio_by_session = {item.session_id: item for item in audio}
    if len(transcript_by_session) != len(transcripts) or len(audio_by_session) != len(audio):
        raise PreparationError("CHiME-6 label inputs repeat a session")
    if set(transcript_by_session) != CHIME6_TRAIN_SESSIONS:
        raise PreparationError("CHiME-6 labels require every frozen train transcript")
    if set(audio_by_session) != CHIME6_TRAIN_SESSIONS:
        raise PreparationError("CHiME-6 labels require every canonical train audio object")
    if destination.exists():
        raise PreparationError("CHiME-6 label destination already exists", {"path": str(destination)})

    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise PreparationError("stale CHiME-6 label destination exists", {"path": str(partial)})
    partial.mkdir(parents=True)
    labels = []
    try:
        for session_id in sorted(CHIME6_TRAIN_SESSIONS):
            transcript = transcript_by_session[session_id]
            canonical_audio = audio_by_session[session_id]
            if transcript.role is not Chime6Role.TRAIN or transcript.party_id != chime6_party_id(session_id):
                raise PreparationError("CHiME-6 transcript has an invalid train identity", {"session_id": session_id})
            if (
                canonical_audio.sample_rate != 16_000
                or canonical_audio.channels != 1
                or canonical_audio.sample_count <= 0
            ):
                raise PreparationError("CHiME-6 audio has an invalid canonical clock", {"session_id": session_id})

            duration = canonical_audio.sample_count / canonical_audio.sample_rate
            intervals = _merge_same_speaker_intervals(transcript, duration=duration)
            rttm_text = "".join(
                "SPEAKER "
                f"{session_id} 1 {_rttm_number(interval.start)} "
                f"{_rttm_number(interval.end - interval.start)} <NA> <NA> "
                f"{interval.speaker} <NA> <NA>\n"
                for interval in intervals
            )
            uem_text = f"{session_id} 1 0 {_rttm_number(duration)}\n"
            rttm_path = partial / f"{session_id}.rttm"
            uem_path = partial / f"{session_id}.uem"
            atomic_write_text(rttm_path, rttm_text)
            atomic_write_text(uem_path, uem_text)
            labels.append(
                Chime6CanonicalLabels(
                    session_id=session_id,
                    party_id=transcript.party_id,
                    rttm_path=destination / rttm_path.name,
                    rttm_sha256=hashlib.sha256(rttm_text.encode()).hexdigest(),
                    uem_path=destination / uem_path.name,
                    uem_sha256=hashlib.sha256(uem_text.encode()).hexdigest(),
                    interval_count=len(intervals),
                    speaker_count=len(transcript.speakers),
                    sample_count=canonical_audio.sample_count,
                    sample_rate=canonical_audio.sample_rate,
                )
            )
        partial.replace(destination)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    return tuple(labels)
