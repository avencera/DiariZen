"""Pure membership, member selection, and ID mapping for tight-label close-talk."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


CLOSE_TALK_SOURCES = (
    "aishell-5",
    "chime-6",
    "icsi",
    "lotusdis",
    "notsofar-real",
)

CHIME6_TRAIN_SESSIONS = frozenset(
    {
        "S03",
        "S04",
        "S05",
        "S06",
        "S07",
        "S08",
        "S12",
        "S13",
        "S16",
        "S17",
        "S18",
        "S19",
        "S20",
        "S22",
        "S23",
        "S24",
    }
)
CHIME6_SESSION_SPEAKERS = {
    "S03": frozenset({"P09", "P10", "P11", "P12"}),
    "S04": frozenset({"P09", "P10", "P11", "P12"}),
    "S05": frozenset({"P13", "P14", "P15", "P16"}),
    "S06": frozenset({"P13", "P14", "P15", "P16"}),
    "S07": frozenset({"P17", "P18", "P19", "P20"}),
    "S08": frozenset({"P21", "P22", "P23", "P24"}),
    "S12": frozenset({"P33", "P34", "P35", "P36"}),
    "S13": frozenset({"P33", "P34", "P35", "P36"}),
    "S16": frozenset({"P21", "P22", "P23", "P24"}),
    "S17": frozenset({"P17", "P18", "P19", "P20"}),
    "S18": frozenset({"P41", "P42", "P43", "P44"}),
    "S19": frozenset({"P49", "P50", "P51", "P52"}),
    "S20": frozenset({"P49", "P50", "P51", "P52"}),
    "S22": frozenset({"P41", "P42", "P43", "P44"}),
    "S23": frozenset({"P53", "P54", "P55", "P56"}),
    "S24": frozenset({"P53", "P54", "P55", "P56"}),
}

_RECORDING_ID_RE = re.compile(r"^([a-z0-9-]+)::(.+)$")
_CHIME6_WORN_RE = re.compile(r"(?:^|/)audio/train/(S\d{2})_(P\d{2})\.wav$")
_AISHELL5_DA_RE = re.compile(r"^train/(\d{3})/(DA0[1-4])\.wav$")
_ICSI_CHANNEL_RE = re.compile(r"^c([0-9A-Fa-f])$")
_LOTUSDIS_LAV_RE = re.compile(r"^(?:.+/)?(Hijack_S\d{3}_T\d{3})_Lav([123])\.wav$")
_NOTSOFAR_CT_RE = re.compile(r"^CT_\d+$")


@dataclass(frozen=True, slots=True)
class CloseTalkMember:
    """One archive or remote close-talk file bound to a training recording."""

    source: str
    recording_id: str
    speaker: str | None
    member: str
    local_name: str
    extra: Mapping[str, object]


def split_recording_id(recording_id: str) -> tuple[str, str]:
    """Split ``source::local-id`` training identifiers."""

    if not isinstance(recording_id, str):
        raise ValueError("recording_id must be a string")
    match = _RECORDING_ID_RE.fullmatch(recording_id)
    if match is None:
        raise ValueError(f"recording_id is not a source-prefixed identity: {recording_id}")
    return match.group(1), match.group(2)


def training_union_ids(manifest: Mapping[str, object]) -> dict[str, frozenset[str]]:
    """Group current training-view recording IDs by source prefix."""

    recordings = manifest.get("recordings")
    if not isinstance(recordings, list):
        raise ValueError("training manifest must contain a recordings array")
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in recordings:
        if not isinstance(row, Mapping):
            raise ValueError("training manifest row must be an object")
        recording_id = row.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError("training manifest row is missing recording_id")
        source, _local = split_recording_id(recording_id)
        grouped[source].add(recording_id)
    return {source: frozenset(ids) for source, ids in grouped.items()}


def close_talk_training_ids(manifest: Mapping[str, object]) -> dict[str, frozenset[str]]:
    """Return training IDs for sources that need close-talk alignment audio."""

    grouped = training_union_ids(manifest)
    selected = {}
    for source in CLOSE_TALK_SOURCES:
        ids = grouped.get(source)
        if not ids:
            raise ValueError(f"training manifest has no recordings for {source}")
        selected[source] = ids
    return selected


def duration_seconds_by_recording(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Map training recording IDs to mix duration in seconds."""

    durations: dict[str, float] = {}
    for row in records:
        recording_id = row["recording_id"]
        samples = int(row["duration_samples"])
        rate = int(row.get("sample_rate_hz") or 16_000)
        if samples <= 0 or rate <= 0:
            raise ValueError(f"invalid duration for {recording_id}")
        durations[str(recording_id)] = samples / rate
    return durations


def duration_within_one_second(actual: float, expected: float) -> bool:
    """Return whether decoded duration matches the training mix within 1 s."""

    return abs(actual - expected) <= 1.0


def chime6_worn_member_suffix(session_id: str, speaker: str) -> str:
    """Return the publisher binaural member suffix for one worn microphone."""

    if session_id not in CHIME6_TRAIN_SESSIONS:
        raise ValueError(f"session {session_id} is not a CHiME-6 train session")
    if speaker not in CHIME6_SESSION_SPEAKERS[session_id]:
        raise ValueError(f"speaker {speaker} is not in CHiME-6 session {session_id}")
    return f"audio/train/{session_id}_{speaker}.wav"


def parse_chime6_worn_member(member: str) -> tuple[str, str] | None:
    """Return ``(session, speaker)`` for a binaural train member, else none."""

    match = _CHIME6_WORN_RE.search(member.replace("\\", "/"))
    if match is None:
        return None
    session_id, speaker = match.group(1), match.group(2)
    if session_id not in CHIME6_TRAIN_SESSIONS:
        return None
    if speaker not in CHIME6_SESSION_SPEAKERS[session_id]:
        return None
    return session_id, speaker


def select_chime6_worn_members(training_ids: Iterable[str]) -> tuple[CloseTalkMember, ...]:
    """Select one binaural file per training-session speaker."""

    selected: list[CloseTalkMember] = []
    for recording_id in sorted(training_ids):
        source, session_id = split_recording_id(recording_id)
        if source != "chime-6":
            raise ValueError(f"CHiME-6 selector received {recording_id}")
        if session_id not in CHIME6_TRAIN_SESSIONS:
            raise ValueError(f"{recording_id} is not a CHiME-6 train session")
        for speaker in sorted(CHIME6_SESSION_SPEAKERS[session_id]):
            suffix = chime6_worn_member_suffix(session_id, speaker)
            selected.append(
                CloseTalkMember(
                    source="chime-6",
                    recording_id=recording_id,
                    speaker=speaker,
                    member=suffix,
                    local_name=f"{session_id}_{speaker}.wav",
                    extra={"session_id": session_id},
                )
            )
    return tuple(selected)


def aishell5_folder_to_recording_id(folder: str) -> str:
    """Map an archive folder number such as ``001`` to a training recording ID."""

    if not isinstance(folder, str) or re.fullmatch(r"\d{1,3}", folder) is None:
        raise ValueError(f"AISHELL-5 folder is invalid: {folder}")
    index = int(folder)
    if index < 1 or index > 568:
        raise ValueError(f"AISHELL-5 folder is outside the training range: {folder}")
    return f"aishell-5::train-{index:03d}"


def parse_aishell5_da_member(member: str) -> tuple[str, str] | None:
    """Return ``(recording_id, stem)`` for a near-field headset WAV member."""

    match = _AISHELL5_DA_RE.fullmatch(member.replace("\\", "/"))
    if match is None:
        return None
    return aishell5_folder_to_recording_id(match.group(1)), match.group(2)


def aishell5_zone_speakers(zone_map: Mapping[str, object]) -> dict[int, dict[int, str]]:
    """Map session number to seat/zone number to speaker ID."""

    sessions = zone_map.get("sessions") if isinstance(zone_map.get("sessions"), Mapping) else zone_map
    if not isinstance(sessions, Mapping):
        raise ValueError("AISHELL-5 zone map must contain sessions")
    mapped: dict[int, dict[int, str]] = {}
    for raw_session, people in sessions.items():
        session_index = int(str(raw_session))
        seats: dict[int, str] = {}
        if not isinstance(people, Sequence):
            raise ValueError(f"AISHELL-5 session {raw_session} has no speaker list")
        for person in people:
            if not isinstance(person, Mapping):
                raise ValueError(f"AISHELL-5 session {raw_session} has an invalid speaker row")
            speaker = str(person["speaker_id"])
            zone = int(person["zone"])
            if zone not in {1, 2, 3, 4}:
                raise ValueError(f"AISHELL-5 zone {zone} is not a headset seat")
            seats[zone] = speaker
        mapped[session_index] = seats
    return mapped


def select_aishell5_headset_members(
    members: Sequence[Mapping[str, object]],
    training_ids: Iterable[str],
    zone_map: Mapping[str, object],
) -> tuple[CloseTalkMember, ...]:
    """Select ``DA0{{seat}}.wav`` members for training folders."""

    wanted = frozenset(training_ids)
    seats_by_session = aishell5_zone_speakers(zone_map)
    selected: list[CloseTalkMember] = []
    seen: set[str] = set()
    for row in members:
        name = str(row.get("name", ""))
        parsed = parse_aishell5_da_member(name)
        if parsed is None:
            continue
        recording_id, stem = parsed
        if recording_id not in wanted:
            continue
        if name in seen:
            raise ValueError(f"AISHELL-5 member is repeated: {name}")
        seen.add(name)
        _source, local_id = split_recording_id(recording_id)
        session_index = int(local_id.split("-")[1])
        seat = int(stem[-1])
        speaker = seats_by_session.get(session_index, {}).get(seat)
        selected.append(
            CloseTalkMember(
                source="aishell-5",
                recording_id=recording_id,
                speaker=speaker,
                member=name,
                local_name=f"{local_id}/{stem}.wav",
                extra={
                    "folder": f"{session_index:03d}",
                    "seat": seat,
                    "size": int(row["size"]) if "size" in row else None,
                },
            )
        )
    return tuple(sorted(selected, key=lambda item: (item.recording_id, item.member)))


def aishell5_missing_headsets(
    selected: Sequence[CloseTalkMember],
    training_ids: Iterable[str],
    zone_map: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Record zone-map speakers whose headset WAV is absent from the archive."""

    present: dict[str, set[int]] = defaultdict(set)
    for item in selected:
        present[item.recording_id].add(int(item.extra["seat"]))
    missing: list[dict[str, object]] = []
    seats_by_session = aishell5_zone_speakers(zone_map)
    for recording_id in sorted(training_ids):
        _source, local_id = split_recording_id(recording_id)
        session_index = int(local_id.split("-")[1])
        for seat, speaker in sorted(seats_by_session.get(session_index, {}).items()):
            if seat not in present.get(recording_id, set()):
                missing.append(
                    {
                        "recording_id": recording_id,
                        "speaker": speaker,
                        "seat": seat,
                        "member": f"train/{session_index:03d}/DA0{seat}.wav",
                    }
                )
    return tuple(missing)


def icsi_channel_filename(channel: str) -> str | None:
    """Map an NXT ``channel="c0"`` attribute onto ``chan0.sph``."""

    if not isinstance(channel, str):
        return None
    match = _ICSI_CHANNEL_RE.fullmatch(channel.strip())
    if match is None:
        return None
    token = match.group(1)
    if token.isalpha():
        token = token.upper()
    return f"chan{token}.sph"


def parse_icsi_dialogue_acts(xml_text: str) -> tuple[tuple[str, str], ...]:
    """Return ``(channel_filename, speaker)`` pairs from one dialogue-act file."""

    root = ET.fromstring(xml_text)
    pairs: list[tuple[str, str]] = []
    for node in root.iter():
        if not node.tag.endswith("dialogueact"):
            continue
        channel = icsi_channel_filename(str(node.attrib.get("channel", "")))
        speaker = str(node.attrib.get("participant", "")).strip()
        if channel is None or not speaker:
            continue
        pairs.append((channel, speaker))
    return tuple(pairs)


def select_icsi_headset_channels(
    meeting_pairs: Mapping[str, Sequence[tuple[str, str]]],
    training_ids: Iterable[str],
) -> tuple[CloseTalkMember, ...]:
    """Choose one headset SPH per speaker by majority NXT channel vote."""

    wanted = {split_recording_id(recording_id)[1]: recording_id for recording_id in training_ids}
    selected: list[CloseTalkMember] = []
    for meeting_id, recording_id in sorted(wanted.items()):
        votes: dict[str, Counter[str]] = defaultdict(Counter)
        for channel, speaker in meeting_pairs.get(meeting_id, ()):
            votes[speaker][channel] += 1
        if not votes:
            continue
        for speaker in sorted(votes):
            channel, _count = votes[speaker].most_common(1)[0]
            selected.append(
                CloseTalkMember(
                    source="icsi",
                    recording_id=recording_id,
                    speaker=speaker,
                    member=f"{meeting_id}/{channel}",
                    local_name=f"{meeting_id}/{channel}",
                    extra={"meeting_id": meeting_id, "channel": channel},
                )
            )
    return tuple(selected)


def icsi_missing_meetings(
    selected: Sequence[CloseTalkMember],
    training_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return training meetings with no headset channel from NXT metadata."""

    present = {item.recording_id for item in selected}
    return tuple(sorted(recording_id for recording_id in training_ids if recording_id not in present))


def select_notsofar_close_talk(
    devices: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
    recording_id: str,
) -> tuple[CloseTalkMember, ...]:
    """Select ``close_talk/CT_*.wav`` files and map them onto participant aliases."""

    source, meeting_id = split_recording_id(recording_id)
    if source != "notsofar-real":
        raise ValueError(f"NOTSOFAR selector received {recording_id}")
    alias_to_device = metadata.get("ParticipantAliasToCtDevice")
    device_to_speaker: dict[str, str] = {}
    if isinstance(alias_to_device, Mapping):
        for alias, device in alias_to_device.items():
            device_to_speaker[str(device)] = str(alias)
    selected: list[CloseTalkMember] = []
    for row in devices:
        if not isinstance(row, Mapping) or row.get("is_close_talk") is not True:
            continue
        device_name = str(row.get("device_name", ""))
        if _NOTSOFAR_CT_RE.fullmatch(device_name) is None:
            continue
        raw_names = row.get("wav_file_names", f"close_talk/{device_name}.wav")
        names = [part.strip().replace("\\", "/") for part in str(raw_names).split(",") if part.strip()]
        for name in names:
            path = PurePosixPath(name)
            if path.name.startswith("._") or "__MACOSX" in path.parts:
                continue
            selected.append(
                CloseTalkMember(
                    source="notsofar-real",
                    recording_id=recording_id,
                    speaker=device_to_speaker.get(device_name),
                    member=f"{meeting_id}/{name}",
                    local_name=f"{meeting_id}/{path.name}",
                    extra={"meeting_id": meeting_id, "device_name": device_name, "repository_path": name},
                )
            )
    return tuple(selected)


def is_skipped_archive_path(member: str) -> bool:
    """Return whether a zip/tar path is an AppleDouble or macOS junk member."""

    path = PurePosixPath(member.replace("\\", "/"))
    if path.name.startswith("._") or path.name.startswith("."):
        return True
    return any(part == "__MACOSX" or part.startswith("._") for part in path.parts)


def parse_lotusdis_lavalier_member(member: str) -> tuple[str, str] | None:
    """Return ``(parent_id, LavN)`` for a lavalier WAV, else none."""

    if is_skipped_archive_path(member):
        return None
    path = PurePosixPath(member.replace("\\", "/"))
    match = _LOTUSDIS_LAV_RE.fullmatch(path.as_posix())
    if match is None:
        match = _LOTUSDIS_LAV_RE.fullmatch(path.name)
    if match is None:
        return None
    return match.group(1), f"Lav{match.group(2)}"


def select_lotusdis_lavalier_members(
    members: Iterable[str],
    training_ids: Iterable[str],
) -> tuple[CloseTalkMember, ...]:
    """Select Lav1–Lav3 files for training parents. Speaker remains unknown."""

    wanted = {split_recording_id(recording_id)[1]: recording_id for recording_id in training_ids}
    selected: list[CloseTalkMember] = []
    seen: set[str] = set()
    for member in members:
        parsed = parse_lotusdis_lavalier_member(member)
        if parsed is None:
            continue
        parent_id, lav = parsed
        recording_id = wanted.get(parent_id)
        if recording_id is None:
            continue
        if member in seen:
            continue
        seen.add(member)
        selected.append(
            CloseTalkMember(
                source="lotusdis",
                recording_id=recording_id,
                speaker=None,
                member=member.replace("\\", "/"),
                local_name=f"{parent_id}/{parent_id}_{lav}.wav",
                extra={"parent_id": parent_id, "channel": lav, "speaker_mapping": "unknown"},
            )
        )
    return tuple(sorted(selected, key=lambda item: (item.recording_id, item.member)))


def lotusdis_missing_lavaliers(
    selected: Sequence[CloseTalkMember],
    training_ids: Iterable[str],
) -> tuple[dict[str, object], ...]:
    """Record training parents that lack one or more Lav1–Lav3 files."""

    present: dict[str, set[str]] = defaultdict(set)
    for item in selected:
        present[item.recording_id].add(str(item.extra["channel"]))
    missing: list[dict[str, object]] = []
    for recording_id in sorted(training_ids):
        have = present.get(recording_id, set())
        for channel in ("Lav1", "Lav2", "Lav3"):
            if channel not in have:
                _source, parent_id = split_recording_id(recording_id)
                missing.append(
                    {
                        "recording_id": recording_id,
                        "channel": channel,
                        "member": f"wav/{parent_id}/{parent_id}_{channel}.wav",
                    }
                )
    return tuple(missing)


def expected_kept_count(source: str, recording_id: str, selected: Sequence[CloseTalkMember]) -> int:
    """Return the number of close-talk files selected for one recording."""

    return sum(item.recording_id == recording_id and item.source == source for item in selected)


def load_icsi_meeting_pairs_from_zip(archive: Path) -> dict[str, tuple[tuple[str, str], ...]]:
    """Read headset channel votes from ICSI NXT dialogue-act XML members."""

    import zipfile

    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            path = PurePosixPath(name)
            if path.parent.name != "DialogueActs" or not path.name.endswith(".dialogue-acts.xml"):
                continue
            meeting_id = path.name.split(".", 1)[0]
            xml_text = source.read(name).decode("latin-1")
            pairs[meeting_id].extend(parse_icsi_dialogue_acts(xml_text))
    return {meeting_id: tuple(items) for meeting_id, items in pairs.items()}
