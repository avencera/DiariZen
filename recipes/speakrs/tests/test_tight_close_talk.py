from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

from recipes.speakrs.large.tight_close_talk import (
    CHIME6_TRAIN_SESSIONS,
    aishell5_folder_to_recording_id,
    aishell5_missing_headsets,
    chime6_worn_member_suffix,
    close_talk_training_ids,
    duration_within_one_second,
    icsi_channel_filename,
    is_skipped_archive_path,
    parse_aishell5_da_member,
    parse_chime6_worn_member,
    parse_icsi_dialogue_acts,
    parse_lotusdis_lavalier_member,
    select_aishell5_headset_members,
    select_chime6_worn_members,
    select_icsi_headset_channels,
    select_lotusdis_lavalier_members,
    select_notsofar_close_talk,
    split_recording_id,
)
from recipes.speakrs.large.tight_labels_acquire import (
    HashingReader,
    extract_selected_tar_gz_members,
)


def test_training_union_keeps_only_close_talk_sources() -> None:
    manifest = {
        "recordings": [
            {"recording_id": "chime-6::S03"},
            {"recording_id": "aishell-5::train-001"},
            {"recording_id": "icsi::Bed002"},
            {"recording_id": "lotusdis::Hijack_S001_T057"},
            {"recording_id": "notsofar-real::MTG_30830"},
            {"recording_id": "ami::ES2002a"},
        ]
    }

    grouped = close_talk_training_ids(manifest)

    assert grouped["chime-6"] == frozenset({"chime-6::S03"})
    assert "ami" not in grouped
    assert split_recording_id("notsofar-real::MTG_30830") == ("notsofar-real", "MTG_30830")


def test_chime6_worn_members_are_binaural_train_files_only() -> None:
    selected = select_chime6_worn_members(["chime-6::S03", "chime-6::S04"])

    assert len(CHIME6_TRAIN_SESSIONS) == 16
    assert chime6_worn_member_suffix("S03", "P09") == "audio/train/S03_P09.wav"
    assert parse_chime6_worn_member("CHiME6_train/CHiME6/audio/train/S03_P09.wav") == ("S03", "P09")
    assert parse_chime6_worn_member("CHiME6_train/CHiME6/audio/train/S03_U06.CH1.wav") is None
    assert parse_chime6_worn_member("audio/train/S02_P05.wav") is None
    assert [item.local_name for item in selected] == [
        "S03_P09.wav",
        "S03_P10.wav",
        "S03_P11.wav",
        "S03_P12.wav",
        "S04_P09.wav",
        "S04_P10.wav",
        "S04_P11.wav",
        "S04_P12.wav",
    ]
    assert all(item.recording_id.startswith("chime-6::") for item in selected)


def test_aishell5_folder_and_da_seat_mapping() -> None:
    assert aishell5_folder_to_recording_id("1") == "aishell-5::train-001"
    assert aishell5_folder_to_recording_id("568") == "aishell-5::train-568"
    assert parse_aishell5_da_member("train/001/DA02.wav") == ("aishell-5::train-001", "DA02")
    assert parse_aishell5_da_member("train/001/DX01C01.wav") is None
    assert parse_aishell5_da_member("train/001/DA02.TextGrid") is None

    members = [
        {"name": "train/001/DA02.wav", "size": 10, "type": "file"},
        {"name": "train/001/DA03.wav", "size": 11, "type": "file"},
        {"name": "train/001/DX01C01.wav", "size": 12, "type": "file"},
        {"name": "train/002/DA04.wav", "size": 13, "type": "file"},
        {"name": "dev/001/DA02.wav", "size": 14, "type": "file"},
    ]
    zone_map = {
        "sessions": {
            "001": [{"speaker_id": "P0089", "zone": 2}, {"speaker_id": "P0090", "zone": 3}],
            "002": [
                {"speaker_id": "P0239", "zone": 4},
                {"speaker_id": "P0240", "zone": 1},
            ],
        }
    }
    training_ids = ["aishell-5::train-001", "aishell-5::train-002"]
    selected = select_aishell5_headset_members(members, training_ids, zone_map)

    assert [item.local_name for item in selected] == [
        "train-001/DA02.wav",
        "train-001/DA03.wav",
        "train-002/DA04.wav",
    ]
    assert selected[0].speaker == "P0089"
    assert selected[2].speaker == "P0239"
    missing = aishell5_missing_headsets(selected, training_ids, zone_map)
    assert missing == (
        {
            "recording_id": "aishell-5::train-002",
            "speaker": "P0240",
            "seat": 1,
            "member": "train/002/DA01.wav",
        },
    )


def test_icsi_channel_codes_map_onto_sph_filenames() -> None:
    assert icsi_channel_filename("c0") == "chan0.sph"
    assert icsi_channel_filename("cB") == "chanB.sph"
    assert icsi_channel_filename("cb") == "chanB.sph"
    assert icsi_channel_filename("2c0") is None
    xml_text = """<?xml version="1.0"?>
    <nite:root xmlns:nite="http://nite.sourceforge.net/">
      <dialogueact channel="c0" participant="me011"/>
      <dialogueact channel="c0" participant="me011"/>
      <dialogueact channel="c2" participant="me010"/>
      <dialogueact channel="2c0" participant="me999"/>
    </nite:root>
    """
    pairs = parse_icsi_dialogue_acts(xml_text)
    assert pairs == (("chan0.sph", "me011"), ("chan0.sph", "me011"), ("chan2.sph", "me010"))
    selected = select_icsi_headset_channels(
        {"Bed002": pairs, "Bmr003": (("chan1.sph", "me013"),)},
        ["icsi::Bed002"],
    )
    assert [(item.local_name, item.speaker) for item in selected] == [
        ("Bed002/chan2.sph", "me010"),
        ("Bed002/chan0.sph", "me011"),
    ]


def test_notsofar_close_talk_uses_devices_and_participant_aliases() -> None:
    devices = [
        {
            "device_name": "CT_21",
            "is_close_talk": True,
            "wav_file_names": "close_talk/CT_21.wav",
        },
        {
            "device_name": "plaza_0",
            "is_close_talk": False,
            "wav_file_names": "sc_plaza_0/ch0.wav",
        },
        {
            "device_name": "CT_22",
            "is_close_talk": True,
            "wav_file_names": "close_talk\\CT_22.wav",
        },
    ]
    metadata = {"ParticipantAliasToCtDevice": {"Peter": "CT_21", "Sophie": "CT_22"}}
    selected = select_notsofar_close_talk(devices, metadata, "notsofar-real::MTG_30830")
    assert [item.local_name for item in selected] == ["MTG_30830/CT_21.wav", "MTG_30830/CT_22.wav"]
    assert selected[0].speaker == "Peter"
    assert selected[1].speaker == "Sophie"


def test_lotusdis_keeps_lav1_to_lav3_and_skips_macos_junk() -> None:
    assert is_skipped_archive_path("__MACOSX/wav/Hijack_S001_T057/._Hijack_S001_T057_Lav1.wav")
    assert is_skipped_archive_path("wav/Hijack_S001_T057/._Hijack_S001_T057_Lav1.wav")
    assert parse_lotusdis_lavalier_member("wav/Hijack_S001_T057/Hijack_S001_T057_Lav1.wav") == (
        "Hijack_S001_T057",
        "Lav1",
    )
    assert parse_lotusdis_lavalier_member("wav/Hijack_S001_T057/Hijack_S001_T057_Lav123.wav") is None
    assert parse_lotusdis_lavalier_member("wav/Hijack_S001_T057/Hijack_S001_T057_Con123.wav") is None
    members = [
        "wav/Hijack_S001_T057/Hijack_S001_T057_Lav1.wav",
        "wav/Hijack_S001_T057/Hijack_S001_T057_Lav2.wav",
        "wav/Hijack_S001_T057/Hijack_S001_T057_Lav3.wav",
        "wav/Hijack_S001_T057/Hijack_S001_T057_Lav123.wav",
        "__MACOSX/wav/Hijack_S001_T057/._Hijack_S001_T057_Lav1.wav",
        "wav/Hijack_S002_T062/Hijack_S002_T062_Lav1.wav",
    ]
    selected = select_lotusdis_lavalier_members(members, ["lotusdis::Hijack_S001_T057"])
    assert [item.local_name for item in selected] == [
        "Hijack_S001_T057/Hijack_S001_T057_Lav1.wav",
        "Hijack_S001_T057/Hijack_S001_T057_Lav2.wav",
        "Hijack_S001_T057/Hijack_S001_T057_Lav3.wav",
    ]
    assert all(item.speaker is None for item in selected)
    assert all(item.extra["speaker_mapping"] == "unknown" for item in selected)


def test_duration_gate_accepts_one_second_window() -> None:
    assert duration_within_one_second(10.0, 10.4)
    assert duration_within_one_second(10.0, 9.0)
    assert not duration_within_one_second(10.0, 11.01)


def test_hashing_reader_and_selected_tar_gz_extract(tmp_path: Path) -> None:
    payload = b"worn-mic-bytes"
    other = b"array-mic-bytes"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        worn = tarfile.TarInfo(name="CHiME6_train/CHiME6/audio/train/S03_P09.wav")
        worn.size = len(payload)
        archive.addfile(worn, io.BytesIO(payload))
        skip = tarfile.TarInfo(name="CHiME6_train/CHiME6/audio/train/S03_U06.CH1.wav")
        skip.size = len(other)
        archive.addfile(skip, io.BytesIO(other))
    compressed = gzip.compress(buffer.getvalue(), mtime=0)
    destination = tmp_path / "kept"
    result = extract_selected_tar_gz_members(
        io.BytesIO(compressed),
        destination,
        wanted={
            "audio/train/S03_P09.wav": destination / "S03_P09.wav",
        },
        minimum_free_bytes=0,
    )

    assert (destination / "S03_P09.wav").read_bytes() == payload
    assert not (destination / "S03_U06.CH1.wav").exists()
    assert result.bytes_streamed == len(compressed)
    hasher = HashingReader(io.BytesIO(compressed))
    hasher.read()
    assert result.stream_sha256 == hasher.hexdigest()
    kept = result.kept[0]
    assert kept.member.endswith("audio/train/S03_P09.wav")
    assert kept.size == len(payload)
