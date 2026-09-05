"""Pinned public sources for the large_cc_v1 release. Hashes are sealed after bytes exist."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    """One remote source. expected_sha256 is None until the bytes are verified."""

    name: str
    corpus: str
    url: str
    licence: str
    licence_url: str
    expected_sha256: str | None
    notes: str
    estimated_gib: float


AMI_AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{session_id}/audio/{session_id}.Array1-01.wav"
)
ICSI_MIX_URL = (
    "https://groups.inf.ed.ac.uk/ami/ICSIcorpusMirror/icsicorpus/{session_id}/audio/{session_id}.Mix-Headset.wav"
)
ICSI_CHANNEL_URL = (
    "https://groups.inf.ed.ac.uk/ami/ICSIcorpusMirror/icsicorpus/{session_id}/audio/{session_id}.{channel}.sph"
)
ICSI_ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip"
AISHELL_ARCHIVE_URLS = {
    "train_L": "https://www.openslr.org/resources/111/train_L.tar.gz",
    "train_M": "https://www.openslr.org/resources/111/train_M.tar.gz",
    "train_S": "https://www.openslr.org/resources/111/train_S.tar.gz",
    "test": "https://www.openslr.org/resources/111/test.tar.gz",
}
ALI_ARCHIVE_URLS = {
    "train": "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz",
    "dev": "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz",
    "test": "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz",
}
VOXCONVERSE_COMMIT = "24bf60be297701cd7e4ef18550c6d390c1b87365"
VOXCONVERSE_ANNOTATION_URL = f"https://github.com/joonson/voxconverse/archive/{VOXCONVERSE_COMMIT}.tar.gz"
VOXCONVERSE_ANNOTATION_SHA256 = "e8c25c91b014657d7e4ad86f9bef4a7eb399929d8d4fab910d8e6c6ab63d1197"
VOXCONVERSE_DEV_AUDIO_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
VOXCONVERSE_DEV_AUDIO_SHA256 = "e83a68b5df3bc945a3cf4544102038792ae79972753c585769e58ea677c523a8"
VOXCONVERSE_TEST_AUDIO_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
VOXCONVERSE_TEST_AUDIO_SHA256 = "472ebf1eaeb1dcb5c311b07a8b5c31bcedcccbf98f386d90a88cde2452da8c68"
NOTSOFAR_HF_DATASET = "microsoft/NOTSOFAR"
NOTSOFAR_REAL_TRAIN = "benchmark-datasets/train_set/240825.1_train"
NOTSOFAR_REAL_DEV1 = "benchmark-datasets/dev_set/240825.1_dev1"
NOTSOFAR_REAL_EVAL = "benchmark-datasets/eval_set/240825.1_eval_full_with_GT"
NOTSOFAR_SIM_PREFIX = "https://notsofarsa.blob.core.windows.net/css-datasets/v1.5/1000hrs/train"
LOTUSDIS_FULL_MEETING_ID = "1ofw99Y5W1p8f1DSaIbJkS0xWtuTI2Hrc"
LOTUSDIS_TEXTGRID_ID = "14fMv_X_8sGDPGbnU-hpJ85Mug43AHlgO"
LOTUSDIS_CSV_ID = "1ut44pgT1tJRd30clNp-IPx6nJiW7co-z"
LOTUSDIS_FULL_MEETING_URL = f"https://drive.google.com/uc?id={LOTUSDIS_FULL_MEETING_ID}&export=download"
LOTUSDIS_TEXTGRID_URL = f"https://drive.google.com/uc?id={LOTUSDIS_TEXTGRID_ID}&export=download"
LOTUSDIS_CSV_URL = f"https://drive.google.com/uc?id={LOTUSDIS_CSV_ID}&export=download"

SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "ami-array1-01",
        "AMI",
        AMI_AUDIO_URL,
        "CC BY 4.0",
        "https://groups.inf.ed.ac.uk/ami/",
        None,
        "Array1 microphone 1 per published session",
        12.0,
    ),
    SourceSpec(
        "alimeeting-far",
        "AliMeeting",
        ALI_ARCHIVE_URLS["train"],
        "CC BY-SA 4.0",
        "https://www.openslr.org/119/",
        None,
        "First far-field channel; OpenSLR 119 archives",
        18.0,
    ),
    SourceSpec(
        "aishell4-openslr111",
        "AISHELL4",
        AISHELL_ARCHIVE_URLS["train_L"],
        "CC BY-SA 4.0",
        "https://www.openslr.org/111/",
        None,
        "Equal mean of 8 channels; reject old downmix sidecars",
        20.0,
    ),
    SourceSpec(
        "voxconverse-dev",
        "VoxConverse",
        VOXCONVERSE_DEV_AUDIO_URL,
        "CC BY 4.0 with source-video rights retained",
        "https://github.com/joonson/voxconverse",
        VOXCONVERSE_DEV_AUDIO_SHA256,
        "Approved 216 published-dev recordings as training",
        6.0,
    ),
    SourceSpec(
        "voxconverse-test",
        "VoxConverse",
        VOXCONVERSE_TEST_AUDIO_URL,
        "CC BY 4.0 with source-video rights retained",
        "https://github.com/joonson/voxconverse",
        VOXCONVERSE_TEST_AUDIO_SHA256,
        "232 published-test recordings remain test",
        6.0,
    ),
    SourceSpec(
        "notsofar-real-240825.1",
        "NOTSOFAR_real",
        f"https://huggingface.co/datasets/{NOTSOFAR_HF_DATASET}",
        "CC BY 4.0",
        "https://github.com/microsoft/NOTSOFAR1-Challenge",
        None,
        "240825.1_train, 240825.1_dev1, 240825.1_eval_full_with_GT; exclude Dev2",
        40.0,
    ),
    SourceSpec(
        "notsofar-sim-v1.5-1000hrs-train",
        "NOTSOFAR_sim",
        NOTSOFAR_SIM_PREFIX,
        "CC BY 4.0",
        "https://github.com/microsoft/NOTSOFAR1-Challenge",
        None,
        "v1.5 1000hrs train only; val stays out; labels from source activity",
        120.0,
    ),
    SourceSpec(
        "icsi-cc-mirror",
        "ICSI",
        ICSI_ANNOTATIONS_URL,
        "CC BY 4.0",
        "https://groups.inf.ed.ac.uk/ami/icsi/license.shtml",
        None,
        "Human labels; distant view or equal-gain close mix if only close channels exist",
        20.0,
    ),
    SourceSpec(
        "lotusdis-full-meeting",
        "LOTUSDIS",
        LOTUSDIS_FULL_MEETING_URL,
        "CC BY-SA 4.0",
        "https://github.com/CAI-NECTEC/LOTUSDIS",
        None,
        "Full meeting audio plus TextGrid/CSV; not utterance-only archives",
        54.0,
    ),
)

PEAK_GIB_ESTIMATE = 300.0
PREPARED_GIB_ESTIMATE = 220.0
