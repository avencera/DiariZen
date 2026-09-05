#!/usr/bin/env python3

"""Prepare the full DiariZen meeting corpus without storing source archives."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
AMI_SESSION_PATTERN = re.compile(r"^(?:EN|ES|IB|IN|IS|TS)\d{4}[a-z]?$")
ALI_SESSION_PATTERN = re.compile(r"^R\d{4}_M\d{4}_MS\d{3}$")
AISHELL_SESSION_PATTERN = re.compile(r"^(?:\d{8}_)?[LMS]_R\d{3}S\d{2}C\d{2}$")
SENSITIVE_ENVIRONMENT_NAMES = (
    "CONTAINER_API_KEY",
    "JUPYTER_TOKEN",
    "OPEN_BUTTON_TOKEN",
)
RECIPE_DIR = Path(__file__).resolve().parent
AUDIO_PROVENANCE_SCHEMA = "speakrs-audio-provenance"
AUDIO_PROVENANCE_SCHEMA_VERSION = 1
AUDIO_PROVENANCE_SUFFIX = ".provenance.json"
PREPARATION_POLICY_NAME = "full-corpus-audio"
# version 2 records the correction from ffmpeg's layout-aware downmix to an
# explicit equal mean for AISHELL-4
PREPARATION_POLICY_VERSION = 2
AISHELL_INPUT_CHANNELS = 8


class Corpus(str, Enum):
    """A source corpus used by the standard DiariZen recipe."""

    AMI = "AMI"
    ALI_MEETING = "AliMeeting"
    AISHELL4 = "AISHELL4"


class Split(str, Enum):
    """A data split used by the standard DiariZen recipe."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class ChannelPolicy(str, Enum):
    """The channel conversion applied to source audio."""

    FIRST = "first"
    MIX = "mix"


def channel_policy_name(channel_policy: ChannelPolicy) -> str:
    """Return the stable provenance name for a channel policy."""

    if channel_policy is ChannelPolicy.FIRST:
        return "first_channel"
    if channel_policy is ChannelPolicy.MIX:
        return "equal_mean"
    raise ValueError(f"Unknown channel policy: {channel_policy}")


def channel_policy_version(channel_policy: ChannelPolicy) -> int:
    """Return the version of a channel policy's audio operation."""

    if channel_policy is ChannelPolicy.FIRST:
        return 1
    if channel_policy is ChannelPolicy.MIX:
        return 2
    raise ValueError(f"Unknown channel policy: {channel_policy}")


def channel_policy_description(channel_policy: ChannelPolicy) -> str:
    """Return a reader-facing description of a channel policy."""

    if channel_policy is ChannelPolicy.FIRST:
        return "first decoded input channel"
    if channel_policy is ChannelPolicy.MIX:
        return "equal arithmetic mean of every input channel"
    raise ValueError(f"Unknown channel policy: {channel_policy}")


@dataclass(frozen=True)
class Recording:
    """A validated recording required by a DiariZen manifest."""

    session_id: str
    corpus: Corpus
    split: Split


@dataclass(frozen=True)
class Archive:
    """A remote archive and the recordings expected in it."""

    name: str
    url: str
    corpus: Corpus
    channel_policy: ChannelPolicy
    session_ids: frozenset[str]
    input_channels: int | None = None


@dataclass(frozen=True)
class ManifestSet:
    """The three files that define one DiariZen dataset split."""

    source_dir: Path
    output_dir: Path
    recordings: tuple[Recording, ...]


AISHELL_ARCHIVE_URLS = {
    "train_L": "https://www.openslr.org/resources/111/train_L.tar.gz",
    "train_M": "https://www.openslr.org/resources/111/train_M.tar.gz",
    "train_S": "https://www.openslr.org/resources/111/train_S.tar.gz",
    "test": "https://www.openslr.org/resources/111/test.tar.gz",
}
ALI_ARCHIVE_URLS = {
    Split.TRAIN: ("https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz"),
    Split.DEV: ("https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz"),
    Split.TEST: ("https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz"),
}
AMI_AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{session_id}/audio/{session_id}.Array1-01.wav"
)


def scrubbed_environment() -> dict[str, str]:
    """Return a child-process environment without host control tokens."""

    environment = os.environ.copy()
    for name in SENSITIVE_ENVIRONMENT_NAMES:
        environment.pop(name, None)

    return environment


def classify_session(session_id: str) -> Corpus:
    """Classify a validated session ID into its source corpus."""

    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(f"Invalid session ID: {session_id!r}")
    if AMI_SESSION_PATTERN.fullmatch(session_id):
        return Corpus.AMI
    if ALI_SESSION_PATTERN.fullmatch(session_id):
        return Corpus.ALI_MEETING
    if AISHELL_SESSION_PATTERN.fullmatch(session_id):
        return Corpus.AISHELL4

    raise ValueError(f"Unknown session ID format: {session_id!r}")


def read_session_ids(path: Path) -> tuple[str, ...]:
    """Read and validate the unique session IDs from a Kaldi-style file."""

    session_ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        session_id = fields[0]
        classify_session(session_id)
        if session_id in seen:
            raise ValueError(f"Duplicate session {session_id!r} in {path}:{line_number}")
        seen.add(session_id)
        session_ids.append(session_id)

    if not session_ids:
        raise ValueError(f"No sessions in {path}")

    return tuple(session_ids)


def read_rttm_session_ids(path: Path) -> set[str]:
    """Read and validate session IDs from an RTTM file."""

    session_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"Invalid RTTM row in {path}:{line_number}")
        classify_session(fields[1])
        session_ids.add(fields[1])

    return session_ids


def load_manifest_set(source_dir: Path, output_dir: Path, split: Split) -> ManifestSet:
    """Parse and cross-check one source manifest set."""

    wav_ids = read_session_ids(source_dir / "wav.scp")
    uem_ids = set(read_session_ids(source_dir / "all.uem"))
    rttm_ids = read_rttm_session_ids(source_dir / "rttm")
    expected_ids = set(wav_ids)
    if uem_ids != expected_ids:
        raise ValueError(f"UEM sessions do not match wav.scp in {source_dir}")
    if rttm_ids != expected_ids:
        raise ValueError(f"RTTM sessions do not match wav.scp in {source_dir}")

    recordings = tuple(Recording(session_id, classify_session(session_id), split) for session_id in wav_ids)
    return ManifestSet(source_dir, output_dir, recordings)


def load_manifests(source_root: Path, output_root: Path) -> tuple[ManifestSet, ...]:
    """Load the combined train/dev sets and all three held-out test sets."""

    manifests = [
        load_manifest_set(source_root / "train", output_root / "train", Split.TRAIN),
        load_manifest_set(source_root / "dev", output_root / "dev", Split.DEV),
    ]
    for corpus in Corpus:
        manifests.append(
            load_manifest_set(
                source_root / "test" / corpus.value,
                output_root / "test" / corpus.value,
                Split.TEST,
            )
        )

    return tuple(manifests)


def aishell_archive_name(recording: Recording) -> str:
    """Return the OpenSLR archive name for one AISHELL-4 recording."""

    if recording.split is Split.TEST:
        return "test"
    for size in ("L", "M", "S"):
        if f"_{size}_" in recording.session_id:
            return f"train_{size}"

    raise ValueError(f"Cannot map AISHELL-4 session to an archive: {recording.session_id}")


def build_archives(recordings: Iterable[Recording]) -> tuple[Archive, ...]:
    """Group required recordings by source archive."""

    grouped: dict[tuple[Corpus, str], set[str]] = {}
    for recording in recordings:
        if recording.corpus is Corpus.AMI:
            continue
        if recording.corpus is Corpus.AISHELL4:
            name = aishell_archive_name(recording)
        else:
            name = recording.split.value
        grouped.setdefault((recording.corpus, name), set()).add(recording.session_id)

    archives: list[Archive] = []
    for (corpus, name), session_ids in sorted(grouped.items(), key=lambda item: (item[0][0].value, item[0][1])):
        if corpus is Corpus.AISHELL4:
            url = AISHELL_ARCHIVE_URLS[name]
        else:
            split = Split(name)
            url = ALI_ARCHIVE_URLS[split]
        channel_policy, input_channels = expected_audio_policy(corpus)
        archives.append(Archive(name, url, corpus, channel_policy, frozenset(session_ids), input_channels))

    return tuple(archives)


def audio_path(output_root: Path, recording: Recording) -> Path:
    """Return the normalized audio path for a recording."""

    return output_root / "audio" / recording.corpus.value / f"{recording.session_id}.flac"


def audio_provenance_path(path: Path) -> Path:
    """Return the sidecar path that records how one audio file was prepared."""

    return path.with_name(path.name + AUDIO_PROVENANCE_SUFFIX)


def expected_audio_policy(corpus: Corpus) -> tuple[ChannelPolicy, int | None]:
    """Return the channel policy and known input count for one corpus."""

    if corpus is Corpus.AISHELL4:
        return ChannelPolicy.MIX, AISHELL_INPUT_CHANNELS
    return ChannelPolicy.FIRST, None


def audio_provenance(channel_policy: ChannelPolicy, input_channels: int | None = None) -> dict[str, object]:
    """Return the current provenance record for one normalized audio file."""

    if channel_policy is ChannelPolicy.MIX:
        if input_channels is None:
            input_channels = AISHELL_INPUT_CHANNELS
        if input_channels <= 0:
            raise ValueError("Mixed audio must have a positive input channel count")
    else:
        input_channels = None

    return {
        "schema": AUDIO_PROVENANCE_SCHEMA,
        "schema_version": AUDIO_PROVENANCE_SCHEMA_VERSION,
        "preparation_policy": PREPARATION_POLICY_NAME,
        "preparation_policy_version": PREPARATION_POLICY_VERSION,
        "format": "mono FLAC at 16 kHz",
        "channel_policy": channel_policy_name(channel_policy),
        "channel_policy_version": channel_policy_version(channel_policy),
        "channel_policy_description": channel_policy_description(channel_policy),
        "input_channels": input_channels,
    }


def audio_matches_policy(
    path: Path,
    channel_policy: ChannelPolicy,
    input_channels: int | None = None,
) -> bool:
    """Return true only when audio and its sidecar match the current policy."""

    if not probe_audio(path):
        return False

    try:
        metadata = json.loads(audio_provenance_path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False

    expected = audio_provenance(channel_policy, input_channels)
    return isinstance(metadata, dict) and all(metadata.get(key) == value for key, value in expected.items())


def write_audio_provenance(path: Path, channel_policy: ChannelPolicy, input_channels: int | None = None) -> None:
    """Write one audio file's policy sidecar atomically."""

    atomic_write(
        audio_provenance_path(path),
        json.dumps(audio_provenance(channel_policy, input_channels), indent=2) + "\n",
    )


def audio_ready(
    path: Path,
    channel_policy: ChannelPolicy,
    input_channels: int | None = None,
) -> bool:
    """Return true when audio matches policy, adopting legacy first-channel output."""

    if audio_matches_policy(path, channel_policy, input_channels):
        return True
    if channel_policy is not ChannelPolicy.FIRST or audio_provenance_path(path).exists() or not probe_audio(path):
        return False

    write_audio_provenance(path, channel_policy, input_channels)
    return True


def preparation_policy_provenance() -> dict[str, object]:
    """Return the current global preparation policy record."""

    return {
        "name": PREPARATION_POLICY_NAME,
        "version": PREPARATION_POLICY_VERSION,
        "audio_sidecar": {
            "suffix": AUDIO_PROVENANCE_SUFFIX,
            "schema": AUDIO_PROVENANCE_SCHEMA,
            "schema_version": AUDIO_PROVENANCE_SCHEMA_VERSION,
        },
        "channel_policies": {
            corpus.value: {
                "name": channel_policy_name(channel_policy),
                "version": channel_policy_version(channel_policy),
                "description": channel_policy_description(channel_policy),
                "input_channels": input_channels,
            }
            for corpus in Corpus
            for channel_policy, input_channels in (expected_audio_policy(corpus),)
        },
    }


def verify_provenance(path: Path) -> None:
    """Verify that the global provenance names the current preparation policy."""

    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read provenance: {path}: {error}") from error

    policy = provenance.get("preparation_policy") if isinstance(provenance, dict) else None
    expected = preparation_policy_provenance()
    if (
        not isinstance(policy, dict)
        or policy.get("name") != expected["name"]
        or policy.get("version") != expected["version"]
    ):
        raise RuntimeError(
            f"Provenance does not match preparation policy {PREPARATION_POLICY_NAME} v{PREPARATION_POLICY_VERSION}: {path}"
        )


def verify_prepared_corpus(manifests: Iterable[ManifestSet], output_root: Path) -> None:
    """Verify all required audio and provenance without changing the output tree."""

    verify_provenance(output_root / "provenance.json")
    invalid: list[Path] = []
    for manifest in manifests:
        for recording in manifest.recordings:
            channel_policy, input_channels = expected_audio_policy(recording.corpus)
            output = audio_path(output_root, recording)
            if not audio_matches_policy(output, channel_policy, input_channels):
                invalid.append(output)

    if invalid:
        paths = ", ".join(path.as_posix() for path in invalid)
        raise RuntimeError(f"Audio is absent, invalid, or has stale policy provenance: {paths}")


def probe_audio(
    path: Path,
    channel_policy: ChannelPolicy | None = None,
    input_channels: int | None = None,
) -> bool:
    """Return true when audio has the required header and optional policy provenance."""

    if not path.is_file() or path.stat().st_size == 0:
        return False
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0:s=x",
            path.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=scrubbed_environment(),
    )
    if result.returncode != 0 or result.stdout.strip() != "16000x1":
        return False
    if channel_policy is None:
        return True
    return audio_matches_policy(path, channel_policy, input_channels)


def equal_mean_filter(input_channels: int) -> str:
    """Return an ffmpeg filter that averages every input channel equally."""

    if input_channels <= 0:
        raise ValueError("Equal mean requires a positive input channel count")
    coefficient = 1 / input_channels
    terms = "+".join(f"{coefficient:.12g}*c{channel}" for channel in range(input_channels))
    return f"pan=mono|c0={terms}"


def transcode(
    stream: BinaryIO,
    output: Path,
    channel_policy: ChannelPolicy,
    input_channels: int | None = None,
) -> None:
    """Transcode one input stream to an atomic mono 16 kHz FLAC file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.flac")
    temporary.unlink(missing_ok=True)
    if channel_policy is ChannelPolicy.FIRST:
        input_channels = None
        channel_args = ["-af", "pan=mono|c0=c0"]
    else:
        if input_channels is None:
            input_channels = AISHELL_INPUT_CHANNELS
        channel_args = ["-af", equal_mean_filter(input_channels)]
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        *channel_args,
        "-ar",
        "16000",
        "-c:a",
        "flac",
        "-compression_level",
        "5",
        "-f",
        "flac",
        "-y",
        temporary.as_posix(),
    ]
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=error_log,
            env=scrubbed_environment(),
        )
        assert process.stdin is not None
        try:
            shutil.copyfileobj(stream, process.stdin, length=1024 * 1024)
            process.stdin.close()
        except BrokenPipeError:
            pass
        return_code = process.wait()
        if return_code != 0:
            error_log.seek(0)
            message = error_log.read().decode(errors="replace").strip()
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed for {output.name}: {message}")

    if not probe_audio(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Invalid transcoded audio: {output}")

    sidecar = audio_provenance_path(output)
    sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".partial")
    sidecar_temporary.unlink(missing_ok=True)
    try:
        sidecar_temporary.write_text(
            json.dumps(audio_provenance(channel_policy, input_channels), indent=2) + "\n",
            encoding="utf-8",
        )
        # publish audio before its sidecar so a crash cannot make stale audio look current
        temporary.replace(output)
        sidecar_temporary.replace(sidecar)
    except BaseException:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)
        raise


def curl_stream(url: str) -> subprocess.Popen[bytes]:
    """Start one retrying, secret-free source download."""

    return subprocess.Popen(
        [
            "curl",
            "--location",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--show-error",
            url,
        ],
        stdout=subprocess.PIPE,
        env=scrubbed_environment(),
    )


def download_ami(recording: Recording, output_root: Path) -> None:
    """Download and normalize one AMI single distant-microphone recording."""

    output = audio_path(output_root, recording)
    if audio_ready(output, ChannelPolicy.FIRST):
        return
    url = AMI_AUDIO_URL.format(session_id=recording.session_id)
    print(f"AMI {recording.session_id}", flush=True)
    curl = curl_stream(url)
    assert curl.stdout is not None
    try:
        transcode(curl.stdout, output, ChannelPolicy.FIRST)
    except BaseException:
        curl.terminate()
        curl.wait()
        raise
    finally:
        curl.stdout.close()
    if curl.wait() != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {url}")


def archive_member_session(member: tarfile.TarInfo) -> str | None:
    """Return a safe audio member's session ID, or none for other members."""

    if not member.isfile():
        return None
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.suffix.lower() not in {".wav", ".flac"}:
        return None
    session_id = path.stem
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        return None

    return session_id


def extract_archive(archive: Archive, output_root: Path) -> None:
    """Stream one archive once and extract only required audio."""

    pending = {
        session_id
        for session_id in archive.session_ids
        if not audio_ready(
            output_root / "audio" / archive.corpus.value / f"{session_id}.flac",
            archive.channel_policy,
            archive.input_channels,
        )
    }
    if not pending:
        return

    print(f"Streaming {archive.corpus.value} {archive.name}: {len(pending)} files", flush=True)
    curl = curl_stream(archive.url)
    assert curl.stdout is not None
    try:
        with tarfile.open(fileobj=curl.stdout, mode="r|gz") as source:
            for member in source:
                session_id = archive_member_session(member)
                if session_id not in pending:
                    continue
                member_stream = source.extractfile(member)
                if member_stream is None:
                    raise RuntimeError(f"Cannot read archive member: {member.name}")
                output = output_root / "audio" / archive.corpus.value / f"{session_id}.flac"
                print(f"  {session_id}", flush=True)
                with member_stream:
                    transcode(member_stream, output, archive.channel_policy, archive.input_channels)
                pending.remove(session_id)
                if not pending:
                    break
    except BaseException:
        curl.terminate()
        curl.wait()
        raise
    finally:
        curl.stdout.close()

    if not pending:
        curl.terminate()
        curl.wait()
        return
    if curl.wait() != 0:
        raise RuntimeError(f"Download failed: {archive.url}")
    raise RuntimeError(f"Archive {archive.name} lacks required sessions: {', '.join(sorted(pending))}")


def retry(operation, label: str, attempts: int = 5) -> None:
    """Retry a convergent operation after a bounded delay."""

    for attempt in range(1, attempts + 1):
        try:
            operation()
            return
        except (OSError, RuntimeError, tarfile.TarError) as error:
            if attempt == attempts:
                raise
            delay = 15 * attempt
            print(
                f"{label} failed on attempt {attempt}: {error}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)


def atomic_write(path: Path, content: str) -> None:
    """Replace a text file only after its complete content is on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_manifests(
    manifests: Iterable[ManifestSet],
    output_root: Path,
    manifest_audio_prefix: Path,
) -> None:
    """Write source annotations and normalized wav.scp files atomically."""

    for manifest in manifests:
        for recording in manifest.recordings:
            output = audio_path(output_root, recording)
            channel_policy, input_channels = expected_audio_policy(recording.corpus)
            if not audio_ready(output, channel_policy, input_channels):
                raise RuntimeError(f"Required audio is absent or invalid: {output}")
        wav_lines = [
            f"{recording.session_id} "
            f"{manifest_audio_prefix / recording.corpus.value / (recording.session_id + '.flac')}"
            for recording in manifest.recordings
        ]
        atomic_write(manifest.output_dir / "wav.scp", "\n".join(wav_lines) + "\n")
        for name in ("rttm", "all.uem"):
            atomic_write(manifest.output_dir / name, (manifest.source_dir / name).read_text())


def print_plan(manifests: Iterable[ManifestSet], archives: Iterable[Archive]) -> None:
    """Print the validated download plan."""

    recordings = [recording for manifest in manifests for recording in manifest.recordings]
    unique = {(recording.corpus, recording.session_id) for recording in recordings}
    for corpus in Corpus:
        count = sum(item[0] is corpus for item in unique)
        print(f"{corpus.value}: {count} unique recordings")
    for archive in archives:
        print(f"{archive.corpus.value}/{archive.name}: {len(archive.session_ids)} files")


def require_executable(name: str) -> None:
    """Fail with a clear message when a required program is absent."""

    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is required")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=RECIPE_DIR.parent / "diar_ssl" / "data" / "AMI_AliMeeting_AISHELL4",
    )
    parser.add_argument("--output-root", type=Path, default=RECIPE_DIR / "data" / "full")
    parser.add_argument(
        "--manifest-audio-prefix",
        type=Path,
        default=Path("../speakrs/data/full/audio"),
    )
    parser.add_argument("--minimum-free-gib", type=float, default=30.0)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify prepared audio and provenance without downloading or writing",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare all standard train, development, and test audio and manifests."""

    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifests = load_manifests(source_root, output_root)
    recordings = tuple(recording for manifest in manifests for recording in manifest.recordings)
    archives = build_archives(recordings)
    print_plan(manifests, archives)
    if args.plan and args.verify:
        raise ValueError("--plan and --verify cannot be combined")
    if args.plan:
        return

    require_executable("ffprobe")
    if args.verify:
        verify_prepared_corpus(manifests, output_root)
        print("Full corpus verification passed", flush=True)
        return

    require_executable("curl")
    require_executable("ffmpeg")
    output_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_root).free / 1024**3
    if free_gib < args.minimum_free_gib:
        raise RuntimeError(f"Only {free_gib:.1f} GiB is free; {args.minimum_free_gib:.1f} GiB is required")

    unique_recordings = {(recording.corpus, recording.session_id): recording for recording in recordings}
    for recording in sorted(unique_recordings.values(), key=lambda item: (item.corpus.value, item.session_id)):
        if recording.corpus is Corpus.AMI:
            retry(
                lambda recording=recording: download_ami(recording, output_root),
                f"AMI {recording.session_id}",
            )
    for archive in archives:
        retry(
            lambda archive=archive: extract_archive(archive, output_root),
            f"{archive.corpus.value} {archive.name}",
        )

    write_manifests(manifests, output_root, args.manifest_audio_prefix)
    provenance = {
        "format": "mono FLAC at 16 kHz",
        "preparation_policy": preparation_policy_provenance(),
        "archives": [
            {
                "corpus": archive.corpus.value,
                "name": archive.name,
                "url": archive.url,
                "recordings": len(archive.session_ids),
                "channel_policy": channel_policy_name(archive.channel_policy),
                "channel_policy_version": channel_policy_version(archive.channel_policy),
                "input_channels": archive.input_channels,
            }
            for archive in archives
        ],
        "ami_audio_url_template": AMI_AUDIO_URL,
    }
    atomic_write(output_root / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    print("Full corpus preparation is complete", flush=True)


if __name__ == "__main__":
    main()
