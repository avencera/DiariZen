"""Source binding contracts for the recorded NOTSOFAR training set."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .acceptance import RttmInterval
from .errors import PreparationError
from .hashing import sha256_file


NOTSOFAR_REAL_REVISION = "ba8fd0f034ce185fe4d24f47e53b4b8194795f07"
NOTSOFAR_REAL_VERSION = "240825.1"
NOTSOFAR_REAL_SUBSET = "240825.1_train"
NOTSOFAR_SELECTED_VIEW = "sc_plaza_0/ch0.wav"
NOTSOFAR_SAMPLE_RATE = 16_000

_RECORDING_ID_RE = re.compile(r"^MTG_[0-9]{5}$")


@dataclass(frozen=True, slots=True)
class NotsofarAudioBinding:
    """Exact source-to-canonical audio identity for one meeting."""

    recording_id: str
    source_path: Path
    source_sha256: str
    canonical_path: Path
    canonical_sha256: str
    decoded_pcm_sha256: str
    sample_count: int


def parse_notsofar_transcript(path: Path, recording_id: str, duration: float) -> tuple[RttmInterval, ...]:
    """Parse and validate publisher speaker activity on the decoded audio clock."""

    if _RECORDING_ID_RE.fullmatch(recording_id) is None:
        raise ValueError("recording_id must be a canonical NOTSOFAR meeting identifier")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError("NOTSOFAR transcript is unreadable", {"recording_id": recording_id}) from error
    if not isinstance(rows, list) or not rows:
        raise PreparationError("NOTSOFAR transcript must contain turns", {"recording_id": recording_id})

    intervals = []
    for line_number, row in enumerate(rows, 1):
        try:
            start = float(row["start_time"])
            end = float(row["end_time"])
            speaker = str(row["speaker_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise PreparationError(
                "NOTSOFAR transcript row is invalid",
                {"recording_id": recording_id, "line": line_number},
            ) from error
        if not speaker or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise PreparationError(
                "NOTSOFAR transcript interval is invalid",
                {"recording_id": recording_id, "line": line_number},
            )
        if end > duration + 1e-9:
            raise PreparationError(
                "NOTSOFAR transcript interval exceeds decoded audio",
                {"recording_id": recording_id, "line": line_number},
            )
        intervals.append(RttmInterval(recording_id, start, end, speaker))

    return tuple(intervals)


def decoded_pcm_identity(path: Path) -> tuple[str, int]:
    """Hash the decoded mono PCM16 stream and return its sample count."""

    digest = hashlib.sha256()
    sample_count = 0
    try:
        with sf.SoundFile(str(path)) as audio:
            if audio.samplerate != NOTSOFAR_SAMPLE_RATE or audio.channels != 1:
                raise PreparationError("NOTSOFAR audio must be mono 16 kHz PCM")
            while True:
                block = audio.read(65_536, dtype="int16", always_2d=True)
                if not len(block):
                    break
                digest.update(block.tobytes(order="C"))
                sample_count += len(block)
    except (OSError, RuntimeError) as error:
        raise PreparationError("NOTSOFAR audio is unreadable", {"path": str(path)}) from error
    if sample_count <= 0:
        raise PreparationError("NOTSOFAR audio contains no samples", {"path": str(path)})

    return digest.hexdigest(), sample_count


def bind_notsofar_audio(recording_id: str, source: Path, canonical: Path) -> NotsofarAudioBinding:
    """Prove that one publisher WAV and canonical FLAC decode to identical PCM."""

    source_pcm_sha256, source_samples = decoded_pcm_identity(source)
    canonical_pcm_sha256, canonical_samples = decoded_pcm_identity(canonical)
    if source_samples != canonical_samples or source_pcm_sha256 != canonical_pcm_sha256:
        raise PreparationError("NOTSOFAR source and canonical audio differ", {"recording_id": recording_id})

    return NotsofarAudioBinding(
        recording_id=recording_id,
        source_path=source,
        source_sha256=sha256_file(source),
        canonical_path=canonical,
        canonical_sha256=sha256_file(canonical),
        decoded_pcm_sha256=source_pcm_sha256,
        sample_count=source_samples,
    )
