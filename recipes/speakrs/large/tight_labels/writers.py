"""RTTM and UNCERTAIN sidecar writers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..jsonio import atomic_write_text
from .domain import RecordingLabels, SpeakerSegment, UncertainInterval, require_identity


TIME_DECIMALS = 9


def format_time(value: float) -> str:
    """Format a time in seconds to nine decimal places."""

    return f"{value:.{TIME_DECIMALS}f}"


def speaker_rttm_row(segment: SpeakerSegment) -> str:
    """Return one SPEAKER row with nine-decimal times."""

    duration = segment.end - segment.start
    return (
        f"SPEAKER {segment.recording_id} 1 {format_time(segment.start)} "
        f"{format_time(duration)} <NA> <NA> {segment.speaker} <NA> <NA>"
    )


def uncertain_rttm_row(region: UncertainInterval) -> str:
    """Return one UNCERTAIN row with the SPEAKER field layout."""

    duration = region.end - region.start
    return (
        f"UNCERTAIN {region.recording_id} 1 {format_time(region.start)} "
        f"{format_time(duration)} <NA> <NA> {region.speaker} <NA> <NA>"
    )


def render_speaker_rttm(segments: Sequence[SpeakerSegment]) -> str:
    """Render SPEAKER rows sorted by onset then speaker."""

    ordered = sorted(segments, key=lambda item: (item.start, item.speaker, item.end))
    if not ordered:
        return ""
    return "\n".join(speaker_rttm_row(item) for item in ordered) + "\n"


def render_uncertain_rttm(regions: Sequence[UncertainInterval]) -> str:
    """Render UNCERTAIN rows; an empty sequence yields an empty file body."""

    ordered = sorted(regions, key=lambda item: (item.start, item.speaker, item.end))
    if not ordered:
        return ""
    return "\n".join(uncertain_rttm_row(item) for item in ordered) + "\n"


def write_speaker_rttm(path: Path, segments: Sequence[SpeakerSegment]) -> None:
    """Write SPEAKER RTTM through an atomic replace."""

    atomic_write_text(path, render_speaker_rttm(segments))


def write_uncertain_rttm(path: Path, regions: Sequence[UncertainInterval]) -> None:
    """Write the UNCERTAIN sidecar, including an empty file."""

    atomic_write_text(path, render_uncertain_rttm(regions))


def write_recording_labels(directory: Path, labels: RecordingLabels) -> tuple[Path, Path]:
    """Write `<recording-id>.rttm` and `<recording-id>.uncertain.rttm`."""

    directory.mkdir(parents=True, exist_ok=True)
    rttm_path = directory / f"{labels.recording_id}.rttm"
    uncertain_path = directory / f"{labels.recording_id}.uncertain.rttm"
    write_speaker_rttm(rttm_path, labels.segments)
    write_uncertain_rttm(uncertain_path, labels.uncertain)
    return rttm_path, uncertain_path


def parse_speaker_rttm(text: str, *, recording_id: str | None = None) -> tuple[SpeakerSegment, ...]:
    """Parse SPEAKER rows into validated segments."""

    segments: list[SpeakerSegment] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if not fields or fields[0].upper() != "SPEAKER":
            continue
        if len(fields) < 8:
            raise ValueError(f"RTTM line is malformed: {line}")
        file_id = require_identity(fields[1], "recording_id")
        if recording_id is not None:
            file_id = recording_id
        start = float(fields[3])
        duration = float(fields[4])
        speaker = fields[7]
        if speaker in {"<NA>", "NA", ""}:
            speaker = fields[8] if len(fields) > 8 else ""
        segments.append(SpeakerSegment(file_id, speaker, start, start + duration))
    return tuple(segments)


def parse_uncertain_rttm(text: str, *, recording_id: str | None = None) -> tuple[UncertainInterval, ...]:
    """Parse UNCERTAIN sidecar rows."""

    regions: list[UncertainInterval] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if not fields or fields[0].upper() != "UNCERTAIN":
            continue
        if len(fields) < 8:
            raise ValueError(f"UNCERTAIN line is malformed: {line}")
        file_id = require_identity(fields[1], "recording_id")
        if recording_id is not None:
            file_id = recording_id
        start = float(fields[3])
        duration = float(fields[4])
        speaker = fields[7]
        regions.append(UncertainInterval(file_id, speaker, start, start + duration))
    return tuple(regions)


def parse_recording_rttm(
    text: str,
    *,
    recording_id: str,
    uncertain_text: str = "",
) -> RecordingLabels:
    """Parse speaker and optional uncertain text for one recording."""

    return RecordingLabels(
        recording_id,
        parse_speaker_rttm(text, recording_id=recording_id),
        parse_uncertain_rttm(uncertain_text, recording_id=recording_id),
    )


def parse_uem_regions(text: str) -> dict[str, tuple[tuple[float, float], ...]]:
    """Parse Kaldi UEM rows keyed by file id."""

    regions: dict[str, list[tuple[float, float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"UEM line is malformed: {line}")
        start = float(fields[-2])
        end = float(fields[-1])
        regions.setdefault(fields[0], []).append((start, end))
    return {key: tuple(value) for key, value in regions.items()}


def rewrite_uem_file_id(text: str, recording_id: str) -> str:
    """Copy UEM times and replace the file id."""

    require_identity(recording_id, "recording_id")
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(raw)
            continue
        fields = stripped.split()
        if len(fields) < 4:
            raise ValueError(f"UEM line is malformed: {raw}")
        fields[0] = recording_id
        lines.append(" ".join(fields))
    body = "\n".join(lines)
    if text.endswith("\n") or not text:
        return body + ("\n" if body or text.endswith("\n") else "")
    return body


def render_uem(recording_id: str, regions: Iterable[tuple[float, float]]) -> str:
    """Render UEM rows for one recording."""

    rows = [f"{recording_id} 1 {format_time(start)} {format_time(end)}" for start, end in regions if end > start]
    if not rows:
        return ""
    return "\n".join(rows) + "\n"
