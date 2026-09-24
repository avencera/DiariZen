"""Build a five-minute listening pack of the most-changed regions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf

from ..jsonio import atomic_write_text
from .domain import RecordingLabels
from .intervals import subtract_spans, union_spans
from .time import NANOSECONDS_PER_SECOND, format_ns, parse_seconds_to_ns


MAX_PACK_SECONDS = 300.0
CLIP_PAD_SECONDS = 0.5
MAX_CLIP_SECONDS = 8.0
MIN_CLIP_SECONDS = 2.0
MAX_CLIPS_PER_RECORDING = 3
SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class ListeningWindow:
    """A source-audio interval chosen for a listening clip."""

    recording_id: str
    start_ns: int
    end_ns: int
    change_ns: int

    @property
    def start(self) -> float:
        """Return clip start in seconds."""

        return self.start_ns / NANOSECONDS_PER_SECOND

    @property
    def end(self) -> float:
        """Return clip end in seconds."""

        return self.end_ns / NANOSECONDS_PER_SECOND

    @property
    def change_seconds(self) -> float:
        """Return changed speech time in seconds."""

        return self.change_ns / NANOSECONDS_PER_SECOND


def change_spans(original: RecordingLabels, new: RecordingLabels) -> list[tuple[int, int]]:
    """Return times where any speaker's activity differs, including UNCERTAIN."""

    speakers = original.speakers | new.speakers
    changed: list[tuple[int, int]] = []
    for speaker in speakers:
        original_union = union_spans(original.segments_for(speaker))
        new_union = union_spans(new.segments_for(speaker))
        changed.extend(subtract_spans(original_union, new_union))
        changed.extend(subtract_spans(new_union, original_union))
        changed.extend((item.start_ns, item.end_ns) for item in new.uncertain_for(speaker))
        changed.extend((item.start_ns, item.end_ns) for item in original.uncertain_for(speaker))
    return union_spans(changed)


def select_listening_windows(
    original: Sequence[RecordingLabels],
    new: Sequence[RecordingLabels],
    *,
    duration_by_recording: Mapping[str, int | float],
    max_pack_seconds: float = MAX_PACK_SECONDS,
    max_clip_seconds: float = MAX_CLIP_SECONDS,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    pad_seconds: float = CLIP_PAD_SECONDS,
    max_clips_per_recording: int = MAX_CLIPS_PER_RECORDING,
) -> tuple[ListeningWindow, ...]:
    """Choose the most-changed regions up to the pack budget."""

    max_pack_ns = parse_seconds_to_ns(max_pack_seconds)
    max_clip_ns = parse_seconds_to_ns(max_clip_seconds)
    min_clip_ns = parse_seconds_to_ns(min_clip_seconds)
    pad_ns = parse_seconds_to_ns(pad_seconds)
    new_by_id = {item.recording_id: item for item in new}
    candidates: list[ListeningWindow] = []
    for source in original:
        produced = new_by_id[source.recording_id]
        audio_end = _duration_ns(duration_by_recording[source.recording_id])
        per_recording = 0
        regions = sorted(change_spans(source, produced), key=lambda span: span[1] - span[0], reverse=True)
        for start_ns, end_ns in regions:
            if per_recording >= max_clips_per_recording:
                break
            change_ns = end_ns - start_ns
            if change_ns <= 0:
                continue
            clip_start = max(0, start_ns - pad_ns)
            clip_end = min(audio_end, end_ns + pad_ns)
            if clip_end - clip_start > max_clip_ns:
                clip_end = min(audio_end, max(start_ns, clip_start) + max_clip_ns)
            if clip_end - clip_start < min_clip_ns:
                extra = min_clip_ns - (clip_end - clip_start)
                clip_start = max(0, clip_start - extra // 2)
                clip_end = min(audio_end, clip_start + min_clip_ns)
            if clip_end <= clip_start:
                continue
            candidates.append(ListeningWindow(source.recording_id, clip_start, clip_end, change_ns))
            per_recording += 1
    candidates.sort(key=lambda item: item.change_ns, reverse=True)
    chosen: list[ListeningWindow] = []
    used = 0
    for window in candidates:
        length = window.end_ns - window.start_ns
        if used + length > max_pack_ns:
            remaining = max_pack_ns - used
            if remaining < min_clip_ns:
                continue
            window = ListeningWindow(
                window.recording_id,
                window.start_ns,
                window.start_ns + remaining,
                window.change_ns,
            )
            length = remaining
        chosen.append(window)
        used += length
        if used >= max_pack_ns:
            break
    return tuple(chosen)


def _duration_ns(value: int | float) -> int:
    if isinstance(value, bool):
        raise ValueError("duration must be nanoseconds or seconds")
    if isinstance(value, int) and value > 10_000:
        return value
    return parse_seconds_to_ns(value)


def _clip_rows(
    window: ListeningWindow,
    clip_id: str,
    original: RecordingLabels,
    new: RecordingLabels,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label_set, labels in (("original", original), ("new", new)):
        for segment in labels.segments:
            start_ns = max(segment.start_ns, window.start_ns)
            end_ns = min(segment.end_ns, window.end_ns)
            if end_ns <= start_ns:
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "recording_id": window.recording_id,
                    "audio_start": format_ns(window.start_ns),
                    "audio_end": format_ns(window.end_ns),
                    "label_set": label_set,
                    "speaker": segment.speaker,
                    "start": format_ns(start_ns - window.start_ns),
                    "end": format_ns(end_ns - window.start_ns),
                }
            )
        for region in labels.uncertain:
            start_ns = max(region.start_ns, window.start_ns)
            end_ns = min(region.end_ns, window.end_ns)
            if end_ns <= start_ns:
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "recording_id": window.recording_id,
                    "audio_start": format_ns(window.start_ns),
                    "audio_end": format_ns(window.end_ns),
                    "label_set": f"{label_set}-uncertain",
                    "speaker": region.speaker,
                    "start": format_ns(start_ns - window.start_ns),
                    "end": format_ns(end_ns - window.start_ns),
                }
            )
    return rows


def write_listening_pack(
    directory: Path,
    *,
    original: Sequence[RecordingLabels],
    new: Sequence[RecordingLabels],
    audio_by_recording: Mapping[str, Path],
    duration_by_recording: Mapping[str, int | float],
) -> dict[str, object]:
    """Cut WAV clips and write a CSV of original and new labels."""

    directory.mkdir(parents=True, exist_ok=True)
    windows = select_listening_windows(
        original,
        new,
        duration_by_recording=duration_by_recording,
    )
    original_by_id = {item.recording_id: item for item in original}
    new_by_id = {item.recording_id: item for item in new}
    csv_path = directory / "clips.csv"
    fieldnames = (
        "clip_id",
        "recording_id",
        "audio_start",
        "audio_end",
        "label_set",
        "speaker",
        "start",
        "end",
    )
    rows: list[dict[str, str]] = []
    total_seconds = 0.0
    for index, window in enumerate(windows, start=1):
        clip_id = f"clip-{index:04d}"
        wav_path = directory / f"{clip_id}.wav"
        audio_path = audio_by_recording[window.recording_id]
        start_frame = (window.start_ns * SAMPLE_RATE) // NANOSECONDS_PER_SECOND
        stop_frame = (window.end_ns * SAMPLE_RATE) // NANOSECONDS_PER_SECOND
        if stop_frame <= start_frame:
            continue
        samples, sample_rate = sf.read(
            audio_path, start=start_frame, stop=stop_frame, dtype="float32", always_2d=False
        )
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"listening audio must be {SAMPLE_RATE} Hz: {audio_path}")
        if np.ndim(samples) == 2:
            samples = samples[:, 0]
        sf.write(wav_path, samples, SAMPLE_RATE, subtype="PCM_16")
        total_seconds += (stop_frame - start_frame) / SAMPLE_RATE
        rows.extend(
            _clip_rows(
                window,
                clip_id,
                original_by_id[window.recording_id],
                new_by_id[window.recording_id],
            )
        )
    handle_text = _csv_text(fieldnames, rows)
    atomic_write_text(csv_path, handle_text)
    return {
        "clip_count": len(windows),
        "total_seconds": total_seconds,
        "csv": str(csv_path),
        "directory": str(directory),
    }


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()
