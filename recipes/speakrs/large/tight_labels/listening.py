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
from .writers import format_time


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
    start: float
    end: float
    change_seconds: float


def change_spans(original: RecordingLabels, new: RecordingLabels) -> list[tuple[float, float]]:
    """Return times where any speaker's activity differs."""

    speakers = original.speakers | new.speakers
    changed: list[tuple[float, float]] = []
    for speaker in speakers:
        original_union = union_spans(original.segments_for(speaker))
        new_union = union_spans(new.segments_for(speaker))
        changed.extend(subtract_spans(original_union, new_union))
        changed.extend(subtract_spans(new_union, original_union))
    return union_spans(changed)


def select_listening_windows(
    original: Sequence[RecordingLabels],
    new: Sequence[RecordingLabels],
    *,
    duration_by_recording: Mapping[str, float],
    max_pack_seconds: float = MAX_PACK_SECONDS,
    max_clip_seconds: float = MAX_CLIP_SECONDS,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    pad_seconds: float = CLIP_PAD_SECONDS,
    max_clips_per_recording: int = MAX_CLIPS_PER_RECORDING,
) -> tuple[ListeningWindow, ...]:
    """Choose the most-changed regions up to the pack budget."""

    new_by_id = {item.recording_id: item for item in new}
    candidates: list[ListeningWindow] = []
    for source in original:
        produced = new_by_id[source.recording_id]
        audio_end = duration_by_recording[source.recording_id]
        per_recording = 0
        regions = sorted(change_spans(source, produced), key=lambda span: span[1] - span[0], reverse=True)
        for start, end in regions:
            if per_recording >= max_clips_per_recording:
                break
            change = end - start
            if change <= 0:
                continue
            clip_start = max(0.0, start - pad_seconds)
            clip_end = min(audio_end, end + pad_seconds)
            if clip_end - clip_start > max_clip_seconds:
                # keep the start of the changed region when the run is long
                clip_end = min(audio_end, max(start, clip_start) + max_clip_seconds)
            if clip_end - clip_start < min_clip_seconds:
                extra = min_clip_seconds - (clip_end - clip_start)
                clip_start = max(0.0, clip_start - extra / 2)
                clip_end = min(audio_end, clip_start + min_clip_seconds)
            if clip_end <= clip_start:
                continue
            candidates.append(ListeningWindow(source.recording_id, clip_start, clip_end, change))
            per_recording += 1
    candidates.sort(key=lambda item: item.change_seconds, reverse=True)
    chosen: list[ListeningWindow] = []
    used = 0.0
    for window in candidates:
        length = window.end - window.start
        if used + length > max_pack_seconds:
            remaining = max_pack_seconds - used
            if remaining < min_clip_seconds:
                continue
            window = ListeningWindow(
                window.recording_id,
                window.start,
                window.start + remaining,
                window.change_seconds,
            )
            length = remaining
        chosen.append(window)
        used += length
        if used >= max_pack_seconds - 1e-9:
            break
    return tuple(chosen)


def _clip_rows(
    window: ListeningWindow,
    clip_id: str,
    original: RecordingLabels,
    new: RecordingLabels,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label_set, labels in (("original", original), ("new", new)):
        for segment in labels.segments:
            start = max(segment.start, window.start)
            end = min(segment.end, window.end)
            if end <= start:
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "recording_id": window.recording_id,
                    "audio_start": format_time(window.start),
                    "audio_end": format_time(window.end),
                    "label_set": label_set,
                    "speaker": segment.speaker,
                    "start": format_time(start - window.start),
                    "end": format_time(end - window.start),
                }
            )
    return rows


def write_listening_pack(
    directory: Path,
    *,
    original: Sequence[RecordingLabels],
    new: Sequence[RecordingLabels],
    audio_by_recording: Mapping[str, Path],
    duration_by_recording: Mapping[str, float],
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
        start_frame = int(round(window.start * SAMPLE_RATE))
        stop_frame = int(round(window.end * SAMPLE_RATE))
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
