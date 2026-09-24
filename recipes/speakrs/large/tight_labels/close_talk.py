"""Close-talk clock offset, lavalier mapping, and energy voicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import correlate, resample_poly

from .domain import Interval, UncertainInterval
from .intervals import intersect_spans, subtract_spans
from .time import NANOSECONDS_PER_SECOND

TARGET_SAMPLE_RATE = 16_000


OFFSET_WINDOWS = 6
OFFSET_WINDOW_SECONDS = 8.0
OFFSET_MAX_ABS_SECONDS = 0.25
OFFSET_AGREE_SECONDS = 0.02
LAV_RATIO_MIN = 1.5
LAV_MIN_EXCLUSIVE_SECONDS = 5.0
VOICED_FRAME_MS = 30
VOICED_HOP_MS = 10
VOICED_ABS_FLOOR = 1e-3
VOICED_RELATIVE = 0.20


@dataclass(frozen=True, slots=True)
class ClockOffset:
    """Measured sample offset of a close-talk channel relative to the mix."""

    offset_samples: int
    offset_seconds: float
    window_offsets_seconds: tuple[float, ...]
    consistent: bool
    reason: str


@dataclass(frozen=True, slots=True)
class LavalierAssignment:
    """One lavalier mapped onto one speaker, with the energy evidence."""

    speaker: str
    channel: str
    ratio: float
    exclusive_seconds: float
    rms_by_speaker: dict[str, float]


@dataclass(frozen=True, slots=True)
class LavalierMapping:
    """Per-recording lavalier-to-speaker map, or a decision to use the mix."""

    assignments: tuple[LavalierAssignment, ...]
    unmapped_speakers: tuple[str, ...]
    use_close_talk: bool
    confidence: float
    reason: str

    def channel_for(self, speaker: str) -> str | None:
        """Return the mapped lavalier name for a speaker."""

        for item in self.assignments:
            if item.speaker == speaker:
                return item.channel
        return None


def downmix_mono(samples: np.ndarray) -> np.ndarray:
    """Return a 1-D float32 signal."""

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return np.ascontiguousarray(array, dtype=np.float32)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample a mono signal with a polyphase Kaiser filter."""

    mono = downmix_mono(samples)
    if source_rate == target_rate:
        return mono
    from math import gcd

    factor = gcd(int(source_rate), int(target_rate))
    return np.asarray(
        resample_poly(mono, int(target_rate) // factor, int(source_rate) // factor),
        dtype=np.float32,
    )


def measure_clock_offset(
    mix: np.ndarray,
    close: np.ndarray,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    windows: int = OFFSET_WINDOWS,
    window_seconds: float = OFFSET_WINDOW_SECONDS,
    max_abs_seconds: float = OFFSET_MAX_ABS_SECONDS,
    agree_seconds: float = OFFSET_AGREE_SECONDS,
) -> ClockOffset:
    """Estimate close-talk delay by cross-correlation on several windows."""

    mix_mono = downmix_mono(mix)
    close_mono = downmix_mono(close)
    length = min(mix_mono.size, close_mono.size)
    if length <= 0:
        return ClockOffset(0, 0.0, (), False, "empty-audio")
    mix_mono = mix_mono[:length]
    close_mono = close_mono[:length]
    window = int(window_seconds * sample_rate)
    if window <= 0 or length < window * 2:
        window = max(1, length // 4)
    if window <= 0 or length <= window:
        return ClockOffset(0, 0.0, (), False, "audio-shorter-than-window")
    starts = np.linspace(0, length - window, num=max(1, windows), dtype=int)
    max_lag = int(max_abs_seconds * sample_rate)
    offsets: list[int] = []
    for start in starts:
        mix_win = mix_mono[start : start + window]
        search_start = max(0, start - max_lag)
        search_end = min(length, start + window + max_lag)
        close_win = close_mono[search_start:search_end]
        if mix_win.size < 16 or close_win.size < mix_win.size:
            continue
        mix_win = mix_win - mix_win.mean()
        close_win = close_win - close_win.mean()
        if np.linalg.norm(mix_win) < 1e-8 or np.linalg.norm(close_win) < 1e-8:
            continue
        corr = correlate(close_win, mix_win, mode="valid")
        peak = int(np.argmax(corr))
        offset = search_start + peak - start
        offsets.append(offset)
    if not offsets:
        return ClockOffset(0, 0.0, (), False, "no-correlation-peak")
    values = np.asarray(offsets, dtype=np.int64)
    median = int(np.median(values))
    seconds = tuple(float(item) / sample_rate for item in values)
    agree_samples = max(1, int(agree_seconds * sample_rate))
    inliers = values[np.abs(values - median) <= agree_samples]
    mean_abs = abs(median) / sample_rate
    majority = len(inliers) >= max(3, (len(values) + 1) // 2)
    consistent = majority and mean_abs <= max_abs_seconds
    spread = float(np.max(np.abs(values - median)) / sample_rate)
    reason = "consistent" if consistent else f"spread={spread:.4f}s abs={mean_abs:.4f}s inliers={len(inliers)}/{len(values)}"
    return ClockOffset(median, median / sample_rate, seconds, consistent, reason)


def shift_close_talk(samples: np.ndarray, offset_samples: int, length: int) -> np.ndarray:
    """Place close-talk samples onto the mix timeline."""

    mono = downmix_mono(samples)
    out = np.zeros(length, dtype=np.float32)
    if offset_samples >= 0:
        src_start = 0
        dst_start = offset_samples
    else:
        src_start = -offset_samples
        dst_start = 0
    copy = min(mono.size - src_start, length - dst_start)
    if copy > 0 and src_start < mono.size and dst_start < length:
        out[dst_start : dst_start + copy] = mono[src_start : src_start + copy]
    return out


def rms_in_spans(samples: np.ndarray, sample_rate: int, spans: Sequence[tuple[int, int]]) -> float:
    """Return RMS energy inside nanosecond spans."""

    squares = 0.0
    count = 0
    for start_ns, end_ns in spans:
        start = _ns_to_frame(start_ns, sample_rate)
        stop = _ns_to_frame(end_ns, sample_rate)
        piece = samples[start:stop]
        if piece.size:
            squares += float(np.square(piece, dtype=np.float64).sum())
            count += piece.size
    if count <= 0:
        return 0.0
    return float(np.sqrt(squares / count))


def map_lavaliers(
    energy: Mapping[str, Mapping[str, float]],
    exclusive_seconds: Mapping[str, float],
    *,
    ratio_min: float = LAV_RATIO_MIN,
    min_exclusive_seconds: float = LAV_MIN_EXCLUSIVE_SECONDS,
) -> LavalierMapping:
    """Assign each lavalier to the speaker with the strongest exclusive energy."""

    channels = sorted(energy)
    speakers = sorted({speaker for row in energy.values() for speaker in row})
    if not channels or not speakers:
        return LavalierMapping((), tuple(speakers), False, 0.0, "no-channels")
    used_speakers: set[str] = set()
    used_channels: set[str] = set()
    assignments: list[LavalierAssignment] = []
    ratios: list[float] = []
    for channel in channels:
        ranked = sorted(
            ((speaker, energy[channel].get(speaker, 0.0)) for speaker in speakers),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked or ranked[0][1] <= 0:
            continue
        best_speaker, best_rms = ranked[0]
        second_rms = ranked[1][1] if len(ranked) > 1 else 0.0
        ratio = best_rms / second_rms if second_rms > 0 else float("inf")
        exclusive = float(exclusive_seconds.get(best_speaker, 0.0))
        if (
            best_speaker in used_speakers
            or channel in used_channels
            or ratio < ratio_min
            or exclusive < min_exclusive_seconds
        ):
            continue
        used_speakers.add(best_speaker)
        used_channels.add(channel)
        assignments.append(
            LavalierAssignment(
                speaker=best_speaker,
                channel=channel,
                ratio=float(ratio) if np.isfinite(ratio) else 99.0,
                exclusive_seconds=exclusive,
                rms_by_speaker={name: float(energy[channel].get(name, 0.0)) for name in speakers},
            )
        )
        ratios.append(assignments[-1].ratio)
    unmapped = tuple(item for item in speakers if item not in used_speakers)
    confidence = min(ratios) if ratios else 0.0
    use = bool(assignments) and not unmapped
    if use:
        reason = "unique-energy-map"
    elif assignments:
        reason = "partial-map"
    else:
        reason = "no-clear-channel"
    return LavalierMapping(tuple(assignments), unmapped, use, float(confidence), reason)


def voiced_spans(
    samples: np.ndarray,
    sample_rate: int,
    *,
    speech_rms: float,
    frame_ms: int = VOICED_FRAME_MS,
    hop_ms: int = VOICED_HOP_MS,
    abs_floor: float = VOICED_ABS_FLOOR,
    relative: float = VOICED_RELATIVE,
) -> list[tuple[int, int]]:
    """Return nanosecond spans where RMS exceeds the voicing floor."""

    mono = downmix_mono(samples)
    frame = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(sample_rate * hop_ms / 1000))
    threshold = max(abs_floor, relative * max(0.0, speech_rms))
    spans: list[tuple[int, int]] = []
    start_ns: int | None = None
    last_end_ns = 0
    for index in range(0, max(0, mono.size - frame + 1), hop):
        rms = float(np.sqrt(np.mean(np.square(mono[index : index + frame], dtype=np.float64))))
        frame_start_ns = (index * NANOSECONDS_PER_SECOND) // sample_rate
        frame_end_ns = ((index + frame) * NANOSECONDS_PER_SECOND) // sample_rate
        if rms >= threshold:
            if start_ns is None:
                start_ns = frame_start_ns
            last_end_ns = frame_end_ns
        elif start_ns is not None:
            if last_end_ns > start_ns:
                spans.append((start_ns, last_end_ns))
            start_ns = None
    if start_ns is not None and last_end_ns > start_ns:
        spans.append((start_ns, last_end_ns))
    return spans


def voicing_uncertain(
    recording_id: str,
    speaker: str,
    human: Interval,
    aligned: Sequence[tuple[int, int]],
    voiced: Sequence[tuple[int, int]],
) -> tuple[UncertainInterval, ...]:
    """Mark voiced close-talk silence inside a human segment as UNCERTAIN."""

    interior = subtract_spans([human], aligned)
    leftover = intersect_spans(interior, voiced)
    return tuple(
        UncertainInterval(recording_id, speaker, start_ns, end_ns) for start_ns, end_ns in leftover if end_ns > start_ns
    )


def voicing_thresholds() -> dict[str, float | int]:
    """Return the recorded energy-VAD thresholds."""

    return {
        "frame_ms": VOICED_FRAME_MS,
        "hop_ms": VOICED_HOP_MS,
        "abs_floor_rms": VOICED_ABS_FLOOR,
        "relative_to_speech_rms": VOICED_RELATIVE,
        "lavalier_ratio_min": LAV_RATIO_MIN,
        "lavalier_min_exclusive_seconds": LAV_MIN_EXCLUSIVE_SECONDS,
        "offset_windows": OFFSET_WINDOWS,
        "offset_window_seconds": OFFSET_WINDOW_SECONDS,
        "offset_max_abs_seconds": OFFSET_MAX_ABS_SECONDS,
        "offset_agree_seconds": OFFSET_AGREE_SECONDS,
    }


def _ns_to_frame(value_ns: int, sample_rate: int) -> int:
    return max(0, (int(value_ns) * sample_rate) // NANOSECONDS_PER_SECOND)
