"""Close-talk clock offset, lavalier mapping, and energy voicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

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
LAV_WINNER_FRACTION_MIN = 0.80
LAV_ELIMINATION_FRACTION_MIN = 0.50
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
    method: Literal["energy", "elimination"]
    winner_fraction: float
    ratio: float
    exclusive_seconds: float
    rms_by_speaker: dict[str, float]


@dataclass(frozen=True, slots=True)
class LavalierSpeakerDecision:
    """Record the mapping evidence for one speaker, including ambiguity."""

    speaker: str
    channel: str | None
    candidate_channel: str | None
    method: Literal["energy", "elimination"] | None
    winner_fraction: float
    ratio: float
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class LavalierMapping:
    """Per-recording lavalier-to-speaker map, or a decision to use the mix."""

    assignments: tuple[LavalierAssignment, ...]
    unmapped_speakers: tuple[str, ...]
    use_close_talk: bool
    confidence: float
    reason: str
    speaker_decisions: tuple[LavalierSpeakerDecision, ...] = ()

    def channel_for(self, speaker: str) -> str | None:
        """Return the mapped lavalier name for a speaker."""

        for item in self.assignments:
            if item.speaker == speaker:
                return item.channel
        return None


@dataclass(frozen=True, slots=True)
class _SpeakerEnergyEvidence:
    """Normalized energy evidence used while selecting one speaker map."""

    candidate_channel: str | None
    pooled: Mapping[str, float]
    ratio: float
    winner_fraction: float
    winner_fractions: Mapping[str, float]


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
    # positive offsets mean the source lags the mix, so trim its delayed prefix
    src_start = max(0, offset_samples)
    dst_start = max(0, -offset_samples)
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
    interval_rms: Mapping[str, Mapping[str, Sequence[float]]],
    interval_seconds: Mapping[str, Sequence[float]],
) -> LavalierMapping:
    """Assign lavaliers from gain-normalized, per-interval speaker energy."""

    channels = sorted(interval_rms)
    speakers = sorted(set(interval_seconds) | {speaker for row in interval_rms.values() for speaker in row})
    if not channels or not speakers:
        decisions = tuple(
            LavalierSpeakerDecision(speaker, None, None, None, 0.0, 0.0, True)
            for speaker in speakers
        )
        return LavalierMapping((), tuple(speakers), False, 0.0, "no-channels", decisions)

    normalized = _normalize_interval_rms(interval_rms, channels, speakers)
    evidence: dict[str, _SpeakerEnergyEvidence] = {}
    for speaker in speakers:
        evidence[speaker] = _speaker_energy_evidence(
            speaker,
            channels,
            normalized,
            interval_seconds.get(speaker, ()),
        )
    eligible: dict[str, str] = {}
    channel_claims: dict[str, list[str]] = {}
    for speaker, item in evidence.items():
        candidate = item.candidate_channel
        if candidate is None or item.winner_fraction < LAV_WINNER_FRACTION_MIN:
            continue
        eligible[speaker] = candidate
        channel_claims.setdefault(candidate, []).append(speaker)

    selected: dict[str, tuple[str, Literal["energy", "elimination"], float]] = {}
    for speaker, channel in eligible.items():
        if len(channel_claims[channel]) == 1:
            selected[speaker] = (channel, "energy", evidence[speaker].winner_fraction)

    unmapped_speakers = [speaker for speaker in speakers if speaker not in selected]
    used_channels = {channel for channel, _, _ in selected.values()}
    remaining_channels = [channel for channel in channels if channel not in used_channels]
    if len(unmapped_speakers) == 1 and len(remaining_channels) == 1:
        speaker = unmapped_speakers[0]
        channel = remaining_channels[0]
        # compare against every channel so the sole remaining one cannot win by default
        fraction = evidence[speaker].winner_fractions.get(channel, 0.0)
        if fraction >= LAV_ELIMINATION_FRACTION_MIN:
            selected[speaker] = (channel, "elimination", fraction)

    assignments = tuple(
        LavalierAssignment(
            speaker=speaker,
            channel=channel,
            method=method,
            winner_fraction=winner_fraction,
            ratio=evidence[speaker].ratio,
            exclusive_seconds=float(sum(interval_seconds.get(speaker, ()))),
            rms_by_speaker={name: evidence[name].pooled.get(channel, 0.0) for name in speakers},
        )
        for speaker, (channel, method, winner_fraction) in sorted(selected.items())
    )
    assigned_by_speaker = {item.speaker: item for item in assignments}
    decisions = tuple(
        LavalierSpeakerDecision(
            speaker=speaker,
            channel=assigned_by_speaker[speaker].channel if speaker in assigned_by_speaker else None,
            candidate_channel=evidence[speaker].candidate_channel,
            method=assigned_by_speaker[speaker].method if speaker in assigned_by_speaker else None,
            winner_fraction=(
                assigned_by_speaker[speaker].winner_fraction
                if speaker in assigned_by_speaker
                else evidence[speaker].winner_fraction
            ),
            ratio=evidence[speaker].ratio,
            ambiguous=speaker not in assigned_by_speaker,
        )
        for speaker in speakers
    )
    unmapped = tuple(item for item in speakers if item not in assigned_by_speaker)
    confidence = min((item.ratio for item in assignments), default=0.0)
    use = bool(assignments)
    if use and not unmapped:
        reason = (
            "unique-energy-map"
            if all(item.method == "energy" for item in assignments)
            else "complete-map"
        )
    elif use:
        reason = "partial-map"
    else:
        reason = "no-clear-channel"
    return LavalierMapping(assignments, unmapped, use, float(confidence), reason, decisions)


def _normalize_interval_rms(
    interval_rms: Mapping[str, Mapping[str, Sequence[float]]],
    channels: Sequence[str],
    speakers: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Normalize each channel by its median interval RMS."""

    normalized: dict[str, dict[str, np.ndarray]] = {}
    for channel in channels:
        rows = {
            speaker: np.asarray(interval_rms[channel].get(speaker, ()), dtype=np.float64)
            for speaker in speakers
        }
        finite_rows = [row[np.isfinite(row) & (row >= 0)] for row in rows.values()]
        observed_rows = [row for row in finite_rows if row.size]
        observed = np.concatenate(observed_rows) if observed_rows else np.array([])
        median = float(np.median(observed)) if observed.size else 0.0
        if median <= 1e-12:
            normalized[channel] = {speaker: np.zeros_like(row) for speaker, row in rows.items()}
            continue
        normalized[channel] = {
            speaker: np.where(np.isfinite(row) & (row >= 0), row / median, 0.0)
            for speaker, row in rows.items()
        }
    return normalized


def _speaker_energy_evidence(
    speaker: str,
    channels: Sequence[str],
    normalized: Mapping[str, Mapping[str, np.ndarray]],
    interval_seconds: Sequence[float],
) -> _SpeakerEnergyEvidence:
    """Return pooled energy, the unique top channel, and interval win fractions."""

    durations = np.asarray(interval_seconds, dtype=np.float64)
    lengths = [durations.size]
    lengths.extend(normalized[channel].get(speaker, np.array([])).size for channel in channels)
    available = min(lengths)
    durations = durations[:available]
    valid = np.isfinite(durations) & (durations > 0)
    durations = durations[valid]
    values = {
        channel: normalized[channel].get(speaker, np.array([]))[:available][valid]
        for channel in channels
    }
    total_duration = float(durations.sum())
    pooled = {
        channel: (
            float(np.sqrt(np.sum(np.square(values[channel], dtype=np.float64) * durations) / total_duration))
            if total_duration > 0
            else 0.0
        )
        for channel in channels
    }
    win_counts = dict.fromkeys(channels, 0)
    if durations.size:
        for index in range(durations.size):
            winner = _unique_maximum({channel: float(values[channel][index]) for channel in channels})
            if winner is not None:
                win_counts[winner] += 1
    winner_fractions = {
        channel: count / durations.size if durations.size else 0.0
        for channel, count in win_counts.items()
    }
    ranked = sorted(pooled.values(), reverse=True)
    ratio = _energy_ratio(ranked[0], ranked[1] if len(ranked) > 1 else 0.0)
    candidate = _unique_maximum(pooled)
    if candidate is None:
        winner_fraction = 0.0
        candidate_channel = None
    else:
        winner_fraction = winner_fractions[candidate]
        candidate_channel = candidate
    return _SpeakerEnergyEvidence(candidate_channel, pooled, ratio, winner_fraction, winner_fractions)


def _unique_maximum(values: Mapping[str, float]) -> str | None:
    """Return the sole positive maximum, or None for a tie or empty evidence."""

    if not values:
        return None
    best = max(values.values())
    if best <= 0:
        return None
    winners = [name for name, value in values.items() if np.isclose(value, best, rtol=1e-9, atol=1e-12)]
    return winners[0] if len(winners) == 1 else None


def _energy_ratio(best: float, second: float) -> float:
    """Return a bounded top-to-runner-up ratio for JSON-safe evidence."""

    if best <= 0:
        return 0.0
    if second <= 0:
        return 99.0
    return float(best / second)


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
        "lavalier_winner_fraction_min": LAV_WINNER_FRACTION_MIN,
        "lavalier_elimination_fraction_min": LAV_ELIMINATION_FRACTION_MIN,
        "offset_windows": OFFSET_WINDOWS,
        "offset_window_seconds": OFFSET_WINDOW_SECONDS,
        "offset_max_abs_seconds": OFFSET_MAX_ABS_SECONDS,
        "offset_agree_seconds": OFFSET_AGREE_SECONDS,
    }


def _ns_to_frame(value_ns: int, sample_rate: int) -> int:
    return max(0, (int(value_ns) * sample_rate) // NANOSECONDS_PER_SECOND)
