"""Measure target loss on the production diarization window pipeline.

The acceptance layer owns source and release records.  This module owns the
model-capacity calculation so it can be imported by acceptance code without a
dependency back to that layer.  It deliberately accepts interval-like values
instead of importing ``RttmInterval``: the production parser and small tests
can both provide the labels without creating an import cycle.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .contracts import CAPACITY_LOSS_LIMIT, CAPACITY_PROFILES, DEFAULT_DATA_PROFILES
from .errors import PreparationError


DEFAULT_OUTPUT_FRAMES_8 = int(DEFAULT_DATA_PROFILES.output_frames_8)
DEFAULT_OUTPUT_FRAMES_16 = DEFAULT_OUTPUT_FRAMES_8 * 2 + 1
DEFAULT_RF_DURATION = float(DEFAULT_DATA_PROFILES.rf_duration)
DEFAULT_RF_STEP = float(DEFAULT_DATA_PROFILES.rf_step)
DEFAULT_CHUNK_SHIFTS = dict(zip(DEFAULT_DATA_PROFILES.chunk_seconds, DEFAULT_DATA_PROFILES.chunk_shifts))


@dataclass(frozen=True)
class CapacityReport:
    """Capacity loss for one window and target-encoder profile.

    ``speaker_seconds`` is the speaker-activity denominator on the production
    target frame grid.  ``uem_speaker_seconds`` keeps the exact unioned UEM
    duration available for audit comparison.  ``slot_lost_seconds`` and
    ``overlap_lost_seconds`` are independent diagnostics against the raw
    target.  ``encoded_lost_seconds`` is the loss after slot selection and the
    real Powerset encode/decode round trip; ``actual_lost_seconds`` includes
    both slot selection and that round-trip loss.
    """

    chunk_seconds: int
    max_overlap: int
    local_slots: int
    speaker_seconds: float
    lost_seconds: float
    loss_fraction: float
    admitted: bool
    slot_lost_seconds: float = 0.0
    overlap_lost_seconds: float = 0.0
    encoded_lost_seconds: float = 0.0
    encode_decode_lost_seconds: float = 0.0
    actual_lost_seconds: float = 0.0
    post_slot_overlap_lost_seconds: float = 0.0
    uem_speaker_seconds: float = 0.0
    chunks: int = 0
    frames: int = 0
    chunk_shift: int = 0
    short_clip_fallback: bool = False
    model_num_frames: int = 0
    model_rf_duration: float = DEFAULT_RF_DURATION
    model_rf_step: float = DEFAULT_RF_STEP

    @property
    def powerset_lost_seconds(self) -> float:
        """Return the loss attributable to Powerset encode/decode."""

        return self.encoded_lost_seconds

    @property
    def speaker_seconds_denominator(self) -> float:
        """Return the frame-grid denominator used for admission."""

        return self.speaker_seconds


def _interval_value(interval: Any, key: str) -> Any:
    if isinstance(interval, Mapping):
        return interval[key]
    return getattr(interval, key)


def _normalise_bounds(
    duration: float | None,
    *,
    uem: tuple[float, float] | None,
    uem_start: float | None,
    uem_end: float | None,
) -> tuple[float, float, float]:
    """Resolve audio and UEM bounds without changing absolute label times."""

    if uem is not None:
        if uem_start is not None or uem_end is not None:
            raise PreparationError("capacity UEM was supplied more than once")
        if len(uem) != 2:
            raise PreparationError("capacity UEM must contain start and end")
        uem_start, uem_end = float(uem[0]), float(uem[1])
    if duration is None:
        if uem_end is None:
            raise PreparationError("capacity measurement requires duration or UEM end")
        duration = float(uem_end)
    audio_duration = float(duration)
    start = 0.0 if uem_start is None else float(uem_start)
    end = audio_duration if uem_end is None else float(uem_end)
    if not all(math.isfinite(value) for value in (audio_duration, start, end)):
        raise PreparationError("capacity bounds must be finite")
    if audio_duration <= 0:
        raise PreparationError("capacity measurement requires positive duration")
    if start < 0 or end <= start:
        raise PreparationError("capacity UEM must have positive non-negative bounds")
    if end > audio_duration + 1e-6:
        raise PreparationError(
            "capacity UEM exceeds the decoded audio",
            {"uem_end": end, "duration": audio_duration},
        )
    return audio_duration, start, min(end, audio_duration)


def _union_intervals(
    intervals: Iterable[Any],
    *,
    uem_start: float,
    uem_end: float,
    recording_id: str | None,
) -> tuple[dict[str, list[tuple[float, float]]], float]:
    """Clip and union intervals per speaker before counting activity."""

    by_speaker: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for interval in intervals:
        if recording_id is not None:
            try:
                if str(_interval_value(interval, "recording_id")) != recording_id:
                    continue
            except (KeyError, AttributeError):
                raise PreparationError("capacity interval has no recording identity") from None
        try:
            start = float(_interval_value(interval, "start"))
            end = float(_interval_value(interval, "end"))
            speaker = str(_interval_value(interval, "speaker"))
        except (KeyError, AttributeError, TypeError, ValueError):
            raise PreparationError("capacity interval is malformed") from None
        if not all(math.isfinite(value) for value in (start, end)):
            raise PreparationError("capacity interval bounds must be finite")
        if not speaker:
            raise PreparationError("capacity interval speaker must be non-empty")
        clipped_start = max(start, uem_start)
        clipped_end = min(end, uem_end)
        if clipped_end <= clipped_start:
            continue
        by_speaker[speaker].append((clipped_start, clipped_end))

    merged: dict[str, list[tuple[float, float]]] = {}
    exact_seconds = 0.0
    for speaker, segments in by_speaker.items():
        segments.sort()
        combined: list[tuple[float, float]] = []
        for start, end in segments:
            if not combined or start > combined[-1][1]:
                combined.append((start, end))
                continue
            combined[-1] = (combined[-1][0], max(combined[-1][1], end))
        merged[speaker] = combined
        exact_seconds += sum(end - start for start, end in combined)
    return merged, exact_seconds


@lru_cache(maxsize=1)
def _load_dataset_module() -> Any:
    """Load the production dataset module without importing recipe runners."""

    dataset_path = Path(__file__).resolve().parents[2] / "diar_ssl" / "dataset.py"
    spec = importlib.util.spec_from_file_location("speakrs_target_capacity_dataset", dataset_path)
    if spec is None or spec.loader is None:
        raise PreparationError("cannot load the production data reader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_powerset_class() -> Any:
    """Load the repository's production Powerset implementation."""

    powerset_path = (
        Path(__file__).resolve().parents[3] / "pyannote-audio" / "pyannote" / "audio" / "utils" / "powerset.py"
    )
    spec = importlib.util.spec_from_file_location("speakrs_target_capacity_powerset", powerset_path)
    if spec is None or spec.loader is None:
        raise PreparationError("cannot load the production Powerset encoder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Powerset


def _frames_for_chunk(chunk_seconds: int, output_frames_8: int) -> int:
    if chunk_seconds == 8:
        return output_frames_8
    if chunk_seconds == 16:
        # production WavLM convolution has 399 frames for 8 s and 799 for
        # 16 s; the old restore helper used 2*n-1 (797), dropping two frames
        return output_frames_8 * 2 + 1
    raise PreparationError("capacity chunk_seconds must be 8 or 16", {"chunk_seconds": chunk_seconds})


def _rttm_line(recording_id: str, speaker: str, start: float, end: float) -> str:
    """Build one synthetic RTTM row for the target-only production reader."""

    return f"SPEAKER {recording_id} 1 {start:.17g} {end - start:.17g} <NA> <NA> {speaker} <NA> <NA>\n"


@contextmanager
def _target_only_dataset(
    dataset_module: Any,
    *,
    by_speaker: Mapping[str, Sequence[tuple[float, float]]],
    uem_start: float,
    uem_end: float,
    chunk_seconds: int,
    chunk_shift: int,
    model_num_frames: int,
    model_rf_duration: float,
    model_rf_step: float,
) -> Iterator[Any]:
    """Construct a production reader whose __getitem__ returns real targets.

    The reader still parses the synthetic RTTM with its own float32 structured
    annotation array and applies its own target discretization. Only waveform
    extraction is replaced because this capacity audit does not validate audio
    decoding.
    """

    class TargetOnlyDataset(dataset_module.DiarizationDataset):
        """Use production target construction without reading audio bytes."""

        def extract_wavforms(self, path, start, end, num_channels=8):
            """Return finite placeholder samples for the production collator."""

            return np.zeros((1, 1), dtype=np.float32)

    # production parser treats recording and speaker values as whitespace-
    # separated tokens. Stable synthetic tokens keep the target audit valid for
    # interval-like test values that contain punctuation or whitespace.
    synthetic_recording = "__capacity_recording__"
    lines: list[str] = []
    for speaker_index, (speaker, segments) in enumerate(sorted(by_speaker.items())):
        synthetic_speaker = f"__capacity_speaker_{speaker_index}__"
        for start, end in segments:
            lines.append(_rttm_line(synthetic_recording, synthetic_speaker, start, end))
    if not lines:
        # rttm2label requires at least one row to derive its structured dtype;
        # keep the dummy outside every measured production chunk.
        lines.append(_rttm_line(synthetic_recording, "__capacity_empty__", uem_end + 1.0, uem_end + 1.1))

    with tempfile.TemporaryDirectory(prefix="speakrs-capacity-") as root:
        work = Path(root)
        wav_scp = work / "wav.scp"
        rttm = work / "targets.rttm"
        uem = work / "targets.uem"
        wav_scp.write_text(f"{synthetic_recording} synthetic.wav\n", encoding="utf-8")
        rttm.write_text("".join(lines), encoding="utf-8")
        uem.write_text(f"{synthetic_recording} 1 {uem_start:.17g} {uem_end:.17g}\n", encoding="utf-8")
        try:
            dataset = TargetOnlyDataset(
                str(wav_scp),
                str(rttm),
                str(uem),
                model_num_frames=model_num_frames,
                model_rf_duration=model_rf_duration,
                model_rf_step=model_rf_step,
                chunk_size=chunk_seconds,
                chunk_shift=chunk_shift,
            )
        except (AssertionError, OSError, TypeError, ValueError) as exc:
            raise PreparationError(
                "capacity UEM cannot produce a production window",
                {
                    "chunk_seconds": chunk_seconds,
                    "chunk_shift": chunk_shift,
                    "uem_start": uem_start,
                    "uem_end": uem_end,
                    "reason": str(exc),
                },
            ) from None
        yield dataset


def _collate_target(dataset_module: Any, sample: tuple[Any, np.ndarray, str], local_slots: int) -> np.ndarray:
    """Run the production slot selection and return its padded target."""

    try:
        collated = dataset_module._collate_fn(
            [sample],
            max_speakers_per_chunk=local_slots,
        )
        result = np.asarray(collated["ts"][0])
    except (KeyError, TypeError, ValueError):
        raise PreparationError("production target collation failed") from None
    target = np.asarray(sample[1])
    if result.shape != (target.shape[0], local_slots):
        raise PreparationError(
            "production target collation returned the wrong shape",
            {"actual": list(result.shape), "expected": [target.shape[0], local_slots]},
        )
    return result.astype(np.float32, copy=False)


def _roundtrip_target(target: np.ndarray, powerset: Any) -> np.ndarray:
    """Encode and decode one target with the actual Powerset class."""

    try:
        import torch

        tensor = torch.from_numpy(target).float().unsqueeze(0)
        encoded = powerset.to_powerset(tensor)
        decoded = powerset.to_multilabel(encoded)
        result = decoded.detach().cpu().numpy()[0]
    except (AttributeError, RuntimeError, TypeError, ValueError):
        raise PreparationError("production Powerset target round-trip failed") from None
    if result.shape != target.shape or not np.isfinite(result).all():
        raise PreparationError(
            "production Powerset target round-trip returned invalid values",
            {"actual": list(result.shape), "expected": list(target.shape)},
        )
    return np.asarray(result, dtype=np.float32)


def measure_capacity_loss(
    intervals: Sequence[Any],
    *,
    duration: float | None = None,
    chunk_seconds: int,
    max_overlap: int,
    local_slots: int,
    uem: tuple[float, float] | None = None,
    uem_start: float | None = None,
    uem_end: float | None = None,
    chunk_shift: int | Mapping[int, int] | None = None,
    model_num_frames: int | Mapping[int, int] | None = None,
    model_rf_duration: float = DEFAULT_RF_DURATION,
    model_rf_step: float = DEFAULT_RF_STEP,
    frame_step: float | None = None,
    output_frames_8: int = DEFAULT_OUTPUT_FRAMES_8,
    recording_id: str | None = None,
) -> CapacityReport:
    """Measure slot, overlap, and actual Powerset target loss.

    Intervals remain immutable.  Same-speaker intervals are unioned before
    frame construction, while different speakers remain simultaneous.  The
    production chunk generation and collation are required.  A short explicit
    UEM that the production generator rejects fails instead of being treated as
    a supported training window.
    """

    chunk_seconds = int(chunk_seconds)
    max_overlap = int(max_overlap)
    local_slots = int(local_slots)
    if chunk_seconds not in {8, 16}:
        raise PreparationError("capacity chunk_seconds must be 8 or 16", {"chunk_seconds": chunk_seconds})
    if max_overlap < 1 or max_overlap > local_slots:
        raise PreparationError(
            "capacity max_overlap must be between one and local_slots",
            {"max_overlap": max_overlap, "local_slots": local_slots},
        )
    if local_slots < 1:
        raise PreparationError("capacity local_slots must be positive")
    if frame_step is not None:
        model_rf_step = float(frame_step)
    if model_rf_duration <= 0 or model_rf_step <= 0:
        raise PreparationError("capacity receptive-field settings must be positive")
    if output_frames_8 < 1:
        raise PreparationError("capacity output_frames_8 must be positive")
    if isinstance(chunk_shift, Mapping):
        chunk_shift = chunk_shift.get(chunk_seconds)
    if chunk_shift is None:
        chunk_shift = DEFAULT_CHUNK_SHIFTS[chunk_seconds]
    chunk_shift = int(chunk_shift)
    if chunk_shift < 1:
        raise PreparationError("capacity chunk_shift must be positive")

    _, start, end = _normalise_bounds(
        duration,
        uem=uem,
        uem_start=uem_start,
        uem_end=uem_end,
    )
    by_speaker, uem_speaker_seconds = _union_intervals(
        intervals,
        uem_start=start,
        uem_end=end,
        recording_id=recording_id,
    )
    module = _load_dataset_module()
    if isinstance(model_num_frames, Mapping):
        model_num_frames = model_num_frames.get(chunk_seconds)
    frames = (
        _frames_for_chunk(chunk_seconds, int(output_frames_8)) if model_num_frames is None else int(model_num_frames)
    )
    if frames < 1:
        raise PreparationError("capacity model_num_frames must be positive")
    with _target_only_dataset(
        module,
        by_speaker=by_speaker,
        uem_start=start,
        uem_end=end,
        chunk_seconds=chunk_seconds,
        chunk_shift=chunk_shift,
        model_num_frames=frames,
        model_rf_duration=model_rf_duration,
        model_rf_step=model_rf_step,
    ) as dataset:
        if len(dataset) == 0:
            raise PreparationError(
                "capacity UEM cannot produce a production window",
                {
                    "chunk_seconds": chunk_seconds,
                    "chunk_shift": chunk_shift,
                    "uem_start": start,
                    "uem_end": end,
                },
            )

        powerset = _load_powerset_class()(local_slots, max_overlap)
        total_raw = 0.0
        total_slot_lost = 0.0
        total_overlap_lost = 0.0
        total_post_slot_overlap_lost = 0.0
        total_encoded_lost = 0.0
        total_actual_lost = 0.0
        measured_chunks = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            target = np.asarray(sample[1])
            if target.ndim != 2 or target.shape[0] != frames:
                raise PreparationError(
                    "production target reader returned the wrong shape",
                    {"actual": list(target.shape), "expected_frames": frames},
                )
            raw_counts = target.sum(axis=1, dtype=np.float64)
            collated = _collate_target(module, sample, local_slots)
            collated_counts = collated.sum(axis=1, dtype=np.float64)
            decoded = _roundtrip_target(collated, powerset)

            raw_total = float(target.sum())
            collated_total = float(collated.sum())
            decoded_total = float(decoded.sum())
            total_raw += raw_total
            total_slot_lost += max(0.0, raw_total - collated_total)
            total_overlap_lost += float(np.maximum(raw_counts - max_overlap, 0.0).sum())
            total_post_slot_overlap_lost += float(np.maximum(collated_counts - max_overlap, 0.0).sum())
            # raw target has one column per speaker; collated and decoded targets
            # have exactly local_slots columns, so compare totals instead of
            # broadcasting columns and counting retained speakers per padded slot
            encoded_lost = max(
                float(np.maximum(collated - decoded, 0.0).sum()),
                collated_total - decoded_total,
                0.0,
            )
            actual_lost = max(raw_total - decoded_total, 0.0)
            total_encoded_lost += encoded_lost
            total_actual_lost += actual_lost
            measured_chunks += 1

    frame_seconds = float(model_rf_step)
    speaker_seconds = total_raw * frame_seconds
    slot_lost_seconds = total_slot_lost * frame_seconds
    overlap_lost_seconds = total_overlap_lost * frame_seconds
    post_slot_overlap_lost_seconds = total_post_slot_overlap_lost * frame_seconds
    encoded_lost_seconds = total_encoded_lost * frame_seconds
    actual_lost_seconds = total_actual_lost * frame_seconds
    loss_fraction = 0.0 if speaker_seconds <= 0 else actual_lost_seconds / speaker_seconds
    return CapacityReport(
        chunk_seconds=chunk_seconds,
        max_overlap=max_overlap,
        local_slots=local_slots,
        speaker_seconds=speaker_seconds,
        lost_seconds=actual_lost_seconds,
        loss_fraction=loss_fraction,
        admitted=loss_fraction <= CAPACITY_LOSS_LIMIT,
        slot_lost_seconds=slot_lost_seconds,
        overlap_lost_seconds=overlap_lost_seconds,
        encoded_lost_seconds=encoded_lost_seconds,
        encode_decode_lost_seconds=encoded_lost_seconds,
        actual_lost_seconds=actual_lost_seconds,
        post_slot_overlap_lost_seconds=post_slot_overlap_lost_seconds,
        uem_speaker_seconds=uem_speaker_seconds,
        chunks=measured_chunks,
        frames=frames,
        chunk_shift=chunk_shift,
        short_clip_fallback=False,
        model_num_frames=frames,
        model_rf_duration=float(model_rf_duration),
        model_rf_step=float(model_rf_step),
    )


def admit_profiles(
    intervals: Sequence[Any],
    *,
    duration: float | None = None,
    profiles: Sequence[Mapping[str, Any]] | None = None,
    uem: tuple[float, float] | None = None,
    uem_start: float | None = None,
    uem_end: float | None = None,
    chunk_shift: int | Mapping[int, int] | None = None,
    model_num_frames: int | Mapping[int, int] | None = None,
    model_rf_duration: float = DEFAULT_RF_DURATION,
    model_rf_step: float = DEFAULT_RF_STEP,
    frame_step: float | None = None,
    output_frames_8: int = DEFAULT_OUTPUT_FRAMES_8,
    recording_id: str | None = None,
) -> list[CapacityReport]:
    """Evaluate the four required window/overlap profiles independently."""

    selected = tuple(profiles or CAPACITY_PROFILES)
    reports: list[CapacityReport] = []
    for profile in selected:
        reports.append(
            measure_capacity_loss(
                intervals,
                duration=duration,
                chunk_seconds=int(profile["chunk_seconds"]),
                max_overlap=int(profile["max_overlap"]),
                local_slots=int(profile.get("local_slots", 4)),
                uem=uem,
                uem_start=uem_start,
                uem_end=uem_end,
                chunk_shift=profile.get("chunk_shift", chunk_shift),
                model_num_frames=profile.get("model_num_frames", model_num_frames),
                model_rf_duration=model_rf_duration,
                model_rf_step=model_rf_step,
                frame_step=frame_step,
                output_frames_8=output_frames_8,
                recording_id=recording_id,
            )
        )
    return reports


__all__ = ["CapacityReport", "admit_profiles", "measure_capacity_loss"]
