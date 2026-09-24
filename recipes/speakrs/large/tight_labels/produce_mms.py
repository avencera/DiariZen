"""Produce tight-v1 MMS releases for Indic DiarBench and LOTUSDIS."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from ..hashing import sha256_paths
from ..indic_diarbench import canonical_speaker_id
from ..jsonio import read_json, write_json
from ..storage import require_free_gib
from . import GATE_POLICY_VERSION, LIBRARY_VERSION
from .classify import (
    AlignmentFailureReason,
    ByDesignReason,
    SegmentClass,
    SegmentOutcome,
    unioned_original_segments,
)
from .close_talk import (
    ClockOffset,
    LavalierMapping,
    map_lavaliers,
    measure_clock_offset,
    resample_audio,
    rms_in_spans,
    shift_close_talk,
    voiced_spans,
    voicing_thresholds,
    voicing_uncertain,
)
from .domain import Interval, RecordingLabels, SpeakerSegment, UncertainInterval
from .gates import GateResult, evaluate_gates
from .intervals import clip_recording_to_uem, duration_ns, overlap_duration_ns
from .listening import write_listening_pack
from .lotus_text import LotusAlignInterval, parse_lotus_align_intervals
from .markers import cleaning_stats_by_language, prepare_transcript
from .merge import MIN_GAP_NS, MIN_GAP_SECONDS, merge_recording
from .mms import (
    INDIC_UROMAN_LCODES,
    MMS_SAMPLE_RATE,
    THAI_UROMAN_LCODE,
    AlignedWord,
    MmsAligner,
    measure_language_coverage,
)
from .produce import RecordingRef, library_commit, release_id_with_version, render_gate_markdown
from .tighten import tighten_recording, tighten_segment
from .time import NANOSECONDS_PER_SECOND, float_bounds_to_rttm_ns, format_ns, ns_to_seconds
from .writers import (
    load_written_recording_labels,
    parse_speaker_rttm,
    render_speaker_rttm,
    rewrite_uem_file_id,
    uem_bounds,
    write_recording_labels,
)


REPO_DIR = Path(__file__).resolve().parents[4]
DV_ROOT = Path("/Volumes/CacheDisk/dev-cache/diarization-data-verification")
UNION_ROOT = DV_ROOT / "next-run/indic-diarbench-union-v1"
TRAINING_VIEW = DV_ROOT / "assets/commercial-data-preparation-v1/final-union-training-v1"
ANNOTATIONS_PATH = TRAINING_VIEW / "annotations.rttm-uem.txt"
ACQUISITION_STAGING = DV_ROOT / "staging/tight-labels-v1"
STAGING_ROOT = ACQUISITION_STAGING
WORK_ROOT = STAGING_ROOT / "align-mms-work"
RELEASE_ROOT = DV_ROOT / "releases"
INDIC_PARQUET_ROOT = DV_ROOT / "sources/indic-diarbench-92877bad8aab6e598167d91c6ee02aa8ca6ede09"
LOTUS_TEXTGRID_ROOT = Path("/Volumes/CacheDisk/dev-cache/wavlm-large-pre-rental/source/lotusdis/extracted/textgrid")
CLOSE_TALK_ROOT = ACQUISITION_STAGING / "close-talk/lotusdis"
MMS_MODEL_PATH = ACQUISITION_STAGING / "models/mms/ctc_alignment_mling_uroman.pt"
MINIMUM_FREE_GIB = 120.0
SAMPLE_RATE = 16_000
PROGRESS_LOG_NAME = "progress.log"

SOURCE_SPECS = {
    "indic-diarbench": {
        "release_id": "indic-diarbench-tight-v1",
        "source_name": "Indic DiarBench",
        "prefix": "indic-diarbench::",
        "listening_name": "indic-diarbench",
        "license": "CC BY 4.0",
        "method": "mms-fa-uroman",
        "require_subset": True,
    },
    "lotusdis": {
        "release_id": "lotusdis-tight-v1",
        "source_name": "LOTUSDIS",
        "prefix": "lotusdis::",
        "listening_name": "lotusdis",
        "license": "CC BY-SA 4.0",
        "method": "mms-fa-uroman",
        "require_subset": True,
    },
}


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One human transcript span matched onto an original RTTM row."""

    speaker: str
    start_ns: int
    end_ns: int
    text: str
    language: str
    source_speaker: str
    overlap: bool = False
    kind: str = "speech"


@dataclass(frozen=True, slots=True)
class MatchReport:
    """Parquet or TextGrid rows matched onto original RTTM segments."""

    matched: int
    original: int
    unmatched_original: int
    unmatched_transcript: int
    mismatches: tuple[str, ...]
    by_language: dict[str, dict[str, int]] = field(default_factory=dict)


def progress_log_path() -> Path:
    """Return the append-only unattended progress log."""

    return WORK_ROOT / PROGRESS_LOG_NAME


def _append_progress(line: str) -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    with progress_log_path().open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{stamp} {message}"
    print(line, flush=True)
    _append_progress(line)


def _write_terminal_status(status: str) -> None:
    print(status, flush=True)
    _append_progress(status)


def load_indic_refs() -> tuple[RecordingRef, ...]:
    """Load Indic training audio and UEM from the Indic union."""

    manifest = read_json(UNION_ROOT / "training-manifest.json")
    uem_text = (UNION_ROOT / "all.uem").read_text(encoding="utf-8")
    uem_by_id: dict[str, tuple[int, int]] = {}
    uem_lines: dict[str, str] = {}
    for raw in uem_text.splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        file_id = raw.split()[0]
        if file_id.startswith("indic-diarbench::"):
            uem_by_id[file_id] = uem_bounds(raw + "\n")
            uem_lines[file_id] = raw + "\n"
    refs: list[RecordingRef] = []
    for record in manifest["recordings"]:
        if record.get("source") != "Indic DiarBench":
            continue
        recording_id = record["recording_id"]
        start_ns, end_ns = uem_by_id[recording_id]
        duration_ns = int(record["duration_samples"]) * NANOSECONDS_PER_SECOND // SAMPLE_RATE
        refs.append(
            RecordingRef(
                recording_id=recording_id,
                audio_path=UNION_ROOT / record["audio_path"],
                uem_path=UNION_ROOT / "all.uem",
                duration_ns=duration_ns,
                uem_start_ns=start_ns,
                uem_end_ns=end_ns,
            )
        )
    refs.sort(key=lambda item: item.recording_id)
    return tuple(refs)


def load_lotusdis_refs() -> tuple[RecordingRef, ...]:
    """Load LOTUSDIS training audio and UEM from the admitted training view."""

    payload = read_json(TRAINING_VIEW / "components/lotusdis.json")
    refs: list[RecordingRef] = []
    for record in payload["records"]:
        uem_path = Path(record["uem"]["source_absolute_path"])
        start_ns, end_ns = uem_bounds(uem_path.read_text(encoding="utf-8"))
        duration_ns = int(record["duration_samples"]) * NANOSECONDS_PER_SECOND // SAMPLE_RATE
        refs.append(
            RecordingRef(
                recording_id=record["recording_id"],
                audio_path=Path(record["audio"]["source_absolute_path"]),
                uem_path=uem_path,
                duration_ns=duration_ns,
                uem_start_ns=start_ns,
                uem_end_ns=end_ns,
            )
        )
    refs.sort(key=lambda item: item.recording_id)
    return tuple(refs)


def load_original_labels(prefix: str, recording_ids: Iterable[str] | None = None) -> dict[str, list[SpeakerSegment]]:
    """Parse SPEAKER rows for one source prefix."""

    wanted = set(recording_ids) if recording_ids is not None else None
    grouped: dict[str, list[SpeakerSegment]] = defaultdict(list)
    path = UNION_ROOT / "all.rttm" if prefix.startswith("indic-diarbench") else ANNOTATIONS_PATH
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.startswith(f"SPEAKER {prefix}"):
                continue
            for segment in parse_speaker_rttm(raw):
                if wanted is not None and segment.recording_id not in wanted:
                    continue
                grouped[segment.recording_id].append(segment)
    return grouped


def load_indic_transcripts() -> dict[str, list[TranscriptSegment]]:
    """Read parquet transcripts without loading embedded audio."""

    by_sample: dict[str, list[TranscriptSegment]] = {}
    for language_dir in sorted(path for path in INDIC_PARQUET_ROOT.iterdir() if path.is_dir()):
        parquet_path = language_dir / "test-00000-of-00001.parquet"
        if not parquet_path.is_file():
            continue
        table = pq.read_table(
            parquet_path,
            columns=["sample_id", "language", "annotated_transcript"],
        )
        payload = table.to_pydict()
        for sample_id, language, rows in zip(
            payload["sample_id"], payload["language"], payload["annotated_transcript"]
        ):
            segments: list[TranscriptSegment] = []
            for item in rows:
                start = item["start_time"]
                end = item["end_time"]
                if (
                    not isinstance(start, (int, float))
                    or not isinstance(end, (int, float))
                    or isinstance(start, bool)
                    or isinstance(end, bool)
                    or end <= start
                    or start != start
                    or end != end
                ):
                    continue
                start_ns, end_ns = float_bounds_to_rttm_ns(start, end)
                segments.append(
                    TranscriptSegment(
                        speaker=canonical_speaker_id(item["speaker_id"]),
                        start_ns=start_ns,
                        end_ns=end_ns,
                        text=item["transcript"],
                        language=language,
                        source_speaker=item["speaker_id"],
                    )
                )
            by_sample[f"indic-diarbench::{sample_id}"] = segments
    return by_sample


def match_segments(
    original: Sequence[SpeakerSegment],
    transcripts: Sequence[TranscriptSegment],
) -> tuple[dict[tuple[str, int, int], TranscriptSegment], MatchReport]:
    """Match transcript rows onto original RTTM rows by speaker and times."""

    remaining = list(transcripts)
    matched: dict[tuple[str, int, int], TranscriptSegment] = {}
    unmatched_original: list[str] = []
    for segment in original:
        key = (segment.speaker, segment.start_ns, segment.end_ns)
        index = next(
            (
                position
                for position, item in enumerate(remaining)
                if item.speaker == segment.speaker
                and item.start_ns == segment.start_ns
                and item.end_ns == segment.end_ns
            ),
            None,
        )
        if index is None:
            index = next(
                (
                    position
                    for position, item in enumerate(remaining)
                    if item.speaker == segment.speaker and item.start_ns == segment.start_ns
                ),
                None,
            )
        if index is None:
            unmatched_original.append(f"{segment.speaker} {format_ns(segment.start_ns)} {format_ns(segment.end_ns)}")
            continue
        matched[key] = remaining.pop(index)
    mismatches = tuple(unmatched_original[:20])
    unmatched_transcript = [
        f"{item.source_speaker} {format_ns(item.start_ns)} {format_ns(item.end_ns)}" for item in remaining
    ]
    return matched, MatchReport(
        matched=len(matched),
        original=len(original),
        unmatched_original=len(unmatched_original),
        unmatched_transcript=len(remaining),
        mismatches=tuple(list(mismatches) + unmatched_transcript[:20]),
    )


def recording_language(recording_id: str, transcripts: Sequence[TranscriptSegment] | None = None) -> str:
    """Return the parquet language, or the recording-id language stem."""

    if transcripts:
        return transcripts[0].language
    stem = recording_id.split("::", 1)[-1]
    return stem.split("_", 1)[0].title()


def held_out_labels(original: RecordingLabels) -> RecordingLabels:
    """Mark every human-segment interior UNCERTAIN with the library fallback."""

    speakers = frozenset(item.speaker for item in original.segments)
    return tighten_recording(original, {}, failed_speakers=speakers)


def overlap_uncertain_regions(
    recording_id: str,
    intervals: Sequence[LotusAlignInterval],
) -> tuple[UncertainInterval, ...]:
    """Mark each listed speaker UNCERTAIN on multi-speaker TextGrid intervals."""

    rows: list[UncertainInterval] = []
    for interval in intervals:
        if interval.non_speech or not interval.overlap:
            continue
        for speaker in interval.speakers:
            rows.append(UncertainInterval(recording_id, speaker, interval.start_ns, interval.end_ns))
    return tuple(rows)


def apply_extra_uncertain(
    labels: RecordingLabels,
    extra: Sequence[UncertainInterval],
) -> RecordingLabels:
    """Add UNCERTAIN regions; RecordingLabels unions them and cuts active speech."""

    if not extra:
        return labels
    return RecordingLabels(
        labels.recording_id,
        labels.segments,
        tuple(labels.uncertain) + tuple(extra),
    )


def _classes_from_row_outcomes(
    original: RecordingLabels,
    row_outcomes: Mapping[tuple[str, int, int], SegmentClass],
) -> tuple[SegmentClass, ...]:
    classes: list[SegmentClass] = []
    for segment in unioned_original_segments([original]):
        hits = [
            item
            for key, item in row_outcomes.items()
            if key[0] == segment.speaker and key[1] < segment.end_ns and key[2] > segment.start_ns
        ]
        if not hits:
            classes.append(
                SegmentClass.from_segment(
                    segment,
                    SegmentOutcome.ALIGNMENT_FAILURE,
                    failure_reason=AlignmentFailureReason.ALIGNER_ERROR,
                )
            )
            continue
        by_design = next((item for item in hits if item.outcome is SegmentOutcome.BY_DESIGN_UNCERTAIN), None)
        if by_design is not None:
            classes.append(
                SegmentClass.from_segment(
                    segment,
                    SegmentOutcome.BY_DESIGN_UNCERTAIN,
                    by_design_reason=by_design.by_design_reason,
                )
            )
            continue
        failed = next((item for item in hits if item.outcome is SegmentOutcome.ALIGNMENT_FAILURE), None)
        if failed is not None:
            classes.append(
                SegmentClass.from_segment(
                    segment,
                    SegmentOutcome.ALIGNMENT_FAILURE,
                    failure_reason=failed.failure_reason,
                )
            )
            continue
        classes.append(SegmentClass.from_segment(segment, SegmentOutcome.ALIGNED))
    return tuple(classes)


def _held_out_classes(original: RecordingLabels) -> tuple[SegmentClass, ...]:
    return tuple(
        SegmentClass.from_segment(
            segment,
            SegmentOutcome.BY_DESIGN_UNCERTAIN,
            by_design_reason=ByDesignReason.HELD_OUT_LANGUAGE,
        )
        for segment in unioned_original_segments([original])
    )


def _mark_low_coverage(
    classes: Sequence[SegmentClass],
    produced: RecordingLabels,
) -> tuple[SegmentClass, ...]:
    adjusted: list[SegmentClass] = []
    for item in classes:
        if item.outcome is SegmentOutcome.ALIGNED:
            covered = overlap_duration_ns(
                [(item.start_ns, item.end_ns)],
                produced.segments_for(item.speaker),
            )
            if covered <= 0:
                adjusted.append(
                    SegmentClass(
                        item.recording_id,
                        item.speaker,
                        item.start_ns,
                        item.end_ns,
                        SegmentOutcome.ALIGNMENT_FAILURE,
                        AlignmentFailureReason.LOW_COVERAGE,
                        None,
                    )
                )
                continue
        adjusted.append(item)
    return tuple(adjusted)


def _classes_to_payload(classes: Sequence[SegmentClass]) -> list[dict[str, str | int | None]]:
    return [item.to_json() for item in classes]


def _classes_from_payload(rows: Sequence[Mapping[str, Any]]) -> tuple[SegmentClass, ...]:
    classes: list[SegmentClass] = []
    for item in rows:
        outcome = SegmentOutcome(str(item["outcome"]))
        failure = item.get("failure_reason")
        by_design = item.get("by_design_reason")
        if outcome is SegmentOutcome.ALIGNMENT_FAILURE and failure is None:
            failure_reason = AlignmentFailureReason.ALIGNER_ERROR
        elif failure is None:
            failure_reason = None
        else:
            failure_reason = AlignmentFailureReason(str(failure))
        classes.append(
            SegmentClass(
                str(item["recording_id"]),
                str(item["speaker"]),
                int(item["start_ns"]),
                int(item["end_ns"]),
                outcome,
                failure_reason,
                None if by_design is None else ByDesignReason(str(by_design)),
            )
        )
    return tuple(classes)


def _reconstruct_classes(
    original: RecordingLabels,
    produced: RecordingLabels,
    *,
    held_out: bool,
    overlap_keys: set[tuple[str, int, int]] | None = None,
) -> tuple[SegmentClass, ...]:
    if held_out:
        return _held_out_classes(original)
    overlap_keys = overlap_keys or set()
    classes: list[SegmentClass] = []
    for segment in unioned_original_segments([original]):
        key = (segment.speaker, segment.start_ns, segment.end_ns)
        if key in overlap_keys or any(
            speaker == segment.speaker and start_ns < segment.end_ns and end_ns > segment.start_ns
            for speaker, start_ns, end_ns in overlap_keys
        ):
            classes.append(
                SegmentClass.from_segment(
                    segment,
                    SegmentOutcome.BY_DESIGN_UNCERTAIN,
                    by_design_reason=ByDesignReason.MULTI_SPEAKER,
                )
            )
            continue
        covered = overlap_duration_ns([(segment.start_ns, segment.end_ns)], produced.segments_for(segment.speaker))
        if covered > 0:
            classes.append(SegmentClass.from_segment(segment, SegmentOutcome.ALIGNED))
        else:
            classes.append(
                SegmentClass.from_segment(
                    segment,
                    SegmentOutcome.ALIGNMENT_FAILURE,
                    failure_reason=AlignmentFailureReason.ALIGNER_ERROR,
                )
            )
    return tuple(classes)


def load_audio_mono(path: Path, start_ns: int | None = None, end_ns: int | None = None) -> tuple[np.ndarray, int]:
    """Read a mix or close-talk file as float32 mono."""

    info = sf.info(str(path))
    if start_ns is None:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    else:
        start = (start_ns * info.samplerate) // NANOSECONDS_PER_SECOND
        stop = None if end_ns is None else (end_ns * info.samplerate) // NANOSECONDS_PER_SECOND
        samples, sample_rate = sf.read(str(path), start=int(start), stop=stop, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        samples = resample_audio(samples, sample_rate, SAMPLE_RATE)
        sample_rate = SAMPLE_RATE
    return samples, sample_rate


def _cache_path(source: str, recording_id: str) -> Path:
    return WORK_ROOT / source / f"{recording_id.replace('::', '__')}.json"


def _word_payload(words: Sequence[AlignedWord]) -> list[dict[str, object]]:
    return [
        {"start_ns": item.start_ns, "end_ns": item.end_ns, "word": item.word, "score": item.score} for item in words
    ]


def _words_from_payload(rows: Sequence[Mapping[str, Any]]) -> tuple[AlignedWord, ...]:
    return tuple(
        AlignedWord(int(item["start_ns"]), int(item["end_ns"]), str(item["word"]), float(item["score"]))
        for item in rows
    )


def align_indic_recording(
    ref: RecordingRef,
    original: RecordingLabels,
    transcripts: Sequence[TranscriptSegment],
    aligner: MmsAligner,
    *,
    poor_languages: frozenset[str],
) -> dict[str, Any]:
    """Align each original Indic segment on the released mix."""

    language = recording_language(ref.recording_id, transcripts)
    if language in poor_languages:
        tight = clip_recording_to_uem(held_out_labels(original), ref.uem_start_ns, ref.uem_end_ns)
        matched, report = match_segments(original.segments, transcripts)
        return {
            "recording_id": ref.recording_id,
            "language": language,
            "held_out": True,
            "held_out_speech_ns": duration_ns(original.segments),
            "match": {
                "matched": report.matched,
                "original": report.original,
                "unmatched_original": report.unmatched_original,
                "unmatched_transcript": report.unmatched_transcript,
                "mismatches": list(report.mismatches),
            },
            "kinds": {"held-out": len(original.segments)},
            "coverage": [],
            "empty_roman": 0,
            "letter_segments": 0,
            "labels": tight,
            "segment_classes": _held_out_classes(original),
            "close_talk": {"used": False, "reason": "indic-mix-only"},
        }
    matched, report = match_segments(original.segments, transcripts)
    mix, _ = load_audio_mono(ref.audio_path)
    aligned_by_speaker: dict[str, list[AlignedWord]] = defaultdict(list)
    failed_segments: list[SpeakerSegment] = []
    kinds: dict[str, int] = defaultdict(int)
    lcode = INDIC_UROMAN_LCODES.get(language)
    coverages: list[float] = []
    empty_roman = 0
    letter_segments = 0
    row_outcomes: dict[tuple[str, int, int], SegmentClass] = {}
    for segment in original.segments:
        key = (segment.speaker, segment.start_ns, segment.end_ns)
        hit = matched.get(key)
        if hit is None:
            kinds["unmatched"] += 1
            failed_segments.append(segment)
            row_outcomes[key] = SegmentClass.from_segment(
                segment,
                SegmentOutcome.ALIGNMENT_FAILURE,
                failure_reason=AlignmentFailureReason.ALIGNER_ERROR,
            )
            continue
        prepared = prepare_transcript(hit.text)
        if not prepared.alignable:
            kinds["marker"] += 1
            failed_segments.append(segment)
            reason = (
                AlignmentFailureReason.EMPTY_TEXT
                if not hit.text.strip()
                else AlignmentFailureReason.MARKER_ONLY
            )
            row_outcomes[key] = SegmentClass.from_segment(
                segment,
                SegmentOutcome.ALIGNMENT_FAILURE,
                failure_reason=reason,
            )
            continue
        letter_segments += 1
        romanized = aligner.romanize(prepared.remainder, lcode=lcode)
        if romanized.empty:
            empty_roman += 1
            kinds["empty-roman"] += 1
            failed_segments.append(segment)
            row_outcomes[key] = SegmentClass.from_segment(
                segment,
                SegmentOutcome.ALIGNMENT_FAILURE,
                failure_reason=AlignmentFailureReason.EMPTY_ROMANIZATION,
            )
            continue
        start = (segment.start_ns * SAMPLE_RATE) // NANOSECONDS_PER_SECOND
        stop = (segment.end_ns * SAMPLE_RATE) // NANOSECONDS_PER_SECOND
        crop = mix[start:stop]
        words = _clip_words(
            aligner.align_samples(crop, SAMPLE_RATE, romanized.words, origin_ns=segment.start_ns),
            segment.start_ns,
            segment.end_ns,
        )
        if not words:
            kinds["align-failed"] += 1
            failed_segments.append(segment)
            row_outcomes[key] = SegmentClass.from_segment(
                segment,
                SegmentOutcome.ALIGNMENT_FAILURE,
                failure_reason=AlignmentFailureReason.ALIGNER_ERROR,
            )
            continue
        aligned_by_speaker[segment.speaker].extend(words)
        covered = sum(item.end_ns - item.start_ns for item in words)
        coverages.append(covered / segment.duration_ns if segment.duration_ns else 0.0)
        kinds["aligned"] += 1
        row_outcomes[key] = SegmentClass.from_segment(segment, SegmentOutcome.ALIGNED)
    tight = _tighten(original, aligned_by_speaker, ref, failed_segments=failed_segments, per_segment=True)
    classes = _mark_low_coverage(_classes_from_row_outcomes(original, row_outcomes), tight)
    return {
        "recording_id": ref.recording_id,
        "language": language,
        "held_out": False,
        "held_out_speech_ns": 0,
        "match": {
            "matched": report.matched,
            "original": report.original,
            "unmatched_original": report.unmatched_original,
            "unmatched_transcript": report.unmatched_transcript,
            "mismatches": list(report.mismatches),
        },
        "kinds": dict(kinds),
        "coverage": coverages,
        "empty_roman": empty_roman,
        "letter_segments": letter_segments,
        "labels": tight,
        "segment_classes": classes,
        "close_talk": {"used": False, "reason": "indic-mix-only"},
    }


def _lotus_textgrid(recording_id: str) -> Path:
    parent = recording_id.split("::", 1)[1]
    return LOTUS_TEXTGRID_ROOT / f"{parent}_Con123.TextGrid"


def _lotus_lav_paths(recording_id: str) -> dict[str, Path]:
    parent = recording_id.split("::", 1)[1]
    directory = CLOSE_TALK_ROOT / parent
    return {f"Lav{index}": directory / f"{parent}_Lav{index}.wav" for index in (1, 2, 3)}


def map_lotusdis_close_talk(
    ref: RecordingRef,
    original: RecordingLabels,
    intervals: Sequence[LotusAlignInterval],
    mix: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, ClockOffset], LavalierMapping]:
    """Infer lavalier-to-speaker maps from exclusive single-speaker energy."""

    exclusive: dict[str, list[tuple[int, int]]] = defaultdict(list)
    exclusive_seconds: dict[str, float] = defaultdict(float)
    for interval in intervals:
        if interval.non_speech or interval.overlap or len(interval.speakers) != 1:
            continue
        speaker = interval.speakers[0]
        exclusive[speaker].append((interval.start_ns, interval.end_ns))
        exclusive_seconds[speaker] += ns_to_seconds(interval.end_ns - interval.start_ns)
    channels: dict[str, np.ndarray] = {}
    offsets: dict[str, ClockOffset] = {}
    energy: dict[str, dict[str, float]] = {}
    for name, path in _lotus_lav_paths(ref.recording_id).items():
        if not path.is_file():
            continue
        samples, _ = load_audio_mono(path)
        offset = measure_clock_offset(mix, samples)
        offsets[name] = offset
        aligned = shift_close_talk(samples, offset.offset_samples if offset.consistent else 0, mix.size)
        channels[name] = aligned
        energy[name] = {
            speaker: rms_in_spans(aligned, SAMPLE_RATE, exclusive.get(speaker, ()))
            for speaker in exclusive
        }
    mapping = map_lavaliers(energy, exclusive_seconds)
    if mapping.use_close_talk:
        for assignment in mapping.assignments:
            offset = offsets.get(assignment.channel)
            if offset is None or not offset.consistent:
                mapping = LavalierMapping(
                    mapping.assignments,
                    mapping.unmapped_speakers,
                    False,
                    mapping.confidence,
                    f"offset-inconsistent:{assignment.channel}",
                )
                break
    return channels, offsets, mapping


def align_lotusdis_recording(
    ref: RecordingRef,
    original: RecordingLabels,
    aligner: MmsAligner,
) -> dict[str, Any]:
    """Align single-speaker LOTUSDIS intervals; mark overlap as UNCERTAIN."""

    intervals = parse_lotus_align_intervals(_lotus_textgrid(ref.recording_id))
    mix, _ = load_audio_mono(ref.audio_path)
    channels, offsets, mapping = map_lotusdis_close_talk(ref, original, intervals, mix)
    by_bounds: dict[tuple[int, int], list[SpeakerSegment]] = defaultdict(list)
    for segment in original.segments:
        by_bounds[(segment.start_ns, segment.end_ns)].append(segment)
    aligned_by_speaker: dict[str, list[AlignedWord]] = defaultdict(list)
    failed_segments: list[SpeakerSegment] = []
    kinds: dict[str, int] = defaultdict(int)
    coverages: list[float] = []
    matched = 0
    unmatched = 0
    row_outcomes: dict[tuple[str, int, int], SegmentClass] = {}
    aligned_keys: set[tuple[str, int, int]] = set()
    for interval in intervals:
        if interval.non_speech:
            continue
        rows = by_bounds.get((interval.start_ns, interval.end_ns), [])
        if not rows:
            unmatched += 1
            continue
        matched += 1
        if interval.overlap:
            kinds["overlap-uncertain"] += 1
            failed_segments.extend(rows)
            for row in rows:
                row_outcomes[(row.speaker, row.start_ns, row.end_ns)] = SegmentClass.from_segment(
                    row,
                    SegmentOutcome.BY_DESIGN_UNCERTAIN,
                    by_design_reason=ByDesignReason.MULTI_SPEAKER,
                )
            continue
        prepared = prepare_transcript(interval.text, keep_square_bracket_words=True)
        if not prepared.alignable:
            kinds["marker"] += 1
            failed_segments.extend(rows)
            reason = (
                AlignmentFailureReason.EMPTY_TEXT
                if not interval.text.strip()
                else AlignmentFailureReason.MARKER_ONLY
            )
            for row in rows:
                row_outcomes[(row.speaker, row.start_ns, row.end_ns)] = SegmentClass.from_segment(
                    row,
                    SegmentOutcome.ALIGNMENT_FAILURE,
                    failure_reason=reason,
                )
            continue
        romanized = aligner.romanize(prepared.remainder, lcode=THAI_UROMAN_LCODE)
        if romanized.empty:
            kinds["empty-roman"] += 1
            failed_segments.extend(rows)
            for row in rows:
                row_outcomes[(row.speaker, row.start_ns, row.end_ns)] = SegmentClass.from_segment(
                    row,
                    SegmentOutcome.ALIGNMENT_FAILURE,
                    failure_reason=AlignmentFailureReason.EMPTY_ROMANIZATION,
                )
            continue
        speaker = interval.speakers[0]
        channel_name = mapping.channel_for(speaker) if mapping.use_close_talk else None
        source = channels[channel_name] if channel_name and channel_name in channels else mix
        start = (interval.start_ns * SAMPLE_RATE) // NANOSECONDS_PER_SECOND
        stop = (interval.end_ns * SAMPLE_RATE) // NANOSECONDS_PER_SECOND
        crop = source[start:stop]
        words = _clip_words(
            aligner.align_samples(crop, SAMPLE_RATE, romanized.words, origin_ns=interval.start_ns),
            interval.start_ns,
            interval.end_ns,
        )
        if not words:
            kinds["align-failed"] += 1
            failed_segments.extend(rows)
            for row in rows:
                row_outcomes[(row.speaker, row.start_ns, row.end_ns)] = SegmentClass.from_segment(
                    row,
                    SegmentOutcome.ALIGNMENT_FAILURE,
                    failure_reason=AlignmentFailureReason.ALIGNER_ERROR,
                )
            continue
        aligned_by_speaker[speaker].extend(words)
        aligned_keys.add((speaker, interval.start_ns, interval.end_ns))
        duration = interval.end_ns - interval.start_ns
        covered = sum(item.end_ns - item.start_ns for item in words)
        coverages.append(covered / duration if duration else 0.0)
        kinds["aligned"] += 1
        for row in rows:
            row_outcomes[(row.speaker, row.start_ns, row.end_ns)] = SegmentClass.from_segment(
                row,
                SegmentOutcome.ALIGNED,
            )
    extra_uncertain = list(overlap_uncertain_regions(ref.recording_id, intervals))
    tight = _tighten(
        original,
        aligned_by_speaker,
        ref,
        failed_segments=failed_segments,
        extra_uncertain=extra_uncertain,
        per_segment=True,
    )
    extra_uncertain = []
    if mapping.use_close_talk:
        for interval in intervals:
            if interval.non_speech or interval.overlap or len(interval.speakers) != 1:
                continue
            speaker = interval.speakers[0]
            if (speaker, interval.start_ns, interval.end_ns) not in aligned_keys:
                continue
            channel_name = mapping.channel_for(speaker)
            if channel_name is None or channel_name not in channels:
                continue
            channel = channels[channel_name]
            speech_rms = rms_in_spans(
                channel,
                SAMPLE_RATE,
                [(item.start_ns, item.end_ns) for item in tight.segments_for(speaker)],
            )
            voiced = voiced_spans(channel, SAMPLE_RATE, speech_rms=speech_rms)
            extra_uncertain.extend(
                voicing_uncertain(
                    ref.recording_id,
                    speaker,
                    Interval(interval.start_ns, interval.end_ns),
                    [(item.start_ns, item.end_ns) for item in tight.segments_for(speaker)],
                    voiced,
                )
            )
        if extra_uncertain:
            tight = apply_extra_uncertain(tight, extra_uncertain)
            tight = clip_recording_to_uem(tight, ref.uem_start_ns, ref.uem_end_ns)
    kinds["voicing-uncertain"] = len(extra_uncertain)
    classes = _mark_low_coverage(_classes_from_row_outcomes(original, row_outcomes), tight)
    return {
        "recording_id": ref.recording_id,
        "language": "Thai",
        "held_out": False,
        "held_out_speech_ns": 0,
        "match": {
            "matched": matched,
            "original": len(original.segments),
            "unmatched_original": unmatched,
            "unmatched_transcript": 0,
            "mismatches": [],
        },
        "kinds": dict(kinds),
        "coverage": coverages,
        "labels": tight,
        "segment_classes": classes,
        "close_talk": {
            "used": mapping.use_close_talk,
            "reason": mapping.reason,
            "confidence": mapping.confidence,
            "assignments": [
                {
                    "speaker": item.speaker,
                    "channel": item.channel,
                    "ratio": item.ratio,
                    "exclusive_seconds": item.exclusive_seconds,
                    "rms_by_speaker": item.rms_by_speaker,
                }
                for item in mapping.assignments
            ],
            "offsets": {
                name: {
                    "offset_seconds": item.offset_seconds,
                    "consistent": item.consistent,
                    "reason": item.reason,
                    "windows": list(item.window_offsets_seconds),
                }
                for name, item in offsets.items()
            },
        },
    }


def _clip_words(words: Sequence[AlignedWord], start_ns: int, end_ns: int) -> tuple[AlignedWord, ...]:
    clipped: list[AlignedWord] = []
    for item in words:
        left = max(item.start_ns, start_ns)
        right = min(item.end_ns, end_ns)
        if right > left:
            clipped.append(AlignedWord(left, right, item.word, item.score))
    return tuple(clipped)


def _tighten(
    original: RecordingLabels,
    aligned_by_speaker: Mapping[str, Sequence[AlignedWord]],
    ref: RecordingRef,
    *,
    failed_segments: Sequence[SpeakerSegment] = (),
    extra_uncertain: Sequence[UncertainInterval] = (),
    per_segment: bool = True,
) -> RecordingLabels:
    aligned = {
        speaker: [Interval(item.start_ns, item.end_ns) for item in words if item.end_ns > item.start_ns]
        for speaker, words in aligned_by_speaker.items()
    }
    failed_keys = {(item.speaker, item.start_ns, item.end_ns) for item in failed_segments}
    active: list[SpeakerSegment] = []
    uncertain: list[UncertainInterval] = []
    for segment in original.segments:
        result = tighten_segment(
            segment,
            aligned.get(segment.speaker, ()),
            alignment_failed=(segment.speaker, segment.start_ns, segment.end_ns) in failed_keys,
        )
        active.extend(result.active)
        uncertain.extend(result.uncertain)
    tight = RecordingLabels(
        original.recording_id,
        tuple(active),
        tuple(uncertain) + tuple(extra_uncertain),
    )
    clipped = clip_recording_to_uem(tight, ref.uem_start_ns, ref.uem_end_ns)
    return merge_recording(clipped, min_gap_ns=MIN_GAP_NS)


def write_aligned_release(
    *,
    source: str,
    refs: Sequence[RecordingRef],
    original: Sequence[RecordingLabels],
    produced: Sequence[RecordingLabels],
    extra_provenance: Mapping[str, Any],
    uem_by_recording: Mapping[str, tuple[int, int]],
    segment_classes: Sequence[SegmentClass] | None,
    release_id: str | None = None,
) -> tuple[Path, GateResult]:
    """Write tight labels, then score gates on the files actually written."""

    spec = SOURCE_SPECS[source]
    written_release_id = release_id or spec["release_id"]
    release_dir = RELEASE_ROOT / written_release_id
    if release_dir.is_dir() and (release_dir / "provenance.json").is_file():
        _log(f"{source} release exists, skip write {release_dir}")
        labels_dir = release_dir / "labels"
        written = tuple(
            load_written_recording_labels(labels_dir, labels.recording_id) for labels in produced
        )
        gate = evaluate_gates(
            original,
            written,
            require_subset=spec["require_subset"],
            uem_by_recording=uem_by_recording,
            segment_classes=segment_classes,
        )
        return release_dir, gate
    if release_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing release {release_dir}")
    staging_dir = release_dir.with_name(release_dir.name + ".partial")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    labels_dir = staging_dir / "labels"
    labels_dir.mkdir(parents=True)
    uem_by_id = {item.recording_id: item.uem_path for item in refs}
    all_rttm: list[str] = []
    all_uem: list[str] = []
    for labels in produced:
        write_recording_labels(labels_dir, labels)
        original_uem = uem_by_id[labels.recording_id].read_text(encoding="utf-8")
        if uem_by_id[labels.recording_id].name == "all.uem":
            original_uem = next(
                raw + "\n"
                for raw in original_uem.splitlines()
                if raw.startswith(labels.recording_id + " ")
            )
        uem_text = rewrite_uem_file_id(original_uem, labels.recording_id)
        (labels_dir / f"{labels.recording_id}.uem").write_text(uem_text, encoding="utf-8")
        all_rttm.append(render_speaker_rttm(labels.segments))
        all_uem.append(uem_text if uem_text.endswith("\n") else uem_text + "\n")
    (staging_dir / "all.rttm").write_text("".join(all_rttm), encoding="utf-8")
    (staging_dir / "all.uem").write_text("".join(all_uem), encoding="utf-8")
    written = tuple(
        load_written_recording_labels(labels_dir, labels.recording_id) for labels in produced
    )
    gate = evaluate_gates(
        original,
        written,
        require_subset=spec["require_subset"],
        uem_by_recording=uem_by_recording,
        segment_classes=segment_classes,
    )
    checksum_paths = [
        staging_dir / "all.rttm",
        staging_dir / "all.uem",
        *sorted(labels_dir.glob("*.rttm")),
        *sorted(labels_dir.glob("*.uem")),
    ]
    provenance = {
        "schema": "speakrs-tight-labels-release",
        "schema_version": 3,
        "release_id": written_release_id,
        "source": spec["source_name"],
        "method": spec["method"],
        "kept_current_labels": False,
        "admission_status": gate.admission_status.value,
        "gate_policy_version": GATE_POLICY_VERSION,
        "library_version": LIBRARY_VERSION,
        "library_commit": library_commit(),
        "min_gap_seconds": MIN_GAP_SECONDS,
        "min_gap_ns": MIN_GAP_NS,
        "recording_count": len(written),
        "licenses": [spec["license"]],
        "label_checksums": sha256_paths(checksum_paths, root=staging_dir),
        "gate_results": gate.to_json(),
        **dict(extra_provenance),
    }
    write_json(staging_dir / "provenance.json", provenance)
    write_json(staging_dir / "gate-report.json", gate.to_json())
    (staging_dir / "gate-report.md").write_text(
        render_gate_markdown(spec["source_name"], written_release_id, gate),
        encoding="utf-8",
    )
    staging_dir.rename(release_dir)
    return release_dir, gate


def _summarise_kinds(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.get("kinds", {}).items():
            totals[key] += int(value)
    return dict(totals)


def _coverage_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    values = [item for row in rows for item in row.get("coverage", [])]
    if not values:
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def produce_source(
    source: str,
    aligner: MmsAligner,
    *,
    limit: int | None = None,
    force: bool = False,
    release_version: str | None = None,
) -> dict[str, Any]:
    """Align one source, score gates, and write the release plus listening pack."""

    spec = SOURCE_SPECS[source]
    written_release_id = release_id_with_version(spec["release_id"], release_version)
    refs = load_indic_refs() if source == "indic-diarbench" else load_lotusdis_refs()
    if limit is not None:
        refs = refs[:limit]
    current = load_original_labels(spec["prefix"], [item.recording_id for item in refs])
    original = [RecordingLabels(item.recording_id, tuple(current[item.recording_id])) for item in refs]
    uem_by_recording = {item.recording_id: (item.uem_start_ns, item.uem_end_ns) for item in refs}
    extra: dict[str, Any] = {
        "aligner": aligner.provenance(),
        "voicing_thresholds": voicing_thresholds(),
        "uroman_lcodes": INDIC_UROMAN_LCODES if source == "indic-diarbench" else {"Thai": THAI_UROMAN_LCODE},
    }
    results: list[dict[str, Any]] = []
    poor_languages: frozenset[str] = frozenset()
    match_summary: dict[str, Any] = {}
    coverage_by_language: dict[str, Any] = {}
    transcripts: dict[str, list[TranscriptSegment]] = {}
    if source == "indic-diarbench":
        transcripts = load_indic_transcripts()
        by_language: dict[str, list[str]] = defaultdict(list)
        language_match: dict[str, dict[str, int]] = defaultdict(lambda: {"matched": 0, "original": 0, "unmatched": 0})
        for labels in original:
            rows = transcripts.get(labels.recording_id, ())
            matched, report = match_segments(labels.segments, rows)
            language = recording_language(labels.recording_id, rows)
            language_match[language]["matched"] += report.matched
            language_match[language]["original"] += report.original
            language_match[language]["unmatched"] += report.unmatched_original
        match_path = WORK_ROOT / "indic-match.json"
        if match_path.is_file() and not force:
            cached_match = read_json(match_path)
            coverage_by_language = cached_match["coverage"]
            poor_languages = frozenset(
                name for name, row in coverage_by_language.items() if row.get("poor")
            )
            _log(
                f"indic match languages={len(language_match)} poor={list(poor_languages) or 'none'} coverage-cache"
            )
        else:
            for rows in transcripts.values():
                for item in rows:
                    prepared = prepare_transcript(item.text)
                    if prepared.alignable:
                        by_language[item.language].append(prepared.remainder)
            coverage_rows = []
            poor = []
            for language, texts in sorted(by_language.items()):
                measured = measure_language_coverage(
                    language, texts, aligner, lcode=INDIC_UROMAN_LCODES.get(language)
                )
                coverage_rows.append(measured)
                if measured.poor:
                    poor.append(language)
            poor_languages = frozenset(poor)
            coverage_by_language = {
                item.language: {
                    "lcode": item.lcode,
                    "empty_roman_rate": item.empty_roman_rate,
                    "leftover_script_rate": item.leftover_script_rate,
                    "expansion_ratio": item.expansion_ratio,
                    "nap_junk_rate": item.nap_junk_rate,
                    "letter_segment_count": item.letter_segment_count,
                    "poor": item.poor,
                    "samples": [{"original": a, "romanized": b} for a, b in item.samples],
                }
                for item in coverage_rows
            }
            _log(
                f"indic match languages={len(language_match)} poor={list(poor_languages) or 'none'}"
            )
        match_summary = {"by_language": dict(language_match), "poor_languages": list(poor_languages)}
        extra["romanizer_coverage"] = coverage_by_language
        extra["transcript_match"] = match_summary
        extra["transcript_cleaning"] = cleaning_stats_by_language(transcripts)
        write_json(match_path, {"match": match_summary, "coverage": coverage_by_language})
    produced_labels: list[RecordingLabels] = []
    close_talk_rows: list[dict[str, Any]] = []
    original_by_id = {item.recording_id: item for item in original}
    for index, ref in enumerate(refs, start=1):
        source_labels = original_by_id[ref.recording_id]
        language = recording_language(ref.recording_id, transcripts.get(ref.recording_id, ()))
        is_held_out = language in poor_languages
        cache = _cache_path(source, ref.recording_id)
        payload: dict[str, Any] | None = None
        if cache.is_file() and not force:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if bool(cached.get("held_out")) == is_held_out:
                payload = cached
        if payload is None:
            if source == "indic-diarbench":
                payload = align_indic_recording(
                    ref,
                    source_labels,
                    transcripts.get(ref.recording_id, ()),
                    aligner,
                    poor_languages=poor_languages,
                )
            else:
                payload = align_lotusdis_recording(ref, source_labels, aligner)
            labels = payload["labels"]
            classes = tuple(payload.get("segment_classes") or ())
            serialisable = {
                "recording_id": ref.recording_id,
                "language": payload.get("language"),
                "held_out": bool(payload.get("held_out")),
                "held_out_speech_ns": int(payload.get("held_out_speech_ns") or 0),
                "match": payload.get("match"),
                "kinds": payload.get("kinds"),
                "coverage": payload.get("coverage"),
                "close_talk": payload.get("close_talk"),
                "segment_classes": _classes_to_payload(classes),
                "segments": [
                    {"speaker": item.speaker, "start_ns": item.start_ns, "end_ns": item.end_ns}
                    for item in labels.segments
                ],
                "uncertain": [
                    {"speaker": item.speaker, "start_ns": item.start_ns, "end_ns": item.end_ns}
                    for item in labels.uncertain
                ],
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            write_json(cache, serialisable)
            from_cache = False
        else:
            labels = RecordingLabels(
                ref.recording_id,
                tuple(
                    SpeakerSegment(ref.recording_id, item["speaker"], int(item["start_ns"]), int(item["end_ns"]))
                    for item in payload["segments"]
                ),
                tuple(
                    UncertainInterval(ref.recording_id, item["speaker"], int(item["start_ns"]), int(item["end_ns"]))
                    for item in payload.get("uncertain", ())
                ),
            )
            from_cache = True
        labels = merge_recording(labels)
        payload["labels"] = labels
        if from_cache:
            if payload.get("segment_classes"):
                try:
                    payload["segment_classes"] = _classes_from_payload(payload["segment_classes"])
                except (KeyError, TypeError, ValueError):
                    payload["segment_classes"] = ()
            if not payload.get("segment_classes"):
                overlap_keys: set[tuple[str, int, int]] = set()
                if source == "lotusdis":
                    for interval in parse_lotus_align_intervals(_lotus_textgrid(ref.recording_id)):
                        if interval.overlap:
                            for speaker in interval.speakers:
                                overlap_keys.add((speaker, interval.start_ns, interval.end_ns))
                payload["segment_classes"] = _reconstruct_classes(
                    source_labels,
                    labels,
                    held_out=is_held_out,
                    overlap_keys=overlap_keys,
                )
        payload["held_out"] = is_held_out
        payload.setdefault("held_out_speech_ns", duration_ns(source_labels.segments) if is_held_out else 0)
        results.append(payload)
        produced_labels.append(labels)
        if source == "lotusdis" and payload.get("close_talk"):
            close_talk_rows.append({"recording_id": ref.recording_id, **payload["close_talk"]})
        kinds = payload.get("kinds", {})
        if from_cache:
            _log(f"{source} {index}/{len(refs)} cache {ref.recording_id}")
        elif is_held_out:
            _log(
                f"{source} {index}/{len(refs)} {ref.recording_id} held-out language={language} "
                f"speech_s={ns_to_seconds(int(payload['held_out_speech_ns'])):.3f}"
            )
        else:
            _log(
                f"{source} {index}/{len(refs)} {ref.recording_id} aligned={kinds.get('aligned', 0)} "
                f"failed={kinds.get('align-failed', 0)} uncertain_rows={len(labels.uncertain)}"
            )
    attempted = list(produced_labels)
    held_ids = {
        item["recording_id"]
        for item in results
        if item.get("held_out")
    }
    segment_classes = tuple(
        item
        for row in results
        for item in row.get("segment_classes") or ()
    )
    held_out_speech_ns = sum(int(item.get("held_out_speech_ns") or 0) for item in results if item.get("held_out"))
    extra["segment_kinds"] = _summarise_kinds(results)
    extra["coverage_distribution"] = _coverage_stats(results)
    extra["close_talk"] = close_talk_rows
    extra["held_out_languages"] = sorted(poor_languages)
    extra["held_out_recordings"] = len(held_ids)
    extra["held_out_speech_seconds"] = ns_to_seconds(held_out_speech_ns)
    extra["held_out_excluded_from_gates"] = False
    extra["gates_scored_on_recordings"] = len(original)
    release_dir, gate = write_aligned_release(
        source=source,
        refs=refs,
        original=original,
        produced=attempted,
        extra_provenance=extra,
        uem_by_recording=uem_by_recording,
        segment_classes=segment_classes,
        release_id=written_release_id,
    )
    extra["uncertain_share_of_original_speech"] = gate.uncertain_share
    extra["uncertain_share_excluding_by_design"] = gate.uncertain_share_excluding_by_design
    extra["by_design_share"] = gate.by_design_share
    extra["failure_share"] = gate.failure_share
    extra["admission_status"] = gate.admission_status.value
    listening_dir = STAGING_ROOT / "listening-v2" / spec["listening_name"]
    listening = write_listening_pack(
        listening_dir,
        original=original,
        new=attempted,
        audio_by_recording={item.recording_id: item.audio_path for item in refs},
        duration_by_recording={item.recording_id: item.duration_ns for item in refs},
    )
    return {
        "source": source,
        "release_id": written_release_id,
        "release_dir": str(release_dir),
        "recording_count": len(attempted),
        "kept_current_labels": False,
        "admission_status": gate.admission_status.value,
        "gate": gate,
        "listening": listening,
        "original": original,
        "produced": attempted,
        "extra": extra,
        "match": match_summary,
        "coverage_by_language": coverage_by_language,
    }


def render_align_report(results: Sequence[Mapping[str, Any]], aligner: MmsAligner) -> str:
    """Render ALIGN-MMS-REPORT.md."""

    lines = [
        "# Tight labels MMS alignment report",
        "",
        f"Library version: {LIBRARY_VERSION}",
        f"Library commit: {library_commit()}",
        f"Aligner: {aligner.bundle_id}",
        f"Checkpoint: `{aligner.model_path}`",
        f"Checkpoint SHA-256: `{aligner.model_sha256}`",
        f"torch {aligner.torch_version}, torchaudio {aligner.torchaudio_version}, uroman {aligner.uroman_version}",
        f"Device: {aligner.device}",
        "",
        "## Sources",
        "",
        "| Source | Recs | Gaps ≤0.3 s/min orig | Gaps ≤0.3 s/min new | Gaps 0.3–1 s/min orig | Gaps 0.3–1 s/min new | Smallest gap new | Failure share | Uncertain share | Admission |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        gate: GateResult = result["gate"]
        lines.append(
            "| {name} | {recs} | {o_short:.2f} | {n_short:.2f} | {o_mid:.2f} | {n_mid:.2f} | {gap} | {fail:.4f} | {unc:.4f} | {status} |".format(
                name=SOURCE_SPECS[result["source"]]["source_name"],
                recs=result["recording_count"],
                o_short=gate.original_tightness.gaps_le_0_3_per_min,
                n_short=gate.tightness.gaps_le_0_3_per_min,
                o_mid=gate.original_tightness.gaps_0_3_to_1_per_min,
                n_mid=gate.tightness.gaps_0_3_to_1_per_min,
                gap="—" if gate.tightness.smallest_gap_s is None else f"{gate.tightness.smallest_gap_s:.4f}",
                fail=gate.failure_share,
                unc=gate.uncertain_share,
                status=gate.admission_status.value,
            )
        )
    for result in results:
        spec = SOURCE_SPECS[result["source"]]
        gate = result["gate"]
        extra = result["extra"]
        lines.extend(
            [
                "",
                f"## {spec['source_name']}",
                "",
                f"- Release: `{result['release_dir']}`",
                f"- Listening pack: `{result['listening']['directory']}`",
                f"- Listening clips: {result['listening']['clip_count']}",
                f"- Listening seconds: {result['listening']['total_seconds']:.1f}",
                f"- Coverage: {extra['coverage_distribution']}",
                f"- Segment kinds: {extra['segment_kinds']}",
                f"- Admission: {gate.admission_status.value}",
                f"- Failure share of all original segments: {gate.failure_share:.4f}",
                f"- Failure share of eligible segments: {gate.failure_share_excluding_by_design:.4f}",
                f"- By-design UNCERTAIN share: {gate.by_design_share:.4f}",
                f"- Uncertain share of original speech: {gate.uncertain_share:.4f}",
                f"- Uncertain share excluding by-design: {gate.uncertain_share_excluding_by_design:.4f}",
                f"- Overlap retention: {gate.overlap_retention}",
                f"- Subset: {gate.subset_ok}",
                f"- UEM: {gate.uem_ok}",
                f"- Speaker retention flagged: {len(gate.flagged_speakers)}",
                f"- Held-out languages: {extra.get('held_out_languages') or 'none'}",
                f"- Held-out recordings: {extra.get('held_out_recordings', 0)}",
                f"- Held-out original speech (s): {extra.get('held_out_speech_seconds', 0):.3f}",
                "- Held-out recordings keep UNCERTAIN interiors and count as by-design, not failures",
            ]
        )
        if result["source"] == "indic-diarbench":
            lines.extend(["", "### Per-language transcript match and romanizer", ""])
            lines.append("| Language | Original segs | Matched | Unmatched | Empty-roman | Leftover script | Expansion | Nap junk | Poor |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
            match = result["match"].get("by_language", {})
            coverage = result["coverage_by_language"]
            for language in sorted(set(match) | set(coverage)):
                row = match.get(language, {})
                cov = coverage.get(language, {})
                lines.append(
                    f"| {language} | {row.get('original', 0)} | {row.get('matched', 0)} | "
                    f"{row.get('unmatched', 0)} | {cov.get('empty_roman_rate', 0):.4f} | "
                    f"{cov.get('leftover_script_rate', 0):.4f} | {cov.get('expansion_ratio', 0):.2f} | "
                    f"{cov.get('nap_junk_rate', 0):.4f} | {cov.get('poor', False)} |"
                )
            cleaning = extra.get("transcript_cleaning") or {}
            if cleaning:
                lines.extend(["", "### Indic transcript cleaning", ""])
                lines.append("| Language | Rows | Changed rows | Changed tokens | Gloss rows | Top tags |")
                lines.append("|---|---:|---:|---:|---:|---|")
                for language, stats in cleaning.items():
                    tags = stats.get("tags") or {}
                    top = ", ".join(f"{name} {count}" for name, count in list(tags.items())[:6]) or "—"
                    lines.append(
                        f"| {language} | {stats.get('rows', 0)} | {stats.get('changed_rows', 0)} | "
                        f"{stats.get('changed_tokens', 0)} | {stats.get('gloss_rows', 0)} | {top} |"
                    )
        if extra.get("close_talk"):
            lines.extend(["", "### LOTUSDIS lavalier mapping", ""])
            used = sum(1 for item in extra["close_talk"] if item.get("used"))
            lines.append(f"Recordings using lavalier audio: {used} of {len(extra['close_talk'])}")
            lines.append("")
            for item in extra["close_talk"]:
                lines.append(
                    f"- `{item['recording_id']}` used={item.get('used')} reason={item.get('reason')} "
                    f"confidence={item.get('confidence')}"
                )
                for assignment in item.get("assignments", ()):
                    lines.append(
                        f"  - {assignment['channel']} -> {assignment['speaker']} "
                        f"ratio={assignment['ratio']:.3f} exclusive_s={assignment['exclusive_seconds']:.1f}"
                    )
        if gate.notes:
            lines.extend(["", "### Gate notes", ""])
            lines.extend(f"- {note}" for note in gate.notes)
    lines.append("")
    return "\n".join(lines) + "\n"


def _configure_paths(
    *,
    output_root: Path | None = None,
    staging_root: Path | None = None,
    work_root: Path | None = None,
) -> None:
    global RELEASE_ROOT, STAGING_ROOT, WORK_ROOT
    if output_root is not None:
        RELEASE_ROOT = Path(output_root) / "releases"
        if staging_root is None:
            STAGING_ROOT = Path(output_root) / "staging"
        if work_root is None:
            WORK_ROOT = Path(output_root) / "work"
    if staging_root is not None:
        STAGING_ROOT = Path(staging_root)
        if work_root is None and output_root is None:
            WORK_ROOT = STAGING_ROOT / "align-mms-work"
    if work_root is not None:
        WORK_ROOT = Path(work_root)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)


def _source_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": item["source"],
        "release_id": item["release_id"],
        "release_dir": item["release_dir"],
        "passed": item["gate"].passed,
        "admission_status": item["admission_status"],
        "kept_current_labels": False,
        "listening": item["listening"],
        "held_out_languages": item["extra"].get("held_out_languages", []),
        "held_out_recordings": item["extra"].get("held_out_recordings", 0),
        "held_out_speech_seconds": item["extra"].get("held_out_speech_seconds", 0.0),
    }


def _write_source_outputs(results: Sequence[Mapping[str, Any]], aligner: MmsAligner) -> Path:
    report_path = STAGING_ROOT / "ALIGN-MMS-REPORT.md"
    report_path.write_text(render_align_report(results, aligner), encoding="utf-8")
    summary = {
        "ok": True,
        "library_version": LIBRARY_VERSION,
        "library_commit": library_commit(),
        "aligner": aligner.provenance(),
        "report": str(report_path),
        "progress_log": str(progress_log_path()),
        "sources": [_source_summary(item) for item in results],
    }
    write_json(WORK_ROOT / "produce-summary.json", summary)
    return report_path


def produce(
    sources: Sequence[str] | None = None,
    *,
    limit: int | None = None,
    force: bool = False,
    output_root: Path | None = None,
    staging_root: Path | None = None,
    work_root: Path | None = None,
    release_version: str | None = None,
) -> dict[str, Any]:
    """Produce the requested MMS source releases and the ALIGN report."""

    _configure_paths(output_root=output_root, staging_root=staging_root, work_root=work_root)
    require_free_gib(DV_ROOT, MINIMUM_FREE_GIB)
    selected = list(sources or SOURCE_SPECS)
    for source in selected:
        if source not in SOURCE_SPECS:
            raise ValueError(f"unknown source {source}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    _log(f"loading MMS aligner device-probe from {MMS_MODEL_PATH}")
    aligner = MmsAligner(MMS_MODEL_PATH)
    _log(f"aligner ready device={aligner.device} sha256={aligner.model_sha256[:12]}")
    results: list[dict[str, Any]] = []
    try:
        for source in selected:
            result = produce_source(
                source, aligner, limit=limit, force=force, release_version=release_version
            )
            results.append(result)
            report_path = _write_source_outputs(results, aligner)
            _log(
                f"{source} finished release={result['release_dir']} "
                f"listening={result['listening']['directory']} report={report_path}"
            )
    except Exception:
        if results:
            try:
                _write_source_outputs(results, aligner)
            except Exception:
                _log("failed to write partial ALIGN-MMS-REPORT.md")
        raise
    summary = read_json(WORK_ROOT / "produce-summary.json")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run MMS tight-label production from the command line."""

    parser = argparse.ArgumentParser(description="Produce tight-v1 MMS label releases")
    parser.add_argument("--source", action="append", choices=sorted(SOURCE_SPECS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Write releases, staging outputs, and work cache under this root",
    )
    parser.add_argument("--staging-root", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument(
        "--release-version",
        default=None,
        help="Replace the tight-vN suffix of the release id. Default: keep the current id",
    )
    args = parser.parse_args(argv)
    _configure_paths(
        output_root=args.output_root,
        staging_root=args.staging_root,
        work_root=args.work_root,
    )
    try:
        summary = produce(
            args.source,
            limit=args.limit,
            force=args.force,
            output_root=args.output_root,
            staging_root=args.staging_root,
            work_root=args.work_root,
            release_version=args.release_version,
        )
    except Exception as error:
        _write_terminal_status(f"FAILED {error}")
        traceback.print_exc()
        return 1
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    _write_terminal_status("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
