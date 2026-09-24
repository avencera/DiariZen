from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recipes.speakrs.large.tight_labels import (
    COVERAGE_RATIO,
    MIN_GAP_NS,
    AlignmentFailureReason,
    ByDesignReason,
    Interval,
    RecordingLabels,
    SegmentClass,
    SegmentOutcome,
    SpeakerSegment,
    UncertainInterval,
    apply_published_replacement,
    clip_recording_to_uem,
    evaluate_gates,
    is_subset_of_original,
    map_speakers_by_overlap,
    measure_tightness,
    merge_short_gaps,
    parse_seconds_to_ns,
    parse_speaker_rttm,
    subtract_uncertain_from_active,
    tighten_recording,
    tighten_segment,
    write_recording_labels,
)
from recipes.speakrs.large.tight_labels.listening import select_listening_windows
from recipes.speakrs.large.tight_labels.time import format_ns
from recipes.speakrs.large.tight_labels.writers import parse_uncertain_rttm, rewrite_uem_file_id, uem_bounds


def _seg(speaker: str, start: object, end: object, recording: str = "rec") -> SpeakerSegment:
    return SpeakerSegment.from_seconds(recording, speaker, start, end)  # type: ignore[arg-type]


def _iv(start: object, end: object) -> Interval:
    return Interval.from_seconds(start, end)  # type: ignore[arg-type]


def _span(start: object, end: object) -> tuple[int, int]:
    return (parse_seconds_to_ns(start), parse_seconds_to_ns(end))  # type: ignore[arg-type]


def _unc(speaker: str, start: object, end: object, recording: str = "rec") -> UncertainInterval:
    return UncertainInterval.from_seconds(recording, speaker, start, end)  # type: ignore[arg-type]


def _cls(
    segment: SpeakerSegment,
    outcome: SegmentOutcome,
    *,
    failure_reason: AlignmentFailureReason | None = None,
    by_design_reason: ByDesignReason | None = None,
) -> SegmentClass:
    return SegmentClass.from_segment(
        segment,
        outcome,
        failure_reason=failure_reason,
        by_design_reason=by_design_reason,
    )


def _spans(items: tuple[SpeakerSegment, ...] | list[SpeakerSegment]) -> list[tuple[int, int]]:
    return [(item.start_ns, item.end_ns) for item in items]


def test_merge_fills_gaps_shorter_than_200ms_and_keeps_exact_200ms() -> None:
    segments = (
        _seg("A", 0.0, 1.0),
        _seg("A", 1.15, 2.0),
        _seg("A", 2.2, 3.0),
        _seg("B", 0.0, 1.0),
        _seg("B", 1.15, 2.0),
    )

    merged = merge_short_gaps(segments, min_gap_ns=MIN_GAP_NS)

    a_spans = _spans([item for item in merged if item.speaker == "A"])
    b_spans = _spans([item for item in merged if item.speaker == "B"])
    assert a_spans == [_span(0, 2), _span("2.2", 3)]
    assert b_spans == [_span(0, 2)]


def test_merge_joins_overlapping_and_touching_segments() -> None:
    segments = (
        _seg("A", 0.0, 1.0),
        _seg("A", 0.5, 1.5),
        _seg("A", 1.5, 2.0),
    )

    merged = merge_short_gaps(segments)

    assert _spans(merged) == [_span(0, 2)]


def test_merge_keeps_exact_200ms_gap_from_published_decimal_text() -> None:
    text = (
        "SPEAKER EN2001a 1 3325.128 3.510 <NA> <NA> EN2001a.E <NA> <NA>\n"
        "SPEAKER EN2001a 1 3328.838 3.680 <NA> <NA> EN2001a.E <NA> <NA>\n"
    )
    start_ns = parse_seconds_to_ns("3325.128")
    duration_ns = parse_seconds_to_ns("3.510")
    next_start_ns = parse_seconds_to_ns("3328.838")
    assert next_start_ns - (start_ns + duration_ns) == MIN_GAP_NS

    merged = merge_short_gaps(parse_speaker_rttm(text))

    assert _spans(merged) == [
        (start_ns, start_ns + duration_ns),
        (next_start_ns, next_start_ns + parse_seconds_to_ns("3.680")),
    ]


def test_float_start_plus_duration_would_fill_the_exact_200ms_gap() -> None:
    start = float("3325.128")
    duration = float("3.510")
    next_start = float("3328.838")
    assert start + duration - next_start != 0.200
    assert next_start - (start + duration) < 0.200


def test_tighten_active_speech_is_subset_of_the_human_segment() -> None:
    human = _seg("A", 10.0, 20.0)
    aligned = (
        _iv(9.0, 11.0),
        _iv(12.0, 13.0),
        _iv(14.0, 17.0),
        _iv(19.5, 22.0),
    )

    result = tighten_segment(human, aligned)

    assert result.fallback is False
    assert result.uncertain == ()
    assert _spans(result.active) == [
        _span(10, 11),
        _span(12, 13),
        _span(14, 17),
        _span("19.5", 20),
    ]
    assert is_subset_of_original(result.active, (human,), speaker="A")


def test_tighten_marks_uncertain_when_coverage_is_below_30_percent() -> None:
    human = _seg("A", 0.0, 10.0)
    aligned = (_iv(0.0, 2.9),)

    result = tighten_segment(human, aligned, min_coverage=COVERAGE_RATIO)

    assert result.fallback is True
    assert result.active == ()
    assert result.uncertain == (UncertainInterval.from_seconds("rec", "A", 0.0, 10.0),)


def test_tighten_keeps_speech_when_coverage_is_exactly_30_percent() -> None:
    human = _seg("A", 0.0, 10.0)
    aligned = (_iv(1.0, 4.0),)

    result = tighten_segment(human, aligned, min_coverage=0.30)

    assert result.fallback is False
    assert _spans(result.active) == [_span(1, 4)]


def test_tighten_keeps_three_tenths_of_one_second() -> None:
    human = _seg("A", "0.0", "1.0")
    aligned = (
        _iv("0.0", "0.1"),
        _iv("0.1", "0.2"),
        _iv("0.2", "0.3"),
    )

    result = tighten_segment(human, aligned, min_coverage=COVERAGE_RATIO)

    assert result.fallback is False
    assert result.coverage == pytest.approx(0.3)
    assert _spans(result.active) == [_span("0.0", "0.3")]


def test_alignment_failure_drops_active_speech() -> None:
    human = _seg("A", 0.0, 2.0)
    aligned = (_iv(0.0, 2.0),)

    result = tighten_segment(human, aligned, alignment_failed=True)

    assert result.fallback is True
    assert result.active == ()
    assert len(result.uncertain) == 1


def test_tighten_merges_short_gaps_inside_a_human_segment() -> None:
    human = _seg("A", 0.0, 5.0)
    aligned = (_iv(0.0, 1.0), _iv(1.1, 2.0))

    result = tighten_segment(human, aligned)

    assert _spans(result.active) == [_span(0, 2)]


def test_tighten_merges_a_40ms_gap_between_human_segments() -> None:
    original = RecordingLabels(
        "rec",
        (_seg("A", 0.0, 1.0), _seg("A", "1.04", 2.0)),
    )
    aligned = {"A": (_iv(0.0, 1.0), _iv("1.04", 2.0))}

    tight = tighten_recording(original, aligned)

    assert _spans(tight.segments) == [_span(0, 2)]
    assert is_subset_of_original(tight.segments, original.segments, speaker="A")


def test_tighten_keeps_a_250ms_gap_between_human_segments() -> None:
    original = RecordingLabels(
        "rec",
        (_seg("A", 0.0, 1.0), _seg("A", "1.25", 2.0)),
    )
    aligned = {"A": (_iv(0.0, 1.0), _iv("1.25", 2.0))}

    tight = tighten_recording(original, aligned)

    assert _spans(tight.segments) == [_span(0, 1), _span("1.25", 2)]
    assert is_subset_of_original(tight.segments, original.segments, speaker="A")


def test_tighten_merge_stays_inside_the_200ms_closed_original() -> None:
    original = RecordingLabels(
        "rec",
        (_seg("A", 0.0, 1.0), _seg("A", "1.04", 2.0)),
    )
    aligned = {"A": (_iv(0.0, 1.0), _iv("1.04", 2.0))}

    tight = tighten_recording(original, aligned)

    assert _spans(tight.segments) == [_span(0, 2)]
    assert is_subset_of_original(tight.segments, original.segments, speaker="A")
    assert not is_subset_of_original((_seg("A", 0.0, "2.1"),), original.segments, speaker="A")


def test_tighten_uncertain_does_not_overlap_merged_speech() -> None:
    original = RecordingLabels(
        "rec",
        (_seg("A", 0.0, 1.0), _seg("A", "1.04", "1.12"), _seg("A", "1.16", 2.0)),
    )
    aligned = {"A": (_iv(0.0, 1.0), _iv("1.16", 2.0))}

    tight = tighten_recording(original, aligned)

    assert _spans(tight.segments) == [_span(0, 2)]
    assert tight.uncertain == ()


def test_tighten_unions_overlapping_human_segments_before_coverage() -> None:
    original = RecordingLabels(
        "rec",
        (_seg("A", 0, 5), _seg("A", 4, 10)),
    )
    aligned = {"A": (_iv(4, 10),)}

    tight = tighten_recording(original, aligned)

    assert _spans(tight.segments) == [_span(4, 10)]
    assert tight.uncertain == ()
    assert is_subset_of_original(tight.segments, original.segments, speaker="A")


def test_tighten_uncertain_does_not_overlap_same_speaker_active_speech() -> None:
    original = RecordingLabels(
        "rec",
        (_seg("A", 0, 5), _seg("A", 4, 10), _seg("A", 20, 21)),
    )
    aligned = {"A": (_iv(4, 10),)}

    tight = tighten_recording(original, aligned)

    assert _spans(tight.segments) == [_span(4, 10)]
    assert _spans(tight.uncertain) == [_span(20, 21)]


def test_subset_check_rejects_speech_outside_the_original_union() -> None:
    original = (_seg("A", 0.0, 1.0),)
    tight = (_seg("A", 0.0, 1.1),)

    assert is_subset_of_original(original, original, speaker="A")
    assert not is_subset_of_original(tight, original, speaker="A")


def test_subset_check_allows_speech_in_a_40ms_original_gap() -> None:
    original = (_seg("A", 0.0, 1.0), _seg("A", "1.04", 2.0))
    tight = (_seg("A", 0.0, 2.0),)

    assert is_subset_of_original(tight, original, speaker="A")


def test_subset_check_rejects_speech_in_a_250ms_original_gap() -> None:
    original = (_seg("A", 0.0, 1.0), _seg("A", "1.25", 2.0))
    tight = (_seg("A", 0.0, 2.0),)

    assert not is_subset_of_original(tight, original, speaker="A")


def test_writers_use_nine_decimals_and_write_empty_uncertain_files(tmp_path: Path) -> None:
    labels = RecordingLabels(
        "ami::EN2001a",
        (_seg("MEE071", 1.25, 2.5, "ami::EN2001a"),),
    )

    rttm_path, uncertain_path = write_recording_labels(tmp_path, labels)

    rttm = rttm_path.read_text(encoding="utf-8")
    assert rttm == ("SPEAKER ami::EN2001a 1 1.250000000 1.250000000 <NA> <NA> MEE071 <NA> <NA>\n")
    assert uncertain_path.read_text(encoding="utf-8") == ""
    parsed = parse_speaker_rttm(rttm)
    assert parsed == labels.segments
    assert parse_uncertain_rttm("") == ()


def test_uncertain_sidecar_matches_the_orchestrator_layout(tmp_path: Path) -> None:
    labels = RecordingLabels(
        "rec",
        (),
        (UncertainInterval.from_seconds("rec", "A", 3.0, 4.5),),
    )

    _, uncertain_path = write_recording_labels(tmp_path, labels)

    assert uncertain_path.read_text(encoding="utf-8") == (
        "UNCERTAIN rec 1 3.000000000 1.500000000 <NA> <NA> A <NA> <NA>\n"
    )


def test_rewrite_uem_keeps_times_and_replaces_the_file_id() -> None:
    text = "EN2001a 1 0.000000000 5250.240063000\n"

    rewritten = rewrite_uem_file_id(text, "ami::EN2001a")

    assert rewritten == "ami::EN2001a 1 0.000000000 5250.240063000\n"
    assert uem_bounds(text) == (0, parse_seconds_to_ns("5250.240063000"))


def test_tightness_counts_short_gaps_per_speaker_minute() -> None:
    segments = (
        _seg("A", 0.0, 1.0),
        _seg("A", 1.25, 2.25),
        _seg("A", 3.25, 4.25),
    )

    stats = measure_tightness(segments)

    assert stats.speaker_seconds == pytest.approx(3.0)
    assert stats.gap_count == 2
    assert stats.smallest_gap_s == pytest.approx(0.25)
    assert stats.smallest_gap_ns == parse_seconds_to_ns("0.25")
    assert stats.gaps_le_0_3_per_min == pytest.approx(20.0)
    assert stats.gaps_0_3_to_1_per_min == pytest.approx(20.0)


def test_speaker_mapping_uses_optimal_assignment_not_greedy() -> None:
    published = RecordingLabels(
        "rec",
        (
            _seg("A", 0.0, 10.0),
            _seg("B", 20.0, 29.0),
        ),
    )
    original = RecordingLabels(
        "rec",
        (
            _seg("X", 0.0, 10.0),
            _seg("X", 20.0, 29.0),
            _seg("Y", 0.0, 9.0),
        ),
    )

    mapping = map_speakers_by_overlap(published, original)

    assert mapping.as_dict() == {"A": "Y", "B": "X"}
    by_pub = {item.published_speaker: item.overlap_seconds for item in mapping.matches}
    assert by_pub["A"] == pytest.approx(9.0)
    assert by_pub["B"] == pytest.approx(9.0)
    assert mapping.unmapped_published == ()
    assert mapping.unmapped_published_ns == 0


def test_published_replacement_rewrites_speakers_and_merges() -> None:
    original = RecordingLabels("ami::EN2001a", (_seg("MEE071", 0.0, 5.0, "ami::EN2001a"),))
    published = RecordingLabels(
        "EN2001a",
        (
            _seg("EN2001a.B", 0.0, 1.0, "EN2001a"),
            _seg("EN2001a.B", 1.1, 2.0, "EN2001a"),
        ),
    )

    replaced, mapping = apply_published_replacement(original, published)

    assert mapping.as_dict() == {"EN2001a.B": "MEE071"}
    assert replaced.recording_id == "ami::EN2001a"
    assert [(item.speaker, item.start_ns, item.end_ns) for item in replaced.segments] == [
        ("MEE071", *_span(0, 2))
    ]


def test_published_replacement_clips_to_uem_before_merge() -> None:
    original = RecordingLabels("rec", (_seg("X", 0, 5),))
    published = RecordingLabels(
        "pub",
        (
            _seg("A", 4, 10, "pub"),
            _seg("A", 10.1, 12, "pub"),
        ),
    )

    replaced, mapping = apply_published_replacement(
        original,
        published,
        uem=(0, parse_seconds_to_ns(5)),
    )

    assert mapping.as_dict() == {"A": "X"}
    assert _spans(replaced.segments) == [_span(4, 5)]


def test_clip_drops_rows_that_start_at_or_after_uem_end() -> None:
    labels = RecordingLabels(
        "rec",
        (
            _seg("A", 0, 2),
            _seg("A", 5, 6),
            _seg("A", 6, 8),
        ),
    )

    clipped = clip_recording_to_uem(labels, 0, parse_seconds_to_ns(5))

    assert _spans(clipped.segments) == [_span(0, 2)]


def test_gates_flag_speakers_below_30_percent_and_outside_speech() -> None:
    original = [
        RecordingLabels(
            "rec",
            (_seg("A", 0.0, 10.0), _seg("B", 0.0, 10.0)),
        )
    ]
    new = [
        RecordingLabels(
            "rec",
            (_seg("A", 0.0, 2.0), _seg("B", 0.0, 8.0), _seg("B", 20.0, 21.0)),
        )
    ]

    gate = evaluate_gates(original, new, require_subset=False)

    speakers = {item.speaker: item.ratio for item in gate.flagged_speakers}
    assert speakers["A"] == pytest.approx(0.2)
    assert "B" not in speakers
    assert "rec" in gate.flagged_recordings
    assert gate.speaker_retention_ok is False
    assert gate.tightness_ok is False
    assert gate.hard_gates_ok is True
    assert gate.passed is True


def test_gates_fail_when_a_row_falls_outside_the_uem() -> None:
    original = [RecordingLabels("rec", (_seg("A", 0, 2),))]
    new = [RecordingLabels("rec", (_seg("A", 0, "2.0003125"),))]

    gate = evaluate_gates(
        original,
        new,
        uem_by_recording={"rec": (0, parse_seconds_to_ns(2))},
    )

    assert gate.uem_ok is False
    assert gate.passed is False
    assert gate.rows_outside_uem


def test_gates_fail_when_a_published_speaker_is_unmapped() -> None:
    original = [RecordingLabels("rec", (_seg("X", 0, 10),))]
    new = [RecordingLabels("rec", (_seg("X", 0, 10),))]
    mapping = map_speakers_by_overlap(
        RecordingLabels("rec", (_seg("A", 0, 10), _seg("B", 20, 25))),
        original[0],
    )

    assert mapping.unmapped_published == ("B",)
    assert mapping.unmapped_published_ns == parse_seconds_to_ns(5)

    gate = evaluate_gates(original, new, mappings={"rec": mapping})

    assert gate.unmapped_speakers_ok is False
    assert gate.passed is False
    assert gate.dropped_unmapped_seconds == pytest.approx(5.0)
    assert gate.unmapped_published == ("rec B",)


def test_listening_windows_prefer_the_longest_changes() -> None:
    original = [
        RecordingLabels("rec", (_seg("A", 0.0, 10.0), _seg("A", 50.0, 51.0))),
    ]
    new = [
        RecordingLabels("rec", (_seg("A", 0.0, 1.0), _seg("A", 50.0, 51.0))),
    ]

    windows = select_listening_windows(
        original,
        new,
        duration_by_recording={"rec": 60.0},
        max_pack_seconds=20.0,
        max_clip_seconds=8.0,
        min_clip_seconds=2.0,
        pad_seconds=0.5,
    )

    assert windows
    assert windows[0].recording_id == "rec"
    assert windows[0].change_seconds == pytest.approx(9.0)


def test_listening_pack_writes_wav_and_csv(tmp_path: Path) -> None:
    from recipes.speakrs.large.tight_labels.listening import write_listening_pack

    audio = tmp_path / "mix.wav"
    sf.write(audio, np.zeros(16_000 * 5, dtype=np.float32), 16_000)
    original = [RecordingLabels("rec", (_seg("A", 0.0, 2.0),))]
    new = [RecordingLabels("rec", (_seg("A", 0.0, 0.4),))]

    result = write_listening_pack(
        tmp_path / "pack",
        original=original,
        new=new,
        audio_by_recording={"rec": audio},
        duration_by_recording={"rec": 5.0},
    )

    assert result["clip_count"] == 1
    csv_text = Path(result["csv"]).read_text(encoding="utf-8")
    assert "original" in csv_text
    assert "new" in csv_text
    assert list((tmp_path / "pack").glob("*.wav"))


def test_format_ns_writes_nine_decimals_from_integers() -> None:
    assert format_ns(MIN_GAP_NS) == "0.200000000"
    assert format_ns(parse_seconds_to_ns("3325.128")) == "3325.128000000"


def test_gates_fail_when_smallest_gap_is_below_200ms() -> None:
    original = [RecordingLabels("rec", (_seg("A", 0.0, 1.0), _seg("A", "1.04", 2.0)))]
    new = [RecordingLabels("rec", (_seg("A", 0.0, 1.0), _seg("A", "1.04", 2.0)))]

    gate = evaluate_gates(original, new, require_subset=True)

    assert gate.smallest_gap_ok is False
    assert gate.hard_gates_ok is False
    assert gate.passed is False


def test_gates_fail_subset_outside_the_200ms_closed_original() -> None:
    original = [RecordingLabels("rec", (_seg("A", 0.0, 1.0), _seg("A", "1.25", 2.0)))]
    new = [RecordingLabels("rec", (_seg("A", 0.0, 2.0),))]

    gate = evaluate_gates(original, new, require_subset=True)

    assert gate.subset_ok is False
    assert gate.hard_gates_ok is False
    assert gate.passed is False


def test_segment_class_rejects_string_outcomes() -> None:
    with pytest.raises(TypeError, match="SegmentOutcome"):
        SegmentClass("rec", "A", 0, parse_seconds_to_ns(1), "alignment_failure")  # type: ignore[arg-type]


def test_gates_pass_failure_share_at_the_025_limit() -> None:
    originals = (_seg("A", 0, 1), _seg("A", 2, 3), _seg("B", 0, 1), _seg("B", 2, 3))
    original = [RecordingLabels("rec", originals)]
    new = [
        RecordingLabels(
            "rec",
            (_seg("A", 0, 1), _seg("A", 2, 3), _seg("B", 0, 1)),
            (_unc("B", 2, 3),),
        )
    ]
    classes = (
        _cls(originals[0], SegmentOutcome.ALIGNED),
        _cls(originals[1], SegmentOutcome.ALIGNED),
        _cls(originals[2], SegmentOutcome.ALIGNED),
        _cls(
            originals[3],
            SegmentOutcome.ALIGNMENT_FAILURE,
            failure_reason=AlignmentFailureReason.LOW_COVERAGE,
        ),
    )

    gate = evaluate_gates(original, new, segment_classes=classes)

    assert gate.failure_share == pytest.approx(0.25)
    assert gate.by_design_share == pytest.approx(0.0)
    assert gate.failure_share_ok is True
    assert gate.passed is True


def test_gates_fail_failure_share_above_025() -> None:
    originals = (_seg("A", 0, 1), _seg("A", 2, 3), _seg("B", 0, 1))
    original = [RecordingLabels("rec", originals)]
    new = [
        RecordingLabels(
            "rec",
            (_seg("A", 0, 1), _seg("A", 2, 3)),
            (_unc("B", 0, 1),),
        )
    ]
    classes = (
        _cls(originals[0], SegmentOutcome.ALIGNED),
        _cls(originals[1], SegmentOutcome.ALIGNED),
        _cls(
            originals[2],
            SegmentOutcome.ALIGNMENT_FAILURE,
            failure_reason=AlignmentFailureReason.ALIGNER_ERROR,
        ),
    )

    gate = evaluate_gates(original, new, segment_classes=classes)

    assert gate.failure_share == pytest.approx(1 / 3)
    assert 0.0 <= gate.failure_share <= 1.0
    assert gate.failure_share_ok is False
    assert gate.hard_gates_ok is True
    assert gate.passed is False


def test_gates_report_uncertain_share_per_recording() -> None:
    original = [
        RecordingLabels(
            "rec",
            (_seg("A", 0.0, 10.0), _seg("B", 0.0, 10.0), _seg("C", 0.0, 10.0), _seg("D", 0.0, 10.0), _seg("E", 0.0, 10.0)),
        )
    ]
    new = [
        RecordingLabels(
            "rec",
            (
                _seg("A", 0.0, 6.0),
                _seg("B", 0.0, 10.0),
                _seg("C", 0.0, 10.0),
                _seg("D", 0.0, 10.0),
                _seg("E", 0.0, 10.0),
            ),
            (UncertainInterval.from_seconds("rec", "A", 6.0, 10.0),),
        )
    ]

    gate = evaluate_gates(original, new)

    assert gate.uncertain_share == pytest.approx(0.08)
    assert len(gate.uncertain_by_recording) == 1
    assert gate.uncertain_by_recording[0].recording_id == "rec"
    assert gate.uncertain_by_recording[0].ratio == pytest.approx(0.08)
    assert gate.uncertain_overlap_ok is True
    payload = gate.to_json()
    assert payload["uncertain_share"] == pytest.approx(0.08)
    assert payload["uncertain_share_excluding_by_design"] == pytest.approx(0.08)
    assert payload["by_design_share"] == pytest.approx(0.0)
    assert payload["keeps_current_labels"] is False
    assert gate.passed is True


def test_gates_count_three_uncertain_rows_as_one_original_segment() -> None:
    human = _seg("A", 0, 4)
    original = [RecordingLabels("rec", (human, _seg("B", 0, 1), _seg("C", 0, 1), _seg("D", 0, 1)))]
    new = [
        RecordingLabels(
            "rec",
            (_seg("B", 0, 1), _seg("C", 0, 1), _seg("D", 0, 1)),
            (_unc("A", 0, 1), _unc("A", 1, 2), _unc("A", 2, 4)),
        )
    ]
    classes = (
        _cls(human, SegmentOutcome.ALIGNMENT_FAILURE, failure_reason=AlignmentFailureReason.MARKER_ONLY),
        _cls(_seg("B", 0, 1), SegmentOutcome.ALIGNED),
        _cls(_seg("C", 0, 1), SegmentOutcome.ALIGNED),
        _cls(_seg("D", 0, 1), SegmentOutcome.ALIGNED),
    )

    gate = evaluate_gates(original, new, segment_classes=classes)

    assert len(new[0].uncertain) == 1
    assert gate.failure_share == pytest.approx(0.25)
    assert 0.0 <= gate.failure_share <= 1.0


def test_gates_exclude_by_design_segments_from_failure_share() -> None:
    originals = (_seg("A", 0, 1), _seg("B", 0, 1), _seg("C", 0, 1), _seg("D", 0, 1))
    original = [RecordingLabels("rec", originals)]
    new = [
        RecordingLabels(
            "rec",
            (_seg("A", 0, 1),),
            (_unc("B", 0, 1), _unc("C", 0, 1), _unc("D", 0, 1)),
        )
    ]
    classes = (
        _cls(originals[0], SegmentOutcome.ALIGNED),
        _cls(
            originals[1],
            SegmentOutcome.BY_DESIGN_UNCERTAIN,
            by_design_reason=ByDesignReason.MULTI_SPEAKER,
        ),
        _cls(
            originals[2],
            SegmentOutcome.BY_DESIGN_UNCERTAIN,
            by_design_reason=ByDesignReason.HELD_OUT_LANGUAGE,
        ),
        _cls(
            originals[3],
            SegmentOutcome.ALIGNMENT_FAILURE,
            failure_reason=AlignmentFailureReason.EMPTY_ROMANIZATION,
        ),
    )

    gate = evaluate_gates(original, new, segment_classes=classes)

    assert gate.failure_share == pytest.approx(0.25)
    assert gate.by_design_share == pytest.approx(0.5)
    assert gate.failure_share_ok is True
    assert gate.uncertain_share == pytest.approx(0.75)
    assert gate.uncertain_share_excluding_by_design == pytest.approx(0.25)
    assert 0.0 <= gate.failure_share <= 1.0
    assert 0.0 <= gate.by_design_share <= 1.0


def test_gates_keep_touching_original_rows_as_separate_segments() -> None:
    originals = (_seg("A", 0, 1), _seg("A", 1, 2))
    original = [RecordingLabels("rec", originals)]
    new = [RecordingLabels("rec", (_seg("A", 0, 1),), (_unc("A", 1, 2),))]
    classes = (
        _cls(originals[0], SegmentOutcome.ALIGNED),
        _cls(
            originals[1],
            SegmentOutcome.BY_DESIGN_UNCERTAIN,
            by_design_reason=ByDesignReason.MULTI_SPEAKER,
        ),
    )

    gate = evaluate_gates(original, new, segment_classes=classes)

    assert gate.failure_share == pytest.approx(0.0)
    assert gate.by_design_share == pytest.approx(0.5)


def test_gates_union_overlapping_original_rows_before_counting() -> None:
    original = [RecordingLabels("rec", (_seg("A", 0, 2), _seg("A", 1, 3)))]
    new = [RecordingLabels("rec", (), (_unc("A", 0, 3),))]
    unioned = _seg("A", 0, 3)
    classes = (
        _cls(unioned, SegmentOutcome.ALIGNMENT_FAILURE, failure_reason=AlignmentFailureReason.LOW_COVERAGE),
    )

    gate = evaluate_gates(original, new, segment_classes=classes)

    assert gate.failure_share == pytest.approx(1.0)
    with pytest.raises(ValueError, match="cover each unioned original segment once"):
        evaluate_gates(
            original,
            new,
            segment_classes=(
                _cls(_seg("A", 0, 2), SegmentOutcome.ALIGNED),
                _cls(
                    _seg("A", 1, 3),
                    SegmentOutcome.ALIGNMENT_FAILURE,
                    failure_reason=AlignmentFailureReason.LOW_COVERAGE,
                ),
            ),
        )


def test_gates_failure_share_stays_in_unit_interval_with_many_uncertain_rows() -> None:
    human = _seg("A", 0, 1)
    original = [RecordingLabels("rec", (human,))]
    new = [RecordingLabels("rec", (), (_unc("A", 0, "0.2"), _unc("A", "0.4", "0.6"), _unc("A", "0.8", 1)))]
    classes = (
        _cls(human, SegmentOutcome.ALIGNMENT_FAILURE, failure_reason=AlignmentFailureReason.ALIGNER_ERROR),
    )

    gate = evaluate_gates(original, new, segment_classes=classes)

    assert gate.failure_share == pytest.approx(1.0)
    assert 0.0 <= gate.failure_share <= 1.0


def test_recording_labels_unions_overlapping_uncertain_per_speaker() -> None:
    labels = RecordingLabels(
        "rec",
        (),
        (_unc("A", 0, 2), _unc("A", 1, 3), _unc("B", 0, 1)),
    )

    assert _spans(labels.uncertain_for("A")) == [_span(0, 3)]
    assert _spans(labels.uncertain_for("B")) == [_span(0, 1)]


def test_recording_labels_unions_nested_uncertain_regions() -> None:
    labels = RecordingLabels("rec", (), (_unc("A", 0, 5), _unc("A", 1, 2), _unc("A", 3, 4)))

    assert _spans(labels.uncertain) == [_span(0, 5)]


def test_recording_labels_unions_touching_uncertain_regions() -> None:
    labels = RecordingLabels("rec", (), (_unc("A", 0, 1), _unc("A", 1, 2)))

    assert _spans(labels.uncertain) == [_span(0, 2)]


def test_recording_labels_subtracts_active_speech_from_uncertain() -> None:
    labels = RecordingLabels(
        "rec",
        (_seg("A", 1, 2),),
        (_unc("A", 0, 3), _unc("A", "0.5", "2.5")),
    )

    assert _spans(labels.segments) == [_span(1, 2)]
    assert _spans(labels.uncertain) == [_span(0, 1), _span(2, 3)]


def test_subtract_uncertain_from_active_unions_then_cuts() -> None:
    remaining = subtract_uncertain_from_active(
        (_unc("A", 0, 2), _unc("A", 1, 4), _unc("A", 4, 5)),
        (_seg("A", 2, 3),),
    )

    assert _spans(remaining) == [_span(0, 2), _span(3, 5)]


def test_gate_report_uses_failure_share_and_retention_parameters() -> None:
    from recipes.speakrs.large.tight_labels.produce import render_gate_markdown

    originals = (_seg("A", 0, 1),)
    original = [RecordingLabels("rec", originals)]
    new = [RecordingLabels("rec", originals)]
    gate = evaluate_gates(
        original,
        new,
        failure_share_limit=0.4,
        speaker_retention_floor=0.15,
    )
    text = render_gate_markdown("AMI", "ami-tight-v3", gate, kept_current=False)

    assert "Failure share ≤ 0.40:" in text
    assert "Flagged speakers below 15% retention:" in text
    assert gate.failure_share_limit == pytest.approx(0.4)
    assert gate.speaker_retention_floor == pytest.approx(0.15)
