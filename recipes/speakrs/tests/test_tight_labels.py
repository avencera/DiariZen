from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recipes.speakrs.large.tight_labels import (
    COVERAGE_RATIO,
    MIN_GAP_SECONDS,
    Interval,
    RecordingLabels,
    SpeakerSegment,
    UncertainInterval,
    apply_published_replacement,
    evaluate_gates,
    is_subset_of_original,
    map_speakers_by_overlap,
    measure_tightness,
    merge_short_gaps,
    parse_speaker_rttm,
    tighten_recording,
    tighten_segment,
    write_recording_labels,
)
from recipes.speakrs.large.tight_labels.listening import select_listening_windows
from recipes.speakrs.large.tight_labels.writers import parse_uncertain_rttm, rewrite_uem_file_id


def _seg(speaker: str, start: float, end: float, recording: str = "rec") -> SpeakerSegment:
    return SpeakerSegment(recording, speaker, start, end)


def test_merge_fills_gaps_shorter_than_200ms_and_keeps_exact_200ms() -> None:
    segments = (
        _seg("A", 0.0, 1.0),
        _seg("A", 1.15, 2.0),
        _seg("A", 2.2, 3.0),
        _seg("B", 0.0, 1.0),
        _seg("B", 1.15, 2.0),
    )

    merged = merge_short_gaps(segments, min_gap=MIN_GAP_SECONDS)

    a_spans = [(item.start, item.end) for item in merged if item.speaker == "A"]
    b_spans = [(item.start, item.end) for item in merged if item.speaker == "B"]
    assert a_spans == [(0.0, 2.0), (2.2, 3.0)]
    assert b_spans == [(0.0, 2.0)]


def test_merge_joins_overlapping_and_touching_segments() -> None:
    segments = (
        _seg("A", 0.0, 1.0),
        _seg("A", 0.5, 1.5),
        _seg("A", 1.5, 2.0),
    )

    merged = merge_short_gaps(segments)

    assert [(item.start, item.end) for item in merged] == [(0.0, 2.0)]


def test_tighten_active_speech_is_subset_of_the_human_segment() -> None:
    human = _seg("A", 10.0, 20.0)
    aligned = (
        Interval(9.0, 11.0),
        Interval(12.0, 13.0),
        Interval(14.0, 17.0),
        Interval(19.5, 22.0),
    )

    result = tighten_segment(human, aligned)

    assert result.fallback is False
    assert result.uncertain == ()
    assert [(item.start, item.end) for item in result.active] == [
        (10.0, 11.0),
        (12.0, 13.0),
        (14.0, 17.0),
        (19.5, 20.0),
    ]
    assert is_subset_of_original(result.active, (human,), speaker="A")


def test_tighten_marks_uncertain_when_coverage_is_below_30_percent() -> None:
    human = _seg("A", 0.0, 10.0)
    aligned = (Interval(0.0, 2.9),)

    result = tighten_segment(human, aligned, min_coverage=COVERAGE_RATIO)

    assert result.fallback is True
    assert result.active == ()
    assert result.uncertain == (UncertainInterval("rec", "A", 0.0, 10.0),)


def test_tighten_keeps_speech_when_coverage_is_exactly_30_percent() -> None:
    human = _seg("A", 0.0, 10.0)
    aligned = (Interval(1.0, 4.0),)

    result = tighten_segment(human, aligned, min_coverage=0.30)

    assert result.fallback is False
    assert [(item.start, item.end) for item in result.active] == [(1.0, 4.0)]


def test_alignment_failure_drops_active_speech() -> None:
    human = _seg("A", 0.0, 2.0)
    aligned = (Interval(0.0, 2.0),)

    result = tighten_segment(human, aligned, alignment_failed=True)

    assert result.fallback is True
    assert result.active == ()
    assert len(result.uncertain) == 1


def test_tighten_merges_short_gaps_inside_a_human_segment() -> None:
    human = _seg("A", 0.0, 5.0)
    aligned = (Interval(0.0, 1.0), Interval(1.1, 2.0))

    result = tighten_segment(human, aligned)

    assert [(item.start, item.end) for item in result.active] == [(0.0, 2.0)]


def test_tighten_does_not_fill_a_gap_between_human_segments() -> None:
    original = RecordingLabels(
        "rec",
        (_seg("A", 0.0, 1.0), _seg("A", 1.1, 2.0)),
    )
    aligned = {"A": (Interval(0.0, 1.0), Interval(1.1, 2.0))}

    tight = tighten_recording(original, aligned)

    assert [(item.start, item.end) for item in tight.segments] == [(0.0, 1.0), (1.1, 2.0)]
    assert is_subset_of_original(tight.segments, original.segments, speaker="A")


def test_subset_check_rejects_speech_outside_the_original_union() -> None:
    original = (_seg("A", 0.0, 1.0),)
    tight = (_seg("A", 0.0, 1.1),)

    assert is_subset_of_original(original, original, speaker="A")
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
        (UncertainInterval("rec", "A", 3.0, 4.5),),
    )

    _, uncertain_path = write_recording_labels(tmp_path, labels)

    assert uncertain_path.read_text(encoding="utf-8") == (
        "UNCERTAIN rec 1 3.000000000 1.500000000 <NA> <NA> A <NA> <NA>\n"
    )


def test_rewrite_uem_keeps_times_and_replaces_the_file_id() -> None:
    text = "EN2001a 1 0.000000000 5250.240063000\n"

    rewritten = rewrite_uem_file_id(text, "ami::EN2001a")

    assert rewritten == "ami::EN2001a 1 0.000000000 5250.240063000\n"


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
    assert [(item.speaker, item.start, item.end) for item in replaced.segments] == [("MEE071", 0.0, 2.0)]


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
