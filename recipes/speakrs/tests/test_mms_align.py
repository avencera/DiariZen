from __future__ import annotations

import numpy as np
import pytest

from recipes.speakrs.large.indic_diarbench import canonical_speaker_id
from recipes.speakrs.large.tight_labels.close_talk import (
    map_lavaliers,
    measure_clock_offset,
    shift_close_talk,
    voiced_spans,
    voicing_uncertain,
)
from recipes.speakrs.large.tight_labels.domain import Interval, RecordingLabels, SpeakerSegment, UncertainInterval
from recipes.speakrs.large.tight_labels.lotus_text import parse_lotus_align_intervals
from recipes.speakrs.large.tight_labels.markers import prepare_transcript
from recipes.speakrs.large.tight_labels.mms import ctc_alignment_fits, leftover_script_rate, mms_words
from recipes.speakrs.large.tight_labels.produce_mms import (
    TranscriptSegment,
    apply_extra_uncertain,
    held_out_labels,
    match_segments,
    overlap_uncertain_regions,
)
from recipes.speakrs.large.tight_labels.time import float_bounds_to_rttm_ns, parse_seconds_to_ns
from recipes.speakrs.large.tight_labels.tighten import tighten_recording, tighten_segment


def test_float_bounds_reproduce_parent_rttm_times() -> None:
    start_ns, end_ns = float_bounds_to_rttm_ns(0.132, 9.792)

    assert start_ns == parse_seconds_to_ns("0.132000000")
    assert end_ns - start_ns == parse_seconds_to_ns("9.660000000")


def test_unintelligible_markers_are_not_alignable() -> None:
    prepared = prepare_transcript("<unintelligible>")

    assert prepared.has_marker is True
    assert prepared.marker_only is True
    assert prepared.alignable is False


def test_mixed_markers_leave_spoken_text() -> None:
    prepared = prepare_transcript("M01 leftover <n> สวัสดี [F] ครับ <n>")

    assert prepared.has_marker is True
    assert prepared.alignable is True
    assert "สวัสดี" in prepared.remainder
    assert "<n>" not in prepared.remainder


def test_mms_words_keep_only_fa_characters() -> None:
    assert mms_words("Namaste Duniyaa!") == ("namaste", "duniyaa")
    assert mms_words("...") == ()


def test_leftover_script_rate_detects_unromanized_letters() -> None:
    assert leftover_script_rate("namaste") == 0.0
    assert leftover_script_rate("नमस्ते") == pytest.approx(1.0)


def test_match_segments_pairs_canonical_speakers_and_rttm_times() -> None:
    speaker = canonical_speaker_id("SPEAKER_03")
    start_ns, end_ns = float_bounds_to_rttm_ns(0.132, 9.792)
    original = (SpeakerSegment("indic-diarbench::assamese_001", speaker, start_ns, end_ns),)
    transcripts = (
        TranscriptSegment(speaker, start_ns, end_ns, "text", "Assamese", "SPEAKER_03"),
    )

    matched, report = match_segments(original, transcripts)

    assert report.matched == 1
    assert report.unmatched_original == 0
    assert matched[(speaker, start_ns, end_ns)].text == "text"


def test_lotus_parser_keeps_single_speaker_text_and_marks_overlap() -> None:
    textgrid = """
File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 4.0
item [1]:
    class = "IntervalTier"
    name = "Con123"
    intervals: size = 3
    intervals [1]:
        xmin = 0.0
        xmax = 1.0
        text = "<n>"
    intervals [2]:
        xmin = 1.0
        xmax = 2.0
        text = "F01, สวัสดี"
    intervals [3]:
        xmin = 2.0
        xmax = 4.0
        text = "F01&M02, overlap text"
"""

    intervals = parse_lotus_align_intervals(textgrid)

    assert intervals[0].non_speech is True
    assert intervals[1].speakers == ("F01",)
    assert intervals[1].text == "สวัสดี"
    assert intervals[1].overlap is False
    assert intervals[2].speakers == ("F01", "M02")
    assert intervals[2].overlap is True


def test_lavalier_map_requires_a_clear_unique_ratio() -> None:
    energy = {
        "Lav1": {"F01": 0.9, "M01": 0.1, "F02": 0.05},
        "Lav2": {"F01": 0.1, "M01": 0.8, "F02": 0.1},
        "Lav3": {"F01": 0.05, "M01": 0.1, "F02": 0.7},
    }
    exclusive = {"F01": 20.0, "M01": 20.0, "F02": 20.0}

    mapping = map_lavaliers(energy, exclusive)

    assert mapping.use_close_talk is True
    assert mapping.channel_for("F01") == "Lav1"
    assert mapping.channel_for("M01") == "Lav2"
    assert mapping.channel_for("F02") == "Lav3"


def test_lavalier_map_falls_back_when_energy_is_tied() -> None:
    energy = {
        "Lav1": {"F01": 0.5, "M01": 0.5},
        "Lav2": {"F01": 0.4, "M01": 0.4},
    }

    mapping = map_lavaliers(energy, {"F01": 20.0, "M01": 20.0})

    assert mapping.use_close_talk is False


def test_clock_offset_recovers_a_known_delay() -> None:
    mix = np.random.default_rng(0).normal(0, 0.1, 16_000 * 20).astype(np.float32)
    delay = 80
    close = np.zeros_like(mix)
    close[delay:] = mix[:-delay]

    measured = measure_clock_offset(mix, close, windows=4, window_seconds=2.0)

    assert measured.consistent is True
    assert abs(measured.offset_samples - delay) <= 2


def test_shift_close_talk_places_the_channel_on_the_mix_clock() -> None:
    samples = np.arange(10, dtype=np.float32)
    shifted = shift_close_talk(samples, 2, 10)

    np.testing.assert_array_equal(shifted[:2], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(shifted[2:], samples[:8])


def test_voicing_marks_close_talk_speech_the_aligner_left_silent() -> None:
    human = Interval.from_seconds(0, 2)
    aligned = ((parse_seconds_to_ns(0), parse_seconds_to_ns(1)),)
    voiced = ((parse_seconds_to_ns("1.2"), parse_seconds_to_ns("1.8")),)

    regions = voicing_uncertain("rec", "F01", human, aligned, voiced)

    assert len(regions) == 1
    assert regions[0].start_ns == parse_seconds_to_ns("1.2")
    assert regions[0].end_ns == parse_seconds_to_ns("1.8")


def test_voiced_spans_ignore_low_energy_frames() -> None:
    samples = np.zeros(16_000, dtype=np.float32)
    samples[8_000:10_000] = 0.5

    spans = voiced_spans(samples, 16_000, speech_rms=0.4, abs_floor=0.05, relative=0.2)

    assert spans
    assert spans[0][0] >= parse_seconds_to_ns("0.4")
    assert spans[0][1] <= parse_seconds_to_ns("0.7")


def test_tighten_marks_unintelligible_original_rows_uncertain() -> None:
    original = RecordingLabels(
        "rec",
        (SpeakerSegment.from_seconds("rec", "A", 0, 2), SpeakerSegment.from_seconds("rec", "A", 5, 7)),
    )
    aligned = {"A": (Interval.from_seconds(0.2, 1.8),)}

    tight = tighten_recording(original, aligned)

    assert tight.segments[0].start_ns == parse_seconds_to_ns("0.2")
    assert tight.uncertain == (UncertainInterval.from_seconds("rec", "A", 5, 7),)


def test_held_out_language_uses_uncertain_interiors() -> None:
    original = RecordingLabels(
        "indic-diarbench::santali_001",
        (
            SpeakerSegment.from_seconds("indic-diarbench::santali_001", "A", 0, 2),
            SpeakerSegment.from_seconds("indic-diarbench::santali_001", "B", 1, 3),
        ),
    )

    labels = held_out_labels(original)

    assert labels.segments == ()
    assert {item.speaker for item in labels.uncertain} == {"A", "B"}
    assert sum(item.duration_ns for item in labels.uncertain) == sum(item.duration_ns for item in original.segments)


def test_lotus_overlap_is_uncertain_beside_aligned_speech() -> None:
    recording = "lotusdis::meeting"
    original = RecordingLabels(
        recording,
        (
            SpeakerSegment.from_seconds(recording, "F01", 0, 2),
            SpeakerSegment.from_seconds(recording, "F01", 2, 4),
            SpeakerSegment.from_seconds(recording, "M02", 2, 4),
        ),
    )
    intervals = parse_lotus_align_intervals(
        """
File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 4.0
item [1]:
    class = "IntervalTier"
    name = "Con123"
    intervals: size = 2
    intervals [1]:
        xmin = 0.0
        xmax = 2.0
        text = "F01, sawasdee"
    intervals [2]:
        xmin = 2.0
        xmax = 4.0
        text = "F01&M02, overlap text"
"""
    )
    extra = overlap_uncertain_regions(recording, intervals)
    active = []
    uncertain = []
    aligned = {"F01": (Interval.from_seconds(0.2, 1.8),)}
    for segment in original.segments:
        failed = (segment.start_ns, segment.end_ns) == (
            parse_seconds_to_ns("2.000000000"),
            parse_seconds_to_ns("4.000000000"),
        )
        result = tighten_segment(segment, aligned.get(segment.speaker, ()), alignment_failed=failed)
        active.extend(result.active)
        uncertain.extend(result.uncertain)
    labels = apply_extra_uncertain(RecordingLabels(recording, tuple(active), tuple(uncertain)), extra)

    assert labels.segments == (SpeakerSegment.from_seconds(recording, "F01", "0.2", "1.8"),)
    speakers = {(item.speaker, item.start_ns, item.end_ns) for item in labels.uncertain}
    assert ("F01", parse_seconds_to_ns("2.000000000"), parse_seconds_to_ns("4.000000000")) in speakers
    assert ("M02", parse_seconds_to_ns("2.000000000"), parse_seconds_to_ns("4.000000000")) in speakers


def test_produce_mms_progress_log_records_failed_and_nonzero_exit(tmp_path, monkeypatch) -> None:
    from recipes.speakrs.large.tight_labels import produce_mms as mod

    def boom(*_args, **_kwargs):
        raise RuntimeError("alignment crashed")

    monkeypatch.setattr(mod, "produce", boom)
    code = mod.main(["--output-root", str(tmp_path)])

    log_text = (tmp_path / "work" / "progress.log").read_text(encoding="utf-8")
    assert code == 1
    assert log_text.strip().endswith("FAILED alignment crashed") or "FAILED alignment crashed" in log_text


def test_ctc_alignment_fits_counts_blank_between_repeated_tokens() -> None:
    # 70 tokens with 7 adjacent repeats need 77 frames; the failing Indic segment had 76
    tokens = [[1, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 7
    assert sum(len(word) for word in tokens) == 70
    assert not ctc_alignment_fits(tokens, 76)
    assert ctc_alignment_fits(tokens, 77)


def test_ctc_alignment_fits_counts_repeats_across_word_boundaries() -> None:
    assert not ctc_alignment_fits([[1, 2], [2, 3]], 4)
    assert ctc_alignment_fits([[1, 2], [2, 3]], 5)


def test_release_id_with_version_replaces_the_suffix() -> None:
    from recipes.speakrs.large.tight_labels.produce import release_id_with_version

    assert release_id_with_version("lotusdis-tight-v1", None) == "lotusdis-tight-v1"
    assert release_id_with_version("lotusdis-tight-v1", "v2") == "lotusdis-tight-v2"
    assert release_id_with_version("lotusdis-tight-v1", "2") == "lotusdis-tight-v2"


def test_mms_tighten_merges_a_40ms_gap_between_human_segments(tmp_path) -> None:
    from recipes.speakrs.large.tight_labels.mms import AlignedWord
    from recipes.speakrs.large.tight_labels.produce import RecordingRef
    from recipes.speakrs.large.tight_labels.produce_mms import _tighten

    original = RecordingLabels(
        "rec",
        (
            SpeakerSegment.from_seconds("rec", "A", 0.0, 1.0),
            SpeakerSegment.from_seconds("rec", "A", "1.04", 2.0),
        ),
    )
    words = {
        "A": (
            AlignedWord(parse_seconds_to_ns(0), parse_seconds_to_ns(1), "a", 1.0),
            AlignedWord(parse_seconds_to_ns("1.04"), parse_seconds_to_ns(2), "b", 1.0),
        )
    }
    ref = RecordingRef(
        "rec",
        tmp_path / "a.wav",
        tmp_path / "a.uem",
        parse_seconds_to_ns(3),
        0,
        parse_seconds_to_ns(3),
    )

    tight = _tighten(original, words, ref)

    assert [(item.start_ns, item.end_ns) for item in tight.segments] == [
        (parse_seconds_to_ns(0), parse_seconds_to_ns(2))
    ]
