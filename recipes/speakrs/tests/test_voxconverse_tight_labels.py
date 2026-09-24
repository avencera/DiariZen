from __future__ import annotations

import numpy as np
import pytest

from recipes.speakrs.large.tight_labels.domain import Interval, RecordingLabels, SpeakerSegment, UncertainInterval
from recipes.speakrs.large.tight_labels.guards import (
    acoustic_guard,
    classify_transcript,
    dictionary_words,
    measure_acoustic,
    normalize_english_words,
    tokenize_words,
)
from recipes.speakrs.large.tight_labels.tighten import tighten_recording
from recipes.speakrs.large.tight_labels.voxconverse import choose_asr, parse_mfa_words, seconds_to_ns


def _tone(frequency: float, duration_s: float, sample_rate: int = 16_000, amplitude: float = 0.2) -> np.ndarray:
    times = np.arange(int(duration_s * sample_rate), dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def _speech_like(duration_s: float = 1.2, sample_rate: int = 16_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.2, int(duration_s * sample_rate))
    times = np.arange(noise.size) / sample_rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * times)
    return (noise * envelope).astype(np.float32)


def test_silence_cut_is_guarded() -> None:
    features = measure_acoustic(np.zeros(16_000, dtype=np.float32))
    assert acoustic_guard(features) == "SILENCE"


def test_constant_tone_is_guarded_as_music() -> None:
    features = measure_acoustic(_tone(440.0, 1.0))
    assert acoustic_guard(features) == "MUSIC_OR_NOISE"


def test_short_tone_skips_the_music_rule() -> None:
    features = measure_acoustic(_tone(440.0, 0.3))
    assert acoustic_guard(features) is None


def test_modulated_noise_passes_the_acoustic_guard() -> None:
    features = measure_acoustic(_speech_like())
    assert acoustic_guard(features) is None
    assert features.modulation_ratio >= 0.15


def test_empty_asr_is_guarded() -> None:
    assert classify_transcript("   ...  ", 2.0) == "EMPTY_ASR"


def test_implausible_character_rate_is_guarded() -> None:
    text = "abcdefghij" * 40
    assert classify_transcript(text, 3.0) == "IMPLAUSIBLE_CHAR_RATE"


def test_short_fast_speech_is_not_rate_guarded() -> None:
    text = "I mean I always look at the"
    assert classify_transcript(text, 0.8) is None


def test_repetition_loop_is_guarded() -> None:
    text = "music " * 12
    assert classify_transcript(text, 4.0) == "REPETITION_LOOP"


def test_normal_transcript_passes() -> None:
    text = "the committee will meet tomorrow morning in london"
    assert classify_transcript(text, 3.5) is None
    assert tokenize_words("Don't stop.") == ("don't", "stop")


def test_english_normalization_keeps_apostrophes_and_drops_punct() -> None:
    words = normalize_english_words("Hello, it's 2020! Café?")
    assert "hello" in words
    assert "it's" in words
    assert "2020" in words


def test_dictionary_words_read_the_first_field() -> None:
    text = "hello\t0.99\t0.14\t1.0\t1.0\th ɛ l ow\n# comment\nworld w ɝ l d\n"
    assert dictionary_words(text.splitlines()) == frozenset({"hello", "world"})


def test_choose_asr_uses_parakeet_for_european_languages() -> None:
    backend, language = choose_asr("fr", 0.9)
    assert backend == "parakeet"
    assert language == "fr"


def test_choose_asr_uses_whisper_outside_parakeet() -> None:
    backend, language = choose_asr("ar", 0.8)
    assert backend == "whisper-small"
    assert language == "ar"


def test_choose_asr_falls_back_to_english_when_lid_is_weak() -> None:
    backend, language = choose_asr("ja", 0.10)
    assert backend == "parakeet"
    assert language == "en"


def test_parse_mfa_json_skips_silence_labels() -> None:
    payload = {
        "tiers": {
            "words": {
                "entries": [
                    [0.0, 0.12, ""],
                    [0.12, 0.40, "hello"],
                    [0.40, 0.50, "sil"],
                    [0.50, 0.80, "world"],
                ]
            }
        }
    }
    assert parse_mfa_words(payload) == [(0.12, 0.40, "hello"), (0.50, 0.80, "world")]


def test_guarded_segment_becomes_uncertain_inside_the_human_bounds() -> None:
    original = RecordingLabels(
        "voxconverse::abjxc",
        (
            SpeakerSegment.from_seconds("voxconverse::abjxc", "spk00", 1.0, 4.0),
            SpeakerSegment.from_seconds("voxconverse::abjxc", "spk00", 6.0, 8.0),
        ),
    )
    aligned = {
        "spk00": (Interval.from_seconds(6.1, 7.2),),
    }

    tight = tighten_recording(original, aligned)

    assert [(item.start, item.end) for item in tight.segments] == pytest.approx([(6.1, 7.2)])
    assert tight.uncertain == (UncertainInterval.from_seconds("voxconverse::abjxc", "spk00", 1.0, 4.0),)


def test_seconds_to_ns_rounds_aligner_floats() -> None:
    assert seconds_to_ns(0.2) == 200_000_000
    assert seconds_to_ns(1.2345678904) == 1_234_567_890


def test_release_id_with_version_replaces_the_suffix() -> None:
    from recipes.speakrs.large.tight_labels.produce import release_id_with_version

    assert release_id_with_version("voxconverse-tight-v1", None) == "voxconverse-tight-v1"
    assert release_id_with_version("voxconverse-tight-v1", "v2") == "voxconverse-tight-v2"
    assert release_id_with_version("voxconverse-tight-v1", "2") == "voxconverse-tight-v2"


def test_classify_voxconverse_jobs_maps_guards_to_typed_outcomes() -> None:
    from recipes.speakrs.large.tight_labels.classify import AlignmentFailureReason, SegmentOutcome
    from recipes.speakrs.large.tight_labels.voxconverse import SegmentJob, classify_voxconverse_jobs

    recording = "voxconverse::abjxc"
    original = [
        RecordingLabels(
            recording,
            (
                SpeakerSegment.from_seconds(recording, "spk00", 0.0, 1.0),
                SpeakerSegment.from_seconds(recording, "spk00", 2.0, 3.0),
            ),
        )
    ]
    produced = original
    jobs = (
        SegmentJob("a", recording, "spk00", 0, 1_000_000_000, "a.wav"),
        SegmentJob("b", recording, "spk00", 2_000_000_000, 3_000_000_000, "a.wav"),
    )
    asr_rows = {"a": {"guard": "SILENCE"}, "b": {}}
    align_rows = {"a": {"ok": False, "error": "SILENCE"}, "b": {"ok": False, "error": "ALIGN_FAILED"}}

    classes = classify_voxconverse_jobs(original, produced, jobs, asr_rows, align_rows)

    outcomes = {item.start_ns: item for item in classes}
    assert outcomes[0].outcome is SegmentOutcome.ALIGNMENT_FAILURE
    assert outcomes[0].failure_reason is AlignmentFailureReason.SILENCE
    assert outcomes[2_000_000_000].outcome is SegmentOutcome.ALIGNMENT_FAILURE
    assert outcomes[2_000_000_000].failure_reason is AlignmentFailureReason.ALIGNER_ERROR


def test_agent_report_renders_speaker_hours() -> None:
    from pathlib import Path

    from recipes.speakrs.large.tight_labels.gates import evaluate_gates
    from recipes.speakrs.large.tight_labels.voxconverse import render_agent_report

    original = [
        RecordingLabels(
            "voxconverse::abjxc",
            (SpeakerSegment.from_seconds("voxconverse::abjxc", "spk00", 0.0, 3600.0),),
        )
    ]
    gate = evaluate_gates(original, original)
    report = render_agent_report(
        refs=[],
        gate=gate,
        kept_current=False,
        summary={
            "unioned_segments": 1,
            "asr_ok": 1,
            "asr_coverage": 1.0,
            "align_ok": 1,
            "alignment_failure_share": 0.0,
            "english_oov_words": 0,
            "language_counts": {},
            "guard_counts": {},
        },
        release_dir=Path("/tmp/voxconverse-tight-v2"),
        listening={"directory": "/tmp", "clip_count": 0, "total_seconds": 0},
    )
    assert "Original speaker hours: 1.000" in report
    assert "Produced speaker hours: 1.000" in report
