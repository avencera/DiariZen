"""Protect typed Open Yap review frames, decisions, and grid conversion."""

from __future__ import annotations

import pytest

from recipes.speakrs.large.errors import ContractError
from recipes.speakrs.large.hashing import sha256_json
from recipes.speakrs.large.review_models import (
    WINDOW_FRAME_COUNT,
    ActivityInterval,
    BoundedRangeScope,
    Clear,
    ClockClear,
    ClockMeasurementEntered,
    ClockNotReviewed,
    ConfirmAction,
    ConfirmedProposal,
    CorrectAction,
    Corrected,
    DefectSet,
    GridProposal,
    NeedsFollowUp,
    NeedsFollowUpAction,
    NoSpeech,
    NoSpeechAction,
    NotReviewed,
    Pending,
    Returned,
    SignedOff,
    SignOffAcceptAction,
    SignOffReturnAction,
    SpeakerRole,
    TranscriptWord,
    Uncertain,
    UncertainAction,
    UnresolvedDefect,
    WholeWindowScope,
    WindowReviewState,
    apply_action,
    clip_words_to_window,
    frames_intersecting_seconds,
    normalize_intervals,
    occupied_frames,
    parse_action,
    parse_decision,
    parse_source_transcript,
    proposal_from_candidate,
    seconds_from_frame,
)


def _interval(speaker: str, start: int, end: int) -> ActivityInterval:
    return ActivityInterval(SpeakerRole(speaker), start, end)


def test_interval_rejects_non_positive_and_out_of_window_ranges() -> None:
    with pytest.raises(ContractError, match="greater than start"):
        ActivityInterval(SpeakerRole.SPEAKER_A, 10, 10)
    with pytest.raises(ContractError, match="outside"):
        ActivityInterval(SpeakerRole.SPEAKER_A, -1, 4)
    with pytest.raises(ContractError, match="outside"):
        ActivityInterval(SpeakerRole.SPEAKER_A, 0, WINDOW_FRAME_COUNT + 1)
    with pytest.raises(ContractError, match="integer"):
        parse_decision(
            {"kind": "corrected", "activity": [{"speaker": "speaker_a", "start_frame": 0.5, "end_frame": 4}]}
        )


def test_normalize_merges_adjacent_same_speaker_and_keeps_overlap() -> None:
    merged = normalize_intervals(
        [
            _interval("speaker_a", 0, 2),
            _interval("speaker_a", 2, 5),
            _interval("speaker_a", 8, 9),
            _interval("speaker_b", 1, 4),
            _interval("speaker_b", 3, 6),
        ]
    )

    assert merged == (
        _interval("speaker_a", 0, 5),
        _interval("speaker_b", 1, 6),
        _interval("speaker_a", 8, 9),
    )
    frames = occupied_frames(merged)
    assert 3 in frames[SpeakerRole.SPEAKER_A]
    assert 3 in frames[SpeakerRole.SPEAKER_B]


def test_short_turn_and_gap_stay_distinct() -> None:
    intervals = normalize_intervals([_interval("speaker_a", 10, 11), _interval("speaker_a", 12, 14)])

    assert intervals == (_interval("speaker_a", 10, 11), _interval("speaker_a", 12, 14))


def test_frame_occupancy_marks_any_intersecting_part_of_a_20ms_frame() -> None:
    assert frames_intersecting_seconds(0.0, 0.02) == (0, 1)
    assert frames_intersecting_seconds(0.0, 0.001) == (0, 1)
    assert frames_intersecting_seconds(0.02, 0.04) == (1, 2)
    assert frames_intersecting_seconds(0.019, 0.021) == (0, 2)
    assert frames_intersecting_seconds(29.99, 30.0) == (1499, 1500)
    assert frames_intersecting_seconds(30.0, 30.2) is None
    assert frames_intersecting_seconds(1.0, 1.0) is None
    assert seconds_from_frame(5) == 0.1
    assert seconds_from_frame(1500) == 30.0


def test_proposal_conversion_is_content_bound_and_speaker_independent() -> None:
    candidate = {
        "window_id": "win-1",
        "speaker_activity": [
            {
                "speaker_role": "speaker_a",
                "intervals": [
                    {"start_seconds": 0.0, "end_seconds": 0.05},
                    {"start_seconds": 0.05, "end_seconds": 0.08},
                ],
            },
            {"speaker_role": "speaker_b", "intervals": [{"start_seconds": 0.04, "end_seconds": 0.07}]},
        ],
    }

    first = proposal_from_candidate(candidate, window_id="win-1", candidate_sha256="a" * 64)
    second = proposal_from_candidate(candidate, window_id="win-1", candidate_sha256="a" * 64)

    assert first.content_sha256 == second.content_sha256
    assert first.intervals == (
        _interval("speaker_a", 0, 4),
        _interval("speaker_b", 2, 4),
    )
    rebuilt = GridProposal.from_dict(first.to_dict())
    assert rebuilt.content_sha256 == first.content_sha256


def test_transcript_schema_rejects_unknown_fields_and_invalid_intervals() -> None:
    speaker = SpeakerRole.SPEAKER_A
    digest = "b" * 64
    words = parse_source_transcript(
        {
            "conversation_id": "conv",
            "speaker_index": "a",
            "language": "en",
            "text": "hello there",
            "corrections_applied": False,
            "words": [
                {"word": "hello", "start": 0.2, "end": 0.6, "type": "word", "corrections_applied": False},
                {"word": "typed", "start": None, "end": None, "type": "word"},
                {"word": "there", "start": 1.0, "end": 1.2, "type": "filler"},
            ],
        },
        speaker=speaker,
        source_sha256=digest,
    )
    assert [item["text"] for item in words] == ["hello", "there"]

    with pytest.raises(ContractError, match="unknown fields"):
        parse_source_transcript({"words": [], "unexpected": True}, speaker=speaker, source_sha256=digest)
    with pytest.raises(ContractError, match="unknown fields"):
        parse_source_transcript(
            {"words": [{"word": "x", "start": 0.0, "end": 0.2, "type": "word", "confidence": 0.9}]},
            speaker=speaker,
            source_sha256=digest,
        )
    skipped = parse_source_transcript(
        {"words": [{"word": "x", "start": 0.4, "end": 0.2, "type": "word"}]},
        speaker=speaker,
        source_sha256=digest,
    )
    assert skipped == []
    with pytest.raises(ContractError, match="speaker_index"):
        parse_source_transcript(
            {"speaker_index": "b", "words": [{"word": "x", "start": 0.0, "end": 0.2, "type": "word"}]},
            speaker=speaker,
            source_sha256=digest,
        )


def test_clipping_retains_only_intersecting_window_words() -> None:
    words = [
        {
            "speaker": SpeakerRole.SPEAKER_A,
            "text": "before",
            "type": "word",
            "start": 0.0,
            "end": 0.4,
            "source_transcript_sha256": "c" * 64,
        },
        {
            "speaker": SpeakerRole.SPEAKER_B,
            "text": "edge",
            "type": "laugh",
            "start": 9.9,
            "end": 10.2,
            "source_transcript_sha256": "c" * 64,
        },
        {
            "speaker": SpeakerRole.SPEAKER_A,
            "text": "inside",
            "type": "word",
            "start": 10.4,
            "end": 10.8,
            "source_transcript_sha256": "c" * 64,
        },
        {
            "speaker": SpeakerRole.SPEAKER_A,
            "text": "after",
            "type": "word",
            "start": 40.0,
            "end": 40.2,
            "source_transcript_sha256": "c" * 64,
        },
    ]

    clipped = clip_words_to_window(words, window_start_seconds=10.0, window_end_seconds=40.0)

    assert [word.text for word in clipped] == ["edge", "inside"]
    assert clipped[0].source_start_seconds == 10.0
    assert clipped[0].window_start_seconds == 0.0
    assert clipped[0].window_end_seconds == pytest.approx(0.2)
    assert clipped[1].window_start_seconds == pytest.approx(0.4)


def test_decision_states_are_tagged_and_reject_unknown_fields() -> None:
    assert parse_decision({"kind": "pending"}) == Pending()
    corrected = parse_decision(
        {"kind": "corrected", "activity": [{"speaker": "speaker_a", "start_frame": 0, "end_frame": 4}]}
    )
    assert isinstance(corrected, Corrected)
    follow = parse_decision({"kind": "needs_follow_up", "reason": "crosstalk", "scope": {"kind": "whole_window"}})
    assert isinstance(follow, NeedsFollowUp)
    with pytest.raises(ContractError, match="unknown fields"):
        parse_decision({"kind": "pending", "ok": True})
    with pytest.raises(ContractError, match="unknown"):
        parse_decision({"kind": "confirmed_proposal", "accepted": True})


def test_confirm_does_not_mark_defects_clear() -> None:
    state = apply_action(WindowReviewState(), ConfirmAction(), actor="reviewer-1", event_hash="d" * 64)

    assert state.decision == ConfirmedProposal()
    assert isinstance(state.defects.identity, NotReviewed)
    assert isinstance(state.defects.clock, ClockNotReviewed)
    assert state.defects.has_unresolved() is True


def test_legal_and_illegal_review_transitions() -> None:
    pending = WindowReviewState()
    confirmed = apply_action(pending, ConfirmAction(), actor="rev", event_hash="1" * 64)
    corrected = apply_action(
        pending,
        CorrectAction((_interval("speaker_a", 0, 3),)),
        actor="rev",
        event_hash="2" * 64,
    )
    silent = apply_action(pending, NoSpeechAction(), actor="rev", event_hash="3" * 64)
    follow = apply_action(
        pending,
        NeedsFollowUpAction("need check", WholeWindowScope()),
        actor="rev",
        event_hash="4" * 64,
    )
    uncertain = apply_action(
        pending,
        UncertainAction("hard", BoundedRangeScope(((10, 20),))),
        actor="rev",
        event_hash="5" * 64,
    )
    signed = apply_action(confirmed, SignOffAcceptAction(), actor="signer", event_hash="6" * 64)
    returned = apply_action(corrected, SignOffReturnAction("fix overlap"), actor="signer", event_hash="7" * 64)

    assert isinstance(confirmed.decision, ConfirmedProposal)
    assert isinstance(corrected.decision, Corrected)
    assert isinstance(silent.decision, NoSpeech)
    assert isinstance(follow.decision, NeedsFollowUp)
    assert isinstance(uncertain.decision, Uncertain)
    assert isinstance(signed.decision, SignedOff)
    assert isinstance(returned.decision, Returned)
    with pytest.raises(ContractError, match="different actor"):
        apply_action(confirmed, SignOffAcceptAction(), actor="rev", event_hash="8" * 64)
    with pytest.raises(ContractError, match="confirmed, corrected, or no-speech"):
        apply_action(uncertain, SignOffAcceptAction(), actor="signer", event_hash="9" * 64)
    with pytest.raises(ContractError, match="cannot be edited"):
        apply_action(signed, ConfirmAction(), actor="rev", event_hash="a" * 64)
    with pytest.raises(ContractError, match="undo must be applied"):
        apply_action(
            confirmed,
            parse_action({"kind": "undo", "reverted_event_hash": "1" * 64}),
            actor="rev",
            event_hash="b" * 64,
        )


def test_clock_measurement_is_present_only_when_entered() -> None:
    defects = DefectSet(
        identity=Clear(),
        clock=ClockClear(ClockMeasurementEntered(0.02, 0.0)),
        synchronization=Clear(),
        redaction=UnresolvedDefect("possible cut"),
    )
    payload = defects.to_dict()
    assert payload["clock"]["measurement"]["kind"] == "entered"
    assert DefectSet.from_dict(payload).redaction.reason == "possible cut"
    with pytest.raises(ContractError, match="unknown fields"):
        DefectSet.from_dict({**payload, "extra": 1})


def test_transcript_word_identity_is_bound() -> None:
    word = TranscriptWord(
        speaker=SpeakerRole.SPEAKER_B,
        text="yes",
        word_type="word",
        source_start_seconds=12.0,
        source_end_seconds=12.2,
        window_start_seconds=2.0,
        window_end_seconds=2.2,
        source_transcript_sha256="e" * 64,
    )
    assert word.to_dict()["source_transcript_sha256"] == "e" * 64
    assert sha256_json(word.to_dict())
