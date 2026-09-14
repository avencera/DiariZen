"""Protect immutable review events, replay, undo, and sign-off."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipes.speakrs.large.errors import ContractError
from recipes.speakrs.large.review_models import (
    ActivityInterval,
    Clear,
    ClockClear,
    ClockMeasurementAbsent,
    ConfirmAction,
    CorrectAction,
    DefectSet,
    NoSpeechAction,
    SetDefectsAction,
    SignOffAcceptAction,
    SpeakerRole,
    UncertainAction,
    UndoAction,
    WholeWindowScope,
)
from recipes.speakrs.large.review_overlay import prepare_review_overlay
from recipes.speakrs.large.review_store import open_review_store
from recipes.speakrs.tests.test_review_overlay import _build_packet


def _session(tmp_path: Path) -> Path:
    archive, packet = _build_packet(tmp_path)
    destination = tmp_path / "session"
    prepare_review_overlay(packet, archive, destination)
    return destination


def _window_id(session: Path) -> str:
    store = open_review_store(session)
    return store.overlay.window_ids()[0]


def _clear_defects() -> DefectSet:
    return DefectSet(
        identity=Clear(),
        clock=ClockClear(ClockMeasurementAbsent()),
        synchronization=Clear(),
        redaction=Clear(),
    )


def test_append_replay_duplicate_stale_undo_and_restart(tmp_path: Path) -> None:
    session = _session(tmp_path)
    window_id = _window_id(session)
    store = open_review_store(session)
    first = store.append(
        window_id=window_id,
        actor="rev-1",
        request_id="req-confirm",
        base_revision=0,
        action=ConfirmAction(),
        timestamp_utc="2026-09-14T00:00:00Z",
    )
    duplicate = store.append(
        window_id=window_id,
        actor="rev-1",
        request_id="req-confirm",
        base_revision=0,
        action=ConfirmAction(),
        timestamp_utc="2026-09-14T00:00:00Z",
    )
    assert duplicate.event_hash == first.event_hash
    assert len(store.events()) == 1
    with pytest.raises(ContractError, match="reused with different content"):
        store.append(
            window_id=window_id,
            actor="rev-1",
            request_id="req-confirm",
            base_revision=0,
            action=NoSpeechAction(),
        )
    with pytest.raises(ContractError, match="stale revision"):
        store.append(
            window_id=window_id,
            actor="rev-1",
            request_id="req-stale",
            base_revision=0,
            action=NoSpeechAction(),
        )
    undone = store.append(
        window_id=window_id,
        actor="rev-1",
        request_id="req-undo",
        base_revision=1,
        action=UndoAction(first.event_hash),
    )
    assert store.window_state(window_id).decision.kind == "pending"
    assert undone.action.kind == "undo"
    restarted = open_review_store(session)
    assert [event.event_hash for event in restarted.events()] == [first.event_hash, undone.event_hash]
    assert restarted.window_state(window_id).decision.kind == "pending"
    assert restarted.progress().as_dict()["reviewed_count"] == 0


def test_crash_residue_is_quarantined_and_ignored(tmp_path: Path) -> None:
    session = _session(tmp_path)
    residue = session / "events" / "event-00000001-deadbeef.json.partial"
    residue.write_text("{", encoding="utf-8")
    store = open_review_store(session)
    assert residue.name in store.quarantined
    assert store.events() == ()


def test_same_actor_cannot_sign_off_and_return_allows_correction(tmp_path: Path) -> None:
    session = _session(tmp_path)
    window_id = _window_id(session)
    store = open_review_store(session)
    store.append(window_id=window_id, actor="rev-1", request_id="c1", base_revision=0, action=ConfirmAction())
    with pytest.raises(ContractError, match="different actor"):
        store.append(
            window_id=window_id,
            actor="rev-1",
            request_id="s1",
            base_revision=1,
            action=SignOffAcceptAction(),
        )
    store.append(
        window_id=window_id,
        actor="signer",
        request_id="ret",
        base_revision=1,
        action={"kind": "sign_off_return", "reason": "overlap is wrong"},
    )
    assert store.window_state(window_id).decision.kind == "returned"
    store.append(
        window_id=window_id,
        actor="rev-1",
        request_id="fix",
        base_revision=2,
        action=CorrectAction((ActivityInterval(SpeakerRole.SPEAKER_A, 0, 8),)),
    )
    store.append(
        window_id=window_id,
        actor="rev-1",
        request_id="def",
        base_revision=3,
        action=SetDefectsAction(_clear_defects()),
    )
    signed = store.append(
        window_id=window_id,
        actor="signer",
        request_id="ok",
        base_revision=4,
        action=SignOffAcceptAction(),
    )
    state = store.window_state(window_id)
    assert state.decision.kind == "signed_off"
    assert state.decision.signer == "signer"
    assert signed.actor == "signer"


def test_uncertain_windows_are_not_signoff_eligible(tmp_path: Path) -> None:
    session = _session(tmp_path)
    window_id = _window_id(session)
    store = open_review_store(session)
    store.append(
        window_id=window_id,
        actor="rev-1",
        request_id="u1",
        base_revision=0,
        action=UncertainAction("hard", WholeWindowScope()),
    )
    with pytest.raises(ContractError, match="confirmed, corrected, or no-speech"):
        store.append(
            window_id=window_id,
            actor="signer",
            request_id="s1",
            base_revision=1,
            action=SignOffAcceptAction(),
        )
