"""Protect export eligibility, QA measurement, and completed-review validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes.speakrs.large.errors import ContractError, PreparationError
from recipes.speakrs.large.review_export import export_review, validate_completed_review
from recipes.speakrs.large.review_models import (
    Clear,
    ClockClear,
    ClockMeasurementAbsent,
    ConfirmAction,
    DefectSet,
    NoSpeechAction,
    SetDefectsAction,
    SignOffAcceptAction,
    UncertainAction,
    WholeWindowScope,
)
from recipes.speakrs.large.review_overlay import prepare_review_overlay
from recipes.speakrs.large.review_packet import validate_review_packet
from recipes.speakrs.large.review_store import open_review_store
from recipes.speakrs.tests.test_review_overlay import _build_packet, _packet_files


def _session(tmp_path: Path) -> tuple[Path, Path]:
    archive, packet = _build_packet(tmp_path)
    destination = tmp_path / "session"
    prepare_review_overlay(packet, archive, destination)
    return destination, packet


def _clear() -> DefectSet:
    return DefectSet(
        identity=Clear(),
        clock=ClockClear(ClockMeasurementAbsent()),
        synchronization=Clear(),
        redaction=Clear(),
    )


def _sign(store, window_id: str, actor: str, signer: str, request_prefix: str, revision: int, action=None) -> int:
    store.append(
        window_id=window_id,
        actor=actor,
        request_id=f"{request_prefix}-act",
        base_revision=revision,
        action=action or ConfirmAction(),
    )
    store.append(
        window_id=window_id,
        actor=actor,
        request_id=f"{request_prefix}-def",
        base_revision=revision + 1,
        action=SetDefectsAction(_clear()),
    )
    store.append(
        window_id=window_id,
        actor=signer,
        request_id=f"{request_prefix}-sign",
        base_revision=revision + 2,
        action=SignOffAcceptAction(),
    )
    return revision + 3


def test_export_excludes_uncertain_returned_and_unsigned(tmp_path: Path) -> None:
    session, packet = _session(tmp_path)
    before = _packet_files(packet)
    store = open_review_store(session)
    windows = store.overlay.window_ids()
    _sign(store, windows[0], "rev-1", "signer", "w0", 0)
    store.append(
        window_id=windows[1],
        actor="rev-1",
        request_id="u1",
        base_revision=0,
        action=UncertainAction("not sure", WholeWindowScope()),
    )
    store.append(window_id=windows[2], actor="rev-1", request_id="c2", base_revision=0, action=ConfirmAction())
    store.append(
        window_id=windows[2],
        actor="signer",
        request_id="ret",
        base_revision=1,
        action={"kind": "sign_off_return", "reason": "check overlap"},
    )
    result = export_review(store)
    assert result["exported_window_count"] == 1
    assert result["excluded_window_count"] == 3
    export_root = Path(result["export_path"])
    exported = json.loads((export_root / "manifest.json").read_text(encoding="utf-8"))
    assert list(exported["exported_windows"]) == [windows[0]]
    annotation = json.loads((export_root / exported["exported_windows"][windows[0]]).read_text(encoding="utf-8"))
    assert annotation["reviewer_id"] == "rev-1"
    assert annotation["independent_signoff"]["signer_id"] == "signer"
    assert annotation["window_disposition"] in {"speech", "no_speech"}
    assert (packet / "windows" / windows[0] / "human-annotation.json").read_text(encoding="utf-8")
    blank = json.loads((packet / "windows" / windows[0] / "human-annotation.json").read_text(encoding="utf-8"))
    assert blank["speaker_activity"] is None
    assert _packet_files(packet) == before
    assert validate_review_packet(packet)["ok"] is True
    qa = json.loads((export_root / "qa-report.json").read_text(encoding="utf-8"))
    assert "human_activity" in qa["overall"]
    assert qa["overall"]["status"] in {"measured", "unavailable"}


def test_no_speech_export_uses_empty_activity(tmp_path: Path) -> None:
    session, _packet = _session(tmp_path)
    store = open_review_store(session)
    window_id = store.overlay.window_ids()[0]
    _sign(store, window_id, "rev-1", "signer", "ns", 0, action=NoSpeechAction())
    result = export_review(store)
    export_root = Path(result["export_path"])
    manifest = json.loads((export_root / "manifest.json").read_text(encoding="utf-8"))
    annotation = json.loads((export_root / manifest["exported_windows"][window_id]).read_text(encoding="utf-8"))
    assert annotation["window_disposition"] == "no_speech"
    assert annotation["speaker_activity"] == []


def test_corrupt_chain_and_same_actor_are_rejected(tmp_path: Path) -> None:
    session, _packet = _session(tmp_path)
    store = open_review_store(session)
    window_id = store.overlay.window_ids()[0]
    store.append(window_id=window_id, actor="rev-1", request_id="c1", base_revision=0, action=ConfirmAction())
    with pytest.raises(ContractError, match="different actor"):
        store.append(
            window_id=window_id, actor="rev-1", request_id="self", base_revision=1, action=SignOffAcceptAction()
        )
    event_path = next((session / "events").glob("event-*.json"))
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["actor"] = "tampered"
    event_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ContractError, PreparationError), match="hash does not match|corrupt"):
        open_review_store(session)


def test_completed_review_validator_recomputes_export(tmp_path: Path) -> None:
    session, _packet = _session(tmp_path)
    store = open_review_store(session)
    window_id = store.overlay.window_ids()[0]
    _sign(store, window_id, "rev-1", "signer", "ok", 0)
    exported = export_review(store)
    report = validate_completed_review(session, Path(exported["export_path"]))
    assert report["ok"] is True
    assert report["eligible_window_count"] == 1
    stale = Path(exported["export_path"]) / "manifest.json"
    manifest = json.loads(stale.read_text(encoding="utf-8"))
    manifest["event_count"] = 0
    stale.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="stale|identity hash"):
        validate_completed_review(session, Path(exported["export_path"]))
