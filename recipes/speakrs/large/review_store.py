"""Immutable hash-chained review events for one Open Yap session."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .errors import ContractError, PreparationError
from .hashing import canonical_json, sha256_json
from .jsonio import atomic_write_text, read_json
from .review_models import (
    GENESIS_HASH,
    ActivityInterval,
    GridProposal,
    NoSpeech,
    ReviewAction,
    SignedOff,
    UndoAction,
    WindowReviewState,
    apply_action,
    decision_activity,
    parse_action,
    parse_non_empty_text,
    parse_sha256,
    reject_unknown_fields,
    require_fields,
    require_mapping,
)
from .review_overlay import ReviewOverlay, load_review_session


EVENT_SCHEMA = "speakrs-open-yap-review-event"
EVENT_SCHEMA_VERSION = 1
EVENT_NAME = "event-{sequence:08d}-{event_hash}.json"
TEMPORARY_SUFFIXES = (".partial", ".tmp", ".temp")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class ReviewEvent:
    """One accepted immutable review event."""

    sequence: int
    event_hash: str
    prior_hash: str
    packet_hash: str
    overlay_hash: str
    window_id: str
    actor: str
    base_revision: int
    request_id: str
    timestamp_utc: str
    action: ReviewAction

    def body(self) -> dict[str, object]:
        return {
            "schema": EVENT_SCHEMA,
            "schema_version": EVENT_SCHEMA_VERSION,
            "sequence": self.sequence,
            "prior_hash": self.prior_hash,
            "packet_hash": self.packet_hash,
            "overlay_hash": self.overlay_hash,
            "window_id": self.window_id,
            "actor": self.actor,
            "base_revision": self.base_revision,
            "request_id": self.request_id,
            "timestamp_utc": self.timestamp_utc,
            "action": self.action.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.body()
        payload["event_hash"] = self.event_hash
        return payload

    @classmethod
    def from_dict(cls, value: object) -> ReviewEvent:
        record = require_mapping(value, "review event")
        reject_unknown_fields(
            record,
            frozenset(
                {
                    "schema",
                    "schema_version",
                    "sequence",
                    "event_hash",
                    "prior_hash",
                    "packet_hash",
                    "overlay_hash",
                    "window_id",
                    "actor",
                    "base_revision",
                    "request_id",
                    "timestamp_utc",
                    "action",
                }
            ),
            "review event",
        )
        require_fields(
            record,
            frozenset(
                {
                    "schema",
                    "schema_version",
                    "sequence",
                    "event_hash",
                    "prior_hash",
                    "packet_hash",
                    "overlay_hash",
                    "window_id",
                    "actor",
                    "base_revision",
                    "request_id",
                    "timestamp_utc",
                    "action",
                }
            ),
            "review event",
        )
        if record.get("schema") != EVENT_SCHEMA or record.get("schema_version") != EVENT_SCHEMA_VERSION:
            raise ContractError("review event schema is invalid")
        body = {key: value for key, value in record.items() if key != "event_hash"}
        event_hash = parse_sha256(record["event_hash"], "event_hash")
        if sha256_json(body) != event_hash:
            raise ContractError("review event hash does not match its body")
        sequence = record["sequence"]
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
            raise ContractError("review event sequence must be a positive integer")
        base_revision = record["base_revision"]
        if not isinstance(base_revision, int) or isinstance(base_revision, bool) or base_revision < 0:
            raise ContractError("review event base_revision must be a non-negative integer")
        return cls(
            sequence=sequence,
            event_hash=event_hash,
            prior_hash=parse_sha256(record["prior_hash"], "prior_hash"),
            packet_hash=parse_sha256(record["packet_hash"], "packet_hash"),
            overlay_hash=parse_sha256(record["overlay_hash"], "overlay_hash"),
            window_id=parse_non_empty_text(record["window_id"], "window_id"),
            actor=parse_non_empty_text(record["actor"], "actor"),
            base_revision=base_revision,
            request_id=parse_non_empty_text(record["request_id"], "request_id"),
            timestamp_utc=parse_non_empty_text(record["timestamp_utc"], "timestamp_utc"),
            action=parse_action(record["action"]),
        )


@dataclass
class _WindowHistory:
    states: list[tuple[str, WindowReviewState]] = field(default_factory=list)

    def current(self) -> WindowReviewState:
        if not self.states:
            return WindowReviewState()
        return self.states[-1][1]

    def state_before(self, event_hash: str) -> WindowReviewState:
        if not self.states or self.states[-1][0] != event_hash:
            raise ContractError("undo must name the latest event for this window")
        if len(self.states) == 1:
            return WindowReviewState()
        return self.states[-2][1]


@dataclass
class ReviewProgress:
    window_ids: tuple[str, ...]
    states: dict[str, WindowReviewState]
    uniform_ids: tuple[str, ...]
    targeted_ids: tuple[str, ...]

    def _finished(self, window_id: str) -> bool:
        decision = self.states[window_id].decision
        return decision.kind in {"signed_off"}

    def _reviewed(self, window_id: str) -> bool:
        kind = self.states[window_id].decision.kind
        return kind not in {"pending"}

    def as_dict(self) -> dict[str, object]:
        next_window = None
        for window_id in self.window_ids:
            if self.states[window_id].decision.kind == "pending":
                next_window = window_id
                break
        if next_window is None:
            for window_id in self.window_ids:
                if self.states[window_id].decision.kind == "returned":
                    next_window = window_id
                    break
        if next_window is None:
            for window_id in self.window_ids:
                if not self._finished(window_id):
                    next_window = window_id
                    break
        return {
            "window_count": len(self.window_ids),
            "reviewed_count": sum(self._reviewed(window_id) for window_id in self.window_ids),
            "signed_count": sum(self._finished(window_id) for window_id in self.window_ids),
            "uniform": {
                "total": len(self.uniform_ids),
                "reviewed": sum(self._reviewed(window_id) for window_id in self.uniform_ids),
                "signed": sum(self._finished(window_id) for window_id in self.uniform_ids),
            },
            "targeted": {
                "total": len(self.targeted_ids),
                "reviewed": sum(self._reviewed(window_id) for window_id in self.targeted_ids),
                "signed": sum(self._finished(window_id) for window_id in self.targeted_ids),
            },
            "next_window_id": next_window,
        }


class ReviewStore:
    """Serialize event creation for one review session."""

    def __init__(self, session_root: Path) -> None:
        loaded = load_review_session(session_root)
        self.root = loaded["root"]
        self.session = loaded["session"]
        self.overlay: ReviewOverlay = loaded["overlay"]
        self.packet_root = loaded["packet_root"]
        self.packet_hash = str(self.session["packet_manifest_sha256"])
        self.overlay_hash = str(self.session["overlay_sha256"])
        self.events_dir = self.root / str(self.session["event_store_path"])
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events: list[ReviewEvent] = []
        self._by_request: dict[str, ReviewEvent] = {}
        self._history: dict[str, _WindowHistory] = {
            window_id: _WindowHistory() for window_id in self.overlay.window_ids()
        }
        self.quarantined: list[str] = []
        self._load_existing()

    def _load_existing(self) -> None:
        quarantined: list[str] = []
        loaded: list[ReviewEvent] = []
        for path in sorted(self.events_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                continue
            if path.name.startswith(".") or path.name.endswith(TEMPORARY_SUFFIXES):
                quarantined.append(path.name)
                continue
            try:
                loaded.append(ReviewEvent.from_dict(read_json(path)))
            except (ContractError, OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                raise PreparationError("review event file is corrupt", {"path": path.as_posix()}) from error
        loaded.sort(key=lambda item: item.sequence)
        sequences = [item.sequence for item in loaded]
        if sequences != list(range(1, len(sequences) + 1)):
            raise ContractError("review event sequences are not contiguous", {"sequences": sequences})
        prior = GENESIS_HASH
        for event in loaded:
            if event.prior_hash != prior:
                raise ContractError("review event chain is broken", {"sequence": event.sequence})
            if event.packet_hash != self.packet_hash or event.overlay_hash != self.overlay_hash:
                raise ContractError("review event is bound to a different packet or overlay")
            if event.window_id not in self._history:
                raise ContractError("review event window is not in the overlay", {"window_id": event.window_id})
            self._apply_loaded(event)
            prior = event.event_hash
        self.quarantined = quarantined

    def _apply_loaded(self, event: ReviewEvent) -> None:
        history = self._history[event.window_id]
        current = history.current()
        if event.base_revision != current.revision:
            raise ContractError("stored event base_revision does not replay")
        if isinstance(event.action, UndoAction):
            restored = history.state_before(event.action.reverted_event_hash)
            next_state = WindowReviewState(
                decision=restored.decision,
                defects=restored.defects,
                last_review_actor=restored.last_review_actor,
                last_review_event_hash=restored.last_review_event_hash,
                revision=current.revision + 1,
            )
        else:
            next_state = apply_action(current, event.action, actor=event.actor, event_hash=event.event_hash)
        history.states.append((event.event_hash, next_state))
        self._events.append(event)
        self._by_request[event.request_id] = event

    def events(self) -> tuple[ReviewEvent, ...]:
        return tuple(self._events)

    def signed_activity(self, window_id: str, proposal: GridProposal) -> tuple[ActivityInterval, ...]:
        """Return the activity that an independent signer accepted."""

        history = self._history[window_id]
        current = history.current()
        if not isinstance(current.decision, SignedOff):
            raise ContractError("window is not signed off", {"window_id": window_id})
        if len(history.states) < 2:
            raise ContractError("signed window is missing the reviewed activity event")
        previous = history.states[-2][1]
        activity = decision_activity(previous.decision, proposal)
        if activity is None:
            raise ContractError("signed window activity is not exportable")
        return activity

    def signed_disposition(self, window_id: str) -> str:
        """Return speech or no_speech from the signed activity decision."""

        history = self._history[window_id]
        current = history.current()
        if not isinstance(current.decision, SignedOff) or len(history.states) < 2:
            raise ContractError("window is not signed off", {"window_id": window_id})
        previous = history.states[-2][1]
        if isinstance(previous.decision, NoSpeech):
            return "no_speech"
        if previous.is_signoff_eligible_activity():
            return "speech"
        raise ContractError("signed window disposition is not exportable")

    def window_state(self, window_id: str) -> WindowReviewState:
        if window_id not in self._history:
            raise ContractError("unknown window", {"window_id": window_id})
        return self._history[window_id].current()

    def progress(self) -> ReviewProgress:
        uniform = []
        targeted = []
        states = {}
        for record in self.overlay.manifest["windows"]:
            window_id = str(record["window_id"])
            states[window_id] = self.window_state(window_id)
            if record.get("selection_kind") == "uniform":
                uniform.append(window_id)
            else:
                targeted.append(window_id)
        return ReviewProgress(
            window_ids=self.overlay.window_ids(),
            states=states,
            uniform_ids=tuple(uniform),
            targeted_ids=tuple(targeted),
        )

    def append(
        self,
        *,
        window_id: str,
        actor: str,
        request_id: str,
        base_revision: int,
        action: ReviewAction | Mapping[str, Any],
        timestamp_utc: str | None = None,
    ) -> ReviewEvent:
        """Create one event or return the prior result for an identical request."""

        parsed = action if isinstance(action, ReviewAction) else parse_action(action)
        with self._lock:
            return self._append_locked(
                window_id=window_id,
                actor=actor,
                request_id=request_id,
                base_revision=base_revision,
                action=parsed,
                timestamp_utc=timestamp_utc,
            )

    def _append_locked(
        self,
        *,
        window_id: str,
        actor: str,
        request_id: str,
        base_revision: int,
        action: ReviewAction,
        timestamp_utc: str | None,
    ) -> ReviewEvent:
        if window_id not in self._history:
            raise ContractError("unknown window", {"window_id": window_id})
        request = parse_non_empty_text(request_id, "request_id")
        actor_id = parse_non_empty_text(actor, "actor")
        if not isinstance(base_revision, int) or isinstance(base_revision, bool) or base_revision < 0:
            raise ContractError("base_revision must be a non-negative integer")
        existing = self._by_request.get(request)
        if existing is not None:
            candidate = {
                "window_id": window_id,
                "actor": actor_id,
                "base_revision": base_revision,
                "action": action.to_dict(),
            }
            prior = {
                "window_id": existing.window_id,
                "actor": existing.actor,
                "base_revision": existing.base_revision,
                "action": existing.action.to_dict(),
            }
            if candidate != prior:
                raise ContractError("request_id was reused with different content")
            return existing
        history = self._history[window_id]
        current = history.current()
        if base_revision != current.revision:
            raise ContractError(
                "stale revision",
                {"base_revision": base_revision, "current_revision": current.revision},
            )
        prior_hash = self._events[-1].event_hash if self._events else GENESIS_HASH
        sequence = len(self._events) + 1
        stamp = timestamp_utc or _utc_now()
        body = {
            "schema": EVENT_SCHEMA,
            "schema_version": EVENT_SCHEMA_VERSION,
            "sequence": sequence,
            "prior_hash": prior_hash,
            "packet_hash": self.packet_hash,
            "overlay_hash": self.overlay_hash,
            "window_id": window_id,
            "actor": actor_id,
            "base_revision": base_revision,
            "request_id": request,
            "timestamp_utc": stamp,
            "action": action.to_dict(),
        }
        event_hash = sha256_json(body)
        event = ReviewEvent.from_dict({**body, "event_hash": event_hash})
        if isinstance(action, UndoAction):
            restored = history.state_before(action.reverted_event_hash)
            next_state = WindowReviewState(
                decision=restored.decision,
                defects=restored.defects,
                last_review_actor=restored.last_review_actor,
                last_review_event_hash=restored.last_review_event_hash,
                revision=current.revision + 1,
            )
        else:
            next_state = apply_action(current, action, actor=actor_id, event_hash=event_hash)
        filename = EVENT_NAME.format(sequence=sequence, event_hash=event_hash)
        destination = self.events_dir / filename
        temporary = destination.with_name(destination.name + ".partial")
        atomic_write_text(temporary, canonical_json(event.to_dict()))
        temporary.replace(destination)
        _fsync_directory(self.events_dir)
        history.states.append((event_hash, next_state))
        self._events.append(event)
        self._by_request[request] = event
        return event


def open_review_store(session_root: Path) -> ReviewStore:
    """Open a session event store and replay durable events."""

    return ReviewStore(session_root)


__all__ = ["EVENT_SCHEMA", "ReviewEvent", "ReviewProgress", "ReviewStore", "open_review_store"]
