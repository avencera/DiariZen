"""Production trusted-controller entry for Cloudeck validation results.

The trainer and supervisor restore both digest-bound stores, import exact
exported result bytes, persist after every transition, and drive drain plus
retention through typed ports. Tests inject fake ports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .contracts import (
    VALIDATION_RESULT_SCHEMA,
    PublishedSlotState,
    PublishedSnapshot,
    Sha256Digest,
    TrainingCompletion,
    ValidatedSlotState,
    ValidationCampaign,
    ValidationContractError,
    training_completion_from_dict,
)
from .controller import (
    CampaignCompletionController,
    CampaignCompletionError,
    CampaignCompletionState,
    CampaignCompletionStateStore,
    CampaignLifecycle,
    ControllerError,
    QueueDrainState,
    RankingRetentionCommit,
    ResultImportError,
    ResultImportFailureKind,
    SelectionController,
    SelectionPolicy,
    TrustedStateError,
    TrustedStateStore,
    ValidationSlotFailureState,
    initial_selection_state,
)


class TrustedImporterError(ControllerError):
    """A production trusted-controller operation cannot proceed."""


class QueueDrainPort(Protocol):
    """Cancel unused queue work and prove pool drain."""

    def request_cancel(self) -> None:
        """Request cancellation of pending and leased unused units."""

    def is_drained(self) -> bool:
        """Return whether the queue and managed pool are empty."""


class RankingRetentionPort(Protocol):
    """Commit retained snapshot files for one ranking revision."""

    def commit(self, receipt: RankingRetentionCommit) -> None:
        """Retain the exact current top-five snapshot files."""


@dataclass(frozen=True)
class FilesystemQueueDrainPort:
    """Record queue cancellation and observe a drain marker file."""

    cancel_path: Path
    drained_path: Path

    def request_cancel(self) -> None:
        """Write the durable cancel request marker."""

        self.cancel_path.parent.mkdir(parents=True, exist_ok=True)
        self.cancel_path.write_text("cancel_requested\n", encoding="utf-8")

    def is_drained(self) -> bool:
        """Return whether the drain marker file exists."""

        return self.drained_path.is_file()


@dataclass(frozen=True)
class FilesystemRankingRetentionPort:
    """Write one canonical ranking-retention receipt."""

    receipt_path: Path

    def commit(self, receipt: RankingRetentionCommit) -> None:
        """Persist the exact ranking receipt next to campaign state."""

        if not isinstance(receipt, RankingRetentionCommit):
            raise TrustedImporterError("retention port requires a typed ranking receipt")
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.receipt_path.write_text(encoded + "\n", encoding="utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("JSON object contains duplicate fields")
        parsed[key] = value
    return parsed


def _load_json_bytes(payload: bytes, context: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"JSON constant is not supported: {value}")),
        )
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ResultImportError(ResultImportFailureKind.INVALID_DOCUMENT, f"{context} is not valid JSON") from error


def _exact_fields(value: Mapping[str, object], required: frozenset[str], context: str) -> None:
    actual = frozenset(value)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        raise ResultImportError(
            ResultImportFailureKind.INVALID_DOCUMENT,
            f"{context} fields are not exact: missing={sorted(missing)}, extra={sorted(extra)}",
        )


def _compact_result_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def parse_exported_cloudeck_result_bytes(payload: bytes) -> tuple[dict[str, object], ...]:
    """Parse exact retained Cloudeck result bytes into compact result documents."""

    document = _load_json_bytes(payload, "exported Cloudeck result")
    if not isinstance(document, Mapping):
        raise ResultImportError(ResultImportFailureKind.INVALID_DOCUMENT, "exported Cloudeck result must be an object")
    if document.get("schema") == VALIDATION_RESULT_SCHEMA:
        return (dict(document),)
    if document.get("result") == "unit_results":
        return _parse_unit_results_envelope(document)
    raise ResultImportError(
        ResultImportFailureKind.INVALID_DOCUMENT,
        "exported Cloudeck result is not a validation result or unit-results envelope",
    )


def _parse_unit_results_envelope(document: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    _exact_fields(document, frozenset({"version", "result", "payload"}), "Cloudeck unit-results envelope")
    if document["version"] != 3:
        raise ResultImportError(ResultImportFailureKind.INVALID_DOCUMENT, "Cloudeck unit-results envelope must be v3")
    payload = document["payload"]
    if not isinstance(payload, Mapping):
        raise ResultImportError(
            ResultImportFailureKind.INVALID_DOCUMENT, "Cloudeck unit-results payload must be an object"
        )
    _exact_fields(payload, frozenset({"queue_id", "units", "next_after_ordinal"}), "Cloudeck unit-results payload")
    units = payload["units"]
    if not isinstance(units, list) or not units:
        raise ResultImportError(
            ResultImportFailureKind.INVALID_DOCUMENT, "Cloudeck unit-results page must contain units"
        )
    results: list[dict[str, object]] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping):
            raise ResultImportError(
                ResultImportFailureKind.INVALID_DOCUMENT,
                f"Cloudeck completed unit {index} must be an object",
            )
        _exact_fields(unit, frozenset({"unit_id", "ordinal", "result_digest", "result"}), "Cloudeck completed unit")
        body = unit["result"]
        if body is None:
            raise ResultImportError(
                ResultImportFailureKind.INVALID_DOCUMENT,
                "Cloudeck completed unit has no retained result body",
            )
        if not isinstance(body, Mapping):
            raise ResultImportError(
                ResultImportFailureKind.INVALID_DOCUMENT,
                "Cloudeck retained result body must be an object",
            )
        digest = unit["result_digest"]
        if not isinstance(digest, str) or hashlib.sha256(_compact_result_bytes(body)).hexdigest() != digest:
            raise ResultImportError(
                ResultImportFailureKind.INVALID_DOCUMENT,
                "Cloudeck retained result digest does not match its exact JSON body",
            )
        results.append(dict(body))
    return tuple(results)


def _snapshots_from_completion(state: CampaignCompletionState) -> tuple[PublishedSnapshot, ...]:
    snapshots: list[PublishedSnapshot] = []
    for _, slot_state in state.slot_states:
        if isinstance(slot_state, (PublishedSlotState, ValidatedSlotState, ValidationSlotFailureState)):
            snapshots.append(slot_state.snapshot)
    return tuple(snapshots)


class TrustedController:
    """Restore both stores, import results, persist, and drive completion."""

    def __init__(
        self,
        campaign: ValidationCampaign,
        policy: SelectionPolicy,
        selection_store: TrustedStateStore,
        completion_store: CampaignCompletionStateStore,
        *,
        drain: QueueDrainPort,
        retention: RankingRetentionPort,
        completion: CampaignCompletionController,
    ) -> None:
        if not isinstance(campaign, ValidationCampaign):
            raise TrustedImporterError("trusted controller campaign must use its typed contract")
        if not isinstance(policy, SelectionPolicy):
            raise TrustedImporterError("trusted controller policy must use its typed contract")
        if not isinstance(selection_store, TrustedStateStore):
            raise TrustedImporterError("trusted controller requires TrustedStateStore")
        if not isinstance(completion_store, CampaignCompletionStateStore):
            raise TrustedImporterError("trusted controller requires CampaignCompletionStateStore")
        if not isinstance(completion, CampaignCompletionController):
            raise TrustedImporterError("trusted controller requires CampaignCompletionController")
        if selection_store.campaign != campaign or completion_store.campaign != campaign:
            raise TrustedImporterError("trusted stores belong to another campaign")
        self.campaign = campaign
        self.policy = policy
        self.selection_store = selection_store
        self.completion_store = completion_store
        self._drain = drain
        self._retention = retention
        self.completion = completion

    @classmethod
    def restore(
        cls,
        campaign: ValidationCampaign,
        policy: SelectionPolicy,
        selection_store: TrustedStateStore,
        completion_store: CampaignCompletionStateStore,
        *,
        drain: QueueDrainPort,
        retention: RankingRetentionPort,
    ) -> TrustedController:
        """Restore both stores and persist if restoration invalidates stale retention."""

        completion_state = completion_store.load()
        snapshots = _snapshots_from_completion(completion_state)
        selection_update = selection_store.load_update()
        selection = SelectionController(
            campaign,
            policy,
            snapshots,
            selection_update.state,
            stop_request=selection_update.stop_request,
        )
        completion = CampaignCompletionController.restore(campaign, policy, completion_store, selection)
        controller = cls(
            campaign,
            policy,
            selection_store,
            completion_store,
            drain=drain,
            retention=retention,
            completion=completion,
        )
        controller.persist()
        return controller

    @property
    def selection(self) -> SelectionController:
        """Return the restored selection owner."""

        return self.completion.selection

    def persist(self) -> None:
        """Persist selection and campaign-completion state after one transition."""

        current = self.selection_store.load()
        if current.revision != self.selection.state.revision:
            self.selection.persist(self.selection_store)
        self.completion.persist(self.completion_store)

    def import_exported_result(self, payload: bytes) -> CampaignCompletionState:
        """Parse exact exported result bytes, import each document, and persist."""

        documents = parse_exported_cloudeck_result_bytes(payload)
        state = self.completion.state
        for document in documents:
            state = self.completion.import_result(document)
            self.persist()
        return state

    def publish_snapshot(self, snapshot: PublishedSnapshot) -> CampaignCompletionState:
        """Register one published snapshot and persist."""

        state = self.completion.publish_snapshot(snapshot)
        self.persist()
        return state

    def mark_trainer_terminal(self, completion: TrainingCompletion) -> CampaignCompletionState:
        """Record durable trainer terminal progress and persist unused-slot mapping."""

        state = self.completion.mark_trainer_terminal(completion)
        self.persist()
        return state

    def mark_validation_failure(self, slot_id: Sha256Digest | str, detail: str) -> CampaignCompletionState:
        """Record a terminal validation failure and persist."""

        state = self.completion.mark_validation_failure(slot_id, detail)
        self.persist()
        return state

    def advance_completion(self) -> CampaignCompletionState:
        """Drive drain, retention, and completion from durable campaign state."""

        state = self.completion.state
        if state.lifecycle is CampaignLifecycle.FAILED or state.lifecycle is CampaignLifecycle.COMPLETED:
            self.persist()
            return state
        if state.training_completion is None:
            return state
        if any(isinstance(slot_state, PublishedSlotState) for _, slot_state in state.slot_states):
            return state
        if state.retention_commit is None or (
            state.retention_commit.selection_revision != self.selection.state.revision
            or state.retention_commit.snapshot_ids != self.selection.state.top_five_snapshot_ids
        ):
            self.completion.commit_ranking_retention()
            if self.completion.state.retention_commit is not None:
                self._retention.commit(self.completion.state.retention_commit)
            self.persist()
        if self.completion.state.queue_state is QueueDrainState.OPEN:
            self._drain.request_cancel()
            self.completion.cancel_queue()
            self.persist()
        if self.completion.state.queue_state is QueueDrainState.CANCEL_REQUESTED and self._drain.is_drained():
            self.completion.mark_queue_drained()
            self.persist()
        if self.completion.ready_to_complete():
            self.completion.complete()
            self.persist()
        return self.completion.state


def _campaign_from_path(path: Path) -> ValidationCampaign:
    payload = _load_json_bytes(path.read_bytes(), "campaign manifest")
    try:
        return ValidationCampaign.from_dict(payload)
    except (TypeError, ValueError, ValidationContractError) as error:
        raise TrustedImporterError("campaign manifest is invalid") from error


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for the trainer or supervisor to restore, import, and complete."""

    parser = argparse.ArgumentParser(description="Import trusted Cloudeck validation results")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--selection-state", type=Path, required=True)
    parser.add_argument("--completion-state", type=Path, required=True)
    parser.add_argument("--patience-limit", type=int, required=True)
    parser.add_argument("--result-file", type=Path, action="append", default=[])
    parser.add_argument("--cancel-marker", type=Path)
    parser.add_argument("--drain-marker", type=Path)
    parser.add_argument("--retention-receipt", type=Path)
    parser.add_argument("--trainer-completion", type=Path)
    args = parser.parse_args(argv)
    try:
        campaign = _campaign_from_path(args.campaign)
        policy = SelectionPolicy(args.patience_limit)
        selection_store = TrustedStateStore(args.selection_state, campaign, policy, initial_selection_state())
        completion_store = CampaignCompletionStateStore(args.completion_state, campaign)
        cancel_marker = args.cancel_marker or args.completion_state.with_name("queue-cancel")
        drain_marker = args.drain_marker or args.completion_state.with_name("queue-drained")
        retention_receipt = args.retention_receipt or args.completion_state.with_name("retention-receipt.json")
        controller = TrustedController.restore(
            campaign,
            policy,
            selection_store,
            completion_store,
            drain=FilesystemQueueDrainPort(cancel_marker, drain_marker),
            retention=FilesystemRankingRetentionPort(retention_receipt),
        )
        for result_file in args.result_file:
            controller.import_exported_result(result_file.read_bytes())
        if args.trainer_completion is not None:
            completion_document = _load_json_bytes(args.trainer_completion.read_bytes(), "trainer completion")
            controller.mark_trainer_terminal(training_completion_from_dict(completion_document))
        controller.advance_completion()
        return 0 if controller.completion.state.lifecycle is not CampaignLifecycle.FAILED else 1
    except (OSError, TypeError, ValueError, ControllerError, TrustedStateError, CampaignCompletionError) as error:
        print(f"trusted controller failed: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FilesystemQueueDrainPort",
    "FilesystemRankingRetentionPort",
    "QueueDrainPort",
    "RankingRetentionPort",
    "TrustedController",
    "TrustedImporterError",
    "main",
    "parse_exported_cloudeck_result_bytes",
]
