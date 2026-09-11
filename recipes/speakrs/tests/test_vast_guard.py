"""Tests for the trusted Mac-side Vast rental deletion guard."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import pytest

from recipes.speakrs.large.errors import RuntimeGateError
from recipes.speakrs.large.vast_guard import arm_rental_guard, destroy_rental, monitor_rental_guard


class _Response:
    def __init__(self, status: int, payload: object | None = None) -> None:
        self.status = status
        self._body = b"" if payload is None else json.dumps(payload).encode("utf-8")

    def read(self, _limit: int) -> bytes:
        return self._body

    def close(self) -> None:
        return None


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _launch(*, now: float = 100.0, cutoff: float = 200.0) -> dict[str, object]:
    epoch = datetime.fromtimestamp(now, timezone.utc)
    return {
        "attempt_id": "a" * 64,
        "launch_id": "l" * 64,
        "offer": {
            "instance_id": "12345",
            "rented_at": epoch.isoformat(),
            "destroy_at": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
            "rates": {
                "gpu_usd_per_hour": 1.0,
                "disk_usd_per_hour": 0.25,
                "total_usd_per_hour": 1.25,
            },
        },
    }


def _key(tmp_path: Path) -> Path:
    path = tmp_path / "vast-api-key"
    path.write_text("private-api-key", encoding="utf-8")
    return path


def _opener(responses: list[object], requests: list[object]):
    def open_url(request, **_kwargs):
        requests.append(request)
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    return open_url


def _running() -> _Response:
    return _Response(200, {"instances": {"id": 12345, "actual_status": "running", "jupyter_token": "do-not-persist"}})


def _writable() -> _Response:
    return _Response(200, {"success": True})


def test_arm_preflights_exact_running_instance_and_publishes_safe_status(tmp_path: Path) -> None:
    requests: list[object] = []
    key = _key(tmp_path)
    status = tmp_path / "status.json"
    launch = _launch()

    result = arm_rental_guard(
        launch,
        api_key_path=key,
        status_path=status,
        opener=_opener([_running(), _writable()], requests),
        now=_Clock(100.0),
    )

    request = requests[0]
    assert [item.method for item in requests] == ["GET", "PUT"]
    assert request.full_url == "https://console.vast.ai/api/v0/instances/12345/"
    assert request.get_header("Authorization") == "Bearer private-api-key"
    assert requests[1].data == b'{"state":"running"}'
    assert result["state"] == "armed"
    status_text = status.read_text(encoding="utf-8")
    assert json.loads(status_text)["state"] == "armed"
    assert "private-api-key" not in status_text
    assert "do-not-persist" not in status_text


def test_arm_rejects_wrong_instance_without_delete(tmp_path: Path) -> None:
    requests: list[object] = []
    status = tmp_path / "status.json"
    wrong = _Response(200, {"instances": {"id": 54321, "actual_status": "running"}})

    with pytest.raises(RuntimeGateError, match="Vast rental guard failed"):
        arm_rental_guard(
            _launch(),
            api_key_path=_key(tmp_path),
            status_path=status,
            opener=_opener([wrong], requests),
            now=_Clock(100.0),
        )

    assert [request.method for request in requests] == ["GET"]
    assert json.loads(status.read_text(encoding="utf-8"))["error_code"] == "wrong_instance"


def test_arm_rejects_expired_cutoff_without_network_request(tmp_path: Path) -> None:
    requests: list[object] = []
    status = tmp_path / "status.json"

    with pytest.raises(RuntimeGateError, match="cutoff"):
        arm_rental_guard(
            _launch(now=0.0, cutoff=100.0),
            api_key_path=_key(tmp_path),
            status_path=status,
            opener=_opener([], requests),
            now=_Clock(100.0),
        )

    assert requests == []
    assert json.loads(status.read_text(encoding="utf-8"))["state"] == "blocked"


def test_arm_rejects_wrong_auth_without_exposing_key(tmp_path: Path) -> None:
    requests: list[object] = []
    status = tmp_path / "status.json"
    unauthorized = HTTPError(
        "https://console.vast.ai/api/v0/instances/12345/",
        401,
        "unauthorized",
        {},
        io.BytesIO(b"private-api-key"),
    )

    with pytest.raises(RuntimeGateError, match="Vast rental guard failed"):
        arm_rental_guard(
            _launch(),
            api_key_path=_key(tmp_path),
            status_path=status,
            opener=_opener([unauthorized], requests),
            now=_Clock(100.0),
        )

    status_text = status.read_text(encoding="utf-8")
    assert json.loads(status_text)["error_code"] == "authorization"
    assert "private-api-key" not in status_text


def test_destroy_publishes_exact_success_receipt_and_fsync_safe_status(tmp_path: Path) -> None:
    requests: list[object] = []
    status = tmp_path / "status.json"
    receipt_path = tmp_path / "receipt.json"
    launch = _launch(now=100.0, cutoff=200.0)

    receipt = destroy_rental(
        launch,
        api_key_path=_key(tmp_path),
        status_path=status,
        receipt_path=receipt_path,
        opener=_opener([_running(), _Response(200, {"success": True})], requests),
        now=_Clock(100.0),
    )

    assert [request.method for request in requests] == ["GET", "DELETE"]
    assert receipt == json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt) == {
        "schema",
        "attempt_id",
        "launch_id",
        "instance_id",
        "rented_at",
        "ended_at",
        "rates",
        "deletion_outcome",
        "spend_usd",
    }
    assert receipt["schema"] == "speakrs-rental-attempt-receipt-v1"
    assert receipt["deletion_outcome"] == "destroyed"
    assert receipt["rates"] == launch["offer"]["rates"]
    assert receipt["spend_usd"] == pytest.approx(0.0)
    assert json.loads(status.read_text(encoding="utf-8"))["state"] == "destroyed"


def test_destroy_retries_transient_failure_then_succeeds(tmp_path: Path) -> None:
    requests: list[object] = []
    sleeps: list[float] = []
    clock = _Clock(100.0)
    receipt_path = tmp_path / "receipt.json"

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.value += seconds

    receipt = destroy_rental(
        _launch(now=0.0, cutoff=200.0),
        api_key_path=_key(tmp_path),
        status_path=tmp_path / "status.json",
        receipt_path=receipt_path,
        opener=_opener(
            [
                _Response(200, {"instances": {"id": 12345, "actual_status": "running"}}),
                _Response(500),
                _running(),
                _Response(200, {"success": True}),
            ],
            requests,
        ),
        now=clock,
        sleep=sleep,
        retry_seconds=1.0,
    )

    assert receipt["deletion_outcome"] == "destroyed"
    assert [request.method for request in requests] == ["GET", "DELETE", "GET", "DELETE"]
    assert sleeps == [1.0]
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["ended_at"] == "1970-01-01T00:01:41+00:00"
    assert receipt["spend_usd"] == pytest.approx(101.0 * 1.25 / 3600.0)


def test_destroy_retries_after_cutoff_with_bounded_timeout_and_visible_overdue_status(tmp_path: Path) -> None:
    requests: list[object] = []
    sleeps: list[float] = []
    timeouts: list[float] = []
    clock = _Clock(100.0)
    key = _key(tmp_path)
    states: list[tuple[str, bool]] = []
    receipt_path = tmp_path / "receipt.json"

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        states.append((status["state"], status["overdue"]))
        key.write_text("rotated-api-key", encoding="utf-8")
        clock.value += seconds

    def open_url(request, **kwargs):
        requests.append(request)
        timeouts.append(kwargs["timeout"])
        response = [
            _running(),
            _Response(500),
            _running(),
            _Response(200, {"success": True}),
        ][len(requests) - 1]
        return response

    receipt = destroy_rental(
        _launch(now=0.0, cutoff=105.0),
        api_key_path=key,
        status_path=tmp_path / "status.json",
        receipt_path=receipt_path,
        opener=open_url,
        now=clock,
        sleep=sleep,
        retry_seconds=90.0,
        timeout_seconds=120.0,
    )

    assert [request.method for request in requests] == ["GET", "DELETE", "GET", "DELETE"]
    assert sleeps == [60.0]
    assert states == [("retrying", False)]
    assert all(request.get_header("Authorization") == "Bearer private-api-key" for request in requests[:2])
    assert all(request.get_header("Authorization") == "Bearer rotated-api-key" for request in requests[2:])
    assert timeouts == [60.0, 60.0, 60.0, 60.0]
    assert receipt["deletion_outcome"] == "destroyed"
    assert receipt["ended_at"] == "1970-01-01T00:02:40+00:00"
    assert receipt["spend_usd"] == pytest.approx(160.0 * 1.25 / 3600.0)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "destroyed"
    assert status["overdue"] is True


def test_destroy_retries_until_a_missing_key_is_repaired(tmp_path: Path) -> None:
    requests: list[object] = []
    clock = _Clock(100.0)
    key = tmp_path / "vast-api-key"
    errors: list[str | None] = []

    def sleep(seconds: float) -> None:
        errors.append(json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["error_code"])
        key.write_text("repaired-private-key", encoding="utf-8")
        clock.value += seconds

    receipt = destroy_rental(
        _launch(now=0.0, cutoff=90.0),
        api_key_path=key,
        status_path=tmp_path / "status.json",
        receipt_path=tmp_path / "receipt.json",
        opener=_opener([_running(), _writable()], requests),
        now=clock,
        sleep=sleep,
        retry_seconds=1.0,
    )

    assert receipt["deletion_outcome"] == "destroyed"
    assert errors == ["credential-unavailable"]
    assert [request.method for request in requests] == ["GET", "DELETE"]


def test_destroy_retries_authorization_after_key_rotation(tmp_path: Path) -> None:
    requests: list[object] = []
    clock = _Clock(100.0)
    key = _key(tmp_path)
    errors: list[str | None] = []
    unauthorized = HTTPError(
        "https://console.vast.ai/api/v0/instances/12345/",
        401,
        "unauthorized",
        {},
        io.BytesIO(b"do-not-persist"),
    )

    def sleep(seconds: float) -> None:
        errors.append(json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["error_code"])
        key.write_text("rotated-private-key", encoding="utf-8")
        clock.value += seconds

    receipt = destroy_rental(
        _launch(now=0.0, cutoff=90.0),
        api_key_path=key,
        status_path=tmp_path / "status.json",
        receipt_path=tmp_path / "receipt.json",
        opener=_opener([unauthorized, _running(), _writable()], requests),
        now=clock,
        sleep=sleep,
        retry_seconds=1.0,
    )

    assert receipt["deletion_outcome"] == "destroyed"
    assert errors == ["authorization"]
    assert [request.method for request in requests] == ["GET", "GET", "DELETE"]
    assert requests[-1].get_header("Authorization") == "Bearer rotated-private-key"


def test_delete_404_is_absent_only_after_verified_get(tmp_path: Path) -> None:
    requests: list[object] = []
    receipt = destroy_rental(
        _launch(),
        api_key_path=_key(tmp_path),
        status_path=tmp_path / "status.json",
        receipt_path=tmp_path / "receipt.json",
        opener=_opener([_running(), _Response(404), _Response(404)], requests),
        now=_Clock(100.0),
    )

    assert receipt["deletion_outcome"] == "already-absent"
    assert [request.method for request in requests] == ["GET", "DELETE", "GET"]


def test_destroy_deletes_a_stopped_but_still_billable_instance(tmp_path: Path) -> None:
    requests: list[object] = []
    stopped = _Response(200, {"instances": {"id": 12345, "actual_status": "stopped"}})

    receipt = destroy_rental(
        _launch(),
        api_key_path=_key(tmp_path),
        status_path=tmp_path / "status.json",
        receipt_path=tmp_path / "receipt.json",
        opener=_opener([stopped, _writable()], requests),
        now=_Clock(100.0),
    )

    assert receipt["deletion_outcome"] == "destroyed"
    assert [request.method for request in requests] == ["GET", "DELETE"]


def test_restart_recovers_absent_receipt_after_deletion_before_publication(tmp_path: Path) -> None:
    requests: list[object] = []
    launch = _launch(now=100.0, cutoff=200.0)
    receipt_path = tmp_path / "receipt.json"

    first = destroy_rental(
        launch,
        api_key_path=_key(tmp_path),
        status_path=tmp_path / "status.json",
        receipt_path=receipt_path,
        opener=_opener([_running(), _writable()], requests),
        now=_Clock(100.0),
    )
    receipt_path.unlink()

    second_requests: list[object] = []
    recovered = destroy_rental(
        launch,
        api_key_path=_key(tmp_path),
        status_path=tmp_path / "status.json",
        receipt_path=receipt_path,
        opener=_opener([_Response(404)], second_requests),
        now=_Clock(120.0),
    )

    assert first["deletion_outcome"] == "destroyed"
    assert recovered["deletion_outcome"] == "already-absent"
    assert recovered["spend_usd"] == pytest.approx(20.0 * 1.25 / 3600.0)
    assert [request.method for request in second_requests] == ["GET"]
    assert recovered == json.loads(receipt_path.read_text(encoding="utf-8"))


def test_restart_returns_existing_terminal_receipt_without_provider_request(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    launch = _launch(now=100.0, cutoff=200.0)

    original = destroy_rental(
        launch,
        api_key_path=_key(tmp_path),
        status_path=tmp_path / "status.json",
        receipt_path=receipt_path,
        opener=_opener([_running(), _writable()], []),
        now=_Clock(100.0),
    )

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("terminal receipt restart must not call Vast")

    recovered = destroy_rental(
        launch,
        api_key_path=_key(tmp_path),
        status_path=tmp_path / "status.json",
        receipt_path=receipt_path,
        opener=unexpected_request,
        now=_Clock(180.0),
    )

    assert recovered == original


def test_replacement_receipt_preserves_prior_spend(tmp_path: Path) -> None:
    requests: list[object] = []
    launch = _launch(now=100.0, cutoff=200.0)
    launch["offer"]["prior_spend_usd"] = 12.5

    receipt = destroy_rental(
        launch,
        api_key_path=_key(tmp_path),
        status_path=tmp_path / "status.json",
        receipt_path=tmp_path / "receipt.json",
        opener=_opener([_running(), _writable()], requests),
        now=_Clock(136.0),
    )

    assert receipt["spend_usd"] == pytest.approx(12.5 + 36.0 * 1.25 / 3600.0)


def test_external_monitor_deletes_after_verified_terminal_backup(tmp_path: Path, monkeypatch) -> None:
    requests: list[object] = []
    launch_path = tmp_path / "launch.json"
    launch = _launch(now=100.0, cutoff=200.0)
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    monkeypatch.setattr("recipes.speakrs.large.vast_guard.parse_launch_lock", lambda payload: payload)
    backup_status = tmp_path / "backup-status.json"
    backup_status.write_text(
        json.dumps(
            {
                "schema": "speakrs-remote-backup-status-v1",
                "launch_id": launch["launch_id"],
                "state": "completed",
                "latest_generation": "update_00060000",
            }
        ),
        encoding="utf-8",
    )

    receipt = monitor_rental_guard(
        launch_path,
        api_key_path=_key(tmp_path),
        backup_status_path=backup_status,
        status_path=tmp_path / "status.json",
        receipt_path=tmp_path / "receipt.json",
        opener=_opener([_running(), _writable()], requests),
        now=_Clock(100.0),
    )

    assert receipt["deletion_outcome"] == "destroyed"
    assert [request.method for request in requests] == ["GET", "DELETE"]
