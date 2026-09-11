"""Trusted Mac-side guard for bounded Vast rental deletion."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import RuntimeGateError
from .jsonio import read_json
from .training_admission import parse_launch_lock


VAST_INSTANCE_URL = "https://console.vast.ai/api/v0/instances/{instance_id}/"
RENTAL_RECEIPT_SCHEMA = "speakrs-rental-attempt-receipt-v1"
STATUS_SCHEMA = "speakrs-vast-rental-guard-status-v1"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_RETRY_SECONDS = 10.0
MAX_REQUEST_TIMEOUT_SECONDS = 60.0
MAX_RETRY_SECONDS = 60.0
DESTROY_REQUEST_RESERVE_SECONDS = 60.0
MAX_RESPONSE_BYTES = 1024 * 1024

__all__ = [
    "RENTAL_RECEIPT_SCHEMA",
    "STATUS_SCHEMA",
    "VAST_INSTANCE_URL",
    "arm_rental_guard",
    "destroy_rental",
    "monitor_rental_guard",
    "preflight_instance",
    "preflight_rental",
]

Clock = Callable[[], float]
Opener = Callable[..., Any]
Sleeper = Callable[[float], None]


class _GuardFailure(Exception):
    def __init__(self, code: str, *, transient: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.transient = transient


class _CutoffReached(Exception):
    pass


@dataclass(frozen=True)
class _Attempt:
    attempt_id: str
    launch_id: str
    instance_id: str
    rented_at: str
    rented_epoch: float
    cutoff_epoch: float
    rates: dict[str, object]
    total_rate: float
    prior_spend: float


def _path(value: Path | str, label: str) -> Path:
    result = Path(value)
    if not str(result):
        raise RuntimeGateError(f"{label} is required")
    return result


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeGateError(f"{label} must be a non-empty string")
    return value


def _instance_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise RuntimeGateError("rental instance_id must be a decimal identifier")
    result = str(value)
    if not result.isdecimal() or int(result) <= 0:
        raise RuntimeGateError("rental instance_id must be a positive decimal identifier")
    return result


def _timestamp(value: object, label: str) -> tuple[str, float]:
    if isinstance(value, datetime):
        parsed = value
        text = value.isoformat()
    elif isinstance(value, str) and value:
        text = value
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise RuntimeGateError(f"{label} must be an ISO-8601 timestamp") from error
    else:
        raise RuntimeGateError(f"{label} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise RuntimeGateError(f"{label} must include a UTC offset")
    return text, parsed.astimezone(timezone.utc).timestamp()


def _epoch(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeGateError(f"{label} must be a finite time")
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    _, result = _timestamp(value, label)
    return result


def _number(value: object, label: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeGateError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or not allow_zero and result == 0:
        adjective = "non-negative" if allow_zero else "positive"
        raise RuntimeGateError(f"{label} must be finite and {adjective}")
    return result


def _rates(value: object) -> tuple[dict[str, object], float]:
    if not isinstance(value, Mapping):
        raise RuntimeGateError("rental rates must be a mapping")
    allowed = {"gpu_usd_per_hour", "disk_usd_per_hour", "total_usd_per_hour"}
    if set(value) not in (allowed, allowed - {"total_usd_per_hour"}):
        raise RuntimeGateError("rental rates fields are not exact")
    gpu = _number(value.get("gpu_usd_per_hour"), "rental gpu rate", allow_zero=False)
    disk = _number(value.get("disk_usd_per_hour"), "rental disk rate", allow_zero=True)
    total = gpu + disk
    if "total_usd_per_hour" in value:
        supplied = _number(value["total_usd_per_hour"], "rental total rate", allow_zero=False)
        if not math.isclose(supplied, total, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeGateError("rental total rate does not equal GPU plus disk rates")
    else:
        supplied = total
    rates = dict(value)
    rates["total_usd_per_hour"] = supplied
    return rates, total


def _attempt(launch: Mapping[str, Any]) -> _Attempt:
    attempt_id = _required_text(launch.get("attempt_id"), "launch attempt_id")
    launch_id = _required_text(launch.get("launch_id"), "launch launch_id")
    offer = launch.get("offer")
    if not isinstance(offer, Mapping):
        raise RuntimeGateError("launch offer is required")
    instance_id = _instance_id(offer.get("instance_id"))
    rented_at, rented_epoch = _timestamp(offer.get("rented_at"), "rental rented_at")
    destroy_at, cutoff_epoch = _timestamp(offer.get("destroy_at"), "rental destroy_at")
    rates, total_rate = _rates(offer.get("rates"))
    prior_spend = _number(offer.get("prior_spend_usd", 0.0), "rental prior spend", allow_zero=True)
    if cutoff_epoch <= rented_epoch:
        raise RuntimeGateError("rental destroy_at must be after rented_at")
    return _Attempt(
        attempt_id=attempt_id,
        launch_id=launch_id,
        instance_id=instance_id,
        rented_at=rented_at,
        rented_epoch=rented_epoch,
        cutoff_epoch=cutoff_epoch,
        rates=rates,
        total_rate=total_rate,
        prior_spend=prior_spend,
    )


def _clock(now: Clock | None) -> Clock:
    return time.time if now is None else now


def _opener(opener: Opener | None) -> Opener:
    return urllib.request.urlopen if opener is None else opener


def _sleeper(sleep: Sleeper | None) -> Sleeper:
    return time.sleep if sleep is None else sleep


def _bounded_positive(value: object, label: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeGateError(f"{label} must be positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise RuntimeGateError(f"{label} must be positive")
    return min(result, maximum)


def _current(now: Clock) -> float:
    try:
        value = float(now())
    except (TypeError, ValueError) as error:
        raise RuntimeGateError("guard clock returned an invalid time") from error
    if not math.isfinite(value):
        raise RuntimeGateError("guard clock returned an invalid time")
    return value


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _read_key(api_key_path: Path | str) -> str:
    path = _path(api_key_path, "Vast API key path")
    try:
        key = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise RuntimeGateError("Vast API key file cannot be read") from error
    if not key:
        raise RuntimeGateError("Vast API key file is empty")
    return key


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        raise RuntimeGateError("guard parent directory cannot be opened for durability") from error
    try:
        os.fsync(directory_fd)
    except OSError as error:
        raise RuntimeGateError("guard parent directory cannot be synced") from error
    finally:
        os.close(directory_fd)


def _write_json(path: Path | str, payload: Mapping[str, object]) -> None:
    destination = _path(path, "guard output path")
    temporary = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except RuntimeGateError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeGateError("guard output could not be published durably") from error


def _status(
    attempt: _Attempt,
    state: str,
    checked_at: float,
    *,
    deletion_outcome: str | None = None,
    attempts: int = 0,
    error_code: str | None = None,
) -> dict[str, object]:
    return {
        "schema": STATUS_SCHEMA,
        "attempt_id": attempt.attempt_id,
        "launch_id": attempt.launch_id,
        "instance_id": attempt.instance_id,
        "state": state,
        "checked_at": _iso(checked_at),
        "overdue": checked_at >= attempt.cutoff_epoch,
        "retrying": state in {"retrying", "overdue"},
        "deletion_outcome": deletion_outcome,
        "attempts": attempts,
        "error_code": error_code,
    }


def _publish_status(
    status_path: Path | str,
    attempt: _Attempt,
    state: str,
    checked_at: float,
    *,
    deletion_outcome: str | None = None,
    attempts: int = 0,
    error_code: str | None = None,
) -> None:
    _write_json(
        status_path,
        _status(
            attempt,
            state,
            checked_at,
            deletion_outcome=deletion_outcome,
            attempts=attempts,
            error_code=error_code,
        ),
    )


def _request(
    instance_id: str,
    api_key: str,
    method: str,
    *,
    opener: Opener,
    timeout: float,
    deadline: float | None,
    now: Clock,
    data: bytes | None = None,
) -> tuple[int, bytes]:
    timeout = _bounded_positive(timeout, "timeout_seconds", MAX_REQUEST_TIMEOUT_SECONDS)
    if deadline is not None:
        remaining = deadline - _current(now)
        if remaining <= 0:
            raise _CutoffReached
        timeout = min(timeout, max(0.001, remaining))
    request = urllib.request.Request(
        VAST_INSTANCE_URL.format(instance_id=instance_id),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if data is not None else {}),
        },
        data=data,
        method=method,
    )
    try:
        response = opener(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        return int(error.code), b""
    except Exception:
        raise _GuardFailure("transport", transient=True) from None
    try:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if isinstance(status, bool) or not isinstance(status, int):
            raise _GuardFailure("invalid_http_status")
        try:
            body = response.read(MAX_RESPONSE_BYTES)
        except TypeError:
            body = response.read()
        if not isinstance(body, bytes):
            body = bytes(body)
        return status, body
    except _GuardFailure:
        raise
    except Exception:
        raise _GuardFailure("transport", transient=True) from None
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


def _json_body(body: bytes, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise _GuardFailure(code, transient=True) from None
    if not isinstance(payload, Mapping):
        raise _GuardFailure(code, transient=True)
    return payload


def _get_instance(
    instance_id: str,
    api_key: str,
    *,
    opener: Opener,
    timeout: float,
    deadline: float | None,
    now: Clock,
) -> tuple[str, Mapping[str, Any] | None]:
    status, body = _request(
        instance_id,
        api_key,
        "GET",
        opener=opener,
        timeout=timeout,
        deadline=deadline,
        now=now,
    )
    if status in {401, 403}:
        raise _GuardFailure("authorization")
    if status == 404:
        return "absent", None
    if status != 200:
        raise _GuardFailure(f"http_{status}", transient=status == 429 or status >= 500)
    payload = _json_body(body, "invalid_get_json")
    instance = payload.get("instances")
    if not isinstance(instance, Mapping):
        raise _GuardFailure("invalid_instance_payload")
    actual_id = instance.get("id")
    try:
        actual_id_text = _instance_id(actual_id)
    except RuntimeGateError as error:
        raise _GuardFailure("invalid_instance_id") from error
    if actual_id_text != instance_id:
        raise _GuardFailure("wrong_instance")
    return "present", instance


def _require_running(instance_id: str, instance: Mapping[str, Any]) -> None:
    if instance.get("actual_status") != "running":
        raise _GuardFailure("not_running")


def _verify_write_access(
    instance_id: str,
    api_key: str,
    *,
    opener: Opener,
    timeout: float,
    deadline: float | None,
    now: Clock,
) -> None:
    status, body = _request(
        instance_id,
        api_key,
        "PUT",
        opener=opener,
        timeout=timeout,
        deadline=deadline,
        now=now,
        data=b'{"state":"running"}',
    )
    if status in {401, 403}:
        raise _GuardFailure("authorization")
    if status != 200:
        raise _GuardFailure(f"http_{status}", transient=status == 429 or status >= 500)
    payload = _json_body(body, "invalid_write_json")
    if payload.get("success") is not True:
        raise _GuardFailure("write_not_authorized")


def _delete_instance(
    instance_id: str,
    api_key: str,
    *,
    opener: Opener,
    timeout: float,
    deadline: float | None,
    now: Clock,
) -> str:
    status, body = _request(
        instance_id,
        api_key,
        "DELETE",
        opener=opener,
        timeout=timeout,
        deadline=deadline,
        now=now,
    )
    if status in {401, 403}:
        raise _GuardFailure("authorization")
    if status == 404:
        return "delete-absent"
    if status != 200:
        raise _GuardFailure(f"http_{status}", transient=status == 429 or status >= 500)
    payload = _json_body(body, "invalid_delete_json")
    if payload.get("success") is not True:
        raise _GuardFailure("delete_not_successful", transient=True)
    return "destroyed"


def _runtime_failure(failure: _GuardFailure) -> RuntimeGateError:
    return RuntimeGateError("Vast rental guard failed", {"code": failure.code})


def preflight_instance(
    instance_id: str | int,
    *,
    api_key_path: Path | str,
    opener: Opener | None = None,
    now: Clock | None = None,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    request_cutoff: float | datetime | str | None = None,
) -> dict[str, object]:
    """Verify one exact Vast instance is authenticated and running.

    The API key is read from ``api_key_path`` for this request only. The returned
    mapping contains no provider response fields, because those fields may contain
    credentials such as a container token.
    """

    expected_id = _instance_id(instance_id)
    clock = _clock(now)
    url_opener = _opener(opener)
    timeout = _bounded_positive(timeout_seconds, "timeout_seconds", MAX_REQUEST_TIMEOUT_SECONDS)
    cutoff = None if request_cutoff is None else _epoch(request_cutoff, "request cutoff")
    current = _current(clock)
    if cutoff is not None and current >= cutoff:
        raise RuntimeGateError("Vast rental request cutoff has passed", {"code": "cutoff"})
    key = _read_key(api_key_path)
    try:
        state, instance = _get_instance(
            expected_id,
            key,
            opener=url_opener,
            timeout=timeout,
            deadline=cutoff,
            now=clock,
        )
    except _CutoffReached:
        raise RuntimeGateError("Vast rental request cutoff has passed", {"code": "cutoff"}) from None
    except _GuardFailure as error:
        raise _runtime_failure(error) from None
    if state == "absent" or instance is None:
        raise RuntimeGateError("Vast rental instance is absent", {"code": "absent"})
    try:
        _require_running(expected_id, instance)
        _verify_write_access(
            expected_id,
            key,
            opener=url_opener,
            timeout=timeout,
            deadline=cutoff,
            now=clock,
        )
    except _CutoffReached:
        raise RuntimeGateError("Vast rental request cutoff has passed", {"code": "cutoff"}) from None
    except _GuardFailure as error:
        raise _runtime_failure(error) from None
    return {
        "ok": True,
        "instance_id": expected_id,
        "actual_status": "running",
        "checked_at": _iso(_current(clock)),
    }


def arm_rental_guard(
    launch: Mapping[str, Any],
    *,
    api_key_path: Path | str,
    status_path: Path | str,
    opener: Opener | None = None,
    now: Clock | None = None,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Preflight and durably arm the trusted guard before training starts.

    ``launch`` must be the durable launch mapping. The status file contains only
    non-secret attempt identity and guard state. A non-running, wrong, absent, or
    unauthenticated instance raises ``RuntimeGateError`` before training may start.
    """

    attempt = _attempt(launch)
    clock = _clock(now)
    current = _current(clock)
    if current >= attempt.cutoff_epoch:
        _publish_status(status_path, attempt, "blocked", min(current, attempt.cutoff_epoch), error_code="cutoff")
        raise RuntimeGateError("Vast rental request cutoff has passed", {"code": "cutoff"})
    try:
        result = preflight_instance(
            attempt.instance_id,
            api_key_path=api_key_path,
            opener=opener,
            now=clock,
            timeout_seconds=timeout_seconds,
            request_cutoff=attempt.cutoff_epoch,
        )
    except RuntimeGateError as error:
        code = error.details.get("code") if isinstance(error.details, Mapping) else None
        _publish_status(
            status_path,
            attempt,
            "blocked",
            min(_current(clock), attempt.cutoff_epoch),
            error_code=code if isinstance(code, str) else "preflight",
        )
        raise
    checked_at = _current(clock)
    _publish_status(status_path, attempt, "armed", checked_at)
    return {
        "ok": True,
        "state": "armed",
        "attempt_id": attempt.attempt_id,
        "launch_id": attempt.launch_id,
        "instance_id": attempt.instance_id,
        "checked_at": result["checked_at"],
        "status_path": str(_path(status_path, "guard status path")),
    }


_RECEIPT_FIELDS = {
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


def _existing_receipt(path: Path | str, attempt: _Attempt) -> dict[str, object] | None:
    try:
        payload = read_json(_path(path, "rental receipt path"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, Mapping) or set(payload) != _RECEIPT_FIELDS:
        return None
    if (
        payload.get("schema") != RENTAL_RECEIPT_SCHEMA
        or payload.get("attempt_id") != attempt.attempt_id
        or payload.get("launch_id") != attempt.launch_id
        or payload.get("instance_id") != attempt.instance_id
        or payload.get("rented_at") != attempt.rented_at
        or payload.get("rates") != attempt.rates
        or payload.get("deletion_outcome") not in {"destroyed", "already-absent"}
        or not isinstance(payload.get("ended_at"), str)
    ):
        return None
    try:
        ended_epoch = _epoch(payload["ended_at"], "rental receipt ended_at")
        spend = _number(payload.get("spend_usd"), "rental receipt spend", allow_zero=True)
    except RuntimeGateError:
        return None
    if ended_epoch < attempt.rented_epoch:
        return None
    expected_spend = attempt.prior_spend + (ended_epoch - attempt.rented_epoch) / 3600.0 * attempt.total_rate
    if not math.isclose(spend, expected_spend, rel_tol=0.0, abs_tol=0.01):
        return None
    return dict(payload)


def destroy_rental(
    launch: Mapping[str, Any],
    *,
    api_key_path: Path | str,
    status_path: Path | str,
    receipt_path: Path | str,
    opener: Opener | None = None,
    now: Clock | None = None,
    sleep: Sleeper | None = None,
    retry_seconds: float = DEFAULT_RETRY_SECONDS,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    request_cutoff: float | datetime | str | None = None,
) -> dict[str, object]:
    """Delete a Vast instance and publish its durable spend receipt.

    Cleanup is restart-safe and deliberately less strict than
    :func:`arm_rental_guard`: a stopped instance is still deleted, an absent
    instance is receipted as already absent, and transient failures continue
    after ``destroy_at``. Every provider request and retry delay is bounded.
    A DELETE 404 is not accepted as proof of deletion until a follow-up GET
    verifies that the instance is absent.
    """

    attempt = _attempt(launch)
    cutoff = attempt.cutoff_epoch
    if request_cutoff is not None:
        supplied_cutoff = _epoch(request_cutoff, "request cutoff")
        if not math.isclose(supplied_cutoff, cutoff, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeGateError("request cutoff differs from launch destroy_at", {"code": "cutoff-mismatch"})
    retry_delay = _bounded_positive(retry_seconds, "retry_seconds", MAX_RETRY_SECONDS)
    request_timeout = _bounded_positive(timeout_seconds, "timeout_seconds", MAX_REQUEST_TIMEOUT_SECONDS)
    clock = _clock(now)
    url_opener = _opener(opener)
    wait = _sleeper(sleep)
    attempts = 0
    current = _current(clock)
    existing_receipt = _existing_receipt(receipt_path, attempt)
    if existing_receipt is not None:
        _publish_status(
            status_path,
            attempt,
            str(existing_receipt["deletion_outcome"]),
            current,
            deletion_outcome=str(existing_receipt["deletion_outcome"]),
        )
        return existing_receipt

    _publish_status(status_path, attempt, "overdue" if current >= cutoff else "destroying", current)

    outcome: str | None = None
    failure: _GuardFailure | None = None
    ended_epoch = current
    while True:
        attempts += 1
        try:
            # Read the file for each attempt so a rotated credential can recover
            # an interrupted deletion without exposing its value.
            key = _read_key(api_key_path)
            state, instance = _get_instance(
                attempt.instance_id,
                key,
                opener=url_opener,
                timeout=request_timeout,
                deadline=None,
                now=clock,
            )
            if state == "absent":
                outcome = "already-absent"
                ended_epoch = _current(clock)
                break
            if instance is None:
                raise _GuardFailure("invalid_instance_payload", transient=True)
            delete_state = _delete_instance(
                attempt.instance_id,
                key,
                opener=url_opener,
                timeout=request_timeout,
                deadline=None,
                now=clock,
            )
            if delete_state == "destroyed":
                outcome = "destroyed"
                ended_epoch = _current(clock)
                break
            verify_state, verify_instance = _get_instance(
                attempt.instance_id,
                key,
                opener=url_opener,
                timeout=request_timeout,
                deadline=None,
                now=clock,
            )
            if verify_state == "absent":
                outcome = "already-absent"
                ended_epoch = _current(clock)
                break
            if verify_instance is None:
                raise _GuardFailure("invalid_instance_payload", transient=True)
            raise _GuardFailure("delete_not_confirmed", transient=True)
        except _GuardFailure as error:
            ended_epoch = _current(clock)
            if not error.transient and error.code != "authorization":
                outcome = "failed"
                failure = error
                break
            _publish_status(
                status_path,
                attempt,
                "overdue" if ended_epoch >= cutoff else "retrying",
                ended_epoch,
                attempts=attempts,
                error_code=error.code,
            )
            wait(retry_delay)
            continue
        except RuntimeGateError:
            ended_epoch = _current(clock)
            _publish_status(
                status_path,
                attempt,
                "overdue" if ended_epoch >= cutoff else "retrying",
                ended_epoch,
                attempts=attempts,
                error_code="credential-unavailable",
            )
            wait(retry_delay)
            continue

    if outcome is None:
        outcome = "failed"
    if outcome in {"destroyed", "already-absent"}:
        _publish_status(
            status_path,
            attempt,
            outcome,
            ended_epoch,
            deletion_outcome=outcome,
            attempts=attempts,
        )
    else:
        _publish_status(
            status_path,
            attempt,
            "failed",
            ended_epoch,
            deletion_outcome="failed",
            attempts=attempts,
            error_code=failure.code if failure is not None else "deletion",
        )
    receipt = _receipt(attempt, ended_epoch, outcome)
    _write_json(receipt_path, receipt)
    if outcome == "failed":
        raise _runtime_failure(failure or _GuardFailure("deletion")) from None
    return receipt


def _backup_is_terminal(path: Path | str, launch_id: str) -> bool:
    try:
        status = read_json(_path(path, "backup status path"))
    except (OSError, UnicodeError, ValueError):
        return False
    return (
        isinstance(status, Mapping)
        and status.get("schema") == "speakrs-remote-backup-status-v1"
        and status.get("launch_id") == launch_id
        and status.get("state") == "completed"
        and isinstance(status.get("latest_generation"), str)
        and bool(status["latest_generation"])
    )


def monitor_rental_guard(
    launch_path: Path | str,
    *,
    api_key_path: Path | str,
    backup_status_path: Path | str,
    status_path: Path | str,
    receipt_path: Path | str,
    poll_seconds: float = DEFAULT_RETRY_SECONDS,
    opener: Opener | None = None,
    now: Clock | None = None,
    sleep: Sleeper | None = None,
) -> dict[str, object]:
    """Own rental deletion outside the worker until trusted backup or cutoff.

    The caller must run this guard on the trusted machine, independently from
    the worker supervisor and backup monitor. A verified terminal backup permits
    early deletion. Otherwise, the guard starts deletion with a full minute left
    before the admitted ``destroy_at`` cutoff. This monitor does not perform the
    strict startup arm: cleanup and recovery must still work after a worker
    crash, stop, or prior deletion.
    """

    poll_delay = _bounded_positive(poll_seconds, "poll_seconds", MAX_RETRY_SECONDS)
    launch = parse_launch_lock(read_json(_path(launch_path, "launch path")))
    attempt = _attempt(launch)
    clock = _clock(now)
    wait = _sleeper(sleep)
    existing_receipt = _existing_receipt(receipt_path, attempt)
    if existing_receipt is not None:
        _publish_status(
            status_path,
            attempt,
            str(existing_receipt["deletion_outcome"]),
            _current(clock),
            deletion_outcome=str(existing_receipt["deletion_outcome"]),
        )
        return existing_receipt
    request_at = attempt.cutoff_epoch - DESTROY_REQUEST_RESERVE_SECONDS
    while True:
        current = _current(clock)
        if _backup_is_terminal(backup_status_path, attempt.launch_id) or current >= request_at:
            break
        _publish_status(status_path, attempt, "monitoring", current)
        wait(min(poll_delay, request_at - current))
    return destroy_rental(
        launch,
        api_key_path=api_key_path,
        status_path=status_path,
        receipt_path=receipt_path,
        opener=opener,
        now=clock,
        sleep=wait,
        request_cutoff=attempt.cutoff_epoch,
    )


def _receipt(attempt: _Attempt, ended_epoch: float, deletion_outcome: str) -> dict[str, object]:
    ended_epoch = max(ended_epoch, attempt.rented_epoch)
    elapsed = ended_epoch - attempt.rented_epoch
    spend = attempt.prior_spend + elapsed / 3600.0 * attempt.total_rate
    if not math.isfinite(spend):
        raise RuntimeGateError("rental spend is not finite")
    return {
        "schema": RENTAL_RECEIPT_SCHEMA,
        "attempt_id": attempt.attempt_id,
        "launch_id": attempt.launch_id,
        "instance_id": attempt.instance_id,
        "rented_at": attempt.rented_at,
        "ended_at": _iso(ended_epoch),
        "rates": dict(attempt.rates),
        "deletion_outcome": deletion_outcome,
        "spend_usd": spend,
    }


def preflight_rental(*args: Any, **kwargs: Any) -> dict[str, object]:
    """Alias for :func:`arm_rental_guard` used by backup-monitor callers."""

    return arm_rental_guard(*args, **kwargs)
