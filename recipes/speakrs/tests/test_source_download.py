"""Tests for bounded, resumable source downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from recipes.speakrs.large import prepare
from recipes.speakrs.large.contracts import DiskLimits
from recipes.speakrs.large.errors import PreparationError


class _Response:
    def __init__(self, status: int, headers: dict[str, str], reads: list[object]) -> None:
        self.status = status
        self.headers = headers
        self._reads = iter(reads)

    def read(self, _size: int) -> bytes:
        value = next(self._reads, b"")
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self) -> None:
        pass


def _limits(root: Path, *, staging_bytes: int = 64) -> DiskLimits:
    staging = root / "staging"
    cache = root / "cache"
    staging.mkdir()
    cache.mkdir()
    return DiskLimits(staging, cache, staging_bytes, 64, 0, 1)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_source_download_resumes_from_the_persisted_partial_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"abcdefgh"
    limits = _limits(tmp_path)
    destination = limits.staging_root / "source.bin"
    responses = [
        _Response(
            206,
            {"Content-Range": "bytes 0-7/8", "Content-Length": "8"},
            [b"abcd", ConnectionResetError("private-token")],
        ),
        _Response(206, {"Content-Range": "bytes 4-7/8", "Content-Length": "4"}, [b"efgh"]),
    ]
    requests = []

    def urlopen(request: object) -> _Response:
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(prepare.urllib.request, "urlopen", urlopen)

    result = prepare.download_source_file(
        "https://source.test/file?token=private-token",
        destination,
        len(payload),
        _digest(payload),
        limits,
        block_size=4,
    )

    assert destination.read_bytes() == payload
    assert not destination.with_name(destination.name + ".part").exists()
    assert [request.headers["Range"] for request in requests] == ["bytes=0-", "bytes=4-"]
    assert result["attempts"] == 2
    assert result["resumed_from_bytes"] == 0
    assert "private-token" not in repr(result)


def test_source_download_rejects_a_full_response_when_resuming_and_preserves_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"abcdefgh"
    limits = _limits(tmp_path)
    destination = limits.staging_root / "source.bin"
    part = destination.with_name(destination.name + ".part")
    part.write_bytes(payload[:4])
    response = _Response(200, {"Content-Length": "8"}, [payload])
    requests = []

    def urlopen(request: object) -> _Response:
        requests.append(request)
        return response

    monkeypatch.setattr(prepare.urllib.request, "urlopen", urlopen)

    with pytest.raises(PreparationError) as raised:
        prepare.download_source_file(
            "https://source.test/file?token=private-token",
            destination,
            len(payload),
            _digest(payload),
            limits,
        )

    assert "private-token" not in str(raised.value)
    assert "private-token" not in repr(raised.value.details)
    assert part.read_bytes() == payload[:4]
    assert not destination.exists()
    assert requests[0].headers["Range"] == "bytes=4-"


def test_source_download_rejects_a_mismatched_content_range_and_preserves_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"abcdefgh"
    limits = _limits(tmp_path)
    destination = limits.staging_root / "source.bin"
    part = destination.with_name(destination.name + ".part")
    part.write_bytes(payload[:4])
    response = _Response(206, {"Content-Range": "bytes 0-7/8", "Content-Length": "8"}, [payload])
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda _request: response)

    with pytest.raises(PreparationError, match="Content-Range"):
        prepare.download_source_file(
            "https://source.test/file?token=private-token",
            destination,
            len(payload),
            _digest(payload),
            limits,
        )

    assert part.read_bytes() == payload[:4]
    assert not destination.exists()


def test_source_download_checks_projected_cap_before_opening_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"abcdefgh"
    limits = _limits(tmp_path, staging_bytes=7)
    destination = limits.staging_root / "source.bin"
    part = destination.with_name(destination.name + ".part")
    part.write_bytes(payload[:4])
    monkeypatch.setattr(prepare.urllib.request, "urlopen", pytest.fail)

    with pytest.raises(PreparationError, match="staging cap exhausted"):
        prepare.download_source_file(
            "https://source.test/file?token=private-token",
            destination,
            len(payload),
            _digest(payload),
            limits,
        )

    assert part.read_bytes() == payload[:4]
    assert not destination.exists()
