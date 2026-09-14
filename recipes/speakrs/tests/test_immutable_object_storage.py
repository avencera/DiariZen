"""Focused tests for streamed conditional immutable object publication."""

from __future__ import annotations

import hashlib
import io
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from recipes.speakrs.large.contracts import ObjectStoreDestination
from recipes.speakrs.large.hashing import sha256_bytes
from recipes.speakrs.large.storage import (
    DEFAULT_S3_SINGLE_PART_MAX_BYTES,
    WRANGLER_R2_REST_MAX_UPLOAD_BYTES,
    ImmutableObjectConflictError,
    ImmutableObjectLimits,
    ImmutableObjectOutcome,
    ImmutableObjectUnsupportedSizeError,
    S3CompatibleBackend,
    WranglerR2Backend,
    publish_immutable_file,
)


class _Response:
    def __init__(
        self,
        request,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        read_sizes: list[int] | None = None,
    ):
        self.request = request
        self.status = status
        self.headers = headers or {}
        self.read_sizes = read_sizes if read_sizes is not None else []

    def __enter__(self):
        if self.request.data is not None:
            chunks = []
            while True:
                chunk = self.request.data.read(3)
                self.read_sizes.append(len(chunk))
                if not chunk:
                    break
                chunks.append(chunk)
            self.payload = b"".join(chunks)
        else:
            self.payload = b""
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size: int = -1) -> bytes:
        return b""


def _destination() -> ObjectStoreDestination:
    return ObjectStoreDestination(
        provider="r2",
        endpoint="https://example.invalid",
        bucket="private",
        prefix="datasets/task",
        credential_reference="rclone:r2",
    )


def _backend(*, limit: int = DEFAULT_S3_SINGLE_PART_MAX_BYTES) -> S3CompatibleBackend:
    return S3CompatibleBackend(
        destination=_destination(),
        access_key="access",
        secret_key="secret",
        immutable_object_limits=ImmutableObjectLimits(max_single_part_bytes=limit),
    )


def _write_file(tmp_path: Path, payload: bytes = b"immutable payload") -> tuple[Path, str]:
    path = tmp_path / "object.bin"
    path.write_bytes(payload)
    return path, sha256_bytes(payload)


def test_s3_publication_streams_exact_body_and_uses_conditional_metadata(tmp_path: Path) -> None:
    path, digest = _write_file(tmp_path)
    backend = _backend()
    read_sizes: list[int] = []

    def urlopen(request, *, timeout):
        assert timeout == backend.timeout
        assert request.get_header("If-none-match") == "*"
        assert request.get_header("Content-length") == str(path.stat().st_size)
        assert request.get_header("X-amz-meta-sha256") == digest
        assert not isinstance(request.data, bytes)
        return _Response(request, read_sizes=read_sizes)

    with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", side_effect=urlopen):
        outcome = publish_immutable_file(
            backend,
            _destination(),
            "datasets/task/object.bin",
            path,
            expected_sha256=digest,
            expected_size=path.stat().st_size,
        )

    assert outcome is ImmutableObjectOutcome.CREATED
    assert read_sizes[-1] == 0
    assert all(0 < size <= 3 for size in read_sizes[:-1])
    assert sum(read_sizes) == path.stat().st_size


def test_s3_precondition_race_verifies_existing_head(tmp_path: Path) -> None:
    path, digest = _write_file(tmp_path)
    backend = _backend()
    precondition = urllib.error.HTTPError(
        "https://example.invalid",
        412,
        "precondition",
        {},
        io.BytesIO(b""),
    )

    def urlopen(request, *, timeout):
        if request.get_method() == "PUT":
            return precondition
        return _Response(
            request,
            headers={"content-length": str(path.stat().st_size), "x-amz-meta-sha256": digest},
        )

    with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", side_effect=urlopen) as mocked:
        outcome = publish_immutable_file(
            backend,
            _destination(),
            "datasets/task/object.bin",
            path,
            expected_sha256=digest,
            expected_size=path.stat().st_size,
        )

    assert outcome is ImmutableObjectOutcome.ALREADY_PRESENT
    assert mocked.call_count == 2


def test_existing_object_with_missing_or_changed_digest_is_conflict(tmp_path: Path) -> None:
    path, digest = _write_file(tmp_path)
    backend = _backend()
    precondition = urllib.error.HTTPError(
        "https://example.invalid",
        412,
        "precondition",
        {},
        io.BytesIO(b""),
    )

    def urlopen(request, *, timeout):
        if request.get_method() == "PUT":
            return precondition
        return _Response(request, headers={"content-length": str(path.stat().st_size)})

    with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", side_effect=urlopen):
        with pytest.raises(ImmutableObjectConflictError):
            publish_immutable_file(
                backend,
                _destination(),
                "datasets/task/object.bin",
                path,
                expected_sha256=digest,
                expected_size=path.stat().st_size,
            )


def test_provider_limit_rejects_without_network_or_hashing_large_file(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"xx")
    backend = _backend(limit=1)

    with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen") as urlopen:
        with pytest.raises(ImmutableObjectUnsupportedSizeError):
            publish_immutable_file(
                backend,
                _destination(),
                "datasets/task/large.bin",
                path,
                expected_sha256=hashlib.sha256(b"xx").hexdigest(),
                expected_size=2,
            )
    urlopen.assert_not_called()


def test_wrangler_rejects_large_file_before_rest_upload(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    with path.open("wb") as stream:
        stream.truncate(WRANGLER_R2_REST_MAX_UPLOAD_BYTES + 1)
    backend = WranglerR2Backend(
        destination=_destination(),
        api_token="token",
        account_id="0123456789abcdef0123456789abcdef",
    )

    with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen") as urlopen:
        with pytest.raises(ImmutableObjectUnsupportedSizeError) as error:
            publish_immutable_file(
                backend,
                _destination(),
                "datasets/task/large.bin",
                path,
                expected_sha256=hashlib.sha256(b"not-read").hexdigest(),
                expected_size=path.stat().st_size,
            )
    urlopen.assert_not_called()
    assert error.value.details["max_single_part_bytes"] == WRANGLER_R2_REST_MAX_UPLOAD_BYTES
