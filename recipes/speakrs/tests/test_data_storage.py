#!/usr/bin/env python3

"""Phase-4 storage failure tests for the object-store owners."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from recipes.speakrs.large.contracts import (  # noqa: E402
    BatchState,
    ObjectState,
    ObjectStoreDestination,  # noqa: E402
    RemoteReleaseState,
    parse_data_preparation_spec,  # noqa: E402
    parse_object_receipt,
)
from recipes.speakrs.large.data import restore_check  # noqa: E402
from recipes.speakrs.large.errors import ContractError, PreparationError, UnresolvedInputError  # noqa: E402
from recipes.speakrs.large.hashing import sha256_bytes, sha256_file, sha256_json  # noqa: E402
from recipes.speakrs.large.jsonio import write_json  # noqa: E402
from recipes.speakrs.large.storage import (  # noqa: E402
    ConsumedSource,
    ContentAddressedWrite,
    LocalCopy,
    LocalCopyState,
    MemoryBackend,
    RemoteRestoreProof,
    WranglerR2Backend,
    assert_no_open_file_handles,
    assert_private_access,
    commit_batch,
    commit_release,
    discard_consumed_source,
    discard_consumed_sources,
    enforce_cap,
    evict_copies,
    evict_copy,
    full_readback_sha256,
    mark_eviction_eligible,
    mark_readback_verified,
    object_key,
    put_content_addressed,
    recover_deletion_journal,
    upload_success_is_not_proof,
    validate_content_addressed_write,
    verify_and_commit_batch,
)
from recipes.speakrs.tests.test_data_acceptance import (  # noqa: E402
    _data_spec_payload,
    _h,
    _rttm,
)


def _audio_and_label(root: Path, recording: str = "rec") -> tuple[Path, Path, str, str]:
    audio = root / f"{recording}.flac"
    samples = np.zeros(16000 * 20, dtype=np.float32)
    samples[::160] = 0.1
    sf.write(audio, samples, 16000)
    label = root / f"{recording}.rttm"
    label.write_text(_rttm(recording, ("spk1", 0.0, 8.0), ("spk2", 4.0, 8.0)), encoding="utf-8")
    return audio, label, sha256_bytes(audio.read_bytes()), sha256_bytes(label.read_bytes())


class _FakeHttpResponse:
    """Small bounded response double for authenticated REST object calls."""

    def __init__(self, status: int, chunks: list[bytes], *, read_error: BaseException | None = None):
        self.status = status
        self._chunks = list(chunks)
        self._read_error = read_error
        self.headers: dict[str, str] = {}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True
        return False

    def read(self, size: int = -1) -> bytes:
        if not self._chunks:
            if self._read_error is not None:
                raise self._read_error
            return b""
        chunk = self._chunks.pop(0)
        if size >= 0 and len(chunk) > size:
            self._chunks.insert(0, chunk[size:])
            return chunk[:size]
        return chunk


class WranglerR2BackendTest(unittest.TestCase):
    _PREFIX = "datasets/diarization-data-verification"

    def _backend(self, *, api_token: str | None = "api-token") -> WranglerR2Backend:
        destination = ObjectStoreDestination(
            provider="r2",
            endpoint="https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
            bucket="praveen",
            prefix=self._PREFIX,
            credential_reference="wrangler",
        )
        return WranglerR2Backend(
            destination,
            api_token=api_token,
            account_id="0123456789abcdef0123456789abcdef",
            api_base_url="https://api.example.invalid/client/v4",
        )

    def _owned_key(self, payload: bytes = b"alpha", extension: str = "txt") -> str:
        return f"{self._PREFIX}/{object_key('AMI', 'official', 'train', sha256_bytes(payload), extension)}"

    def test_put_get_and_same_key_conflict(self):
        from unittest import mock

        backend = self._backend()
        missing = urllib.error.HTTPError(
            "https://api.example.invalid",
            404,
            "not found",
            {},
            io.BytesIO(b"missing"),
        )
        with mock.patch(
            "recipes.speakrs.large.storage.urllib.request.urlopen",
            side_effect=[missing, _FakeHttpResponse(200, [b""]), _FakeHttpResponse(200, [b"alpha"])],
        ) as urlopen:
            key = self._owned_key()
            backend.put_bytes(key, b"alpha")
            self.assertEqual(backend.get_bytes(key), b"alpha")
            self.assertEqual(urlopen.call_count, 3)
            self.assertIsNone(urlopen.call_args_list[1].args[0].get_header("If-none-match"))
        with mock.patch(
            "recipes.speakrs.large.storage.urllib.request.urlopen",
            return_value=_FakeHttpResponse(200, [b"alpha"]),
        ):
            with self.assertRaises(PreparationError):
                backend.put_bytes(self._owned_key(), b"alpha")

        with mock.patch(
            "recipes.speakrs.large.storage.urllib.request.urlopen",
            side_effect=[_FakeHttpResponse(200, [b"beta"]), _FakeHttpResponse(200, [b"beta"])],
        ) as urlopen:
            with self.assertRaises(PreparationError):
                backend.put_bytes(self._owned_key(), b"alpha")
        self.assertEqual(urlopen.call_count, 2)

    def test_direct_put_does_not_require_external_task_root(self):
        from unittest import mock

        backend = self._backend()
        missing = urllib.error.HTTPError("https://api.example.invalid", 404, "not found", {}, io.BytesIO())
        with mock.patch(
            "recipes.speakrs.large.storage.urllib.request.urlopen",
            side_effect=[missing, _FakeHttpResponse(200, [b""])],
        ) as urlopen:
            backend.put_bytes(self._owned_key(), b"alpha")
        self.assertEqual(urlopen.call_count, 2)

    def test_direct_put_uses_provider_etag_only(self):
        from unittest import mock

        backend = self._backend()
        missing = urllib.error.HTTPError("https://api.example.invalid", 404, "not found", {}, io.BytesIO())
        response = _FakeHttpResponse(200, [b""])
        response.headers["ETag"] = '"provider-etag"'
        with mock.patch(
            "recipes.speakrs.large.storage.urllib.request.urlopen",
            side_effect=[missing, response],
        ):
            result = backend.put_bytes(self._owned_key(), b"alpha")
        self.assertEqual(result["etag"], "provider-etag")

        existing_head = _FakeHttpResponse(200, [b"alpha"])
        existing_head.headers["ETag"] = '"provider-etag"'
        existing_get = _FakeHttpResponse(200, [b"alpha"])
        with mock.patch(
            "recipes.speakrs.large.storage.urllib.request.urlopen",
            side_effect=[existing_head, existing_get],
        ):
            result = backend.put_bytes(self._owned_key(), b"alpha")
        self.assertEqual(result["etag"], "")

    def test_generic_key_rejected_before_network_access(self):
        from unittest import mock

        backend = self._backend()
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen") as urlopen:
            with self.assertRaises(ContractError):
                backend.put_bytes("k1", b"alpha")
        urlopen.assert_not_called()

    def test_direct_backend_does_not_claim_undocumented_atomic_create(self):
        self.assertFalse(self._backend().atomic_create_supported)

    def test_content_addressed_writer_rejects_bad_keys_before_network_access(self):
        from unittest import mock

        backend = self._backend()
        payload = b"alpha"
        malformed_marker_base = {
            "schema": "speakrs-remote-batch-v1",
            "state": "committed",
            "objects": [],
            "label_policy_id": "label-policy",
            "split_id": "frozen",
            "qa_policy_sha256": sha256_bytes(b"qa"),
            "acceptance_sha256": sha256_bytes(b"acceptance"),
            "inventory": {"complete": True},
        }
        malformed_marker_digest = sha256_json(malformed_marker_base)
        bad_keys = (
            f"{self._PREFIX}/custom/alpha.txt",
            f"{self._PREFIX}/datasets/AMI/official/train/not-a-digest.flac",
            f"{self._PREFIX}/_commits/batches/{malformed_marker_digest}.json",
        )
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen") as urlopen:
            for key in bad_keys:
                with self.subTest(key=key), self.assertRaises(ContractError):
                    backend.put_content_addressed(key, payload, content_type="application/json")
        urlopen.assert_not_called()

    def test_content_addressed_function_validates_before_backend_put(self):
        from unittest import mock

        backend = self._backend()
        payload = b"alpha"
        key = self._owned_key(payload)
        with mock.patch.object(backend, "put_content_addressed") as writer:
            result = put_content_addressed(backend, backend.destination, key, payload)
        self.assertIs(result, writer.return_value)
        writer.assert_called_once_with(key, payload, content_type="application/octet-stream")

    def test_content_addressed_writer_rejects_marker_content_type(self):
        from unittest import mock

        base = {
            "schema": "speakrs-remote-batch-v1",
            "state": "committed",
            "objects": [],
            "label_policy_id": "label-policy",
            "split_id": "frozen",
            "qa_policy_sha256": sha256_bytes(b"qa"),
            "acceptance_sha256": sha256_bytes(b"acceptance"),
            "inventory": {"complete": True},
        }
        digest = sha256_json(base)
        marker = {**base, "batch_sha256": digest}
        payload = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        key = f"{self._PREFIX}/_commits/batches/{digest}.json"
        backend = self._backend()
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen") as urlopen:
            with self.assertRaises(ContractError):
                backend.put_content_addressed(key, payload, content_type="text/plain")
        urlopen.assert_not_called()

    def test_content_addressed_writer_accepts_nested_production_batch_marker(self):
        from unittest import mock

        base = {
            "schema": "speakrs-remote-batch-v1",
            "state": BatchState.COMMITTED.value,
            "objects": [],
            "label_policy_id": "label-policy",
            "split_id": "frozen",
            "qa_policy_sha256": sha256_bytes(b"qa"),
            "acceptance_sha256": sha256_bytes(b"acceptance"),
            "inventory": {"complete": True},
        }
        digest = sha256_json(base)
        marker = {**base, "batch_sha256": digest}
        payload = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        key = f"{self._PREFIX}/datasets/AMI/official/train/_commits/batches/{digest}.json"
        missing = urllib.error.HTTPError("https://api.example.invalid", 404, "not found", {}, io.BytesIO())
        with mock.patch(
            "recipes.speakrs.large.storage.urllib.request.urlopen",
            side_effect=[missing, _FakeHttpResponse(200, [b""])],
        ) as urlopen:
            result = self._backend().put_content_addressed(key, payload, content_type="application/json")

        self.assertEqual(result["key"], key)
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args_list[1].args[0].get_header("Content-type"), "application/json")

    def test_concurrent_same_key_write_uses_identical_content(self):
        class BarrierBackend(MemoryBackend):
            def __init__(self):
                super().__init__()
                self.barrier = threading.Barrier(8)

            def put_content_addressed(self, key, payload, *, content_type="application/octet-stream"):
                self.barrier.wait(timeout=5)
                return super().put_content_addressed(key, payload, content_type=content_type)

        backend = BarrierBackend()
        payload = b"alpha"
        key = self._owned_key(payload)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(backend.put_content_addressed, key, payload) for _ in range(8)]
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual({result["key"] for result in results}, {key})
        self.assertEqual(backend.get_bytes(key), payload)
        self.assertEqual(len(backend.objects), 1)

    def test_content_addressed_validation_rejects_hash_mismatch_before_network_access(self):
        from unittest import mock

        backend = self._backend()
        key = self._owned_key(b"different")
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen") as urlopen:
            with self.assertRaises(ContractError):
                backend.put_bytes(key, b"alpha")
        urlopen.assert_not_called()

    def test_content_addressed_validation_rejects_noncanonical_marker_before_network_access(self):
        from unittest import mock

        base = {
            "schema": "speakrs-remote-batch-v1",
            "state": "committed",
            "objects": [],
            "label_policy_id": "label-policy",
            "split_id": "frozen",
            "qa_policy_sha256": sha256_bytes(b"qa"),
            "acceptance_sha256": sha256_bytes(b"acceptance"),
            "inventory": {"complete": True},
        }
        digest = sha256_json(base)
        marker = {**base, "batch_sha256": digest}
        noncanonical = ('{\n  "batch_sha256": ' + json.dumps(digest) + "\n}\n").encode("utf-8")
        key = f"{self._PREFIX}/_commits/batches/{digest}.json"
        backend = self._backend()
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen") as urlopen:
            with self.assertRaises(ContractError):
                backend.put_bytes(key, noncanonical, content_type="application/json")
        urlopen.assert_not_called()
        canonical = (
            json.dumps(marker, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
        )
        self.assertEqual(
            validate_content_addressed_write(
                backend.destination, key, canonical, content_type="application/json"
            ).kind,
            "batches-marker",
        )
        self.assertEqual(
            ContentAddressedWrite.validate(
                backend.destination, key, canonical, content_type="application/json"
            ).digest,
            digest,
        )

    def test_exists_does_not_treat_wrangler_permission_failure_as_missing(self):
        from unittest import mock

        backend = self._backend()
        denied = urllib.error.HTTPError("https://api.example.invalid", 403, "denied", {}, io.BytesIO(b"denied"))
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", side_effect=denied):
            with self.assertRaises(UnresolvedInputError):
                backend.exists(self._owned_key())

    def test_exists_classifies_only_explicit_not_found_as_missing(self):
        from unittest import mock

        backend = self._backend()
        missing = urllib.error.HTTPError("https://api.example.invalid", 404, "not found", {}, io.BytesIO())
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", side_effect=missing):
            self.assertFalse(backend.exists(self._owned_key()))

    def test_iter_bytes_enforces_bound_and_streams_without_temp_files(self):
        from unittest import mock

        backend = self._backend()
        response = _FakeHttpResponse(200, [b"ab", b"cd", b"ef"])
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(
                list(backend.iter_bytes("folder/a file.txt", chunk_size=2, max_bytes=6)), [b"ab", b"cd", b"ef"]
            )
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://api.example.invalid/client/v4/accounts/0123456789abcdef0123456789abcdef/r2/buckets/praveen/objects/folder/a%20file.txt",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer api-token")
        self.assertEqual(request.get_header("User-agent"), "wrangler/4.129.0")
        self.assertTrue(response.closed)

        response = _FakeHttpResponse(200, [b"abcd"])
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", return_value=response):
            with self.assertRaises(PreparationError):
                list(backend.iter_bytes("bounded", max_bytes=3))

    def test_interrupted_and_corrupt_gets_fail_closed(self):
        from unittest import mock

        backend = self._backend()
        interrupted = _FakeHttpResponse(200, [b"good"], read_error=OSError("connection reset"))
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", return_value=interrupted):
            with self.assertRaises(UnresolvedInputError):
                list(backend.iter_bytes("interrupted"))
        corrupt = _FakeHttpResponse(200, [b"wrong"])
        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", return_value=corrupt):
            with self.assertRaises(PreparationError):
                full_readback_sha256(backend, "corrupt", sha256_bytes(b"expected"))

    def test_oversized_put_is_unresolved_before_network_access(self):
        from unittest import mock

        backend = self._backend()
        with mock.patch("recipes.speakrs.large.storage.WRANGLER_R2_REST_MAX_UPLOAD_BYTES", 3):
            with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen") as urlopen:
                with self.assertRaises(UnresolvedInputError) as raised:
                    backend.put_bytes(self._owned_key(b"four"), b"four")
        self.assertEqual(raised.exception.details["max_bytes"], 3)
        urlopen.assert_not_called()

    def test_object_url_escapes_reserved_key_bytes_but_not_slashes(self):
        backend = self._backend()
        self.assertEqual(
            backend._direct_object_url("folder/a file+%.txt", "0123456789abcdef0123456789abcdef"),
            "https://api.example.invalid/client/v4/accounts/0123456789abcdef0123456789abcdef/r2/buckets/praveen/objects/folder/a%20file%2B%25.txt",
        )

    def test_oauth_cache_refreshes_after_expiry(self):
        from unittest import mock

        backend = self._backend(api_token=None)
        now = datetime.now(timezone.utc)
        states = iter(
            (
                ("token-a", now + timedelta(minutes=2)),
                ("token-a", now - timedelta(minutes=2)),
                ("token-b", now + timedelta(hours=1)),
            )
        )
        with mock.patch("recipes.speakrs.large.storage._load_wrangler_oauth_state", side_effect=states):
            with mock.patch(
                "recipes.speakrs.large.storage._wrangler_whoami",
                return_value=mock.Mock(returncode=0, stdout="{}"),
            ) as refresh:
                first = backend._direct_credentials()
                backend._credentials_cache = (first[0], first[1], now - timedelta(seconds=1))
                second = backend._direct_credentials()
        self.assertEqual(first, ("0123456789abcdef0123456789abcdef", "token-a"))
        self.assertEqual(second, ("0123456789abcdef0123456789abcdef", "token-b"))
        refresh.assert_called_once()

    def test_oauth_cache_uses_token_until_actual_expiry(self):
        from unittest import mock

        backend = self._backend(api_token=None)
        now = datetime.now(timezone.utc)
        with mock.patch(
            "recipes.speakrs.large.storage._load_wrangler_oauth_state",
            return_value=("token-a", now + timedelta(seconds=30)),
        ):
            with mock.patch("recipes.speakrs.large.storage._wrangler_whoami") as refresh:
                first = backend._direct_credentials()
                second = backend._direct_credentials()
        self.assertEqual(first, ("0123456789abcdef0123456789abcdef", "token-a"))
        self.assertEqual(second, first)
        refresh.assert_not_called()

    def test_object_read_retries_once_after_authenticated_expiry(self):
        from unittest import mock

        backend = self._backend(api_token=None)
        now = datetime.now(timezone.utc)
        states = iter(
            (
                ("token-a", now + timedelta(hours=1)),
                ("token-b", now + timedelta(hours=1)),
                ("token-b", now + timedelta(hours=1)),
            )
        )
        denied = urllib.error.HTTPError("https://api.example.invalid", 401, "expired", {}, io.BytesIO())
        with mock.patch("recipes.speakrs.large.storage._load_wrangler_oauth_state", side_effect=states):
            with mock.patch(
                "recipes.speakrs.large.storage._wrangler_whoami",
                return_value=mock.Mock(returncode=0, stdout="{}"),
            ) as refresh:
                with mock.patch(
                    "recipes.speakrs.large.storage.urllib.request.urlopen",
                    side_effect=[denied, _FakeHttpResponse(200, [b"alpha"])],
                ) as urlopen:
                    self.assertEqual(list(backend.iter_bytes("k1")), [b"alpha"])
        self.assertEqual(urlopen.call_args_list[0].args[0].get_header("Authorization"), "Bearer token-a")
        self.assertEqual(urlopen.call_args_list[1].args[0].get_header("Authorization"), "Bearer token-b")
        refresh.assert_called_once()

    def test_provider_policy_evidence_is_bound_to_authenticated_bucket(self):
        from unittest import mock

        destination = ObjectStoreDestination(
            provider="r2",
            endpoint="https://example.invalid.r2.cloudflarestorage.com",
            bucket="praveen",
            prefix="datasets/diarization-data-verification",
            credential_reference="wrangler",
        )
        backend = WranglerR2Backend(destination, api_token="api-token", account_id="a" * 32)
        with mock.patch.object(backend, "exists", return_value=True):
            with mock.patch.object(
                backend,
                "_cloudflare_api_json",
                return_value={
                    "success": True,
                    "result": {
                        "name": "praveen",
                        "location": "WNAM",
                        "storage_class": "Standard",
                    },
                },
            ) as api_json:
                evidence = backend.provider_encryption_evidence("k1")
        self.assertEqual(evidence["provider"], "cloudflare-r2")
        self.assertEqual(evidence["policy"], "AES-256")
        self.assertEqual(evidence["bucket"]["name"], "praveen")
        api_json.assert_called_once_with("a" * 32, "api-token", resource="", action="bucket")

    def test_verify_and_commit_batch_counts_real_wrangler_requests(self):
        from unittest import mock

        backend = self._backend()
        payload = b"alpha"
        key = self._owned_key(payload)
        expected = [
            {
                "key": key,
                "sha256": sha256_bytes(payload),
                "size": len(payload),
                "purpose": "train-audio",
                "parent_id": "p",
                "source": "AMI",
                "version": "official",
                "codec": "flac",
            }
        ]
        remote = {key: payload}
        requests = []

        def response(payload_bytes: bytes, status: int = 200):
            return _FakeHttpResponse(status, [payload_bytes])

        def fake_urlopen(request, timeout=60):
            del timeout
            requests.append(request)
            parsed = urllib.parse.urlparse(request.full_url)
            method = request.get_method()
            path = parsed.path
            if "/objects/" in path and method == "GET":
                object_key_value = urllib.parse.unquote(path.split("/objects/", 1)[1])
                if object_key_value not in remote:
                    raise urllib.error.HTTPError(
                        request.full_url,
                        404,
                        "not found",
                        {},
                        io.BytesIO(b"missing"),
                    )
                return response(remote[object_key_value])
            if path.endswith("/objects") and method == "GET":
                body = {
                    "success": True,
                    "result": [{"key": object_key_value} for object_key_value in sorted(remote)],
                    "result_info": {"is_truncated": False},
                }
                return response(json.dumps(body).encode("utf-8"))
            if path.endswith("/r2/buckets/praveen") and method == "GET":
                return response(json.dumps({"success": True, "result": {"name": "praveen"}}).encode("utf-8"))
            if path.endswith("/domains/managed") and method == "GET":
                return response(json.dumps({"success": True, "result": {"enabled": False}}).encode("utf-8"))
            if path.endswith("/domains/custom") and method == "GET":
                return response(json.dumps({"success": True, "result": {"domains": []}}).encode("utf-8"))
            if path.startswith("/praveen") and method == "GET":
                return response(b"", status=403)
            if "/objects/" in path and method == "PUT":
                object_key_value = urllib.parse.unquote(path.split("/objects/", 1)[1])
                remote[object_key_value] = bytes(request.data or b"")
                return response(b"")
            raise AssertionError(f"unexpected request: {method} {request.full_url}")

        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", side_effect=fake_urlopen):
            result = verify_and_commit_batch(
                [{**expected[0], "state": ObjectState.UPLOADED.value}],
                state=BatchState.DRAFT,
                expected_objects=expected,
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix=self._PREFIX,
                privacy_prefix=self._PREFIX,
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )

        marker_key = result["marker"]["key"]
        authenticated_object_gets = [
            request
            for request in requests
            if request.get_method() == "GET"
            and request.get_header("Authorization")
            and "/objects/" in urllib.parse.urlparse(request.full_url).path
        ]
        marker_gets = [request for request in authenticated_object_gets if marker_key in request.full_url]
        object_gets = [request for request in authenticated_object_gets if key in request.full_url]
        self.assertEqual(len(object_gets), 1)
        self.assertEqual(len(marker_gets), 2)
        self.assertEqual(
            sum(
                request.get_method() == "GET" and urllib.parse.urlparse(request.full_url).path.endswith("/objects")
                for request in requests
            ),
            2,
        )
        self.assertEqual(sum(request.full_url.endswith("/r2/buckets/praveen") for request in requests), 2)
        self.assertEqual(sum("/domains/managed" in request.full_url for request in requests), 2)
        self.assertEqual(sum("/domains/custom" in request.full_url for request in requests), 2)
        self.assertEqual(
            sum(
                request.full_url.startswith("https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com")
                for request in requests
            ),
            4,
        )

    def test_provider_privacy_evidence_requires_disabled_managed_route_and_no_custom_domain(self):
        from unittest import mock

        destination = ObjectStoreDestination(
            provider="r2",
            endpoint="https://example.invalid.r2.cloudflarestorage.com",
            bucket="praveen",
            prefix="datasets/diarization-data-verification",
            credential_reference="wrangler",
        )
        backend = WranglerR2Backend(destination, api_token="api-token", account_id="a" * 32)
        responses = [
            {"success": True, "result": {"enabled": False}},
            {"success": True, "result": {"domains": []}},
        ]
        with mock.patch.object(backend, "_cloudflare_api_json", side_effect=responses) as api_json:
            evidence = backend.provider_privacy_evidence()
        self.assertFalse(evidence["managed_public_access_enabled"])
        self.assertEqual(evidence["custom_domains_count"], 0)
        self.assertEqual(
            [call.kwargs["resource"] for call in api_json.call_args_list], ["/domains/managed", "/domains/custom"]
        )

        with mock.patch.object(
            backend,
            "_cloudflare_api_json",
            return_value={"success": True, "result": {"enabled": True}},
        ):
            with self.assertRaises(PreparationError):
                backend.provider_privacy_evidence()

    def test_readback_and_commit_preserve_version_identity(self):
        backend = MemoryBackend()
        payload = b"versioned-audio"
        digest = sha256_bytes(payload)
        key = object_key("AMI", "official", "train", digest, "flac")
        backend.put_bytes(key, payload)
        verified = mark_readback_verified(
            {
                "key": key,
                "sha256": digest,
                "size": len(payload),
                "purpose": "train-audio",
                "parent_id": "p",
                "source": "AMI",
                "version": "official",
                "state": ObjectState.UPLOADED.value,
                "codec": "flac",
            },
            backend=backend,
            expected_sha256=digest,
        )
        self.assertEqual(verified["version"], "official")
        batch = commit_batch(
            [verified],
            state=BatchState.DRAFT,
            expected_objects=[verified],
            acceptance_sha256=_h("acceptance"),
            backend=backend,
            inventory_prefix="datasets",
            label_policy_id="label-policy",
            split_id="frozen",
            qa_policy_sha256=_h("qa"),
        )
        self.assertEqual(batch["objects"][0]["version"], "official")

    def test_list_prefix_uses_bounded_cloudflare_api_pagination(self):
        from unittest import mock

        destination = ObjectStoreDestination(
            provider="r2",
            endpoint="https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
            bucket="praveen",
            prefix="datasets/diarization-data-verification",
            credential_reference="wrangler",
        )
        backend = WranglerR2Backend(destination, api_token="api-token")
        responses = iter(
            (
                {
                    "success": True,
                    "result": [{"key": "datasets/example/a"}],
                    "result_info": {"is_truncated": True, "cursor": "cursor-1"},
                },
                {
                    "success": True,
                    "result": [{"key": "datasets/example/b"}],
                    "result_info": {"is_truncated": False},
                },
            )
        )
        requests: list[tuple[dict[str, list[str]], str | None]] = []

        class Response:
            status = 200

            def __init__(self, payload):
                self.body = json.dumps(payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _size=-1):
                body, self.body = self.body, b""
                return body

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 60)
            requests.append(
                (
                    urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query),
                    request.get_header("Authorization"),
                )
            )
            return Response(next(responses))

        with mock.patch("recipes.speakrs.large.storage.urllib.request.urlopen", side_effect=fake_urlopen):
            keys = backend.list_prefix("datasets/example", max_keys=3)

        self.assertEqual(keys, ["datasets/example/a", "datasets/example/b"])
        self.assertEqual(requests[0][0], {"prefix": ["datasets/example"], "per_page": ["3"]})
        self.assertEqual(requests[1][0], {"prefix": ["datasets/example"], "per_page": ["2"], "cursor": ["cursor-1"]})
        self.assertEqual({auth for _, auth in requests}, {"Bearer api-token"})


class OpenFileCheckTest(unittest.TestCase):
    def test_open_file_check_blocks_reported_processes(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.bin"
            path.write_bytes(b"task")
            completed = mock.Mock(returncode=0, stdout="p123\nf7\n", stderr="")
            with mock.patch("recipes.speakrs.large.storage.shutil.which", return_value="/usr/sbin/lsof"):
                with mock.patch("recipes.speakrs.large.storage.subprocess.run", return_value=completed):
                    with self.assertRaises(PreparationError) as raised:
                        assert_no_open_file_handles(path)
            self.assertIn("open file handles", str(raised.exception))
            self.assertEqual(raised.exception.details["open_pids"], [123])

    def test_open_file_check_fails_closed_when_unavailable_or_ambiguous(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.bin"
            path.write_bytes(b"task")
            with mock.patch("recipes.speakrs.large.storage.shutil.which", return_value=None):
                with self.assertRaises(UnresolvedInputError):
                    assert_no_open_file_handles(path)
            completed = mock.Mock(returncode=1, stdout="", stderr="permission denied")
            with mock.patch("recipes.speakrs.large.storage.shutil.which", return_value="/usr/sbin/lsof"):
                with mock.patch("recipes.speakrs.large.storage.subprocess.run", return_value=completed):
                    with self.assertRaises(UnresolvedInputError):
                        assert_no_open_file_handles(path)


class ConsumedSourceCleanupTest(unittest.TestCase):
    def _fixture(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        staging = root / "staging"
        source_path = staging / "ami" / "raw.wav"
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"raw-source")
        canonical_path = staging / "selections" / "AMI" / "p1.flac"
        source_sha256 = sha256_bytes(source_path.read_bytes())
        canonical_sha256 = sha256_bytes(b"canonical")
        label_sha256 = sha256_bytes(b"label")
        uem_sha256 = sha256_bytes(b"uem")
        transform_facts = {"kind": "canonical", "target": "lossless FLAC, 16 kHz, mono"}
        parent = {
            "schema": "speakrs-parent-preparation",
            "schema_version": 2,
            "parent_id": "p1",
            "source_audio": str(source_path),
            "source_sha256": source_sha256,
            "audio": {
                "path": str(canonical_path),
                "sha256": canonical_sha256,
                "sample_count": 1,
                "sample_rate": 16000,
                "channels": 1,
            },
            "rttm_path": str(staging / "p1.rttm"),
            "uem_path": str(staging / "p1.uem"),
            "label_sha256": label_sha256,
            "uem_sha256": uem_sha256,
            "transform": transform_facts,
            "transform_sha256": sha256_json(transform_facts),
        }
        transform = {"schema": "speakrs-source-transforms-v1", "source": "AMI", "parents": [parent]}
        transform_path = staging / "time-transform.json"
        write_json(transform_path, transform)
        portable_transform = {
            "schema": "speakrs-source-transforms-v1",
            "source": "AMI",
            "parents": [
                {
                    key: value
                    for key, value in parent.items()
                    if key not in {"source_audio", "rttm_path", "uem_path", "bounds"}
                }
            ],
        }
        portable_transform["parents"][0]["audio"] = {
            key: value for key, value in parent["audio"].items() if key != "path"
        }
        backend = MemoryBackend()
        planned = []
        references = {}
        for field, payload, extension in (
            ("audio", b"canonical", "flac"),
            ("rttm", b"label", "rttm"),
            ("uem", b"uem", "uem"),
        ):
            digest = sha256_bytes(payload)
            key = object_key("AMI", "official", "train", digest, extension)
            backend.put_bytes(key, payload)
            references[field] = {"key": key, "sha256": digest, "size": len(payload), "codec": extension}
            planned.append(
                {
                    "key": key,
                    "sha256": digest,
                    "size": len(payload),
                    "purpose": "train-audio" if field == "audio" else "train-label",
                    "parent_id": "p1",
                    "source": "AMI",
                    "version": "official",
                    "state": ObjectState.UPLOADED.value,
                    "codec": extension,
                }
            )
        acceptance_sha256 = sha256_bytes(b"acceptance")
        portable = {
            "schema": "speakrs-portable-training-selection-v1",
            "source": "AMI",
            "version": "official",
            "acceptance_sha256": acceptance_sha256,
            "evidence_hashes": {"time_transform": sha256_bytes(transform_path.read_bytes())},
            "provenance": {"time_transform": portable_transform},
            "recordings": [{"recording_id": "p1", **references}],
        }
        portable_path = root / "minimal-package.json"
        write_json(portable_path, portable)
        manifest_sha256 = sha256_bytes(portable_path.read_bytes())
        manifest_key = object_key("AMI", "official", "train", manifest_sha256, "json")
        backend.put_bytes(manifest_key, portable_path.read_bytes())
        planned.append(
            {
                "key": manifest_key,
                "sha256": manifest_sha256,
                "size": portable_path.stat().st_size,
                "purpose": "manifest",
                "parent_id": None,
                "source": "AMI",
                "version": "official",
                "state": ObjectState.UPLOADED.value,
                "codec": "json",
            }
        )
        verified = [mark_readback_verified(item, backend=backend, expected_sha256=item["sha256"]) for item in planned]
        batch = commit_batch(
            verified,
            state=BatchState.DRAFT,
            expected_objects=planned,
            acceptance_sha256=acceptance_sha256,
            backend=backend,
            inventory_prefix="datasets/AMI/official/train",
            label_policy_id="label-policy",
            split_id="frozen",
            qa_policy_sha256=sha256_bytes(b"qa"),
        )
        cache = root / "cache"
        cache.mkdir()
        restored = []
        for item in verified:
            restored_path = cache / Path(item["key"]).name
            restored_path.write_bytes(backend.get_bytes(item["key"]))
            restored.append(
                {
                    "key": item["key"],
                    "sha256": item["sha256"],
                    "size": item["size"],
                    "restored_path": str(restored_path),
                }
            )
        restore = {
            "schema": "speakrs-cold-restore-v1",
            "command": "restore-check",
            "ok": True,
            "acceptance_sha256": acceptance_sha256,
            "marker": batch["marker"],
            "objects": restored,
        }
        restore_path = root / "restore.json"
        write_json(restore_path, restore)
        return {
            "source_path": source_path,
            "staging": staging,
            "source_sha256": source_sha256,
            "transform_path": transform_path,
            "portable_path": portable_path,
            "verified": verified,
            "restore_path": restore_path,
            "backend": backend,
        }

    def _grouped_copies(self, fixture, count=2):
        restore = json.loads(fixture["restore_path"].read_text())
        restore_sha256 = sha256_file(fixture["restore_path"])
        marker = restore["marker"]
        item = fixture["verified"][0]
        restored = next(row for row in restore["objects"] if row["key"] == item["key"])
        copies = []
        for index in range(count):
            path = fixture["staging"] / f"grouped-copy-{index}.bin"
            path.write_bytes(fixture["backend"].get_bytes(item["key"]))
            proof = RemoteRestoreProof(
                object_key=item["key"],
                object_sha256=item["sha256"],
                object_size=item["size"],
                acceptance_sha256=restore["acceptance_sha256"],
                marker_key=marker["key"],
                marker_sha256=marker["sha256"],
                restore_receipt_sha256=restore_sha256,
                restored_sha256=restored["sha256"],
                restored_size=restored["size"],
                restore_receipt_path=fixture["restore_path"],
            )
            copies.append(
                LocalCopy(
                    path=path,
                    state=LocalCopyState.RETAINED,
                    source="AMI",
                    sha256=item["sha256"],
                    labels_accepted=True,
                    remote_proof=proof,
                )
            )
        return copies, item["key"]

    def test_grouped_eviction_reads_each_remote_object_once_for_duplicate_copies(self):
        fixture = self._fixture()
        copies, duplicate_key = self._grouped_copies(fixture)
        backend = fixture["backend"]
        restore = json.loads(fixture["restore_path"].read_text())
        counts = {}
        real_iter_bytes = backend.iter_bytes

        def counted_iter(key, *, chunk_size=1024 * 1024, max_bytes=8 * 1024**3):
            counts[key] = counts.get(key, 0) + 1
            yield from real_iter_bytes(key, chunk_size=chunk_size, max_bytes=max_bytes)

        backend.iter_bytes = counted_iter
        gone = evict_copies(copies, restore_receipt=fixture["restore_path"], backend=backend)
        self.assertEqual(len(gone), 2)
        expected_keys = {restore["marker"]["key"], *(item["key"] for item in fixture["verified"])}
        self.assertEqual({key: counts[key] for key in expected_keys}, dict.fromkeys(expected_keys, 1))
        self.assertEqual(counts[duplicate_key], 1)
        self.assertTrue(all(copy.state is LocalCopyState.EVICTED for copy in gone))

    def test_grouped_eviction_corrupt_remote_fails_before_local_deletion(self):
        fixture = self._fixture()
        copies, corrupt_key = self._grouped_copies(fixture)
        fixture["backend"].corrupt_get_keys.add(corrupt_key)
        with self.assertRaises(PreparationError):
            evict_copies(copies, restore_receipt=fixture["restore_path"], backend=fixture["backend"])
        self.assertTrue(all(copy.path.is_file() for copy in copies))

    def test_grouped_eviction_rejects_a_caller_fabricated_remote_proof(self):
        fixture = self._fixture()
        copies, key = self._grouped_copies(fixture)
        fake_digest = sha256_bytes(b"fabricated")
        real = copies[0].remote_proof
        assert isinstance(real, RemoteRestoreProof)
        copies[0].remote_proof = RemoteRestoreProof(
            object_key=key,
            object_sha256=fake_digest,
            object_size=len(b"fabricated"),
            acceptance_sha256=real.acceptance_sha256,
            marker_key=real.marker_key,
            marker_sha256=real.marker_sha256,
            restore_receipt_sha256=real.restore_receipt_sha256,
            restored_sha256=fake_digest,
            restored_size=len(b"fabricated"),
            restore_receipt_path=real.restore_receipt_path,
        )
        with self.assertRaises(PreparationError):
            evict_copies(copies, restore_receipt=fixture["restore_path"], backend=fixture["backend"])
        self.assertTrue(all(copy.path.is_file() for copy in copies))

    def test_grouped_eviction_rechecks_each_local_copy_before_delete(self):
        from unittest import mock

        fixture = self._fixture()
        copies, _ = self._grouped_copies(fixture)
        first, second = copies
        real_persist = __import__(
            "recipes.speakrs.large.storage", fromlist=["_persist_deletion_intent"]
        )._persist_deletion_intent
        calls = {"count": 0}

        def mutate_after_first_intent(intent):
            result = real_persist(intent)
            calls["count"] += 1
            if calls["count"] == 1:
                second.path.write_bytes(b"mutated")
            return result

        with mock.patch(
            "recipes.speakrs.large.storage._persist_deletion_intent",
            side_effect=mutate_after_first_intent,
        ):
            with self.assertRaises(PreparationError) as raised:
                evict_copies(copies, restore_receipt=fixture["restore_path"], backend=fixture["backend"])
        self.assertIn("changed after remote proof", str(raised.exception))
        self.assertFalse(first.path.exists())
        self.assertTrue(second.path.is_file())

    def test_discard_requires_complete_transform_and_restore_proof(self):
        fixture = self._fixture()
        request = ConsumedSource(
            path=fixture["source_path"],
            parent_id="p1",
            source_sha256=fixture["source_sha256"],
            task_root=fixture["staging"],
        )
        deletion = discard_consumed_source(
            request,
            transform_receipt=fixture["transform_path"],
            portable_manifest=fixture["portable_path"],
            accepted_outputs=fixture["verified"],
            restore_receipt=fixture["restore_path"],
            backend=fixture["backend"],
        )
        self.assertFalse(fixture["source_path"].exists())
        self.assertEqual(deletion["state"], "consumed-source")
        self.assertTrue(deletion["deleted"])
        self.assertEqual(deletion["parent_id"], "p1")
        self.assertEqual(len(deletion["accepted_outputs"]), 4)

    def test_grouped_discard_requires_intact_restored_files_before_raw_deletion(self):
        for mutation in ("missing", "corrupt", "symlink"):
            with self.subTest(mutation=mutation):
                fixture = self._fixture()
                restore = json.loads(fixture["restore_path"].read_text())
                restored_path = Path(restore["objects"][0]["restored_path"])
                if mutation == "missing":
                    restored_path.unlink()
                elif mutation == "corrupt":
                    restored_path.write_bytes(b"changed")
                else:
                    restored_path.unlink()
                    restored_path.symlink_to(fixture["source_path"])
                request = ConsumedSource(
                    path=fixture["source_path"],
                    parent_id="p1",
                    source_sha256=fixture["source_sha256"],
                    task_root=fixture["staging"],
                )
                with self.assertRaises(PreparationError):
                    discard_consumed_sources(
                        [request],
                        transform_receipt=fixture["transform_path"],
                        portable_manifest=fixture["portable_path"],
                        accepted_outputs=fixture["verified"],
                        restore_receipt=fixture["restore_path"],
                        backend=fixture["backend"],
                    )
                self.assertTrue(fixture["source_path"].is_file())

    def test_discard_rejects_wrong_source_identity_and_active_reader(self):
        fixture = self._fixture()
        request = ConsumedSource(
            path=fixture["source_path"],
            parent_id="p1",
            source_sha256=sha256_bytes(b"different"),
            task_root=fixture["staging"],
        )
        with self.assertRaises(PreparationError):
            discard_consumed_source(
                request,
                transform_receipt=fixture["transform_path"],
                portable_manifest=fixture["portable_path"],
                accepted_outputs=fixture["verified"],
                restore_receipt=fixture["restore_path"],
                backend=fixture["backend"],
            )
        request = ConsumedSource(
            path=fixture["source_path"],
            parent_id="p1",
            source_sha256=fixture["source_sha256"],
            task_root=fixture["staging"],
            live_readers=1,
        )
        with self.assertRaises(PreparationError):
            discard_consumed_source(
                request,
                transform_receipt=fixture["transform_path"],
                portable_manifest=fixture["portable_path"],
                accepted_outputs=fixture["verified"],
                restore_receipt=fixture["restore_path"],
                backend=fixture["backend"],
            )
        self.assertTrue(fixture["source_path"].exists())

    def test_discard_resumes_from_durable_intent_after_completion_interrupt(self):
        from unittest import mock

        fixture = self._fixture()
        request = ConsumedSource(
            path=fixture["source_path"],
            parent_id="p1",
            source_sha256=fixture["source_sha256"],
            task_root=fixture["staging"],
        )
        with mock.patch(
            "recipes.speakrs.large.storage._persist_deletion_completion",
            side_effect=RuntimeError("simulated process stop"),
        ):
            with self.assertRaises(RuntimeError):
                discard_consumed_source(
                    request,
                    transform_receipt=fixture["transform_path"],
                    portable_manifest=fixture["portable_path"],
                    accepted_outputs=fixture["verified"],
                    restore_receipt=fixture["restore_path"],
                    backend=fixture["backend"],
                )

        self.assertFalse(fixture["source_path"].exists())
        resumed = discard_consumed_source(
            request,
            transform_receipt=fixture["transform_path"],
            portable_manifest=fixture["portable_path"],
            accepted_outputs=fixture["verified"],
            restore_receipt=fixture["restore_path"],
            backend=fixture["backend"],
        )
        self.assertTrue(resumed["deleted"])
        self.assertEqual(resumed["state"], "consumed-source")


class UploadProofTest(unittest.TestCase):
    def test_upload_success_head_and_etag_cannot_enable_eviction(self):
        with self.assertRaises(PreparationError):
            upload_success_is_not_proof({"ok": True})
        with self.assertRaises(PreparationError) as raised:
            upload_success_is_not_proof({"etag": "abc"})
        self.assertIn("ETag", str(raised.exception))
        with self.assertRaises(PreparationError) as raised:
            upload_success_is_not_proof({"head": {"size": "12"}})
        self.assertIn("HEAD", str(raised.exception))


class _CountingMemoryBackend(MemoryBackend):
    def __init__(self):
        super().__init__()
        self.read_keys: list[str] = []
        self.head_keys: list[str] = []
        self.anonymous_get_keys: list[str] = []
        self.anonymous_list_prefixes: list[str] = []

    def iter_bytes(self, key, *, chunk_size=1024 * 1024, max_bytes=8 * 1024**3):
        self.read_keys.append(key)
        yield from super().iter_bytes(key, chunk_size=chunk_size, max_bytes=max_bytes)

    def head(self, key):
        self.head_keys.append(key)
        return super().head(key)

    def anonymous_get(self, key):
        self.anonymous_get_keys.append(key)
        return super().anonymous_get(key)

    def anonymous_list(self, prefix):
        self.anonymous_list_prefixes.append(prefix)
        return super().anonymous_list(prefix)


class VerifyAndCommitBatchTest(unittest.TestCase):
    def _objects(self):
        payloads = {"a": b"audio-a", "b": b"audio-b"}
        expected = []
        for parent_id, payload in payloads.items():
            digest = sha256_bytes(payload)
            expected.append(
                {
                    "key": object_key("AMI", "official", "train", digest, "flac"),
                    "sha256": digest,
                    "size": len(payload),
                    "purpose": "train-audio",
                    "parent_id": parent_id,
                    "source": "AMI",
                    "version": "official",
                    "codec": "flac",
                }
            )
        return expected, payloads

    def test_owner_reads_each_data_object_once_and_marker_separately(self):
        backend = _CountingMemoryBackend()
        expected, payloads = self._objects()
        for item, payload in zip(expected, payloads.values()):
            backend.put_bytes(item["key"], payload)
        uploaded = [{**item, "state": ObjectState.UPLOADED.value} for item in expected]

        batch = verify_and_commit_batch(
            uploaded,
            state=BatchState.DRAFT,
            expected_objects=expected,
            acceptance_sha256=_h("acceptance"),
            backend=backend,
            inventory_prefix="datasets/AMI/official/train",
            privacy_prefix="datasets",
            label_policy_id="label-policy",
            split_id="frozen",
            qa_policy_sha256=_h("qa"),
        )

        data_keys = [item["key"] for item in expected]
        self.assertEqual(backend.read_keys.count(data_keys[0]), 1)
        self.assertEqual(backend.read_keys.count(data_keys[1]), 1)
        marker_key = batch["marker"]["key"]
        self.assertEqual(backend.read_keys.count(marker_key), 1)
        self.assertEqual(backend.head_keys, [*sorted(data_keys), marker_key])
        self.assertEqual(backend.anonymous_get_keys.count(data_keys[0]), 1)
        self.assertEqual(backend.anonymous_get_keys.count(data_keys[1]), 1)
        self.assertEqual(backend.anonymous_get_keys.count(marker_key), 1)
        self.assertEqual(backend.anonymous_list_prefixes, ["datasets", "datasets"])
        self.assertEqual([item["version"] for item in batch["objects"]], ["official", "official"])

    def test_owner_rejects_forged_verified_state_before_remote_reads(self):
        backend = _CountingMemoryBackend()
        expected, payloads = self._objects()
        for item, payload in zip(expected, payloads.values()):
            backend.put_bytes(item["key"], payload)
        forged = [{**item, "state": ObjectState.READBACK_VERIFIED.value} for item in expected]

        with self.assertRaises(PreparationError):
            verify_and_commit_batch(
                forged,
                state=BatchState.DRAFT,
                expected_objects=expected,
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets/AMI/official/train",
                privacy_prefix="datasets",
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )
        self.assertEqual(backend.read_keys, [])
        self.assertFalse(any("/_commits/" in key for key in backend.objects))

    def test_corrupt_object_fails_before_marker_publication(self):
        backend = _CountingMemoryBackend()
        expected, payloads = self._objects()
        for item, payload in zip(expected, payloads.values()):
            backend.put_bytes(item["key"], payload)
        backend.corrupt_get_keys.add(expected[1]["key"])
        uploaded = [{**item, "state": ObjectState.UPLOADED.value} for item in expected]

        with self.assertRaises(PreparationError):
            verify_and_commit_batch(
                uploaded,
                state=BatchState.DRAFT,
                expected_objects=expected,
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets/AMI/official/train",
                privacy_prefix="datasets",
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )
        self.assertEqual([key for key in backend.objects if "/_commits/" in key], [])

    def test_owner_rejects_missing_encryption_and_public_object(self):
        backend = _CountingMemoryBackend()
        backend.encryption_default = None
        expected, payloads = self._objects()
        for item, payload in zip(expected, payloads.values()):
            backend.put_bytes(item["key"], payload)
        uploaded = [{**item, "state": ObjectState.UPLOADED.value} for item in expected]

        with self.assertRaises(PreparationError):
            verify_and_commit_batch(
                uploaded,
                state=BatchState.DRAFT,
                expected_objects=expected,
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets/AMI/official/train",
                privacy_prefix="datasets",
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )
        self.assertFalse(any("/_commits/" in key for key in backend.objects))

        backend = _CountingMemoryBackend()
        for item, payload in zip(expected, payloads.values()):
            backend.put_bytes(item["key"], payload)
        backend.objects[expected[0]["key"]].public = True
        with self.assertRaises(PreparationError):
            verify_and_commit_batch(
                uploaded,
                state=BatchState.DRAFT,
                expected_objects=expected,
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets/AMI/official/train",
                privacy_prefix="datasets",
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )
        self.assertFalse(any("/_commits/" in key for key in backend.objects))

    def test_public_commit_rechecks_forged_readback_verified_rows(self):
        backend = _CountingMemoryBackend()
        expected, payloads = self._objects()
        for item, payload in zip(expected, payloads.values()):
            backend.put_bytes(item["key"], payload)
        forged = [
            {
                **item,
                "state": ObjectState.READBACK_VERIFIED.value,
                "encryption": "AES256",
                "public": False,
            }
            for item in expected
        ]
        backend.corrupt_get_keys.add(expected[0]["key"])

        with self.assertRaises(PreparationError):
            commit_batch(
                forged,
                state=BatchState.DRAFT,
                expected_objects=expected,
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets/AMI/official/train",
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )
        self.assertEqual(backend.read_keys.count(expected[0]["key"]), 1)
        self.assertFalse(any("/_commits/" in key for key in backend.objects))

    def test_owner_rejects_duplicate_and_mismatched_inventory_before_readback(self):
        backend = _CountingMemoryBackend()
        expected, payloads = self._objects()
        for item, payload in zip(expected, payloads.values()):
            backend.put_bytes(item["key"], payload)
        uploaded = [{**item, "state": ObjectState.UPLOADED.value} for item in expected]

        with self.assertRaises(PreparationError):
            verify_and_commit_batch(
                [uploaded[0], uploaded[0]],
                state=BatchState.DRAFT,
                expected_objects=expected,
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets/AMI/official/train",
                privacy_prefix="datasets",
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )
        self.assertEqual(backend.read_keys, [])

        mismatch = [{**uploaded[0], "size": uploaded[0]["size"] + 1}, uploaded[1]]
        with self.assertRaises(PreparationError):
            verify_and_commit_batch(
                mismatch,
                state=BatchState.DRAFT,
                expected_objects=expected,
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets/AMI/official/train",
                privacy_prefix="datasets",
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )
        self.assertEqual(backend.read_keys, [])

    def test_public_privacy_wrapper_uses_its_own_single_readback(self):
        backend = _CountingMemoryBackend()
        expected, payloads = self._objects()
        backend.put_bytes(expected[0]["key"], payloads["a"])

        evidence = assert_private_access(backend, expected[0]["key"], "datasets")

        self.assertEqual(evidence["authenticated_sha256"], expected[0]["sha256"])
        self.assertEqual(backend.read_keys, [expected[0]["key"]])
        self.assertEqual(backend.anonymous_get_keys, [expected[0]["key"]])
        self.assertEqual(backend.anonymous_list_prefixes, ["datasets"])

    def test_partial_corrupt_missing_cannot_commit(self):
        backend = MemoryBackend()
        payload = b"hello-audio"
        digest = sha256_bytes(payload)
        key = object_key("AMI", "official", "train", digest, "flac")
        backend.put_bytes(key, payload)
        backend.corrupt_get_keys.add(key)
        with self.assertRaises(PreparationError) as raised:
            full_readback_sha256(backend, key, digest)
        self.assertIn("cannot commit", str(raised.exception))
        with self.assertRaises(PreparationError):
            full_readback_sha256(backend, "missing-key", digest)
        planned = {
            "key": key,
            "sha256": digest,
            "size": len(payload),
            "purpose": "train-audio",
            "parent_id": "ES2002a",
            "source": "AMI",
            "state": ObjectState.UPLOADED.value,
            "codec": "flac",
        }
        backend.corrupt_get_keys.clear()
        verified = mark_readback_verified(planned, backend=backend, expected_sha256=digest)
        with self.assertRaises(PreparationError):
            commit_batch(
                [verified],
                state=BatchState.DRAFT,
                expected_objects=[verified, {"key": "datasets/AMI/missing", "sha256": _h("missing"), "size": 1}],
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets",
                label_policy_id="label-qa-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )

    def test_same_key_different_content_fails(self):
        backend = MemoryBackend()
        digest = _h("body-a")
        key = object_key("AMI", "official", "train", digest, "flac")
        backend.put_bytes(key, b"body-a")
        with self.assertRaises(PreparationError) as raised:
            backend.put_bytes(key, b"body-b")
        self.assertIn("same-key different-content", str(raised.exception))

    def test_public_anonymous_and_missing_encryption_fail(self):
        backend = MemoryBackend()
        payload = b"private"
        digest = sha256_bytes(payload)
        key = object_key("AMI", "official", "train", digest, "flac")
        backend.put_bytes(key, payload)
        backend.objects[key].public = True
        with self.assertRaises(PreparationError) as raised:
            assert_private_access(backend, key, "datasets/")
        self.assertIn("anonymous", str(raised.exception).lower())
        backend.objects[key].public = False
        backend.encryption_default = None
        backend.objects[key].encryption = None
        planned = {
            "key": key,
            "sha256": digest,
            "size": len(payload),
            "purpose": "train-audio",
            "parent_id": "ES2002a",
            "source": "AMI",
            "state": ObjectState.UPLOADED.value,
            "codec": "flac",
        }
        with self.assertRaises(PreparationError) as raised:
            mark_readback_verified(planned, backend=backend, expected_sha256=digest)
        self.assertIn("encryption", str(raised.exception))

    def test_expired_credentials_fail(self):
        backend = MemoryBackend(expired=True)
        with self.assertRaises(PreparationError) as raised:
            backend.put_bytes("k", b"x")
        self.assertIn("expired", str(raised.exception))

    def test_etag_is_not_sha256(self):
        with self.assertRaises(ContractError):
            parse_object_receipt(
                {
                    "key": "datasets/AMI/x.flac",
                    "sha256": "a" * 64,
                    "size": 4,
                    "purpose": "train-audio",
                    "parent_id": "p",
                    "source": "AMI",
                    "state": ObjectState.READBACK_VERIFIED.value,
                    "codec": "flac",
                    "etag": "a" * 64,
                    "encryption": "AES256",
                    "public": False,
                }
            )


class InterruptAndCapTest(unittest.TestCase):
    def test_interrupted_upload_does_not_false_seal(self):
        backend = MemoryBackend(put_interrupt_after=0)
        digest = _h("payload")
        key = object_key("AMI", "official", "train", digest, "flac")
        with self.assertRaises(PreparationError):
            backend.put_bytes(key, b"payload")
        self.assertFalse(backend.exists(key))
        backend.put_interrupt_after = None
        backend.put_bytes(key, b"payload")
        self.assertEqual(backend.get_bytes(key), b"payload")

    def test_interrupted_readback_preserves_uploaded_not_verified(self):
        backend = MemoryBackend()
        payload = b"abc123"
        digest = sha256_bytes(payload)
        key = object_key("AMI", "official", "train", digest, "flac")
        backend.put_bytes(key, payload)
        backend.fail_get_keys.add(key)
        planned = {
            "key": key,
            "sha256": digest,
            "size": len(payload),
            "purpose": "train-audio",
            "parent_id": "p",
            "source": "AMI",
            "state": ObjectState.UPLOADED.value,
            "codec": "flac",
        }
        with self.assertRaises(PreparationError):
            mark_readback_verified(planned, backend=backend, expected_sha256=digest)
        self.assertTrue(backend.exists(key))
        backend.fail_get_keys.clear()
        verified = mark_readback_verified(planned, backend=backend, expected_sha256=digest)
        self.assertEqual(verified["state"], ObjectState.READBACK_VERIFIED.value)

    def test_cap_exhaustion_and_live_readers_block_deletion(self):
        with self.assertRaises(PreparationError) as raised:
            enforce_cap(10, 4, "staging")
        self.assertIn("cap exhausted", str(raised.exception))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "keep.bin"
            path.write_bytes(b"task")
            copy = LocalCopy(
                path=path,
                state=LocalCopyState.RETAINED,
                source="AMI",
                sha256=_h("task"),
                live_readers=1,
                task_owned=True,
                labels_accepted=True,
                remote_verified=True,
            )
            with self.assertRaises(PreparationError) as raised:
                mark_eviction_eligible(copy)
            self.assertIn("live readers", str(raised.exception))
            self.assertTrue(path.exists())
            foreign = LocalCopy(
                path=Path(temporary) / "other.bin",
                state=LocalCopyState.RETAINED,
                source="AMI",
                sha256=_h("other"),
                live_readers=0,
                task_owned=False,
                labels_accepted=True,
                remote_verified=True,
            )
            with self.assertRaises(PreparationError):
                mark_eviction_eligible(foreign)

    def test_interrupted_eviction_keeps_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.bin"
            path.write_bytes(b"task")
            backend = MemoryBackend()
            digest = sha256_bytes(path.read_bytes())
            key = object_key("AMI", "official", "train", digest, "bin")
            backend.put_bytes(key, path.read_bytes())
            verified = mark_readback_verified(
                {
                    "key": key,
                    "sha256": digest,
                    "size": path.stat().st_size,
                    "purpose": "train-audio",
                    "parent_id": "p",
                    "source": "AMI",
                    "state": ObjectState.UPLOADED.value,
                    "codec": "bin",
                },
                backend=backend,
                expected_sha256=digest,
            )
            batch = commit_batch(
                [verified],
                state=BatchState.DRAFT,
                expected_objects=[verified],
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets",
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )
            restore_receipt_path = Path(temporary) / "restore.json"
            write_json(
                restore_receipt_path,
                {
                    "schema": "speakrs-cold-restore-v1",
                    "command": "restore-check",
                    "ok": True,
                    "acceptance_sha256": _h("acceptance"),
                    "marker": batch["marker"],
                    "objects": [verified],
                    "samples": [{"recording_id": "p", "finite": True}],
                },
            )
            restore_receipt_sha256 = sha256_bytes(restore_receipt_path.read_bytes())
            proof = RemoteRestoreProof(
                object_key=key,
                object_sha256=digest,
                object_size=path.stat().st_size,
                acceptance_sha256=_h("acceptance"),
                marker_key=batch["marker"]["key"],
                marker_sha256=batch["marker"]["sha256"],
                restore_receipt_sha256=restore_receipt_sha256,
                restored_sha256=digest,
                restored_size=path.stat().st_size,
                restore_receipt_path=restore_receipt_path,
            )
            copy = LocalCopy(
                path=path,
                state=LocalCopyState.RETAINED,
                source="AMI",
                sha256=digest,
                labels_accepted=True,
                remote_verified=True,
                remote_proof=proof,
            )
            eligible = mark_eviction_eligible(copy, backend=backend)
            with self.assertRaises(PreparationError):
                evict_copy(eligible, preserve_on_interrupt=True)
            self.assertTrue(path.exists())
            gone = evict_copy(eligible, backend=backend)
            self.assertEqual(gone.state, LocalCopyState.EVICTED)
            self.assertFalse(path.exists())

    def test_eviction_resumes_from_durable_intent_after_completion_interrupt(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.bin"
            path.write_bytes(b"task")
            backend = MemoryBackend()
            digest = sha256_bytes(path.read_bytes())
            key = object_key("AMI", "official", "train", digest, "bin")
            backend.put_bytes(key, path.read_bytes())
            verified = mark_readback_verified(
                {
                    "key": key,
                    "sha256": digest,
                    "size": path.stat().st_size,
                    "purpose": "train-audio",
                    "parent_id": "p",
                    "source": "AMI",
                    "state": ObjectState.UPLOADED.value,
                    "codec": "bin",
                },
                backend=backend,
                expected_sha256=digest,
            )
            batch = commit_batch(
                [verified],
                state=BatchState.DRAFT,
                expected_objects=[verified],
                acceptance_sha256=_h("acceptance"),
                backend=backend,
                inventory_prefix="datasets",
                label_policy_id="label-policy",
                split_id="frozen",
                qa_policy_sha256=_h("qa"),
            )
            restore_receipt_path = Path(temporary) / "restore.json"
            write_json(
                restore_receipt_path,
                {
                    "schema": "speakrs-cold-restore-v1",
                    "command": "restore-check",
                    "ok": True,
                    "acceptance_sha256": _h("acceptance"),
                    "marker": batch["marker"],
                    "objects": [verified],
                    "samples": [{"recording_id": "p", "finite": True}],
                },
            )
            restore_receipt_sha256 = sha256_bytes(restore_receipt_path.read_bytes())
            proof = RemoteRestoreProof(
                object_key=key,
                object_sha256=digest,
                object_size=path.stat().st_size,
                acceptance_sha256=_h("acceptance"),
                marker_key=batch["marker"]["key"],
                marker_sha256=batch["marker"]["sha256"],
                restore_receipt_sha256=restore_receipt_sha256,
                restored_sha256=digest,
                restored_size=path.stat().st_size,
                restore_receipt_path=restore_receipt_path,
            )
            copy = LocalCopy(
                path=path,
                state=LocalCopyState.RETAINED,
                source="AMI",
                sha256=digest,
                labels_accepted=True,
                remote_verified=True,
                remote_proof=proof,
            )
            eligible = mark_eviction_eligible(copy, backend=backend)
            with mock.patch(
                "recipes.speakrs.large.storage._persist_deletion_completion",
                side_effect=RuntimeError("simulated process stop"),
            ):
                with self.assertRaises(RuntimeError):
                    evict_copy(eligible, backend=backend)

            self.assertFalse(path.exists())
            recovered = recover_deletion_journal(
                restore_receipt_path,
                operation="eviction",
                path=path,
                expected_identity={
                    "copy_sha256": digest,
                    "object_key": key,
                    "object_sha256": digest,
                    "object_size": proof.object_size,
                    "acceptance_sha256": _h("acceptance"),
                    "marker_key": batch["marker"]["key"],
                    "marker_sha256": batch["marker"]["sha256"],
                    "restore_receipt_sha256": restore_receipt_sha256,
                },
            )
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["state"], "intent")
            resumed = evict_copy(eligible, backend=backend)
            self.assertEqual(resumed.state, LocalCopyState.EVICTED)
            self.assertTrue(resumed.deletion_receipt["deleted"])

    def test_batch_is_not_a_complete_release(self):
        backend = MemoryBackend()
        payload = b"audio"
        digest = sha256_bytes(payload)
        key = object_key("AMI", "official", "train", digest, "flac")
        backend.put_bytes(key, payload)
        verified = mark_readback_verified(
            {
                "key": key,
                "sha256": digest,
                "size": len(payload),
                "purpose": "train-audio",
                "parent_id": "p",
                "source": "AMI",
                "state": ObjectState.UPLOADED.value,
                "codec": "flac",
            },
            backend=backend,
            expected_sha256=digest,
        )
        batch = commit_batch(
            [verified],
            state=BatchState.DRAFT,
            expected_objects=[verified],
            acceptance_sha256=_h("acceptance"),
            backend=backend,
            inventory_prefix="datasets",
            label_policy_id="label-qa-policy",
            split_id="frozen",
            qa_policy_sha256=_h("qa"),
        )
        self.assertEqual(batch["state"], BatchState.COMMITTED.value)
        with self.assertRaises(UnresolvedInputError):
            commit_release(
                [batch],
                required_sources=["AMI", "AliMeeting"],
                state=RemoteReleaseState.DRAFT,
            )

    def test_release_commit_reads_each_object_once(self):
        from unittest import mock

        backend = MemoryBackend()
        payload = b"audio"
        digest = sha256_bytes(payload)
        key = object_key("AMI", "official", "train", digest, "flac")
        backend.put_bytes(key, payload)
        verified = mark_readback_verified(
            {
                "key": key,
                "sha256": digest,
                "size": len(payload),
                "purpose": "train-audio",
                "parent_id": "p",
                "source": "AMI",
                "state": ObjectState.UPLOADED.value,
                "codec": "flac",
            },
            backend=backend,
            expected_sha256=digest,
        )
        batch = commit_batch(
            [verified],
            state=BatchState.DRAFT,
            expected_objects=[verified],
            acceptance_sha256=_h("acceptance"),
            backend=backend,
            inventory_prefix="datasets",
            label_policy_id="label-qa-policy",
            split_id="frozen",
            qa_policy_sha256=_h("qa"),
        )
        backend.put_bytes("datasets/_canary/owner-probe.bin", b"speakrs-diarization-r2-owner-canary")
        backend.put_bytes("datasets/_canary/probe.txt", b"speakrs-diarization-r2-canary")

        with mock.patch.object(backend, "iter_bytes", wraps=backend.iter_bytes) as iter_bytes:
            result = commit_release(
                [batch],
                required_sources=["AMI"],
                state=RemoteReleaseState.DRAFT,
                backend=backend,
                inventory_prefix="datasets",
                required_objects=batch["objects"],
                release_identity={
                    "required_batches": {batch["batch_sha256"]: batch["acceptance_sha256"]},
                },
            )

        self.assertEqual(result["state"], RemoteReleaseState.COMMITTED.value)
        object_reads = [call for call in iter_bytes.call_args_list if call.args[0] == key]
        self.assertEqual(len(object_reads), 1)
        for relative, payload in (
            ("owner-probe.bin", b"speakrs-diarization-r2-owner-canary"),
            ("probe.txt", b"speakrs-diarization-r2-canary"),
        ):
            canary_key = f"datasets/_canary/{relative}"
            canary_reads = [call for call in iter_bytes.call_args_list if call.args[0] == canary_key]
            self.assertEqual(len(canary_reads), 1)
            self.assertEqual(canary_reads[0].kwargs["max_bytes"], len(payload))

        backend.put_bytes("datasets/_canary/unknown.bin", b"unknown")
        with self.assertRaisesRegex(PreparationError, "inventory is not an exact accepted-selection closure"):
            commit_release(
                [batch],
                required_sources=["AMI"],
                state=RemoteReleaseState.DRAFT,
                backend=backend,
                inventory_prefix="datasets",
                required_objects=batch["objects"],
                release_identity={
                    "required_batches": {batch["batch_sha256"]: batch["acceptance_sha256"]},
                },
            )

    def test_release_commit_rejects_changed_control_canary(self):
        backend = MemoryBackend()
        payload = b"audio"
        digest = sha256_bytes(payload)
        key = object_key("AMI", "official", "train", digest, "flac")
        backend.put_bytes(key, payload)
        verified = mark_readback_verified(
            {
                "key": key,
                "sha256": digest,
                "size": len(payload),
                "purpose": "train-audio",
                "parent_id": "p",
                "source": "AMI",
                "state": ObjectState.UPLOADED.value,
                "codec": "flac",
            },
            backend=backend,
            expected_sha256=digest,
        )
        batch = commit_batch(
            [verified],
            state=BatchState.DRAFT,
            expected_objects=[verified],
            acceptance_sha256=_h("acceptance"),
            backend=backend,
            inventory_prefix="datasets",
            label_policy_id="label-qa-policy",
            split_id="frozen",
            qa_policy_sha256=_h("qa"),
        )
        expected_canary = b"speakrs-diarization-r2-canary"
        backend.put_bytes("datasets/_canary/probe.txt", b"x" * len(expected_canary))

        with self.assertRaisesRegex(PreparationError, "partial/corrupt/missing remote content"):
            commit_release(
                [batch],
                required_sources=["AMI"],
                state=RemoteReleaseState.DRAFT,
                backend=backend,
                inventory_prefix="datasets",
                required_objects=batch["objects"],
                release_identity={
                    "required_batches": {batch["batch_sha256"]: batch["acceptance_sha256"]},
                },
            )

    def test_generic_backend_uses_its_encryption_evidence_contract(self):
        class ProviderAttestedBackend(MemoryBackend):
            def head(self, key):
                return {name: value for name, value in super().head(key).items() if name != "encryption"}

            def encryption_evidence(self, key):
                return "provider-attested"

        backend = ProviderAttestedBackend()
        payload = b"audio"
        digest = sha256_bytes(payload)
        key = object_key("AMI", "official", "train", digest, "flac")
        backend.put_bytes(key, payload)

        verified = mark_readback_verified(
            {
                "key": key,
                "sha256": digest,
                "size": len(payload),
                "purpose": "train-audio",
                "parent_id": "p",
                "source": "AMI",
                "state": ObjectState.UPLOADED.value,
                "codec": "flac",
            },
            backend=backend,
            expected_sha256=digest,
        )

        self.assertEqual(verified["encryption"], "provider-attested")


class RestoreReaderTest(unittest.TestCase):
    def test_caller_samples_cannot_replace_committed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = parse_data_preparation_spec(_data_spec_payload(root))
            with self.assertRaisesRegex(PreparationError, "committed portable manifest"):
                restore_check(
                    spec,
                    root / "fabricated-receipt.json",
                    root / "restore.json",
                    backend=MemoryBackend(),
                    samples=[{"recording_id": "rec", "source_path": str(root / "arbitrary.wav")}],
                )


if __name__ == "__main__":
    unittest.main()
