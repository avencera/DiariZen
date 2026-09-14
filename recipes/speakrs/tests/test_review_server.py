"""Protect the localhost review API, media policy, and bundled UI."""

from __future__ import annotations

import json
import subprocess
from http.client import HTTPConnection
from pathlib import Path

from recipes.speakrs.large.review_overlay import prepare_review_overlay
from recipes.speakrs.large.review_server import UI_ROOT, start_review_server
from recipes.speakrs.large.review_store import open_review_store
from recipes.speakrs.tests.test_review_overlay import _build_packet


def _start(tmp_path: Path, **kwargs):
    archive, packet = _build_packet(tmp_path)
    session = tmp_path / "session"
    prepare_review_overlay(packet, archive, session)
    server, thread, host, port = start_review_server(session, actor_id="rev-1", **kwargs)
    return server, session, host, port


def _request(
    host: str,
    port: int,
    method: str,
    path: str,
    *,
    origin: str | None = None,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    host_header: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection(host, port, timeout=10)
    extra = dict(headers or {})
    if origin:
        extra["Origin"] = origin
    extra["Host"] = host_header or f"{host}:{port}"
    if body is not None:
        extra.setdefault("Content-Type", "application/json")
        extra["Content-Length"] = str(len(body))
    connection.request(method, path, body=body, headers=extra)
    response = connection.getresponse()
    payload = response.read()
    header_map = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, header_map, payload


def test_origin_headers_and_path_policy(tmp_path: Path) -> None:
    server, session, host, port = _start(tmp_path)
    try:
        status, headers, body = _request(host, port, "GET", "/api/v1/session")
        assert status == 200
        assert headers["cache-control"] == "no-store"
        assert "default-src 'self'" in headers["content-security-policy"]
        assert "set-cookie" not in headers
        session_payload = json.loads(body)
        window_id = session_payload["windows"][0]["window_id"]
        status, _, _ = _request(
            host,
            port,
            "POST",
            f"/api/v1/windows/{window_id}/events",
            origin="http://example.com",
            body=b"{}",
        )
        assert status == 403
        status, _, _ = _request(host, port, "GET", "/api/v1/windows/../secret")
        assert status in {400, 404}
        status, _, _ = _request(
            host,
            port,
            "GET",
            f"/api/v1/windows/{window_id}/media/emitted",
            headers={"Range": "bytes=0-15"},
        )
        assert status == 206
        status, _, body = _request(host, port, "GET", "/")
        assert status == 200
        assert b"Confirm" in body
        status, _, _ = _request(
            host,
            port,
            "POST",
            f"/api/v1/windows/{window_id}/events",
            origin=f"http://{host}:{port}",
            body=b"x" * (2 * 1024 * 1024),
        )
        assert status == 413
        wav_status, _, wav_body = _request(
            host,
            port,
            "GET",
            f"/api/v1/windows/{window_id}/media/emitted?container=wav",
        )
        assert wav_status == 200
        assert wav_body[:4] == b"RIFF"
    finally:
        server.shutdown()
        server.server_close()


def test_read_only_disables_writes(tmp_path: Path) -> None:
    server, session, host, port = _start(tmp_path, read_only=True)
    try:
        payload = json.loads(_request(host, port, "GET", "/api/v1/session")[2])
        window_id = payload["windows"][0]["window_id"]
        status, _, body = _request(
            host,
            port,
            "POST",
            f"/api/v1/windows/{window_id}/events",
            origin=f"http://{host}:{port}",
            body=json.dumps({"request_id": "x", "base_revision": 0, "action": {"kind": "confirm"}}).encode(),
        )
        assert status == 403
        assert json.loads(body)["error"]["code"] == "read_only"
        assert open_review_store(session).events() == ()
    finally:
        server.shutdown()
        server.server_close()


def test_save_replay_duplicate_stale_undo_and_signoff_modes(tmp_path: Path) -> None:
    server, session, host, port = _start(tmp_path)
    origin = f"http://{host}:{port}"
    try:
        payload = json.loads(_request(host, port, "GET", "/api/v1/session")[2])
        window_id = payload["windows"][0]["window_id"]
        event_body = {
            "request_id": "confirm-1",
            "base_revision": 0,
            "action": {"kind": "confirm"},
        }
        status, _, body = _request(
            host,
            port,
            "POST",
            f"/api/v1/windows/{window_id}/events",
            origin=origin,
            body=json.dumps(event_body).encode(),
        )
        assert status == 200
        first = json.loads(body)
        status, _, body = _request(
            host,
            port,
            "POST",
            f"/api/v1/windows/{window_id}/events",
            origin=origin,
            body=json.dumps(event_body).encode(),
        )
        assert json.loads(body)["event"]["event_hash"] == first["event"]["event_hash"]
        stale = dict(event_body)
        stale["request_id"] = "stale"
        status, _, _ = _request(
            host,
            port,
            "POST",
            f"/api/v1/windows/{window_id}/events",
            origin=origin,
            body=json.dumps(stale).encode(),
        )
        assert status == 409
        undo_body = {
            "request_id": "undo-1",
            "base_revision": 1,
            "action": {"kind": "undo", "reverted_event_hash": first["event"]["event_hash"]},
        }
        status, _, body = _request(
            host,
            port,
            "POST",
            f"/api/v1/windows/{window_id}/events",
            origin=origin,
            body=json.dumps(undo_body).encode(),
        )
        assert status == 200
        assert json.loads(body)["state"]["decision"]["kind"] == "pending"
    finally:
        server.shutdown()
        server.server_close()

    store = open_review_store(session)
    window_id = store.overlay.window_ids()[0]
    store.append(window_id=window_id, actor="rev-1", request_id="c2", base_revision=2, action={"kind": "confirm"})
    server, thread, host, port = start_review_server(session, actor_id="signer", signoff_mode=True)
    origin = f"http://{host}:{port}"
    try:
        status, _, body = _request(
            host,
            port,
            "POST",
            f"/api/v1/windows/{window_id}/signoff",
            origin=origin,
            body=json.dumps({"request_id": "sign-1", "base_revision": 3, "decision": "accept"}).encode(),
        )
        assert status == 200
        assert json.loads(body)["state"]["decision"]["kind"] == "signed_off"
        status, _, _ = _request(
            host,
            port,
            "POST",
            f"/api/v1/windows/{window_id}/events",
            origin=origin,
            body=json.dumps({"request_id": "nope", "base_revision": 4, "action": {"kind": "confirm"}}).encode(),
        )
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()


def test_ui_assets_have_no_external_urls_and_keyboard_logic(tmp_path: Path) -> None:
    for path in UI_ROOT.glob("*"):
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text
        assert "https://" not in text
        assert "cdn." not in text
        assert "token" not in text
        assert "Authorization" not in text
    html = (UI_ROOT / "index.html").read_text(encoding="utf-8")
    for label in ("Confirm", "Deny", "No speech", "Uncertain", "Undo", "Previous", "Next"):
        assert label in html
    logic = UI_ROOT / "logic.js"
    result = subprocess.run(
        [
            "node",
            "-e",
            "const l=require(process.argv[1]); const a=l.applyKey({playheadFrame:10,playing:false},'ArrowRight',{}); if(a.playheadFrame!==11) process.exit(2); if(l.seekWord({window_start_seconds:0.4})!==20) process.exit(3); if(l.snapFrame(0.03)!==2) process.exit(4);",
            str(logic),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
