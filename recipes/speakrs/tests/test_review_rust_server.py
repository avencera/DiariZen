"""Protect the Rust review server API, media policy, and Python event compatibility."""

from __future__ import annotations

import io
import json
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPConnection
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import soundfile as sf

from recipes.speakrs.large.hashing import canonical_json
from recipes.speakrs.large.review_export import export_review, validate_completed_review
from recipes.speakrs.large.review_models import (
    Clear,
    ClockClear,
    ClockMeasurementEntered,
    ConfirmAction,
    DefectSet,
    SetDefectsAction,
)
from recipes.speakrs.large.review_overlay import prepare_review_overlay
from recipes.speakrs.large.review_store import ReviewEvent, open_review_store
from recipes.speakrs.tests.test_review_overlay import _build_packet


REPO_ROOT = Path(__file__).resolve().parents[3]
UI_ROOT = REPO_ROOT / "recipes" / "speakrs" / "review-ui"
BUILD_COMMAND = "pnpm --dir recipes/speakrs/review-ui install && pnpm --dir recipes/speakrs/review-ui build"
CLEAR_DEFECTS = {
    "identity": {"kind": "clear"},
    "clock": {"kind": "clear", "measurement": {"kind": "absent"}},
    "synchronization": {"kind": "clear"},
    "redaction": {"kind": "clear"},
}


@pytest.fixture(scope="session")
def review_binary() -> Path:
    result = subprocess.run(
        ["cargo", "build", "-p", "open-yap-review", "--message-format=json-render-diagnostics"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        message = json.loads(line)
        target = message.get("target", {})
        if message.get("reason") == "compiler-artifact" and target.get("name") == "open-yap-review":
            if message.get("executable"):
                return Path(message["executable"])
    raise AssertionError("cargo did not report the open-yap-review executable")


@pytest.fixture
def ui_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "ui-dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Open Yap review stub</title>\n", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("export {};\n", encoding="utf-8")
    return dist


@pytest.fixture
def session(tmp_path: Path) -> Path:
    archive, packet = _build_packet(tmp_path)
    destination = tmp_path / "session"
    prepare_review_overlay(packet, archive, destination)
    return destination


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict:
        return json.loads(self.body)


@dataclass
class RunningServer:
    host: str
    port: int

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        origin: str | None = None,
        headers: dict[str, str] | None = None,
        host_header: str | None = None,
    ) -> Response:
        connection = HTTPConnection(self.host, self.port, timeout=30)
        extra = dict(headers or {})
        if origin:
            extra["Origin"] = origin
        extra["Host"] = host_header or f"{self.host}:{self.port}"
        if body is not None:
            extra.setdefault("Content-Type", "application/json")
            extra["Content-Length"] = str(len(body))
        connection.request(method, path, body=body, headers=extra)
        response = connection.getresponse()
        payload = response.read()
        header_map = {key.lower(): value for key, value in response.getheaders()}
        connection.close()
        return Response(response.status, header_map, payload)

    def get(self, path: str, **kwargs) -> Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, payload: object) -> Response:
        return self.request("POST", path, body=json.dumps(payload).encode(), origin=self.origin)


def _command(binary: Path, session: Path, dist: Path, actor: str, *flags: str) -> list[str]:
    return [
        str(binary),
        "serve",
        "--session",
        str(session),
        "--reviewer-id",
        actor,
        "--port",
        "0",
        "--ui-dist",
        str(dist),
        *flags,
    ]


@contextmanager
def _serve(binary: Path, session: Path, dist: Path, actor: str = "rev-1", *flags: str) -> Iterator[RunningServer]:
    process = subprocess.Popen(
        _command(binary, session, dist, actor, *flags),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = process.stdout.readline() if process.stdout else ""
        if not line:
            _, stderr = process.communicate(timeout=30)
            raise AssertionError(f"review server did not start: {stderr}")
        ready = json.loads(line)
        assert ready["ok"] is True
        assert ready["host"] == "127.0.0.1"
        assert ready["url"] == f"http://127.0.0.1:{ready['port']}/"
        assert ready["reviewer_id"] == actor
        yield RunningServer(ready["host"], int(ready["port"]))
    finally:
        process.terminate()
        try:
            process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


def _window_ids(server: RunningServer) -> list[str]:
    return [window["window_id"] for window in server.get("/api/v1/session").json()["windows"]]


def _event(server: RunningServer, window_id: str, request_id: str, base_revision: int, action: dict) -> Response:
    return server.post(
        f"/api/v1/windows/{window_id}/events",
        {"request_id": request_id, "base_revision": base_revision, "action": action},
    )


def _packet_media(session: Path, window_id: str, key: str) -> Path:
    stored = json.loads((session / "session.json").read_text(encoding="utf-8"))
    manifest = json.loads((session / "overlay" / "overlay-manifest.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["windows"] if item["window_id"] == window_id)
    return Path(stored["packet_path"]) / record["packet_files"][key]


def test_origin_headers_and_path_policy(review_binary: Path, session: Path, ui_dist: Path) -> None:
    with _serve(review_binary, session, ui_dist) as server:
        response = server.get("/api/v1/session")
        assert response.status == 200
        assert response.headers["cache-control"] == "no-store"
        assert "default-src 'self'" in response.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "set-cookie" not in response.headers
        payload = response.json()
        assert payload["actor_id"] == "rev-1"
        assert payload["read_only"] is False
        assert payload["signoff_mode"] is False
        assert payload["progress"]["window_count"] == len(payload["windows"])
        window_id = payload["windows"][0]["window_id"]

        foreign = server.request(
            "POST", f"/api/v1/windows/{window_id}/events", origin="http://example.com", body=b"{}"
        )
        assert foreign.status == 403
        assert foreign.json()["error"]["message"] == "origin is not allowed"
        rebinding = server.get("/api/v1/session", host_header="evil.example")
        assert rebinding.status == 403
        assert rebinding.json() == {"ok": False, "error": {"code": "forbidden", "message": "host is not 127.0.0.1"}}
        same_origin = server.get("/api/v1/session", origin=server.origin)
        assert same_origin.headers["access-control-allow-origin"] == server.origin
        assert same_origin.headers["vary"] == "Origin"
        assert server.request("OPTIONS", "/api/v1/session", origin=server.origin).status == 204

        assert server.get("/api/v1/windows/../secret").status in {400, 404}
        assert server.get("/api/v1/windows/missing").status == 404
        assert server.get(f"/api/v1/windows/{window_id}/media/other").status == 404

        emitted = server.get(f"/api/v1/windows/{window_id}/media/emitted", headers={"Range": "bytes=0-15"})
        assert emitted.status == 206
        assert len(emitted.body) == 16
        assert emitted.headers["content-range"].startswith("bytes 0-15/")
        assert emitted.headers["content-type"] == "audio/flac"
        bad_range = server.get(f"/api/v1/windows/{window_id}/media/emitted", headers={"Range": "bytes=9-1"})
        assert bad_range.status == 416
        assert bad_range.headers["content-range"].startswith("bytes */")

        index = server.get("/")
        assert index.status == 200
        assert b"Open Yap review stub" in index.body
        assert index.headers["content-type"].startswith("text/html")
        asset = server.get("/assets/app.js")
        assert asset.status == 200
        assert asset.headers["content-type"].startswith("text/javascript")
        assert server.get("/assets/../index.html").status == 404
        assert server.get("/missing.js").status == 404

        too_large = server.request(
            "POST",
            f"/api/v1/windows/{window_id}/events",
            origin=server.origin,
            body=b"x" * (2 * 1024 * 1024),
        )
        assert too_large.status == 413
        invalid = server.request("POST", f"/api/v1/windows/{window_id}/events", origin=server.origin, body=b"{")
        assert invalid.status == 400
        assert invalid.json()["error"]["code"] == "invalid_json"
        unknown_post = server.post("/api/v1/export", {})
        assert unknown_post.status == 400

        connection = HTTPConnection(server.host, server.port, timeout=30)
        connection.putrequest("POST", f"/api/v1/windows/{window_id}/events", skip_host=True)
        connection.putheader("Host", f"{server.host}:{server.port}")
        connection.endheaders()
        assert connection.getresponse().status == 411
        connection.close()

        wav = server.get(f"/api/v1/windows/{window_id}/media/emitted?container=wav")
        assert wav.status == 200
        assert wav.body[:4] == b"RIFF"
        assert wav.headers["content-type"] == "audio/wav"
        decoded, rate = sf.read(io.BytesIO(wav.body), dtype="int16")
        source, source_rate = sf.read(_packet_media(session, window_id, "emitted"), dtype="int16")
        assert rate == source_rate
        assert np.array_equal(decoded, source)
        reference = server.get(f"/api/v1/windows/{window_id}/media/speaker-b?container=wav")
        reference_samples, reference_rate = sf.read(io.BytesIO(reference.body), dtype="int16")
        expected, expected_rate = sf.read(_packet_media(session, window_id, "reference_speaker_b"), dtype="int16")
        assert reference_rate == expected_rate
        assert np.array_equal(reference_samples, expected)
        wav_range = server.get(
            f"/api/v1/windows/{window_id}/media/emitted?container=wav", headers={"Range": "bytes=0-43"}
        )
        assert wav_range.status == 206
        assert wav_range.body == wav.body[:44]


def test_read_only_disables_writes(review_binary: Path, session: Path, ui_dist: Path) -> None:
    with _serve(review_binary, session, ui_dist, "rev-1", "--read-only") as server:
        payload = server.get("/api/v1/session").json()
        assert payload["read_only"] is True
        window_id = payload["windows"][0]["window_id"]
        response = _event(server, window_id, "x", 0, {"kind": "confirm"})
        assert response.status == 403
        assert response.json()["error"]["code"] == "read_only"
        assert server.get(f"/api/v1/windows/{window_id}/media/emitted?container=wav").status == 200
    assert open_review_store(session).events() == ()


def test_save_replay_duplicate_stale_undo_and_signoff_modes(review_binary: Path, session: Path, ui_dist: Path) -> None:
    with _serve(review_binary, session, ui_dist) as server:
        window_id = _window_ids(server)[0]
        first = _event(server, window_id, "confirm-1", 0, {"kind": "confirm"})
        assert first.status == 200
        first_payload = first.json()
        assert first_payload["state"]["decision"]["kind"] == "confirmed_proposal"
        assert first_payload["progress"]["reviewed_count"] == 1
        duplicate = _event(server, window_id, "confirm-1", 0, {"kind": "confirm"})
        assert duplicate.status == 200
        assert duplicate.json()["event"]["event_hash"] == first_payload["event"]["event_hash"]
        reused = _event(server, window_id, "confirm-1", 0, {"kind": "no_speech"})
        assert reused.status == 409
        stale = _event(server, window_id, "stale", 0, {"kind": "confirm"})
        assert stale.status == 409
        assert stale.json()["error"]["details"] == {"base_revision": 0, "current_revision": 1}
        assert _event(server, "missing", "unknown", 0, {"kind": "confirm"}).status == 404
        signoff_in_review = server.post(
            f"/api/v1/windows/{window_id}/signoff", {"request_id": "s", "base_revision": 1, "decision": "accept"}
        )
        assert signoff_in_review.status == 400
        undo = _event(
            server,
            window_id,
            "undo-1",
            1,
            {"kind": "undo", "reverted_event_hash": first_payload["event"]["event_hash"]},
        )
        assert undo.status == 200
        assert undo.json()["state"]["decision"]["kind"] == "pending"

    store = open_review_store(session)
    store.append(window_id=window_id, actor="rev-1", request_id="c2", base_revision=2, action={"kind": "confirm"})
    with _serve(review_binary, session, ui_dist, "signer", "--signoff") as server:
        assert server.get("/api/v1/session").json()["signoff_mode"] is True
        signed = server.post(
            f"/api/v1/windows/{window_id}/signoff",
            {"request_id": "sign-1", "base_revision": 3, "decision": "accept"},
        )
        assert signed.status == 200
        assert signed.json()["state"]["decision"]["kind"] == "signed_off"
        assert signed.json()["latest_event_kind"] == "sign_off_accept"
        refused = _event(server, window_id, "nope", 4, {"kind": "confirm"})
        assert refused.status == 400


def test_set_defects_then_undo_uses_latest_event_hash(review_binary: Path, session: Path, ui_dist: Path) -> None:
    with _serve(review_binary, session, ui_dist) as server:
        window_id = _window_ids(server)[0]
        confirmed = _event(server, window_id, "c1", 0, {"kind": "confirm"}).json()
        defects = _event(server, window_id, "d1", 1, {"kind": "set_defects", "defects": CLEAR_DEFECTS})
        assert defects.status == 200
        defects_payload = defects.json()
        assert defects_payload["latest_event_hash"] == defects_payload["event"]["event_hash"]
        assert defects_payload["latest_event_kind"] == "set_defects"
        assert defects_payload["state"]["last_review_actor"] == "rev-1"

        old_target = _event(
            server,
            window_id,
            "u-old",
            2,
            {"kind": "undo", "reverted_event_hash": confirmed["event"]["event_hash"]},
        )
        assert old_target.status == 400
        undo = _event(
            server,
            window_id,
            "u1",
            2,
            {"kind": "undo", "reverted_event_hash": defects_payload["latest_event_hash"]},
        )
        assert undo.status == 200
        undo_payload = undo.json()
        assert undo_payload["state"]["decision"]["kind"] == "confirmed_proposal"
        assert undo_payload["state"]["defects"]["identity"]["kind"] == "not_reviewed"
        assert undo_payload["latest_event_kind"] == "undo"

        window = server.get(f"/api/v1/windows/{window_id}").json()
        assert window["latest_event_hash"] == undo_payload["event"]["event_hash"]
        assert window["latest_event_kind"] == "undo"
        assert window["state"]["revision"] == 3


def test_reviewed_activity_after_correct(review_binary: Path, session: Path, ui_dist: Path) -> None:
    with _serve(review_binary, session, ui_dist) as server:
        first, second = _window_ids(server)[:2]
        pending = server.get(f"/api/v1/windows/{first}").json()
        assert pending["reviewed_activity"] is None
        assert pending["latest_event_hash"] is None
        assert pending["latest_event_kind"] is None
        assert pending["window"]["window_id"] == first
        assert pending["transcript"]["display_only"] is True
        assert pending["proposal"]["conversion_policy"] == "speakrs-open-yap-grid-proposal-v1"
        assert pending["media"]["speaker_a"] == f"/api/v1/windows/{first}/media/speaker-a"

        activity = [
            {"speaker": "speaker_b", "start_frame": 40, "end_frame": 60},
            {"speaker": "speaker_a", "start_frame": 10, "end_frame": 20},
            {"speaker": "speaker_a", "start_frame": 20, "end_frame": 25},
        ]
        corrected = _event(server, first, "fix", 0, {"kind": "correct", "activity": activity})
        assert corrected.status == 200
        expected = [
            {"speaker": "speaker_a", "start_frame": 10, "end_frame": 25},
            {"speaker": "speaker_b", "start_frame": 40, "end_frame": 60},
        ]
        assert corrected.json()["reviewed_activity"] == expected
        assert server.get(f"/api/v1/windows/{first}").json()["reviewed_activity"] == expected

        confirmed = _event(server, second, "ok", 0, {"kind": "confirm"}).json()
        proposal = server.get(f"/api/v1/windows/{second}").json()["proposal"]["intervals"]
        assert confirmed["reviewed_activity"] == proposal
        no_speech = server.post(
            f"/api/v1/windows/{second}/events",
            {"request_id": "quiet", "base_revision": 1, "action": {"kind": "no_speech"}},
        )
        assert no_speech.json()["reviewed_activity"] == []


def _encoded(window_id: str) -> str:
    return "".join(f"%{ord(character):02X}" if character == "-" else character for character in window_id)


def test_percent_encoded_window_ids_and_response_contract(review_binary: Path, session: Path, ui_dist: Path) -> None:
    contract_fields = {
        "ok",
        "event",
        "state",
        "progress",
        "latest_event_hash",
        "latest_event_kind",
        "reviewed_activity",
    }
    with _serve(review_binary, session, ui_dist) as server:
        window_id = _window_ids(server)[0]
        encoded = _encoded(window_id)
        assert encoded != window_id

        window = server.get(f"/api/v1/windows/{encoded}", origin=server.origin)
        assert window.status == 200
        assert window.json()["window"]["window_id"] == window_id
        for url in window.json()["media"].values():
            assert "?" not in url
        wav = server.get(f"/api/v1/windows/{encoded}/media/speaker-a?container=wav", origin=server.origin)
        assert wav.status == 200
        assert wav.body[:4] == b"RIFF"
        assert wav.body[8:16] == b"WAVEfmt "
        assert wav.body[36:40] == b"data"

        no_speech = _event(server, encoded, "quiet", 0, {"kind": "no_speech"})
        assert no_speech.status == 200
        assert set(no_speech.json()) == contract_fields
        assert no_speech.json()["event"]["window_id"] == window_id
        assert no_speech.json()["reviewed_activity"] == []
        assert no_speech.json()["latest_event_kind"] == "no_speech"

        undo = _event(
            server,
            encoded,
            "undo-quiet",
            1,
            {"kind": "undo", "reverted_event_hash": no_speech.json()["latest_event_hash"]},
        )
        assert set(undo.json()) == contract_fields
        assert undo.json()["reviewed_activity"] is None
        assert _event(server, encoded, "confirm", 2, {"kind": "confirm"}).status == 200

        stale = _event(server, encoded, "stale", 0, {"kind": "confirm"})
        reused = _event(server, encoded, "confirm", 0, {"kind": "no_speech"})
        for response in (stale, reused):
            assert response.status == 409
            assert set(response.json()) == {"ok", "error"}
            assert response.json()["ok"] is False
            assert set(response.json()["error"]) == {"code", "message", "details"}
        assert server.get("/api/v1/windows/w%zz").status == 404

    with _serve(review_binary, session, ui_dist, "signer", "--signoff") as server:
        signed = server.post(
            f"/api/v1/windows/{encoded}/signoff",
            {"request_id": "sign", "base_revision": 3, "decision": "accept"},
        )
        assert signed.status == 200
        assert set(signed.json()) == contract_fields
        assert signed.json()["latest_event_kind"] == "sign_off_accept"
        proposal = server.get(f"/api/v1/windows/{encoded}").json()["proposal"]["intervals"]
        assert signed.json()["reviewed_activity"] == proposal


def test_missing_ui_dist_fails_startup(review_binary: Path, session: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        _command(review_binary, session, tmp_path / "no-dist", "rev-1"),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert BUILD_COMMAND in result.stderr
    assert result.stdout == ""


def test_second_server_on_same_session_fails(review_binary: Path, session: Path, ui_dist: Path) -> None:
    with _serve(review_binary, session, ui_dist):
        result = subprocess.run(
            _command(review_binary, session, ui_dist, "rev-2", "--read-only"),
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode != 0
        assert "locked" in result.stderr
    with _serve(review_binary, session, ui_dist, "rev-2", "--read-only") as server:
        assert server.get("/api/v1/session").status == 200


def _assert_python_byte_compatible(session: Path) -> None:
    for path in sorted((session / "events").glob("event-*.json")):
        text = path.read_text(encoding="utf-8")
        event = ReviewEvent.from_dict(json.loads(text))
        assert canonical_json(event.to_dict()) == text
        assert path.name == f"event-{event.sequence:08d}-{event.event_hash}.json"


def test_python_export_accepts_rust_review(review_binary: Path, session: Path, ui_dist: Path) -> None:
    correction = [
        {"speaker": "speaker_a", "start_frame": 0, "end_frame": 30},
        {"speaker": "speaker_b", "start_frame": 25, "end_frame": 50},
    ]
    with _serve(review_binary, session, ui_dist) as server:
        first, second, third = _window_ids(server)[:3]
        assert _event(server, first, "r-confirm", 0, {"kind": "confirm"}).status == 200
        assert _event(server, second, "r-correct", 0, {"kind": "correct", "activity": correction}).status == 200
        for window_id in (first, second):
            response = _event(
                server, window_id, f"r-def-{window_id}", 1, {"kind": "set_defects", "defects": CLEAR_DEFECTS}
            )
            assert response.status == 200
        entered = {
            "identity": {"kind": "unresolved", "reason": " voices swap "},
            "clock": {
                "kind": "unresolved",
                "reason": "drift",
                "measurement": {"kind": "entered", "offset_seconds": 0, "drift_seconds_per_second": 0.00001},
            },
            "synchronization": {"kind": "not_reviewed"},
            "redaction": {"kind": "clear"},
        }
        measured = _event(server, third, "r-measure", 0, {"kind": "set_defects", "defects": entered})
        assert measured.status == 200
        stored_clock = measured.json()["event"]["action"]["defects"]["clock"]
        assert stored_clock["measurement"]["offset_seconds"] == 0.0
        assert measured.json()["event"]["action"]["defects"]["identity"]["reason"] == "voices swap"

    with _serve(review_binary, session, ui_dist, "signer", "--signoff") as server:
        for window_id in (first, second):
            response = server.post(
                f"/api/v1/windows/{window_id}/signoff",
                {"request_id": f"s-{window_id}", "base_revision": 2, "decision": "accept"},
            )
            assert response.status == 200
        second_window = server.get(f"/api/v1/windows/{second}").json()
        assert second_window["reviewed_activity"] == correction

    _assert_python_byte_compatible(session)
    assert re.search(r'"offset_seconds":0\.0', next((session / "events").glob("event-00000005-*.json")).read_text())
    store = open_review_store(session)
    assert len(store.events()) == 7
    assert store.window_state(first).decision.kind == "signed_off"
    assert store.window_state(second).decision.signer == "signer"
    assert store.window_state(third).defects.clock.measurement == ClockMeasurementEntered(0.0, 1e-05)

    result = export_review(store)
    assert result["exported_window_count"] == 2
    export_root = Path(result["export_path"])
    manifest = json.loads((export_root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["exported_windows"]) == {first, second}
    annotation = json.loads((export_root / manifest["exported_windows"][second]).read_text(encoding="utf-8"))
    assert annotation["reviewer_id"] == "rev-1"
    assert annotation["independent_signoff"]["signer_id"] == "signer"
    assert [item["speaker"] for item in annotation["speaker_activity"]] == ["speaker_a", "speaker_b"]
    report = validate_completed_review(session, export_root)
    assert report["ok"] is True
    assert report["eligible_window_count"] == 2


def test_rust_server_continues_python_events(review_binary: Path, session: Path, ui_dist: Path) -> None:
    store = open_review_store(session)
    window_id = store.overlay.window_ids()[0]
    store.append(window_id=window_id, actor="rev-1", request_id="py-confirm", base_revision=0, action=ConfirmAction())
    defects = DefectSet(
        identity=Clear(),
        clock=ClockClear(ClockMeasurementEntered(0.02, 3e-07)),
        synchronization=Clear(),
        redaction=Clear(),
    )
    python_head = store.append(
        window_id=window_id,
        actor="rev-1",
        request_id="py-defects",
        base_revision=1,
        action=SetDefectsAction(defects),
    )

    with _serve(review_binary, session, ui_dist) as server:
        session_payload = server.get("/api/v1/session").json()
        assert session_payload["windows"][0]["revision"] == 2
        window = server.get(f"/api/v1/windows/{window_id}").json()
        assert window["latest_event_hash"] == python_head.event_hash
        assert window["state"]["defects"]["clock"]["measurement"]["drift_seconds_per_second"] == 3e-07
        duplicate = _event(
            server,
            window_id,
            "py-defects",
            1,
            {"kind": "set_defects", "defects": defects.to_dict()},
        )
        assert duplicate.status == 200
        assert duplicate.json()["event"]["event_hash"] == python_head.event_hash
        response = _event(server, window_id, "rust-no-speech", 2, {"kind": "no_speech"})
        assert response.status == 200
        event = response.json()["event"]
        assert event["sequence"] == 3
        assert event["prior_hash"] == python_head.event_hash

    _assert_python_byte_compatible(session)
    replayed = open_review_store(session)
    assert [item.request_id for item in replayed.events()] == ["py-confirm", "py-defects", "rust-no-speech"]
    assert replayed.window_state(window_id).decision.kind == "no_speech"
    assert replayed.window_state(window_id).revision == 3


def test_review_ui_sources_have_no_external_urls_or_tokens() -> None:
    sources = []
    if (UI_ROOT / "src").is_dir():
        sources.extend(path for path in (UI_ROOT / "src").rglob("*") if path.is_file())
    if (UI_ROOT / "index.html").is_file():
        sources.append(UI_ROOT / "index.html")
    if not sources:
        pytest.skip("review UI sources are not present")
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for needle in ("http://", "https://", "token", "Authorization"):
            assert needle not in text, f"{needle} found in {path.relative_to(REPO_ROOT)}"
