"""Private localhost review API, media ranges, and bundled UI."""

from __future__ import annotations

import io
import json
import mimetypes
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import soundfile as sf

from .errors import ContractError, LargeError, PreparationError
from .jsonio import read_json
from .review_export import export_review
from .review_models import parse_action, parse_non_empty_text
from .review_overlay import load_review_session
from .review_store import open_review_store


UI_ROOT = Path(__file__).resolve().parent / "review_ui"
MAX_JSON_BYTES = 1 * 1024 * 1024
MAX_FULL_MEDIA_BYTES = 32 * 1024 * 1024
MEDIA_KINDS = {
    "emitted": "emitted",
    "speaker-a": "reference_speaker_a",
    "speaker-b": "reference_speaker_b",
}
UI_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.css": "app.css",
    "/app.js": "app.js",
    "/logic.js": "logic.js",
}


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


class ReviewService:
    """In-memory server configuration for one review session."""

    def __init__(
        self,
        session_root: Path,
        *,
        actor_id: str,
        read_only: bool = False,
        signoff_mode: bool = False,
    ) -> None:
        loaded = load_review_session(session_root)
        self.session_root = loaded["root"]
        self.packet_root = loaded["packet_root"]
        self.overlay = loaded["overlay"]
        self.store = open_review_store(self.session_root)
        self.actor_id = parse_non_empty_text(actor_id, "actor_id")
        self.read_only = bool(read_only)
        self.signoff_mode = bool(signoff_mode)
        self.media_paths = self._media_allow_list()
        self.window_index = {record["window_id"]: record for record in self.overlay.manifest["windows"]}
        self._wav_cache: OrderedDict[str, bytes] = OrderedDict()
        self._wav_lock = threading.Lock()

    def wav_bytes(self, path: Path) -> bytes:
        """Convert one allow-listed media file to WAV for browser decode."""

        key = path.as_posix()
        with self._wav_lock:
            cached = self._wav_cache.get(key)
            if cached is not None:
                self._wav_cache.move_to_end(key)
                return cached
        samples, rate = sf.read(path, dtype="float32", always_2d=False)
        buffer = io.BytesIO()
        sf.write(buffer, samples, int(rate), format="WAV", subtype="PCM_16")
        payload = buffer.getvalue()
        with self._wav_lock:
            self._wav_cache[key] = payload
            self._wav_cache.move_to_end(key)
            while len(self._wav_cache) > 9:
                self._wav_cache.popitem(last=False)
        return payload

    def _media_allow_list(self) -> dict[tuple[str, str], Path]:
        allowed: dict[tuple[str, str], Path] = {}
        packet = self.packet_root.resolve()
        for record in self.overlay.manifest["windows"]:
            window_id = str(record["window_id"])
            files = record["packet_files"]
            for kind, key in MEDIA_KINDS.items():
                relative = files.get(key)
                if not isinstance(relative, str):
                    raise PreparationError("packet media path is missing", {"window_id": window_id, "kind": kind})
                path = (packet / relative).resolve()
                if packet not in path.parents or not path.is_file():
                    raise PreparationError("packet media path is unsafe or missing", {"path": path.as_posix()})
                allowed[(window_id, kind)] = path
        return allowed

    def origin_allowed(self, origin: str | None, host: str) -> bool:
        expected = f"http://{host}"
        if origin in (None, "", "null"):
            return True
        return origin == expected

    def session_payload(self) -> dict[str, Any]:
        progress = self.store.progress().as_dict()
        windows = []
        for record in self.overlay.manifest["windows"]:
            window_id = str(record["window_id"])
            state = self.store.window_state(window_id)
            windows.append(
                {
                    "window_id": window_id,
                    "parent_id": record["parent_id"],
                    "selection_kind": record.get("selection_kind"),
                    "stratum": record.get("stratum"),
                    "revision": state.revision,
                    "decision": state.decision.to_dict(),
                }
            )
        return {
            "actor_id": self.actor_id,
            "read_only": self.read_only,
            "signoff_mode": self.signoff_mode,
            "overlay_sha256": self.overlay.overlay_sha256,
            "packet_manifest_sha256": self.store.packet_hash,
            "progress": progress,
            "quarantined": self.store.quarantined,
            "windows": windows,
        }

    def window_payload(self, window_id: str) -> dict[str, Any]:
        record = self.window_index.get(window_id)
        if record is None:
            raise ContractError("unknown window", {"window_id": window_id})
        overlay_root = self.overlay.root
        transcript = read_json(overlay_root / str(record["transcript"]["path"]))
        proposal = read_json(overlay_root / str(record["proposal"]["path"]))
        state = self.store.window_state(window_id)
        return {
            "window": {
                "window_id": window_id,
                "parent_id": record["parent_id"],
                "selection_kind": record.get("selection_kind"),
                "stratum": record.get("stratum"),
                "source_clock": record.get("source_clock"),
            },
            "transcript": transcript,
            "proposal": proposal,
            "state": state.to_dict(),
            "media": {
                "emitted": f"/api/v1/windows/{window_id}/media/emitted",
                "speaker_a": f"/api/v1/windows/{window_id}/media/speaker-a",
                "speaker_b": f"/api/v1/windows/{window_id}/media/speaker-b",
            },
        }


def _error_payload(error: Exception) -> tuple[int, dict[str, Any]]:
    if isinstance(error, LargeError):
        status = 400
        if "stale revision" in error.message:
            status = 409
        if "unknown window" in error.message:
            status = 404
        if "request_id was reused" in error.message:
            status = 409
        return status, error.to_json()
    return 400, {"ok": False, "error": {"code": "error", "message": str(error), "details": {}}}


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "OpenYapReview/1"
    close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        return

    @property
    def service(self) -> ReviewService:
        return self.server.review_service  # type: ignore[attr-defined]

    def _host(self) -> str:
        host = self.headers.get("Host", "")
        return host

    def _allowed_host(self) -> bool:
        host = self._host()
        return host.startswith("127.0.0.1:") or host == "127.0.0.1"

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; media-src 'self' blob:; "
            "img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        origin = self.headers.get("Origin")
        if origin and self.service.origin_allowed(origin, self._host()):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _require_origin(self) -> bool:
        if not self._allowed_host():
            self._send_json(403, {"ok": False, "error": {"code": "forbidden", "message": "host is not 127.0.0.1"}})
            return False
        origin = self.headers.get("Origin")
        if origin and not self.service.origin_allowed(origin, self._host()):
            self._send_json(403, {"ok": False, "error": {"code": "forbidden", "message": "origin is not allowed"}})
            return False
        return True

    def do_OPTIONS(self) -> None:
        if not self._allowed_host():
            self._send_json(403, {"ok": False, "error": {"code": "forbidden", "message": "host is not 127.0.0.1"}})
            return
        origin = self.headers.get("Origin")
        extra = {
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Origin",
        }
        if origin and self.service.origin_allowed(origin, self._host()):
            extra["Access-Control-Allow-Origin"] = origin
        self._send(204, b"", "text/plain", extra)

    def do_GET(self) -> None:
        if not self._require_origin():
            return
        parsed = urlparse(self.path)
        path = parsed.path
        if path in UI_FILES:
            self._serve_ui(UI_FILES[path])
            return
        if path == "/api/v1/session":
            self._send_json(200, {"ok": True, **self.service.session_payload()})
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:3] == ["api", "v1", "windows"]:
            try:
                payload = self.service.window_payload(parts[3])
            except Exception as error:
                status, body = _error_payload(error)
                self._send_json(status, body)
                return
            self._send_json(200, {"ok": True, **payload})
            return
        if len(parts) == 6 and parts[:3] == ["api", "v1", "windows"] and parts[4] == "media":
            self._serve_media(parts[3], parts[5])
            return
        self._send_json(404, {"ok": False, "error": {"code": "not_found", "message": "unknown path"}})

    def do_POST(self) -> None:
        if not self._require_origin():
            return
        if self.service.read_only:
            self._send_json(
                403, {"ok": False, "error": {"code": "read_only", "message": "review actions are disabled"}}
            )
            return
        length_header = self.headers.get("Content-Length", "")
        try:
            length = int(length_header)
        except ValueError:
            self._send_json(411, {"ok": False, "error": {"code": "length", "message": "Content-Length is required"}})
            return
        if length < 0 or length > MAX_JSON_BYTES:
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self._send_json(
                413, {"ok": False, "error": {"code": "payload_too_large", "message": "JSON payload exceeds the bound"}}
            )
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "error": {"code": "invalid_json", "message": "body is not JSON"}})
            return
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        try:
            result = self._dispatch_post(parts, payload)
        except Exception as error:
            status, body = _error_payload(error)
            self._send_json(status, body)
            return
        self._send_json(200, result)

    def _dispatch_post(self, parts: list[str], payload: Any) -> dict[str, Any]:
        if parts == ["api", "v1", "export"]:
            if self.service.signoff_mode is False and not self.service.read_only:
                # export is allowed after review; keep it available to the current actor
                pass
            result = export_review(self.service.store)
            return {"ok": True, **result}
        if len(parts) == 5 and parts[:3] == ["api", "v1", "windows"] and parts[4] == "events":
            if self.service.signoff_mode:
                raise ContractError("reviewer events are disabled in sign-off mode")
            return self._create_event(parts[3], payload)
        if len(parts) == 5 and parts[:3] == ["api", "v1", "windows"] and parts[4] == "signoff":
            if not self.service.signoff_mode:
                raise ContractError("sign-off is disabled in reviewer mode")
            return self._create_signoff(parts[3], payload)
        raise ContractError("unknown path")

    def _create_event(self, window_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ContractError("event body must be an object")
        request_id = parse_non_empty_text(payload.get("request_id"), "request_id")
        base_revision = payload.get("base_revision")
        action = parse_action(payload.get("action"))
        event = self.service.store.append(
            window_id=window_id,
            actor=self.service.actor_id,
            request_id=request_id,
            base_revision=base_revision,
            action=action,
        )
        state = self.service.store.window_state(window_id)
        return {
            "ok": True,
            "event": event.to_dict(),
            "state": state.to_dict(),
            "progress": self.service.store.progress().as_dict(),
        }

    def _create_signoff(self, window_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ContractError("sign-off body must be an object")
        decision = payload.get("decision")
        if decision == "accept":
            action_payload = {"kind": "sign_off_accept"}
        elif decision == "return":
            action_payload = {"kind": "sign_off_return", "reason": payload.get("reason")}
        else:
            raise ContractError("sign-off decision must be accept or return")
        event = self.service.store.append(
            window_id=window_id,
            actor=self.service.actor_id,
            request_id=parse_non_empty_text(payload.get("request_id"), "request_id"),
            base_revision=payload.get("base_revision"),
            action=action_payload,
        )
        state = self.service.store.window_state(window_id)
        return {
            "ok": True,
            "event": event.to_dict(),
            "state": state.to_dict(),
            "progress": self.service.store.progress().as_dict(),
        }

    def _serve_ui(self, name: str) -> None:
        path = (UI_ROOT / name).resolve()
        if UI_ROOT.resolve() not in path.parents and path != UI_ROOT.resolve():
            self._send_json(404, {"ok": False, "error": {"code": "not_found", "message": "unknown asset"}})
            return
        if not path.is_file():
            self._send_json(404, {"ok": False, "error": {"code": "not_found", "message": "ui asset is missing"}})
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        self._send(200, path.read_bytes(), content_type)

    def _serve_media(self, window_id: str, kind: str) -> None:
        path = self.service.media_paths.get((window_id, kind))
        if path is None:
            self._send_json(
                404, {"ok": False, "error": {"code": "not_found", "message": "media path is not allow-listed"}}
            )
            return
        query = parse_qs(urlparse(self.path).query)
        if "wav" in query.get("container", []):
            self._send_media_bytes(self.service.wav_bytes(path), "audio/wav")
            return
        data_path = path
        size = data_path.stat().st_size
        content_type = "audio/flac" if data_path.suffix == ".flac" else "audio/wav"
        range_header = self.headers.get("Range")
        if range_header:
            start, end = _parse_byte_range(range_header, size)
            if start is None:
                self._send(416, b"", content_type, {"Content-Range": f"bytes */{size}"})
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with data_path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    block = handle.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
            return
        if size > MAX_FULL_MEDIA_BYTES:
            self._send_json(
                416, {"ok": False, "error": {"code": "range_required", "message": "media requires a byte range"}}
            )
            return
        self._send(200, data_path.read_bytes(), content_type, {"Accept-Ranges": "bytes"})

    def _send_media_bytes(self, payload: bytes, content_type: str) -> None:
        size = len(payload)
        range_header = self.headers.get("Range")
        if range_header:
            start, end = _parse_byte_range(range_header, size)
            if start is None:
                self._send(416, b"", content_type, {"Content-Range": f"bytes */{size}"})
                return
            self._send(
                206,
                payload[start : end + 1],
                content_type,
                {
                    "Content-Range": f"bytes {start}-{end}/{size}",
                    "Accept-Ranges": "bytes",
                },
            )
            return
        if size > MAX_FULL_MEDIA_BYTES:
            self._send_json(
                416, {"ok": False, "error": {"code": "range_required", "message": "media requires a byte range"}}
            )
            return
        self._send(200, payload, content_type, {"Accept-Ranges": "bytes"})


def _parse_byte_range(header: str, size: int) -> tuple[int | None, int]:
    if not header.startswith("bytes=") or "," in header:
        return None, 0
    spec = header[len("bytes=") :]
    if "-" not in spec:
        return None, 0
    start_text, end_text = spec.split("-", 1)
    try:
        if start_text == "":
            suffix = int(end_text)
            if suffix <= 0:
                return None, 0
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError:
        return None, 0
    if start < 0 or end < start or start >= size:
        return None, 0
    end = min(end, size - 1)
    return start, end


class ReviewHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], service: ReviewService) -> None:
        super().__init__(address, ReviewHandler)
        self.review_service = service


def serve_review(
    session_root: Path,
    *,
    actor_id: str,
    read_only: bool = False,
    signoff_mode: bool = False,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ReviewHTTPServer, str, int]:
    """Bind a localhost review server."""

    if host != "127.0.0.1":
        raise PreparationError("review server host must be 127.0.0.1")
    service = ReviewService(session_root, actor_id=actor_id, read_only=read_only, signoff_mode=signoff_mode)
    server = ReviewHTTPServer((host, port), service)
    bound_host, bound_port = server.server_address[:2]
    return server, str(bound_host), int(bound_port)


def start_review_server(
    session_root: Path,
    *,
    actor_id: str,
    read_only: bool = False,
    signoff_mode: bool = False,
    port: int = 0,
) -> tuple[ReviewHTTPServer, threading.Thread, str, int]:
    """Start the review server on a background thread."""

    server, host, bound_port = serve_review(
        session_root,
        actor_id=actor_id,
        read_only=read_only,
        signoff_mode=signoff_mode,
        port=port,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, host, bound_port


__all__ = ["ReviewService", "serve_review", "start_review_server"]
