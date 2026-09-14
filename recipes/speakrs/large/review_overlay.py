"""Bounded transcript overlay bound to an immutable Open Yap review packet.

The overlay extracts only selected-window source words in one sequential archive
pass, converts machine candidates onto the integer 20 ms grid, and never writes
back into the original packet.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import ContractError, PreparationError
from .hashing import sha256_file, sha256_json
from .jsonio import read_json, write_json
from .review_models import (
    CONVERSION_POLICY,
    FRAME_SECONDS,
    SpeakerRole,
    TranscriptWord,
    clip_words_to_window,
    parse_non_empty_text,
    parse_sha256,
    parse_source_transcript,
    proposal_from_candidate,
    reject_unknown_fields,
    require_mapping,
)
from .review_packet import (
    MAX_JSON_MEMBER_BYTES,
    SPEAKER_ROLES,
    load_archive_inventory,
    validate_review_packet,
)


OVERLAY_SCHEMA = "speakrs-open-yap-review-overlay"
OVERLAY_SCHEMA_VERSION = 1
SESSION_SCHEMA = "speakrs-open-yap-review-session"
SESSION_SCHEMA_VERSION = 1
PROGRESS = Callable[[Mapping[str, object]], None]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise PreparationError("overlay path escapes the overlay root")
    return resolved.relative_to(root_resolved).as_posix()


def _load_packet_manifest(packet_root: Path) -> dict[str, Any]:
    validation = validate_review_packet(packet_root)
    if validation.get("ok") is not True:
        raise PreparationError("review packet failed validation")
    manifest_path = packet_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError("review packet manifest is unreadable") from error
    return require_mapping(manifest, "packet manifest")


def _packet_windows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    windows = manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise PreparationError("review packet has no windows")
    records: list[dict[str, Any]] = []
    for item in windows:
        record = require_mapping(item, "packet window")
        window = require_mapping(record.get("window"), "packet window.window")
        files = require_mapping(record.get("files"), "packet window.files")
        records.append({"window": dict(window), "files": dict(files), "raw": dict(record)})
    return records


def _policy_identity(manifest: Mapping[str, Any], packet_root: Path) -> tuple[str, str, Path]:
    policy = require_mapping(manifest.get("policy"), "packet policy")
    policy_path_value = policy.get("global_policy_path")
    stored_hash = policy.get("global_policy_sha256")
    if not isinstance(policy_path_value, str) or not isinstance(stored_hash, str):
        raise PreparationError("packet policy identity is incomplete")
    policy_path = Path(policy_path_value)
    if not policy_path.is_file():
        bundled = packet_root / "source-procedure.json"
        if not bundled.is_file():
            raise PreparationError("frozen QA policy file is missing")
        # the packet already bound the policy hash; require the live file when present
        raise PreparationError("frozen QA policy file is missing", {"path": policy_path.as_posix()})
    digest = sha256_file(policy_path)
    if digest != stored_hash:
        raise PreparationError("frozen QA policy hash does not match the packet")
    return stored_hash, str(policy.get("policy_id") or ""), policy_path


def _archive_identity(manifest: Mapping[str, Any], archive_path: Path) -> tuple[str, int]:
    archive = require_mapping(manifest.get("archive"), "packet archive")
    expected_sha256 = parse_sha256(archive.get("sha256"), "archive.sha256")
    expected_size = archive.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise PreparationError("packet archive size is invalid")
    if not archive_path.is_file():
        raise PreparationError("source archive is missing", {"path": archive_path.as_posix()})
    observed_size = archive_path.stat().st_size
    if observed_size != expected_size:
        raise PreparationError(
            "source archive size does not match the packet",
            {"expected": expected_size, "observed": observed_size},
        )
    return expected_sha256, expected_size


def _selected_transcript_members(windows: list[dict[str, Any]], inventory: Any) -> dict[str, dict[str, Any]]:
    parents_by_id = {parent.parent_id: parent for parent in inventory.parents}
    members: dict[str, dict[str, Any]] = {}
    for record in windows:
        window = record["window"]
        parent_id = window.get("parent_id")
        parent = parents_by_id.get(parent_id)
        if parent is None:
            raise PreparationError("selected parent is missing from the archive inventory", {"parent_id": parent_id})
        for role in SPEAKER_ROLES:
            member_name = parent.transcript_members[role]
            facts = members.get(member_name)
            record_facts = {
                "parent_id": parent_id,
                "role": role,
                "sha256": parent.member_sha256[f"{role}_transcript.json"],
                "size_bytes": parent.member_sizes_bytes[f"{role}_transcript.json"],
            }
            if facts is None:
                members[member_name] = record_facts
                continue
            if facts != record_facts:
                raise PreparationError("selected transcript member identity is inconsistent", {"member": member_name})
    return members


def _read_allowlisted_member(archive: tarfile.TarFile, member: tarfile.TarInfo, expected_size: int) -> bytes:
    if member.size < 0 or member.size > MAX_JSON_MEMBER_BYTES:
        raise PreparationError(
            "source transcript member exceeds the bounded JSON limit",
            {"size_bytes": member.size, "limit_bytes": MAX_JSON_MEMBER_BYTES},
        )
    if member.size != expected_size:
        raise PreparationError(
            "source transcript member size does not match the inventory",
            {"expected": expected_size, "observed": member.size},
        )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise PreparationError("source transcript member cannot be read")
    parts: list[bytes] = []
    remaining = member.size
    while remaining:
        block = extracted.read(min(1024 * 1024, remaining))
        if not block:
            raise PreparationError("source transcript member ended before its declared size")
        parts.append(block)
        remaining -= len(block)
    if extracted.read(1):
        raise PreparationError("source transcript member is longer than its declared size")
    return b"".join(parts)


def _extract_selected_transcripts(
    archive_path: Path,
    members: Mapping[str, Mapping[str, Any]],
    *,
    progress: PROGRESS | None,
) -> dict[str, dict[str, Any]]:
    remaining = set(members)
    seen: set[str] = set()
    extracted: dict[str, dict[str, Any]] = {}
    scanned = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            scanned += 1
            if progress is not None and scanned % 250 == 0:
                progress(
                    {
                        "stage": "archive_scan",
                        "members_scanned": scanned,
                        "transcripts_remaining": len(remaining),
                    }
                )
            name = member.name
            if name not in members:
                continue
            if name in seen:
                raise PreparationError("source transcript member is duplicated", {"member": name})
            seen.add(name)
            facts = members[name]
            payload = _read_allowlisted_member(archive, member, int(facts["size_bytes"]))
            digest = _sha256_bytes(payload)
            if digest != facts["sha256"]:
                raise PreparationError(
                    "source transcript member hash does not match the inventory",
                    {"member": name},
                )
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PreparationError("source transcript member is not valid UTF-8 JSON") from error
            role = SpeakerRole(str(facts["role"]))
            words = parse_source_transcript(value, speaker=role, source_sha256=digest)
            extracted[name] = {
                "parent_id": facts["parent_id"],
                "role": role,
                "sha256": digest,
                "size_bytes": len(payload),
                "words": words,
            }
            remaining.discard(name)
    if remaining:
        raise PreparationError(
            "selected source transcript members are missing from the archive",
            {"missing_count": len(remaining)},
        )
    if progress is not None:
        progress({"stage": "archive_scan_complete", "members_scanned": scanned, "transcripts": len(extracted)})
    return extracted


def _window_transcript_payload(
    window: Mapping[str, Any],
    extracted: Mapping[str, Mapping[str, Any]],
    parent_members: Mapping[str, str],
) -> dict[str, object]:
    words: list[TranscriptWord] = []
    sources: dict[str, dict[str, object]] = {}
    for role in SPEAKER_ROLES:
        member_name = parent_members[role]
        payload = extracted[member_name]
        clipped = clip_words_to_window(
            payload["words"],
            window_start_seconds=float(window["start_seconds"]),
            window_end_seconds=float(window["end_seconds"]),
        )
        words.extend(clipped)
        sources[role] = {
            "member": member_name,
            "sha256": payload["sha256"],
            "size_bytes": payload["size_bytes"],
        }
    words.sort(key=lambda item: (item.window_start_seconds, item.speaker.value, item.text))
    return {
        "schema": "speakrs-open-yap-window-transcript",
        "schema_version": 1,
        "window_id": window["window_id"],
        "parent_id": window["parent_id"],
        "source_clock": {
            "start_seconds": window["start_seconds"],
            "end_seconds": window["end_seconds"],
        },
        "source_members": sources,
        "display_only": True,
        "not_human_activity": True,
        "words": [word.to_dict() for word in words],
    }


def _write_window_artifacts(
    overlay_root: Path,
    packet_root: Path,
    records: list[dict[str, Any]],
    extracted: Mapping[str, Mapping[str, Any]],
    inventory: Any,
) -> list[dict[str, object]]:
    parents_by_id = {parent.parent_id: parent for parent in inventory.parents}
    window_records: list[dict[str, object]] = []
    for record in records:
        window = record["window"]
        files = record["files"]
        window_id = parse_non_empty_text(window.get("window_id"), "window_id")
        parent = parents_by_id[window["parent_id"]]
        candidate_rel = files["candidate_annotation"]
        candidate_path = (packet_root / str(candidate_rel)).resolve()
        if packet_root.resolve() not in candidate_path.parents:
            raise PreparationError("candidate annotation path is unsafe")
        candidate = read_json(candidate_path)
        candidate_sha256 = sha256_file(candidate_path)
        proposal = proposal_from_candidate(candidate, window_id=window_id, candidate_sha256=candidate_sha256)
        window_dir = overlay_root / "windows" / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        transcript_payload = _window_transcript_payload(window, extracted, parent.transcript_members)
        transcript_path = window_dir / "transcript.json"
        proposal_payload = proposal.to_dict()
        proposal_payload["content_sha256"] = proposal.content_sha256
        proposal_path = window_dir / "proposal.json"
        write_json(transcript_path, transcript_payload)
        write_json(proposal_path, proposal_payload)
        window_records.append(
            {
                "window_id": window_id,
                "parent_id": window["parent_id"],
                "selection_kind": window.get("selection_kind"),
                "stratum": window.get("stratum"),
                "source_clock": {
                    "start_seconds": window["start_seconds"],
                    "end_seconds": window["end_seconds"],
                },
                "packet_files": {
                    "candidate_annotation": str(candidate_rel),
                    "emitted": files.get("emitted"),
                    "reference_speaker_a": files.get("reference_speaker_a"),
                    "reference_speaker_b": files.get("reference_speaker_b"),
                    "human_annotation_template": files.get("human_annotation_template"),
                },
                "candidate_sha256": candidate_sha256,
                "transcript": {
                    "path": _safe_relative(transcript_path, overlay_root),
                    "sha256": sha256_file(transcript_path),
                    "word_count": len(transcript_payload["words"]),
                },
                "proposal": {
                    "path": _safe_relative(proposal_path, overlay_root),
                    "sha256": sha256_file(proposal_path),
                    "content_sha256": proposal.content_sha256,
                    "interval_count": len(proposal.intervals),
                },
            }
        )
    window_records.sort(key=lambda item: str(item["window_id"]))
    return window_records


def _overlay_manifest_payload(
    *,
    packet_root: Path,
    packet_manifest_sha256: str,
    archive_sha256: str,
    archive_size_bytes: int,
    policy_sha256: str,
    policy_id: str,
    window_records: list[dict[str, object]],
    transcript_members: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "schema": OVERLAY_SCHEMA,
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "conversion_policy": CONVERSION_POLICY,
        "frame_seconds": FRAME_SECONDS,
        "packet": {
            "path": packet_root.as_posix(),
            "manifest_sha256": packet_manifest_sha256,
        },
        "archive": {
            "sha256": archive_sha256,
            "size_bytes": archive_size_bytes,
        },
        "policy": {
            "policy_id": policy_id,
            "sha256": policy_sha256,
        },
        "privacy": {
            "transcript_text_retained": True,
            "retention_scope": "selected-window-words-only",
            "full_parent_transcripts_retained": False,
        },
        "membership": {
            "window_count": len(window_records),
            "parent_count": len({item["parent_id"] for item in window_records}),
            "transcript_member_count": len(transcript_members),
        },
        "source_transcript_members": {
            name: {"sha256": facts["sha256"], "size_bytes": facts["size_bytes"], "parent_id": facts["parent_id"]}
            for name, facts in sorted(transcript_members.items())
        },
        "windows": window_records,
    }


def _session_payload(
    *,
    packet_root: Path,
    archive_path: Path,
    overlay_root: Path,
    overlay_sha256: str,
    packet_manifest_sha256: str,
    archive_sha256: str,
    archive_size_bytes: int,
    policy_sha256: str,
) -> dict[str, object]:
    return {
        "schema": SESSION_SCHEMA,
        "schema_version": SESSION_SCHEMA_VERSION,
        "packet_path": packet_root.as_posix(),
        "archive_path": archive_path.as_posix(),
        "overlay_path": "overlay",
        "event_store_path": "events",
        "packet_manifest_sha256": packet_manifest_sha256,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "policy_sha256": policy_sha256,
        "overlay_sha256": overlay_sha256,
        "privacy": {
            "transcript_text_retained": True,
            "retention_scope": "selected-window-words-only",
            "localhost_only": True,
        },
    }


def _remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@dataclass(frozen=True)
class ReviewOverlay:
    """A validated overlay bound to one immutable packet."""

    root: Path
    manifest: Mapping[str, Any]
    overlay_sha256: str

    def window(self, window_id: str) -> dict[str, Any]:
        """Return one overlay window record."""

        for record in self.manifest["windows"]:
            if record["window_id"] == window_id:
                return dict(record)
        raise ContractError("overlay does not contain this window", {"window_id": window_id})

    def window_ids(self) -> tuple[str, ...]:
        return tuple(record["window_id"] for record in self.manifest["windows"])


def validate_review_overlay(root: Path, *, packet_root: Path | None = None) -> dict[str, object]:
    """Validate overlay identities, membership, transcripts, and proposals."""

    overlay_root = Path(root).expanduser().resolve()
    manifest_path = overlay_root / "overlay-manifest.json"
    try:
        manifest = read_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError("review overlay manifest is unreadable") from error
    record = require_mapping(manifest, "overlay manifest")
    if record.get("schema") != OVERLAY_SCHEMA or record.get("schema_version") != OVERLAY_SCHEMA_VERSION:
        raise PreparationError("review overlay schema is invalid")
    reject_unknown_fields(
        record,
        frozenset(
            {
                "schema",
                "schema_version",
                "conversion_policy",
                "frame_seconds",
                "packet",
                "archive",
                "policy",
                "privacy",
                "membership",
                "source_transcript_members",
                "windows",
                "overlay_sha256",
            }
        ),
        "overlay manifest",
    )
    windows = record.get("windows")
    if not isinstance(windows, list) or not windows:
        raise PreparationError("review overlay has no windows")
    packet_info = require_mapping(record.get("packet"), "overlay packet")
    bound_packet = Path(str(packet_info["path"])).expanduser().resolve()
    if packet_root is not None and bound_packet != Path(packet_root).expanduser().resolve():
        raise PreparationError("overlay packet path does not match the requested packet")
    packet_manifest_sha256 = sha256_file(bound_packet / "manifest.json")
    if packet_manifest_sha256 != packet_info.get("manifest_sha256"):
        raise PreparationError("original packet manifest changed after overlay preparation")
    packet_manifest = _load_packet_manifest(bound_packet)
    packet_windows = {item["window"]["window_id"]: item for item in _packet_windows(packet_manifest)}
    if len(windows) != len(packet_windows):
        raise PreparationError("overlay window membership does not match the packet")
    seen: set[str] = set()
    for item in windows:
        window = require_mapping(item, "overlay window")
        window_id = parse_non_empty_text(window.get("window_id"), "window_id")
        if window_id in seen:
            raise PreparationError("overlay window IDs must be unique")
        seen.add(window_id)
        packet_window = packet_windows.get(window_id)
        if packet_window is None:
            raise PreparationError("overlay contains a window that is not in the packet", {"window_id": window_id})
        transcript_rel = require_mapping(window.get("transcript"), "overlay transcript")["path"]
        proposal_rel = require_mapping(window.get("proposal"), "overlay proposal")["path"]
        transcript_path = (overlay_root / str(transcript_rel)).resolve()
        proposal_path = (overlay_root / str(proposal_rel)).resolve()
        if overlay_root not in transcript_path.parents or overlay_root not in proposal_path.parents:
            raise PreparationError("overlay artifact path is unsafe")
        if sha256_file(transcript_path) != window["transcript"]["sha256"]:
            raise PreparationError("overlay transcript hash does not match")
        if sha256_file(proposal_path) != window["proposal"]["sha256"]:
            raise PreparationError("overlay proposal hash does not match")
        transcript = read_json(transcript_path)
        require_mapping(transcript, "window transcript")
        if transcript.get("not_human_activity") is not True or transcript.get("display_only") is not True:
            raise PreparationError("overlay transcript must remain a display aid")
        for word in transcript.get("words", []):
            TranscriptWord.from_dict(word)
        candidate_path = bound_packet / str(packet_window["files"]["candidate_annotation"])
        candidate_sha256 = sha256_file(candidate_path)
        if candidate_sha256 != window.get("candidate_sha256"):
            raise PreparationError("overlay candidate hash does not match the packet file")
        rebuilt = proposal_from_candidate(
            read_json(candidate_path),
            window_id=window_id,
            candidate_sha256=candidate_sha256,
        )
        stored_proposal = read_json(proposal_path)
        if rebuilt.content_sha256 != stored_proposal.get("content_sha256"):
            raise PreparationError("overlay proposal is not a deterministic conversion of the candidate")
    identity = dict(record)
    identity.pop("overlay_sha256", None)
    overlay_sha256 = sha256_json(identity)
    stored = record.get("overlay_sha256")
    if stored is not None and stored != overlay_sha256:
        raise PreparationError("overlay identity hash does not match")
    return {
        "ok": True,
        "overlay_sha256": overlay_sha256,
        "window_count": len(windows),
        "parent_count": record["membership"]["parent_count"],
        "transcript_text_retained": True,
        "packet_manifest_sha256": packet_manifest_sha256,
    }


def load_review_overlay(root: Path) -> ReviewOverlay:
    """Load a validated overlay."""

    overlay_root = Path(root).expanduser().resolve()
    result = validate_review_overlay(overlay_root)
    manifest = read_json(overlay_root / "overlay-manifest.json")
    return ReviewOverlay(root=overlay_root, manifest=manifest, overlay_sha256=str(result["overlay_sha256"]))


def _matching_complete_overlay(destination: Path, expected: Mapping[str, str]) -> bool:
    overlay_root = destination / "overlay"
    if not overlay_root.is_dir():
        return False
    try:
        loaded = load_review_overlay(overlay_root)
    except (ContractError, PreparationError, OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    packet = loaded.manifest.get("packet", {})
    archive = loaded.manifest.get("archive", {})
    policy = loaded.manifest.get("policy", {})
    return (
        packet.get("manifest_sha256") == expected["packet_manifest_sha256"]
        and archive.get("sha256") == expected["archive_sha256"]
        and policy.get("sha256") == expected["policy_sha256"]
    )


def prepare_review_overlay(
    packet: Path,
    archive: Path,
    destination: Path,
    *,
    progress: PROGRESS | None = None,
) -> dict[str, object]:
    """Create a versioned overlay and empty review session for one packet."""

    packet_root = Path(packet).expanduser().resolve()
    archive_path = Path(archive).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if progress is not None:
        progress({"stage": "validate_packet"})
    packet_manifest = _load_packet_manifest(packet_root)
    packet_manifest_sha256 = sha256_file(packet_root / "manifest.json")
    archive_sha256, archive_size = _archive_identity(packet_manifest, archive_path)
    policy_sha256, policy_id, _policy_path = _policy_identity(packet_manifest, packet_root)
    expected = {
        "packet_manifest_sha256": packet_manifest_sha256,
        "archive_sha256": archive_sha256,
        "policy_sha256": policy_sha256,
    }
    if output.exists() and _matching_complete_overlay(output, expected):
        session = read_json(output / "session.json")
        if progress is not None:
            progress({"stage": "reuse_complete_overlay", "overlay_sha256": session.get("overlay_sha256")})
        return {
            "ok": True,
            "reused": True,
            "session_path": output.as_posix(),
            "overlay_sha256": session["overlay_sha256"],
            "window_count": session_window_count(output),
        }
    if output.exists():
        events_dir = output / "events"
        if events_dir.exists() and any(events_dir.iterdir()):
            raise PreparationError("refusing to replace a review session that already has events")
        raise PreparationError(
            "destination exists and is not a matching complete overlay",
            {"path": output.as_posix()},
        )
    inventory_path = packet_root / "archive-inventory.json"
    inventory = load_archive_inventory(
        inventory_path,
        archive_path=archive_path,
        expected_sha256=archive_sha256,
        expected_size_bytes=archive_size,
    )
    windows = _packet_windows(packet_manifest)
    members = _selected_transcript_members(windows, inventory)
    if progress is not None:
        progress({"stage": "extract_transcripts", "member_count": len(members)})
    extracted = _extract_selected_transcripts(archive_path, members, progress=progress)
    partial = output.with_name(output.name + ".partial")
    _remove_if_exists(partial)
    overlay_root = partial / "overlay"
    overlay_root.mkdir(parents=True)
    (partial / "events").mkdir()
    window_records = _write_window_artifacts(overlay_root, packet_root, windows, extracted, inventory)
    manifest_payload = _overlay_manifest_payload(
        packet_root=packet_root,
        packet_manifest_sha256=packet_manifest_sha256,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size,
        policy_sha256=policy_sha256,
        policy_id=policy_id,
        window_records=window_records,
        transcript_members=members,
    )
    overlay_sha256 = sha256_json(manifest_payload)
    manifest_payload["overlay_sha256"] = overlay_sha256
    write_json(overlay_root / "overlay-manifest.json", manifest_payload)
    write_json(
        partial / "session.json",
        _session_payload(
            packet_root=packet_root,
            archive_path=archive_path,
            overlay_root=overlay_root,
            overlay_sha256=overlay_sha256,
            packet_manifest_sha256=packet_manifest_sha256,
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size,
            policy_sha256=policy_sha256,
        ),
    )
    validate_review_overlay(overlay_root, packet_root=packet_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(output)
    if progress is not None:
        progress({"stage": "overlay_complete", "overlay_sha256": overlay_sha256, "window_count": len(window_records)})
    return {
        "ok": True,
        "reused": False,
        "session_path": output.as_posix(),
        "overlay_sha256": overlay_sha256,
        "window_count": len(window_records),
        "parent_count": len({item["parent_id"] for item in window_records}),
        "transcript_member_count": len(members),
        "transcript_text_retained": True,
    }


def session_window_count(session_root: Path) -> int:
    """Return the overlay window count for a session directory."""

    overlay = load_review_overlay(Path(session_root) / "overlay")
    return len(overlay.window_ids())


def load_review_session(path: Path) -> dict[str, Any]:
    """Load and validate a review session directory."""

    root = Path(path).expanduser().resolve()
    session = require_mapping(read_json(root / "session.json"), "review session")
    if session.get("schema") != SESSION_SCHEMA or session.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise PreparationError("review session schema is invalid")
    overlay = load_review_overlay(root / str(session["overlay_path"]))
    if overlay.overlay_sha256 != session.get("overlay_sha256"):
        raise PreparationError("session overlay hash does not match the overlay manifest")
    packet_root = Path(str(session["packet_path"])).expanduser().resolve()
    if sha256_file(packet_root / "manifest.json") != session.get("packet_manifest_sha256"):
        raise PreparationError("session packet manifest changed")
    return {"root": root, "session": session, "overlay": overlay, "packet_root": packet_root}


__all__ = [
    "OVERLAY_SCHEMA",
    "ReviewOverlay",
    "load_review_overlay",
    "load_review_session",
    "prepare_review_overlay",
    "validate_review_overlay",
]
