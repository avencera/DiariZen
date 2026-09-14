#!/usr/bin/env python3
"""Thin CLI for Open Yap review overlay preparation, validation, and export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _print(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Private Open Yap review overlay, validation, and export. "
            "Serve the review UI with: cargo run --release -p open-yap-review -- serve "
            "--session PATH --reviewer-id ID"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build a transcript overlay and empty event store")
    prepare.add_argument("--packet", required=True, type=Path)
    prepare.add_argument("--archive", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    validate = sub.add_parser("validate", help="validate a review session or completed export")
    validate.add_argument("--session", required=True, type=Path)
    validate.add_argument("--export", type=Path)

    export = sub.add_parser("export", help="write a versioned human-annotation export")
    export.add_argument("--session", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    from recipes.speakrs.large.errors import LargeError
    from recipes.speakrs.large.review_export import export_review, validate_completed_review
    from recipes.speakrs.large.review_overlay import load_review_session, prepare_review_overlay
    from recipes.speakrs.large.review_store import open_review_store

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_review_overlay(args.packet, args.archive, args.output, progress=_print)
            _print(result)
            return 0
        if args.command == "validate":
            loaded = load_review_session(args.session)
            overlay = loaded["overlay"]
            store = open_review_store(args.session)
            payload = {
                "ok": True,
                "session_path": Path(args.session).expanduser().resolve().as_posix(),
                "window_count": len(overlay.window_ids()),
                "parent_count": overlay.manifest["membership"]["parent_count"],
                "overlay_sha256": overlay.overlay_sha256,
                "event_count": len(store.events()),
                "quarantined": store.quarantined,
                "progress": store.progress().as_dict(),
            }
            if args.export is not None:
                payload["completed"] = validate_completed_review(args.session, args.export)
            _print(payload)
            return 0
        store = open_review_store(args.session)
        _print(export_review(store))
        return 0
    except LargeError as error:
        _print(error.to_json())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
