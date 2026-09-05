#!/usr/bin/env python3
"""Thin CLI for the Speakrs WavLM Large recipe."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv=None) -> int:
    """Dispatch to the recipe command package."""

    from recipes.speakrs.large.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
