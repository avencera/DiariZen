"""Speakrs WavLM Large recipe package.

Public commands enter through ``recipes/speakrs/large_run.py``. Shared trainer
code must import typed state from this package only through injected objects,
never through recipe filesystem paths.
"""

from __future__ import annotations


PACKAGE_NAME = "speakrs.large"
SCHEMA_VERSION = 1
