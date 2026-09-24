"""Transcript markers that are not alignable speech."""

from __future__ import annotations

import re
from dataclasses import dataclass


MARKER_RE = re.compile(
    r"<\s*/?\s*(unintelligible|inaudible|unk(?:nown)?|noise|n|sil(?:ence)?|"
    r"laugh(?:ter)?|foreign|crosstalk|overlap|td)\s*>"
    r"|\[\s*(unintelligible|inaudible|unk(?:nown)?|noise|laughter|foreign|"
    r"topic|[mf])\s*\]"
    r"|\(\s*(unintelligible|inaudible)\s*\)",
    re.IGNORECASE,
)
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TranscriptText:
    """Speech remainder after non-alignable markers are removed."""

    original: str
    remainder: str
    marker_only: bool
    has_marker: bool

    @property
    def alignable(self) -> bool:
        """Return True when leftover text still contains a letter."""

        return bool(self.remainder) and LETTER_RE.search(self.remainder) is not None


def prepare_transcript(text: str) -> TranscriptText:
    """Strip unintelligible, noise, laughter, and foreign-speech markers."""

    original = text if isinstance(text, str) else ""
    stripped, count = MARKER_RE.subn(" ", original)
    remainder = re.sub(r"\s+", " ", stripped).strip()
    has_letter = LETTER_RE.search(remainder) is not None
    marker_only = (not has_letter) and (count > 0 or not remainder)
    if not original.strip():
        marker_only = True
    return TranscriptText(original, remainder, marker_only, count > 0)
