"""Transcript markers that are not alignable speech."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence


ANGLE_TAG_RE = re.compile(r"<[^>]*>")
SQUARE_TAG_RE = re.compile(r"\[[^\]]*\]")
TAG_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")
ASCII_GLOSS_RE = re.compile(
    r"(?<=[^\x00-\x7F])\s*\(\s*(?=[A-Za-z0-9' .,-]*[A-Za-z])[A-Za-z0-9' .,-]*\s*\)"
)
PAREN_NOISE_RE = re.compile(r"\(\s*(unintelligible|inaudible)\s*\)", re.IGNORECASE)
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class TranscriptText:
    """Speech remainder after non-alignable markers are removed."""

    original: str
    remainder: str
    marker_only: bool
    has_marker: bool
    stripped_tags: tuple[str, ...]
    stripped_glosses: tuple[str, ...]

    @property
    def alignable(self) -> bool:
        """Return True when leftover text still contains a letter."""

        return bool(self.remainder) and LETTER_RE.search(self.remainder) is not None


def tag_label(raw: str) -> str:
    """Return the normalised inner text of a bracket or angle-bracket tag."""

    text = raw.strip()
    if len(text) >= 2 and text[0] in "<[" and text[-1] in ">]":
        text = text[1:-1]
    return re.sub(r"\s+", " ", text).strip().lower()


def _keep_square_inner(match: re.Match[str]) -> str:
    inner = tag_label(match.group(0))
    return f" {inner} " if inner else " "


def prepare_transcript(text: str, *, keep_square_bracket_words: bool = False) -> TranscriptText:
    """Strip non-speech marks and native-script ASCII glosses.

    Indic square brackets are tags. LOTUSDIS square brackets mark spoken
    English loanwords, so keep_square_bracket_words keeps their inner text.
    """

    original = text if isinstance(text, str) else ""
    if keep_square_bracket_words:
        tags = tuple(tag_label(item) for item in ANGLE_TAG_RE.findall(original) if tag_label(item))
        stripped = ANGLE_TAG_RE.sub(" ", original)
        stripped = SQUARE_TAG_RE.sub(_keep_square_inner, stripped)
    else:
        tags = tuple(tag_label(item) for item in TAG_RE.findall(original) if tag_label(item))
        stripped = TAG_RE.sub(" ", original)
    glosses = tuple(item.strip() for item in ASCII_GLOSS_RE.findall(original) if item.strip())
    stripped = ASCII_GLOSS_RE.sub("", stripped)
    stripped, noise_count = PAREN_NOISE_RE.subn(" ", stripped)
    remainder = re.sub(r"\s+", " ", stripped).strip()
    has_letter = LETTER_RE.search(remainder) is not None
    has_marker = bool(tags) or bool(glosses) or noise_count > 0
    marker_only = (not has_letter) and (has_marker or not remainder)
    if not original.strip():
        marker_only = True
    return TranscriptText(original, remainder, marker_only, has_marker, tags, glosses)


def transcript_cleaning_row(prepared: TranscriptText) -> dict[str, object]:
    """Return tag and gloss counts for one cleaned transcript row."""

    return {
        "tags": prepared.stripped_tags,
        "glosses": prepared.stripped_glosses,
        "marker_only": prepared.marker_only,
        "token_count": len(TOKEN_RE.findall(prepared.remainder)),
    }


def summarise_cleaning(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate per-language row, token, tag, and gloss counts."""

    tag_counts: Counter[str] = Counter()
    gloss_rows = 0
    marker_only_rows = 0
    total_rows = 0
    for row in rows:
        total_rows += 1
        if row.get("glosses"):
            gloss_rows += 1
        if row.get("marker_only"):
            marker_only_rows += 1
        for tag in row.get("tags") or ():
            tag_counts[str(tag)] += 1
    return {
        "rows": total_rows,
        "gloss_rows": gloss_rows,
        "marker_only_rows": marker_only_rows,
        "tags": dict(tag_counts.most_common()),
    }


def cleaning_stats_by_language(
    transcripts: Mapping[str, Sequence[object]],
    *,
    keep_square_bracket_words: bool = False,
) -> dict[str, dict[str, object]]:
    """Return current cleaning counts grouped by transcript language."""

    by_language: dict[str, list[dict[str, object]]] = {}
    for rows in transcripts.values():
        for item in rows:
            language = getattr(item, "language", None) or "unknown"
            text = getattr(item, "text", "")
            prepared = prepare_transcript(text, keep_square_bracket_words=keep_square_bracket_words)
            by_language.setdefault(str(language), []).append(transcript_cleaning_row(prepared))
    return {language: summarise_cleaning(rows) for language, rows in sorted(by_language.items())}


__all__ = [
    "TranscriptText",
    "cleaning_stats_by_language",
    "prepare_transcript",
    "summarise_cleaning",
    "tag_label",
    "transcript_cleaning_row",
]
