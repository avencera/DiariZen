"""Transcript markers that are not alignable speech."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence


TAG_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")
ASCII_GLOSS_RE = re.compile(r"(?<=[^\x00-\x7F])\s*\(\s*[A-Za-z][A-Za-z0-9' .,-]*\s*\)")
PAREN_NOISE_RE = re.compile(r"\(\s*(unintelligible|inaudible)\s*\)", re.IGNORECASE)
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
TOKEN_RE = re.compile(r"\S+")
LEGACY_MARKER_RE = re.compile(
    r"<\s*/?\s*(unintelligible|inaudible|unk(?:nown)?|noise|n|sil(?:ence)?|"
    r"laugh(?:ter)?|foreign|crosstalk|overlap|td)\s*>"
    r"|\[\s*(unintelligible|inaudible|unk(?:nown)?|noise|laughter|foreign|"
    r"topic|[mf])\s*\]"
    r"|\(\s*(unintelligible|inaudible)\s*\)",
    re.IGNORECASE,
)


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


def prepare_transcript(text: str) -> TranscriptText:
    """Strip bracket tags, angle-bracket tags, and native-script ASCII glosses."""

    original = text if isinstance(text, str) else ""
    tags = tuple(tag_label(item) for item in TAG_RE.findall(original) if tag_label(item))
    glosses = tuple(item.strip() for item in ASCII_GLOSS_RE.findall(original) if item.strip())
    stripped = TAG_RE.sub(" ", original)
    stripped = ASCII_GLOSS_RE.sub("", stripped)
    stripped, noise_count = PAREN_NOISE_RE.subn(" ", stripped)
    remainder = re.sub(r"\s+", " ", stripped).strip()
    has_letter = LETTER_RE.search(remainder) is not None
    has_marker = bool(tags) or bool(glosses) or noise_count > 0
    marker_only = (not has_letter) and (has_marker or not remainder)
    if not original.strip():
        marker_only = True
    return TranscriptText(original, remainder, marker_only, has_marker, tags, glosses)


def _legacy_remainder(text: str) -> str:
    stripped = LEGACY_MARKER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", stripped).strip()


def cleaning_delta(text: str) -> dict[str, object]:
    """Compare generic tag cleaning with the previous listed-tag strip."""

    prepared = prepare_transcript(text)
    legacy = _legacy_remainder(text)
    old_tokens = TOKEN_RE.findall(legacy)
    new_tokens = TOKEN_RE.findall(prepared.remainder)
    return {
        "row_changed": prepared.remainder != legacy,
        "token_delta": abs(len(new_tokens) - len(old_tokens)),
        "old_token_count": len(old_tokens),
        "new_token_count": len(new_tokens),
        "tags": prepared.stripped_tags,
        "glosses": prepared.stripped_glosses,
    }


def summarise_cleaning(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate per-language row, token, tag, and gloss counts."""

    tag_counts: Counter[str] = Counter()
    gloss_rows = 0
    changed_rows = 0
    changed_tokens = 0
    total_rows = 0
    for row in rows:
        total_rows += 1
        if row.get("row_changed"):
            changed_rows += 1
        changed_tokens += int(row.get("token_delta") or 0)
        if row.get("glosses"):
            gloss_rows += 1
        for tag in row.get("tags") or ():
            tag_counts[str(tag)] += 1
    return {
        "rows": total_rows,
        "changed_rows": changed_rows,
        "changed_tokens": changed_tokens,
        "gloss_rows": gloss_rows,
        "tags": dict(tag_counts.most_common()),
    }


def cleaning_stats_by_language(
    transcripts: Mapping[str, Sequence[object]],
) -> dict[str, dict[str, object]]:
    """Return cleaning deltas grouped by transcript language."""

    by_language: dict[str, list[dict[str, object]]] = {}
    for rows in transcripts.values():
        for item in rows:
            language = getattr(item, "language", None) or "unknown"
            text = getattr(item, "text", "")
            by_language.setdefault(str(language), []).append(cleaning_delta(text))
    return {language: summarise_cleaning(rows) for language, rows in sorted(by_language.items())}


__all__ = [
    "TranscriptText",
    "cleaning_delta",
    "cleaning_stats_by_language",
    "prepare_transcript",
    "summarise_cleaning",
    "tag_label",
]
