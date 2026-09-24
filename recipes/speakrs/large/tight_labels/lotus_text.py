"""LOTUSDIS TextGrid intervals with speaker lists and transcript text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .time import float_bounds_to_rttm_ns


_SPEECH_MARK_RE = re.compile(r"^([FM]\d{2}(?:&[FM]\d{2})*),(.*)$", re.DOTALL)
_NON_SPEECH_MARKS = frozenset({"<n>", "<sil>", "<unk>", "<td>"})
_INTERVAL_RE = re.compile(r"^\s*intervals\s*\[\s*(\d+)\s*\]:\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class LotusAlignInterval:
    """One TextGrid interval with speakers, text, and RTTM-matching bounds."""

    start_ns: int
    end_ns: int
    speakers: tuple[str, ...]
    text: str
    raw_mark: str
    non_speech: bool

    @property
    def overlap(self) -> bool:
        """Return True when more than one speaker is listed."""

        return len(self.speakers) > 1


def parse_lotus_align_intervals(source: str | Path) -> tuple[LotusAlignInterval, ...]:
    """Parse the publisher interval tier and keep transcript text."""

    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).is_file()):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_INTERVAL_RE.finditer(text))
    intervals: list[LotusAlignInterval] = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        start = _number(_field(block, "xmin"))
        stop = _number(_field(block, "xmax"))
        mark = _quoted(_field(block, "text")).strip()
        start_ns, end_ns = float_bounds_to_rttm_ns(start, stop)
        if not mark or mark in _NON_SPEECH_MARKS:
            intervals.append(LotusAlignInterval(start_ns, end_ns, (), "", mark, True))
            continue
        found = _SPEECH_MARK_RE.fullmatch(mark)
        if found is None:
            intervals.append(LotusAlignInterval(start_ns, end_ns, (), mark, mark, True))
            continue
        speakers = tuple(found.group(1).split("&"))
        intervals.append(
            LotusAlignInterval(start_ns, end_ns, speakers, found.group(2).strip(), mark, False)
        )
    return tuple(intervals)


def _field(block: str, name: str) -> str:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(.*?)\s*$", block, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"TextGrid interval is missing {name}")
    return match.group(1).strip()


def _quoted(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        body = value[3:-3] if value.startswith('"""') and value.endswith('"""') and len(value) >= 6 else value[1:-1]
        return body.replace('""', '"')
    return value


def _number(value: str) -> float:
    return float(value)
