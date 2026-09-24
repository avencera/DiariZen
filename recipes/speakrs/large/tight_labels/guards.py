"""Non-speech and invented-text guards for VoxConverse segment ASR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


SAMPLE_RATE = 16_000
FRAME_MS = 30
HOP_MS = 10
VOICED_RMS_DBFS = -42.0
SILENCE_MEAN_DBFS = -48.0
SILENCE_VOICED_MAX = 0.10
MUSIC_VOICED_MAX = 0.22
MUSIC_MEAN_DBFS_MIN = -38.0
MUSIC_MODULATION_MAX = 0.15
MIN_ALIGN_SECONDS = 0.080
MIN_DURATION_FOR_MUSIC_S = 0.60
MAX_CHARS_PER_SEC = 25.0
MAX_WORDS_PER_SEC = 8.0
MIN_DURATION_FOR_CHAR_RATE_S = 0.40
MIN_DURATION_FOR_WORD_RATE_S = 0.50
MIN_WORDS_FOR_RATE = 4
REPETITION_FRACTION = 0.60
REPETITION_MIN_TOKENS = 8
EPS = 1e-12
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", re.UNICODE)
ENGLISH_KEEP_RE = re.compile(r"[^a-z0-9']+")
MMS_KEEP_RE = re.compile(r"[^a-z']+")

GUARD_RULES = {
    "TOO_SHORT": ("Human segment duration is below 80 ms, so ASR and alignment cannot place words with confidence."),
    "SILENCE": (
        "Mean RMS is below -48 dBFS and the share of 30 ms frames at or above "
        "-42 dBFS is below 0.10. The segment is treated as silence."
    ),
    "MUSIC_OR_NOISE": (
        "On a cut of at least 0.60 s, mean RMS is at or above -38 dBFS and "
        "3-8 Hz envelope modulation is below 0.15 of 0.5-20 Hz energy, or "
        "the voiced-frame share is below 0.22 with the same energy floor. "
        "The segment is treated as music or noise."
    ),
    "EMPTY_ASR": "ASR returned no word tokens after punctuation stripping.",
    "IMPLAUSIBLE_CHAR_RATE": (
        "Letter and digit count divided by duration exceeds 25 per second on a segment of at least 0.40 s."
    ),
    "IMPLAUSIBLE_WORD_RATE": (
        "Word count divided by duration exceeds 8 per second on a segment of at least 0.50 s with at least 4 words."
    ),
    "REPETITION_LOOP": (
        "The most frequent token is at least 60% of tokens and the token "
        "count is at least 8. This pattern matches ASR hallucination loops."
    ),
    "ASR_FAILED": "The ASR model raised an error on this segment cut.",
    "LID_UNSUPPORTED": (
        "Spoken-language identification is outside Parakeet's 25 languages and no fallback ASR covers the language."
    ),
    "G2P_FAILED": "MFA G2P could not produce a pronunciation for at least one word.",
    "ALIGN_FAILED": "Forced alignment returned no word intervals or raised an error.",
    "NO_ALIGNABLE_WORDS": "Normalized ASR text contained no alignable word tokens.",
}


@dataclass(frozen=True, slots=True)
class AcousticFeatures:
    """Energy and modulation measurements for one segment cut."""

    duration_s: float
    mean_rms_dbfs: float
    voiced_fraction: float
    modulation_ratio: float
    frame_count: int


@dataclass(frozen=True, slots=True)
class TranscriptStats:
    """Length and repetition measurements for one ASR string."""

    text: str
    char_count: int
    word_count: int
    char_rate: float
    word_rate: float
    top_token_fraction: float
    tokens: tuple[str, ...]


def _frame_params(sample_rate: int) -> tuple[int, int]:
    frame = max(1, int(sample_rate * FRAME_MS / 1000))
    hop = max(1, int(sample_rate * HOP_MS / 1000))
    return frame, hop


def rms_dbfs(samples: np.ndarray) -> float:
    """Return the full-cut RMS level in dBFS."""

    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64))) + EPS))
    return 20.0 * np.log10(rms + EPS)


def frame_rms_dbfs(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return per-frame RMS in dBFS for 30 ms frames with a 10 ms hop."""

    frame, hop = _frame_params(sample_rate)
    if samples.size < frame:
        return np.array([rms_dbfs(samples)], dtype=np.float64)
    window = np.hanning(frame)
    views = np.lib.stride_tricks.sliding_window_view(samples.astype(np.float64), frame)[::hop]
    rms = np.sqrt(np.mean(np.square(views * window), axis=1) + EPS)
    return 20.0 * np.log10(rms + EPS)


def voiced_fraction(frame_dbfs: np.ndarray, threshold_dbfs: float = VOICED_RMS_DBFS) -> float:
    """Return the share of frames at or above the voicing RMS threshold."""

    if frame_dbfs.size == 0:
        return 0.0
    return float(np.mean(frame_dbfs >= threshold_dbfs))


def speech_modulation_ratio(frame_dbfs: np.ndarray, hop_s: float) -> float:
    """Return 3-8 Hz envelope energy divided by 0.5-20 Hz envelope energy.

    Speech has a syllabic peak near 4 Hz. Music and noise usually do not.
    """

    if frame_dbfs.size < 16 or hop_s <= 0:
        return 0.0
    envelope = np.power(10.0, frame_dbfs / 20.0)
    envelope = envelope - np.mean(envelope)
    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(envelope.size, d=hop_s)
    band = (freqs >= 0.5) & (freqs <= 20.0)
    speech = (freqs >= 3.0) & (freqs <= 8.0)
    denom = float(np.sum(spectrum[band]) + EPS)
    numer = float(np.sum(spectrum[speech]))
    return numer / denom


def measure_acoustic(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> AcousticFeatures:
    """Measure energy and modulation on a mono float cut."""

    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=-1)
    duration_s = float(mono.size / sample_rate) if sample_rate else 0.0
    frames = frame_rms_dbfs(mono, sample_rate)
    hop_s = HOP_MS / 1000.0
    return AcousticFeatures(
        duration_s=duration_s,
        mean_rms_dbfs=rms_dbfs(mono),
        voiced_fraction=voiced_fraction(frames),
        modulation_ratio=speech_modulation_ratio(frames, hop_s),
        frame_count=int(frames.size),
    )


def acoustic_guard(features: AcousticFeatures) -> str | None:
    """Return a non-speech guard name, or None when the cut may be speech."""

    if features.duration_s < MIN_ALIGN_SECONDS:
        return "TOO_SHORT"
    if features.mean_rms_dbfs < SILENCE_MEAN_DBFS and features.voiced_fraction < SILENCE_VOICED_MAX:
        return "SILENCE"
    if features.duration_s >= MIN_DURATION_FOR_MUSIC_S and features.mean_rms_dbfs >= MUSIC_MEAN_DBFS_MIN:
        low_modulation = features.modulation_ratio < MUSIC_MODULATION_MAX
        low_voicing = features.voiced_fraction < MUSIC_VOICED_MAX
        if low_modulation or low_voicing:
            return "MUSIC_OR_NOISE"
    return None


def tokenize_words(text: str) -> tuple[str, ...]:
    """Split ASR text into word tokens, dropping punctuation-only pieces."""

    return tuple(match.group(0).lower().replace("’", "'") for match in WORD_RE.finditer(text or ""))


def letter_digit_count(text: str) -> int:
    """Count letters and digits in an ASR string."""

    return sum(1 for char in text if char.isalnum())


def measure_transcript(text: str, duration_s: float) -> TranscriptStats:
    """Measure ASR length against the human segment duration."""

    tokens = tokenize_words(text)
    char_count = letter_digit_count(text)
    word_count = len(tokens)
    safe_duration = duration_s if duration_s > 0 else 0.0
    top_fraction = 0.0
    if tokens:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        top_fraction = max(counts.values()) / len(tokens)
    return TranscriptStats(
        text=text,
        char_count=char_count,
        word_count=word_count,
        char_rate=(char_count / safe_duration) if safe_duration else float("inf"),
        word_rate=(word_count / safe_duration) if safe_duration else float("inf"),
        top_token_fraction=top_fraction,
        tokens=tokens,
    )


def classify_transcript(text: str, duration_s: float) -> str | None:
    """Guard ASR text against empty and implausible length or loops."""

    stats = measure_transcript(text, duration_s)
    if stats.word_count == 0:
        return "EMPTY_ASR"
    if duration_s >= MIN_DURATION_FOR_CHAR_RATE_S and stats.char_rate > MAX_CHARS_PER_SEC:
        return "IMPLAUSIBLE_CHAR_RATE"
    if (
        duration_s >= MIN_DURATION_FOR_WORD_RATE_S
        and stats.word_count >= MIN_WORDS_FOR_RATE
        and stats.word_rate > MAX_WORDS_PER_SEC
    ):
        return "IMPLAUSIBLE_WORD_RATE"
    if stats.top_token_fraction >= REPETITION_FRACTION and len(stats.tokens) >= REPETITION_MIN_TOKENS:
        return "REPETITION_LOOP"
    return None


def normalize_english_words(text: str) -> tuple[str, ...]:
    """Lowercase English ASR text and keep letters, digits, and apostrophes."""

    cleaned = (text or "").lower().replace("’", "'").replace("`", "'")
    cleaned = ENGLISH_KEEP_RE.sub(" ", cleaned)
    return tuple(token for token in cleaned.split() if token and token != "'")


def normalize_mms_words(text: str, romanized: str | None = None) -> tuple[str, ...]:
    """Keep romanized letters for MMS CTC alignment."""

    source = romanized if romanized is not None else text
    cleaned = (source or "").lower().replace("’", "'")
    cleaned = MMS_KEEP_RE.sub(" ", cleaned)
    return tuple(token for token in cleaned.split() if token and token != "'")


def dictionary_words(lines: Iterable[str]) -> frozenset[str]:
    """Return word forms from an MFA pronunciation dictionary."""

    words: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        word = line.split()[0]
        words.add(word)
    return frozenset(words)
