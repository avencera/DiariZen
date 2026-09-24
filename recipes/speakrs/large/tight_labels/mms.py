"""MMS forced alignment with uroman romanization."""

from __future__ import annotations

import hashlib
import inspect
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from uroman import Uroman

from .domain import Interval
from .time import NANOSECONDS_PER_SECOND, parse_seconds_to_ns


MMS_SAMPLE_RATE = 16_000
MMS_CHARSET = frozenset("abcdefghijklmnopqrstuvwxyz'")
MMS_MODEL_ID = "torchaudio.pipelines.MMS_FA"
MIN_ALIGN_SAMPLES = 1_600
POOR_ROMAN_EMPTY_RATE = 0.30
POOR_ROMAN_LEFTOVER_RATE = 0.05
POOR_ROMAN_NAP_RATE = 0.40

INDIC_UROMAN_LCODES = {
    "Assamese": "asm",
    "Bengali": "ben",
    "Bodo": "brx",
    "Dogri": "doi",
    "Gujarati": "guj",
    "Hindi": "hin",
    "Kannada": "kan",
    "Kashmiri": "kas",
    "Konkani": "kok",
    "Maithili": "mai",
    "Malayalam": "mal",
    "Manipuri": "mni",
    "Marathi": "mar",
    "Nepali": "nep",
    "Odia": "ori",
    "Punjabi": "pan",
    "Sanskrit": "san",
    "Santali": "sat",
    "Sindhi": "snd",
    "Tamil": "tam",
    "Telugu": "tel",
    "Urdu": "urd",
}
THAI_UROMAN_LCODE = "tha"

_NON_LATIN_SCRIPTS = frozenset(
    {
        "DEVANAGARI",
        "BENGALI",
        "GURMUKHI",
        "GUJARATI",
        "ORIYA",
        "TAMIL",
        "TELUGU",
        "KANNADA",
        "MALAYALAM",
        "MEETEI MAYEK",
        "OL CHIKI",
        "ARABIC",
        "THAI",
    }
)


@dataclass(frozen=True, slots=True)
class AlignedWord:
    """One romanized word placed on the source timeline."""

    start_ns: int
    end_ns: int
    word: str
    score: float

    @property
    def bounds(self) -> Interval:
        """Return the unlabelled span."""

        return Interval(self.start_ns, self.end_ns)


@dataclass(frozen=True, slots=True)
class Romanization:
    """Romanized text and the MMS word tokens taken from it."""

    original: str
    romanized: str
    words: tuple[str, ...]
    leftover_script_rate: float
    empty: bool


@dataclass(frozen=True, slots=True)
class LanguageCoverage:
    """Per-language romanizer quality measured on transcript remainders."""

    language: str
    lcode: str | None
    segment_count: int
    letter_segment_count: int
    empty_roman_count: int
    leftover_script_count: int
    empty_roman_rate: float
    leftover_script_rate: float
    expansion_ratio: float
    nap_junk_rate: float
    poor: bool
    samples: tuple[tuple[str, str], ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_device() -> str:
    """Prefer MPS when a tensor can be created there."""

    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return "mps"
        except RuntimeError:
            return "cpu"
    return "cpu"


def _script_name(char: str) -> str | None:
    try:
        name = unicodedata.name(char)
    except ValueError:
        return None
    for script in _NON_LATIN_SCRIPTS:
        if script in name:
            return script
    return None


def leftover_script_rate(text: str) -> float:
    """Return the share of letters that still belong to a source script."""

    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    leftover = sum(1 for char in letters if _script_name(char) is not None)
    return leftover / len(letters)


def mms_words(romanized: str) -> tuple[str, ...]:
    """Keep only MMS-FA characters and drop empty tokens."""

    words: list[str] = []
    for raw in romanized.lower().split():
        chars = [char for char in raw if char in MMS_CHARSET]
        if chars:
            words.append("".join(chars))
    return tuple(words)


class MmsAligner:
    """Load the local MMS_FA checkpoint and align one cropped utterance."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str | None = None,
        uroman: Uroman | None = None,
    ) -> None:
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MMS checkpoint is missing: {self.model_path}")
        self.model_sha256 = sha256_file(self.model_path)
        self.bundle_id = MMS_MODEL_ID
        self.torchaudio_version = torchaudio.__version__
        self.torch_version = torch.__version__
        try:
            import uroman as uroman_mod

            self.uroman_version = str(getattr(uroman_mod, "__version__", "unknown"))
        except Exception:
            self.uroman_version = "unknown"
        self.device = device or select_device()
        self._uroman = uroman or Uroman()
        self._bundle = bundle
        self._model = bundle.get_model(
            with_star=True,
            dl_kwargs={
                "model_dir": str(self.model_path.parent),
                "file_name": self.model_path.name,
                "map_location": "cpu",
                "progress": False,
            },
        )
        try:
            self._model.to(self.device)
            torch.zeros(1, device=self.device)
        except RuntimeError:
            self.device = "cpu"
            self._model.to("cpu")
        self._model.eval()
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()
        self.forced_align_devices = _forced_align_devices()

    def provenance(self) -> dict[str, object]:
        """Return aligner and model identity for a release receipt."""

        return {
            "aligner": self.bundle_id,
            "checkpoint": str(self.model_path),
            "checkpoint_sha256": self.model_sha256,
            "torch": self.torch_version,
            "torchaudio": self.torchaudio_version,
            "uroman": self.uroman_version,
            "device": self.device,
            "sample_rate": MMS_SAMPLE_RATE,
            "labels": list(self._bundle.get_labels()),
        }

    def romanize(self, text: str, *, lcode: str | None = None) -> Romanization:
        """Romanize transcript text and keep MMS-FA word tokens."""

        original = text if isinstance(text, str) else ""
        romanized = self._uroman.romanize_string(original, lcode=lcode) if original else ""
        if not isinstance(romanized, str):
            romanized = str(romanized)
        words = mms_words(romanized)
        leftover = leftover_script_rate(romanized)
        return Romanization(original, romanized, words, leftover, not words)

    def align_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
        words: Sequence[str],
        *,
        origin_ns: int = 0,
    ) -> tuple[AlignedWord, ...]:
        """Align romanized words on a mono waveform and shift onto origin_ns."""

        if not words:
            return ()
        waveform = _as_mono_waveform(samples, sample_rate)
        if waveform.numel() < MIN_ALIGN_SAMPLES:
            return ()
        tokens = self._tokenizer(list(words))
        if not tokens or any(not item for item in tokens):
            return ()
        align_device = self.device if self.device in self.forced_align_devices else "cpu"
        try:
            with torch.inference_mode():
                emission, _ = self._model(waveform.to(self.device))
                emission = emission.to(align_device)
                token_spans = self._aligner(emission[0], tokens)
        except RuntimeError:
            if self.device != "cpu":
                with torch.inference_mode():
                    cpu_model = self._model.to("cpu")
                    emission, _ = cpu_model(waveform)
                    token_spans = self._aligner(emission[0], tokens)
                    self._model.to(self.device)
            else:
                raise
        if len(token_spans) != len(words):
            return ()
        num_frames = int(emission.size(1))
        if num_frames <= 0:
            return ()
        seconds_per_frame = waveform.size(1) / float(MMS_SAMPLE_RATE) / float(num_frames)
        aligned: list[AlignedWord] = []
        for word, spans in zip(words, token_spans):
            if not spans:
                continue
            start_frame = min(item.start for item in spans)
            end_frame = max(item.end for item in spans)
            if end_frame <= start_frame:
                continue
            start_ns = origin_ns + round(start_frame * seconds_per_frame * NANOSECONDS_PER_SECOND)
            end_ns = origin_ns + round(end_frame * seconds_per_frame * NANOSECONDS_PER_SECOND)
            if end_ns <= start_ns:
                continue
            weight = sum((item.end - item.start) * item.score for item in spans)
            length = sum(item.end - item.start for item in spans)
            score = float(weight / length) if length else 0.0
            aligned.append(AlignedWord(int(start_ns), int(end_ns), word, score))
        return tuple(aligned)


def _as_mono_waveform(samples: np.ndarray, sample_rate: int) -> torch.Tensor:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if sample_rate != MMS_SAMPLE_RATE:
        array = _resample(array, sample_rate, MMS_SAMPLE_RATE)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
    return tensor


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    from math import gcd

    from scipy.signal import resample_poly

    factor = gcd(source_rate, target_rate)
    return np.asarray(resample_poly(samples, target_rate // factor, source_rate // factor), dtype=np.float32)


def _forced_align_devices() -> frozenset[str]:
    try:
        import torchaudio.functional as functional

        source = inspect.getsource(functional.forced_align)
    except (OSError, TypeError):
        return frozenset({"cpu"})
    devices = {"cpu"}
    if "mps" in source:
        devices.add("mps")
    if "cuda" in source:
        devices.add("cuda")
    return frozenset(devices)


def measure_language_coverage(
    language: str,
    texts: Sequence[str],
    romanizer: MmsAligner,
    *,
    lcode: str | None = None,
    sample_limit: int = 8,
) -> LanguageCoverage:
    """Measure empty-token and leftover-script rates for one language."""

    empty = 0
    leftover = 0
    nap_junk = 0
    expansions: list[float] = []
    letter_count = 0
    samples: list[tuple[str, str]] = []
    for text in texts:
        if not any(char.isalpha() for char in text):
            continue
        letter_count += 1
        romanized = romanizer.romanize(text, lcode=lcode)
        if romanized.empty:
            empty += 1
        if romanized.leftover_script_rate > 0:
            leftover += 1
        original_letters = sum(1 for char in text if char.isalpha())
        roman_letters = sum(1 for char in romanized.romanized if char.isalpha())
        if original_letters > 0:
            expansions.append(roman_letters / original_letters)
        if "nap" in romanized.romanized.lower():
            nap_junk += 1
        if len(samples) < sample_limit:
            samples.append((text[:120], romanized.romanized[:120]))
    empty_rate = (empty / letter_count) if letter_count else 0.0
    leftover_rate = (leftover / letter_count) if letter_count else 0.0
    expansion = float(sum(expansions) / len(expansions)) if expansions else 1.0
    nap_rate = (nap_junk / letter_count) if letter_count else 0.0
    poor = letter_count > 0 and (
        empty_rate > POOR_ROMAN_EMPTY_RATE
        or leftover_rate > POOR_ROMAN_LEFTOVER_RATE
        or nap_rate > POOR_ROMAN_NAP_RATE
    )
    return LanguageCoverage(
        language=language,
        lcode=lcode,
        segment_count=len(texts),
        letter_segment_count=letter_count,
        empty_roman_count=empty,
        leftover_script_count=leftover,
        empty_roman_rate=empty_rate,
        leftover_script_rate=leftover_rate,
        expansion_ratio=expansion,
        nap_junk_rate=nap_rate,
        poor=poor,
        samples=tuple(samples),
    )
