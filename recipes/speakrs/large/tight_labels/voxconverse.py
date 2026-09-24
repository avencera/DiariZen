"""Produce VoxConverse tight-v1 labels from per-segment ASR and alignment."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from ..jsonio import read_json, write_json
from ..storage import require_free_gib
from . import LIBRARY_VERSION
from .domain import Interval, RecordingLabels, SpeakerSegment
from .gates import GateResult, evaluate_gates
from .guards import (
    GUARD_RULES,
    SAMPLE_RATE,
    AcousticFeatures,
    acoustic_guard,
    classify_transcript,
    dictionary_words,
    measure_acoustic,
    measure_transcript,
    normalize_english_words,
    normalize_mms_words,
)
from .intervals import clip_recording_to_uem
from .listening import write_listening_pack
from .merge import MIN_GAP_NS, MIN_GAP_SECONDS
from .produce import (
    DV_ROOT,
    MINIMUM_FREE_GIB,
    STAGING_ROOT,
    TRAINING_VIEW,
    RecordingRef,
    library_commit,
    load_current_labels,
    release_id_with_version,
    render_gate_markdown,
)
from .tighten import tighten_recording, union_human_segments
from .time import NANOSECONDS_PER_SECOND, ns_to_seconds
from .writers import (
    render_speaker_rttm,
    rewrite_uem_file_id,
    write_recording_labels,
)


RELEASE_ID = "voxconverse-tight-v1"
SOURCE_NAME = "VoxConverse"
PREFIX = "voxconverse::"
COMPONENT = "voxconverse"
LICENSE = "CC BY 4.0"
MFA_JOBS = 6
PARAKEET_CHUNK_SECONDS = 120.0
PARAKEET_OVERLAP_SECONDS = 15.0
WHISPER_LID_REPO = "mlx-community/whisper-tiny"
WHISPER_ASR_REPO = "mlx-community/whisper-small-mlx"
LID_MIN_PROB = 0.35
WORK_ROOT = STAGING_ROOT / "voxconverse-work"
VOX_MODELS = STAGING_ROOT / "voxconverse-models"
PARAKEET_DIR = STAGING_ROOT / "models/parakeet-tdt-0.6b-v3"
MFA_ACOUSTIC = STAGING_ROOT / "models/mfa/acoustic/english_mfa.zip"
MFA_DICTIONARY = STAGING_ROOT / "models/mfa/dictionary/english_us_mfa.dict"
MFA_G2P = STAGING_ROOT / "models/mfa/g2p/english_us_mfa.zip"
MMS_WEIGHTS = STAGING_ROOT / "models/mms/ctc_alignment_mling_uroman.pt"
MFA_BIN = STAGING_ROOT / "tools/mfa-conda/bin/mfa"
MODELS_RECEIPT = STAGING_ROOT / "models/receipt.json"

PARAKEET_LANGUAGES = frozenset(
    {
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "it",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "es",
        "sv",
        "ru",
        "uk",
    }
)
WHISPER_LANGUAGES = frozenset(
    {
        "en",
        "zh",
        "de",
        "es",
        "ru",
        "ko",
        "fr",
        "ja",
        "pt",
        "tr",
        "pl",
        "ca",
        "nl",
        "ar",
        "sv",
        "it",
        "id",
        "hi",
        "fi",
        "vi",
        "he",
        "uk",
        "el",
        "ms",
        "cs",
        "ro",
        "da",
        "hu",
        "ta",
        "no",
        "th",
        "ur",
        "hr",
        "bg",
        "lt",
        "la",
        "mi",
        "ml",
        "cy",
        "sk",
        "te",
        "fa",
        "lv",
        "bn",
        "sr",
        "az",
        "sl",
        "kn",
        "et",
        "mk",
        "br",
        "eu",
        "is",
        "hy",
        "ne",
        "mn",
        "bs",
        "kk",
        "sq",
        "sw",
        "gl",
        "mr",
        "pa",
        "si",
        "km",
        "sn",
        "yo",
        "so",
        "af",
        "oc",
        "ka",
        "be",
        "tg",
        "sd",
        "gu",
        "am",
        "yi",
        "lo",
        "uz",
        "fo",
        "ht",
        "ps",
        "tk",
        "nn",
        "mt",
        "sa",
        "lb",
        "my",
        "bo",
        "tl",
        "mg",
        "as",
        "tt",
        "haw",
        "ln",
        "ha",
        "ba",
        "jw",
        "su",
        "yue",
    }
)
ISO1_TO3 = {
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "uk": "ukr",
    "pl": "pol",
    "nl": "nld",
    "sv": "swe",
    "da": "dan",
    "fi": "fin",
    "el": "ell",
    "hu": "hun",
    "cs": "ces",
    "sk": "slk",
    "sl": "slv",
    "hr": "hrv",
    "bg": "bul",
    "ro": "ron",
    "et": "est",
    "lv": "lav",
    "lt": "lit",
    "mt": "mlt",
    "ar": "ara",
    "zh": "zho",
    "ja": "jpn",
    "ko": "kor",
    "hi": "hin",
    "tr": "tur",
    "th": "tha",
    "vi": "vie",
    "he": "heb",
    "fa": "fas",
    "ur": "urd",
    "bn": "ben",
    "ta": "tam",
    "te": "tel",
    "id": "ind",
    "ms": "msa",
    "ca": "cat",
    "no": "nor",
}
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
SILENCE_WORDS = frozenset({"", "sil", "sp", "spn", "<unk>", "<eps>", "unk"})


@dataclass(frozen=True, slots=True)
class SegmentJob:
    """One unioned human segment to transcribe and align on its own."""

    segment_id: str
    recording_id: str
    speaker: str
    start_ns: int
    end_ns: int
    audio_path: str

    @property
    def duration_ns(self) -> int:
        """Return the human-segment duration in nanoseconds."""

        return self.end_ns - self.start_ns

    @property
    def duration_s(self) -> float:
        """Return the human-segment duration in seconds."""

        return ns_to_seconds(self.duration_ns)

    @property
    def safe_id(self) -> str:
        """Return a filesystem-safe identifier."""

        return SAFE_ID_RE.sub("_", self.segment_id)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seconds_to_ns(value: float) -> int:
    """Round a float second count to integer nanoseconds."""

    return int(round(float(value) * NANOSECONDS_PER_SECOND))


def write_status(work_dir: Path, **fields: Any) -> None:
    """Write a small status file for progress notes."""

    payload = {"updated_utc": utc_now(), **fields}
    write_json(work_dir / "status.json", payload)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object as a line and flush it to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Load a JSONL file keyed by segment_id."""

    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["segment_id"]] = row
    return rows


def load_refs() -> tuple[RecordingRef, ...]:
    """Load VoxConverse training audio, UEM, and duration."""

    payload = read_json(TRAINING_VIEW / "components" / f"{COMPONENT}.json")
    refs: list[RecordingRef] = []
    for record in payload["records"]:
        uem_path = Path(record["uem"]["source_absolute_path"])
        from .writers import uem_bounds

        uem_start_ns, uem_end_ns = uem_bounds(uem_path.read_text(encoding="utf-8"))
        duration_samples = int(record["duration_samples"])
        duration_ns = duration_samples * NANOSECONDS_PER_SECOND // SAMPLE_RATE
        refs.append(
            RecordingRef(
                recording_id=record["recording_id"],
                audio_path=Path(record["audio"]["source_absolute_path"]),
                uem_path=uem_path,
                duration_ns=duration_ns,
                uem_start_ns=uem_start_ns,
                uem_end_ns=uem_end_ns,
            )
        )
    refs.sort(key=lambda item: item.recording_id)
    return tuple(refs)


def build_jobs(
    refs: Sequence[RecordingRef],
    current_by_id: Mapping[str, list[SpeakerSegment]],
) -> tuple[SegmentJob, ...]:
    """Union overlapping same-speaker human segments, then clip to UEM."""

    jobs: list[SegmentJob] = []
    for ref in refs:
        original = RecordingLabels(ref.recording_id, tuple(current_by_id[ref.recording_id]))
        clipped = clip_recording_to_uem(original, ref.uem_start_ns, ref.uem_end_ns)
        for segment in union_human_segments(clipped.segments):
            jobs.append(
                SegmentJob(
                    segment_id=f"{segment.recording_id}|{segment.speaker}|{segment.start_ns}",
                    recording_id=segment.recording_id,
                    speaker=segment.speaker,
                    start_ns=segment.start_ns,
                    end_ns=segment.end_ns,
                    audio_path=str(ref.audio_path),
                )
            )
    jobs.sort(key=lambda item: (item.recording_id, item.start_ns, item.speaker))
    return tuple(jobs)


def cut_samples(job: SegmentJob) -> np.ndarray:
    """Read the human-segment audio as mono float32 at 16 kHz."""

    start_frame = (job.start_ns * SAMPLE_RATE) // NANOSECONDS_PER_SECOND
    stop_frame = (job.end_ns * SAMPLE_RATE) // NANOSECONDS_PER_SECOND
    if stop_frame <= start_frame:
        return np.zeros(0, dtype=np.float32)
    samples, sample_rate = sf.read(
        job.audio_path,
        start=start_frame,
        stop=stop_frame,
        dtype="float32",
        always_2d=False,
    )
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"training audio must be {SAMPLE_RATE} Hz: {job.audio_path}")
    if np.ndim(samples) == 2:
        samples = np.mean(samples, axis=1)
    return np.asarray(samples, dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray) -> None:
    """Write a 16 kHz PCM16 wav for MFA."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")


def acoustic_row(features: AcousticFeatures) -> dict[str, float | int]:
    """Return JSON-ready acoustic measurements."""

    return {
        "duration_s": features.duration_s,
        "mean_rms_dbfs": features.mean_rms_dbfs,
        "voiced_fraction": features.voiced_fraction,
        "modulation_ratio": features.modulation_ratio,
        "frame_count": features.frame_count,
    }


def ensure_hf_snapshot(repo: str, dest: Path) -> Path:
    """Download a Hugging Face repo once into staging."""

    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "config.json").is_file():
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo, local_dir=str(dest))
    return dest


class WhisperLid:
    """Spoken-language identification with Whisper tiny on MLX."""

    def __init__(self, repo: str = WHISPER_LID_REPO, cache_dir: Path | None = None) -> None:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.load_models import load_model
        from mlx_whisper.tokenizer import get_tokenizer

        self._mx = mx
        self._log_mel = log_mel_spectrogram
        self._pad = pad_or_trim
        self._n_frames = N_FRAMES
        self._n_samples = N_SAMPLES
        local = ensure_hf_snapshot(repo, cache_dir or (VOX_MODELS / "whisper-tiny"))
        self.model = load_model(str(local))
        self.tokenizer = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages)
        self.repo = repo
        self.local_dir = local

    def detect(self, samples: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Return the top language code, its probability, and the full distribution."""

        audio = samples.astype(np.float32)
        mel = self._log_mel(audio, n_mels=self.model.dims.n_mels, padding=self._n_samples)
        mel_segment = self._pad(mel, self._n_frames, axis=-2)
        _tokens, probs = self.model.detect_language(mel_segment, tokenizer=self.tokenizer)
        if isinstance(probs, list):
            probs = probs[0]
        language, probability = max(probs.items(), key=lambda item: item[1])
        self._mx.eval(self.model.parameters())
        return language, float(probability), {key: float(value) for key, value in probs.items()}


class ParakeetAsr:
    """Parakeet TDT 0.6B v3 on MLX, one model load for every segment."""

    def __init__(self, model_dir: Path) -> None:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        self._mx = mx
        self._get_logmel = get_logmel
        self.model = from_pretrained(str(model_dir))
        self.sample_rate = int(self.model.preprocessor_config.sample_rate)

    def transcribe(self, samples: np.ndarray) -> str:
        """Transcribe a 16 kHz mono cut. Chunk long audio."""

        if samples.size == 0:
            return ""
        audio = self._mx.array(np.ascontiguousarray(samples, dtype=np.float32))
        duration = samples.size / self.sample_rate
        if duration <= PARAKEET_CHUNK_SECONDS + 10.0:
            mel = self._get_logmel(audio, self.model.preprocessor_config)
            return self.model.generate(mel)[0].text.strip()
        chunk = int(PARAKEET_CHUNK_SECONDS * self.sample_rate)
        overlap = int(PARAKEET_OVERLAP_SECONDS * self.sample_rate)
        texts: list[str] = []
        start = 0
        while start < audio.shape[0]:
            end = min(start + chunk, audio.shape[0])
            if end - start < self.model.preprocessor_config.hop_length:
                break
            mel = self._get_logmel(audio[start:end], self.model.preprocessor_config)
            texts.append(self.model.generate(mel)[0].text.strip())
            if end >= audio.shape[0]:
                break
            start = end - overlap
        return " ".join(part for part in texts if part).strip()


class WhisperAsr:
    """Whisper small on MLX for languages outside Parakeet's 25."""

    def __init__(self, repo: str = WHISPER_ASR_REPO) -> None:
        import mlx_whisper

        self._transcribe = mlx_whisper.transcribe
        self.local_dir = ensure_hf_snapshot(repo, VOX_MODELS / "whisper-small")
        self.repo = str(self.local_dir)

    def transcribe(self, samples: np.ndarray, language: str) -> str:
        """Transcribe a cut in a known Whisper language."""

        result = self._transcribe(
            np.ascontiguousarray(samples, dtype=np.float32),
            path_or_hf_repo=self.repo,
            language=language,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        return str(result.get("text") or "").strip()


def choose_asr(language: str, probability: float) -> tuple[str | None, str]:
    """Select the ASR backend from spoken LID.

    Low-confidence LID falls back to English because VoxConverse is mostly English.
    """

    if probability < LID_MIN_PROB:
        language = "en"
    if language in PARAKEET_LANGUAGES:
        return "parakeet", language
    if language in WHISPER_LANGUAGES:
        return "whisper-small", language
    return None, "LID_UNSUPPORTED"


def run_asr_pass(
    jobs: Sequence[SegmentJob],
    work_dir: Path,
    *,
    limit: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Cut each human segment, run LID and ASR, and apply invented-text guards."""

    asr_path = work_dir / "asr.jsonl"
    done = load_jsonl(asr_path)
    selected = jobs if limit is None else jobs[:limit]
    remaining = [job for job in selected if job.segment_id not in done]
    write_status(
        work_dir,
        phase="asr",
        done=len(done),
        remaining=len(remaining),
        total=len(selected),
    )
    if not remaining:
        return load_jsonl(asr_path)

    lid = WhisperLid(cache_dir=VOX_MODELS / "whisper-tiny")
    parakeet: ParakeetAsr | None = None
    whisper: WhisperAsr | None = None
    started = time.time()
    processed = 0
    for job in remaining:
        samples = cut_samples(job)
        features = measure_acoustic(samples)
        guard = acoustic_guard(features)
        row: dict[str, Any] = {
            "segment_id": job.segment_id,
            "recording_id": job.recording_id,
            "speaker": job.speaker,
            "start_ns": job.start_ns,
            "end_ns": job.end_ns,
            "duration_s": job.duration_s,
            "acoustic": acoustic_row(features),
            "guard": guard,
            "language": None,
            "language_prob": None,
            "asr_backend": None,
            "text": "",
            "char_rate": None,
            "word_rate": None,
            "word_count": 0,
        }
        if guard is None and samples.size > 0:
            language, probability, _dist = lid.detect(samples)
            row["language"] = language
            row["language_prob"] = probability
            backend, selected_language = choose_asr(language, probability)
            row["language"] = selected_language if backend else language
            if backend is None:
                row["guard"] = "LID_UNSUPPORTED"
            else:
                row["asr_backend"] = backend
                language = selected_language
                try:
                    if backend == "parakeet":
                        if parakeet is None:
                            parakeet = ParakeetAsr(PARAKEET_DIR)
                        text = parakeet.transcribe(samples)
                    else:
                        if whisper is None:
                            whisper = WhisperAsr()
                        text = whisper.transcribe(samples, language)
                except Exception as error:  # noqa: BLE001
                    row["guard"] = "ASR_FAILED"
                    row["error"] = str(error)
                    text = ""
                row["text"] = text
                stats = measure_transcript(text, job.duration_s)
                row["char_rate"] = stats.char_rate if stats.char_rate != float("inf") else None
                row["word_rate"] = stats.word_rate if stats.word_rate != float("inf") else None
                row["word_count"] = stats.word_count
                if row["guard"] is None:
                    row["guard"] = classify_transcript(text, job.duration_s)
        append_jsonl(asr_path, row)
        processed += 1
        if processed % 20 == 0 or processed == len(remaining):
            write_status(
                work_dir,
                phase="asr",
                done=len(done) + processed,
                remaining=len(remaining) - processed,
                total=len(selected),
                elapsed_s=round(time.time() - started, 1),
                last_segment=job.segment_id,
            )
            print(
                f"[voxconverse] asr {len(done) + processed}/{len(selected)} "
                f"last={job.recording_id} guard={row['guard']}",
                flush=True,
            )
    return load_jsonl(asr_path)


def parse_mfa_words(payload: Mapping[str, Any]) -> list[tuple[float, float, str]]:
    """Parse MFA JSON word intervals."""

    entries: list[Any] = []
    tiers = payload.get("tiers")
    if isinstance(tiers, dict):
        words = tiers.get("words") or tiers.get("word")
        if isinstance(words, dict):
            entries = list(words.get("entries") or [])
    if not entries and isinstance(payload.get("words"), list):
        for item in payload["words"]:
            if isinstance(item, dict):
                entries.append((item.get("start"), item.get("end"), item.get("word") or item.get("label")))
    words_out: list[tuple[float, float, str]] = []
    for item in entries:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        start, end, label = item[0], item[1], item[2]
        word = str(label or "").strip()
        if word.lower() in SILENCE_WORDS:
            continue
        start_f = float(start)
        end_f = float(end)
        if end_f > start_f:
            words_out.append((start_f, end_f, word))
    return words_out


def parse_textgrid_words(text: str) -> list[tuple[float, float, str]]:
    """Parse word intervals from a long TextGrid."""

    words: list[tuple[float, float, str]] = []
    in_words = False
    xmin = xmax = None
    text_value = None
    for raw in text.splitlines():
        line = raw.strip()
        lowered = line.lower()
        if lowered.startswith("name") and "word" in lowered:
            in_words = True
        if not in_words:
            continue
        if lowered.startswith("name") and "word" not in lowered and "xmin" not in lowered:
            if words or xmin is not None:
                break
        if line.startswith("xmin") and "=" in line:
            xmin = float(line.split("=", 1)[1])
        elif line.startswith("xmax") and "=" in line:
            xmax = float(line.split("=", 1)[1])
        elif line.startswith("text") and "=" in line:
            text_value = line.split("=", 1)[1].strip().strip('"')
            if xmin is not None and xmax is not None and text_value is not None:
                if text_value.lower() not in SILENCE_WORDS and xmax > xmin:
                    words.append((xmin, xmax, text_value))
            xmin = xmax = text_value = None
    return words


def run_g2p(words: Sequence[str], work_dir: Path) -> tuple[Path, tuple[str, ...]]:
    """Generate pronunciations for OOV words. Return the extra dict and leftovers."""

    unique = sorted({word for word in words if word})
    if not unique:
        extra = work_dir / "mfa" / "oov.dict"
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("", encoding="utf-8")
        return extra, ()
    wordlist = work_dir / "mfa" / "oov_words.txt"
    extra = work_dir / "mfa" / "oov.dict"
    wordlist.write_text("\n".join(unique) + "\n", encoding="utf-8")
    command = [
        str(MFA_BIN),
        "g2p",
        str(wordlist),
        str(MFA_G2P),
        str(extra),
        "--num_jobs",
        str(MFA_JOBS),
        "--temporary_directory",
        str(work_dir / "mfa" / "g2p-tmp"),
        "--clean",
    ]
    subprocess.run(command, check=True)
    generated = dictionary_words(extra.read_text(encoding="utf-8").splitlines()) if extra.is_file() else frozenset()
    missing = tuple(word for word in unique if word not in generated)
    return extra, missing


def prepare_mfa_corpus(
    jobs: Sequence[SegmentJob],
    asr_rows: Mapping[str, Mapping[str, Any]],
    work_dir: Path,
) -> tuple[Path, dict[str, SegmentJob], dict[str, int], tuple[str, ...]]:
    """Write wav+lab for English segments that passed ASR guards."""

    corpus = work_dir / "mfa" / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    corpus.mkdir(parents=True)
    known = dictionary_words(MFA_DICTIONARY.read_text(encoding="utf-8").splitlines())
    mapping: dict[str, SegmentJob] = {}
    needed_oov: set[str] = set()
    per_file_words: dict[str, tuple[str, ...]] = {}
    for job in jobs:
        row = asr_rows.get(job.segment_id)
        if row is None or row.get("guard") or row.get("language") != "en":
            continue
        words = normalize_english_words(str(row.get("text") or ""))
        if not words:
            continue
        per_file_words[job.safe_id] = words
        mapping[job.safe_id] = job
        for word in words:
            if word not in known:
                needed_oov.add(word)
    extra_dict, missing = run_g2p(sorted(needed_oov), work_dir)
    missing_set = set(missing)
    combined = work_dir / "mfa" / "english_plus_oov.dict"
    combined.write_text(
        MFA_DICTIONARY.read_text(encoding="utf-8")
        + (extra_dict.read_text(encoding="utf-8") if extra_dict.is_file() else ""),
        encoding="utf-8",
    )
    oov_counts: dict[str, int] = {}
    kept: dict[str, SegmentJob] = {}
    g2p_failed: list[str] = []
    for safe_id, job in mapping.items():
        words = per_file_words[safe_id]
        oov = [word for word in words if word not in known]
        oov_counts[job.segment_id] = len(oov)
        if any(word in missing_set for word in words):
            g2p_failed.append(job.segment_id)
            continue
        samples = cut_samples(job)
        write_wav(corpus / f"{safe_id}.wav", samples)
        (corpus / f"{safe_id}.lab").write_text(" ".join(words) + "\n", encoding="utf-8")
        kept[safe_id] = job
    return combined, kept, oov_counts, tuple(g2p_failed)


def run_mfa_align(corpus_dict: Path, work_dir: Path) -> Path:
    """Run MFA English alignment with six jobs on CacheDisk temp space."""

    output = work_dir / "mfa" / "output"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    config_path = work_dir / "mfa" / "align.yaml"
    config_path.write_text("beam: 10\nretry_beam: 40\n", encoding="utf-8")
    command = [
        str(MFA_BIN),
        "align",
        str(work_dir / "mfa" / "corpus"),
        str(corpus_dict),
        str(MFA_ACOUSTIC),
        str(output),
        "--config_path",
        str(config_path),
        "--g2p_model_path",
        str(MFA_G2P),
        "--output_format",
        "json",
        "--num_jobs",
        str(MFA_JOBS),
        "--temporary_directory",
        str(work_dir / "mfa" / "align-tmp"),
        "--single_speaker",
        "--clean",
        "--overwrite",
        "--use_mp",
    ]
    log_path = work_dir / "mfa" / "align.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return output


def load_mms(device: str):
    """Load the local MMS forced-alignment bundle on MPS or CPU."""

    import torch
    from torchaudio.pipelines import MMS_FA

    bundle = MMS_FA
    model = bundle.get_model(
        with_star=False,
        dl_kwargs={
            "model_dir": str(MMS_WEIGHTS.parent),
            "file_name": MMS_WEIGHTS.name,
            "progress": False,
        },
    )
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    return bundle, model, tokenizer, aligner, torch_device


def align_mms_words(
    samples: np.ndarray,
    words: Sequence[str],
    *,
    model,
    tokenizer,
    aligner,
    device,
    sample_rate: int,
) -> list[tuple[float, float, str]]:
    """Align romanized words with MMS CTC forced alignment."""

    import torch

    if not words or samples.size == 0:
        return []
    waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(list(words)))
    ratio = waveform.size(1) / emission.size(1) / sample_rate
    aligned: list[tuple[float, float, str]] = []
    for spans, word in zip(token_spans, words):
        if not spans:
            continue
        start = float(spans[0].start) * ratio
        end = float(spans[-1].end) * ratio
        if end > start:
            aligned.append((start, end, word))
    return aligned


def run_alignment(
    jobs: Sequence[SegmentJob],
    asr_rows: Mapping[str, Mapping[str, Any]],
    work_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Align English with MFA and other languages with MMS."""

    align_path = work_dir / "align.jsonl"
    done = load_jsonl(align_path)
    write_status(work_dir, phase="align", done=len(done), total=len(jobs))
    pending_ids = {job.segment_id for job in jobs if job.segment_id not in done}
    if not pending_ids:
        return done

    english_jobs = [
        job
        for job in jobs
        if job.segment_id in pending_ids
        and asr_rows.get(job.segment_id, {}).get("language") == "en"
        and not asr_rows.get(job.segment_id, {}).get("guard")
    ]
    other_jobs = [
        job
        for job in jobs
        if job.segment_id in pending_ids
        and asr_rows.get(job.segment_id, {}).get("language") not in {None, "en"}
        and not asr_rows.get(job.segment_id, {}).get("guard")
    ]

    oov_counts: dict[str, int] = {}
    g2p_failed: set[str] = set()
    mfa_output: Path | None = None
    mfa_kept: dict[str, SegmentJob] = {}
    if english_jobs:
        write_status(work_dir, phase="mfa-prepare", english=len(english_jobs))
        combined, mfa_kept, oov_counts, failed = prepare_mfa_corpus(english_jobs, asr_rows, work_dir)
        g2p_failed.update(failed)
        if mfa_kept:
            write_status(work_dir, phase="mfa-align", files=len(mfa_kept))
            print(f"[voxconverse] mfa align files={len(mfa_kept)} jobs={MFA_JOBS}", flush=True)
            mfa_output = run_mfa_align(combined, work_dir)

    for job in english_jobs:
        row: dict[str, Any] = {
            "segment_id": job.segment_id,
            "recording_id": job.recording_id,
            "speaker": job.speaker,
            "aligner": "mfa-english-v3.1.0",
            "ok": False,
            "words": [],
            "oov_count": oov_counts.get(job.segment_id, 0),
            "error": None,
        }
        if job.segment_id in g2p_failed:
            row["error"] = "G2P_FAILED"
            append_jsonl(align_path, row)
            continue
        if mfa_output is None or job.safe_id not in mfa_kept:
            row["error"] = "NO_ALIGNABLE_WORDS"
            append_jsonl(align_path, row)
            continue
        json_path = mfa_output / f"{job.safe_id}.json"
        grid_path = mfa_output / f"{job.safe_id}.TextGrid"
        try:
            if json_path.is_file():
                pairs = parse_mfa_words(read_json(json_path))
            elif grid_path.is_file():
                pairs = parse_textgrid_words(grid_path.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(str(json_path))
            words = []
            for start, end, label in pairs:
                start_ns = job.start_ns + seconds_to_ns(start)
                end_ns = job.start_ns + seconds_to_ns(end)
                end_ns = min(end_ns, job.end_ns)
                start_ns = max(start_ns, job.start_ns)
                if end_ns > start_ns:
                    words.append({"text": label, "start_ns": start_ns, "end_ns": end_ns})
            row["words"] = words
            row["ok"] = bool(words)
            if not words:
                row["error"] = "ALIGN_FAILED"
        except Exception as error:  # noqa: BLE001
            row["error"] = f"ALIGN_FAILED:{error}"
        append_jsonl(align_path, row)

    if other_jobs:
        write_status(work_dir, phase="mms", remaining=len(other_jobs))
        print(f"[voxconverse] mms align segments={len(other_jobs)}", flush=True)
        import torch
        from uroman import Uroman

        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
        bundle, model, tokenizer, aligner, device = load_mms(device_name)
        romanizer = Uroman()
        for index, job in enumerate(other_jobs, start=1):
            asr = asr_rows[job.segment_id]
            language = str(asr.get("language") or "")
            romanized = romanizer.romanize_string(str(asr.get("text") or ""), lcode=ISO1_TO3.get(language, language))
            words = normalize_mms_words(str(asr.get("text") or ""), romanized=str(romanized))
            row = {
                "segment_id": job.segment_id,
                "recording_id": job.recording_id,
                "speaker": job.speaker,
                "aligner": "mms-fa",
                "ok": False,
                "words": [],
                "oov_count": 0,
                "error": None,
                "romanized": str(romanized),
            }
            if not words:
                row["error"] = "NO_ALIGNABLE_WORDS"
                append_jsonl(align_path, row)
                continue
            try:
                samples = cut_samples(job)
                pairs = align_mms_words(
                    samples,
                    words,
                    model=model,
                    tokenizer=tokenizer,
                    aligner=aligner,
                    device=device,
                    sample_rate=int(bundle.sample_rate),
                )
                aligned_words = []
                for start, end, label in pairs:
                    start_ns = job.start_ns + seconds_to_ns(start)
                    end_ns = job.start_ns + seconds_to_ns(end)
                    end_ns = min(end_ns, job.end_ns)
                    start_ns = max(start_ns, job.start_ns)
                    if end_ns > start_ns:
                        aligned_words.append({"text": label, "start_ns": start_ns, "end_ns": end_ns})
                row["words"] = aligned_words
                row["ok"] = bool(aligned_words)
                if not aligned_words:
                    row["error"] = "ALIGN_FAILED"
            except Exception as error:  # noqa: BLE001
                row["error"] = f"ALIGN_FAILED:{error}"
            append_jsonl(align_path, row)
            if index % 20 == 0 or index == len(other_jobs):
                print(f"[voxconverse] mms {index}/{len(other_jobs)} last={job.recording_id}", flush=True)

    written = set(load_jsonl(align_path))
    for job in jobs:
        if job.segment_id in written or job.segment_id not in pending_ids:
            continue
        asr = asr_rows.get(job.segment_id, {})
        append_jsonl(
            align_path,
            {
                "segment_id": job.segment_id,
                "recording_id": job.recording_id,
                "speaker": job.speaker,
                "aligner": None,
                "ok": False,
                "words": [],
                "oov_count": 0,
                "error": asr.get("guard") or "ALIGN_FAILED",
            },
        )
    write_status(work_dir, phase="align-done", total=len(jobs))
    return load_jsonl(align_path)


def labels_from_alignment(
    refs: Sequence[RecordingRef],
    current_by_id: Mapping[str, list[SpeakerSegment]],
    align_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[RecordingLabels], list[RecordingLabels]]:
    """Tighten each recording from per-segment word intervals."""

    words_by_recording: dict[str, dict[str, list[Interval]]] = defaultdict(lambda: defaultdict(list))
    for row in align_rows.values():
        if not row.get("ok"):
            continue
        recording_id = row["recording_id"]
        speaker = row["speaker"]
        for word in row.get("words") or []:
            start_ns = int(word["start_ns"])
            end_ns = int(word["end_ns"])
            if end_ns > start_ns:
                words_by_recording[recording_id][speaker].append(Interval(start_ns, end_ns))

    original: list[RecordingLabels] = []
    produced: list[RecordingLabels] = []
    for ref in refs:
        source = clip_recording_to_uem(
            RecordingLabels(ref.recording_id, tuple(current_by_id[ref.recording_id])),
            ref.uem_start_ns,
            ref.uem_end_ns,
        )
        original.append(source)
        aligned = words_by_recording.get(ref.recording_id, {})
        tight = tighten_recording(source, aligned)
        produced.append(clip_recording_to_uem(tight, ref.uem_start_ns, ref.uem_end_ns))
    return original, produced


def model_hashes() -> dict[str, Any]:
    """Record aligner and ASR identities from the acquisition receipt plus local files."""

    receipt = read_json(MODELS_RECEIPT) if MODELS_RECEIPT.is_file() else {}
    kept = {item.get("id"): item for item in receipt.get("kept_members", []) if isinstance(item, dict)}
    return {
        "parakeet": kept.get("parakeet-tdt-0.6b-v3-mlx"),
        "mfa_english_acoustic": kept.get("acoustic-english_mfa-v3.1.0"),
        "mfa_english_dictionary": kept.get("dictionary-english_us_mfa-v3.1.0"),
        "mfa_english_g2p": kept.get("g2p-english_us_mfa-v3.0.0"),
        "mms_fa": kept.get("torchaudio-mms-fa"),
        "whisper_lid": {
            "id": WHISPER_LID_REPO,
            "license": "MIT",
            "role": "spoken-language-identification",
        },
        "whisper_asr_fallback": {
            "id": WHISPER_ASR_REPO,
            "license": "MIT",
            "role": "asr-outside-parakeet-languages",
        },
        "mfa_binary": {
            "path": str(MFA_BIN),
            "version": "3.4.2",
        },
    }


def summarize_pass(
    jobs: Sequence[SegmentJob],
    asr_rows: Mapping[str, Mapping[str, Any]],
    align_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Count ASR coverage, language mix, guards, and alignment failures."""

    guard_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    asr_ok = 0
    align_ok = 0
    oov_words = 0
    alignable = 0
    for job in jobs:
        asr = asr_rows.get(job.segment_id, {})
        guard = asr.get("guard")
        if guard:
            guard_counts[str(guard)] += 1
        else:
            asr_ok += 1
            backend_counts[str(asr.get("asr_backend") or "unknown")] += 1
        language = asr.get("language")
        if language:
            language_counts[str(language)] += 1
        aligned = align_rows.get(job.segment_id, {})
        if aligned.get("ok"):
            align_ok += 1
        elif not guard:
            guard_counts[str(aligned.get("error") or "ALIGN_FAILED")] += 1
        if asr.get("language") == "en" and not guard:
            alignable += 1
            oov_words += int(aligned.get("oov_count") or 0)
    return {
        "unioned_segments": len(jobs),
        "asr_ok": asr_ok,
        "asr_coverage": asr_ok / len(jobs) if jobs else 0.0,
        "align_ok": align_ok,
        "alignment_failure_share": 1.0 - (align_ok / len(jobs) if jobs else 1.0),
        "guard_counts": dict(guard_counts),
        "language_counts": dict(language_counts),
        "backend_counts": dict(backend_counts),
        "english_oov_words": oov_words,
        "english_alignable_segments": alignable,
        "guard_rules": GUARD_RULES,
    }


def write_release(
    refs: Sequence[RecordingRef],
    original: Sequence[RecordingLabels],
    produced: Sequence[RecordingLabels],
    gate: GateResult,
    kept_current: bool,
    summary: Mapping[str, Any],
    *,
    release_id: str = RELEASE_ID,
) -> Path:
    """Write the voxconverse-tight-v1 release directory."""

    release_dir = DV_ROOT / "releases" / release_id
    if release_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing release {release_dir}")
    labels_dir = release_dir / "labels"
    labels_dir.mkdir(parents=True)
    uem_by_id = {item.recording_id: item.uem_path for item in refs}
    written = original if kept_current else produced
    all_rttm: list[str] = []
    all_uem: list[str] = []
    for labels in written:
        write_recording_labels(labels_dir, labels)
        uem_text = rewrite_uem_file_id(
            uem_by_id[labels.recording_id].read_text(encoding="utf-8"),
            labels.recording_id,
        )
        (labels_dir / f"{labels.recording_id}.uem").write_text(uem_text, encoding="utf-8")
        all_rttm.append(render_speaker_rttm(labels.segments))
        all_uem.append(uem_text if uem_text.endswith("\n") else uem_text + "\n")
    (release_dir / "all.rttm").write_text("".join(all_rttm), encoding="utf-8")
    (release_dir / "all.uem").write_text("".join(all_uem), encoding="utf-8")
    provenance = {
        "schema": "speakrs-tight-labels-release",
        "schema_version": 2,
        "release_id": release_id,
        "source": SOURCE_NAME,
        "method": "keep-current-labels" if kept_current else "asr-per-segment-then-forced-alignment",
        "kept_current_labels": kept_current,
        "library_version": LIBRARY_VERSION,
        "library_commit": library_commit(),
        "min_gap_seconds": MIN_GAP_SECONDS,
        "min_gap_ns": MIN_GAP_NS,
        "recording_count": len(written),
        "licenses": [LICENSE],
        "models": model_hashes(),
        "thresholds": {
            "coverage_ratio": 0.3,
            "lid_min_prob": LID_MIN_PROB,
            "mfa_jobs": MFA_JOBS,
            "guard_rules": GUARD_RULES,
        },
        "asr_alignment_summary": dict(summary),
        "gate_results": gate.to_json(),
    }
    write_json(release_dir / "provenance.json", provenance)
    write_json(release_dir / "gate-report.json", gate.to_json())
    (release_dir / "gate-report.md").write_text(
        render_gate_markdown(SOURCE_NAME, release_id, gate, kept_current=kept_current),
        encoding="utf-8",
    )
    return release_dir


def render_agent_report(
    *,
    refs: Sequence[RecordingRef],
    gate: GateResult,
    kept_current: bool,
    summary: Mapping[str, Any],
    release_dir: Path,
    listening: Mapping[str, Any],
) -> str:
    """Render ALIGN-VOXCONVERSE-REPORT.md in short factual English."""

    tightness = gate.tightness
    original = gate.original_tightness
    languages = summary.get("language_counts") or {}
    language_rows = "\n".join(
        f"| {code} | {count} |" for code, count in sorted(languages.items(), key=lambda item: (-item[1], item[0]))
    )
    guard_rows = "\n".join(
        f"| `{name}` | {count} | {GUARD_RULES.get(name, '')} |"
        for name, count in sorted((summary.get("guard_counts") or {}).items(), key=lambda item: (-item[1], item[0]))
    )
    status = "PASS" if gate.passed and not kept_current else "FAIL"
    lines = [
        "# VoxConverse tight labels (ALIGN-VOXCONVERSE)",
        "",
        f"Status: {status}",
        f"Library commit: `{library_commit()}`",
        f"Release: `{release_dir}`",
        "",
        "## Method",
        "",
        "Each human speaker segment is cut from the training mix. Overlapping "
        "same-speaker segments are unioned first. Text never crosses a segment "
        "boundary. Spoken language comes from Whisper tiny (MIT). Segments in "
        "Parakeet's 25 European languages are transcribed with Parakeet TDT "
        "0.6B v3 on MLX. Other Whisper languages use Whisper small. English "
        "text is aligned with MFA English v3.1.0. Other languages are aligned "
        "with MMS FA after uroman. Word intervals go to `tighten`. Coverage "
        "below 30% or an alignment failure marks the human-segment interior "
        "UNCERTAIN.",
        "",
        "## Inventory",
        "",
        f"- Recordings: {len(refs)}",
        f"- Unioned human segments: {summary['unioned_segments']}",
        f"- Original speaker hours: {original.speaker_seconds / 3600:.3f}",
        f"- Produced speaker hours: {tightness.speaker_seconds / 3600:.3f}",
        "",
        "## ASR coverage and language mix",
        "",
        f"- ASR accepted: {summary['asr_ok']} / {summary['unioned_segments']} ({summary['asr_coverage']:.4f})",
        f"- Alignment accepted: {summary['align_ok']} / {summary['unioned_segments']}",
        f"- Alignment failure share: {summary['alignment_failure_share']:.4f}",
        f"- English OOV word tokens: {summary['english_oov_words']}",
        "",
        "| Language | Segments |",
        "|---|---:|",
        language_rows or "| — | 0 |",
        "",
        "## Invented-text and non-speech guards",
        "",
        "| Rule | Count | Definition |",
        "|---|---:|---|",
        guard_rows or "| — | 0 | |",
        "",
        "## Tightness",
        "",
        "| Measure | Original | Produced | Forced AMI |",
        "|---|---:|---:|---:|",
        f"| Gaps ≤0.3 s / min | {original.gaps_le_0_3_per_min:.4f} | {tightness.gaps_le_0_3_per_min:.4f} | 4.63 |",
        f"| Gaps 0.3–1.0 s / min | {original.gaps_0_3_to_1_per_min:.4f} | {tightness.gaps_0_3_to_1_per_min:.4f} | 17.00 |",
        f"| Smallest gap s | {original.smallest_gap_s} | {tightness.smallest_gap_s} | 0.200 |",
        f"| Median segment s | {original.median_segment_s} | {tightness.median_segment_s} | 0.970 |",
        "",
        "## Gates",
        "",
        f"- Rows inside UEM: {'pass' if gate.uem_ok else 'fail'}",
        f"- Subset of 200 ms-closed original: {'pass' if gate.subset_ok else 'fail'}",
        f"- Smallest same-speaker gap ≥ 0.200 s: {'pass' if gate.smallest_gap_ok else 'fail'}",
        f"- UNCERTAIN overlapping same-speaker speech: {'pass' if gate.uncertain_overlap_ok else 'fail'}",
        f"- Failure share ≤ 0.25: {'pass' if gate.failure_share_ok else 'fail'} ({gate.failure_share:.4f})",
        f"- Tightness band (reported): {'pass' if gate.tightness_ok else 'outside reference'}",
        f"- Speaker retention (reported): {len(gate.flagged_speakers)} speakers below 30%",
        f"- Overlap retention (reported): {gate.overlap_retention}",
        f"- UNCERTAIN share of original speech: {gate.uncertain_share:.4f}",
        f"- Kept current labels: {kept_current}",
        f"- Flagged speakers below 30% retention: {len(gate.flagged_speakers)}",
        "",
    ]
    if kept_current:
        lines.extend(
            [
                "The produced labels are the current labels because the source "
                "failed a hard gate or the failure-share limit.",
                "",
            ]
        )
    lines.extend(
        [
            "## Listening pack",
            "",
            f"- Directory: `{listening.get('directory')}`",
            f"- Clips: {listening.get('clip_count')}",
            f"- Seconds: {listening.get('total_seconds')}",
            "",
        ]
    )
    if gate.notes:
        lines.extend(["## Notes", "", *[f"- {note}" for note in gate.notes], ""])
    return "\n".join(lines)


def produce(*, limit: int | None = None, phase: str = "all", release_version: str | None = None) -> dict[str, Any]:
    """Run ASR, alignment, tightening, gates, release, and the agent report."""

    require_free_gib(DV_ROOT, MINIMUM_FREE_GIB)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    VOX_MODELS.mkdir(parents=True, exist_ok=True)
    refs = load_refs()
    if limit is not None:
        refs = refs[:limit]
    current_by_id = load_current_labels([PREFIX])
    current_by_id = {ref.recording_id: current_by_id[ref.recording_id] for ref in refs}
    jobs = build_jobs(refs, current_by_id)
    write_json(
        WORK_ROOT / "jobs.json",
        {
            "recording_count": len(refs),
            "segment_count": len(jobs),
            "created_utc": utc_now(),
        },
    )
    asr_rows: dict[str, dict[str, Any]] = {}
    align_rows: dict[str, dict[str, Any]] = {}
    if phase in {"all", "asr"}:
        asr_rows = run_asr_pass(jobs, WORK_ROOT, limit=None)
    else:
        asr_rows = load_jsonl(WORK_ROOT / "asr.jsonl")
    if phase == "asr":
        return {"phase": "asr", "asr_rows": len(asr_rows), "jobs": len(jobs)}
    if phase in {"all", "align"}:
        align_rows = run_alignment(jobs, asr_rows, WORK_ROOT)
    else:
        align_rows = load_jsonl(WORK_ROOT / "align.jsonl")
    if phase == "align":
        return {"phase": "align", "align_rows": len(align_rows), "jobs": len(jobs)}

    original, produced = labels_from_alignment(refs, current_by_id, align_rows)
    uem_by_recording = {ref.recording_id: (ref.uem_start_ns, ref.uem_end_ns) for ref in refs}
    gate = evaluate_gates(
        original,
        produced,
        require_subset=True,
        uem_by_recording=uem_by_recording,
    )
    kept_current = not gate.passed
    summary = summarize_pass(jobs, asr_rows, align_rows)
    uncertain_ns = sum(region.duration_ns for labels in produced for region in labels.uncertain)
    original_ns = sum(segment.duration_ns for labels in original for segment in labels.segments)
    summary["uncertain_share_of_original_speech"] = (uncertain_ns / original_ns) if original_ns else 0.0
    summary["failure_share_gate"] = gate.failure_share
    write_json(WORK_ROOT / "summary.json", {**summary, "gate": gate.to_json(), "kept_current": kept_current})
    release_id = release_id_with_version(RELEASE_ID, release_version)
    release_dir = write_release(
        refs, original, produced, gate, kept_current, summary, release_id=release_id
    )
    listening_dir = STAGING_ROOT / "listening-v2" / "voxconverse"
    listening = write_listening_pack(
        listening_dir,
        original=original,
        new=produced,
        audio_by_recording={item.recording_id: item.audio_path for item in refs},
        duration_by_recording={item.recording_id: item.duration_ns for item in refs},
    )
    report = render_agent_report(
        refs=refs,
        gate=gate,
        kept_current=kept_current,
        summary=summary,
        release_dir=release_dir,
        listening=listening,
    )
    report_path = STAGING_ROOT / "ALIGN-VOXCONVERSE-REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    result = {
        "ok": gate.passed and not kept_current,
        "release_dir": str(release_dir),
        "report": str(report_path),
        "listening": listening,
        "kept_current_labels": kept_current,
        "gate_passed": gate.passed,
        "summary": summary,
    }
    write_json(WORK_ROOT / "produce-result.json", result)
    write_status(work_dir=WORK_ROOT, phase="done", **{k: v for k, v in result.items() if k != "summary"})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the VoxConverse tight-label producer."""

    parser = argparse.ArgumentParser(description="Produce VoxConverse tight-v1 labels")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N recordings")
    parser.add_argument(
        "--phase",
        choices=("all", "asr", "align", "produce"),
        default="all",
        help="Stop after ASR, after alignment, or run the full release",
    )
    parser.add_argument(
        "--release-version",
        default=None,
        help="Replace the tight-vN suffix of the release id. Default: keep the current id",
    )
    args = parser.parse_args(argv)
    summary = produce(limit=args.limit, phase=args.phase, release_version=args.release_version)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
