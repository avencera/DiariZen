from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.notsofar_real import bind_notsofar_audio, parse_notsofar_transcript


def _write_audio(path: Path, samples: np.ndarray) -> None:
    sf.write(path, samples, 16_000, subtype="PCM_16")


def test_audio_binding_accepts_sample_identical_containers(tmp_path: Path) -> None:
    samples = np.arange(-400, 400, dtype=np.int16)
    source = tmp_path / "source.wav"
    canonical = tmp_path / "canonical.flac"
    _write_audio(source, samples)
    _write_audio(canonical, samples)

    binding = bind_notsofar_audio("MTG_30830", source, canonical)

    assert binding.sample_count == len(samples)
    assert binding.decoded_pcm_sha256


def test_audio_binding_rejects_changed_samples(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    canonical = tmp_path / "canonical.flac"
    _write_audio(source, np.array([1, 2, 3], dtype=np.int16))
    _write_audio(canonical, np.array([1, 2, 4], dtype=np.int16))

    with pytest.raises(PreparationError, match="source and canonical audio differ"):
        bind_notsofar_audio("MTG_30830", source, canonical)


def test_transcript_rejects_interval_past_audio(tmp_path: Path) -> None:
    transcript = tmp_path / "gt_transcription.json"
    transcript.write_text(
        json.dumps([{"speaker_id": "speaker", "start_time": 1.0, "end_time": 2.1}]),
        encoding="utf-8",
    )

    with pytest.raises(PreparationError, match="exceeds decoded audio"):
        parse_notsofar_transcript(transcript, "MTG_30830", 2.0)
