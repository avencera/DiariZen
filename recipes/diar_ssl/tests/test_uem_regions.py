"""Verify UEM region and per-recording speaker handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipes.diar_ssl.dataset import DiarizationDataset, _gen_chunk_indices, load_uem


def _write_inputs(tmp_path: Path, *, uem: str, rttm: str = "") -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    scp = tmp_path / "wav.scp"
    rttm_path = tmp_path / "rttm"
    uem_path = tmp_path / "all.uem"
    scp.write_text("rec-a /tmp/rec-a.flac\nrec-b /tmp/rec-b.flac\n", encoding="utf-8")
    rttm_path.write_text(rttm, encoding="utf-8")
    uem_path.write_text(uem, encoding="utf-8")
    return scp, rttm_path, uem_path


def _dataset(tmp_path: Path, *, uem: str, rttm: str = "", chunk_size: int = 0, chunk_shift: int = 5):
    scp, rttm_path, uem_path = _write_inputs(tmp_path, uem=uem, rttm=rttm)
    return DiarizationDataset(
        str(scp),
        str(rttm_path),
        str(uem_path),
        model_num_frames=10,
        model_rf_duration=0.1,
        model_rf_step=0.1,
        chunk_size=chunk_size,
        chunk_shift=chunk_shift,
    )


def test_load_uem_preserves_sorted_nonoverlapping_regions(tmp_path: Path) -> None:
    uem_path = tmp_path / "all.uem"
    uem_path.write_text(
        "# rec channel start end\nrec-a 1 30 40\nrec-a 1 0 10\nrec-a 1 15 20\n",
        encoding="utf-8",
    )

    assert load_uem(str(uem_path)) == {"rec-a": [(0.0, 10.0), (15.0, 20.0), (30.0, 40.0)]}


@pytest.mark.parametrize(
    "line",
    [
        "rec-a 1 1 1\n",
        "rec-a 1 2 1\nrec-a 1 1 3\n",
        "rec-a 1 0 nan\n",
        "rec-a 1 -1 3\n",
    ],
)
def test_load_uem_rejects_invalid_regions(tmp_path: Path, line: str) -> None:
    uem_path = tmp_path / "all.uem"
    uem_path.write_text(line, encoding="utf-8")

    with pytest.raises(ValueError, match="UEM"):
        load_uem(str(uem_path))


def test_chunk_grid_does_not_cross_unknown_region_gaps(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        uem="rec-a 1 0 20\nrec-a 1 30 50\n",
        chunk_size=8,
        chunk_shift=8,
    )

    assert dataset.chunk_indices == [
        ("rec-a", "/tmp/rec-a.flac", 1, 9),
        ("rec-a", "/tmp/rec-a.flac", 9, 17),
        ("rec-a", "/tmp/rec-a.flac", 31, 39),
        ("rec-a", "/tmp/rec-a.flac", 39, 47),
    ]
    assert all(not (end > 20 and start < 30) for _, _, start, end in dataset.chunk_indices)


def test_single_region_keeps_existing_chunk_grid_and_short_region_fails(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, uem="rec-a 1 0 20\n", chunk_size=8, chunk_shift=8)

    assert dataset.chunk_indices == [
        ("rec-a", "/tmp/rec-a.flac", start, end) for start, end in _gen_chunk_indices(0, 20, 8, 8)
    ]

    with pytest.raises(ValueError, match="too short"):
        _dataset(tmp_path / "short", uem="rec-a 1 0 8\n", chunk_size=8)


def test_rttm_speaker_ids_are_scoped_per_interleaved_recording(tmp_path: Path) -> None:
    rttm = "\n".join(
        [
            "SPEAKER rec-a 1 0 1 <NA> <NA> a <NA> <NA>",
            "SPEAKER rec-b 1 0 1 <NA> <NA> b <NA> <NA>",
            "SPEAKER rec-a 1 2 1 <NA> <NA> c <NA> <NA>",
            "SPEAKER rec-b 1 2 1 <NA> <NA> a <NA> <NA>",
            "SPEAKER rec-a 1 4 1 <NA> <NA> a <NA> <NA>",
        ]
    )
    dataset = _dataset(tmp_path, uem="rec-a 1 0 10\nrec-b 1 0 10\n", rttm=rttm)

    labels = [int(item["label_idx"]) for item in dataset.annotations]
    sessions = [int(item["session_idx"]) for item in dataset.annotations]
    assert sessions == [0, 1, 0, 1, 0]
    assert labels == [0, 0, 1, 1, 0]
