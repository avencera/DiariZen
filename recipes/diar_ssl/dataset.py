# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import math
import os
from typing import Dict, Iterator

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def get_dtype(value: int) -> str:
    """Return the most suitable type for storing the
    value passed in parameter in memory.

    Parameters
    ----------
    value: int
        value whose type is best suited to storage in memory

    Returns
    -------
    str:
        numpy formatted type
        (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
    """
    # signe byte (8 bits), signed short (16 bits), signed int (32 bits):
    types_list = [(127, "b"), (32_768, "i2"), (2_147_483_648, "i")]
    filtered_list = [(max_val, type) for max_val, type in types_list if max_val > abs(value)]
    if not filtered_list:
        return "i8"  # signed long (64 bits)
    return filtered_list[0][1]


def load_scp(scp_file: str) -> Dict[str, str]:
    """return dictionary { rec: wav_rxfilename }"""
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}


def load_uem(uem_file: str) -> Dict[str, list[tuple[float, float]]] | None:
    """Return every positive, non-overlapping ``(start, end)`` span per recording."""

    if not os.path.exists(uem_file):
        return None
    regions: Dict[str, list[tuple[float, float]]] = {}
    with open(uem_file) as file:
        for line_number, line in enumerate(file, 1):
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 4:
                raise ValueError(f"UEM line {line_number} must contain recording, channel, start, and end")
            try:
                start = float(fields[-2])
                end = float(fields[-1])
            except ValueError as error:
                raise ValueError(f"UEM line {line_number} has non-numeric bounds") from error
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError(f"UEM line {line_number} must have finite positive bounds")
            regions.setdefault(fields[0], []).append((start, end))

    for recording, spans in regions.items():
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"UEM recording {recording} contains overlapping spans")
    return regions


def _gen_chunk_indices(
    init_posi: int,
    data_len: int,
    size: int,
    step: int,
) -> Iterator[tuple[int, int]]:
    if size <= 0 or step <= 0:
        raise ValueError("chunk size and shift must be positive")
    init_posi = int(init_posi + 1)
    data_len = int(data_len - 1)
    cur_len = data_len - init_posi
    if cur_len <= size:
        raise ValueError("UEM span is too short for the requested chunk size")
    num_chunks = int((cur_len - size + step) / step)
    if num_chunks <= 0:
        raise ValueError("UEM span cannot produce a complete chunk")

    for i in range(num_chunks):
        yield init_posi + (i * step), init_posi + (i * step) + size


def _collate_fn(batch, max_speakers_per_chunk=4) -> torch.Tensor:
    collated_x = []
    collated_y = []
    collated_names = []

    for x, y, name in batch:
        # remove inactive columns before ranking so zero-padded speakers cannot
        # displace active speakers when the target is stored as uint8
        activity = np.asarray(y, dtype=np.int64).sum(axis=0)
        active_indices = np.flatnonzero(activity > 0)
        y = y[:, active_indices]

        if y.shape[-1] > max_speakers_per_chunk:
            # sort speakers in descending talkativeness order
            indices = np.argsort(-activity[active_indices], kind="stable")
            # keep only the most talkative speakers
            y = y[:, indices[:max_speakers_per_chunk]]

        if y.shape[-1] < max_speakers_per_chunk:
            # create inactive speakers by zero padding
            y = np.pad(
                y,
                ((0, 0), (0, max_speakers_per_chunk - y.shape[-1])),
                mode="constant",
            )
        collated_x.append(x)
        collated_y.append(y)
        collated_names.append(name)

    return {
        "xs": torch.from_numpy(np.stack(collated_x)).float(),
        "ts": torch.from_numpy(np.stack(collated_y)),
        "names": collated_names,
    }


class DiarizationDataset(Dataset):
    def __init__(
        self,
        scp_file: str,
        rttm_file: str,
        uem_file: str,
        model_num_frames: int,  # default: wavlm_base
        model_rf_duration: float,  # model.receptive_field.duration, seconds
        model_rf_step: float,  # model.receptive_field.step, seconds
        chunk_size: int = 5,  # seconds
        chunk_shift: int = 5,  # seconds
        sample_rate: int = 16000,
    ):
        self.chunk_indices = []

        self.sample_rate = sample_rate

        self.model_rf_step = model_rf_step
        self.model_rf_duration = model_rf_duration
        self.model_num_frames = model_num_frames

        self.rec_scp = load_scp(scp_file)
        self.reco2dur = load_uem(uem_file)
        if self.reco2dur is None:
            raise ValueError(f"UEM file does not exist: {uem_file}")

        for rec, spans in self.reco2dur.items():
            if rec not in self.rec_scp:
                raise ValueError(f"UEM recording is absent from wav.scp: {rec}")

            for start_sec, end_sec in spans:
                if chunk_size > 0:
                    for st, ed in _gen_chunk_indices(start_sec, end_sec, chunk_size, chunk_shift):
                        self.chunk_indices.append((rec, self.rec_scp[rec], st, ed))  # seconds
                else:
                    self.chunk_indices.append((rec, self.rec_scp[rec], start_sec, end_sec))

        self.annotations = self.rttm2label(rttm_file)

    def get_session_idx(self, session):
        """
        convert session to session idex
        """
        session_keys = list(self.rec_scp.keys())
        return session_keys.index(session)

    def rttm2label(self, rttm_file):
        """
        SPEAKER train100_306 1 15.71 1.76 <NA> <NA> 5456 <NA> <NA>
        """
        annotations = []
        speaker_indices = {}
        with open(rttm_file, "r") as file:
            for line in file:
                line = line.split()
                if not line or line[0].startswith("#"):
                    continue
                session, start, dur = line[1], line[3], line[4]

                start = float(start)
                end = start + float(dur)
                spk = line[-2] if line[-2] != "<NA>" else line[-3]

                session_speakers = speaker_indices.setdefault(session, {})
                label_idx = session_speakers.setdefault(spk, len(session_speakers))

                annotations.append((self.get_session_idx(session), start, end, label_idx))

        if not annotations:
            return np.empty(
                0,
                dtype=[("session_idx", "i1"), ("start", "f"), ("end", "f"), ("label_idx", "i1")],
            )

        segment_dtype = [
            (
                "session_idx",
                get_dtype(max(a[0] for a in annotations)),
            ),
            ("start", "f"),
            ("end", "f"),
            ("label_idx", get_dtype(max(a[3] for a in annotations))),
        ]

        return np.array(annotations, dtype=segment_dtype)

    def extract_wavforms(self, path, start, end, num_channels=8):
        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)
        data, sample_rate = sf.read(path, start=start, stop=end)
        assert sample_rate == self.sample_rate
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = np.einsum("tc->ct", data)
        return data[:num_channels, :]

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        session, path, chunk_start, chunk_end = self.chunk_indices[idx]
        data = self.extract_wavforms(path, chunk_start, chunk_end)  # [start, end)

        # chunked annotations
        session_idx = self.get_session_idx(session)
        annotations_session = self.annotations[self.annotations["session_idx"] == session_idx]
        chunked_annotations = annotations_session[
            (annotations_session["start"] < chunk_end) & (annotations_session["end"] > chunk_start)
        ]

        # discretize chunk annotations at model output resolution
        step = self.model_rf_step
        half = 0.5 * self.model_rf_duration

        start = np.maximum(chunked_annotations["start"], chunk_start) - chunk_start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunked_annotations["end"], chunk_end) - chunk_start - half
        end_idx = np.round(end / step).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunked_annotations["label_idx"]))
        num_labels = len(labels)

        mask_label = np.zeros((self.model_num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}
        for start, end, label in zip(start_idx, end_idx, chunked_annotations["label_idx"]):
            mapped_label = mapping[label]
            mask_label[start : end + 1, mapped_label] = 1

        return data, mask_label, session
