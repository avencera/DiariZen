# Licensed under the MIT license.
# Adopted from https://github.com/espnet/espnet/blob/master/egs2/chime8_task1/diar_asr1/local/pyannote_diarize.py
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import toml
import torch
import torchaudio
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation
from scipy.ndimage import median_filter
from torch.torch_version import TorchVersion

from diarizen.ckpt_utils import load_metric_summary
from diarizen.inference_identity import build_engine_identity


# pytorch 2.6+ needs an explicit allowlist for these checkpoint metadata types
torch.serialization.add_safe_globals([TorchVersion, Problem, Resolution, Specifications])


RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_VERSION = 4
RTTM_COMPLETION_MARKER_SUFFIX = ".complete"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one inference input."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def describe_file(path: Path) -> dict[str, Any]:
    """Return stable identity fields for one inference input file."""

    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def describe_segmentation(segmentation) -> dict[str, Any]:
    """Describe a single checkpoint or a ranked checkpoint average."""

    if isinstance(segmentation, list):
        checkpoints = []
        for record in segmentation:
            checkpoint = describe_file(Path(record["bin_path"]))
            checkpoint["epoch"] = int(record["epoch"])
            checkpoint["Loss"] = float(record["Loss"])
            checkpoints.append(checkpoint)

        return {"kind": "checkpoint_average", "checkpoints": checkpoints}

    if segmentation:
        return {
            "kind": "checkpoint",
            "checkpoint": describe_file(Path(segmentation)),
        }

    return {"kind": "implicit"}


def describe_audio_inputs(wav_scp_path: Path) -> dict[str, Any]:
    """Describe the manifest and audio files that produce an inference run."""

    resolved_manifest = wav_scp_path.expanduser().resolve(strict=True)
    recordings = []
    for session, audio_path in sorted(load_scp(resolved_manifest.as_posix()).items()):
        recordings.append(
            {
                "session": session,
                **describe_file(Path(audio_path)),
            }
        )

    return {
        "wav_scp": describe_file(resolved_manifest),
        "recordings": recordings,
    }


def build_run_manifest(args, config_path: Path, segmentation) -> dict[str, Any]:
    """Build the identity of one resumable inference run."""

    embedding_model = Path(args.embedding_model).expanduser().resolve(strict=True)
    return {
        "version": RUN_MANIFEST_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input": describe_audio_inputs(Path(args.in_wav_scp)),
        "configuration": describe_file(config_path),
        "segmentation": describe_segmentation(segmentation),
        "embedding": describe_file(embedding_model),
        "engine": build_engine_identity(args.clustering_method, getattr(args, "diarizen_hub", None)),
        "inference": {
            "seg_duration": args.seg_duration,
            "segmentation_step": args.segmentation_step,
            "batch_size": args.batch_size,
            "apply_median_filtering": args.apply_median_filtering,
        },
        "clustering": {
            "method": args.clustering_method,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
            "ahc_criterion": args.ahc_criterion,
            "ahc_threshold": args.ahc_threshold,
            "min_cluster_size": args.min_cluster_size,
            "Fa": args.Fa,
            "Fb": args.Fb,
            "lda_dim": args.lda_dim,
            "max_iters": args.max_iters,
        },
    }


def rttm_completion_marker(rttm_path: Path) -> Path:
    """Return the completion marker path associated with an RTTM file."""

    return rttm_path.with_name(rttm_path.name + RTTM_COMPLETION_MARKER_SUFFIX)


def is_completed_rttm(rttm_path: Path) -> bool:
    """Return whether an RTTM file is complete, including an empty result."""

    if not rttm_path.is_file():
        return False

    return rttm_path.stat().st_size > 0 or rttm_completion_marker(rttm_path).is_file()


def write_rttm_atomically(rttm_path: Path, rttm: str) -> None:
    """Publish an RTTM and mark empty output complete after the publish."""

    temporary_rttm = rttm_path.with_suffix(".partial.rttm")
    temporary_rttm.write_text(rttm)

    completion_marker = rttm_completion_marker(rttm_path)
    completion_marker.unlink(missing_ok=True)
    temporary_rttm.replace(rttm_path)

    if not rttm.strip():
        temporary_marker = completion_marker.with_name(completion_marker.name + ".partial")
        temporary_marker.write_text("complete\n")
        temporary_marker.replace(completion_marker)


def initialize_run_directory(output_dir: Path, manifest: dict[str, Any]) -> None:
    """Create or validate the identity of a resumable inference directory."""

    output_dir.mkdir(exist_ok=True, parents=True)
    manifest_path = output_dir / RUN_MANIFEST_FILENAME
    completed_rttms = [
        path
        for path in output_dir.glob("*.rttm")
        if not path.name.endswith(".partial.rttm") and is_completed_rttm(path)
    ]

    if manifest_path.exists():
        saved_manifest = json.loads(manifest_path.read_text())
        if saved_manifest != manifest:
            raise ValueError(f"Inference settings do not match {manifest_path}; use a new output directory")
        return

    if completed_rttms:
        raise ValueError(
            f"Cannot validate {len(completed_rttms)} existing RTTM files in {output_dir}; use a new output directory"
        )

    temporary_manifest = manifest_path.with_suffix(".partial.json")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)


def load_ranked_checkpoints(index_path: Path, count: int):
    """Load the best model-only checkpoint paths from a ranked index."""

    records = json.loads(index_path.read_text())
    if len(records) < count:
        raise ValueError(f"Ranked checkpoint index has {len(records)} entries; {count} are required")
    checkpoints = []
    for record in records[:count]:
        epoch = int(record["epoch"])
        checkpoint = index_path.parent / f"epoch_{epoch:04d}" / "pytorch_model.bin"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoints.append(
            {
                "epoch": epoch,
                "bin_path": checkpoint,
                "Loss": float(record["score"]),
            }
        )

    return checkpoints


def load_scp(scp_file: str) -> dict[str, str]:
    """Return the unique session-to-audio mapping from a Kaldi-style file."""

    recordings = {}
    for line_number, line in enumerate(Path(scp_file).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            raise ValueError(f"Invalid wav.scp row at {scp_file}:{line_number}")
        session, audio_path = fields
        if session in recordings:
            raise ValueError(f"Duplicate session in {scp_file}:{line_number}: {session}")
        recordings[session] = audio_path

    if not recordings:
        raise ValueError(f"No recordings found in {scp_file}")

    return recordings


def _has_activity(data: np.ndarray) -> bool:
    """Return whether segmentation or speaker-count data contains activity."""

    return bool(np.any(np.nan_to_num(data, nan=0.0) > 0))


def diarize_session(sess_name, in_wav, pipeline, min_speakers=1, max_speakers=20, apply_median_filtering=True):
    print("Extracting segmentations...")
    waveform, sample_rate = torchaudio.load(in_wav)
    waveform = torch.unsqueeze(waveform[0], 0)  # force to use the SDM data
    segmentations = pipeline.get_segmentations({"waveform": waveform, "sample_rate": sample_rate}, soft=False)

    if apply_median_filtering:
        segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode="reflect")

    # binarize segmentation
    binarized_segmentations = segmentations  # powerset

    if not _has_activity(binarized_segmentations.data):
        return Annotation(uri=sess_name)

    # estimate frame-level number of instantaneous speakers
    count = pipeline.speaker_count(
        binarized_segmentations,
        pipeline._segmentation.model._receptive_field,
        warm_up=(0.0, 0.0),
    )

    if not _has_activity(count.data):
        return Annotation(uri=sess_name)

    print("Extracting Embeddings.")
    embeddings = pipeline.get_embeddings(
        {"waveform": waveform, "sample_rate": sample_rate},
        binarized_segmentations,
        exclude_overlap=pipeline.embedding_exclude_overlap,
    )

    #  shape: (num_chunks, local_num_speakers, dimension)
    print("Clustering.")
    hard_clusters, _, _ = pipeline.clustering(
        embeddings=embeddings,
        segmentations=binarized_segmentations,
        min_clusters=min_speakers,
        max_clusters=max_speakers,
    )

    # during counting, we could possibly overcount the number of instantaneous
    # speakers due to segmentation errors, so we cap the maximum instantaneous number
    # of speakers by the `max_speakers` value
    count.data = np.minimum(count.data, max_speakers).astype(np.int8)

    # keep track of inactive speakers
    inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
    #   shape: (num_chunks, num_speakers)

    # reconstruct discrete diarization from raw hard clusters
    hard_clusters[inactive_speakers] = -2
    discrete_diarization, _ = pipeline.reconstruct(
        segmentations,
        hard_clusters,
        count,
    )

    # convert to annotation
    to_annotation = Binarize(onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0)
    result = to_annotation(discrete_diarization)
    result.uri = sess_name

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "This script performs diarization using DiariZen pipeline ",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required arguments
    parser.add_argument(
        "-C",
        "--configuration",
        type=str,
        required=True,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "-i",
        "--in_wav_scp",
        type=str,
        required=True,
        help="test wav.scp.",
        dest="in_wav_scp",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model.",
    )

    # Optional arguments
    parser.add_argument("--diarizen_hub", type=str, help="Path to DiariZen model hub directory.")
    parser.add_argument(
        "--avg_ckpt_num",
        type=int,
        default=5,
        help="the number of chckpoints of model averaging",
    )
    parser.add_argument(
        "--val_metric",
        type=str,
        default="Loss",
        help="validation metric",
        choices=["Loss", "DER"],
    )
    parser.add_argument(
        "--val_mode",
        type=str,
        default="best",
        help="validation metric mode",
        choices=["best", "prev", "center"],
    )
    parser.add_argument(
        "--val_metric_summary",
        type=str,
        default="",
        help="val_metric_summary",
    )
    parser.add_argument(
        "--ranked_checkpoint_index",
        type=Path,
        help="Path to the ranked model-only checkpoint index",
    )
    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="",
        help="Path to pretrained segmentation model.",
    )

    # Inference parameters
    parser.add_argument(
        "--seg_duration",
        type=int,
        default=16,
        help="Segment duration in seconds.",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for inference.",
    )
    parser.add_argument(
        "--apply_median_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply median filtering to segmentation output.",
    )

    # Clustering parameters
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="VBxClustering",
        choices=["VBxClustering", "AgglomerativeClustering"],
        help="Clustering method to use.",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=1,
        help="Minimum number of speakers.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=20,
        help="Maximum number of speakers.",
    )
    parser.add_argument(
        "--ahc_criterion",
        type=str,
        default="distance",
        help="AHC criterion (for VBx).",
    )
    parser.add_argument(
        "--ahc_threshold",
        type=float,
        default=0.6,
        help="AHC threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=13,
        help="Minimum cluster size (for AHC).",
    )
    parser.add_argument(
        "--Fa",
        type=float,
        default=0.07,
        help="VBx Fa parameter.",
    )
    parser.add_argument(
        "--Fb",
        type=float,
        default=0.8,
        help="VBx Fb parameter.",
    )
    parser.add_argument(
        "--lda_dim",
        type=int,
        default=128,
        help="VBx LDA dimension.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="VBx maximum iterations.",
    )

    args = parser.parse_args()
    print(args)

    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    ckpt_path = config_path.parent / "checkpoints"
    segmentation = args.segmentation_model
    if args.ranked_checkpoint_index:
        if args.val_metric_summary:
            raise ValueError("Use either --ranked_checkpoint_index or --val_metric_summary, not both")
        segmentation = load_ranked_checkpoints(args.ranked_checkpoint_index.expanduser().resolve(), args.avg_ckpt_num)
    elif args.val_metric_summary:
        val_metric_lst = load_metric_summary(args.val_metric_summary, ckpt_path)
        val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[args.val_metric])
        best_val_metric_idx = val_metric_lst.index(val_metric_lst_sorted[0])
        if args.val_mode == "best":
            segmentation = val_metric_lst_sorted[: args.avg_ckpt_num]
        elif args.val_mode == "prev":
            segmentation = val_metric_lst[best_val_metric_idx - args.avg_ckpt_num + 1 : best_val_metric_idx + 1]
        else:
            segmentation = val_metric_lst[
                best_val_metric_idx - args.avg_ckpt_num // 2 : best_val_metric_idx + args.avg_ckpt_num // 2 + 1
            ]
        assert len(segmentation) == args.avg_ckpt_num

    output_dir = Path(args.out_dir).expanduser().resolve()
    run_manifest = build_run_manifest(args, config_path, segmentation)
    initialize_run_directory(output_dir, run_manifest)

    # create, instantiate and apply the pipeline
    diarization_pipeline = SpeakerDiarizationPipeline(
        config=config,
        seg_duration=args.seg_duration,
        segmentation=segmentation,
        segmentation_step=args.segmentation_step,
        embedding=args.embedding_model,
        embedding_exclude_overlap=True,
        clustering=args.clustering_method,
        embedding_batch_size=args.batch_size,
        segmentation_batch_size=args.batch_size,
        device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    )

    if args.clustering_method == "AgglomerativeClustering":
        PIPELINE_PARAMS = {
            "clustering": {
                "method": "centroid",
                "min_cluster_size": args.min_cluster_size,
                "threshold": args.ahc_threshold,
            }
        }
    elif args.clustering_method == "VBxClustering":
        PIPELINE_PARAMS = {
            "clustering": {
                "ahc_criterion": args.ahc_criterion,
                "ahc_threshold": args.ahc_threshold,
                "Fa": args.Fa,
                "Fb": args.Fb,
            }
        }
        diarization_pipeline.clustering.plda_dir = os.path.join(args.diarizen_hub, "plda")
        diarization_pipeline.clustering.lda_dim = args.lda_dim
        diarization_pipeline.clustering.maxIters = args.max_iters
    else:
        raise ValueError(f"Unsupported clustering method: {args.clustering_method}")

    diarization_pipeline.instantiate(PIPELINE_PARAMS)

    audio_dict = load_scp(args.in_wav_scp)

    for sess, in_wav in audio_dict.items():
        rttm_out = output_dir / f"{sess}.rttm"
        if is_completed_rttm(rttm_out):
            print(f"Reusing completed session: {sess}")
            continue
        print(f"Diarizing Session: {sess}")
        diar_result = diarize_session(
            sess_name=sess,
            in_wav=in_wav,
            pipeline=diarization_pipeline,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            apply_median_filtering=args.apply_median_filtering,
        )
        write_rttm_atomically(rttm_out, diar_result.to_rttm())
