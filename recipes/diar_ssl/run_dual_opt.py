# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
from functools import partial
from pathlib import Path

import toml
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import GradientAccumulationPlugin, set_seed
from dataset import _collate_fn
from torch.utils.data import DataLoader

from diarizen.ckpt_utils import average_ckpt
from diarizen.logger import init_logging_logger
from diarizen.source_sampling import build_source_weighted_sampler
from diarizen.utils import instantiate
from diarizen.warm_start import WARM_START_MODE, PowersetWarmStartPolicy, warm_start_powerset_model
from recipes.speakrs.large.validation.config import ValidationMode, parse_validation_config
from recipes.speakrs.large.validation.trainer_bridge import SnapshotPublisher, TrainerBridge


def run(
    config,
    resume,
    *,
    mode: tuple[str, ...] | list[str] | None = None,
    validation_bridge: TrainerBridge | None = None,
    validation_publisher: SnapshotPublisher | None = None,
):
    init_logging_logger(config)
    selected_modes = tuple(mode) if mode is not None else tuple(getattr(globals().get("args"), "mode", ("train",)))
    validation_config = parse_validation_config(config)
    external_validation = validation_config.mode is ValidationMode.EXTERNAL
    if external_validation and "validate" in selected_modes:
        raise ValueError("external validation mode does not load or run validation data")
    if external_validation and validation_bridge is None:
        if validation_publisher is None:
            from recipes.speakrs.large.contracts import ObjectStoreDestination
            from recipes.speakrs.large.storage import backend_from_destination
            from recipes.speakrs.large.validation.publication import SnapshotPublication

            destination_config = validation_config.object_store_destination
            runtime_destination = ObjectStoreDestination(
                provider=destination_config.provider,
                endpoint="wrangler",
                bucket=destination_config.bucket,
                prefix=destination_config.prefix,
                credential_reference="wrangler",
            )
            validation_publisher = SnapshotPublication(
                runtime_destination,
                backend_from_destination(runtime_destination),
                validation_config.publication_transaction_root,
            )
        validation_bridge = TrainerBridge.from_config(validation_config, validation_publisher)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # keep the qualified effective batch across epoch boundaries
    accumulation = GradientAccumulationPlugin(
        num_steps=config["trainer"]["args"]["gradient_accumulation_steps"],
        sync_with_dataloader=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=accumulation,
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(config["meta"]["seed"], device_specific=True)

    model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info

    if config["finetune"]["finetune"]:
        accelerator.print("fine-tuning...")
        if config["finetune"].get("mode") == WARM_START_MODE:
            warm_start_powerset_model(model, PowersetWarmStartPolicy.from_config(config["finetune"]))
        else:
            model = average_ckpt(config["finetune"]["ckpt_dir"], model)

    optimizer_small = instantiate(
        config["optimizer_small"]["path"],
        args={"params": model.wavlm_model.parameters()}
        | config["optimizer_small"]["args"]
        | {"lr": config["optimizer_small"]["args"]["lr"]},
    )
    optimizer_big = instantiate(
        config["optimizer_big"]["path"],
        args={"params": model.non_wavlm_parameters()}
        | config["optimizer_big"]["args"]
        | {"lr": config["optimizer_big"]["args"]["lr"]},
    )

    (model, optimizer_small, optimizer_big) = accelerator.prepare(model, optimizer_small, optimizer_big)

    # pass model receptive field info to dataset
    train_dataset_config = config["train_dataset"]["args"]
    train_dataset_config["model_num_frames"] = model_num_frames
    train_dataset_config["model_rf_duration"] = model_rf_duration
    train_dataset_config["model_rf_step"] = model_rf_step

    collate_fn_partial = partial(_collate_fn, max_speakers_per_chunk=config["model"]["args"]["max_speakers_per_chunk"])

    train_dataloader = None
    validate_dataloader = None
    if "train" in selected_modes:
        train_dataset = instantiate(config["train_dataset"]["path"], args=train_dataset_config)
        sampling_config = config["train_dataset"].get("sampling")
        if sampling_config is None:
            train_sampler = None
            train_generator = torch.Generator().manual_seed(config["meta"]["seed"])
        else:
            if set(sampling_config) != {"policy_file", "bundle_file"}:
                raise ValueError("training sampling configuration fields are invalid")
            train_sampler = build_source_weighted_sampler(
                policy_path=sampling_config["policy_file"],
                bundle_path=sampling_config["bundle_file"],
                chunk_recording_ids=train_dataset.chunk_recording_ids,
            )
            train_generator = train_sampler.generator
        train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=collate_fn_partial,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=train_generator,
            **config["train_dataset"]["dataloader"],
        )
        train_dataloader = accelerator.prepare(train_dataloader)

    if not external_validation and ("train" in selected_modes or "validate" in selected_modes):
        validate_dataset_config = config["validate_dataset"]["args"]
        validate_dataset_config["model_num_frames"] = model_num_frames
        validate_dataset_config["model_rf_duration"] = model_rf_duration
        validate_dataset_config["model_rf_step"] = model_rf_step
        validate_dataset = instantiate(config["validate_dataset"]["path"], args=validate_dataset_config)
        validate_dataloader = DataLoader(
            dataset=validate_dataset,
            collate_fn=collate_fn_partial,
            shuffle=False,
            **config["validate_dataset"]["dataloader"],
        )
        validate_dataloader = accelerator.prepare(validate_dataloader)

    trainer_arguments = {
        "accelerator": accelerator,
        "config": config,
        "resume": resume,
        "model": model,
        "optimizer_small": optimizer_small,
        "optimizer_big": optimizer_big,
    }
    if validation_bridge is not None:
        trainer_arguments["validation_bridge"] = validation_bridge
    trainer = instantiate(config["trainer"]["path"], initialize=False)(
        **trainer_arguments,
    )

    for flag in selected_modes:
        if flag == "train":
            trainer.train(train_dataloader, validate_dataloader)
        elif flag == "validate":
            trainer.validate(validate_dataloader)
        else:
            raise ValueError(f"Unknown mode: {flag}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-ZEN based EEND framework")
    parser.add_argument(
        "-C",
        "--configuration",
        required=True,
        type=str,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "-M",
        "--mode",
        nargs="+",
        type=str,
        default=["train"],
        choices=["train", "validate"],
        help="Mode of the experiment.",
    )
    parser.add_argument(
        "-R",
        "--resume",
        action="store_true",
        help="Resume the experiment from latest checkpoint.",
    )
    parser.add_argument(
        "-FT",
        "--finetune",
        action="store_true",
        help="Label of fine-tuning.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint path for fine-tuning.",
    )

    args = parser.parse_args()

    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    config["meta"]["exp_id"] = config_path.stem
    config["meta"]["config_path"] = config_path.as_posix()

    run(config, args.resume)
