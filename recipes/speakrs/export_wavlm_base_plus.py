#!/usr/bin/env python3
"""Export the torchaudio WavLM Base+ checkpoint in DiariZen format."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torchaudio

from diarizen.models.module.wav2vec2.model import wav2vec2_model
from diarizen.models.module.wavlm_config import get_config


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def convert_state_dict(state_dict: dict[str, torch.Tensor], layer_count: int) -> dict[str, torch.Tensor]:
    """Convert torchaudio attention parameters to the DiariZen layout."""
    converted = dict(state_dict)
    if converted and all(key.startswith("model.") for key in converted):
        converted = {key[len("model.") :]: value for key, value in converted.items()}
    converted["feature_extractor.dummy_weight"] = torch.ones(512)

    for layer in range(layer_count):
        prefix = f"encoder.transformer.layers.{layer}.attention"
        projection_weight = converted.pop(f"{prefix}.attention.in_proj_weight")
        projection_bias = converted.pop(f"{prefix}.attention.in_proj_bias")
        query_weight, key_weight, value_weight = projection_weight.chunk(3)
        query_bias, key_bias, value_bias = projection_bias.chunk(3)

        converted[f"{prefix}.q_proj.weight"] = query_weight
        converted[f"{prefix}.q_proj.bias"] = query_bias
        converted[f"{prefix}.k_proj.weight"] = key_weight
        converted[f"{prefix}.k_proj.bias"] = key_bias
        converted[f"{prefix}.v_proj.weight"] = value_weight
        converted[f"{prefix}.v_proj.bias"] = value_bias
        converted[f"{prefix}.out_proj.weight"] = converted.pop(f"{prefix}.attention.out_proj.weight")
        converted[f"{prefix}.out_proj.bias"] = converted.pop(f"{prefix}.attention.out_proj.bias")

    return converted


def main() -> None:
    """Download, verify, and export WavLM Base+ weights."""
    output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "wavlm-base-plus-torchaudio.pt"
    provenance_path = output_dir / "wavlm-base-plus-torchaudio.json"

    config = get_config("wavlm_base")
    source_model = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model()
    state_dict = convert_state_dict(source_model.state_dict(), config["encoder_num_layers"])

    target_model = wav2vec2_model(**config)
    incompatible = target_model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Incompatible WavLM state: {incompatible}")

    temporary_path = output_path.with_suffix(".pt.part")
    torch.save({"config": config, "state_dict": state_dict}, temporary_path)
    temporary_path.replace(output_path)

    provenance = {
        "artifact": output_path.name,
        "artifact_sha256": sha256(output_path),
        "source": "torchaudio.pipelines.WAVLM_BASE_PLUS",
        "torchaudio_version": torchaudio.__version__,
        "torch_version": torch.__version__,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
