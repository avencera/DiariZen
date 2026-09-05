"""Shared WavLM export used by Base+ and Large."""

from __future__ import annotations

from pathlib import Path

from .errors import LargeError
from .jsonio import write_json


def export_variant(variant: str, output_dir: Path) -> dict:
    """Export Base+ or Large through the shared conversion layout."""

    import torch
    import torchaudio

    from diarizen.models.module.wav2vec2.model import wav2vec2_model
    from diarizen.models.module.wavlm_config import get_config
    from recipes.speakrs.export_wavlm_base_plus import convert_state_dict, sha256

    output_dir.mkdir(parents=True, exist_ok=True)
    if variant == "base_plus":
        config_name = "wavlm_base"
        pipeline = torchaudio.pipelines.WAVLM_BASE_PLUS
        artifact_name = "wavlm-base-plus-torchaudio.pt"
        provenance_name = "wavlm-base-plus-torchaudio.json"
        layer_num = 13
        feat_dim = 768
        source_name = "torchaudio.pipelines.WAVLM_BASE_PLUS"
    elif variant == "large":
        config_name = "wavlm_large"
        pipeline = torchaudio.pipelines.WAVLM_LARGE
        artifact_name = "wavlm-large-torchaudio.pt"
        provenance_name = "wavlm-large-torchaudio.json"
        layer_num = 25
        feat_dim = 1024
        source_name = "torchaudio.pipelines.WAVLM_LARGE"
    else:
        raise LargeError("export", f"unknown WavLM variant {variant}")

    config = get_config(config_name)
    source_model = pipeline.get_model()
    source_model.eval()
    state_dict = convert_state_dict(source_model.state_dict(), config["encoder_num_layers"])
    target_model = wav2vec2_model(**config)
    incompatible = target_model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise LargeError("export", f"Incompatible WavLM state: {incompatible}")

    output_path = output_dir / artifact_name
    temporary_path = output_path.with_suffix(".pt.part")
    torch.save({"config": config, "state_dict": state_dict}, temporary_path)
    temporary_path.replace(output_path)

    parity = _parity_report(source_model, target_model, config) if variant == "large" else {}
    diarization = None
    if variant == "large":
        diarization = _diarization_shape(output_path, layer_num, feat_dim)

    provenance = {
        "artifact": output_path.name,
        "artifact_sha256": sha256(output_path),
        "source": source_name,
        "source_sha256": _module_identity(source_model),
        "torchaudio_version": torchaudio.__version__,
        "torch_version": torch.__version__,
        "variant": variant,
        "wavlm_blocks": config["encoder_num_layers"],
        "wavlm_width": config["encoder_embed_dim"],
        "wavlm_heads": config["encoder_total_num_heads"][0],
        "normalize_waveform": config["normalize_waveform"],
        "strict_load": True,
        "parity": parity,
        "diarization": diarization,
    }
    write_json(output_dir / provenance_name, provenance)
    if variant == "large":
        write_json(output_dir / "wavlm-large-parity.json", parity)
    return provenance


def _module_identity(model) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _relative_l2(left, right) -> float:
    import torch

    baseline = torch.linalg.vector_norm(left.float())
    delta = torch.linalg.vector_norm((left.float() - right.float()))
    if float(baseline) == 0.0:
        return float(delta)
    return float(delta / baseline)


def _parity_report(source_model, target_model, config) -> dict:
    import torch

    target_model.eval()
    source_model.eval()
    lengths = [16000, 32000, 8 * 16000]
    layer_errors = []
    with torch.no_grad():
        for samples in lengths:
            audio = torch.randn(1, samples)
            if config["normalize_waveform"]:
                source_audio = torch.nn.functional.layer_norm(audio, audio.shape)
            else:
                source_audio = audio
            source_out = source_model.extract_features(source_audio)
            if isinstance(source_out, tuple):
                source_layers = source_out[0]
            else:
                source_layers = source_out
            target_layers, _ = target_model.extract_features(audio)
            # Target includes encoder input as extra representation 0.
            comparable_target = target_layers[1:] if len(target_layers) == len(source_layers) + 1 else target_layers
            if len(comparable_target) != len(source_layers):
                raise LargeError(
                    "export",
                    "layer count mismatch during parity",
                    {"source": len(source_layers), "target": len(target_layers)},
                )
            for source_layer, target_layer in zip(source_layers, comparable_target):
                layer_errors.append(_relative_l2(source_layer, target_layer))
    maximum = max(layer_errors) if layer_errors else 1.0
    if maximum > 1e-4:
        raise LargeError("export", "WavLM Large parity exceeded 1e-4", {"max_relative_l2": maximum})
    return {
        "max_relative_l2": maximum,
        "threshold": 1e-4,
        "lengths": lengths,
        "target_representations": 25,
        "source_layers": 24,
        "normalize_waveform": True,
    }


def _diarization_shape(wavlm_path: Path, layer_num: int, feat_dim: int) -> dict:
    import torch

    from diarizen.models.eend.model_wavlm_conformer import Model

    model = Model(
        wavlm_src=str(wavlm_path),
        wavlm_layer_num=layer_num,
        wavlm_feat_dim=feat_dim,
        attention_in=256,
        ffn_hidden=1024,
        num_head=4,
        num_layer=4,
        kernel_size=31,
        dropout=0.1,
        chunk_size=8,
        max_speakers_per_chunk=4,
        max_speakers_per_frame=2,
        strict_wavlm_load=True,
    )
    model.eval()
    audio = torch.randn(2, 1, 8 * 16000)
    with torch.no_grad():
        output = model(audio)
    shape = list(output.shape)
    if shape != [2, 399, 11]:
        raise LargeError("export", "Large diarization output shape mismatch", {"shape": shape})
    return {"shape": shape, "batch": 2, "frames": 399, "classes": 11}
