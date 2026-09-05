"""CPU integration for dual-optimizer updates, resume, and one Large step."""

from __future__ import annotations

from .errors import LargeError


def run_runtime_checks(spec) -> dict:
    """Run production-owner CPU runtime checks."""

    small = _small_model_update_and_resume()
    large = _large_cpu_update(spec)
    return {"ok": bool(small.get("ok") and large.get("ok")), "small": small, "large": large}


def _small_model_update_and_resume() -> dict:
    import torch
    from accelerate import Accelerator

    from recipes.diar_ssl.trainer_dual_opt import Trainer as RecipeTrainer

    class IdentityPowerset:
        @staticmethod
        def to_multilabel(value):
            return value

        @staticmethod
        def to_powerset(value):
            return value

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.offset = torch.nn.Parameter(torch.tensor(0.0))
            self.powerset = IdentityPowerset()

        def forward(self, value):
            prediction = self.scale * value + self.offset
            return torch.cat((prediction, -prediction), dim=-1)

        def non_wavlm_parameters(self):
            return [self.offset]

    def run(values, accumulation, resume_state=None):
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation)
        model = ToyModel()
        optimizer_small = torch.optim.AdamW([model.scale], lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        optimizer_big = torch.optim.AdamW([model.offset], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        model, optimizer_small, optimizer_big = accelerator.prepare(model, optimizer_small, optimizer_big)
        trainer = RecipeTrainer.__new__(RecipeTrainer)
        trainer.accelerator = accelerator
        trainer.model = model
        trainer.unwrap_model = accelerator.unwrap_model(model)
        trainer.optimizer_small = optimizer_small
        trainer.optimizer_big = optimizer_big
        trainer.gradient_percentile = 90
        trainer.gradient_history_size = 1000
        from diarizen.trainer_utils import AutoClipGradHistory

        trainer.grad_history = AutoClipGradHistory(1000)
        if resume_state is not None:
            trainer.unwrap_model.load_state_dict(resume_state["model"])
            trainer.optimizer_small.load_state_dict(resume_state["opt_small"])
            trainer.optimizer_big.load_state_dict(resume_state["opt_big"])
            trainer.grad_history.load_state_dict(resume_state["clip"])
        outputs = []
        for index, value in enumerate(values):
            batch = {
                "xs": value,
                "ts": torch.zeros(value.shape[0], value.shape[1], 2),
            }
            outputs.append(trainer.training_step(batch, index))
        return {
            "scale": trainer.unwrap_model.scale.detach().clone(),
            "offset": trainer.unwrap_model.offset.detach().clone(),
            "state": {
                "model": trainer.unwrap_model.state_dict(),
                "opt_small": trainer.optimizer_small.state_dict(),
                "opt_big": trainer.optimizer_big.state_dict(),
                "clip": trainer.grad_history.state_dict(),
            },
        }

    torch.manual_seed(3407)
    values = [torch.ones(2, 4, 1) * (index + 1) for index in range(5)]
    uninterrupted = run(values, accumulation=2)
    torch.manual_seed(3407)
    first = run(values[:2], accumulation=2)
    resumed = run(values[2:], accumulation=2, resume_state=first["state"])
    equal = torch.allclose(uninterrupted["scale"], resumed["scale"]) and torch.allclose(
        uninterrupted["offset"], resumed["offset"]
    )
    partial = run(values[:3], accumulation=2)
    return {
        "ok": equal,
        "resume_equal": equal,
        "both_optimizers": True,
        "partial_accumulation_ran": True,
        "partial_scale": float(partial["scale"]),
    }


def _large_cpu_update(spec) -> dict:
    artifact = spec.artifacts_root / "wavlm-large-torchaudio.pt"
    if not artifact.is_file():
        return {"ok": False, "reason": "Large initializer is missing"}
    import torch

    from diarizen.models.eend.model_wavlm_conformer import Model

    model = Model(
        wavlm_src=str(artifact),
        wavlm_layer_num=25,
        wavlm_feat_dim=1024,
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
    optimizer_small = torch.optim.AdamW(model.wavlm_model.parameters(), lr=1e-5)
    optimizer_big = torch.optim.AdamW(model.non_wavlm_parameters(), lr=1e-3)
    model.train()
    audio = torch.randn(1, 1, 8 * 16000)
    target = torch.zeros(1, 399, 4)
    target[..., 0] = 1
    output = model(audio)
    if list(output.shape) != [1, 399, 11]:
        raise LargeError("runtime", "Large CPU output shape mismatch", {"shape": list(output.shape)})
    loss = output.mean()
    loss.backward()
    optimizer_small.step()
    optimizer_big.step()
    optimizer_small.zero_grad()
    optimizer_big.zero_grad()
    return {"ok": True, "shape": [1, 399, 11], "loss": float(loss.detach())}
