"""Portable adapter-only files with base identity verification.

PiSSA files contain INITIAL factors as well as trained factors. This avoids
depending on SVD signs, randomized SVD RNG, or implementation versions.
"""

import torch
from .lora import LoRALinear
from .pissa import PiSSALinear
from ..utils.io import save_torch


def save_adapter(model, path, base_identity):
    layers = {}
    for name, layer in model.named_modules():
        if isinstance(layer, LoRALinear):
            # Only factorized state and magnitude: no dense frozen weights.
            state = {
                key: value.detach().cpu()
                for key, value in layer.state_dict().items()
                if isinstance(value, torch.Tensor) and not key.startswith("base.")
            }
            layers[name] = {"config": layer.get_extra_state(), "state": state}
    if not layers:
        raise ValueError("Model contains no adapters")
    save_torch(path, {"format": 1, "base_identity": base_identity, "layers": layers})


def load_adapter(model, path, base_identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != 1 or payload["base_identity"] != base_identity:
        raise ValueError("Wrong checkpoint format or base identity")
    live = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if set(live) != set(payload["layers"]):
        raise ValueError("Adapter target modules differ")
    for name, layer in live.items():
        record = payload["layers"][name]
        if layer.get_extra_state() != record["config"]:
            raise ValueError(f"Adapter configuration mismatch at {name}")
        with torch.no_grad():
            if isinstance(layer, PiSSALinear):
                # Undo the new constructor's decomposition, then use the saved one.
                original = layer.original_weight().clone()
            for key, tensor in record["state"].items():
                layer.state_dict()[key].copy_(tensor)
            if isinstance(layer, PiSSALinear):
                layer.base.weight.copy_(original - layer.initial_component())
        layer.unmerge()
    return model
