"""Explicit projection roles; fail on unmatched requests rather than guessing."""

from torch import nn
from .lora import LoRALinear
from .dora import DoRALinear
from .pissa import PiSSALinear

ROLE_NAMES = {
    "query": ("q_proj", "query"),
    "key": ("k_proj", "key"),
    "value": ("v_proj", "value"),
    "output": ("o_proj", "out_proj"),
    "up": ("up_proj", "fc1"),
    "down": ("down_proj", "fc2"),
    "gate": ("gate_proj",),
}
GROUPS = {"attention": ["query", "key", "value", "output"], "ffn": ["up", "down", "gate"]}
GROUPS["combined"] = GROUPS["attention"] + GROUPS["ffn"]


def projection_names(model, targets="attention"):
    """Return explicit nn.Linear paths. Llama/Qwen/OPT/tiny are supported.

    GPT-2 Conv1D/fused QKV are deliberately unsupported, never silently
    transposed. Custom architectures can pass an explicit list of suffixes.
    """
    suffixes = (
        [n for role in GROUPS[targets] for n in ROLE_NAMES[role]]
        if isinstance(targets, str)
        else targets
    )
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in suffixes
    ]
    if not names:
        raise ValueError(
            f"No nn.Linear projections matched {targets}; inspect model.named_modules()"
        )
    return names


def inject_adapters(model, method="lora", rank=4, alpha=8, targets="attention", **options):
    """Freeze all parameters and replace selected linear layers atomically."""
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("Model already contains adapters")
    classes = {"lora": LoRALinear, "dora": DoRALinear, "pissa": PiSSALinear}
    cls = classes[method]
    names = projection_names(model, targets)
    # Wrapping PiSSA changes residual storage. Construct on copies before mutation.
    import copy

    replacements = [
        (name, cls(copy.deepcopy(model.get_submodule(name)), rank=rank, alpha=alpha, **options))
        for name in names
    ]
    model.requires_grad_(False)
    for name, layer in replacements:
        parent_path, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_path), child, layer)
    return names


def restore_linear(model, keep_adaptation=True):
    """Replace wrappers by ordinary frozen Linear layers, merged or original."""
    import torch

    for name, layer in list(model.named_modules()):
        if isinstance(layer, LoRALinear):
            with torch.no_grad():
                weight = layer.effective_weight() if keep_adaptation else layer.original_weight()
                layer.base.weight.copy_(weight)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, layer.base)
    return model


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0,
    }
