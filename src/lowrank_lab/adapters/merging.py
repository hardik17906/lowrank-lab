"""Explicit inference merge lifecycle."""

from .lora import LoRALinear


def merge_adapters(model):
    model.eval()
    for layer in model.modules():
        if isinstance(layer, LoRALinear):
            layer.merge()
    return model


def unmerge_adapters(model):
    for layer in model.modules():
        if isinstance(layer, LoRALinear):
            layer.unmerge()
    return model
