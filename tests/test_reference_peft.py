"""Optional independent implementation checks. Core never imports PEFT."""

import copy
import pytest
import torch
from torch import nn
from lowrank_lab.adapters import LoRALinear, DoRALinear, PiSSALinear

peft = pytest.importorskip("peft")


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
def test_peft_forward_and_factor_gradients(method):
    from peft.tuners.lora.layer import Linear

    base = nn.Linear(7, 5, bias=True)
    classes = {"lora": LoRALinear, "dora": DoRALinear, "pissa": PiSSALinear}
    manual = classes[method](copy.deepcopy(base), rank=3, alpha=6)
    cfg = peft.LoraConfig(
        r=3,
        lora_alpha=6,
        lora_dropout=0.0,
        init_lora_weights="pissa" if method == "pissa" else True,
        use_dora=method == "dora",
    )
    reference = Linear(copy.deepcopy(base), adapter_name="default", config=cfg, r=3, lora_alpha=6)
    with torch.no_grad():
        # Align factor values rather than relying on RNG order or SVD signs.
        reference.lora_A["default"].weight.copy_(manual.A)
        reference.lora_B["default"].weight.copy_(manual.B)
        reference.base_layer.weight.copy_(manual.base.weight)
        manual.B.add_(0.03)
        reference.lora_B["default"].weight.copy_(manual.B)
        if method == "dora":
            reference.lora_magnitude_vector["default"].weight.copy_(manual.magnitude)
    x = torch.randn(4, 7)
    torch.testing.assert_close(manual(x), reference(x), atol=2e-6, rtol=2e-5)
    manual(x).square().sum().backward()
    reference(x).square().sum().backward()
    for actual, expected in [
        (manual.A.grad, reference.lora_A["default"].weight.grad),
        (manual.B.grad, reference.lora_B["default"].weight.grad),
    ]:
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    if method == "dora":
        torch.testing.assert_close(
            manual.magnitude.grad,
            reference.lora_magnitude_vector["default"].weight.grad,
            atol=3e-5,
            rtol=3e-5,
        )
