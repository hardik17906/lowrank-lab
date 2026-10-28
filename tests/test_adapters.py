import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from lowrank_lab.adapters import (
    LoRALinear,
    DoRALinear,
    PiSSALinear,
    inject_adapters,
    restore_linear,
)
from lowrank_lab.adapters.checkpoint import save_adapter, load_adapter
from lowrank_lab.models import TinyTransformer, TinyConfig


@pytest.mark.parametrize("cls", [LoRALinear, DoRALinear, PiSSALinear])
@pytest.mark.parametrize("rank", [0, 1, 3, 5])
def test_shapes_initial_function_and_freeze(cls, rank):
    base = nn.Linear(7, 5).double()
    x = torch.randn(2, 4, 7, dtype=torch.float64)
    original = copy.deepcopy(base)
    layer = cls(base, rank=rank, alpha=6)
    assert layer.A.shape == (rank, 7)
    assert layer.B.shape == (5, rank)
    torch.testing.assert_close(layer(x), original(x), atol=1e-12, rtol=1e-12)
    assert not layer.base.weight.requires_grad
    assert not layer.base.bias.requires_grad
    if cls == LoRALinear:
        assert torch.count_nonzero(layer.delta_weight()) == 0


@pytest.mark.parametrize("cls", [LoRALinear, DoRALinear, PiSSALinear])
def test_gradients_and_frozen_base(cls):
    layer = cls(nn.Linear(7, 5).double(), rank=3)
    original = layer.base.weight.detach().clone()
    optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.01)
    x, target = torch.randn(4, 7).double(), torch.randn(4, 5).double()
    for step in range(2):
        optimizer.zero_grad()
        F.mse_loss(layer(x), target).backward()
        assert layer.B.grad.abs().sum() > 0
        if step == 1 or cls == PiSSALinear:
            assert layer.A.grad.abs().sum() > 0
        if cls == DoRALinear:
            assert layer.magnitude.grad.abs().sum() > 0
        optimizer.step()
    assert layer.base.weight.grad is None
    torch.testing.assert_close(layer.base.weight, original, atol=0, rtol=0)


@pytest.mark.parametrize("cls", [LoRALinear, DoRALinear, PiSSALinear])
def test_merge_unmerge_idempotence_state_dict_and_disabled(cls):
    layer = cls(nn.Linear(7, 5).double(), rank=3)
    with torch.no_grad():
        layer.B.add_(0.04)
    x = torch.randn(3, 7).double()
    layer.eval()
    reference = layer(x)
    before = layer.base.weight.detach().clone()
    layer.merge().merge()
    torch.testing.assert_close(layer(x), reference, atol=1e-12, rtol=1e-12)
    other = cls(nn.Linear(7, 5).double(), rank=3)
    other.load_state_dict(layer.state_dict())
    torch.testing.assert_close(other(x), reference)
    assert not other.merged
    layer.unmerge().unmerge()
    torch.testing.assert_close(layer(x), reference)
    torch.testing.assert_close(layer.base.weight, before, atol=0, rtol=0)
    layer.merge().train()
    assert not layer.merged
    layer.enabled = False
    torch.testing.assert_close(layer(x), F.linear(x, layer.original_weight(), layer.base.bias))


@pytest.mark.parametrize("detach", [False, True])
def test_dora_reference_forward_gradient_axes_and_bias(detach):
    base = nn.Linear(3, 2, bias=True).double()
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]))
        base.bias.copy_(torch.tensor([0.7, -0.4]))
    layer = DoRALinear(base, rank=2, alpha=2, detach_norm=detach)
    torch.testing.assert_close(layer.magnitude, torch.tensor([5.0, 2.0], dtype=torch.float64))
    with torch.no_grad():
        layer.B.normal_()
        layer.magnitude.add_(0.5)
    x = torch.randn(4, 3).double()
    a, b, m = [p.detach().clone().requires_grad_() for p in [layer.A, layer.B, layer.magnitude]]
    # Independent row-by-row scalar reference, matching NVIDIA's dora_simple option.
    outputs = []
    for j in range(2):
        v = base.weight[j] + b[j] @ a
        denominator = (v.square().sum()).sqrt()
        if detach:
            denominator = denominator.detach()
        outputs.append((x * v).sum(-1) * m[j] / denominator + base.bias[j])
    reference = torch.stack(outputs, dim=-1)
    torch.testing.assert_close(layer(x), reference, atol=1e-12, rtol=1e-12)
    g = torch.randn_like(reference)
    expected = torch.autograd.grad(reference, [a, b, m], g)
    actual = torch.autograd.grad(layer(x), [layer.A, layer.B, layer.magnitude], g)
    for e, v in zip(expected, actual):
        torch.testing.assert_close(e, v, atol=1e-10, rtol=1e-10)


def test_dora_full_norm_gradcheck_and_zero_rows_stable():
    layer = DoRALinear(nn.Linear(3, 2).double(), rank=2, detach_norm=False)
    with torch.no_grad():
        layer.B.normal_()
    x = torch.randn(2, 3).double().requires_grad_()
    assert torch.autograd.gradcheck(lambda t: layer(t), (x,), eps=1e-6, atol=1e-5)
    with torch.no_grad():
        layer.base.weight.zero_()
        layer.B.zero_()
        layer.magnitude.zero_()
    assert torch.isfinite(layer(x)).all()


@pytest.mark.parametrize("rank", [1, 2, 4])
@pytest.mark.parametrize("svd", ["exact", "fast"])
def test_pissa_principal_residual_and_gradients(rank, svd):
    weight = torch.diag(torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float64))
    base = nn.Linear(4, 4, bias=False).double()
    base.weight.data.copy_(weight)
    layer = PiSSALinear(base, rank=rank, alpha=7, svd=svd, svd_iterations=6)
    torch.testing.assert_close(layer.effective_weight(), weight, atol=1e-12, rtol=1e-12)
    if svd == "exact":
        expected = weight.clone()
        expected[rank:, rank:] = 0
        torch.testing.assert_close(layer.delta_weight(), expected, atol=1e-12, rtol=1e-12)
    layer(torch.ones(2, 4).double()).square().sum().backward()
    assert layer.A.grad.abs().sum() > 0 and layer.B.grad.abs().sum() > 0


@pytest.mark.parametrize("cls", [LoRALinear, DoRALinear, PiSSALinear])
@pytest.mark.parametrize(
    "dtype,tolerance", [(torch.float32, 2e-6), (torch.float16, 0.01), (torch.bfloat16, 0.06)]
)
def test_supported_dtype_conversion_and_numerical_stability(cls, dtype, tolerance):
    base = nn.Linear(8, 8)
    expected = base(torch.ones(2, 8)).detach()
    layer = cls(copy.deepcopy(base), rank=4).to(dtype=dtype)
    x = torch.ones(2, 8, dtype=dtype)
    result = layer(x)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.float(), expected, atol=tolerance, rtol=tolerance)
    result.float().square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
def test_injection_restoration_and_portable_checkpoint(method, tmp_path):
    base = TinyTransformer(TinyConfig(width=16, heads=2, layers=1))
    pristine, model = copy.deepcopy(base), copy.deepcopy(base)
    x = torch.randint(0, 32, (2, 10))
    options = {"svd": "fast"} if method == "pissa" else {}
    names = inject_adapters(model, method, rank=2, targets="combined", **options)
    assert len(names) == 6
    torch.testing.assert_close(model(x).logits, base(x).logits, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, LoRALinear):
                layer.B.add_(0.1)
    reference = model(x).logits
    save_adapter(model, tmp_path / "adapter.pt", {"revision": "test"})
    # Different randomized SVD factors must still load the saved residual correctly.
    torch.manual_seed(987)
    inject_adapters(pristine, method, rank=2, targets="combined", **options)
    load_adapter(pristine, tmp_path / "adapter.pt", {"revision": "test"})
    torch.testing.assert_close(pristine(x).logits, reference, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError):
        load_adapter(pristine, tmp_path / "adapter.pt", {"revision": "wrong"})
    restore_linear(pristine, keep_adaptation=True)
    assert not any(isinstance(m, LoRALinear) for m in pristine.modules())
    torch.testing.assert_close(pristine(x).logits, reference, atol=2e-6, rtol=2e-6)
    restore_linear(model, keep_adaptation=False)
    torch.testing.assert_close(model(x).logits, base(x).logits, atol=2e-6, rtol=2e-6)


def test_fail_explicitly_on_invalid_config():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(2, 2), rank=3)
    with pytest.raises(ValueError):
        PiSSALinear(nn.Linear(2, 2), rank=1, dropout=0.1)
    model = TinyTransformer()
    with pytest.raises(ValueError):
        inject_adapters(model, targets=["absent"])
    inject_adapters(model)
    with pytest.raises(ValueError):
        inject_adapters(model)
