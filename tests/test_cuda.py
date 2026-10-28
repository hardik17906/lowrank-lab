import copy
import pytest
import torch
from torch import nn
from lowrank_lab.adapters import LoRALinear, DoRALinear, PiSSALinear
from lowrank_lab.models import TinyConfig, TinyTransformer
from lowrank_lab.data.preprocessing import synthetic_splits
from lowrank_lab.training.trainer import Trainer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable in this torch build"
)


@pytest.mark.parametrize("cls", [LoRALinear, DoRALinear, PiSSALinear])
def test_cuda_forward_backward_conversion_and_merge(cls):
    base = nn.Linear(8, 8)
    expected = base(torch.ones(2, 8)).detach()
    layer = cls(copy.deepcopy(base), rank=4).cuda().half()
    x = torch.ones(2, 8, device="cuda", dtype=torch.float16)
    output = layer(x)
    torch.testing.assert_close(output.float().cpu(), expected, atol=0.005, rtol=0.005)
    output.float().square().sum().backward()
    assert layer.B.grad is not None and torch.isfinite(layer.B.grad).all()
    layer.eval().merge()
    torch.testing.assert_close(layer(x), output, atol=0.003, rtol=0.003)
    layer.train()
    assert not layer.merged


def test_cuda_mixed_precision_training_and_memory(tmp_path):
    model = TinyTransformer(TinyConfig(width=16, heads=2, layers=1)).cuda()
    records = synthetic_splits(length=12)
    config = {
        "seed": 7,
        "max_steps": 2,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "learning_rate": 0.003,
        "mixed_precision": "fp16",
        "validation_interval": 2,
    }
    trainer = Trainer(model, records["train"][:8], records["validation"][:4], config, tmp_path)
    result = trainer.fit()
    assert result["completed_steps"] == 2
    assert result["peak_allocated_gpu_bytes"] > 0
