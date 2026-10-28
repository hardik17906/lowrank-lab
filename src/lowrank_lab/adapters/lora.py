"""Manual low-rank adaptation in PyTorch's (out_features, in_features) layout."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """y = x W0.T + (alpha/r) (dropout(x) A.T) B.T + bias.

    A: [r,in], B: [out,r]. Base weights and bias are frozen. r=0 is a
    frozen identity wrapper. Kaiming-uniform A matches microsoft/loralib's
    Linear implementation; Gaussian A is an explicit initialization ablation.
    Merge caches an inference weight without mutating the frozen parameter.
    """

    def __init__(self, base: nn.Linear, rank=4, alpha=8.0, dropout=0.0, init="kaiming", **kwargs):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Only unquantized nn.Linear is supported")
        if not 0 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("rank must lie in [0, min(in,out)]")
        if alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("alpha must be positive; dropout must lie in [0,1)")
        if kwargs:
            raise TypeError(f"Unexpected adapter options: {kwargs}")
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        self.scaling = alpha / rank if rank else 0.0
        self.dropout = nn.Dropout(dropout)
        self.init = init
        self.enabled = True
        self.register_buffer("_merged_weight", None, persistent=False)
        self.A = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.B = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        if rank:
            if init == "kaiming":
                nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            elif init == "gaussian":
                nn.init.normal_(self.A, std=1 / rank)
            else:
                raise ValueError(f"Unknown initialization: {init}")
        else:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)

    @property
    def weight(self):
        """Frozen storage weight (residual for PiSSA), for inspection only."""
        return self.base.weight

    @property
    def merged(self):
        return self._merged_weight is not None

    def delta_weight(self):
        """Return scaled update [out,in], retaining the adapter gradient graph."""
        return (self.B @ self.A) * self.scaling

    def original_weight(self):
        return self.base.weight

    def effective_weight(self):
        return self.base.weight + self.delta_weight()

    def forward(self, x):
        if not self.enabled:
            return F.linear(x, self.original_weight(), self.base.bias)
        if self.merged:
            return F.linear(x, self._merged_weight, self.base.bias)
        result = self.base(x)
        if self.rank:
            update = F.linear(F.linear(self.dropout(x).to(self.A.dtype), self.A), self.B)
            result = result + (update * self.scaling).to(result.dtype)
        return result

    @torch.no_grad()
    def merge(self):
        if self.training:
            raise RuntimeError("Call eval() before merging for inference")
        if not self.merged:
            self._merged_weight = self.effective_weight().detach().clone()
        return self

    def unmerge(self):
        self._merged_weight = None
        return self

    def train(self, mode=True):
        if mode:
            self.unmerge()
        return super().train(mode)

    def get_extra_state(self):
        # Metadata makes full state dictionaries reject incompatible scaling.
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "method": type(self).__name__,
            "dropout": self.dropout.p,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError(f"Adapter configuration mismatch: {state}")
        self.unmerge()
