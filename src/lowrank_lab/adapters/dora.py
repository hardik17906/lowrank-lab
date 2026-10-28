"""DoRA, including the paper's detached-denominator training variant."""

import torch
from torch import nn
from torch.nn import functional as F
from .lora import LoRALinear


class DoRALinear(LoRALinear):
    """W' = diag(m / ||W0 + s BA||row) (W0 + s BA).

    The paper's column norms use input-by-output weights. nn.Linear stores
    the transpose, hence dim=1 here. Bias is added AFTER magnitude scaling.
    detach_norm=True follows the paper section 4.3 memory-saving convention.
    False retains the full derivative of the normalization (section 4.2).
    Dropout is intentionally rejected to keep this exact dense formulation
    unambiguous across the two gradient variants.
    """

    def __init__(self, base, rank=4, alpha=8.0, dropout=0.0, detach_norm=True, **kwargs):
        if dropout != 0:
            raise ValueError("This DoRA implementation requires dropout=0")
        super().__init__(base, rank, alpha, dropout, **kwargs)
        self.detach_norm = bool(detach_norm)
        self.magnitude = nn.Parameter(self._row_norm(base.weight).to(base.weight.dtype))
        if rank == 0:
            self.magnitude.requires_grad_(False)

    @staticmethod
    def _row_norm(weight):
        # Accumulate norms in fp32 for half/bfloat16. Preserve float64 tests.
        norm_weight = weight if weight.dtype == torch.float64 else weight.float()
        return torch.linalg.vector_norm(norm_weight, dim=1)

    def effective_weight(self):
        direction = self.base.weight + self.delta_weight()
        norm = self._row_norm(direction).clamp_min(1e-12)
        if self.detach_norm:
            norm = norm.detach()
        ratio = self.magnitude.to(norm.dtype) / norm
        return direction * ratio.to(direction.dtype).unsqueeze(1)

    def forward(self, x):
        if not self.enabled or self.merged or self.rank == 0:
            return super().forward(x)
        return F.linear(x, self.effective_weight(), self.base.bias)

    def get_extra_state(self):
        return {**super().get_extra_state(), "detach_norm": self.detach_norm}
