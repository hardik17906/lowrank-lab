"""Principal-component adaptation with a frozen residual, not an additive SVD."""

import torch
from .lora import LoRALinear


class PiSSALinear(LoRALinear):
    """W0=Ur diag(Sr) Vr.T + R; train principal part while freezing R.

    With s=alpha/r, B0=Ur sqrt(Sr/s), A0=sqrt(Sr/s) Vr.T.
    Then s B0 A0 reconstructs the principal component exactly in SVD
    precision. R=W0-s B0 A0. Both factors begin nonzero. Initial factors
    are buffers so adapter-only files can reconstruct R on a clean base.
    """

    def __init__(
        self, base, rank=4, alpha=8.0, dropout=0.0, svd="exact", svd_iterations=4, **kwargs
    ):
        if dropout != 0:
            raise ValueError("PiSSA requires dropout=0 to preserve the initial function")
        super().__init__(base, rank, alpha, dropout, **kwargs)
        if svd not in {"exact", "fast"} or svd_iterations < 0:
            raise ValueError("svd must be exact/fast and iterations nonnegative")
        self.svd, self.svd_iterations = svd, svd_iterations
        with torch.no_grad():
            if rank:
                w = base.weight.detach()
                w = w if w.dtype == torch.float64 else w.float()
                if svd == "exact":
                    u, values, vh = torch.linalg.svd(w, full_matrices=False)
                    u, values, vh = u[:, :rank], values[:rank], vh[:rank]
                else:
                    # Randomized approximation; reconstruction still preserves W0
                    # because its own approximate component is subtracted below.
                    u, values, v = torch.svd_lowrank(w, q=rank, niter=svd_iterations)
                    vh = v.T
                root = (values / self.scaling).sqrt()
                self.B.copy_(u * root.unsqueeze(0))
                self.A.copy_(root.unsqueeze(1) * vh)
            self.register_buffer("initial_A", self.A.detach().clone())
            self.register_buffer("initial_B", self.B.detach().clone())
            base.weight.sub_(self.initial_component())

    def initial_component(self):
        return (self.initial_B @ self.initial_A) * self.scaling

    def original_weight(self):
        return self.base.weight + self.initial_component()

    def get_extra_state(self):
        return {**super().get_extra_state(), "svd": self.svd, "svd_iterations": self.svd_iterations}
