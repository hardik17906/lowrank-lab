# PiSSA paper reading companion

Read [PiSSA section 3 and the fast-SVD appendix](https://arxiv.org/html/2404.02948v4),
guide section 12, and `adapters/pissa.py`. Start with a diagonal matrix so
its singular vectors are ordinary axes. The largest singular value identifies
the strongest direction of that pretrained map, not automatically the most
useful direction for the downstream task.

PiSSA changes which part of W0 is trainable. The principal component moves
into the factors, and the residual remains frozen. Their sum initially equals
the original function. This differs from putting an SVD-shaped update on
top of unchanged W0, which would double-count the component.

Scaling belongs inside the square-root factor initialization as well as in
the forward path. Otherwise alpha/r would rescale the principal component
at initialization. Check the guide's scale-two example and the diagonal
reconstruction test. Both factors are nonzero initially and can receive
gradients on the first backward.

Fast SVD approximates the leading subspace, but the residual subtracts the
actual initialized component. It therefore still reconstructs W0 up to
rounding. Its factors can differ across random seeds, signs, implementations
and devices. Saving initial factors makes portable residual reconstruction
possible on a clean base without assuming a second SVD is identical.

Exercise: compare adapter-only LoRA and PiSSA file sizes. Explain why PiSSA
saves more than its current trainable factors. Next inspect `load_adapter`:
why must it undo the fresh initialization before applying saved initial
factors? Finally explain why zero dropout matters for preserving the initial
training forward, not just the evaluation function.
