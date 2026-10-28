# DoRA paper reading companion

Read [DoRA sections 4.1–4.3](https://arxiv.org/html/2402.09353v2), guide
section 11, and `adapters/dora.py`. Separate three ideas: the forward
normalization, the gradient through normalization, and the memory-saving
detached denominator. They are related but not interchangeable.

Work the `[3,4]` row example. Its magnitude is five. Dividing by five gives
a unit direction; multiplying by a learned magnitude reconstructs or rescales
the row. The matrix convention matters: the paper's columns correspond to
our PyTorch output rows. The bias is not part of the direction vector and
must be added after the magnitude scaling.

The default detached denominator has the same numerical forward as full
differentiation but a different backward rule. This is intentional and named
in configs. A gradient test must match that choice. Finite differences of
the ordinary normalized expression validate full differentiation, while
the detached variant needs a reference graph with the same detach.

The independent scalar-row test checks A, B and magnitude derivatives,
normalization axes and bias. The optional PEFT test checks the default
variant against a separate implementation. Numerical agreement on small
matrices supports implementation correctness; it does not reproduce a paper's
downstream scores or optimized memory results.

Exercise: double magnitude while holding direction fixed and inspect the
effective weight and output bias. Then set `detach_norm=False` and compare
gradient directions. Explain why dense effective-weight materialization can
consume more memory than factorized LoRA despite similar trainable counts.
