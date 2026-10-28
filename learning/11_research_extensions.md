# Design the next research question

The optional hypothesis is that uniform rank may allocate parameters poorly
across layers. Spectra provide a transparent starting signal, but pretrained
weight energy is not task importance. A sensitive classifier can depend on
a direction with low singular energy, while a large-energy direction may
already work well without adaptation.

Establish baseline behavior before implementing an allocation algorithm.
Count the actual factor cost of rank one in every matrix: input width plus
output width, with DoRA magnitude overhead accounted for separately. A rank
assignment that spends fewer or more parameters is not a fixed-budget test.
Integer ranks and unequal matrix widths can make exact budgets difficult.

A useful protocol: decide a spectral score using only pretrained weights,
define an integer allocation procedure and its tie-breaking rule, enforce
the budget exactly or report the unused remainder, then compare with uniform
LoRA/PiSSA using identical seeds, examples and steps. Use validation for
choices. Report if the allocation performs worse or if its benefit is within
seed variability. Do not search many allocation rules on test scores.

Other natural extensions are a second public task, larger held-out sets,
longer training, several learning rates per method, separately matched time
budgets, full-norm DoRA, exact versus fast SVD costs and repeated checkpoint
transfer across devices. Quantization and fused-QKV support require their
own correctness tests and are not consequences of ordinary linear support.

Exercise: write a one-page preregistration containing the hypothesis, controls,
metric, selection procedure, stopping rule and failure criteria. State one
negative result that would change your belief. The optional adaptive-rank
algorithm is not implemented in the baseline laboratory; it remains a proposed
extension rather than an untested feature labelled complete.
