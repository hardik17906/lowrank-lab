# LoRA paper reading companion

Read [the original methodology](https://arxiv.org/html/2106.09685v2) and
guide section 10, then `adapters/lora.py`. The source audit in
`reports/SOURCES.md` distinguishes the paper from the reference library.
The project's worked matrices are independent teaching examples.

Focus on the coordinate system: PyTorch weights are output-by-input, A is
rank-by-input and B is output-by-rank. On a real batch, the adapter first
compresses features to rank coordinates, then maps those coordinates to
outputs. It restricts the update matrix, while leaving the full base matrix
available in the forward computation.

Initial zero B means exactly zero update, even though A is random. This
preserves the starting pretrained function. Initializing both factors with
zeros would leave both derivatives zero, preventing useful learning. The
reference Linear implementation's Kaiming initialization is the default;
Gaussian is available as an explicitly labelled initialization study.

Inspect `test_merge_unmerge_idempotence_state_dict_and_disabled`. Merging
is an inference lifecycle operation, not training. This project caches the
merged weight while leaving frozen storage alone; ordinary-layer export can
make the merged result permanent. Explain why repeatedly adding the same
update to base storage without a guard would corrupt inference.

Exercise: calculate adapter parameters for one rectangular FFN matrix at
ranks four and eight. Compare with an attention projection of different
dimensions. Does equal rank mean equal parameter budget? Then inspect the
fixed-alpha rank grid and explain how scale changes with rank. These are
reasons a rank ablation needs precise configuration reporting.
