# Full fine-tuning as the unrestricted baseline

Read guide sections 9, 16 and 18. In full fine-tuning all parameters remain
trainable: embeddings, attention, FFN, norms and output head. The code does
not replace the model with adapters in this branch. AdamW therefore needs
moments and gradients for every parameter, increasing memory substantially.

The no-fine-tuning baseline freezes everything and skips optimizer creation.
It still evaluates the same prompt and labels. These two controls bracket
the PEFT comparison: does adaptation help, and what can unrestricted updates
achieve under this step budget? Neither control guarantees an upper or lower
bound on held-out performance; optimization and overfitting can reverse them.

Inspect `runner.run` around the method dispatch. There is one shared setup,
loss and evaluation path. This avoids a common confound where adapter runs
receive a different task formulation. Full training has more degrees of
freedom, but hyperparameters can suit methods differently. Shared learning
rate is a controlled first comparison, not proof of equally tuned methods.

The report's runtime includes checkpoint saving. Full checkpoints contain
many more tensors than adapter-only exports, and this affects deployment
storage as well as measured runtime. Training resume checkpoints still store
the full state even in adapter runs, so resume storage and adapter export
storage are explicitly different metrics.

Exercise: use the static memory estimate before enabling the GPU. Add room
for activations and temporary operations; do not rely only on checkpoint
file size. Compare exact parameter counts in actual results, and explain
why the adapter trainable fraction does not predict the full memory ratio.
