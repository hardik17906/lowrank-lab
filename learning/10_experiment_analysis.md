# Read results without overclaiming

Read guide sections 14, 17 and 18 and `reports/RESEARCH_REPORT.md`.
Every completed run has config, environment, model revision, data manifest,
encoded splits, result and optional history/predictions. A YAML without a
completed result is a prepared experiment, not a measurement.

Begin with `reports/figures/comparison.csv`. Check model, revision, dataset
hash, step count, seed, precision, placement and learning rate before
comparing metrics. Tiny and pretrained results answer different questions.
The generated plots separate their model IDs but can still include multiple
settings, so visual proximity alone is not evidence of a controlled comparison.

Study train/validation curves together. Lower training loss can accompany
overfitting. Runtime curves include interval validation and checkpoint I/O,
while initialization is timed separately. Peak GPU allocation is a PyTorch
allocator measurement; sampled host RSS is a different quantity. Null GPU
measurements are missing data, not zero memory.

Macro F1 and confusion matrices matter because SMS is imbalanced. On 64
test examples, one prediction changes accuracy by 1.5625 percentage points.
A small apparent gain may be fragile. Multiple training seeds estimate one
source of variation; they do not enlarge the underlying held-out sample.
Undefined single-seed SD must not be represented as perfect stability.

Exercise: find two runs sharing all controls except method. State what their
comparison can establish, then list what remains uncertain. Find a dataset-size
comparison and explain how fixed optimizer steps change the number of epochs
seen at each fraction. Propose a larger confirmatory experiment whose test
set is untouched until model and hyperparameters are selected on validation.
