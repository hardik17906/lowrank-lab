# Project code walkthrough

All paths in this document are relative to the repository root. Python code
lives under `src/lowrank_lab/`; the import prefix is `lowrank_lab`. Read this
with `COMPLETE_LEARNING_GUIDE.md`. The full generated callable inventory is
`CODE_REFERENCE.md`; the sections below explain execution and mathematical
decisions rather than merely listing names.

## Dependency map and reading order

```text
cli.main
 ├─ utils.environment.environment_report
 ├─ experiments.grids.generate_grid / run_suite
 ├─ experiments.analysis.analyze
 └─ experiments.runner.read_config / run
     ├─ setup
     │   ├─ utils.reproducibility.seed_everything
     │   ├─ models.tiny_transformer OR models.pretrained.load_pretrained
     │   └─ data.dataset_loader + data.preprocessing
     ├─ adapters.injection.inject_adapters
     │   ├─ LoRALinear
     │   ├─ DoRALinear (LoRA + magnitude normalization)
     │   └─ PiSSALinear (LoRA + principal initialization / frozen residual)
     ├─ training.trainer.Trainer
     │   ├─ training.optimizers
     │   ├─ training.losses.causal_loss_sum
     │   ├─ evaluation.benchmark.evaluate_loss
     │   └─ utils.io / utils.reproducibility
     └─ evaluation + adapters.checkpoint
```

Read in this order: `TinyConfig`, `TinyTransformer.forward`,
`CausalSelfAttention.forward`, `TransformerBlock.forward`, `causal_loss_sum`,
`LoRALinear`, `DoRALinear`, `PiSSALinear`, injection/restoration, data,
evaluation, trainer, runner, CLI, grids and analysis. Studying adapter
injection before ordinary linear layers makes the wrappers seem mysterious.

## 1. Configuration enters the program

`cli.main(argv=None)` takes an optional list of command-line strings and
dispatches a subcommand. It returns no research metric itself; `run`
returns the result dictionary. `python -m lowrank_lab` calls this same
function through `__main__.py`. Use `--help` to see actual supported flags.

`read_config(path)` reads YAML into a dictionary. If `extends` is present,
it recursively reads that file relative to the child file and merges nested
dictionaries. Child scalar/list values replace base values. Thus
`configs/lora.yaml` inherits the CPU pretrained settings, which inherit
smoke defaults. The irrelevant retained `model.tiny` dictionary is ignored
when kind is pretrained; it is not a second model. A resolved config flattens
inheritance and pins the model revision. Breakpoint: inspect the dictionary
just after `read_config`, before any model loading.

`runner.run(config, allow_download=False, resume=None, evaluate_only=False)`
creates one run directory, checks for accidental overwrites, builds the
experiment, trains when requested and evaluates. Inputs are plain dictionaries
and paths; output is a JSON-serializable nested result. On exception it
writes `failure.json` with the actual traceback and re-raises. It never
retries at a smaller sequence length or changes method after an OOM.

## 2. Construct the model and dataset

`setup(config, allow_download)` seeds Python/NumPy/PyTorch, limits CPU
threads, and chooses the tiny or pretrained branch. It returns six values:
model, tokenizer or None, encoded split dictionary, raw SMS splits or None,
model identity dictionary, and dataset provenance dictionary. These are
separate so evaluation can use class labels without changing training input.

`TinyConfig` declares vocabulary size, width, head count, depth, context
limit and dropout. `TinyTransformer.__init__` registers token and position
embeddings, a ModuleList of blocks, final layer norm and an untied output
head. It receives configuration, creates parameters, and returns a model
object. No pretrained knowledge is implied. `load_pretrained` instead loads
actual public safetensors weights and the matching tokenizer. It checks
host RAM/disk before downloads, resolves an allowed download's revision,
sets right padding and disables the generation cache for training. Gradient
checkpointing uses non-reentrant mode so frozen embeddings do not silently
break adapter gradients. Tests instantiate a tiny Llama architecture offline
to validate projection injection, while actual benchmark runs load real weights.

`load_sms(cache_dir, split_seed, allow_download)` returns raw split records
and an audit manifest. It downloads only with explicit CLI authorization,
hashes the archive, parses labels/messages, deduplicates and stratifies.
`normalized_text` standardizes duplicate keys; it does not replace the raw
text sent to the tokenizer. `deduplicate` removes conflicts entirely.
`take_subset` produces a deterministic nested prefix after fixed shuffling.
Its `fraction` applies only to training. Shape here means Python records:
`{'text': str, 'label': 0 or 1}`, not tensors yet.

`prompt_tokens(tokenizer,text,max_length)` returns a list of prompt IDs.
It reserves room for both label candidates and keeps the answer delimiter.
`encode_record` appends the selected candidate's IDs and produces parallel
`input_ids` and `labels` lists; prompt positions have label `-100`.
For a toy prompt `[4,5,6]` and answer `[7,8]`, the encoding is IDs
`[4,5,6,7,8]`, labels `[-100,-100,-100,7,8]`. At logit position two,
the first answer ID 7 is supervised after causal shifting. This is why
masking prompt labels must not mask the final prompt logit.

`remove_token_collisions` compares visible prompt ID tuples, excluding
answers, after truncation. It returns cleaned splits plus per-split removal
counts. Test examples get priority over validation and training. It prevents
identical local model inputs leaking across splits but cannot identify all
semantic near duplicates. `split_fingerprint` hashes deterministic JSON of
encoded examples; identities include both combined and individual split hashes.

`synthetic_splits(seed,length)` returns disjoint repeated-pair token examples
for offline teaching. IDs 2–31 denote pattern symbols, ID 1 starts a sequence.
Train/validation/test use different ordered pairs. It has no text tokenizer.
`collate(records,pad_token_id)` pads ID/label/mask tensors to the longest
sequence in that microbatch. Returned tensors are `[B,T]`, integer-valued;
only attention masks use zeros for padding and only labels use -100.
Tests: `test_data_and_integration.py` covers conflicts, nested subsets,
truncation collisions, padding, split disjointness and a real Transformers
architecture. Breakpoint: inspect the three tensors returned by collate.

## 3. Tiny-model forward pass, shape by shape

For the default smoke batch, IDs are `[8,24]`. Embeddings produce `[8,24,48]`.
Position vectors are `[24,48]` and broadcast over the batch. Each block keeps
the outer shape unchanged. `CausalSelfAttention.__init__` creates four
width-to-width linear maps and checks divisibility by head count.

`CausalSelfAttention.forward(x,attention_mask)` takes `[B,T,D]`, applies
q/k/v maps and reshapes each result to `[B,H,T,D/H]`. At defaults these are
`[8,4,24,12]`. Dot-product scores are `[8,4,24,24]`; triangular and padding
masks forbid invalid key positions. Softmax runs over the final key axis.
Value aggregation returns `[8,4,24,12]`. Transpose/reshape concatenates
heads into `[8,24,48]`; output projection retains that shape. It returns
hidden values, not probabilities or language logits. The attention test
changes IDs at future positions and requires earlier logits to stay identical.

`TransformerBlock.forward` first normalizes features, applies attention and
adds the residual. It normalizes again, maps to `[B,T,4D]`, applies GELU,
maps back and adds a second residual. Norms are pre-normalization; this is
one deliberate tiny architecture, not an imitation of every pretrained model.
Breakpoints: before masking scores, after softmax, and after each residual.

`TinyTransformer.forward(input_ids,attention_mask=None,labels=None)` loops
through blocks, applies final norm and projects to `[B,T,V]`. With default
V=32, logits are `[8,24,32]`. It returns a namespace with `.logits` and
optional `.loss`; the trainer deliberately uses its own common summed loss.
When provided labels, the model's educational loss shifts once and computes
mean CE. Passing sequences beyond the learned context limit raises an error.

`TinyTransformer.generate` takes `[B,T]` IDs and returns `[B,T+new_tokens]`.
It uses eval/no-grad, repeatedly evaluates the context, chooses an ID and
appends it, then restores the previous training mode. Temperature zero is
greedy. This explicit loop makes autoregression visible; it does not cache
keys/values. Tests cover generation length and causal independence.

## 4. LoRA forward and lifecycle

`LoRALinear(base,rank,alpha,dropout,init)` takes a real `nn.Linear`, creates
A `[r,in]`, B `[out,r]`, freezes base and bias, validates options and
registers a nonpersistent merge cache. Base must be unquantized and rank
cannot exceed either matrix dimension. Defaults follow loralib Linear's
Kaiming A/zero B convention; Gaussian is an explicit alternative.

`delta_weight()` returns `[out,in]` scaled B@A with an intact graph.
`effective_weight()` adds it to frozen storage; `original_weight()` returns
the base for LoRA. These dense inspection helpers are used for merging,
testing and restoration. `forward(x)` accepts any leading axes ending in
`in`, returns the same leading axes ending in `out`, and computes the
adapter via two factorized linear operations. For x `[2,5,7]`, A `[3,7]`,
B `[4,3]`, the intermediate is `[2,5,3]` and update `[2,5,4]`.

`merge()` requires eval mode and caches one detached effective weight.
It returns self and is idempotent. `unmerge()` discards the cache and returns
self. `train(True)` unmerges before enabling training. Base storage remains
unchanged by merge/unmerge, avoiding repeated rounding. `enabled=False`
uses original weights even if a merge cache exists. `get_extra_state` and
`set_extra_state` serialize/validate scaling/method metadata in full state
dicts; loading incompatible rank/alpha is rejected rather than silently wrong.

Tests in `test_adapters.py` cover dimensions, zero update, frozen base,
two-step gradients, disabled behavior, dtype conversion, initial function,
state dicts and repeated merge/unmerge. Breakpoint: watch `B.grad` on step
one and `A.grad` on step two. Inspect base `.grad`, which should be None.

## 5. DoRA forward and norm gradients

`DoRALinear.__init__` adds one magnitude parameter per output row, initialized
to the base row norm. It accepts `detach_norm`; nonzero dropout is rejected.
`_row_norm(weight)` maps `[out,in]` to `[out]`, summing norms in float32 for
half types and preserving float64 for precise tests. `effective_weight`
constructs direction, clamps its norm, optionally detaches only the
denominator and multiplies by magnitude/norm. `forward` applies a dense
linear map and adds frozen bias afterward. Input/output shapes match LoRA.

Example: base rows `[3,4,0]` and `[0,0,2]` yield magnitudes `[5,2]`, not
the three input-channel norms. `detach_norm=False` propagates the derivative
through norm, rotating the gradient away from the current direction;
True omits that path as the original paper's efficiency variant does.
Dense effective-weight materialization makes the formula inspectable but
may need more training memory than optimized factorized implementations.

The independent row-by-row reference checks both variants' outputs and
factor/magnitude gradients with bias. PEFT checks the detached variant
separately when installed. Breakpoint: inspect `direction.shape`,
`norm.shape`, `magnitude.shape` and `norm.requires_grad`.

## 6. PiSSA initialization and restoration

`PiSSALinear.__init__` computes SVD in float32 (float64 retained for tests),
selects principal components, divides singular values by scale before square
rooting, fills A/B and saves initial factors as buffers. It subtracts the
initialized component from frozen base storage. That storage now means
residual, not original W0. Exact mode uses economy SVD; fast mode uses
`torch.svd_lowrank` with configured iteration count. Its random subspace
changes initialization, but its residual correction still preserves W0.

`initial_component()` returns scaled initial_B@initial_A `[out,in]`.
`original_weight()` adds that to residual; inherited `effective_weight`
adds the current trained factors to residual. Disabling adaptation must use
the first, not residual alone. Standard factorized forward is inherited.
Both trainable factors start nonzero. Do not interpret a nonzero adapter
product as a nonzero initial change to the complete model.

Tests use a diagonal matrix with known singular values, several ranks,
exact/fast reconstruction and nonzero gradients. State dicts include
initial buffers; portable adapter loading is tested with different randomized
initialization seeds. Breakpoint: compare residual + initial component to a
clone of the weight immediately before initialization.

## 7. Injection and ordinary-layer export

`projection_names(model,targets)` scans named modules and returns nn.Linear
paths matching explicit role suffixes. It returns strings, not layers.
Llama/Qwen grouped-query output widths are read from actual matrices.
`inject_adapters` rejects an already-adapted model, constructs replacement
wrappers on copies, freezes everything else, and installs each child on its
parent. It returns replaced paths, saved in `target_modules.json`.
Constructing replacements before installation avoids half-injected models
when validation fails, though temporary copies cost host memory.

`restore_linear(model,keep_adaptation)` copies either the effective adapted
weight or reconstructed original weight into ordinary linear layers and
replaces wrappers. It returns the same model object with frozen linears.
`parameter_counts` returns total/trainable counts and their ratio; it excludes
buffers and includes DoRA's magnitude. `merge_adapters`/`unmerge_adapters`
apply the cache lifecycle across all wrappers. These are inference helpers,
not optimizer steps. Test restoration before and after actual factor changes.

## 8. Forward loss, backward and optimizer step

`causal_loss_sum(logits,labels)` returns `(sum_loss,valid_count)`. Inputs
are `[B,T,V]` and `[B,T]`. It shifts once, flattens token positions, computes
float32 CE with reduction=sum and counts nonignored targets. It raises if
the batch has no valid target. Example IDs `[1,2,3]`, labels `[-100,-100,3]`
supervise only logit position one predicting 3. It never supervises the
last logit because no next token is present.

`make_optimizer(model,config)` selects AdamW or SGD on trainable parameters.
`make_scheduler` returns a LambdaLR with constant/linear/cosine multiplier
and optional warmup. Its counter advances once per optimizer update.
These helpers isolate standard utilities from the custom mathematical layers.

`Trainer.__init__` stores datasets, verifies nonempty splits and settings,
constructs optimizer/scheduler/scaler and initializes counters. It requires
the model already be on its target device. `autocast()` returns the selected
context manager or a no-op context. Unsupported CPU FP16 or CUDA BF16 fails
explicitly. `_batches()` builds the current epoch's deterministic shuffled
index lists with a private seed+epoch generator, separate from dropout RNG.

`fit(stop_after=None)` is the central loop. It takes the next accumulation
window, collates microbatches and counts every supervised token. Each
microbatch forward contributes summed CE divided by the window token count.
Backward adds gradients; it does not update weights. After all microbatches,
the loop unscales, checks/clips gradients, steps the optimizer/scaler,
advances scheduler and cursor, updates completed-epoch summaries and logs
Python scalar metrics. Validation runs at configured intervals. On a better
validation score it stores trainable parameter tensors in `best.pt`.
Matched-step defaults use final weights and disable early stopping. Optional
best selection is explicit and uses validation only.

Breakpoints: after normalized loss, after backward before clipping, after
optimizer step and before checkpoint save. Compare one factor value before
and after the optimizer step. `test_gradient_accumulation_unequal_target_counts`
and `test_tiny_training_loss_reduction` verify learning and normalization.

## 9. Checkpoints and exact resume

`save_checkpoint(path)` writes full model state, optimizer, scheduler,
scaler, RNG, epoch, cursor, metrics and identity at an optimizer boundary.
`load_checkpoint(path)` validates configuration/model/data identity, restores
all state and restores RNG last. Local resume files include Python RNG
objects and are trusted-local-only, whereas adapter files use weights-only
loading. Carry the original run directory, including `best.pt`, if selecting
best weights after resuming.

`seed_everything` sets seeds and deterministic flags. `rng_state` returns
Python/NumPy/CPU/CUDA states; `restore_rng` restores them. Neither function
proves cross-version determinism. `save_torch` and `write_json` write a
temporary sibling then atomically replace the destination. JSON refuses NaN
and infinity so invalid metrics cannot silently pollute analysis.

`save_adapter(model,path,base_identity)` stores trained factor state,
PiSSA initial factors, DoRA magnitudes and per-layer metadata without dense
base weights. `load_adapter` checks identity/targets/configuration and
installs factors. For PiSSA it reconstructs the clean original weight using
the constructor's initial factors, then subtracts the saved initial component.
This avoids sign/RNG dependency of a second SVD. The caller must supply the
same genuine base identity; identity strings are not a cryptographic proof
of arbitrary caller-provided weights. Tests reload onto a fresh identical base
and compare outputs after factor updates.

## 10. Evaluation flow

`evaluate_loss(model,records,batch_size,pad_token_id)` switches to eval mode,
uses no-grad and sums losses/tokens across batches. It returns CE, perplexity
and token count, then restores the previous training mode. It weights by
tokens rather than averaging differently sized batch means. `evaluate_classification`
constructs both answer candidates for each raw SMS, forwards the candidates,
gathers answer-token log probabilities, sums and chooses the higher score.
For B messages it evaluates 2B candidates. It returns accuracy, macro F1,
matrix, support, labels, predictions and raw candidate scores. No generation
heuristic or answer extraction is needed. `classification_metrics` defines
true-row/predicted-column confusion matrices and explicit zero-division behavior.

Runner evaluates validation and held-out test after training, writes predictions
and `result.json`, and saves adapter/model files. `evaluate` reloads the saved
final weights with the same configuration and writes `evaluation.json` without
replacing the original result. This is a useful end-to-end checkpoint audit.

## 11. Experiment grids, metrics and figures

`generate_grid(base_path,output_dir,stage)` expands one named study, writes
flattened YAMLs and a manifest, and returns file paths. It never trains by
itself. `run_suite(manifest,allow_download,limit)` executes sequentially in a
fresh process per run, skips completed runs and stops on failures. Process
isolation prevents earlier optimizers/models contaminating peak memory.
A limit lets the owner stage a
large manifest after checking one run's memory and timing.

`analyze(root,output)` reads completed results and returns a pandas frame.
It writes comparison/seed CSVs, a Markdown table and actual curves. It
separates random tiny and pretrained model panels, records model/data hashes,
and groups seed summaries only under matching controls. Empty GPU measurements
are shown as unmeasured. Rank/size scatter plots are exploratory when other
controls vary; inspect config hashes before interpreting a line as causal.
No test metric is invented for a missing run.

`environment_report()` records OS, Python, installed packages, PyTorch CUDA
build/availability, torch GPU properties, nvidia-smi, host RAM and free disk.
Driver discovery and a CUDA-enabled torch build are separate checks. A driver
can exist while torch reports CPU-only. The report records both rather than
concealing a resource limitation.

## Debugging one concrete example

Start with `tests/test_adapters.py::test_dora_reference_forward_gradient_axes_and_bias`.
Set a breakpoint just before `effective_weight`. Inspect base `[2,3]`,
A `[2,3]`, B `[2,2]` and magnitude `[2]`. Compute row norms `[5,2]` by
hand at initialization. Next step through the row reference and compare bias
addition. Repeat with norm detach disabled and explain the changed derivative.
This small case is more instructive than looking at thousands of model modules.

Then open `tests/test_training.py::test_checkpoint_exact_resume_with_dropout_and_epoch_transition`.
At load time inspect epoch/cursor and RNG restoration. The exact equality
assertion checks final parameters and loss history, not just successful file
loading. It is a concrete example of reproducibility as behavior.
