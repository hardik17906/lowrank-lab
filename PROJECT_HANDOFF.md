# PROJECT_HANDOFF — tutor and owner entry point

Completed on 2026-10-28, Asia/Calcutta. Paths are relative to the repository root.
Read this document first, then `learning/COMPLETE_LEARNING_GUIDE.md` and
`learning/PROJECT_CODE_WALKTHROUGH.md`. Do not infer large-model reproduction
from successful smoke tests. The project owner knows basic machine learning;
teach tensors/autograd/attention before asking them to modify experiments.

## A. What the completed project does

Independent manual PyTorch LoRA, DoRA and PiSSA; tiny educational causal
Transformer; genuine public pretrained integration; SMS completion-based
classification; full/no-update controls; deterministic data preparation;
token-normalized training; resume/adapter exports; GPU/CPU metrics; staged
ablations; actual charts/tables; numerical and functional tests. The core never
uses PEFT; PEFT is an optional independent reference dependency for tests.

The completed study is an affordable pilot, not a full original-paper
reproduction. Prepared 0.5B/1.5B settings require a suitable machine. No
quantization, distributed training, generic Conv1D/fused-QKV adaptation or
task-general accuracy claim is made.

## B. Repository architecture

| Path | Actual purpose |
|---|---|
| README.md | Installation entry points, project scope and documentation links |
| pyproject.toml / requirements.txt / requirements-lock.txt | Installable package, broad dependencies, final execution lock |
| configs/smoke.yaml | Offline random tiny demonstration |
| configs/pretrained_cpu.yaml / pretrained_gpu.yaml | Pinned 135M pilot |
| configs/baseline.yaml / full_finetune.yaml / lora.yaml / dora.yaml / pissa.yaml | Individual CPU method commands |
| configs/limited_gpu.yaml / larger_gpu.yaml | Prepared 0.5B and 1.5B tiers |
| configs/experiments | Concrete all-stage YAMLs and manifests; only selected subset executed |
| src/lowrank_lab/models/tiny_transformer.py | Embeddings, explicit causal attention, residual FFN blocks, logits, CE, generation |
| src/lowrank_lab/models/pretrained.py | Resource checks, pinned safetensors/tokenizer, local snapshot loading |
| src/lowrank_lab/adapters/lora.py | Frozen base, low-rank factors, scaling, cached merge lifecycle |
| src/lowrank_lab/adapters/dora.py | Output-row magnitude/direction, detached or full norm gradients |
| src/lowrank_lab/adapters/pissa.py | Exact/fast SVD, scale correction, frozen residual, initial-factor buffers |
| src/lowrank_lab/adapters/injection.py / merging.py | Projection roles, replacement, counts, restoration and model-wide merging |
| src/lowrank_lab/adapters/checkpoint.py | Adapter-only save/load and portable PiSSA residual reconstruction |
| src/lowrank_lab/data/dataset_loader.py | Verified public archive, normalization/dedup, fixed stratified splits |
| src/lowrank_lab/data/preprocessing.py | Prompts, completion labels, truncation collision removal, padding, synthetic data, hashes |
| src/lowrank_lab/training/trainer.py | Explicit accumulation, clipping, AMP, validation, checkpoints, deterministic resume |
| src/lowrank_lab/training/losses.py / optimizers.py | Common causal token objective and standard optimizer/scheduler selection |
| src/lowrank_lab/evaluation | Token-weighted CE/PPL and constrained classification metrics |
| src/lowrank_lab/experiments | Runner, staged grid generation, suite execution and measured-result analysis |
| src/lowrank_lab/utils | Atomic I/O, seeding/RNG and hardware/environment reporting |
| tests | Mathematical, gradient, dtype, causal, checkpoint, CUDA, PEFT and workflow checks |
| scripts | Selected study, tiny continuation demo, code inventory and report generation |
| notebooks/01_inspect_adapters.ipynb | Runnable teaching cells; no fabricated saved outputs |
| results | Actual configs, manifests, encoded splits, histories, predictions, checkpoints and result JSONs |
| benchmarks/sms-pilot-2026-10-28 | Published text snapshot of all 32 recorded runs, hashes, hardware and validation metadata |
| reports | Research/reproduction/resource/source/test/event audits and actual plots/CSVs |
| learning | Complete first-principles guide, chapters 00–11, execution walkthrough and exact callable inventory |
| benchmarks/sms-pilot-2026-10-28/manifest.json | Published artifact index and checksums |

## C. Mathematical implementation map

- LoRA factor construction: `src/lowrank_lab/adapters/lora.py:9` (`LoRALinear`); update `src/lowrank_lab/adapters/lora.py:58` (`delta_weight`).
- DoRA row norms: `src/lowrank_lab/adapters/dora.py:30` (`_row_norm`); normalized effective weight `src/lowrank_lab/adapters/dora.py:35` (`effective_weight`).
- PiSSA scale-correct SVD/residual initialization: `src/lowrank_lab/adapters/pissa.py:7` (`PiSSALinear`); original component `src/lowrank_lab/adapters/pissa.py:44` (`initial_component`).
- Token-summed causal CE: `src/lowrank_lab/training/losses.py:4` (`causal_loss_sum`); accumulation and update `src/lowrank_lab/training/trainer.py:132` (`fit`).
- Conditional candidate scoring: `src/lowrank_lab/evaluation/benchmark.py:36` (`evaluate_classification`).

Full derivations, shapes, symbol definitions, worked examples, failure modes
and tests are in the learning guide. Full state dictionaries include adapter
metadata; inference caches are nonpersistent. Adapters freeze base bias as well
as weight. Magnitudes are output-channel scalars, not input-channel scalars.

## D. Execution status

- Recorded pytest execution: 60 executed, 0 failures, 0 errors, 0 skipped; 10.56 seconds. Initial CPU-only suite: 51 passed;
  independent reference checks: three passed; expanded CUDA suite executed.
- 32 completed saved runs under results, including 20
  selected genuine pretrained GPU runs plus 5 fresh-process repeats,
  one CPU pretrained baseline, five
  tiny baselines and the separate smoke demonstration.
- Real tiny greedy continuations saved, including imperfect outputs.
- Final dependency check passed; hardware/package reports and lock saved.
- All four original pretrained model/adapter exports reloaded with matching
  validation/test metrics; see `reports/checkpoint_roundtrip.json`.
- `reports/AUDIT.json` records the final artifact checks: fingerprints, visible
  split disjointness, recomputed metrics, budgets, CLI parsers, links and notebook.
- Primary efficiency comparison uses five isolated processes, with original
  validation/test metrics exactly reproduced. Original shared-process peaks
  are retained but excluded from the primary memory/runtime figures.
- See `reports/RESOURCE_PLAN.md` for hardware constraints and the shared-process
  measurement correction. Machine-specific execution logs remain local.
- The failed full-GPU preflight JSON is retained alongside its later successful
  result locally; it was not counted as a successful measurement or silently retried
  with different training settings.
- Exact selected configs are in `experiments/executed_manifest.json`.
  Published metrics are under `benchmarks/sms-pilot-2026-10-28/runs`; runtime
  weights and outputs are intentionally excluded from version control.
  Unselected grid entries, 0.5B/1.5B training, OOD and adaptive rank were not run.

## E. Real experiment summary

Pinned model `HuggingFaceTB/SmolLM2-135M` at
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. UCI SMS, archive
`1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3`. 5574 raw rows;
415 duplicate/conflicting rows removed;
fixed 70/15/15 splits followed by tokenized prompt collision filtering.
Pilot uses {'test': 64, 'validation': 32, 'train': 128}; test support 56 legitimate / 8 spam.
Seed-seven baseline settings: rank four, alpha eight, attention-only adapters;
12 matched AdamW steps, microbatch two × accumulation two, LR 0.0002,
FP32, context 64, final weights and no early stopping.

| Method | Test CE | Accuracy | Macro F1 | Trainable | Training s | Peak GPU MiB | Adapter KiB |
|---|---|---|---|---|---|---|---|
| baseline | 9.26662 | 0.12500 | 0.11111 | 0 | 0.00000 | unmeasured | n/a |
| full | 0.36963 | 0.82812 | 0.58466 | 134515008 | 56.41940 | 2670.70000 | n/a |
| lora | 7.61692 | 0.12500 | 0.11111 | 460800 | 32.27221 | 799.22000 | 1876.11000 |
| dora | 7.37432 | 0.12500 | 0.11111 | 506880 | 17.99648 | 1003.45000 | 2087.91000 |
| pissa | 0.99723 | 0.87500 | 0.46667 | 460800 | 17.32419 | 800.97000 | 3741.15000 |


- lora: seeds [7, 17, 27], target CE mean 7.56600, sample SD 0.05377; macro F1 mean 0.11111, sample SD 0.00000.
- pissa: seeds [7, 17, 27], target CE mean 0.99041, sample SD 0.04268; macro F1 mean 0.46667, sample SD 0.00000.

The no-update model predicts every example as spam. Majority legitimate alone
would achieve 87.5% accuracy; use macro F1 and confusion matrices to judge
minority behavior. This is not a method-specific tuned or converged benchmark.
The full real tables/plots and ablation interpretation are in the research
report and `reports/figures`. Source author claims remain separate.

The Gaussian-A LoRA ablation reaches test CE 0.98868 and macro F1 0.46667, while still predicting all messages as legitimate. Its A standard deviation is 1/r (0.25 at rank four), versus approximately 0.0241 for Kaiming at input width 576. Thus this ablation changes factor scale as well as distribution. The large early-loss difference between default LoRA and PiSSA cannot isolate the value of a principal subspace; initialization scale and optimization dynamics are material confounds. The Gaussian-vs-PiSSA seed-seven CE gap is smaller than PiSSA's observed three-seed CE SD (0.04268); Gaussian LoRA itself was not replicated across seeds, so no stable quality ranking follows.

## F. Learning prerequisites

Start from supervised learning, train/validation/test splits and ordinary
algebra. Teach dot products and matrix shapes, derivatives/chain rule,
gradient descent and Adam moments, tensors/modules/parameters/buffers,
computation graphs, embeddings, Q/K/V, softmax and masking, residuals/norms,
next-token shifting and ignored labels, rank/SVD, low-rank gradients,
serialization, resource accounting and experimental uncertainty. Do not assume
the owner already understands any of the deep-learning terminology.

## G. Recommended teaching order

1. Read chapter 00 and identify the separate tiny/pretrained tracks.
2. Chapters 01–03: affine operations, nonlinear networks, autograd and optimizers.
3. Chapter 04 plus TinyConfig/TinyTransformer/CausalSelfAttention: trace real shapes.
4. Chapter 05 and causal_loss_sum: explain shifted answer-only targets and padding.
5. Chapter 06 and the full baseline: connect forward/backward/optimizer step.
6. Chapter 07 and LoRALinear: inspect zero B, initial zero A gradient, later flow.
7. Chapter 08 and DoRALinear: work the [3,4] norm example and two backward variants.
8. Chapter 09 and PiSSALinear: diagonal SVD example, scale correction, residual.
9. Injection/restoration/checkpoint code: trace names, buffers and portable state.
10. Data and evaluation: duplicate policy, fixed splits, candidate log scores, F1.
11. Trainer and exact-resume tests: unequal target counts, RNG/cursor/Adam state.
12. Chapters 10–11 and the report: interpret controlled comparisons and uncertainty,
    then design a preregistered extension.

The walkthrough provides function inputs/outputs, important variables, shapes,
step-by-step flow, rationale, tests and suggested debugger breakpoints.
CODE_REFERENCE.md is generated from actual source to avoid invented names.

## H. Important implementation decisions

- Proper src-layout package improves installation; functions stay plain and inspectable.
- Public small Llama-style checkpoint fits a real pilot on the 4 GiB GPU;
  0.5B/1.5B are prepared tiers rather than assumed feasible from weight size.
- UCI task has public attribution/license and manageable text examples. Its
  imbalance is recorded rather than hidden by accuracy alone.
- Kaiming A follows loralib Linear; Gaussian is a separate LoRA initialization.
- Detached DoRA denominator matches the original efficiency variant; full
  derivative remains selectable and independently tested.
- PiSSA subtracts the initialized component and stores initial factor buffers;
  portable files do not assume SVD signs or randomized seeds are reproduced.
- Cached merge preserves frozen storage across repeated merge/unmerge calls.
- Explicit projection roles avoid silently adapting unsupported architecture layouts.
- All caches/temp paths use D: because C: was nearly full. No unrelated
  files, running apps or processes were removed or stopped.
- No automatic precision/rank/sequence fallback; failures are recorded.

## I. Known limitations

Only one task/checkpoint and a short 12-step budget; small imbalanced test;
three seeds only for LoRA/PiSSA; no hyperparameter sweep/confirmatory test;
possible public-data pretraining overlap; no semantic near-duplicate detector;
Windows-only execution; runtime affected by host load/checkpoint I/O;
DoRA uses dense effective matrices; full resume files include frozen weights;
CPU RSS is sampled. Best-weight resume requires carrying the original run
directory. Adapter identity checking trusts the caller's genuine base identity,
not an arbitrary model relabelled with the same string. Quantization,
Conv1D/fused QKV, distributed/OOD/adaptive-rank experiments remain outside scope.

## J. Next research questions

Do more steps and separately validated learning rates change the observed
short-budget behavior? Does minority detection improve on a larger held-out
sample? Is initialization cost amortized by convergence savings? Does full
DoRA normalization help under a controlled budget? Does a second task change
the result? Finally, can spectral rank allocation improve a fixed-budget
uniform baseline without confusing weight energy with task importance?

## Reproduction entry points

Read `reports/REPRODUCTION_GUIDE.md` for exact Windows/Linux setup, tests,
baseline, methods, resume, saved-weight evaluation, ablations and plotting.
Run `python -m pytest -q`, then inspect existing results before launching
new training. Completed directories are protected; choose a new output_dir
for repeat runs. The baseline implementation is complete; the listed research extensions
are future work rather than unimplemented core components.
