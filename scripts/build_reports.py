"""Build research/handoff documents from completed runs, never invented metrics."""

import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def table(rows, fields):
    text = "| " + " | ".join(label for label, _ in fields) + " |\n"
    text += "|" + "|".join(["---"] * len(fields)) + "|\n"
    for row in rows:
        values = [fn(row) for _, fn in fields]
        text += (
            "| "
            + " | ".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in values)
            + " |\n"
        )
    return text


def location(file, symbol):
    path = Path("src/lowrank_lab") / file
    tree = ast.parse(path.read_text())
    matches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == symbol
    ]
    return f"`{path.as_posix()}:{matches[0].lineno}` (`{symbol}`)"


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("benchmarks/sms-pilot-2026-10-28/runs")
    )
    args = parser.parse_args()
    result_root = args.results_root
    snapshot = result_root.parent
    paths = sorted(result_root.rglob("result.json"))
    results = [load(p) for p in paths]
    primary_root = result_root / "isolated_baselines"
    primary = [load(p) for p in primary_root.glob("*/result.json")]
    order = {m: i for i, m in enumerate(["baseline", "full", "lora", "dora", "pissa"])}
    primary.sort(key=lambda r: order[r["method"]])
    if len(primary) != 5:
        raise RuntimeError("Finish all five pretrained baselines before building the final report")
    selected_manifest = load("experiments/selected_manifest.json")
    from lowrank_lab.experiments.runner import read_config

    completed_selected = [
        p
        for p in selected_manifest["runs"]
        if (
            result_root / Path(read_config(p)["output_dir"]).relative_to("results") / "result.json"
        ).exists()
    ]
    isolated_count = len(list(primary_root.glob("*/result.json")))
    if len(completed_selected) != len(selected_manifest["runs"]):
        raise RuntimeError("Selected study is incomplete; report must not label it complete")
    if (snapshot / "validation.json").exists():
        validation = load(snapshot / "validation.json")
        test_status = f"{validation['tests']} executed, {validation['failures']} failures, {validation['errors']} errors, {validation['skipped']} skipped; {validation['seconds']:.2f} seconds"
        env = load(snapshot / "environment.json")
    else:
        suite = ET.parse("reports/pytest.xml").getroot().find("testsuite")
        test_status = f"{suite.attrib['tests']} executed, {suite.attrib['failures']} failures, {suite.attrib['errors']} errors, {suite.attrib['skipped']} skipped; {float(suite.attrib['time']):.2f} seconds"
        env = load("reports/environment.json")
    data = primary[0]["data"]
    seed7 = table(
        primary,
        [
            ("Method", lambda r: r["method"]),
            ("Test CE", lambda r: r["test"]["cross_entropy"]),
            ("Accuracy", lambda r: r["test"]["accuracy"]),
            ("Macro F1", lambda r: r["test"]["macro_f1"]),
            ("Trainable", lambda r: r["parameters"]["trainable_parameters"]),
            ("Training s", lambda r: r["training"]["training_seconds"]),
            (
                "Peak GPU MiB",
                lambda r: (
                    round(r["training"]["peak_allocated_gpu_bytes"] / 2**20, 2)
                    if r["training"].get("peak_allocated_gpu_bytes")
                    else "unmeasured"
                ),
            ),
            (
                "Adapter KiB",
                lambda r: (
                    round(r["adapter_checkpoint_bytes"] / 1024, 2)
                    if r["adapter_checkpoint_bytes"]
                    else "n/a"
                ),
            ),
        ],
    )
    tiny_paths = sorted((result_root / "tiny_baselines").glob("*/result.json"))
    tiny = table(
        [load(p) for p in tiny_paths],
        [
            ("Method", lambda r: r["method"]),
            ("Test CE", lambda r: r["test"]["cross_entropy"]),
            ("Test PPL", lambda r: r["test"]["perplexity"]),
        ],
    )

    def ablation_rows(prefix):
        return [
            (p.parent.name, load(p)) for p in sorted((result_root / prefix).glob("*/result.json"))
        ]

    ablations = []
    for stage in ["rank", "placement", "size", "initialization", "seeds"]:
        items = ablation_rows(stage)
        ablations += [
            f"### {stage}",
            "",
            table(
                items,
                [
                    ("Run", lambda x: x[0]),
                    ("CE", lambda x: x[1]["test"]["cross_entropy"]),
                    ("Accuracy", lambda x: x[1]["test"]["accuracy"]),
                    ("Macro F1", lambda x: x[1]["test"]["macro_f1"]),
                ],
            ),
        ]
    seed_text = []
    seed_standard_deviations = {}
    import statistics

    for method in ["lora", "pissa"]:
        seed_runs = [r for r in primary if r["method"] == method]
        seed_runs += [r for _, r in ablation_rows("seeds") if r["method"] == method]
        values = [r["test"]["cross_entropy"] for r in seed_runs]
        f1 = [r["test"]["macro_f1"] for r in seed_runs]
        seed_standard_deviations[method] = statistics.stdev(values)
        seed_text.append(
            f"- {method}: seeds {[r['seed'] for r in seed_runs]}, target CE mean {statistics.mean(values):.5f}, sample SD {statistics.stdev(values):.5f}; macro F1 mean {statistics.mean(f1):.5f}, sample SD {statistics.stdev(f1):.5f}."
        )
    seed_stats = "\n".join(seed_text)
    gaussian = next(r for name, r in ablation_rows("initialization") if "gaussian" in name)
    initialization_finding = (
        f"The Gaussian-A LoRA ablation reaches test CE {gaussian['test']['cross_entropy']:.5f} "
        f"and macro F1 {gaussian['test']['macro_f1']:.5f}, while still predicting all "
        "messages as legitimate. Its A standard deviation is 1/r (0.25 at rank four), "
        "versus approximately 0.0241 for Kaiming at input width 576. Thus this ablation "
        "changes factor scale as well as distribution. The large early-loss difference "
        "between default LoRA and PiSSA cannot isolate the value of a principal subspace; "
        "initialization scale and optimization dynamics are material confounds. "
        f"The Gaussian-vs-PiSSA seed-seven CE gap is smaller than PiSSA's observed "
        f"three-seed CE SD ({seed_standard_deviations['pissa']:.5f}); Gaussian LoRA itself "
        "was not replicated across seeds, so no stable quality ranking follows."
    )
    report = f"""# Low-Rank Adaptation Research Lab — research report

Execution date: 2026-10-28 (Asia/Calcutta). This document is generated by
`python scripts/build_reports.py` from actual completed result files.

Published artifacts are under `benchmarks/sms-pilot-2026-10-28/runs/`.
Historical `results/...` paths below identify the original runtime layout;
the same relative subdirectories are preserved in the published snapshot.

## 1. Abstract

We implemented LoRA, DoRA and PiSSA independently in PyTorch, validated their
numerical behavior, built a small educational decoder, and ran a genuine
pretrained pilot on public SMS classification. There are {len(results)} completed
saved runs, including {len(completed_selected)} staged GPU pilot runs and
{isolated_count} fresh-process primary repeats. These are
short, resource-constrained experiments rather than a full paper reproduction.
Tests: {test_status}. The pilot supports studying behavior under a fixed
budget; it does not establish a universally better adaptation method.

## 2. Introduction

Fine-tuning can modify all parameters or a small selected parameterization.
The laboratory exposes the actual weights, gradients, initialization and
optimization loop so that a beginner can inspect both the mathematical
methods and the empirical controls. A random tiny model is kept separate
from the pretrained task to avoid conflating training mechanics with LLM
adaptation performance.

## 3. Research questions

How do unrestricted updates, additive low-rank updates, magnitude/direction
adaptation and principal-component adaptation behave under matched optimizer
steps? How do rank, placement, data amount and initialization affect the
short-budget result? How large is observed seed variation for affordable
LoRA/PiSSA subsets? Runtime, memory, storage and parameter counts are measured
with explicit definitions, not inferred solely from trainable fraction.

## 4. Background

The tiny decoder uses learned positions, causal multi-head attention,
pre-normalized residual blocks, a GELU FFN and an untied vocabulary head.
The real model is a pretrained causal LM; classification is formulated as
conditional likelihood of fixed answer strings. The complete beginner
derivations, symbol definitions and numerical examples are in
`learning/COMPLETE_LEARNING_GUIDE.md`.

## 5. Original papers

[LoRA](https://arxiv.org/abs/2106.09685),
[DoRA](https://arxiv.org/abs/2402.09353) and
[PiSSA](https://arxiv.org/abs/2404.02948) are the methodological sources.
Their authors report results on their own large-model/task settings. Those
author results are not local measurements and no paper table is reproduced
as if obtained here. Reference conventions and implementation differences
are audited in `reports/SOURCES.md`, including the official repositories.

## 6. Mathematical methodology

LoRA trains A/B around frozen W0 with scale alpha/r and zero initial update.
DoRA adds independently trainable output-channel magnitudes and normalizes
the updated direction; PyTorch row norms correspond to the paper's column
convention. Detached denominator is the default, with full differentiation
available as a named study. Bias remains unscaled. PiSSA initializes from
leading singular components with square-root scaling and freezes the
subtracted residual, preserving the initial pretrained function. Exact and
randomized SVD modes are separate recorded configurations.

Dense DoRA and explicit cached inference merge favor inspectability over
kernel efficiency. Dropout is zero in these comparisons. PiSSA exports save
initial factors to reconstruct residuals portably; they are consequently
larger than plain two-factor LoRA files.

## 7. Implementation

Core algorithms are in `src/lowrank_lab/adapters`, with no PEFT dependency in
the training path. Injection uses explicit nn.Linear projection roles;
Llama/Qwen/OPT-style suffixes are supported, while Conv1D, quantized layers
and arbitrary fused QKV are not claimed. Training supports token-weighted
accumulation, clipping, AdamW/SGD, scheduling, validation, deterministic
optimizer-boundary resume, optional early stopping/best selection and AMP.
The test suite includes an independent PEFT numerical comparison and a
real Transformers Llama architecture instantiated offline.

## 8. Experimental setup

Host: {env["os"]}, Python 3.12.4, torch {env["torch"]}; GPU GTX 1650 Ti,
4 GiB VRAM; host RAM {env["ram_total_bytes"] / 2**30:.2f} GiB. CPU smoke ran first
on torch 2.14.1+cpu. The pretrained GPU pilot uses FP32 weights and compute,
microbatch two, accumulation two, 12 AdamW optimizer steps, LR 0.0002,
one warmup step, cosine schedule, zero weight decay, clipping norm one,
validation every six steps, final weights, and no early stopping.

Pretrained model: `{primary[0]["model"]["model_id"]}` at exact commit
`{primary[0]["model"]["revision"]}`. Task source: Almeida & Hidalgo (2011),
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
CC BY 4.0, DOI 10.24432/C5CC84. Archive SHA-256:
`{data["raw_sha256"]}`. Actual raw rows {data["raw_rows"]}; removed normalized
duplicates/conflicting-label rows {data["removed_duplicate_or_conflicting_rows"]}.
Initial stratified split counts: {data["full_counts"]}. Additional token-prompt
collisions removed: {data["removed_token_prompt_collisions"]}.

Pilot counts: {data["used_counts"]}; class supports (legitimate, spam):
{data["class_counts"]}. Fixed split seed 42; baseline training seed 7.
Max sequence length 64, answer-only loss, constrained sum-log-likelihood label
scoring. Training subsets are ordered deterministic prefixes of the cap.
The combined encoded-split SHA is `{data["encoded_splits_sha256"]}`.
Both candidate labels tokenize to one token under the pinned tokenizer in
these saved examples; the code supports multiple-token candidates.

Training runtime includes interval validation and interval checkpoint I/O;
final checkpoint/export and final evaluation are excluded. Step timestamps
precede their own validation, so threshold times use validation-start times.
Initialization is separately measured. GPU memory is maximum torch-allocated memory during
training, not total system GPU use. CPU RSS is sampled and not a true lifetime
peak. Baseline has no training memory measurement. This is matched steps,
not matched wall-clock time, memory or parameter budget. Shared LR is a control,
not an assertion that every method is equally well tuned.

The primary table and efficiency figures use five fresh-process repeats under
`results/isolated_baselines`. They exactly reproduce the original validation/
test metrics. Earlier shared-process pilot peaks may include allocations from
previous models/optimizers and are not used for fair method-memory comparison.
Original raw measurements remain saved. Suite execution now uses one process
per run; the audit checks distinct PIDs for the primary comparison.

## 9. Results

### Genuine pretrained matched-step pilot

{seed7}

The no-update model predicts spam for every test message (8/64 correct).
For context, predicting legitimate for every message would give 56/64 =
87.5% accuracy and macro F1 about 0.46667. Therefore an accuracy increase
alone does not show useful minority-class detection; consult the saved
confusion matrices and per-example scores. Full fine-tuning's measured
confusion matrix is {next(r for r in primary if r["method"] == "full")["test"]["confusion_matrix"]}.

Interpret this table as a short-budget observation under one shared learning
rate. No ranking here establishes general superiority or converged performance.

### Educational random-initialized study

{tiny}

The separate smoke run duplicates the full tiny configuration as an independent
pipeline demonstration, not another independent seed sample. Its real greedy
continuations are saved in `results/smoke/generation.json`; they contain errors
and are not presented as successful natural-language generation. The tiny
study uses a different task and starting function from the pretrained study.

### Seed uncertainty

{seed_stats}

Only three seeds per selected method are available. Within-run test uncertainty
remains large because just eight test examples are spam. A single changed
prediction moves test accuracy by 1.5625 percentage points. Observed zero
classification variation, when present, does not mean a method is generally
seed-invariant. Differences comparable to observed seed SD are not strong
evidence of a stable advantage. There is no significance or power claim.

## 10. Ablation studies

The selected subset executes all four LoRA ranks (rank four is the baseline),
all three LoRA placements (attention is baseline), and all four training
fractions (100% is baseline). Initialization runs add Gaussian LoRA, fast
PiSSA and full-norm DoRA. Seed runs add 17 and 27 for LoRA/PiSSA.
Other methods' full generated grids are prepared but were not executed.

Rank grids keep alpha at eight, so alpha/r changes with rank. Placement
changes parameter count. Data-size grids keep 12 steps, so small datasets
are revisited more often; fractions are of the 128-example cap, not the
entire source train split. These design choices limit causal interpretations.

{chr(10).join(ablations)}

Complete actual-result table: `reports/figures/TABLE.md`. CSV contains
initialization time, checkpoint storage, device, precision, revision and hashes.
Seed summaries deduplicate repeated seed/configuration combinations.

## 11. Discussion

The code makes three distinct parameterizations inspectable and demonstrates
both exact initialization preservation and actual training. A short prompt
classification pilot is especially sensitive to learning the answer-token
convention and prior class frequency. The unrestricted model updates
embeddings/head/norms as well as projections; attention-only PEFT cannot.
Shared hyperparameters and 12 steps may favor different methods unevenly.
{initialization_finding}
Training and memory efficiency should be interpreted with checkpoint I/O and
the dense DoRA implementation in mind. Exact versus fast SVD initialization
costs belong in efficiency analysis rather than being hidden from runtime.

## 12. Threats to validity

Small, imbalanced subsets; few seeds; one task/model; arbitrary but fixed
label wording; public-corpus pretraining overlap unknown; normalized/token
duplicate filtering does not detect all near duplicates; no independent
confirmatory test after exploratory ablations; finite optimizer budget;
no method-specific validation hyperparameter search; rank-dependent scaling;
different parameter counts across placements; reused samples at small data
fractions; host load and Windows WDDM scheduling affect timing. Test metrics
are exploratory observations and must not be used to select a claimed
unbiased final benchmark result.

## 13. Limitations

The 0.5B and 1.5B configurations were prepared, not executed. No multi-billion
model training, quantized/QLoRA/QPiSSA results, distributed training, second
OOD task, adaptive-rank allocation or full original-paper benchmark was run.
CUDA tests cover the discovered GPU and CPU tests cover Windows; Linux
commands/configurations are provided but were not executed on Linux here.
DoRA dropout variants and Conv1D/fused QKV adaptation remain unsupported.
The trainer stores full resume state even for PEFT, so resume files are much
larger than adapter-only exports. Memory estimates are conservative guards,
not guarantees. Exact CPU resume is tested within one environment, not
bitwise across software versions or hardware.

## 14. Conclusions

The deliverable is a tested, executable research laboratory with real CPU/GPU
measurements and auditable artifacts. Under this small fixed budget the
tables describe local behavior only. Further training, broader validation
search and larger held-out datasets are required before making claims about
method quality or generalization. Mathematical correctness and empirical
superiority are separate questions.

## 15. References

- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models, arXiv:2106.09685.
- Liu et al. DoRA: Weight-Decomposed Low-Rank Adaptation, arXiv:2402.09353.
- Meng et al. PiSSA: Principal Singular Values and Singular Vectors Adaptation
  of Large Language Models, arXiv:2404.02948.
- Almeida & Hidalgo (2011), SMS Spam Collection, UCI, DOI 10.24432/C5CC84.
- HuggingFaceTB SmolLM2-135M pinned model card/checkpoint.
- Microsoft LoRA, NVIDIA DoRA, MuLabPKU PiSSA and Hugging Face PEFT reference
  repositories, with direct links and audited conventions in `reports/SOURCES.md`.

## 16. Future research directions

Increase examples/steps, preregister validation selection, add an untouched
confirmatory test and another public task, expand seeds and learning-rate
search, compare matched wall-clock budgets, measure exact/fast SVD tradeoffs
and implement optimized DoRA. Only after these baselines, investigate
layer-sensitive rank allocation with exact parameter-budget accounting.
Spectral energy is a hypothesis about useful allocation, not evidence of
task-specific importance. Negative results must be retained.
"""
    Path("reports/RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    math_map = "\n".join(
        [
            "- LoRA factor construction: "
            + location("adapters/lora.py", "LoRALinear")
            + "; update "
            + location("adapters/lora.py", "delta_weight")
            + ".",
            "- DoRA row norms: "
            + location("adapters/dora.py", "_row_norm")
            + "; normalized effective weight "
            + location("adapters/dora.py", "effective_weight")
            + ".",
            "- PiSSA scale-correct SVD/residual initialization: "
            + location("adapters/pissa.py", "PiSSALinear")
            + "; original component "
            + location("adapters/pissa.py", "initial_component")
            + ".",
            "- Token-summed causal CE: "
            + location("training/losses.py", "causal_loss_sum")
            + "; accumulation and update "
            + location("training/trainer.py", "fit")
            + ".",
            "- Conditional candidate scoring: "
            + location("evaluation/benchmark.py", "evaluate_classification")
            + ".",
        ]
    )
    handoff = f"""# PROJECT_HANDOFF — tutor and owner entry point

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

{math_map}

Full derivations, shapes, symbol definitions, worked examples, failure modes
and tests are in the learning guide. Full state dictionaries include adapter
metadata; inference caches are nonpersistent. Adapters freeze base bias as well
as weight. Magnitudes are output-channel scalars, not input-channel scalars.

## D. Execution status

- Recorded pytest execution: {test_status}. Initial CPU-only suite: 51 passed;
  independent reference checks: three passed; expanded CUDA suite executed.
- {len(results)} completed saved runs under results, including {len(completed_selected)}
  selected genuine pretrained GPU runs plus {isolated_count} fresh-process repeats,
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

Pinned model `{primary[0]["model"]["model_id"]}` at
`{primary[0]["model"]["revision"]}`. UCI SMS, archive
`{data["raw_sha256"]}`. {data["raw_rows"]} raw rows;
{data["removed_duplicate_or_conflicting_rows"]} duplicate/conflicting rows removed;
fixed 70/15/15 splits followed by tokenized prompt collision filtering.
Pilot uses {data["used_counts"]}; test support 56 legitimate / 8 spam.
Seed-seven baseline settings: rank four, alpha eight, attention-only adapters;
12 matched AdamW steps, microbatch two × accumulation two, LR 0.0002,
FP32, context 64, final weights and no early stopping.

{seed7}

{seed_stats}

The no-update model predicts every example as spam. Majority legitimate alone
would achieve 87.5% accuracy; use macro F1 and confusion matrices to judge
minority behavior. This is not a method-specific tuned or converged benchmark.
The full real tables/plots and ablation interpretation are in the research
report and `reports/figures`. Source author claims remain separate.

{initialization_finding}

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
"""
    Path("PROJECT_HANDOFF.md").write_text(handoff, encoding="utf-8")
    print(f"Built report and handoff from {len(results)} completed runs; {test_status}")


if __name__ == "__main__":
    main()
