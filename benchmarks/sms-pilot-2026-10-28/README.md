# SMS classification pilot — 2026-10-28

A dated record of **32 actual completed runs**. New experiments write to
the ignored `results/` directory; this snapshot can be inspected without
downloading a model or running training.

## Contents

- `runs/`: configs, results, histories, tokenized splits, predictions and data manifests.
- `environment.json`: recorded hardware and package versions, without host-specific identifiers.
- `validation.json`: the recorded 60-test CUDA execution summary.
- `checkpoint_roundtrip.json`: actual pretrained-export reload checks.
- `isolated_comparison_audit.json`: exact metric reproduction in independent processes.
- `manifest.json`: per-file SHA-256 values, byte counts and the run index.

The directory below `runs/` preserves the original relative run layout. For
example, historical `results/isolated_baselines/method-lora/result.json` is
published as `runs/isolated_baselines/method-lora/result.json`. Configs retain
their original runtime output paths; no metrics were edited during export.

## Verify or analyze

From the repository root with the package installed:

```bash
python scripts/verify_benchmark.py
python -m lowrank_lab analyze --root benchmarks/sms-pilot-2026-10-28/runs --output reports/figures
python scripts/build_reports.py
```

The verifier checks hashes, split counts/disjointness, target fingerprints,
prediction-derived metrics and step histories. It does **not** verify weights
or rerun the experiments: model/adapter/optimizer checkpoints and the raw
archive are excluded. See the [reproduction guide](../../reports/REPRODUCTION_GUIDE.md)
to execute the study from the pinned public checkpoint.

## Interpretation

The main comparison uses SmolLM2-135M, 128 training examples, a 12-step
matched budget and 64 test examples, of which eight are spam. It is a pilot,
with class-prior collapse and initialization-scale confounds documented in the
[research report](../../reports/RESEARCH_REPORT.md). It does not establish a
general method ranking.

Use `runs/isolated_baselines` for memory/runtime comparison. The original
20-run shared-process pilot remains available, but its peak allocations may
include earlier models/optimizers. The five independent repeats reproduce
the original validation/test metrics exactly and avoid that contamination.

## Attribution

Tokenized SMS examples derive from Almeida and Hidalgo (2011),
[UCI SMS Spam Collection](https://doi.org/10.24432/C5CC84), CC BY 4.0.
The raw archive and split hashes identify the source and transformations.
See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
