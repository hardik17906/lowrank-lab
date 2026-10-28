# Reproduction guide

Run commands from the repository root. Prepared GPU configurations are not
measurements until executed. A failed command exits nonzero and records a
failure for experiment runs. No command silently changes hardware or precision.

## Environment setup

Windows PowerShell (from your cloned repository):

```powershell
Set-Location path\to\lowrank-lab
py -3.12 -m venv .venv
New-Item -ItemType Directory -Force .cache/tmp,.cache/pip | Out-Null
$env:TEMP = "$PWD\.cache\tmp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "$PWD\.cache\pip"
$env:HF_HUB_DISABLE_XET = '1'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

If PowerShell script activation is blocked, prefix commands with
`.\.venv\Scripts\python.exe` instead of changing system execution policy.
Recorded artifacts can be read without recreating the training environment. For GPU execution on this specific GTX 1650 Ti, the tested
wheel is `torch==2.6.0` from the CUDA 12.4 index. For other GPUs choose a
compatible official build; do not infer build compatibility from nvidia-smi's
driver CUDA version alone.

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install sympy==1.13.1
python -m lowrank_lab environment --output reports/environment.json
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
mkdir -p .cache/tmp .cache/pip
export TMPDIR="$PWD/.cache/tmp" PIP_CACHE_DIR="$PWD/.cache/pip"
export HF_HUB_DISABLE_XET=1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m lowrank_lab environment
```

`requirements-lock.txt` records the final Windows execution environment.
It includes a platform-specific CUDA torch version; for a CPU machine install
its CPU equivalent first, then install matching portable packages. The broad
`requirements.txt` describes supported dependencies; the lock is execution
provenance, not a universal cross-platform lock.

## Tests and offline demonstration

```text
python -m pytest -q --junitxml=reports/pytest.xml
python -m lowrank_lab run --config configs/smoke.yaml
python -m lowrank_lab evaluate --config configs/smoke.yaml
python -m lowrank_lab grid --base configs/smoke.yaml --output configs/experiments/tiny_baselines --stage baselines
python -m lowrank_lab suite --manifest configs/experiments/tiny_baselines/manifest.json
```

Existing completed run directories are protected. To rerun, copy a YAML and
change `output_dir`, or generate a grid into a newly named directory. Do not
delete completed results just to make a command succeed. Suite deliberately
skips completed runs. For example, rerunning a saved manifest is a safe way
to fill missing runs without overwriting existing ones.

PEFT is optional and never used by the main implementation:

```text
python -m pip install peft==0.21.2 accelerate==1.15.0
python -m pytest -q tests/test_reference_peft.py
```

Without PEFT, only its dedicated optional reference test module is skipped;
all manual numerical tests still run. CUDA-specific tests skip on CPU builds.
Final test status records actual executed/skipped counts.

## Prepare the public model and data once

```text
python -m lowrank_lab prepare --config configs/pretrained_cpu.yaml --allow-download
```

This checks resources, downloads only public checkpoint/tokenizer/data files,
deduplicates splits and writes `configs/pretrained_cpu.resolved.yaml` plus a
data manifest. The checked-in base configuration already pins the observed
SmolLM2 commit. The generated resolved file is a flattened copy suitable for
grids. The dataset archive SHA and encoded splits are saved in each actual run.
Subsequent runs use local snapshots and work without network access.

## Baseline inference and method training

CPU commands for individual methods:

```text
python -m lowrank_lab run --config configs/baseline.yaml
python -m lowrank_lab run --config configs/full_finetune.yaml
python -m lowrank_lab run --config configs/lora.yaml
python -m lowrank_lab run --config configs/dora.yaml
python -m lowrank_lab run --config configs/pissa.yaml
```

The actual affordable GPU pilot uses the same model/task/step settings:

```text
python -m lowrank_lab grid --base configs/pretrained_gpu.yaml --output configs/experiments/pretrained_baselines --stage baselines
python -m lowrank_lab suite --manifest configs/experiments/pretrained_baselines/manifest.json --limit 1
python -m lowrank_lab suite --manifest configs/experiments/pretrained_baselines/manifest.json
```

The first limited call stages baseline inference; the remaining call skips
that completed run and trains the four methods sequentially. These are matched
12-step pilots by default, not converged benchmark claims.

## Resume and reevaluate

```text
python -m lowrank_lab run --config configs/lora.yaml --resume results/lora/checkpoint.pt
python -m lowrank_lab evaluate --config configs/lora.yaml
```

Resume requires an interrupted run without a completed result and identical
configuration/data identity. Use the YAML for the actual run, not a different
method or output directory. Checkpoints are saved at validation and final
optimizer boundaries. To rehearse interruption behavior without manual process
killing, run the exact-resume test. `evaluate` reloads the final model or adapter,
then writes `evaluation.json` while retaining original `result.json`.

GPU saved runs can be reevaluated with their exact generated YAML, e.g.:

```text
python -m lowrank_lab evaluate --config configs/experiments/pretrained_baselines/02_method-lora.yaml
```

## Generate ablations, then stage execution

```text
python -m lowrank_lab grid --base configs/pretrained_gpu.yaml --output configs/experiments/rank --stage rank
python -m lowrank_lab grid --base configs/pretrained_gpu.yaml --output configs/experiments/placement --stage placement
python -m lowrank_lab grid --base configs/pretrained_gpu.yaml --output configs/experiments/size --stage size
python -m lowrank_lab grid --base configs/pretrained_gpu.yaml --output configs/experiments/seeds --stage seeds
python -m lowrank_lab grid --base configs/pretrained_gpu.yaml --output configs/experiments/initialization --stage initialization
python -m lowrank_lab suite --manifest configs/experiments/rank/manifest.json --limit 1
```

Rank includes 4/8/16/32; placement attention/FFN/combined; training size
10/25/50/100% of the configured training cap; seeds 7/17/27. Initialization
compares Kaiming/Gaussian LoRA, exact/fast PiSSA and detached/full-norm DoRA.
Generated manifests explicitly describe matched-step fairness. Full grids
are not launched automatically. The checked-in `experiments/executed_manifest.json`
and report identify the actual subset executed during the implementation.

For a CPU-only ablation substitute `configs/smoke.yaml` as base and use a
different output name, documenting that the result is educational synthetic
training. Do not label it pretrained adaptation.

## Resource-tier configurations

```text
python -m lowrank_lab prepare --config configs/limited_gpu.yaml --allow-download
python -m lowrank_lab run --config configs/limited_gpu.resolved.yaml
python -m lowrank_lab prepare --config configs/larger_gpu.yaml --allow-download
python -m lowrank_lab run --config configs/larger_gpu.resolved.yaml
```

Only run these after checking the target device. A 4 GiB device is not promised
to fit every method at 0.5B. The larger tier needs BF16 support and substantial
VRAM. The host's free RAM matters for loading and CPU SVD as well.

## Produce actual tables and plots

```text
python -m lowrank_lab analyze --root results --output reports/figures
python scripts/generate_code_reference.py
```

Read comparison.csv, seed_summary.csv and TABLE.md together with each run's
config. Figures include loss curves, method metric panels, rank/size/parameter
scatter plots and time-to-validation curves. Absent measurements are not
fabricated. Final report tables identify the matched comparison subset.

## Exact selected comparison on a fresh workspace

Install the recorded environment, prepare the pinned checkpoint, then run
the saved `experiments/executed_manifest.json` using `suite`. On the original
workspace completed runs are skipped. On a fresh workspace the same configs
execute in their recorded order. Compare dataset raw/encoded hashes before
interpreting numerical differences; cross-device or cross-version results
can be statistically comparable without being bitwise identical.

The final primary efficiency comparison is a separate five-run repeat:

```text
python scripts/run_isolated_baselines.py
```

Suite launches each experiment in a fresh process. The original 20 pilot
results used a shared process; their raw peak-memory values are retained but
excluded from fair standalone method-memory comparisons. The isolated
repeats under `results/isolated_baselines` drive the final primary plots and
report table. Distinct process IDs and exact metric reproduction are audited.

To rebuild and audit all final artifacts after the selected and isolated
studies have completed:

```text
python scripts/verify_saved_models.py
python -m lowrank_lab analyze --root results --output reports/figures
python scripts/build_reports.py
python scripts/audit_project.py
```


## Published snapshot and a fresh clone

Recorded results are committed under `benchmarks/sms-pilot-2026-10-28/runs/`.
The original runtime `results/` directory, weights, virtual environment and
caches are excluded from Git. A fresh clone therefore has an empty runtime
workspace and the smoke command can run immediately.

```text
python scripts/verify_benchmark.py
python -m lowrank_lab analyze --root benchmarks/sms-pilot-2026-10-28/runs --output reports/figures
python scripts/build_reports.py
```

These commands need no weights or network. The recorded CUDA validation
summary belongs to the dated snapshot; a new CPU CI execution reports its
own test results and skips unsupported CUDA checks.

`scripts/audit_project.py`, `scripts/verify_saved_models.py`, and
`scripts/tiny_demo.py` audit or load local run files and require those files
to exist. They are not intended to succeed against a fresh clone without
first rerunning the relevant experiment. Use `verify_benchmark.py` to audit
the published text artifacts independently.
