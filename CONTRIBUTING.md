# Contributing

Changes should preserve the ability to inspect the mathematics and reproduce
the result. Explain the problem, the proposed behavior, and the evidence in
the pull request.

## Local checks

Install a suitable CPU or CUDA PyTorch build, then:

```bash
python -m pip install -e ".[dev,reference]"
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest -q
python scripts/verify_benchmark.py
```

The test suite needs no pretrained-model download. CUDA checks skip on CPU
hosts; optional PEFT reference tests skip when PEFT is absent. CI installs
the reference extra and runs CPU checks on Linux and Windows.

## Algorithm changes

State the weight orientation, factor shapes, scaling convention and
initialization. Describe whether the pretrained function is preserved and
which tensors receive gradients. Add a small independent numerical example
when changing the mathematics. Initialization, normalization or checkpoint
changes should retain merge and save/load behavior.

## Experimental changes

Use a new output directory. Record the complete configuration, model commit,
dataset fingerprint, seed, token budget, evaluation rule and measurement
scope. Keep exploratory comparisons separate from selected-model claims.
Negative or inconclusive results are useful and should remain visible.

The dated benchmark directory is a record of completed runs. Do not edit its
metrics by hand or overwrite it to update a plot. Add a separately identified
study and regenerate figures from that study's machine-readable outputs.

Do not commit model weights, credentials, private datasets, virtual
environments or local execution logs. Dataset-derived snapshots require
source attribution and a compatible redistribution license.
