# Third-party notices

The MIT license covers the original code and documentation in this project.
It does not replace the licenses of datasets, model checkpoints, dependencies,
or the original papers.

## Dataset and derived records

Almeida, T. and Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine
Learning Repository. [DOI: 10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
The [UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
licenses the dataset under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Tokenized SMS snapshots in the recorded benchmark are derived from that
dataset. Changes include normalized duplicate detection, conflicting-label
removal, stratified splitting, tokenization, truncation and loss masking.
Dataset-derived records retain the dataset's attribution and license.
Source archive and encoded split hashes are recorded in each run manifest.
The raw archive is downloaded separately and is not committed.

## Pretrained models

The recorded study uses
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
at commit `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
Its model card specifies Apache-2.0. Weights are not distributed in this
repository. Candidate Qwen configurations do not distribute their weights.

## Method references

- Hu et al. [LoRA](https://arxiv.org/abs/2106.09685),
  [Microsoft reference](https://github.com/microsoft/LoRA).
- Liu et al. [DoRA](https://arxiv.org/abs/2402.09353),
  [NVIDIA reference](https://github.com/NVlabs/DoRA).
- Meng et al. [PiSSA](https://arxiv.org/abs/2404.02948),
  [author repository](https://github.com/MuLabPKU/PiSSA).

The adapter implementations were written independently. Reference source is
not vendored. Optional comparison tests execute the separately installed
Hugging Face PEFT package. See [SOURCES.md](reports/SOURCES.md) for the audit.
