# Primary sources and method audit

Read on 2026-10-28. Core algorithms were independently implemented; reference
source was inspected for conventions, not vendored. Web branches can change;
the installed environment lock and resolved model revision record this run's
software/model state. This is a small-method reproduction, not a reproduction
of any paper's large-model results.

| Source | What was checked | Project decision |
|---|---|---|
| [Hu et al., LoRA, arXiv:2106.09685v2](https://arxiv.org/html/2106.09685v2), §4 and hyperparameter appendix | Factor shapes, alpha/r, initialization, frozen base, merge | Manual factors; zero B; explicit rank and scale |
| [Microsoft loralib Linear](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py) | Actual linear initialization and dropout/merge behavior | Kaiming A by default, Gaussian alternative; explicit cached inference merge |
| [Liu et al., DoRA, arXiv:2402.09353v2](https://arxiv.org/html/2402.09353v2), §4.1–4.3 and appendix | Weight orientation, magnitude, gradients, detached norm | Output-channel row norms; detached default and full derivative option |
| [NVIDIA DoRA reference](https://github.com/NVlabs/DoRA/blob/main/commonsense_reasoning/peft/src/peft/tuners/dora.py) | dim=1 norms, dora_simple denominator detach, unscaled bias | Dense no-dropout formulation; bias added once after scaling |
| [Meng et al., PiSSA, arXiv:2404.02948v4](https://arxiv.org/html/2404.02948v4), §3, experiments, fast-SVD appendix | Principal components and residual; exact/fast SVD | Scale-correct square-root factors and frozen residual; zero dropout |
| [PiSSA official repository](https://github.com/MuLabPKU/PiSSA) (redirects from GraphPKU/PiSSA) | Author usage, PEFT integration, fast initialization settings | Source identification; original vendored layer URL was absent at inspection |
| [PEFT PiSSA implementation](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py) | pissa_init scaling, residual subtraction | Independent reference check; PEFT is optional, never the training implementation |
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), [DOI](https://doi.org/10.24432/C5CC84) | Public data, labels, license, download | CC BY 4.0; cite Almeida & Hidalgo (2011); archive SHA-256 in each run |
| [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | Causal LM architecture and availability | Public Apache-2.0 checkpoint; pinned commit in resolved config |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | Resource-tier choices | Apache-2.0 model cards; prepared GPU configurations, not assumed fit |
| [Official PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/) | Windows CUDA wheel availability | CPU smoke first; CUDA 12.4 torch 2.6 for this older GPU generation |

## Differences and deliberate constraints

The paper and reference LoRA initialization descriptions differ (Gaussian in
the paper; Kaiming uniform for linear layers in loralib). Both zero-update
variants are implemented and named. No paper hyperparameters are silently
transferred to this unrelated small task.

DoRA's detached and full-denominator variants have identical forwards but
different derivatives. Neither should be judged by the other's gradient
test. The detached variant follows the overhead reduction discussed in the
paper; full differentiation remains available. Adapter dropout is rejected
for DoRA/PiSSA rather than choosing an ambiguous variant.

PiSSA factor names here follow PyTorch's output-by-input convention. The
paper uses a different naming orientation. Randomized SVD subtracts exactly
the approximate component it initialized, preserving the original function
up to rounding even though its subspace approximates the principal one.

The original papers study different tasks, models, data volumes and budgets.
Their author-reported gains motivate hypotheses; no author numbers are
relabelled as local measurements. Consult the papers for their tables.

The training-configuration audit also checked LoRA appendix D.3's multi-epoch
GPT-2 setup and validation-based reporting, and DoRA appendix A.3's different
rank/scale, dropout, projection-placement and warmup choices for language and
multimodal models. Our 12-step shared-LR SMS pilot intentionally differs from
those task-specific schedules. In particular zero dropout is a local controlled
choice, not a claim to copy DoRA's commonsense-reasoning recipe. PiSSA's
limitations section leaves adaptive rank and wider theoretical explanations
open; the proposed spectral allocation extension here is likewise untested.

No quantization, QLoRA, distributed training, fused-QKV splitting, Conv1D
adaptation, or optimized DoRA kernels are claimed. Dense DoRA materializes
an effective weight and is deliberately easy to inspect, with a memory cost.
