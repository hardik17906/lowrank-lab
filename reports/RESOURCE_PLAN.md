# Resource plan and measurement definitions

Initial host: Windows 11, Python 3.12.4, 16 GiB installed RAM (about 2–4 GiB
free during discovery), GTX 1650 Ti with 4096 MiB VRAM, approximately 95 GiB
free on D:. C: had only about 0.4 GiB free. The first dependency installation
failed because temporary/cache files went to C:. All subsequent task caches
and temporary files were redirected to D:. No unrelated files or processes
were removed or stopped.

The initial PyTorch build was CPU-only; the driver being present did not make
torch CUDA-enabled. CPU mathematical tests and the educational study ran
first. An official torch 2.6.0 CUDA 12.4 Windows wheel was then downloaded
to D: after checking resources, for this GPU's older compute generation.
The final environment report records actual runtime availability.

## A — CPU and offline smoke

`configs/smoke.yaml`: width-48, two-layer random tiny decoder, 24-token
sequences, microbatch eight, accumulation two, float32, 60 optimizer steps.
No network or pretrained weights needed. Test suite additionally uses smaller
models and exact double-precision matrix examples. `pretrained_cpu.yaml`
is a genuine 135M pretrained fallback with 64-token examples; it is slower
and not substituted silently if a GPU run fails.

## B — Limited GPU

`configs/pretrained_gpu.yaml` runs the affordable 135M pilot on the discovered
4 GiB device. `configs/limited_gpu.yaml` prepares the requested approximately
0.5B tier using Qwen2.5-0.5B, batch one, FP16 autocast, gradient checkpointing
and accumulation 16. This is a candidate configuration, not a guarantee of
fit on 4 GiB. Its FP32 base alone occupies roughly 2 GB; activations,
gradients, optimizer state, dense DoRA operations and initialization copies
all need additional room. Run one LoRA configuration before expanding a grid.
Full fine-tuning at this tier is expected to exceed the conservative 4 GiB
budget and must not be assumed feasible merely from weight storage.

## C — Larger GPU

`configs/larger_gpu.yaml`: Qwen2.5-1.5B, context 256, BF16 autocast,
gradient checkpointing, batch two, accumulation eight. Select a GPU with
substantial headroom (typically 16–24 GiB for the PEFT configuration;
full fine-tuning can need much more). This is planning guidance, not a
measured requirement. A 3B experiment can use the same supported projection
mapping after selecting and reviewing a suitable public checkpoint and
license; no 3B download was performed. BF16 requires explicit hardware support.

## What the guards can and cannot establish

Before allowed model downloads, inspect remote safetensors file sizes and
require spare disk/host RAM. Before moving to CUDA, estimate FP32 static
weights plus gradients and Adam moments for trainable parameters and reserve
at least 20% of free VRAM for additional allocations. This cannot predict all
activation or kernel workspace peaks; an OOM still raises and records failure.
Exact SVD runs on CPU before moving the model to the GPU, trading host
initialization work for reduced GPU pressure. Fast SVD is an explicit option.

Peak GPU bytes are torch's maximum allocated bytes during training. CUDA
allocator resets do not free live model allocations; they remain counted.
CPU RSS is sampled after steps, so short-lived peaks may be missed. Baseline
has no training-memory measurement. Training seconds include optimizer work,
interval validation and interval checkpoint I/O; initialization/download time
is recorded separately. The final checkpoint/export and end-of-run
validation/test evaluation are outside this timer. Per-step timestamps are
captured before that step's validation; time-to-threshold charts therefore
use the validation-start timestamp, not its completion timestamp.
Cross-device runtime measurements must be labelled and cannot be pooled.

The initial 20-run pilot shared one process. Its peak-memory readings can
include allocations left by previous model/optimizer objects, so they are
not fair standalone method-memory comparisons. All five primary baselines
were repeated under `results/isolated_baselines`, one process per run; those
readings drive the main efficiency plot/table. Their environment JSONs record
different process IDs, and an audit checks exact equality of validation/test
metrics to the original runs. Future suites launch one process per run.
