# Executed mathematical and functional checks

The machine-readable final pytest report is `reports/pytest.xml`; the final
execution counts/time are recorded in the research report and handoff. Tests
run actual forwards/backwards and compare tensors. A generated test file alone
is never counted as a pass.

| Required behavior | Actual test(s) |
|---|---|
| LoRA dimensions / zero update / frozen weights | `test_shapes_initial_function_and_freeze` across methods and ranks |
| Nonzero factors' gradients after appropriate steps | `test_gradients_and_frozen_base` |
| Merge/unmerge / repeated merge / disabled path | `test_merge_unmerge_idempotence_state_dict_and_disabled` |
| DoRA axes / magnitudes / bias / gradient variants | `test_dora_reference_forward_gradient_axes_and_bias` |
| Full-norm differentiability / zero rows | `test_dora_full_norm_gradcheck_and_zero_rows_stable` |
| PiSSA principal component / frozen residual / scale | `test_pissa_principal_residual_and_gradients` |
| PiSSA initial function preservation | shapes test and injection test, exact and fast SVD |
| Full state dict / adapter-only loading | merge/state-dict test and portable-checkpoint test |
| Exact training checkpoint resume | `test_checkpoint_exact_resume_with_dropout_and_epoch_transition` |
| CPU forward/backward | adapter, trainer and offline Llama tests |
| Causal future masking | `test_causal_mask_no_future_leakage_and_generation` |
| Token-weighted accumulation | `test_gradient_accumulation_unequal_target_counts` |
| Tiny training loss reduction | `test_tiny_training_loss_reduction` |
| Injection and ordinary-linear restoration | `test_injection_restoration_and_portable_checkpoint` |
| Supported dtypes | `test_supported_dtype_conversion_and_numerical_stability` |
| GPU conversion, inference merge, backward | `test_cuda_forward_backward_conversion_and_merge` |
| Actual FP16 scaler/autocast training | `test_cuda_mixed_precision_training_and_memory` |
| Independent PEFT output/factor/magnitude gradients | `test_peft_forward_and_factor_gradients` |
| Truncation collisions / label masking / split integrity | `test_data_and_integration.py` |
| Config and all grid stages / save-evaluate workflow | `test_workflow.py` |

## Tolerance reasoning

Double-precision small matrices use absolute/relative tolerances near 1e-12
for algebraic equality and 1e-10 for separately constructed gradients. Those
allow a few operation-order rounding differences without accepting a wrong
axis, omitted residual or bias scaling. Finite-difference gradcheck uses
epsilon 1e-6 and atol 1e-5 because differencing subtracts nearby values.

Float32 end-to-end comparisons use approximately 1e-6–1e-5 tolerances.
Accumulated gradients can add in a different order from a concatenated batch.
PEFT gradient checks allow about 3e-5 because its factorized DoRA evaluation
has a different summation order from our dense formulation.

CPU half/bfloat16 initial-output tests allow 0.01/0.06 respectively, reflecting
their much coarser representation after conversion; GPU FP16 tests use tighter
0.003–0.005 tolerances on these small values. All supported low-precision
checks also require finite outputs and gradients. These are bounded small
examples, not guarantees against overflow on arbitrary training runs.

Exact CPU resume uses zero tolerance: final parameter tensors and training
loss sequences must be identical in the same environment. Freezing checks
also use exact equality, because the optimizer must not alter base storage.
Assertions do not hide failures by changing precision, rank or seeds.

Optional reference dependencies skip only their dedicated module when absent;
CUDA tests skip only if the runtime has no CUDA. The actual final run on this
host had PEFT and CUDA available. Consult the final XML for executed counts.
