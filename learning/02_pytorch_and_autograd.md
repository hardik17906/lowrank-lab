# PyTorch and autograd practice

Read guide sections 2 and 4. Create a two-entry vector with gradient tracking,
square it, sum it, and call backward. Confirm that the gradient has the same
shape. Repeat backward after a fresh forward without zeroing gradients and
observe accumulation. That behavior is useful within an accumulation window
and a bug when accidental across windows.

Inspect `LoRALinear.named_parameters()`. Base weight and bias are registered
parameters even when frozen. A/B are trainable. PiSSA initial_A/initial_B
instead appear in `named_buffers()`: they move with the module and serialize,
but should never be optimized. A Python tensor attribute lacking registration
would not have those properties.

A computation graph records operations, not just values. Calling `.detach()`
returns values without their preceding gradient path. `no_grad()` prevents
graph recording in an entire block. Neither changes whether dropout is
active: `.train()`/`.eval()` control that separately. The evaluation helpers
use both eval and no-grad, then restore the previous mode.

Follow `test_gradients_and_frozen_base`. Its assertions check frozen tensors
stay exactly fixed across optimizer updates, not merely that gradients happen
to be small. LoRA's first A gradient can be zero by design. B must first move
before the A path opens; the test therefore checks two steps.

Exercise: print parameter dtype/device before and after `.to(dtype=torch.bfloat16)`.
Check buffers too. Predict why SVD initialization is performed in float32,
then inspect the tolerance assigned to low-precision reconstruction tests.
Compare the purpose of a full training checkpoint with an adapter-only file:
one reproduces an optimization trajectory; the other transfers learned
adaptation onto the same base. Loading only factors cannot restore Adam moments.
