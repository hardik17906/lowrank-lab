# Complete learning guide

This guide starts from basic supervised machine learning. Work slowly with
paper, a Python prompt, and the real files. A formula below is a recipe for
arrays, not a prerequisite you must already know. All numerical examples
are teaching calculations, not experimental measurements.

## How to use the guide

First read through neural networks and PyTorch. Next trace the tiny model
before studying adapters. Then compare the three weight parameterizations.
Finally study data control, training and the report. For each formula:
identify the axes, work the small example, locate the code, and predict a
failure before running its test. The accompanying walkthrough maps the
important functions to inputs, outputs, intermediate tensors and tests.

## 1. From a perceptron to a network

A feature is a number describing an input. A weight says how much that
feature contributes. A bias supplies an offset independent of the input.
A perceptron combines weighted features then applies a nonlinear rule.

**Formula: affine map.** `z_j = sum_i W[j,i] x_i + b_j`, or `z = W x + b`.
Here `x` has `d_in` features; `W` is a `d_out × d_in` weight matrix; `b`
has `d_out` entries; `z` has `d_out` output features. Index `i` enumerates
inputs and `j` outputs. We need this operation to combine evidence learned
from data. Think of each output row as one weighted question about the input.
For `W=[[1,2],[-1,0]], x=[3,4], b=[1,2]`, the output is `[12,-1]`.
The first number is `1*3+2*4+1`. In the code, `nn.Linear` stores exactly
this orientation; batched inputs compute `x @ W.T + b`. Reversing the matrix
orientation either raises a shape error or changes the meaning of DoRA's
normalization. See `tests/test_adapters.py::test_shapes_initial_function_and_freeze`.

Matrix multiplication is repeated dot products. It is not elementwise
multiplication. An array shaped `[2,3]` multiplied by `[3,4]` yields
`[2,4]`: the shared axis is summed out. Batch/time axes may be retained.
Write the dimensions before doing algebra. The final axis is the feature
axis in this repository's linear layers.

**Formula: activation.** `ReLU(z)=max(0,z)`; in the tiny model we instead
use `GELU(z)=z Φ(z)`, where `Φ` is the standard normal cumulative
distribution function and `z` is a scalar activation. Both introduce
nonlinearity. Without a nonlinear step, composing affine layers remains
one affine map, unable to represent many useful relationships. ReLU of
`[-2,0,3]` is `[0,0,3]`; GELU of `1` is approximately `0.8413`.
`TransformerBlock.forward` applies `F.gelu` between `up_proj` and
`down_proj`. Removing the activation changes the function class; placing it
after the wrong projection changes the trained architecture.

A network's forward pass computes predictions. A loss assigns a badness
score. Backpropagation calculates how changing every trainable number
would change that score. An optimizer then changes those numbers. This
cycle is what training means; constructing a model alone does not train it.

**Formula: squared error.** `L=(1/N) sum_n (yhat_n-y_n)^2`. Here `N` is
the number of scalar predictions, `yhat_n` a prediction, `y_n` its target,
and `L` the mean error. It makes larger errors costly. For predictions
`[2,4]` and targets `[1,2]`, `L=(1+4)/2=2.5`. The derivative for prediction
`n` is `2(yhat_n-y_n)/N`, obtained by differentiating a square. Adapter
gradient tests use `F.mse_loss` because small regression examples isolate
weight behavior. The language-model task uses cross-entropy instead.

## 2. Derivatives, chain rule and backpropagation

A derivative measures a local rate of change. If changing a weight slightly
increases loss, its derivative is positive. Moving against that derivative
locally reduces loss. A gradient is the collection of derivatives for a
tensor. It has the same shape as the tensor whose values it describes.

**Formula: chain rule.** If `z=w*x` and `L=z^2`, then
`dL/dw=(dL/dz)(dz/dw)=2*z*x`. Here `x` is a fixed scalar input, `w` a
trainable scalar, `z` the intermediate product and `L` the scalar loss.
For `x=3,w=2`, `z=6,L=36,dL/dw=36`. This rule joins local derivatives
through a computation graph. PyTorch performs it when `Trainer.fit` calls
`backward`. Detaching an intermediate removes a chain-rule path; detaching
the adapter output would prevent learning. DoRA deliberately offers a
specific detached norm, discussed later, rather than detaching everything.

**Formula: gradient descent.** `theta_new=theta_old-eta*g`, where `theta`
is any parameter, `g=dL/dtheta` its gradient, and `eta>0` the learning rate.
For `theta=2,g=3,eta=.1`, the next value is `1.7`. It is the first-order
local rule for reducing loss; too large a step can overshoot. Implemented
by SGD in `training/optimizers.py::make_optimizer`; the default uses AdamW.
The gradient is accumulated by backward, so forgetting to zero gradients
changes the update into a sum across unrelated windows.

## 3. AdamW, regularization and overfitting

SGD treats each coordinate similarly. Adam tracks recent gradient direction
and size independently per coordinate. It can help when coordinates have
different scales, but its extra state consumes memory.

**Formula: Adam moments.** `m_t=beta1*m_(t-1)+(1-beta1)*g_t` and
`v_t=beta2*v_(t-1)+(1-beta2)*g_t^2`. `t` is the optimizer step; `g_t`
is its gradient; `m_t` averages gradients; `v_t` averages squared gradients;
`beta1,beta2` are decay coefficients between zero and one. Squares and
products here act coordinatewise. Starting from zero with `g=2,beta1=.9,
beta2=.99` gives `m_1=.2,v_1=.04`. The averages initially underestimate
their stationary values, so Adam uses `mhat_t=m_t/(1-beta1^t)` and
`vhat_t=v_t/(1-beta2^t)`. At step one these are `2` and `4`.

**Formula: AdamW update.**
`theta_new=(1-eta*lambda)*theta_old-eta*mhat/(sqrt(vhat)+epsilon)`.
Here `lambda` is weight decay, `epsilon>0` prevents division by zero, and
the remaining symbols are defined above. It rescales a gradient by recent
magnitude while separately shrinking the parameter. With zero weight decay,
`eta=.01,mhat=2,vhat=4,epsilon` negligible, the update is about `-.01`.
This is delegated to tested `torch.optim.AdamW`, not reimplemented here.
Checkpointing only weights would lose `m` and `v`; exact resumed training
would then diverge. `test_checkpoint_exact_resume_with_dropout_and_epoch_transition`
checks the complete optimization trajectory.

Regularization discourages fitting accidents in the training set. Weight
decay shrinks parameters; dropout randomly suppresses some intermediate
values during training. Overfitting means training improves while
performance on separate examples stalls or worsens. Validation guides
choices; test data estimates the final chosen procedure. Looking repeatedly
at test performance while changing the procedure turns the test into a
development set. This repository's exploratory ablations share held-out data;
the report must not treat their best observed test score as an unbiased
selected model estimate.

## 4. PyTorch tensors, modules and autograd

A tensor has values, shape, dtype and device. `torch.long` is used for token
IDs. Trainable weights are floating point. CPU and GPU tensors must be
placed consistently. `model.to(device)` moves parameters and registered
buffers; an unregistered tensor attribute would not move automatically.

Try this independently:

```python
import torch
w = torch.tensor(2.0, requires_grad=True)
loss = (w * 3) ** 2
loss.backward()
print(w.grad)  # 36
```

`requires_grad=True` asks autograd to track operations affecting that tensor.
The graph exists for the current forward computation. `loss.backward()`
adds derivatives into leaf `.grad` fields. `optimizer.zero_grad(set_to_none=True)`
starts the next accumulation window. It is normal for LoRA's A gradient to
be zero at initialization: zero B blocks that path initially. It should
become nonzero after B has changed, which the gradient tests check.

`nn.Module` organizes children, parameters, buffers, train/eval modes and
serialization. Assigning an `nn.Parameter` registers a trainable candidate.
Freezing it with `requires_grad_(False)` stops its own gradient, but does
not stop gradients flowing through its operation to earlier adapters. A
buffer is persistent tensor state that an optimizer should not update.
PiSSA's original factors are buffers; its current factors are parameters.

`model.train()` enables dropout; `model.eval()` disables it. Evaluation mode
does not disable autograd. `torch.no_grad()` disables recording operations;
the evaluation functions use it to reduce memory. Inference merge caches
are invalidated when training resumes. A state dictionary maps registered
names to tensor state and adapter metadata. It is not the same as an
optimizer checkpoint. Full resume also needs sampler position, RNG and
scheduler/scaler state.

Floating point is approximate. Float32 has more precision than float16 or
bfloat16. Half types store fewer bytes but can round reconstruction or
overflow gradients. Autocast chooses lower-precision operations while
allowing sensitive calculations to stay in float32. FP16 GradScaler scales
loss to keep small gradients representable, then unscales before clipping.
Unsupported device/precision choices raise errors rather than switching
settings. No BF16 support is assumed for the GTX 1650 Ti.

## 5. Tokens, embeddings and positions

Tokenization converts text to IDs in a finite vocabulary. A token can be a
word fragment, punctuation, whitespace, or a special marker. IDs are
indices, not continuous feature measurements: token 10 is not twice token 5.
The tokenizer and pretrained checkpoint must agree on the vocabulary.

**Formula: embedding lookup and position.** `x[b,t,:]=E[id[b,t],:]+P[t,:]`.
Here `b` indexes examples, `t` positions; `id` is a `[B,T]` integer array;
`E` is a `[V,D]` token embedding table; `P` is a `[Tmax,D]` learned position
table; `x` is `[B,T,D]`. `B` is batch size, `T` sequence length, `V`
vocabulary size, `D` hidden width and `Tmax` the context limit. This gives
each ID a learned vector and tells the model where it occurs. If an
embedding is `[1,2]` and position vector is `[.1,-.2]`, their sum is
`[1.1,1.8]`. Implemented at the start of `TinyTransformer.forward`.
Without position information, attention cannot distinguish many reordered
sequences. Pretrained models may use rotary positions instead; do not replace
their positional scheme with the tiny model's learned table.

## 6. Attention from a single weighted average

Imagine each current token asking which earlier tokens contain useful
information. A query expresses what it wants. A key describes what another
position offers. A value is the information retrieved. All three are learned
linear projections of the current hidden vectors.

**Formula: Q/K/V and scaled attention.** `Q=X Wq.T`, `K=X Wk.T`,
`Vv=X Wv.T`; `S=Q K.T/sqrt(dh)`; `A=softmax(S+M)`;
`C=A Vv`. Here `X` is the hidden sequence, `Wq/Wk/Wv` projection weights,
`dh` the per-head feature width, `S` pairwise scores, `M` an additive mask,
`A` attention probabilities and `C` retrieved context. `Vv` denotes values,
not the vocabulary size. For one head the sequence matrices are `[T,dh]`;
`S` and `A` are `[T,T]`. The square-root scale limits growth of dot-product
variance as head width increases. A query `[1,0]` and keys `[1,0],[0,1]`
at width two yield scores `[.7071,0]`; softmax gives approximately
`[.6698,.3302]`. With values `[2,0],[0,4]`, the context is approximately
`[1.3396,1.3208]`. `CausalSelfAttention.forward` computes these steps
directly. Mixing up key transpose or the softmax axis makes a different
operation, even if some shapes happen to work.

**Formula: softmax.** `p_i=exp(s_i)/sum_j exp(s_j)`, where `s_i` is a
score and `p_i` its nonnegative probability. The index `j` ranges over
candidate keys or vocabulary entries. Probabilities sum to one. Equal
scores `[0,0]` yield `[.5,.5]`. Adding a common constant leaves the result
unchanged; implementations use this for numerical stability. Tiny attention
uses `F.softmax(scores.float(), dim=-1)` and casts probabilities back.
Normalizing along the head axis would mix heads instead of positions.

**Formula: causal mask.** `M[t,u]=0` for `u<=t` and negative infinity
otherwise. `t` is a query position and `u` a key position. A current token
must not read future answers during next-token training. For three tokens,
allowed rows are `[1,0,0]`, `[1,1,0]`, `[1,1,1]`. Masked logits receive
zero softmax probability. Padding keys are additionally hidden using
`attention_mask`. The tiny model requires nonempty right-padded sequences,
avoiding all-masked rows. `test_causal_mask_no_future_leakage_and_generation`
changes future tokens and requires earlier logits to be identical.

Multi-head attention divides feature coordinates into parallel attention
systems. In the tiny model, `[B,T,D]` becomes `[B,H,T,D/H]`, where `H`
is the number of heads. Each head computes its own weighted averages; the
results are concatenated and projected back to width `D`. A four-head,
48-wide configuration has head width 12. Heads need not learn human-labelled
roles. Pretrained Qwen/SmolLM models may use grouped-query attention, where
key/value projections have fewer output channels than queries. Injection
uses actual linear dimensions rather than assuming all projections are square.

## 7. Residuals, layer normalization and feed-forward blocks

**Formula: residual connection.** `Y=X+f(X)`. `X` is a hidden tensor and
`f` a shape-preserving sublayer; `Y` is the updated tensor. If a sublayer
returns `[.1,-.2]` for input `[1,2]`, the residual output is `[1.1,1.8]`.
The identity path makes it possible to learn a correction to useful existing
representations. Tiny blocks use a pre-normalized attention residual and a
pre-normalized feed-forward residual. Removing a residual changes both
information flow and gradient propagation.

**Formula: layer norm.** `LN(x)=gamma*(x-mu)/sqrt(var+epsilon)+beta`.
`x` is one token's `D` features; `mu` their mean; `var` their population
variance; `gamma,beta` learned per-feature scale and bias; `epsilon` a small
positive stabilizer. It normalizes each token independently rather than
mixing examples. For `[1,3]`, mean is 2, variance is 1, so with unit scale,
zero bias and negligible epsilon the output is `[-1,1]`. `nn.LayerNorm`
implements this in each tiny block. Normalizing the time axis accidentally
couples positions and can invalidate causal behavior.

**Formula: feed-forward network.** `FFN(x)=Wdown*GELU(Wup*x+bup)+bdown`.
`x` has width `D`; `Wup` maps `D` to `4D`; `Wdown` maps `4D` back to
`D`; `bup,bdown` are biases. This applies nonlinear feature transformations
independently to each token. A one-dimensional teaching analogue is
`Wup=2,Wdown=3,biases=0,x=1`, giving `3*GELU(2)≈5.8635`.
In `TransformerBlock.forward`, the up projection produces `[B,T,4D]`
before down projection. Pretrained Llama-style networks may use an
additional multiplicative gate; the explicit `gate` role is included in
feed-forward placement. Omitting gates in a declared all-FFN ablation
would make the placement description misleading.

## 8. Next-token prediction and causal loss

A language-model head maps each hidden vector to vocabulary logits.
Logits are unconstrained scores. Their softmax is the probability of the
next token. Training teaches the model to assign high probability to the
observed next token, using earlier tokens as context.

**Formula: causal cross-entropy.**
`L=-(1/N) sum_(b,t in valid) log p_theta(id[b,t+1] | id[b,0:t+1])`.
`theta` is the set of trainable parameters; `b` indexes sequences; `t`
indexes predictor positions; `id[b,0:t+1]` includes positions zero through
`t`; `N` counts valid target tokens. `p_theta` comes from softmax logits.
The negative log turns multiplying probabilities into adding penalties.
If a target has probability `.25`, its loss is `-log(.25)≈1.3863`.
`causal_loss_sum` uses logits `[:, :-1]` and labels `[:, 1:]`. The model
must predict the next ID, not copy the ID at the same position. A double
shift or missing shift trains the wrong task. The trainer recomputes loss
from logits explicitly rather than depending on architecture loss reductions.

Labels use `-100` for ignored targets. For a prompt followed by answer
tokens, only answer positions are supervised. The prompt still participates
in attention and gradients: ignored loss does not mean ignored input.
Padding labels are masked by sequence length, not by `id==pad_id`.
This matters when a tokenizer uses the same ID for EOS and padding.

**Formula: perplexity.** `PPL=exp(L)`, where `L` is mean natural-log
cross-entropy per valid target token. For `L=log(4)`, perplexity is 4,
roughly the effective uncertainty of four equally plausible alternatives.
`evaluate_loss` computes a token-weighted mean before exponentiating.
On SMS, this is answer-label conditional perplexity, not perplexity of the
whole message. Comparing it to general corpus perplexity is invalid.

Autoregressive generation repeatedly chooses a next token, appends it, and
runs the model again. Greedy decoding picks the maximum score. Sampling
uses a probability distribution and a temperature controlling sharpness.
`TinyTransformer.generate` supports both and truncates context to its
position-table limit. Its simple implementation recomputes the full prefix;
it has no key/value cache and is not intended as a production decoder.

## 9. Pretraining, full fine-tuning and PEFT

Pretraining learns broad language structure using large corpora. Fine-tuning
adapts a pretrained function to a narrower task. Starting a tiny model from
random weights is training, but does not establish adaptation of pretrained
knowledge. Full fine-tuning updates all model parameters, including embeddings,
norms and the language-model head. PEFT updates a smaller chosen set.

Freezing a parameter prevents its gradient and optimizer state. It does not
eliminate forward computations or all activation storage. Adapters at earlier
layers still need gradients through later frozen layers. Consequently the
fraction of trainable parameters is not the fraction of total training memory.
Full fine-tuning also updates parameter families adapters do not, so equal
step count is a useful but incomplete fairness definition.

## 10. Rank and LoRA

Rank counts the independent directions a matrix can transmit. Columns
`[1,2]` and `[2,4]` point in the same direction, so a matrix containing only
those columns has rank one. Low rank is a restriction on a matrix's update,
not on the rank of the entire pretrained model.

**Formula: LoRA.** `W_eff=W0+s*B*A`, with `s=alpha/r`.
`W0` is a frozen `[d_out,d_in]` pretrained weight; `A` is trainable
`[r,d_in]`; `B` is trainable `[d_out,r]`; `r` is adapter rank; `alpha`
is positive scaling; `W_eff` is the effective weight. The product has rank
at most `r`. Instead of training `d_out*d_in` values, train
`r*(d_in+d_out)` values. For a `100×100` matrix and rank four, that is
800 rather than 10,000. The base still exists in memory.

The forward derivation distributes a matrix product:
`W_eff*x=W0*x+s*B*(A*x)`. It lets the adapter use two small projections
without materializing a dense update. For `W0=I_2,A=[[1,2]],B=[[0],[0]],
alpha=2,r=1`, the initial update is exactly zero. If B later becomes
`[[1],[-1]]`, the update matrix is `[[2,4],[-2,-4]]`. For input `[1,0]`,
the output changes from `[1,0]` to `[3,-2]` before bias.

`LoRALinear.__init__`, `delta_weight` and `forward` implement this recipe.
The zero B initialization preserves the pretrained function. Initializing
both factors to zero blocks both gradients. Random A plus zero B lets B
learn first; after B changes, A can learn too. Kaiming-uniform A follows
the reference Linear implementation; Gaussian A is a named alternative.
The exact initial distributions matter: Kaiming with this layer's setting is
uniform on `[-1/sqrt(d_in),1/sqrt(d_in)]`, with standard deviation
`1/sqrt(3*d_in)`. Here `d_in` is input width; the variance of a uniform
distribution on `[-a,a]` is `a^2/3`. The Gaussian alternative uses mean zero
and standard deviation `1/r`, where r is adapter rank. At width 576 and rank
four these standard deviations are about 0.0241 and 0.25. This is a large
scale difference, not just a change in distribution shape. Both initial
complete functions are identical because B is zero, but their gradients and
early effective update sizes differ. These settings appear in
`LoRALinear.__init__`. A fair investigation of subspace alone would also
control factor scale, rather than attributing all convergence changes to SVD.
Alpha is fixed in the rank grids, so `alpha/r` changes with rank. This
matches the stated scale convention but means a rank study also changes
factorization dynamics. It is not a pure comparison of model capacity.

`r=0` is a frozen no-op wrapper in this implementation. `enabled=False`
evaluates the original pretrained weight. Merging computes an inference
cache once; repeated merge calls do not add the update twice. Training
invalidates the cache automatically. Exporting through `restore_linear`
can retain adaptation or restore the original function. This deliberately
differs from in-place subtractive unmerge in the reference library: frozen
storage is never rounded by repeated add/subtract cycles.

## 11. DoRA: separate length and direction

A vector can change how long it is and where it points. DoRA lets each
output channel learn a magnitude separately from a low-rank direction change.
Its effective update generally is not itself rank `r` after normalization.

**Formula: DoRA.** For output row `j`,
`v_j=W0[j,:]+s*(B*A)[j,:]`, `m0_j=||W0[j,:]||2`,
`W_eff[j,:]=m_j*v_j/max(||v_j||2,epsilon)`.
Here `j` indexes output channels, `v_j` is the updated direction vector,
`m_j` a trainable scalar magnitude, `m0_j` its initialization, and `epsilon`
a positive numerical floor. Other symbols retain their LoRA definitions.
The Euclidean norm `||v||2=sqrt(sum_i v_i^2)` is vector length. A row
`[3,4]` has length five and direction `[.6,.8]`. At zero directional
update, magnitude five reconstructs `[3,4]`; magnitude ten produces `[6,8]`.
Bias is added afterward, so a bias of one is not doubled when magnitude doubles.

The paper writes column norms in input-by-output layout. PyTorch stores
output-by-input weights, so `dim=1` is essential. Using `dim=0` normalizes
input channels and is a different method. `DoRALinear._row_norm` sums in
float32 for half inputs; `effective_weight` applies the ratio; `forward`
adds the original frozen bias exactly once. Near a zero row the norm floor
avoids division by zero. At exactly zero original row, initial magnitude
is zero: finite behavior is guaranteed, meaningful direction is not.

**Formula: normalization derivative.** Away from the floor, for one
row `w=m*v/||v||`, `dL/dv=(m/||v||)*(I-u*u.T)*g`, where `u=v/||v||`
is the unit direction, `g=dL/dw` the output-weight gradient, `I` the
identity matrix and `L` the loss. The matrix `I-u*u.T` removes the part of
the gradient parallel to the current direction. This follows by applying
the product rule to `v/||v||` and differentiating `||v||` as `v/||v||`.
For `v=[3,4],m=5,g=[1,0]`, the full derivative is `[.64,-.48]`.
If the norm is detached, the derivative instead becomes `(m/||v||)*g`,
or `[1,0]` in this example. The forwards are identical; the gradients differ.

The default `detach_norm=True` follows the paper's overhead-reduction
variant and NVIDIA's `dora_simple` behavior. False implements full
normalization differentiation. A finite-difference derivative of the
mathematical forward cannot validate a deliberately detached denominator's
surrogate derivative; compare it against a matching detached reference.
Both variants are tested row by row. Adapter dropout is rejected here to
keep the exact dense no-dropout formula visible and its merge unambiguous.

## 12. SVD and PiSSA from fundamentals

SVD describes a linear map using orthogonal input directions, nonnegative
strengths and orthogonal output directions. Orthogonal vectors are at right
angles; orthonormal means additionally length one. A rotation followed by
axis-wise stretching and another rotation can represent any real matrix.

**Formula: singular value decomposition.** `W=U*diag(sigma)*V.T`.
`W` is `[d_out,d_in]`; `k=min(d_out,d_in)`; economy-size `U` is
`[d_out,k]`, `sigma` has `k` descending nonnegative entries and `V.T`
is `[k,d_in]`. Each column of U and corresponding row of V.T define a
rank-one contribution, scaled by its singular value. `diag` puts values
on a square matrix's diagonal. For `W=[[4,0],[0,1]]`, U and V are
identity and singular values are `[4,1]`. The rank-one principal component
is `[[4,0],[0,0]]`; the residual is `[[0,0],[0,1]]`.

**Formula: PiSSA with scale correction.** Set
`B0=U_r*diag(sqrt(sigma_r/s))`,
`A0=diag(sqrt(sigma_r/s))*V_r.T`,
`R=W0-s*B0*A0`, and train `W_eff=R+s*B*A` starting from A0/B0.
Subscript `r` selects the largest r singular components; `s=alpha/r` is
positive. `R` is the frozen residual. `A,B` are the current factors;
`A0,B0` are their initial values. Multiplying the two square-root factors
restores `sigma_r/s`; multiplying by s restores the original principal
component. Adding R therefore reconstructs W0 at initialization.

For the diagonal example at rank one and scale two, B0 is `[sqrt(2),0].T`,
A0 is `[sqrt(2),0]`, and `2*B0*A0=[[4,0],[0,0]]`. Adding residual
`[[0,0],[0,1]]` restores the original matrix. If the residual were left as
W0, the initial effective matrix would incorrectly be `[[8,0],[0,1]]`.
This is the central PiSSA correctness condition, not merely a good
initialization heuristic. `PiSSALinear.__init__` implements these steps;
reconstruction and initial-function tests cover several ranks and both SVD modes.

LoRA starts with a zero update around W0. PiSSA starts with nonzero trainable
factors and a changed frozen residual, while preserving the same total
function. Both factors can receive gradients immediately. The initial
subspace is informed by pretrained weights, but that does not prove the
subspace is optimal for SMS or another task. This is an empirical question.

Exact SVD decomposes the whole matrix even if only a few singular vectors
are retained. Its temporary memory and initialization time matter.
`torch.svd_lowrank` approximates the leading subspace using randomized
iterations; `svd_iterations` controls extra refinement. Because the
implementation subtracts its actual initialized component, it preserves
the original function even when the subspace approximation is imperfect.
Half precision adds rounding error; initialize in float32 where possible.
Zero dropout avoids randomly deleting the principal component during the
first training forward, which would violate initial-function preservation.

An adapter-only PiSSA file must include enough information to reconstruct
the residual on a clean pretrained model. This project saves initial factors
as well as trained factors. On load it undoes the new constructor's
decomposition and subtracts the saved initial component. Random SVD signs
and RNG choices therefore do not silently change the residual. The file is
larger than a two-factor LoRA file; that is part of the measured storage cost.

## 13. Adapter injection and gradient flow

Injection replaces selected `nn.Linear` children with wrappers. It first
finds explicit projection suffixes, constructs replacements, freezes all
other parameters and registers the wrappers at their original module paths.
Supported roles are query, key, value, attention output, FFN up/down/gate.
Attention, FFN and combined groups are inspectable in `ROLE_NAMES`/`GROUPS`.

Do not identify every linear layer as equivalent. An output vocabulary head
can be huge and its meaning differs from attention projections. A GPT-2
`Conv1D` stores weights differently; fused QKV layers require slicing rules.
This version rejects architectures with no supported matching linear roles.
It supports ordinary Llama/Qwen/OPT-style projection names and explicit
suffix lists, not arbitrary architectures by assumption.

During backward, a frozen W0 receives no `.grad`. The input to its linear
operation can still receive gradients because changing that input changes
loss. This allows an early adapter to learn through frozen later blocks.
The optimizer explicitly receives only parameters with `requires_grad=True`.
Parameter counts include adapter factors and DoRA magnitudes but exclude
buffers. Checkpoint byte count includes serialization overhead and PiSSA
initial buffers, so it is a distinct measurement.

## 14. Dataset preparation and evaluation

The raw UCI archive is hashed, parsed and normalized for duplicate detection.
Normalization collapses whitespace and case. Duplicate text is kept once;
texts with conflicting labels are removed entirely. Only then are stratified
70/15/15 splits produced with a fixed split seed. This seed is separate
from the training seed: seed studies must not silently change the task.
The archive's actual row count is measured, rather than forced to match a
catalogue's rounded or differing count.

Token truncation can make different raw messages look identical. The
pipeline therefore removes identical visible prompt-token sequences across
splits, retaining test first, validation next, training last. This operation
does not use label values. Saved encoded examples make the result auditable.
This prevents exact local contamination, not unknown pretraining overlap or
near duplicates. Such overlap cannot be ruled out for a public pretrained LM.

The prompt describes the task and ends in `Label:`. Label strings begin
with a space. Prompt and answer tokens are deliberately encoded separately
in both training and evaluation, so the same token boundary convention is
used everywhere. Message tokens are truncated with room reserved for the
longer candidate and the fixed prompt suffix. No true label is put in the
prompt. Subsets are fixed ordered prefixes after a deterministic shuffle;
10% is nested in 25%, 50% and 100% of the configured training cap.

**Formula: constrained classification.**
`score(c)=sum_j log p(a_c[j] | prompt, a_c[:j])`,
`prediction=argmax_c score(c)`. Here `c` is a candidate class (legitimate
or spam), `a_c` its tokenized answer string and `j` an answer-token index.
The model scores both alternatives without unconstrained generation. If
candidate A has single-token probability `.6` and B `.4`, log scores are
about `-.511` and `-.916`, so A wins. For multiple tokens, conditional
probabilities multiply, hence logs add. Longer labels can be penalized by
this sum: the rule is fixed across methods and recorded, not claimed to be
calibrated class probabilities. `evaluate_classification` implements it.

**Formula: accuracy.** `accuracy=correct/Nexamples`. `correct` counts
matching predictions and `Nexamples` is held-out example count. Three
correct of four gives `.75`. A majority-class predictor can have high
accuracy on imbalanced SMS, so macro F1 and confusion matrices are required.

**Formula: precision, recall and F1.** For class c, `P_c=TP_c/(TP_c+FP_c)`,
`R_c=TP_c/(TP_c+FN_c)`, `F1_c=2*P_c*R_c/(P_c+R_c)`, and
`macroF1=(F1_0+F1_1)/2`. `TP` counts correct predictions of c; `FP`
incorrect predictions as c; `FN` missed true c examples. Undefined ratios
are set to zero. If `TP=2,FP=1,FN=1`, P and R are both 2/3 and F1 is
2/3. The confusion matrix uses true classes as rows and predicted classes
as columns. `metrics.py` delegates these conventions to scikit-learn.
Reporting a single weighted F1 as macro F1 would hide minority failures.

## 15. A training step and fair accumulation

Each step samples a deterministic window of microbatches, runs forwards,
computes supervised token losses, runs backwards, clips gradients, updates
the optimizer and advances the scheduler. Validation never calls optimizer
step. Test evaluation occurs at the end. No computation graph is stored
inside a metric history: `.item()` converts loss to a Python number.

**Formula: accumulation.** `Lwindow=(sum_k Sk)/(sum_k Nk)`, where k
indexes microbatches, `Sk` is summed supervised token cross-entropy in that
microbatch, and `Nk` its valid target-token count. Backward on each
`Sk/Nwindow` gives the same gradient as backward once on the concatenated
batch, up to floating-point summation, provided dropout and other
batch-dependent behavior match. Example: one microbatch has sum 6 over
3 tokens; another sum 4 over 1 token. Correct loss is `10/4=2.5`.
Averaging their means gives `(2+4)/2=3`, a different objective.
`Trainer.fit` computes the denominator before backward. The unequal-token
accumulation test compares actual parameter gradients with a full batch.

**Formula: gradient clipping.** `g_clipped=g*min(1,c/(||g||2+epsilon))`.
`g` denotes all trainable gradients concatenated conceptually; `c` is the
maximum norm and epsilon avoids division by zero. A gradient `[3,4]` has
norm 5; clipping at 1 produces `[.6,.8]`. Clipping prevents unusually large
updates; it is applied after accumulation and unscaling. Clipping each
microbatch separately would change the objective's effective direction.
`clip_grad_norm_` returns the preclip norm, which is what the run logs.

Schedulers change learning rate with optimizer steps, not microsteps.
Constant, linear-decay and cosine-decay schedules are implemented; warmup
uses smaller early multipliers. Resume must restore the scheduler because
restarting warmup changes the experiment even with identical weights.
Early stopping monitors validation cross-entropy, with a configured patience.
Default matched-step studies disable early stopping and use final weights.

## 16. Reproducibility and checkpoints

Reproducibility has levels: same data/configuration, statistically similar
results, and bitwise-identical trajectories. This repository tests exact CPU
resume with dropout, uneven final batches and an epoch boundary. It does
not promise bitwise equivalence across PyTorch versions, GPUs or CPU/GPU.

Training checkpoints contain the model, optimizer moments, scheduler,
mixed-precision scaler, all RNG states, epoch permutation position, history,
best validation score and configuration identity. They are written at
optimizer boundaries and atomically renamed. They use trusted local Python
checkpoint loading because NumPy RNG objects are included. Never load an
untrusted arbitrary resume file. Adapter-only files use weights-only loading.

Model downloads resolve `main` to a concrete commit and write a resolved
configuration. Reproduction should use that commit, the raw archive hash,
encoded split hash, package lock and fixed seeds. A mutable model tag is a
convenience for first preparation, not sufficient research provenance.
Resuming with a different dataset or learning rate raises an explicit error.

## 17. Experimental controls and uncertainty

A baseline is a reference procedure: no fine-tuning answers whether any
update helps; full fine-tuning measures unrestricted adaptation. LoRA, DoRA
and PiSSA test restricted parameterizations. An ablation deliberately
changes one named factor, such as rank or placement, while holding others
fixed. A grid definition is not evidence until its runs finish.

Matched steps means equal optimizer updates under the same microbatch and
accumulation policy. It does not mean equal runtime, number of trainable
parameters or memory. Rank/placement change parameter counts; fixed-alpha
rank grids change scale. Dataset-size grids at fixed steps reuse smaller
datasets more often, so they measure data exposure under a fixed update
budget rather than matched epochs. The report must state these choices.

**Formula: sample variability.** For seed results `x_1,...,x_n`,
`mean=(sum_i x_i)/n`, `sampleSD=sqrt(sum_i (x_i-mean)^2/(n-1))` for n>1.
`x_i` is one run's metric; n is the number of independent training seeds.
For `[.6,.7,.8]`, mean is `.7` and sample SD is `.1`. One run has no
estimable seed SD, not zero SD. `analysis.py` writes blank standard
deviations for single observations. Seed variance is not the same as
uncertainty from having a small held-out test set. Three seeds are still
weak evidence; repeated testing and tuned hyperparameters add selection bias.

Do not combine runs differing in model commit, data hash, initialization,
step count, precision or learning rate into a seed statistic. Paired seed
comparisons are more informative than comparing unrelated runs. Negative
or inconclusive results are useful: they show that a proposed advantage
did not establish itself under this budget. They do not refute a paper's
experiments on different tasks or much larger models.

## 18. Memory and resource planning

**Formula: static training-memory estimate.** For float32 parameters with
AdamW, `Mstatic≈4*Ptotal+12*Ptrain` bytes. `Ptotal` counts model and adapter
parameter elements; `Ptrain` counts trainable elements. The first term
stores weights, the second adds gradients and two moment arrays. A model
with 100M total and 1M trainable parameters needs approximately 412M bytes
before activations, allocator overhead, temporary updates and buffers. Full
fine-tuning with all 100M trainable needs approximately 1.6 billion bytes.
`runner.run` applies a conservative static VRAM gate, but it cannot guarantee
fit. Real peak allocation must still be measured during training.

Attention score storage grows with sequence length squared in the explicit
attention implementation. DoRA's dense direction and normalization need
temporary storage. PiSSA SVD has initialization costs. Gradient checkpointing
trades recomputation for smaller saved activation memory. Accumulation permits
a smaller microbatch but does not make the base checkpoint smaller.
Autocast is not quantization; FP32 weights may remain even when compute uses
FP16. The driver's supported CUDA version is not the installed PyTorch build.

CPU RSS is sampled after steps and is not a true process lifetime peak.
CUDA peak allocated bytes describe PyTorch allocations on that device,
not all GPU memory used by Windows or other processes. Missing GPU memory
measurements are null, never zero. Runtime includes training, interval
validation and interval checkpoint I/O; initialization is recorded separately.
Final checkpoint/export and final evaluation are outside the training timer.
Step timestamps precede their own validation, so convergence-time plots use
validation-start timestamps rather than completed-validation timestamps.
Use these measurement definitions when interpreting efficiency charts.

## 19. Quantization, QLoRA and research extensions

Quantization represents weights with fewer bits plus scales or codebooks.
QLoRA combines a quantized frozen base with low-rank trainable adapters;
it is a separate method with additional numerical and memory behavior.
This repository does not quantize base weights or claim QLoRA support.
Likewise, PiSSA is not QPiSSA merely because its factors use less precision.

A possible extension assigns different ranks to layers under a fixed
parameter budget. Pretrained singular-value energy is one transparent
signal, but high energy need not mean high task importance. First establish
uniform-rank baselines. Then preregister an allocation rule, account for
different matrix dimensions, enforce the budget, and report failures or
negative outcomes. Do not infer task-specific importance from spectra alone.

## 20. Guided study exercises

1. Calculate the shapes of every tensor in a two-head, width-16, length-8
   attention forward. Verify them at a debugger breakpoint.
2. Predict which LoRA factor has zero gradient at the first step. Inspect
   `.grad` before and after two SGD steps.
3. Change a DoRA reference norm from dim=1 to dim=0. Explain the failed
   numerical example using output-channel versus input-channel meaning.
4. On the diagonal PiSSA example, omit residual subtraction by hand. Explain
   why the initial function changes although SVD was calculated correctly.
5. Compare accumulated gradients with a concatenated batch whose target
   counts differ. Explain why equal averaging of microbatch losses fails.
6. Interrupt a tiny training run at an optimizer boundary and resume it.
   Explain each checkpoint field needed for identical dropout and ordering.
7. Read the final report and identify one claim supported by local runs,
   one author claim, and one untested hypothesis.
8. Design a follow-up study with larger validation/test sets and a stopping
   rule decided before inspecting test outcomes. State what it can establish.

You understand the project when you can explain what is frozen, which
tensor receives a gradient, how a label is scored, and which claims the
available measurements can actually support.
