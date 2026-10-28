# Build intuition for a network's computation

Read guide sections 1–3 and 7. First think of a linear layer as outputs made
from weighted input sums. Apply its weights row by row. Then introduce a
nonlinearity. Multiple linear maps without nonlinearities collapse to a single
linear map; this is why the FFN's GELU belongs between its projections.

The tiny FFN temporarily expands 48 features to 192. Expansion provides room
for nonlinear combinations, then down projection returns to the residual
width. It acts independently at each token. Attention is what mixes token
positions; FFN is what transforms features within a position.

Residual connections add a sublayer's correction to its input. Layer norm
controls feature scale per token. These are distinct operations with distinct
axes. Normalizing across batch or time would be a different model. Use the
guide's two-feature layer-norm example to check the axis by hand.

Regularization and optimization are also distinct. Weight decay shrinks
trainable parameters; gradient clipping limits unusually large update
directions; dropout changes training forwards; early stopping chooses when
to stop using validation behavior. None fixes contaminated test data.

Exercise: run the tiny loss-reduction test, then inspect its train and
validation examples. Why can memorizing the 16 training pairs lower train
loss without proving robust language understanding? Describe what held-out
pairs measure. Next compare full fine-tuning with adapters: full training
can change embeddings and the output head, whereas attention-only adapters
cannot. A larger gain by full training on a random model is therefore not
surprising and does not settle the pretrained PEFT question.
