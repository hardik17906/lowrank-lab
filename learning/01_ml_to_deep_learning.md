# From familiar machine learning to learned representations

In a simple tabular classifier, you choose features such as message length
or number of currency symbols. A neural network can learn intermediate
features from less structured inputs. This changes where the work happens:
instead of manually deciding every useful feature, you choose an architecture,
loss, data and optimization procedure that can discover useful combinations.

The basic supervised-learning contract is unchanged: inputs, targets,
predictions, loss and held-out evaluation. More parameters do not remove the
need for a clean test set. Text models introduce additional details: converting
strings to token IDs, modeling context, predicting token sequences and
controlling the boundary between prompt and answer.

Read guide sections 1–3. Start with the affine example and calculate its two
outputs. A matrix is a collection of learned linear questions. A network
adds nonlinear steps so later questions can depend on earlier combinations.
Depth means multiple learned transformations, not merely a large matrix.

Full training and fine-tuning share the same mechanical loop. Their starting
functions differ. The tiny model starts randomly and learns repeated pairs.
The public-task model starts from a checkpoint already trained on language.
Always ask whether a reported improvement starts from pretrained knowledge.

Exercise: list the input, target, model output and loss for ordinary binary
logistic regression. Then list them for this repository's SMS task. The
semantic target is binary in both, but the causal LM produces vocabulary
logits and scores answer strings. It does not acquire a separate binary head.

Read `data/preprocessing.py` next. Explain why a prompt can affect learning
even though its labels are ignored. Then read the no-fine-tuning result:
it establishes the pretrained model's initial task behavior. The random tiny
baseline establishes a different fact and must not be mixed into that claim.
