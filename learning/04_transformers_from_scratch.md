# Trace the decoder from IDs to logits

Read guide sections 5–8. Open `models/tiny_transformer.py`. For a batch of
two sequences, length eight, width sixteen and two heads, write these shapes
before debugging: IDs `[2,8]`, embedding `[2,8,16]`, each split Q/K/V
`[2,2,8,8]`, scores `[2,2,8,8]`, concatenated context `[2,8,16]`,
vocabulary logits `[2,8,32]`. The equal-looking score dimensions represent
different things: query position and key position.

In attention, queries ask, keys match, values carry content. They are learned
projections, not hand-coded text meanings. Softmax converts key scores into
a weighted average for each query. Multiple heads make several such systems
in parallel; the output projection mixes their results.

The triangular mask blocks future keys. Next-token training would otherwise
allow the model to read the answer. Padding masks additionally block nonexistent
tokens. The tiny implementation checks right padding because a fully masked
query row would produce an undefined softmax. Inspect the actual test that
changes future IDs and compares earlier logits; this checks behavior, not
just the existence of a triangle tensor.

The output head predicts vocabulary scores at every position. The position
at the last prompt token predicts the first answer token. Work this out on
three prompt IDs plus two answer IDs. Draw the shifted logits and labels.
Do not shift the labels twice simply because the pretrained architecture
also offers a `.loss` attribute: the common trainer uses logits directly.

Exercise: run greedy generation from a one-token prefix after tiny training.
Explain why every generated token becomes input to the next prediction.
Then inspect the pretrained model's named modules: grouped-query attention
can have narrower key/value matrices, and rotary positions differ from the
tiny learned position table. The educational model explains the pipeline;
it is not a claim that all Transformer architectures are identical.
