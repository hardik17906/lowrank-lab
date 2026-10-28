# Language-model training and supervised completions

Read guide sections 8, 14 and 15. A causal LM assigns probabilities to next
tokens. Training on a classification task can still use that objective by
representing labels as answer strings. Here the answer candidates are fixed,
and the loss supervises only their token positions.

Inspect `encode_record`. Prompt IDs enter the network normally. Their label
positions are -100, which excludes those targets from CE. Attention masks
represent real sequence length, while loss masks represent supervised
positions. These masks answer different questions and must not be reused
interchangeably. An EOS token serving as padding also must not cause real
answer tokens with the same ID to be ignored accidentally.

In `causal_loss_sum`, sum reduction preserves exact target-token counts.
The trainer divides by the count across the entire accumulation window.
Use the guide's unequal-count example and compare the correct 2.5 objective
with the incorrect 3.0 average of microbatch means. This matters whenever
answers or sequence lengths differ.

At evaluation, conditional answer cross-entropy and classification accuracy
measure different behavior. A model can improve correct-answer probability
without changing which label wins, or increase confidence in a wrong label.
Macro F1 helps expose minority-class failures in imbalanced SMS.

Exercise: inspect saved candidate log probabilities for one correct and one
incorrect example. Calculate the chosen class manually. Then read the
confusion matrix using true rows/predicted columns. A model predicting every
message as legitimate can score high accuracy; determine what happens to
spam recall and macro F1. Never describe conditional answer perplexity as
whole-corpus language-model perplexity.
