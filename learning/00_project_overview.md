# Start here: what this laboratory actually investigates

A pretrained language model has already learned to predict text. Fine-tuning
changes its behavior using a smaller, more specific collection. Our public
task asks a causal language model to choose between two SMS labels. The
same pretrained checkpoint, prompt construction, data splits, evaluation
procedure and optimizer-step budget are reused across methods.

There are two separate tracks. `models/tiny_transformer.py` is a randomly
initialized educational decoder. Its repeated-pair dataset makes attention,
autograd and optimization easy to inspect. The pretrained track loads a
real Hugging Face checkpoint and uses public UCI messages. Tiny-model
learning is evidence that the pipeline works; it is not evidence of LLM
fine-tuning performance.

Read `COMPLETE_LEARNING_GUIDE.md` for the mathematical foundation, then
`PROJECT_CODE_WALKTHROUGH.md` beside the source. Chapters 01–11 provide
smaller study sessions. `reports/RESEARCH_REPORT.md` separates measured
results from unexecuted configurations. `PROJECT_HANDOFF.md` is the tutor's
entry point and records the finished state.

Use this question whenever a tensor is confusing: what does each axis
represent? A batch axis enumerates examples; a time axis enumerates tokens;
a feature axis represents learned coordinates. A low-rank axis is neither
tokens nor classes: it is a narrow intermediate space for a weight update.

Suggested first exercise: run tests, open `configs/smoke.yaml`, and follow
one batch through `TinyTransformer.forward`. Do not start by downloading a
billion-parameter model. The smoke configuration is enough to understand
the mechanism; the pretrained study answers a different empirical question.
