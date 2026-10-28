import math
import torch
from ..training.losses import causal_loss_sum
from ..data.preprocessing import collate, encode_record
from .metrics import classification_metrics


@torch.no_grad()
def evaluate_loss(model, records, batch_size=4, pad_token_id=0):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, tokens = 0.0, 0
    try:
        for start in range(0, len(records), batch_size):
            batch = {
                k: v.to(device)
                for k, v in collate(records[start : start + batch_size], pad_token_id).items()
            }
            loss, count = causal_loss_sum(model(**batch).logits, batch["labels"])
            total += loss.item()
            tokens += count
        if not tokens:
            raise ValueError("Empty evaluation set")
        ce = total / tokens
        return {
            "cross_entropy": ce,
            "perplexity": math.exp(ce) if ce < 700 else None,
            "supervised_tokens": tokens,
        }
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate_classification(model, raw_records, tokenizer, max_length, batch_size=4):
    """Constrained classification by SUM log probability of each candidate label.

    Candidate strings are separately tokenized identically in train and eval.
    No answer text appears in the prompt. Both labels get the same prompt
    truncation budget. Padding/EOS equality never masks a real target token.
    """
    was_training = model.training
    model.eval()
    predictions, labels, scores = [], [], []
    device = next(model.parameters()).device
    try:
        for start in range(0, len(raw_records), batch_size):
            rows = raw_records[start : start + batch_size]
            candidates = [
                encode_record(tokenizer, row, max_length, label) for row in rows for label in [0, 1]
            ]
            batch = {
                k: v.to(device) for k, v in collate(candidates, tokenizer.pad_token_id).items()
            }
            logits = model(**batch).logits[:, :-1].float()
            targets = batch["labels"][:, 1:]
            mask = targets != -100
            token_logp = (
                logits.log_softmax(-1).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            )
            candidate_scores = (token_logp * mask).sum(-1).reshape(-1, 2)
            predictions.extend(candidate_scores.argmax(-1).cpu().tolist())
            scores.extend(candidate_scores.cpu().tolist())
            labels.extend(row["label"] for row in rows)
        return {
            **classification_metrics(labels, predictions),
            "predictions": predictions,
            "labels": labels,
            "candidate_log_probabilities": scores,
        }
    finally:
        model.train(was_training)
