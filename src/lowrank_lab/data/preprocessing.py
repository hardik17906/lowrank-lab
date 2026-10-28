import hashlib
import json
import random
import torch

LABELS = [" legitimate", " spam"]


def prompt_tokens(tokenizer, text, max_length):
    """Truncate message tokens while preserving instruction and answer delimiter."""
    prefix = tokenizer.encode(
        "Classify the SMS as legitimate or spam.\nSMS: ", add_special_tokens=False
    )
    suffix = tokenizer.encode("\nLabel:", add_special_tokens=False)
    reserve = max(len(tokenizer.encode(s, add_special_tokens=False)) for s in LABELS)
    room = max_length - len(prefix) - len(suffix) - reserve
    if room < 1:
        raise ValueError("max_length too small for task prompt and labels")
    return prefix + tokenizer.encode(text, add_special_tokens=False)[:room] + suffix


def encode_record(tokenizer, row, max_length, label=None):
    prompt = prompt_tokens(tokenizer, row["text"], max_length)
    answer = tokenizer.encode(
        LABELS[row["label"] if label is None else label], add_special_tokens=False
    )
    return {
        "input_ids": prompt + answer,
        "labels": [-100] * len(prompt) + answer,
        "class_label": row["label"],
    }


def remove_token_collisions(splits, tokenizer, max_length):
    """Remove all but one identical visible prompt, across and within splits.

    Keep test before validation before train, independently of labels.
    This protects held-out evaluation even after truncation. Raw disjointness
    cannot guarantee tokenized disjointness. Report removed counts explicitly.
    """
    seen, output, removed = set(), {}, {}
    for split in ["test", "validation", "train"]:
        output[split] = []
        for row in splits[split]:
            key = tuple(prompt_tokens(tokenizer, row["text"], max_length))
            if key not in seen:
                seen.add(key)
                output[split].append(row)
        removed[split] = len(splits[split]) - len(output[split])
    return output, removed


def collate(records, pad_token_id=0):
    length = max(len(r["input_ids"]) for r in records)
    ids, labels, masks = [], [], []
    for row in records:
        n = len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_token_id] * (length - n))
        labels.append(row["labels"] + [-100] * (length - n))
        masks.append([1] * n + [0] * (length - n))
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(masks),
    }


def synthetic_splits(seed=42, length=24):
    """Educational repeated-pair LM; disjoint pairs, never a downstream benchmark."""
    pairs = [(a, b) for a in range(2, 32) for b in range(2, 32) if a != b]
    random.Random(seed).shuffle(pairs)

    def rows(items):
        result = []
        for a, b in items:
            tokens = [1] + ([a, b] * length)[: length - 1]
            result.append({"input_ids": tokens, "labels": list(tokens)})
        return result

    return {
        "train": rows(pairs[:128]),
        "validation": rows(pairs[128:160]),
        "test": rows(pairs[160:192]),
    }


def split_fingerprint(splits):
    encoded = json.dumps(splits, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()
