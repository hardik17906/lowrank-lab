import copy
import pytest
import torch
from lowrank_lab.data.dataset_loader import deduplicate, take_subset
from lowrank_lab.data.preprocessing import (
    encode_record,
    collate,
    remove_token_collisions,
    synthetic_splits,
)
from lowrank_lab.evaluation.metrics import classification_metrics
from lowrank_lab.adapters import inject_adapters


class ToyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 31 + 1 for c in text]


def test_deduplication_conflicts_and_subset_nesting():
    clean, removed = deduplicate(
        [
            {"text": " A B ", "label": 0},
            {"text": "a  b", "label": 0},
            {"text": "conflict", "label": 0},
            {"text": "CONFLICT", "label": 1},
        ]
    )
    assert len(clean) == 1 and removed == 3
    rows = list(range(100))
    assert take_subset(rows, 100, 0.1) == take_subset(rows, 100, 0.25)[:10]


def test_prompt_masking_padding_and_truncation_collisions():
    tokenizer = ToyTokenizer()
    row = {"text": "some SMS message " * 10, "label": 1}
    result = encode_record(tokenizer, row, 90)
    assert len(result["input_ids"]) <= 90
    assert result["labels"].count(-100) > 0
    assert all(t != -100 for t in result["labels"][-5:])
    batch = collate([result, encode_record(tokenizer, {"text": "a", "label": 0}, 90)])
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    splits = {"train": [row], "validation": [row], "test": [row]}
    clean, removed = remove_token_collisions(splits, tokenizer, 90)
    assert [len(clean[k]) for k in ["train", "validation", "test"]] == [0, 0, 1]


def test_synthetic_splits_disjoint_and_metrics():
    splits = synthetic_splits()
    sets = [set(tuple(r["input_ids"]) for r in rows) for rows in splits.values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
def test_real_transformers_llama_architecture_offline(method):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(config).eval()
    pristine = copy.deepcopy(model)
    x = torch.randint(0, 32, (2, 8))
    names = inject_adapters(model, method, rank=2, targets="combined")
    assert len(names) == 7
    torch.testing.assert_close(model(x).logits, pristine(x).logits, atol=1e-6, rtol=1e-5)
    loss = model(x, labels=x).loss
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad
    )
