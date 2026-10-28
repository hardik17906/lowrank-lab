import copy
import pytest
import torch
from lowrank_lab.models import TinyTransformer, TinyConfig
from lowrank_lab.data.preprocessing import synthetic_splits, collate
from lowrank_lab.training.trainer import Trainer
from lowrank_lab.training.losses import causal_loss_sum
from lowrank_lab.evaluation.benchmark import evaluate_loss
from lowrank_lab.utils.reproducibility import seed_everything


def config(**kwargs):
    return {
        "seed": 7,
        "max_steps": 4,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "learning_rate": 0.003,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "weight_decay": 0.01,
        "validation_interval": 2,
        "max_grad_norm": 1.0,
        "mixed_precision": "none",
        **kwargs,
    }


def test_causal_mask_no_future_leakage_and_generation():
    model = TinyTransformer(TinyConfig(width=16, heads=2, layers=1)).eval()
    x = torch.randint(0, 32, (2, 10))
    changed = x.clone()
    changed[:, 6:] = (changed[:, 6:] + 4) % 32
    torch.testing.assert_close(model(x).logits[:, :6], model(changed).logits[:, :6], atol=0, rtol=0)
    assert model.generate(x[:, :3], max_new_tokens=4).shape == (2, 7)


def test_gradient_accumulation_unequal_target_counts():
    model = TinyTransformer(TinyConfig(width=16, heads=2, layers=1))
    accumulated = copy.deepcopy(model)
    rows = [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, 4]},
        {"input_ids": [1, 5, 6], "labels": [-100, -100, 6]},
    ]
    batch = collate(rows)
    loss, n = causal_loss_sum(model(**batch).logits, batch["labels"])
    (loss / n).backward()
    for row in rows:
        micro = collate([row])
        micro_loss, _ = causal_loss_sum(accumulated(**micro).logits, micro["labels"])
        (micro_loss / n).backward()
    for p, q in zip(model.parameters(), accumulated.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)


def test_checkpoint_exact_resume_with_dropout_and_epoch_transition(tmp_path):
    rows = synthetic_splits(length=12)
    seed_everything(7)
    base = TinyTransformer(TinyConfig(width=16, heads=2, layers=1, dropout=0.1))
    first, second = copy.deepcopy(base), copy.deepcopy(base)
    settings = config(max_steps=5, batch_size=3)
    seed_everything(99)
    complete = Trainer(
        first, rows["train"][:11], rows["validation"][:4], settings, tmp_path / "complete"
    )
    complete.fit()
    seed_everything(99)
    interrupted = Trainer(
        second, rows["train"][:11], rows["validation"][:4], settings, tmp_path / "interrupted"
    )
    interrupted.fit(stop_after=3)
    resumed_model = TinyTransformer(TinyConfig(width=16, heads=2, layers=1, dropout=0.1))
    resumed = Trainer(
        resumed_model, rows["train"][:11], rows["validation"][:4], settings, tmp_path / "resumed"
    )
    resumed.load_checkpoint(tmp_path / "interrupted" / "checkpoint.pt")
    resumed.fit()
    for p, q in zip(first.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert [r["training_loss"] for r in complete.history] == [
        r["training_loss"] for r in resumed.history
    ]
    assert complete.scheduler.state_dict() == resumed.scheduler.state_dict()


def test_tiny_training_loss_reduction(tmp_path):
    rows = synthetic_splits(length=12)
    model = TinyTransformer(TinyConfig(width=24, heads=2, layers=1))
    subset = rows["train"][:16]
    before = evaluate_loss(model, subset)["cross_entropy"]
    trainer = Trainer(
        model,
        subset,
        rows["validation"][:4],
        config(
            max_steps=50,
            batch_size=8,
            gradient_accumulation=1,
            learning_rate=0.01,
            validation_interval=25,
        ),
        tmp_path,
    )
    trainer.fit()
    after = evaluate_loss(model, subset)["cross_entropy"]
    assert after < before * 0.65


def test_resume_rejects_different_settings_and_empty_labels(tmp_path):
    rows = synthetic_splits(length=12)
    model = TinyTransformer(TinyConfig(width=16, heads=2, layers=1))
    trainer = Trainer(model, rows["train"][:4], rows["validation"][:2], config(), tmp_path)
    trainer.fit(stop_after=1)
    other = Trainer(
        copy.deepcopy(model),
        rows["train"][:4],
        rows["validation"][:2],
        config(learning_rate=0.5),
        tmp_path / "other",
    )
    with pytest.raises(ValueError):
        other.load_checkpoint(tmp_path / "checkpoint.pt")
    with pytest.raises(ValueError):
        causal_loss_sum(torch.randn(1, 3, 32), torch.full((1, 3), -100))
