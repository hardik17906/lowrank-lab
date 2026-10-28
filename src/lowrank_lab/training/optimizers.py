import math
import torch


def make_optimizer(model, config):
    parameters = [p for p in model.parameters() if p.requires_grad]
    name = config.get("optimizer", "adamw")
    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.0)
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.0)
        )
    raise ValueError(f"Unsupported optimizer {name}")


def make_scheduler(optimizer, config):
    total = config["max_steps"]
    warmup = config.get("warmup_steps", 0)
    schedule = config.get("scheduler", "cosine")
    if schedule not in {"constant", "linear", "cosine"}:
        raise ValueError(f"Unknown scheduler {schedule}")

    def multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, max(0, (step - warmup) / max(1, total - warmup)))
        return {
            "constant": 1.0,
            "linear": 1 - progress,
            "cosine": (1 + math.cos(math.pi * progress)) / 2,
        }[schedule]

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
