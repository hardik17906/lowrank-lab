from torch.nn import functional as F


def causal_loss_sum(logits, labels):
    """Sum over valid next-token targets; normalize ONCE per accumulation window.

    logits [B,T,V] at t predict labels [B,T] at t+1. Prompt/padding labels=-100.
    Returns (scalar loss sum, Python valid-token count).
    """
    shifted = labels[:, 1:]
    count = int((shifted != -100).sum().item())
    if not count:
        raise ValueError("Batch has no supervised next-token labels")
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        shifted.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, count
