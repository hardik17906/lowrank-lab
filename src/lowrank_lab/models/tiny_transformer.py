"""A small decoder-only Transformer written for shape-by-shape inspection."""

import math
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TinyConfig:
    vocab_size: int = 32
    width: int = 48
    heads: int = 4
    layers: int = 2
    max_length: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.width % config.heads:
            raise ValueError("width must be divisible by heads")
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.q_proj = nn.Linear(config.width, config.width)
        self.k_proj = nn.Linear(config.width, config.width)
        self.v_proj = nn.Linear(config.width, config.width)
        self.o_proj = nn.Linear(config.width, config.width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        batch, length, width = x.shape

        def split(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)

        q, k, v = [split(proj(x)) for proj in (self.q_proj, self.k_proj, self.v_proj)]
        # q,k,v: [batch,heads,time,head_width]. Scores: [batch,heads,time,time].
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        allowed = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        if attention_mask is not None:
            if not torch.all(attention_mask[:, 0]):
                raise ValueError("Tiny model requires right padding and a nonempty first token")
            allowed = allowed[None, None] & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, float("-inf"))
        probabilities = F.softmax(scores.float(), dim=-1).to(x.dtype)
        context = (self.dropout(probabilities) @ v).transpose(1, 2).contiguous()
        return self.o_proj(context.reshape(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.width)
        self.up_proj = nn.Linear(config.width, 4 * config.width)
        self.down_proj = nn.Linear(4 * config.width, config.width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.dropout(self.attention(self.ln1(x), attention_mask))
        hidden = self.down_proj(F.gelu(self.up_proj(self.ln2(x))))
        return x + self.dropout(hidden)


class TinyTransformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TinyConfig()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocab_size, c.width)
        self.position_embedding = nn.Embedding(c.max_length, c.width)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.layers)])
        self.final_norm = nn.LayerNorm(c.width)
        self.lm_head = nn.Linear(c.width, c.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        _, length = input_ids.shape
        if length > self.config.max_length:
            raise ValueError("Sequence exceeds learned positional embedding table")
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, attention_mask)
        logits = self.lm_head(self.final_norm(x))  # [batch,time,vocab]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=10, temperature=0.0):
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.max_length :]
                scores = self(context).logits[:, -1]
                token = (
                    scores.argmax(-1, keepdim=True)
                    if temperature == 0
                    else torch.multinomial(F.softmax(scores / temperature, dim=-1), 1)
                )
                input_ids = torch.cat((input_ids, token), dim=1)
            return input_ids
        finally:
            self.train(was_training)
