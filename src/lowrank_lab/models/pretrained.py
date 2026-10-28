"""Load pinned, public Hugging Face causal LMs with conservative resource checks."""

import os
import shutil
from pathlib import Path
import psutil
import torch


def load_pretrained(config, allow_download=False):
    # All caches stay on the project drive (C: is almost full on the initial host).
    cache = Path(config.get("cache_dir", ".cache/huggingface")).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi, snapshot_download

    model_id = config["model_id"]
    revision = config.get("revision", "main")
    local_only = not allow_download
    resolved = revision
    if allow_download:
        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
        resolved = info.sha
        weight_bytes = sum(
            f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors")
        )
        if not weight_bytes:
            raise RuntimeError(
                "No safetensors checkpoint found; unsafe pickle download is not supported"
            )
        if shutil.disk_usage(cache).free < max(2 * weight_bytes, 2 * 2**30):
            raise RuntimeError("Insufficient disk for weights, cache and checkpoints")
        # Staging + model + FP32 SVD + optimizer working memory need more than file size.
        if psutil.virtual_memory().available < max(3 * weight_bytes, 1 * 2**30):
            raise RuntimeError(
                "Insufficient available host RAM for conservative model initialization"
            )
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        config.get("dtype", "float32")
    ]
    # Loading a local snapshot also prevents optional PEFT auto-detection from
    # making an unexpected adapter_config HEAD request despite local_files_only.
    snapshot = snapshot_download(
        model_id,
        revision=resolved,
        cache_dir=str(cache),
        local_files_only=local_only,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "vocab.*",
            "merges.txt",
            "*.model",
            "model*.safetensors",
            "model.safetensors.index.json",
        ],
    )
    if local_only:
        resolved = Path(snapshot).name
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has neither padding nor EOS token")
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        use_safetensors=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return (
        model,
        tokenizer,
        {
            "model_id": model_id,
            "revision": getattr(model.config, "_commit_hash", None) or resolved,
            "dtype": str(dtype),
            "architecture": type(model).__name__,
        },
    )
