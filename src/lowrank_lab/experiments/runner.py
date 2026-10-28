"""One consistent interface for random tiny demonstrations and genuine pretrained runs."""

import copy
import json
import time
import traceback
import gc
from pathlib import Path
import torch
import yaml
from ..adapters import inject_adapters, parameter_counts
from ..adapters.checkpoint import save_adapter, load_adapter
from ..data.dataset_loader import load_sms, take_subset
from ..data.preprocessing import (
    synthetic_splits,
    encode_record,
    remove_token_collisions,
    split_fingerprint,
)
from ..models import TinyConfig, TinyTransformer
from ..models.pretrained import load_pretrained
from ..training.trainer import Trainer
from ..evaluation.benchmark import evaluate_loss, evaluate_classification
from ..utils.environment import environment_report
from ..utils.reproducibility import seed_everything
from ..utils.io import write_json, save_torch


def read_config(path):
    # Manifests created on Windows should also be readable on POSIX.
    path = Path(str(path).replace("\\", "/"))
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" in config:
        base = read_config(Path(path).parent / config.pop("extends"))

        def merge(a, b):
            for key, value in b.items():
                if isinstance(value, dict) and isinstance(a.get(key), dict):
                    merge(a[key], value)
                else:
                    a[key] = value
            return a

        config = merge(base, config)
    if "output_dir" in config:
        config["output_dir"] = config["output_dir"].replace("\\", "/")
    return config


def setup(config, allow_download=False):
    seed_everything(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    if config["model"]["kind"] == "tiny":
        model = TinyTransformer(TinyConfig(**config["model"].get("tiny", {})))
        tokenizer, raw = None, None
        records = synthetic_splits(
            config["data"].get("split_seed", 42), config["data"]["max_length"]
        )
        for key in records:
            records[key] = take_subset(
                records[key],
                config["data"].get(key + "_samples"),
                config["data"].get("fraction", 1.0) if key == "train" else 1.0,
            )
        provenance = {
            "dataset": "synthetic repeated-pair LM",
            "purpose": "educational, not a pretrained benchmark",
        }
        identity = {
            "model_id": "random-tiny-transformer",
            "revision": "local",
            "seed": config["seed"],
            "tiny": config["model"].get("tiny", {}),
        }
    else:
        model, tokenizer, identity = load_pretrained(config["model"], allow_download)
        raw, provenance = load_sms(
            config["data"].get("cache_dir", "data/raw"),
            config["data"].get("split_seed", 42),
            allow_download,
        )
        raw, removed = remove_token_collisions(raw, tokenizer, config["data"]["max_length"])
        provenance["removed_token_prompt_collisions"] = removed
        raw = {
            key: take_subset(
                rows,
                config["data"].get(key + "_samples"),
                config["data"].get("fraction", 1.0) if key == "train" else 1.0,
                config["data"].get("split_seed", 42),
            )
            for key, rows in raw.items()
        }
        records = {
            key: [encode_record(tokenizer, row, config["data"]["max_length"]) for row in rows]
            for key, rows in raw.items()
        }
    provenance["used_counts"] = {k: len(v) for k, v in records.items()}
    provenance["encoded_splits_sha256"] = split_fingerprint(records)
    provenance["split_hashes"] = {k: split_fingerprint({k: v}) for k, v in records.items()}
    if raw:
        provenance["class_counts"] = {
            k: [sum(r["label"] == c for r in rows) for c in [0, 1]] for k, rows in raw.items()
        }
    return model, tokenizer, records, raw, identity, provenance


def run(config, allow_download=False, resume=None, evaluate_only=False):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() and not evaluate_only:
        raise FileExistsError(f"Completed run already exists at {output}; choose a new output_dir")
    if (output / "checkpoint.pt").exists() and not resume and not evaluate_only:
        raise FileExistsError("Interrupted run exists; pass --resume or choose a new output_dir")
    if not evaluate_only and not resume:
        write_json(output / "config.json", config)
    try:
        initialization_start = time.perf_counter()
        model, tokenizer, records, raw, identity, provenance = setup(config, allow_download)
        resolved = copy.deepcopy(config)
        resolved["model"]["revision"] = identity["revision"]
        if not evaluate_only and not resume:
            write_json(output / "resolved_config.json", resolved)
            write_json(output / "environment.json", environment_report())
            write_json(output / "data_manifest.json", provenance)
        # Includes input IDs and masked labels, never makes downloading mandatory for smoke.
        if not evaluate_only and not resume:
            write_json(output / "encoded_splits.json", records)
        method = config["method"]
        names = []
        if method in {"lora", "dora", "pissa"}:
            adapter = copy.deepcopy(config["adapter"])
            if method != "dora":
                adapter.pop("detach_norm", None)
            if method != "pissa":
                adapter.pop("svd", None)
                adapter.pop("svd_iterations", None)
            names = inject_adapters(model, method=method, **adapter)
        elif method == "baseline":
            model.requires_grad_(False)
        elif method != "full":
            raise ValueError(f"Unknown method {method}")
        write_json(output / "target_modules.json", names)
        counts = parameter_counts(model)
        device = torch.device(config.get("device", "cpu"))
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested but unavailable; install a compatible CUDA-enabled torch build"
                )
            # Sequential runs may leave FREE blocks in PyTorch's caching allocator.
            # Release those before querying driver-free memory; live tensors stay.
            gc.collect()
            torch.cuda.empty_cache()
            # FP32 Adam estimates: weights + gradients + 2 moments, excluding activations.
            parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            minimum = parameter_bytes + counts["trainable_parameters"] * 12
            free, _ = torch.cuda.mem_get_info(device)
            if minimum > 0.80 * free:
                raise RuntimeError(
                    f"Static estimate {minimum / 2**30:.2f} GiB leaves insufficient VRAM for activations"
                )
        model.to(device)
        if evaluate_only:
            if method in {"lora", "dora", "pissa"}:
                load_adapter(model, output / "adapter.pt", identity)
            elif method == "full":
                model.load_state_dict(
                    torch.load(output / "model.pt", map_location=device, weights_only=True)
                )
        initialization_seconds = time.perf_counter() - initialization_start
        pad = tokenizer.pad_token_id if tokenizer else 0
        initial = evaluate_loss(model, records["validation"], config["training"]["batch_size"], pad)
        training = {"completed_steps": 0, "training_seconds": 0.0, "peak_allocated_gpu_bytes": None}
        if method != "baseline" and not evaluate_only:
            training_config = {**config["training"], "seed": config["seed"]}
            trainer = Trainer(
                model,
                records["train"],
                records["validation"],
                training_config,
                output,
                pad,
                identity={
                    "model": identity,
                    "data": provenance["encoded_splits_sha256"],
                    "method": method,
                    "adapter": config.get("adapter"),
                },
            )
            if resume:
                trainer.load_checkpoint(resume)
            training = trainer.fit()
            if method in {"lora", "dora", "pissa"}:
                save_adapter(model, output / "adapter.pt", identity)
            else:
                save_torch(output / "model.pt", model.state_dict())
        validation = evaluate_loss(
            model, records["validation"], config["training"]["batch_size"], pad
        )
        test = evaluate_loss(model, records["test"], config["training"]["batch_size"], pad)
        if tokenizer:
            val_classes = evaluate_classification(
                model,
                raw["validation"],
                tokenizer,
                config["data"]["max_length"],
                config["training"]["batch_size"],
            )
            test_classes = evaluate_classification(
                model,
                raw["test"],
                tokenizer,
                config["data"]["max_length"],
                config["training"]["batch_size"],
            )
            write_json(output / "validation_predictions.json", val_classes)
            write_json(output / "test_predictions.json", test_classes)
            validation.update(
                {
                    k: val_classes[k]
                    for k in ["accuracy", "macro_f1", "confusion_matrix", "class_support"]
                }
            )
            test.update(
                {
                    k: test_classes[k]
                    for k in ["accuracy", "macro_f1", "confusion_matrix", "class_support"]
                }
            )
        result = {
            "status": "completed",
            "method": method,
            "seed": config["seed"],
            "model": identity,
            "data": provenance,
            "initial_validation": initial,
            "validation": validation,
            "test": test,
            "training": training,
            "parameters": counts,
            "initialization_seconds": initialization_seconds,
            "adapter": config.get("adapter"),
            "run_config": config,
            "adapter_checkpoint_bytes": (output / "adapter.pt").stat().st_size
            if (output / "adapter.pt").exists()
            else None,
        }
        write_json(output / ("evaluation.json" if evaluate_only else "result.json"), result)
        print(
            json.dumps({"output": str(output), "method": method, "test": test}, indent=2),
            flush=True,
        )
        return result
    except Exception as error:
        write_json(
            output / "failure.json",
            {"error": repr(error), "traceback": traceback.format_exc(), "settings_changed": False},
        )
        raise
