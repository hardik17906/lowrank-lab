"""A single-device, inspectable trainer with deterministic optimizer-step resume."""

import contextlib
import time
from pathlib import Path
import psutil
import torch
from .losses import causal_loss_sum
from .optimizers import make_optimizer, make_scheduler
from ..data.preprocessing import collate
from ..evaluation.benchmark import evaluate_loss
from ..utils.reproducibility import rng_state, restore_rng
from ..utils.io import save_torch, write_json


class Trainer:
    """Checkpoint only at optimizer boundaries; no unsaved partial gradients.

    Epoch permutations use a private generator seeded by seed+epoch. Resume
    restores epoch, cursor, optimizer, scheduler, scaler, RNG and counters.
    Changing settings other than output/checkpoint paths is rejected.
    """

    def __init__(
        self,
        model,
        train_records,
        validation_records,
        config,
        output_dir,
        pad_token_id=0,
        identity=None,
    ):
        self.model = model
        self.train_records, self.validation_records = train_records, validation_records
        self.config, self.pad = dict(config), pad_token_id
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        if not train_records or not validation_records:
            raise ValueError("Nonempty train/validation sets required")
        for key in ["max_steps", "batch_size", "gradient_accumulation"]:
            if config[key] < 1:
                raise ValueError(f"{key} must be positive")
        self.device = next(model.parameters()).device
        precision = config.get("mixed_precision", "none")
        if precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be none, fp16 or bf16")
        if self.device.type == "cpu" and precision == "fp16":
            raise ValueError("CPU fp16 training is unsupported; choose none/bf16 explicitly")
        if (
            self.device.type == "cuda"
            and precision == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("This GPU does not support bf16")
        self.precision = precision
        self.optimizer = make_optimizer(model, config)
        self.scheduler = make_scheduler(self.optimizer, config)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda" and precision == "fp16"
        )
        self.step = self.epoch = self.cursor = 0
        self.history, self.epochs = [], []
        self.best, self.bad_validations = float("inf"), 0
        self.elapsed = 0.0
        self.identity = identity
        self.epoch_loss_sum, self.epoch_tokens = 0.0, 0
        self.peak_rss = 0

    def autocast(self):
        if self.precision == "none":
            return contextlib.nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _batches(self):
        generator = torch.Generator().manual_seed(self.config.get("seed", 0) + self.epoch)
        order = torch.randperm(len(self.train_records), generator=generator).tolist()
        size = self.config["batch_size"]
        return [order[i : i + size] for i in range(0, len(order), size)]

    def save_checkpoint(self, path=None):
        save_torch(
            path or self.output / "checkpoint.pt",
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "cursor": self.cursor,
                "history": self.history,
                "epochs": self.epochs,
                "best": self.best,
                "bad_validations": self.bad_validations,
                "elapsed": self.elapsed,
                "epoch_loss_sum": self.epoch_loss_sum,
                "epoch_tokens": self.epoch_tokens,
                "rng": rng_state(),
                "config": self.config,
                "identity": self.identity,
                "peak_rss": self.peak_rss,
            },
        )

    def load_checkpoint(self, path):
        # Trusted LOCAL experiment files contain Python/NumPy RNG objects.
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["config"] != self.config or state["identity"] != self.identity:
            raise ValueError("Resume configuration/data/model identity mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        for key in [
            "step",
            "epoch",
            "cursor",
            "history",
            "epochs",
            "best",
            "bad_validations",
            "elapsed",
            "epoch_loss_sum",
            "epoch_tokens",
            "peak_rss",
        ]:
            setattr(self, key, state[key])
        restore_rng(state["rng"])

    def fit(self, stop_after=None):
        """stop_after is an absolute boundary for interruption/resume tests."""
        self.model.train()
        start = time.perf_counter()
        previous_elapsed = self.elapsed
        target = min(stop_after or self.config["max_steps"], self.config["max_steps"])
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        while self.step < target:
            batches = self._batches()
            window = batches[self.cursor : self.cursor + self.config["gradient_accumulation"]]
            if not window:
                raise RuntimeError("Invalid checkpoint batch cursor")
            microbatches = [
                collate([self.train_records[i] for i in indices], self.pad) for indices in window
            ]
            token_count = sum(int((b["labels"][:, 1:] != -100).sum()) for b in microbatches)
            if not token_count:
                raise ValueError("Accumulation window has no supervised tokens")
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for batch in microbatches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with self.autocast():
                    loss, _ = causal_loss_sum(self.model(**batch).logits, batch["labels"])
                    normalized = loss / token_count
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; no training settings were changed")
                self.scaler.scale(normalized).backward()
                loss_sum += loss.item()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.get("max_grad_norm", 1.0),
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    "Nonfinite gradient norm; stopping without silent fallback"
                )
            lr = self.optimizer.param_groups[0]["lr"]
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.step += 1
            self.cursor += len(window)
            self.epoch_loss_sum += loss_sum
            self.epoch_tokens += token_count
            if self.cursor == len(batches):
                self.epochs.append(
                    {
                        "epoch": self.epoch + 1,
                        "step": self.step,
                        "training_loss": self.epoch_loss_sum / self.epoch_tokens,
                        "supervised_tokens": self.epoch_tokens,
                    }
                )
                self.epoch += 1
                self.cursor = 0
                self.epoch_loss_sum, self.epoch_tokens = 0.0, 0
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
            self.elapsed = previous_elapsed + time.perf_counter() - start
            row = {
                "step": self.step,
                "epoch": self.epoch,
                "training_loss": loss_sum / token_count,
                "learning_rate": lr,
                "gradient_norm": float(grad_norm),
                "elapsed_seconds": self.elapsed,
                "supervised_tokens": token_count,
            }
            interval = self.config.get("validation_interval", 10)
            if self.step % interval == 0 or self.step == self.config["max_steps"]:
                metrics = evaluate_loss(
                    self.model, self.validation_records, self.config["batch_size"], self.pad
                )
                row["validation_loss"] = metrics["cross_entropy"]
                if metrics["cross_entropy"] < self.best - self.config.get("min_delta", 0.0):
                    self.best, self.bad_validations = metrics["cross_entropy"], 0
                    # Save only trainable tensors; frozen base never changes.
                    save_torch(
                        self.output / "best.pt",
                        {
                            n: p.detach().cpu()
                            for n, p in self.model.named_parameters()
                            if p.requires_grad
                        },
                    )
                else:
                    self.bad_validations += 1
                print(
                    f"step={self.step} loss={row['training_loss']:.4f} val={row['validation_loss']:.4f}",
                    flush=True,
                )
            self.history.append(row)
            self.elapsed = previous_elapsed + time.perf_counter() - start
            if "validation_loss" in row:
                self.save_checkpoint()
            patience = self.config.get("early_stopping_patience", 0)
            if patience and self.bad_validations >= patience:
                break
        self.elapsed = previous_elapsed + time.perf_counter() - start
        self.save_checkpoint()
        if self.config.get("use_best", False) and stop_after is None:
            state = torch.load(self.output / "best.pt", map_location=self.device, weights_only=True)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in state:
                        param.copy_(state[name])
        write_json(self.output / "history.json", self.history)
        write_json(self.output / "epochs.json", self.epochs)
        return {
            "completed_steps": self.step,
            "completed_epochs": self.epoch,
            "training_seconds": self.elapsed,
            "best_validation_loss": self.best if self.best < float("inf") else None,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None,
            "peak_sampled_cpu_rss_bytes": self.peak_rss,
        }
