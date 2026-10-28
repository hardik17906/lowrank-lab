"""All tables and figures are derived exclusively from completed result.json files."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def analyze(root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, histories = [], []
    for path in sorted(Path(root).rglob("result.json")):
        result = json.loads(path.read_text())
        if result["status"] != "completed":
            continue
        adapter = result.get("adapter") or {}
        cfg = result["run_config"]
        adapted = result["method"] in {"lora", "dora", "pissa"}
        row = {
            "run": str(path.parent),
            "method": result["method"],
            "seed": result["seed"],
            "model": result["model"]["model_id"],
            "revision": result["model"]["revision"],
            "data_hash": result["data"]["encoded_splits_sha256"],
            "test_hash": result["data"]["split_hashes"]["test"],
            "train_samples": result["data"]["used_counts"]["train"],
            "test_samples": result["data"]["used_counts"]["test"],
            "max_length": cfg["data"]["max_length"],
            "device": cfg["device"],
            "measurement_isolation": cfg.get("measurement_isolation", "unspecified"),
            "learning_rate": cfg["training"]["learning_rate"],
            "precision": cfg["training"].get("mixed_precision", "none"),
            "rank": adapter.get("rank") if adapted else None,
            "targets": str(adapter.get("targets")) if adapted else "n/a",
            "fraction": cfg["data"].get("fraction", 1.0),
            "alpha": adapter.get("alpha") if adapted else None,
            "init": adapter.get("init", "kaiming")
            if result["method"] in {"lora", "dora"}
            else "n/a",
            "svd": adapter.get("svd", "exact") if result["method"] == "pissa" else "n/a",
            "detach_norm": adapter.get("detach_norm", True) if result["method"] == "dora" else None,
            "validation_loss": result["validation"]["cross_entropy"],
            "test_loss": result["test"]["cross_entropy"],
            "test_accuracy": result["test"].get("accuracy"),
            "test_macro_f1": result["test"].get("macro_f1"),
            "trainable_parameters": result["parameters"]["trainable_parameters"],
            "total_parameters": result["parameters"]["total_parameters"],
            "training_seconds": result["training"]["training_seconds"],
            "initialization_seconds": result["initialization_seconds"],
            "steps": result["training"]["completed_steps"],
            "peak_gpu_mib": result["training"].get("peak_allocated_gpu_bytes"),
            "adapter_kib": result.get("adapter_checkpoint_bytes"),
        }
        if row["peak_gpu_mib"] is not None:
            row["peak_gpu_mib"] /= 2**20
        if row["adapter_kib"] is not None:
            row["adapter_kib"] /= 1024
        rows.append(row)
        hist = path.parent / "history.json"
        if hist.exists():
            history = json.loads(hist.read_text())
            histories.append((row, history))
            threshold = result["initial_validation"]["cross_entropy"] * 0.5
            hits = [p for p in history if p.get("validation_loss", float("inf")) <= threshold]
            row["steps_to_half_initial_validation_loss"] = hits[0]["step"] if hits else None
            row["seconds_to_half_initial_validation_loss"] = (
                hits[0]["elapsed_seconds"] if hits else None
            )
    if not rows:
        raise ValueError("No completed runs found")
    df = pd.DataFrame(rows)
    df.to_csv(output / "comparison.csv", index=False)
    # Avoid combining random tiny and pretrained results in the same panel.
    for model, subset in df.groupby("model"):
        slug = model.replace("/", "_")
        primary = subset[subset["run"].map(lambda p: Path(p).parent.name.endswith("baselines"))]
        isolated = primary[primary["measurement_isolation"] == "fresh_process"]
        if not isolated.empty:
            primary = isolated
        if primary.empty:
            primary = subset
        method_order = {m: i for i, m in enumerate(["baseline", "full", "lora", "dora", "pissa"])}
        primary = primary.loc[primary["method"].map(method_order).sort_values().index]
        chosen_paths = set(primary["run"])
        chosen = [(r, h) for r, h in histories if r["run"] in chosen_paths]
        chosen.sort(key=lambda pair: method_order.get(pair[0]["method"], 99))
        for metric in ["training_loss", "validation_loss"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for row, history in chosen:
                entries = [p for p in history if metric in p]
                ax.plot(
                    [p["step"] for p in entries],
                    [p[metric] for p in entries],
                    alpha=0.7,
                    label=Path(row["run"]).name,
                )
            ax.set(xlabel="Optimizer step", ylabel=metric.replace("_", " "), title=model)
            if chosen and len(chosen) <= 12:
                ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(output / f"{slug}_{metric}.png", dpi=150)
            plt.close(fig)
        metrics = [
            "test_accuracy",
            "test_macro_f1",
            "trainable_parameters",
            "peak_gpu_mib",
            "training_seconds",
            "test_loss",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, metric in zip(axes.flat, metrics):
            valid = primary.dropna(subset=[metric])
            if valid.empty:
                ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.bar(range(len(valid)), valid[metric])
                ax.set_xticks(
                    range(len(valid)), [Path(p).name for p in valid["run"]], rotation=90, fontsize=6
                )
            ax.set_title(metric.replace("_", " "))
        fig.suptitle(model + " — exploratory runs; consult configs before comparing")
        fig.tight_layout()
        fig.savefig(output / f"{slug}_comparison.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        default_initialization = (
            ((~subset["method"].isin(["lora", "dora"])) | (subset["init"] == "kaiming"))
            & ((subset["method"] != "pissa") | (subset["svd"] == "exact"))
            & ((subset["method"] != "dora") | subset["detach_norm"].eq(True))
        )
        controlled = subset[
            (subset["seed"] == 7)
            & (subset["targets"] == "attention")
            & default_initialization
            & subset["method"].isin(["lora", "dora", "pissa"])
        ]
        for method, group in controlled.groupby("method"):
            rank_group = group[group["fraction"] == 1.0]
            size_group = group[group["rank"] == 4]
            axes[0].scatter(rank_group["rank"], rank_group["test_loss"], label=method)
            axes[1].scatter(size_group["fraction"], size_group["test_loss"], label=method)
        for method, group in primary.groupby("method"):
            axes[2].scatter(group["trainable_parameters"], group["test_loss"], label=method)
        for ax, label in zip(
            axes, ["Adapter rank", "Training data fraction", "Trainable parameters"]
        ):
            ax.set(xlabel=label, ylabel="Held-out target cross-entropy")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"{slug}_sensitivity.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        for row, history in chosen:
            entries = [p for p in history if "validation_loss" in p]
            ax.plot(
                [p["elapsed_seconds"] for p in entries],
                [p["validation_loss"] for p in entries],
                alpha=0.7,
            )
        ax.set(
            xlabel="Training and interval-validation seconds",
            ylabel="Validation target cross-entropy",
            title=model,
        )
        fig.tight_layout()
        fig.savefig(output / f"{slug}_convergence_time.png", dpi=150)
        plt.close(fig)
        # Show every observed seed instead of hiding small-n variation in a bar.
        seed_runs = subset[
            (subset["rank"] == 4)
            & (subset["fraction"] == 1.0)
            & (subset["targets"] == "attention")
            & default_initialization
        ]
        seed_runs = seed_runs.drop_duplicates(["method", "seed", "device", "data_hash"])
        fig, ax = plt.subplots(figsize=(8, 4))
        for method, group in seed_runs.groupby("method"):
            ax.scatter(group["seed"], group["test_loss"], label=method, alpha=0.8)
        ax.set(xlabel="Training seed", ylabel="Held-out target cross-entropy", title=model)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"{slug}_seed_variability.png", dpi=150)
        plt.close(fig)
        matrices = []
        for _, row in primary.iterrows():
            original = json.loads((Path(row["run"]) / "result.json").read_text())
            if "confusion_matrix" in original["test"]:
                matrices.append((row["method"], original["test"]["confusion_matrix"]))
        if matrices:
            fig, axes = plt.subplots(
                1, len(matrices), figsize=(3 * len(matrices), 3.2), squeeze=False
            )
            common_max = max(max(max(row) for row in matrix) for _, matrix in matrices)
            for ax, (method, matrix) in zip(axes.flat, matrices):
                ax.imshow(matrix, cmap="Blues", vmin=0, vmax=common_max)
                for i in range(2):
                    for j in range(2):
                        ax.text(
                            j,
                            i,
                            str(matrix[i][j]),
                            ha="center",
                            va="center",
                            color="white" if matrix[i][j] > common_max / 2 else "black",
                        )
                ax.set_xticks([0, 1], ["legit", "spam"])
                ax.set_yticks([0, 1], ["legit", "spam"])
                ax.set(xlabel="Predicted", ylabel="True", title=method)
            fig.tight_layout()
            fig.savefig(output / f"{slug}_confusion_matrices.png", dpi=150)
            plt.close(fig)
    keys = [
        "model",
        "revision",
        "data_hash",
        "method",
        "rank",
        "targets",
        "fraction",
        "steps",
        "alpha",
        "init",
        "svd",
        "detach_norm",
        "learning_rate",
        "device",
        "precision",
        "max_length",
        "measurement_isolation",
    ]
    # Repeating the same seed/configuration is not an independent seed sample.
    independent_seeds = df.drop_duplicates(subset=keys + ["seed"])
    summary = independent_seeds.groupby(keys, dropna=False)[
        ["test_loss", "test_accuracy", "test_macro_f1", "training_seconds"]
    ].agg(["mean", "std", "count"])
    summary.to_csv(output / "seed_summary.csv")
    # Single observations have undefined SD (blank), never report a false zero.
    columns = [
        "method",
        "seed",
        "model",
        "rank",
        "targets",
        "fraction",
        "steps",
        "test_loss",
        "test_accuracy",
        "test_macro_f1",
        "trainable_parameters",
        "training_seconds",
    ]

    def cell(x):
        return f"{x:.5g}" if isinstance(x, float) and pd.notna(x) else "—" if pd.isna(x) else str(x)

    table = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    table += "\n".join(
        "| " + " | ".join(cell(row[c]) for c in columns) + " |" for _, row in df.iterrows()
    )
    (output / "TABLE.md").write_text(
        "# Actual completed runs\n\n"
        + table
        + "\n\nDo not pool different data hashes, initialization modes, or model revisions. Target perplexity is conditional on labels for SMS, not whole-text perplexity.\n",
        encoding="utf-8",
    )
    return df
