"""Audit actual saved outputs and published commands without rerunning training."""

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from lowrank_lab.data.preprocessing import split_fingerprint
from lowrank_lab.evaluation.metrics import classification_metrics
from lowrank_lab.experiments.runner import read_config
from lowrank_lab.utils.io import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    checked, completed = [], list((root / "results").rglob("result.json"))
    assert completed, "No completed experimental outputs"
    raw_hash = hashlib.sha256((root / "data/raw/sms_spam_collection.zip").read_bytes()).hexdigest()

    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)

    for path in completed:
        result = json.loads(path.read_text())
        assert finite(result), f"Nonfinite result {path}"
        splits = json.loads((path.parent / "encoded_splits.json").read_text())
        assert split_fingerprint(splits) == result["data"]["encoded_splits_sha256"]
        keys = {}
        for split, rows in splits.items():
            assert len(rows) == result["data"]["used_counts"][split]
            assert result["data"]["split_hashes"][split] == split_fingerprint({split: rows})

            def visible(row):
                if -100 not in row["labels"]:
                    return tuple(row["input_ids"])
                prompt_length = next(i for i, x in enumerate(row["labels"]) if x != -100)
                return tuple(row["input_ids"][:prompt_length])

            keys[split] = {visible(r) for r in rows}
            assert len(keys[split]) == len(rows), f"Duplicate visible inputs in {path}/{split}"
        assert not keys["train"] & keys["validation"]
        assert not keys["train"] & keys["test"]
        assert not keys["validation"] & keys["test"]
        if "raw_sha256" in result["data"]:
            assert result["data"]["raw_sha256"] == raw_hash
            for split in ["validation", "test"]:
                predictions = json.loads((path.parent / f"{split}_predictions.json").read_text())
                measured = classification_metrics(predictions["labels"], predictions["predictions"])
                for metric in ["accuracy", "macro_f1", "confusion_matrix", "class_support"]:
                    assert measured[metric] == result[split][metric]
        if result["adapter_checkpoint_bytes"] is not None:
            assert (path.parent / "adapter.pt").stat().st_size == result["adapter_checkpoint_bytes"]
        history = path.parent / "history.json"
        if history.exists():
            events = json.loads(history.read_text())
            assert len(events) == result["training"]["completed_steps"]
            assert [p["step"] for p in events] == list(range(1, len(events) + 1))
        checked.append(str(path.relative_to(root)))
    # Matched primary baseline controls really are equal except for method.
    primary_paths = sorted((root / "results/isolated_baselines").glob("*/result.json"))
    primary = [json.loads(p.read_text()) for p in primary_paths]
    assert len(primary) == 5
    process_ids = [
        json.loads((p.parent / "environment.json").read_text())["process_id"] for p in primary_paths
    ]
    assert len(set(process_ids)) == 5, "Primary memory comparison did not use separate processes"
    for field in ["model", "data"]:
        assert all(r[field] == primary[0][field] for r in primary)
    assert all(r["run_config"]["training"] == primary[0]["run_config"]["training"] for r in primary)
    # Every explicitly selected configuration completed at its requested budget.
    manifest = json.loads((root / "experiments/executed_manifest.json").read_text())
    for config_path in manifest["runs"]:
        config = read_config(config_path)
        result = json.loads((root / config["output_dir"] / "result.json").read_text())
        if config["method"] != "baseline":
            assert result["training"]["completed_steps"] == config["training"]["max_steps"]
    missing_links = []
    for directory in [root, root / "learning", root / "reports"]:
        for file in directory.glob("*.md"):
            for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
                if "://" in link or link.startswith("#"):
                    continue
                target = file.parent / link.split("#")[0]
                if not target.exists():
                    missing_links.append(f"{file.name}: {link}")
    assert not missing_links, missing_links
    command_audit = []
    for command in [None, "environment", "prepare", "run", "evaluate", "grid", "suite", "analyze"]:
        args = [sys.executable, "-m", "lowrank_lab"] + ([command] if command else []) + ["--help"]
        response = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=root)
        assert response.returncode == 0, response.stderr
        command_audit.append(" ".join(args[2:]))
    notebook = json.loads((root / "notebooks/01_inspect_adapters.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "01_inspect_adapters.ipynb", "exec"), namespace)
    report = {
        "status": "passed",
        "completed_runs_audited": len(checked),
        "result_files": checked,
        "selected_runs": len(manifest["runs"]),
        "checks": [
            "finite outputs",
            "raw/encoded fingerprints",
            "visible-input split disjointness",
            "record counts",
            "metrics recomputed from predictions",
            "artifact byte sizes",
            "step histories",
            "primary matched controls",
            "selected budgets",
            "documentation links",
            "all CLI help parsers",
            "notebook code execution",
        ],
        "cli_commands": command_audit,
        "missing_documentation_links": missing_links,
    }
    write_json(root / "reports/AUDIT.json", report)
    print(f"Audit passed: {len(checked)} completed runs, all CLI subcommands and notebook cells")


if __name__ == "__main__":
    main()
