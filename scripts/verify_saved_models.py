"""End-to-end round trips of ACTUAL pretrained exports, after the study finishes."""

import json
import math
from pathlib import Path
from lowrank_lab.experiments.runner import read_config, run
from lowrank_lab.utils.io import write_json


def main():
    checks = []
    root = Path("configs/experiments/pretrained_baselines")
    for method, index in [("full", 1), ("lora", 2), ("dora", 3), ("pissa", 4)]:
        config = read_config(root / f"{index:02d}_method-{method}.yaml")
        original = json.loads((Path(config["output_dir"]) / "result.json").read_text())
        evaluated = run(config, evaluate_only=True)
        for split in ["validation", "test"]:
            assert math.isclose(
                original[split]["cross_entropy"],
                evaluated[split]["cross_entropy"],
                abs_tol=3e-5,
                rel_tol=3e-5,
            )
            for metric in ["accuracy", "macro_f1", "confusion_matrix"]:
                assert original[split][metric] == evaluated[split][metric], (method, metric)
        checks.append(
            {
                "method": method,
                "test_ce_original": original["test"]["cross_entropy"],
                "test_ce_loaded": evaluated["test"]["cross_entropy"],
                "status": "passed",
            }
        )
    write_json("reports/checkpoint_roundtrip.json", checks)
    print("All four pretrained model/adapter export round trips passed")


if __name__ == "__main__":
    main()
