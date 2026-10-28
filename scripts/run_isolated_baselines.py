"""Fresh-process repeats for valid memory/runtime comparison; preserve originals."""

import json
from pathlib import Path
import yaml
from lowrank_lab.experiments.runner import read_config
from lowrank_lab.experiments.grids import generate_grid, run_suite
from lowrank_lab.utils.io import write_json


def main():
    paths = generate_grid(
        "configs/pretrained_gpu.yaml", "configs/experiments/isolated_baselines", "baselines"
    )
    for path in paths:
        cfg = read_config(path)
        cfg["measurement_isolation"] = "fresh_process"
        Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run_suite("configs/experiments/isolated_baselines/manifest.json")
    checks = []
    for path in paths:
        cfg = read_config(path)
        actual = json.loads((Path(cfg["output_dir"]) / "result.json").read_text())
        previous = json.loads(
            (
                Path("results/pretrained_baselines") / ("method-" + cfg["method"]) / "result.json"
            ).read_text()
        )
        # There is no algorithm/hyperparameter change: exact GPU metrics should repeat.
        for split in ["validation", "test"]:
            assert actual[split] == previous[split], (cfg["method"], split)
        checks.append(
            {
                "method": cfg["method"],
                "metric_reproduction": "exact",
                "peak_allocated_gpu_bytes": actual["training"].get("peak_allocated_gpu_bytes"),
            }
        )
    original = json.loads(Path("experiments/selected_manifest.json").read_text())["runs"]
    write_json(
        "experiments/executed_manifest.json",
        {
            "runs": original + paths,
            "status": "20 staged pilots plus five fresh-process primary repeats completed",
        },
    )
    write_json("reports/isolated_comparison_audit.json", checks)


if __name__ == "__main__":
    main()
