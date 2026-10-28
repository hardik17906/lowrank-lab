"""Prepare all requested grids, then run an affordable staged pretrained subset.

Baselines run first. Every later study is explicit and sequential. Do not
mistake the full generated grids for completed measurements. Run from root.
"""

from pathlib import Path
from lowrank_lab.experiments.grids import generate_grid, run_suite
from lowrank_lab.utils.io import write_json


def main():
    base = "configs/pretrained_gpu.yaml"
    root = Path("configs/experiments")
    stages = ["baselines", "rank", "placement", "size", "initialization", "seeds"]
    generated = {
        stage: generate_grid(
            base, root / ("pretrained_" + stage if stage == "baselines" else stage), stage
        )
        for stage in stages
    }
    selected = generated["baselines"]
    # Full LoRA rank/placement/size axes; two affordable methods for seed variance.
    selected += [generated["rank"][i] for i in [1, 2, 3]]
    selected += [generated["placement"][i] for i in [1, 2]]
    selected += [generated["size"][i] for i in [0, 1, 2]]
    selected += [generated["initialization"][i] for i in [1, 3, 5]]
    selected += [generated["seeds"][i] for i in [4, 5, 10, 11]]
    manifest = "experiments/selected_manifest.json"
    write_json(manifest, {"runs": selected, "design": "staged matched-12-step pretrained pilot"})
    run_suite(manifest)
    write_json(
        "experiments/executed_manifest.json",
        {
            "runs": selected,
            "status": "all selected runs completed; unselected grid entries were not executed",
        },
    )


if __name__ == "__main__":
    main()
