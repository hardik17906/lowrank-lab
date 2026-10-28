"""Staged study manifests; generation never launches a large grid implicitly."""

import copy
import itertools
from pathlib import Path
import yaml
from .runner import read_config
from ..utils.io import write_json


def generate_grid(base_path, output_dir, stage):
    base = read_config(base_path)
    entries = []
    if stage == "baselines":
        variations = [{"method": m} for m in ["baseline", "full", "lora", "dora", "pissa"]]
    elif stage == "rank":
        variations = [
            {"method": m, "rank": r}
            for m, r in itertools.product(["lora", "dora", "pissa"], [4, 8, 16, 32])
        ]
    elif stage == "placement":
        variations = [
            {"method": m, "targets": t}
            for m, t in itertools.product(
                ["lora", "dora", "pissa"], ["attention", "ffn", "combined"]
            )
        ]
    elif stage == "size":
        variations = [
            {"method": m, "fraction": f}
            for m, f in itertools.product(["lora", "dora", "pissa"], [0.1, 0.25, 0.5, 1.0])
        ]
    elif stage == "seeds":
        variations = [
            {"method": m, "seed": s}
            for m, s in itertools.product(["full", "lora", "dora", "pissa"], [7, 17, 27])
        ]
    elif stage == "initialization":
        variations = [
            {"method": "lora", "init": "kaiming"},
            {"method": "lora", "init": "gaussian"},
            {"method": "pissa", "svd": "exact"},
            {"method": "pissa", "svd": "fast"},
            {"method": "dora", "detach_norm": True},
            {"method": "dora", "detach_norm": False},
        ]
    else:
        raise ValueError(stage)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for i, changes in enumerate(variations):
        config = copy.deepcopy(base)
        for key, value in changes.items():
            target = (
                config["adapter"]
                if key in {"rank", "targets", "init", "svd", "detach_norm"}
                else config["data"]
                if key == "fraction"
                else config
            )
            target[key] = value
        slug = "_".join(f"{k}-{v}" for k, v in changes.items())
        config["output_dir"] = (Path("results") / destination.name / slug).as_posix()
        config["measurement_isolation"] = "fresh_process"
        path = destination / f"{i:02d}_{slug}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        entries.append(path.as_posix())
    write_json(
        destination / "manifest.json",
        {
            "stage": stage,
            "runs": entries,
            "fairness": "matched optimizer steps; alpha fixed across ranks; NOT matched wall time or parameters",
        },
    )
    return entries


def run_suite(manifest, allow_download=False, limit=None):
    import json
    import subprocess
    import sys

    paths = json.loads(Path(manifest).read_text(encoding="utf-8"))["runs"]
    for path in paths[:limit]:
        config = read_config(path)
        result = Path(config["output_dir"]) / "result.json"
        if result.exists():
            print(f"Skipping completed run {path}", flush=True)
            continue
        # A process boundary prevents previous optimizers/models from inflating
        # a later run's live GPU allocations and measured training memory.
        command = [
            sys.executable,
            "-m",
            "lowrank_lab",
            "run",
            "--config",
            str(path).replace("\\", "/"),
        ]
        if allow_download:
            command.append("--allow-download")
        subprocess.run(command, check=True)
