"""Export a dated, checksummed text snapshot from completed local runs."""

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from lowrank_lab.utils.io import write_json

SNAPSHOT = Path("benchmarks/sms-pilot-2026-10-28")
ARTIFACTS = {
    "result.json",
    "config.json",
    "resolved_config.json",
    "data_manifest.json",
    "encoded_splits.json",
    "history.json",
    "epochs.json",
    "target_modules.json",
    "test_predictions.json",
    "validation_predictions.json",
    "generation.json",
}


def export(destination=SNAPSHOT):
    destination = Path(destination)
    if (destination / "manifest.json").exists():
        raise FileExistsError(
            "Snapshot exists; choose a new study directory instead of overwriting it"
        )
    runs = sorted(Path("results").rglob("result.json"))
    if not runs:
        raise ValueError("No completed local runs")
    recorded = []
    for result_path in runs:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "completed":
            continue
        relative = result_path.parent.relative_to("results")
        output = destination / "runs" / relative
        output.mkdir(parents=True, exist_ok=True)
        for artifact in sorted(result_path.parent.iterdir()):
            if artifact.name in ARTIFACTS:
                shutil.copyfile(artifact, output / artifact.name)
        recorded.append(relative.as_posix())
    environment = json.loads(Path("reports/environment.json").read_text())
    public_environment = {
        key: environment[key]
        for key in [
            "os",
            "torch",
            "cuda_build",
            "cuda_available",
            "torch_gpus",
            "ram_total_bytes",
            "packages",
        ]
    }
    public_environment["python"] = environment["python"].split()[0]
    write_json(destination / "environment.json", public_environment)
    suite = ET.parse("reports/pytest.xml").getroot().find("testsuite")
    validation = {key: int(suite.attrib[key]) for key in ["tests", "failures", "errors", "skipped"]}
    validation["seconds"] = float(suite.attrib["time"])
    validation["scope"] = "Recorded Windows CUDA environment, including PEFT reference checks"
    write_json(destination / "validation.json", validation)
    for name in ["checkpoint_roundtrip.json", "isolated_comparison_audit.json"]:
        shutil.copyfile(Path("reports") / name, destination / name)
    files = {}
    for path in sorted(destination.rglob("*.json")):
        data = path.read_bytes()
        files[path.relative_to(destination).as_posix()] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "study": "sms-pilot-2026-10-28",
            "completed_runs": len(recorded),
            "runs": recorded,
            "files": files,
            "omitted": [
                "model and adapter weights",
                "optimizer checkpoints",
                "raw dataset",
                "machine-specific logs, hostnames, process IDs and filesystem paths",
            ],
            "primary_efficiency_runs": "runs/isolated_baselines",
            "historical_output_root": "results",
            "source_date": "2026-10-28",
        },
    )
    print(
        f"Exported {len(recorded)} completed runs and {len(files)} text artifacts to {destination}"
    )


if __name__ == "__main__":
    export()
