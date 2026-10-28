"""Verify the published benchmark without weights, network access or training."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from lowrank_lab.data.preprocessing import split_fingerprint
from lowrank_lab.evaluation.metrics import classification_metrics


def verify(snapshot):
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot.resolve()):
            raise ValueError(f"Artifact escapes snapshot: {relative}")
        data = path.read_bytes()
        assert len(data) == expected["bytes"], f"Size mismatch: {relative}"
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], f"Hash mismatch: {relative}"
    assert len(manifest["runs"]) == manifest["completed_runs"]
    for relative in manifest["runs"]:
        root = snapshot / "runs" / relative
        result = json.loads((root / "result.json").read_text())
        splits = json.loads((root / "encoded_splits.json").read_text())
        assert split_fingerprint(splits) == result["data"]["encoded_splits_sha256"]
        visible = {}
        for split, rows in splits.items():
            assert len(rows) == result["data"]["used_counts"][split]
            assert split_fingerprint({split: rows}) == result["data"]["split_hashes"][split]
            keys = []
            for row in rows:
                first = next((i for i, label in enumerate(row["labels"]) if label != -100), 0)
                keys.append(tuple(row["input_ids"][:first] if first else row["input_ids"]))
            assert len(set(keys)) == len(keys), f"Duplicate visible input: {relative}/{split}"
            visible[split] = set(keys)
        assert not visible["train"] & visible["validation"]
        assert not visible["train"] & visible["test"]
        assert not visible["validation"] & visible["test"]
        if "raw_sha256" in result["data"]:
            for split in ["validation", "test"]:
                predictions = json.loads((root / f"{split}_predictions.json").read_text())
                measured = classification_metrics(predictions["labels"], predictions["predictions"])
                for metric in ["accuracy", "macro_f1", "confusion_matrix", "class_support"]:
                    assert measured[metric] == result[split][metric], (relative, metric)
                assert math.isfinite(result[split]["cross_entropy"])
        history = root / "history.json"
        if history.exists():
            events = json.loads(history.read_text())
            assert len(events) == result["training"]["completed_steps"]
            assert [event["step"] for event in events] == list(range(1, len(events) + 1))
    print(
        f"Verified {manifest['completed_runs']} recorded runs and {len(manifest['files'])} artifact hashes"
    )
    return manifest["completed_runs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("benchmarks/sms-pilot-2026-10-28"))
    args = parser.parse_args()
    verify(args.snapshot)


if __name__ == "__main__":
    main()
