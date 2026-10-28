from pathlib import Path
import pytest
from lowrank_lab.experiments.runner import read_config, run
from lowrank_lab.experiments.grids import generate_grid


def test_all_configurations_parse_and_grids_are_concrete(tmp_path):
    for path in Path("configs").glob("*.yaml"):
        cfg = read_config(path)
        assert cfg["training"]["max_steps"] > 0
        assert cfg["data"]["max_length"] <= 256
    for stage, expected in [
        ("baselines", 5),
        ("rank", 12),
        ("placement", 9),
        ("size", 12),
        ("seeds", 12),
        ("initialization", 6),
    ]:
        paths = generate_grid("configs/smoke.yaml", tmp_path / stage, stage)
        assert len(paths) == expected
        assert all(
            read_config(p)["method"] in {"baseline", "full", "lora", "dora", "pissa"} for p in paths
        )


def test_complete_tiny_cli_workflow_and_reevaluation(tmp_path):
    cfg = read_config("configs/smoke.yaml")
    cfg["output_dir"] = str(tmp_path / "workflow")
    cfg["model"]["tiny"].update(width=16, heads=2, layers=1)
    cfg["data"].update(train_samples=8, validation_samples=4, test_samples=4, max_length=12)
    cfg["training"].update(max_steps=2, batch_size=2, validation_interval=1)
    cfg["method"] = "pissa"
    result = run(cfg)
    original_config = (tmp_path / "workflow" / "config.json").read_bytes()
    evaluated = run(cfg, evaluate_only=True)
    assert result["test"] == evaluated["test"]
    assert (tmp_path / "workflow" / "config.json").read_bytes() == original_config
    with pytest.raises(FileExistsError):
        run(cfg)
