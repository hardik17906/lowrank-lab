import argparse
import json
from pathlib import Path
from .experiments.runner import read_config, run, setup
from .experiments.grids import generate_grid, run_suite
from .experiments.analysis import analyze
from .utils.environment import environment_report
from .utils.io import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lowrank-lab")
    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("environment", help="Inspect hardware and dependency versions")
    env.add_argument("--output", default="reports/environment.json")
    for name in ["run", "evaluate", "prepare"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--allow-download", action="store_true")
        if name == "run":
            p.add_argument("--resume")
    grid = sub.add_parser("grid", help="Generate configurations without executing them")
    grid.add_argument("--base", required=True)
    grid.add_argument("--output", required=True)
    grid.add_argument(
        "--stage",
        choices=["baselines", "rank", "placement", "size", "seeds", "initialization"],
        required=True,
    )
    suite = sub.add_parser("suite")
    suite.add_argument("--manifest", required=True)
    suite.add_argument("--limit", type=int)
    suite.add_argument("--allow-download", action="store_true")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--root", default="results")
    analysis.add_argument("--output", default="reports/figures")
    args = parser.parse_args(argv)
    if args.command == "environment":
        result = environment_report()
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "prepare":
        config = read_config(args.config)
        _, _, _, _, identity, provenance = setup(config, args.allow_download)
        config["model"]["revision"] = identity["revision"]
        import yaml

        destination = Path(args.config).with_suffix(".resolved.yaml")
        destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        write_json("reports/prepared_data_manifest.json", provenance)
        print(f"Prepared pinned config: {destination}")
    elif args.command in {"run", "evaluate"}:
        run(
            read_config(args.config),
            args.allow_download,
            getattr(args, "resume", None),
            args.command == "evaluate",
        )
    elif args.command == "grid":
        print(json.dumps(generate_grid(args.base, args.output, args.stage), indent=2))
    elif args.command == "suite":
        run_suite(args.manifest, args.allow_download, args.limit)
    elif args.command == "analyze":
        frame = analyze(args.root, args.output)
        print(f"Analyzed {len(frame)} completed runs")


if __name__ == "__main__":
    main()
