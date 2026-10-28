"""Inspect real greedy continuations from the completed educational smoke run."""

import torch
from lowrank_lab.models import TinyConfig, TinyTransformer
from lowrank_lab.experiments.runner import read_config
from lowrank_lab.utils.io import write_json


def main():
    config = read_config("configs/smoke.yaml")
    model = TinyTransformer(TinyConfig(**config["model"]["tiny"]))
    model.load_state_dict(
        torch.load("results/smoke/model.pt", map_location="cpu", weights_only=True)
    )
    prefixes = torch.tensor([[1, 8, 9, 8], [1, 4, 12, 4]])
    generated = model.generate(prefixes, max_new_tokens=8)
    result = {
        "purpose": "educational random-initialized model, not a pretrained benchmark",
        "prefixes": prefixes.tolist(),
        "greedy_sequences": generated.tolist(),
    }
    write_json("results/smoke/generation.json", result)
    print(result)


if __name__ == "__main__":
    main()
