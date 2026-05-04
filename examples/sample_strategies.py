"""Sample named and EBM-generated strategy embeddings."""

from __future__ import annotations

import json

import torch

from strategy_games.models.ebm import EnergyMLP
from strategy_games.strategies.embeddings import available_heuristic_strategies, named_strategy_embedding
from strategy_games.strategies.sampler import langevin_sample


def main() -> None:
    strategy_dim = 8
    ebm = EnergyMLP(strategy_dim=strategy_dim, hidden_dim=32)
    named = torch.stack([named_strategy_embedding(label, strategy_dim).vector for label in available_heuristic_strategies()])
    sampled = langevin_sample(ebm, num_samples=4, strategy_dim=strategy_dim, steps=5)
    payload = {
        "named_labels": list(available_heuristic_strategies()),
        "named_shape": list(named.shape),
        "sampled_shape": list(sampled.shape),
        "sampled_energy": [float(x) for x in ebm(sampled).detach()],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
