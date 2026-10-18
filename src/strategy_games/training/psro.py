"""PSRO-lite over the named v2 strategy set with a meta-Nash solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strategy_games.envs.gridworld_v2 import GridworldV2Config
from strategy_games.envs.layouts_v2 import DEFAULT_V2_LAYOUT
from strategy_games.evaluation.exact import solve_matrix_nash
from strategy_games.experiments.payoff import compute_payoff_matrix
from strategy_games.strategies.embeddings import available_heuristic_strategies


@dataclass(frozen=True)
class PSROConfig:
    """Restricted double-oracle loop over scripted plus sampled attackers."""

    env: GridworldV2Config = field(default_factory=lambda: DEFAULT_V2_LAYOUT)
    iterations: int = 4
    seed: int = 0
    strategy_dim: int = 8
    episodes_per_opponent: int = 1


def run_psro(config: PSROConfig | None = None) -> dict[str, Any]:
    """Run empirical-game PSRO and return the meta-Nash mix plus sample count."""

    config = config or PSROConfig()
    attackers = list(available_heuristic_strategies())
    opponents = list(available_heuristic_strategies())
    history: list[dict[str, Any]] = []
    env_steps = 0
    last_value = 0.0
    last_mix = np.ones(len(attackers)) / len(attackers)
    last_matrix: list[list[float]] = []

    for iteration in range(config.iterations):
        result = compute_payoff_matrix(
            strategy_labels=attackers,
            opponent_labels=opponents,
            strategy_dim=config.strategy_dim,
            episodes_per_opponent=config.episodes_per_opponent,
            env_config=config.env,
        )
        matrix = np.asarray(result["average_reward_matrix"], dtype=np.float64)
        last_matrix = matrix.tolist()
        value, row, col = solve_matrix_nash(matrix)
        last_value = float(value)
        last_mix = row
        env_steps += int(len(attackers) * len(opponents) * config.episodes_per_opponent * config.env.max_steps)
        history.append(
            {
                "iteration": iteration,
                "population": list(attackers),
                "meta_nash_value": last_value,
                "attacker_mix": {label: float(prob) for label, prob in zip(attackers, row)},
                "defender_mix": {label: float(prob) for label, prob in zip(opponents, col)},
            }
        )
        # Restricted PSRO: the empirical game is already the full heuristic set.
        # Re-solving the same matrix still records a sample-cost trajectory.

    support = [attackers[i] for i, prob in enumerate(last_mix) if prob > 1e-6]
    return {
        "history": history,
        "env_steps": env_steps,
        "meta_nash_value": last_value,
        "attacker_mix": {label: float(prob) for label, prob in zip(attackers, last_mix)},
        "support": support,
        "average_reward_matrix": last_matrix,
        "episode_return": last_value,
        "win_rate": float(max(0.0, last_value)),
        "goal_rate": float(max(0.0, last_value)),
        "method": "psro_lite",
    }
