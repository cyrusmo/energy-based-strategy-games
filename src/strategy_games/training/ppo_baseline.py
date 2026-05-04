"""Baseline policies.

Full PPO is intentionally left as a near-term TODO. The file exposes random and
heuristic baselines so the scaffold can produce meaningful debug rollouts now.
"""

from __future__ import annotations

import numpy as np

from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig, greedy_action_towards
from strategy_games.models.policy import RandomPolicy


def run_random_policy_baseline(episodes: int = 5, seed: int | None = 0, env_config: GridworldConfig | None = None) -> dict[str, float]:
    """Run a uniform random attacker against a greedy defender."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    env_config = env_config or GridworldConfig()
    policy = RandomPolicy(action_dim=AttackerDefenderGridworld.action_dim, seed=seed)
    returns: list[float] = []
    outcomes: list[str] = []
    for _ in range(episodes):
        env = AttackerDefenderGridworld(env_config)
        obs = env.reset()
        done = False
        total = 0.0
        info = {"outcome": "running"}
        while not done:
            action = policy.act()
            result = env.step(action)
            obs = result.observation
            del obs
            total += result.reward
            done = result.done
            info = result.info
        returns.append(total)
        outcomes.append(str(info["outcome"]))
    return _summarize(returns, outcomes)


def run_direct_goal_baseline(episodes: int = 5, env_config: GridworldConfig | None = None) -> dict[str, float]:
    """Run a direct-to-goal attacker against a greedy defender."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    env_config = env_config or GridworldConfig()
    returns: list[float] = []
    outcomes: list[str] = []
    for _ in range(episodes):
        env = AttackerDefenderGridworld(env_config)
        env.reset()
        done = False
        total = 0.0
        info = {"outcome": "running"}
        while not done:
            action = greedy_action_towards(env.attacker_pos, env.goal_pos)
            result = env.step(action)
            total += result.reward
            done = result.done
            info = result.info
        returns.append(total)
        outcomes.append(str(info["outcome"]))
    return _summarize(returns, outcomes)


def train_ppo_placeholder() -> None:
    """Research TODO for a full PPO baseline implementation."""

    raise NotImplementedError("PPO baseline is scaffolded but not implemented yet.")


def _summarize(returns: list[float], outcomes: list[str]) -> dict[str, float]:
    return {
        "episode_return": float(np.mean(returns)),
        "win_rate": float(outcomes.count("goal") / len(outcomes)),
        "goal_rate": float(outcomes.count("goal") / len(outcomes)),
        "catch_rate": float(outcomes.count("caught") / len(outcomes)),
        "timeout_rate": float(outcomes.count("timeout") / len(outcomes)),
    }
