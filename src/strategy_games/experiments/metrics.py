"""Metrics used by public examples and early experiments."""

from __future__ import annotations

import numpy as np
from torch import Tensor

from strategy_games.strategies.embeddings import pairwise_diversity


def summarize_episode_returns(returns: list[float]) -> dict[str, float]:
    """Return common scalar summaries for episode returns."""

    if not returns:
        raise ValueError("returns must be non-empty")
    array = np.asarray(returns, dtype=np.float32)
    return {
        "episode_return": float(array.mean()),
        "episode_return_std": float(array.std()),
        "episode_return_min": float(array.min()),
        "episode_return_max": float(array.max()),
    }


def compute_strategy_diversity(strategies: Tensor) -> float:
    """Mean pairwise distance among strategy embeddings."""

    return pairwise_diversity(strategies)


def outcome_rates(outcomes: list[str]) -> dict[str, float]:
    """Compute goal/catch/timeout rates."""

    if not outcomes:
        raise ValueError("outcomes must be non-empty")
    n = len(outcomes)
    return {
        "win_rate": float(outcomes.count("goal") / n),
        "goal_rate": float(outcomes.count("goal") / n),
        "catch_rate": float(outcomes.count("caught") / n),
        "timeout_rate": float(outcomes.count("timeout") / n),
    }
