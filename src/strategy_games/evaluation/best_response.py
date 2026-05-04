"""Approximate game-theoretic evaluation via sampled opponent responses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from strategy_games.envs.gridworld import (
    ACTION_TO_DELTA,
    AttackerDefenderGridworld,
    GridworldConfig,
    Position,
    clip_position,
    greedy_action_away_from,
    greedy_action_towards,
    manhattan,
)
from strategy_games.evaluation.exploitability import exploitability_proxy
from strategy_games.evaluation.robustness import robustness_score
from strategy_games.strategies.embeddings import available_heuristic_strategies, named_strategy_embedding


@dataclass(frozen=True)
class RolloutResult:
    """Summary of one candidate-vs-opponent rollout."""

    total_reward: float
    outcome: str
    steps: int
    opponent_label: str


class GameTheoreticEvaluator:
    """Evaluate candidate attacker strategies against sampled defender responses."""

    def __init__(
        self,
        env_config: GridworldConfig | None = None,
        opponent_labels: tuple[str, ...] | None = None,
        episodes_per_opponent: int = 2,
        strategy_dim: int = 8,
        seed: int | None = None,
    ) -> None:
        if episodes_per_opponent < 1:
            raise ValueError("episodes_per_opponent must be positive")
        if strategy_dim < 1:
            raise ValueError("strategy_dim must be positive")
        self.env_config = env_config or GridworldConfig()
        self.opponent_labels = opponent_labels or available_heuristic_strategies()
        self.episodes_per_opponent = episodes_per_opponent
        self.strategy_dim = strategy_dim
        self.rng = np.random.default_rng(seed)

    def evaluate_strategy(self, strategy: Tensor, label: str | None = None) -> dict[str, float | str]:
        """Return approximate values and robustness metrics for one strategy."""

        if strategy.ndim != 1:
            raise ValueError(f"strategy must be 1D, got {tuple(strategy.shape)}")

        results: list[RolloutResult] = []
        for opponent_label in self.opponent_labels:
            for _ in range(self.episodes_per_opponent):
                results.append(self.rollout(strategy, opponent_label, label=label))

        values = np.asarray([result.total_reward for result in results], dtype=np.float32)
        average_value = float(values.mean())
        worst_value = float(values.min())
        robustness = robustness_score(values)
        exploitability = exploitability_proxy(average_value, worst_value)

        outcomes = [result.outcome for result in results]
        goal_rate = outcomes.count("goal") / len(outcomes)
        catch_rate = outcomes.count("caught") / len(outcomes)
        timeout_rate = outcomes.count("timeout") / len(outcomes)
        worst_idx = int(values.argmin())
        best_response_label = results[worst_idx].opponent_label

        return {
            "average_case_value": average_value,
            "worst_case_value": worst_value,
            "robustness_score": robustness,
            "exploitability_proxy": exploitability,
            "goal_rate": float(goal_rate),
            "catch_rate": float(catch_rate),
            "timeout_rate": float(timeout_rate),
            "win_rate": float(goal_rate),
            "best_response_label": best_response_label,
        }

    def rollout(self, strategy: Tensor, opponent_label: str, label: str | None = None) -> RolloutResult:
        """Run one true-environment rollout for a candidate strategy."""

        env = AttackerDefenderGridworld(self.env_config)
        env.reset()
        total_reward = 0.0
        done = False
        info: dict[str, object] = {"outcome": "running", "steps": 0}

        while not done:
            attacker_action = attacker_heuristic_action(env, strategy, label=label)
            defender_action = defender_heuristic_action(env, opponent_label)
            result = env.step(attacker_action, defender_action)
            total_reward += result.reward
            done = result.done
            info = result.info

        return RolloutResult(
            total_reward=float(total_reward),
            outcome=str(info["outcome"]),
            steps=int(info["steps"]),
            opponent_label=opponent_label,
        )


def attacker_heuristic_action(env: AttackerDefenderGridworld, strategy: Tensor, label: str | None = None) -> int:
    """Map a latent strategy to an attacker action for evaluator rollouts."""

    inferred_label = label or infer_strategy_label(strategy)
    if inferred_label == "direct_goal":
        return greedy_action_towards(env.attacker_pos, env.goal_pos)
    if inferred_label == "evasive":
        return evasive_goal_action(env.attacker_pos, env.goal_pos, env.defender_pos, env.config.grid_size)
    if inferred_label == "aggressive":
        return greedy_action_towards(env.attacker_pos, env.goal_pos)
    if inferred_label == "patient":
        if env.steps < env.config.max_steps // 3:
            return greedy_action_away_from(env.attacker_pos, env.defender_pos, env.config.grid_size)
        return greedy_action_towards(env.attacker_pos, env.goal_pos)
    if inferred_label == "feint":
        waypoint = (env.config.grid_size - 1, env.config.grid_size - 1)
        target = waypoint if env.steps < env.config.max_steps // 3 else env.goal_pos
        return greedy_action_towards(env.attacker_pos, target)
    return greedy_action_towards(env.attacker_pos, env.goal_pos)


def defender_heuristic_action(env: AttackerDefenderGridworld, label: str) -> int:
    """Return a defender action for a named approximate best response."""

    if label == "aggressive":
        return greedy_action_towards(env.defender_pos, env.attacker_pos)
    if label == "direct_goal":
        return greedy_action_towards(env.defender_pos, env.goal_pos)
    if label == "patient":
        intercept = midpoint(env.attacker_pos, env.goal_pos)
        return greedy_action_towards(env.defender_pos, intercept)
    if label == "feint":
        target = env.goal_pos if env.steps % 4 < 2 else env.attacker_pos
        return greedy_action_towards(env.defender_pos, target)
    if label == "evasive":
        return greedy_action_towards(env.defender_pos, env.attacker_pos)
    return greedy_action_towards(env.defender_pos, env.attacker_pos)


def infer_strategy_label(strategy: Tensor) -> str:
    """Infer the closest named heuristic prototype by Euclidean distance."""

    strategy_dim = int(strategy.shape[0])
    labels = available_heuristic_strategies()
    prototypes = torch.stack([named_strategy_embedding(label, strategy_dim).vector for label in labels], dim=0)
    distances = torch.norm(prototypes - strategy.detach().cpu(), dim=-1)
    return labels[int(torch.argmin(distances).item())]


def evasive_goal_action(source: Position, goal: Position, defender: Position, grid_size: int) -> int:
    """Choose a move that trades off goal progress against defender distance."""

    best_action = 0
    best_score = -float("inf")
    for action, delta in ACTION_TO_DELTA.items():
        candidate = clip_position((source[0] + delta[0], source[1] + delta[1]), grid_size)
        score = -manhattan(candidate, goal) + 0.75 * manhattan(candidate, defender)
        if score > best_score:
            best_action = action
            best_score = score
    return best_action


def midpoint(a: Position, b: Position) -> Position:
    """Return rounded midpoint between two positions."""

    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
