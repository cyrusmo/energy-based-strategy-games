"""Approximate game-theoretic evaluation via sampled opponent responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from strategy_games.envs.gridworld_v2 import (
    AttackerDefenderGridworldV2,
    GridworldV2Config,
    is_v2_env,
    make_gridworld,
)
from strategy_games.envs.pathfinding import apply_move, bfs_distance, bfs_first_action, nearest_goal, shortest_path_first_steps
from strategy_games.evaluation.exploitability import sampled_response_gap
from strategy_games.evaluation.robustness import robustness_score, robustness_score_g
from strategy_games.strategies.controllers import parametric_attacker_action
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
        env_config: GridworldConfig | GridworldV2Config | None = None,
        opponent_labels: tuple[str, ...] | None = None,
        episodes_per_opponent: int = 2,
        strategy_dim: int = 8,
        seed: int | None = None,
        compute_exact: bool = False,
        robustness_alpha: float = 0.5,
    ) -> None:
        if episodes_per_opponent < 1:
            raise ValueError("episodes_per_opponent must be positive")
        if strategy_dim < 1:
            raise ValueError("strategy_dim must be positive")
        if not 0.0 <= robustness_alpha <= 1.0:
            raise ValueError("robustness_alpha must be in [0, 1]")
        self.env_config = env_config or GridworldConfig()
        self.compute_exact = compute_exact
        self.robustness_alpha = robustness_alpha
        self.opponent_labels = opponent_labels or available_heuristic_strategies()
        self.episodes_per_opponent = episodes_per_opponent
        self.strategy_dim = strategy_dim
        self.rng = np.random.default_rng(seed)

    def evaluate_strategy(
        self,
        strategy: Tensor,
        label: str | None = None,
        policy: Any | None = None,
    ) -> dict[str, float | str]:
        """Return approximate values and robustness metrics for one strategy."""

        if strategy.ndim != 1:
            raise ValueError(f"strategy must be 1D, got {tuple(strategy.shape)}")

        results: list[RolloutResult] = []
        for opponent_label in self.opponent_labels:
            for _ in range(self.episodes_per_opponent):
                results.append(self.rollout(strategy, opponent_label, label=label, policy=policy))

        values = np.asarray([result.total_reward for result in results], dtype=np.float32)
        average_value = float(values.mean())
        worst_value = float(values.min())
        robustness = robustness_score(values)
        robustness_g = robustness_score_g(worst_value, average_value, self.robustness_alpha)
        response_gap = sampled_response_gap(average_value, worst_value)

        outcomes = [result.outcome for result in results]
        goal_rate = outcomes.count("goal") / len(outcomes)
        catch_rate = outcomes.count("caught") / len(outcomes)
        timeout_rate = outcomes.count("timeout") / len(outcomes)
        worst_idx = int(values.argmin())
        best_response_label = results[worst_idx].opponent_label

        metrics: dict[str, float | str] = {
            "average_case_value": average_value,
            "worst_case_value": worst_value,
            "robustness_score": robustness,
            "robustness_score_g": robustness_g,
            "sampled_response_gap": response_gap,
            "exploitability_proxy": response_gap,
            "goal_rate": float(goal_rate),
            "catch_rate": float(catch_rate),
            "timeout_rate": float(timeout_rate),
            "win_rate": float(goal_rate),
            "best_response_label": best_response_label,
            "env_steps": float(sum(result.steps for result in results)),
        }
        if self.compute_exact and isinstance(self.env_config, GridworldV2Config):
            from strategy_games.evaluation.exact import exact_metrics_for_strategy

            metrics.update(exact_metrics_for_strategy(self.env_config, strategy, label=label, policy=policy))
        return metrics

    def rollout(
        self,
        strategy: Tensor,
        opponent_label: str,
        label: str | None = None,
        policy: Any | None = None,
    ) -> RolloutResult:
        """Run one true-environment rollout for a candidate strategy."""

        env = make_gridworld(self.env_config)
        env.reset()
        total_reward = 0.0
        done = False
        info: dict[str, object] = {"outcome": "running", "steps": 0}

        while not done:
            if policy is None:
                attacker_action = attacker_heuristic_action(env, strategy, label=label)
            else:
                attacker_action = policy_action(env, policy, strategy)
            defender_action = defender_heuristic_action(env, opponent_label)
            result = env.step(attacker_action, defender_action)
            total_reward += result.reward
            done = result.done
            info = result.info

        raw_steps = info.get("steps", 0)
        steps = int(raw_steps) if isinstance(raw_steps, int | float | str) else 0
        return RolloutResult(
            total_reward=float(total_reward),
            outcome=str(info["outcome"]),
            steps=steps,
            opponent_label=opponent_label,
        )


def policy_action(env: AttackerDefenderGridworld, policy: Any, strategy: Tensor) -> int:
    """Return an action from a learned strategy-conditioned policy."""

    device = next(policy.parameters()).device
    state = torch.as_tensor(env.observe(), dtype=torch.float32, device=device)
    strategy_device = strategy.detach().float().to(device)
    return int(policy.act(state, strategy_device, deterministic=True))


def attacker_heuristic_action(env: AttackerDefenderGridworld, strategy: Tensor, label: str | None = None) -> int:
    """Map a latent strategy to an attacker action for evaluator rollouts."""

    if is_v2_env(env):
        if label is None:
            return parametric_attacker_action(env, strategy)
        return attacker_v2_heuristic_action(env, label)
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

    if is_v2_env(env):
        return defender_v2_heuristic_action(env, label)
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


def attacker_v2_heuristic_action(env: AttackerDefenderGridworldV2, label: str) -> int:
    """Behaviorally distinct attacker programs for v2 layouts."""

    grid_size = env.config.grid_size
    obstacles = env.obstacles
    primary = env.goal_positions[0]
    decoy = env.goal_positions[-1]
    if label == "direct_goal":
        target = nearest_goal(env.attacker_pos, env.goal_positions, grid_size, obstacles)
        return bfs_first_action(env.attacker_pos, target, grid_size, obstacles)
    if label == "aggressive":
        steps = shortest_path_first_steps(env.attacker_pos, primary, grid_size, obstacles)
        best_action, best_dist = steps[0][0], manhattan(steps[0][1], env.defender_pos)
        for action, nxt in steps:
            dist = manhattan(nxt, env.defender_pos)
            if dist < best_dist:
                best_action, best_dist = action, dist
        return best_action
    if label == "evasive":
        return evasive_v2_action(env)
    if label == "feint":
        guarded = nearest_goal(env.defender_pos, env.goal_positions, grid_size, obstacles)
        other = decoy if guarded == primary else primary
        if len(env.goal_positions) > 2:
            other = next(goal for goal in env.goal_positions if goal != guarded)
        commit_step = max(env.config.max_steps // 4, 2)
        target = guarded if env.steps < commit_step else other
        return bfs_first_action(env.attacker_pos, target, grid_size, obstacles)
    if label == "patient":
        hold_horizon = int(0.55 * env.config.max_steps)
        if env.steps < hold_horizon and manhattan(env.attacker_pos, env.defender_pos) > 1:
            return _hide_from_defender(env)
        open_goal = max(
            env.goal_positions,
            key=lambda goal: manhattan(env.defender_pos, goal),
        )
        return bfs_first_action(env.attacker_pos, open_goal, grid_size, obstacles)
    target = nearest_goal(env.attacker_pos, env.goal_positions, grid_size, obstacles)
    return bfs_first_action(env.attacker_pos, target, grid_size, obstacles)


def defender_v2_heuristic_action(env: AttackerDefenderGridworldV2, label: str) -> int:
    """Named defender programs occupying distinct points of the v2 strategy space."""

    grid_size = env.config.grid_size
    obstacles = env.obstacles
    primary = env.goal_positions[0]
    decoy = env.goal_positions[-1]
    if label == "aggressive":
        if manhattan(env.defender_pos, env.attacker_pos) <= 1:
            return 0
        return bfs_first_action(env.defender_pos, env.attacker_pos, grid_size, obstacles)
    if label == "direct_goal":
        return bfs_first_action(env.defender_pos, primary, grid_size, obstacles)
    if label == "patient":
        goal = nearest_goal(env.attacker_pos, env.goal_positions, grid_size, obstacles)
        intercept = ((env.attacker_pos[0] + goal[0]) // 2, (env.attacker_pos[1] + goal[1]) // 2)
        intercept = _nearest_free(intercept, grid_size, obstacles)
        return bfs_first_action(env.defender_pos, intercept, grid_size, obstacles)
    if label == "feint":
        target = primary if env.steps % 4 < 2 else env.attacker_pos
        return bfs_first_action(env.defender_pos, target, grid_size, obstacles)
    if label == "evasive":
        return bfs_first_action(env.defender_pos, decoy, grid_size, obstacles)
    if manhattan(env.defender_pos, env.attacker_pos) <= 1:
        return 0
    return bfs_first_action(env.defender_pos, env.attacker_pos, grid_size, obstacles)


def evasive_v2_action(env: AttackerDefenderGridworldV2) -> int:
    """Trade BFS progress to the nearest goal against distance from the defender."""

    grid_size = env.config.grid_size
    obstacles = env.obstacles
    target = nearest_goal(env.attacker_pos, env.goal_positions, grid_size, obstacles)
    best_action = 0
    best_score = -float("inf")
    for action in ACTION_TO_DELTA:
        nxt = apply_move(env.attacker_pos, action, grid_size, obstacles)
        goal_dist = bfs_distance(nxt, target, grid_size, obstacles)
        if goal_dist is None:
            continue
        score = -float(goal_dist) + 0.75 * float(manhattan(nxt, env.defender_pos))
        if nxt == env.defender_pos:
            score -= 10.0
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _hide_from_defender(env: AttackerDefenderGridworldV2) -> int:
    """Stay put unless a neighbor is strictly farther from the defender."""

    grid_size = env.config.grid_size
    obstacles = env.obstacles
    best_action = 0
    best_dist = manhattan(env.attacker_pos, env.defender_pos)
    for action in ACTION_TO_DELTA:
        nxt = apply_move(env.attacker_pos, action, grid_size, obstacles)
        if nxt == env.defender_pos:
            continue
        dist = manhattan(nxt, env.defender_pos)
        if dist > best_dist:
            best_dist = dist
            best_action = action
    return best_action


def _nearest_free(pos: Position, grid_size: int, obstacles: frozenset[Position]) -> Position:
    from strategy_games.envs.pathfinding import is_free

    if is_free(pos, grid_size, obstacles):
        return pos
    for radius in range(1, grid_size + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                candidate = (pos[0] + dy, pos[1] + dx)
                if is_free(candidate, grid_size, obstacles):
                    return candidate
    return pos


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
