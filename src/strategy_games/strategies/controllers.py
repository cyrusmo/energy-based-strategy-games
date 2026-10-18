"""Continuous sigma-to-action controllers for v2 strategy embeddings.

Named heuristic prototypes occupy points in this space. Arbitrary latent
vectors are executed by the same weighted controller so the evaluator can
score continuous strategies rather than a 5-class lookup.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from strategy_games.envs.gridworld import ACTION_TO_DELTA, manhattan
from strategy_games.envs.pathfinding import apply_move, bfs_distance, nearest_goal

STRATEGY_WEIGHT_NAMES = (
    "directness",
    "evasiveness",
    "aggression",
    "patience",
    "feint",
    "commit_time",
)


def strategy_weights(strategy: Tensor) -> dict[str, float]:
    """Read the interpretable controller weights from a latent vector."""

    values = [float(strategy[i].item()) if i < strategy.numel() else 0.0 for i in range(len(STRATEGY_WEIGHT_NAMES))]
    commit = values[5]
    commit_frac = max(0.15, min(0.75, 0.35 + 0.25 * commit))
    return {
        "directness": values[0],
        "evasiveness": values[1],
        "aggression": values[2],
        "patience": values[3],
        "feint": values[4],
        "commit_frac": commit_frac,
    }


def defender_committed(env: Any) -> bool:
    """True when the defender is closer to a goal than to the attacker."""

    goals = getattr(env, "goal_positions", (env.goal_pos,))
    obstacles = getattr(env, "obstacles", frozenset())
    grid_size = env.config.grid_size
    to_attacker = manhattan(env.defender_pos, env.attacker_pos)
    goal_distances = [
        bfs_distance(env.defender_pos, goal, grid_size, obstacles) for goal in goals
    ]
    reachable = [dist for dist in goal_distances if dist is not None]
    if not reachable:
        return False
    return min(reachable) < to_attacker


def parametric_attacker_action(env: Any, strategy: Tensor) -> int:
    """Choose an attacker action from continuous controller weights."""

    weights = strategy_weights(strategy)
    obstacles = getattr(env, "obstacles", frozenset())
    grid_size = env.config.grid_size
    goals = getattr(env, "goal_positions", (env.goal_pos,))
    primary = goals[0]
    decoy = goals[-1]
    commit_step = int(weights["commit_frac"] * env.config.max_steps)
    if weights["feint"] >= 0.5 and env.steps < commit_step:
        preferred = decoy
    elif weights["directness"] >= weights["aggression"]:
        preferred = nearest_goal(env.attacker_pos, goals, grid_size, obstacles)
    else:
        preferred = primary

    if weights["patience"] >= 0.5 and not defender_committed(env) and env.steps < int(0.66 * env.config.max_steps):
        return _safest_action(env)

    best_action = 0
    best_score = -float("inf")
    for action in ACTION_TO_DELTA:
        nxt = apply_move(env.attacker_pos, action, grid_size, obstacles)
        goal_dist = bfs_distance(nxt, preferred, grid_size, obstacles)
        if goal_dist is None:
            continue
        def_dist = float(manhattan(nxt, env.defender_pos))
        score = (
            weights["directness"] * (-float(goal_dist))
            + weights["evasiveness"] * def_dist
            + weights["aggression"] * (-def_dist)
        )
        if nxt in goals:
            score += 10.0
        if nxt == env.defender_pos:
            score -= 5.0
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def parametric_defender_action(env: Any, strategy: Tensor) -> int:
    """Choose a defender action from the same latent controller space."""

    weights = strategy_weights(strategy)
    obstacles = getattr(env, "obstacles", frozenset())
    grid_size = env.config.grid_size
    goals = getattr(env, "goal_positions", (env.goal_pos,))
    primary = goals[0]
    decoy = goals[-1]
    if manhattan(env.defender_pos, env.attacker_pos) <= 1 and weights["aggression"] >= 0.3:
        return 0
    if weights["feint"] >= 0.5 and env.steps % 4 >= 2:
        target = env.attacker_pos
    elif weights["evasiveness"] >= 0.5:
        target = decoy
    elif weights["patience"] >= 0.5:
        target = _intercept_point(env)
    elif weights["directness"] >= weights["aggression"]:
        target = primary
    else:
        target = env.attacker_pos
    from strategy_games.envs.pathfinding import bfs_first_action

    return bfs_first_action(env.defender_pos, target, grid_size, obstacles)


def _safest_action(env: Any) -> int:
    obstacles = getattr(env, "obstacles", frozenset())
    grid_size = env.config.grid_size
    best_action = 0
    best_dist = manhattan(env.attacker_pos, env.defender_pos)
    for action in ACTION_TO_DELTA:
        nxt = apply_move(env.attacker_pos, action, grid_size, obstacles)
        dist = manhattan(nxt, env.defender_pos)
        if nxt == env.defender_pos:
            continue
        if dist > best_dist:
            best_dist = dist
            best_action = action
    return best_action


def _intercept_point(env: Any) -> tuple[int, int]:
    goals = getattr(env, "goal_positions", (env.goal_pos,))
    obstacles = getattr(env, "obstacles", frozenset())
    grid_size = env.config.grid_size
    goal = nearest_goal(env.attacker_pos, goals, grid_size, obstacles)
    return ((env.attacker_pos[0] + goal[0]) // 2, (env.attacker_pos[1] + goal[1]) // 2)
