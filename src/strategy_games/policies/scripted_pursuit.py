"""Deterministic and random scripted policies for pursuit/evasion traces."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitEnv
from strategy_games.envs.pursuit_actions import (
    ACTIONS,
    Position,
    greedy_action_away_from,
    greedy_action_towards,
    manhattan,
    move_position,
)

PURSUER_POLICIES = ("pursuer_greedy_nearest", "pursuer_random")
EVADER_POLICIES = ("evader_flee_nearest", "evader_random", "evader_goal_directed", "evader_feint")


def scripted_pursuit_actions(
    env: MultiEvaderPursuitEnv,
    pursuer_policy: str = "pursuer_greedy_nearest",
    evader_policy: str = "evader_flee_nearest",
    rng: np.random.Generator | None = None,
    step_index: int = 0,
    feint_steps: int = 3,
) -> dict[str, str]:
    """Return actions for all agents under named scripted policies."""

    rng = rng or np.random.default_rng(0)
    if pursuer_policy not in PURSUER_POLICIES:
        raise ValueError(f"unsupported pursuer policy: {pursuer_policy}")
    if evader_policy not in EVADER_POLICIES:
        raise ValueError(f"unsupported evader policy: {evader_policy}")

    actions: dict[str, str] = {}
    for pursuer_id in env.pursuer_ids:
        actions[pursuer_id] = _pursuer_action(env, pursuer_id, pursuer_policy, rng)
    for evader_id in env.evader_ids:
        if evader_id not in env.active_evaders:
            actions[evader_id] = "stay"
        else:
            actions[evader_id] = _evader_action(env, evader_id, evader_policy, rng, step_index, feint_steps)
    return actions


def _pursuer_action(
    env: MultiEvaderPursuitEnv,
    pursuer_id: str,
    policy: str,
    rng: np.random.Generator,
) -> str:
    if policy == "pursuer_random":
        return str(rng.choice(ACTIONS))
    if not env.active_evaders:
        return "stay"
    position = env.pursuer_positions[pursuer_id]
    target_id = _nearest(position, {evader_id: env.evader_positions[evader_id] for evader_id in env.active_evaders})
    return greedy_action_towards(position, env.evader_positions[target_id], env.grid_size)


def _evader_action(
    env: MultiEvaderPursuitEnv,
    evader_id: str,
    policy: str,
    rng: np.random.Generator,
    step_index: int,
    feint_steps: int,
) -> str:
    if policy == "evader_random":
        return str(rng.choice(ACTIONS))
    if policy == "evader_goal_directed":
        return _evader_goal_directed(env, evader_id)
    if policy == "evader_feint" and step_index < feint_steps:
        return _evader_goal_directed(env, evader_id)
    return _evader_flee_nearest(env, evader_id)


def _evader_goal_directed(env: MultiEvaderPursuitEnv, evader_id: str) -> str:
    return greedy_action_towards(env.evader_positions[evader_id], env.evader_goals[evader_id], env.grid_size)


def _evader_flee_nearest(env: MultiEvaderPursuitEnv, evader_id: str) -> str:
    position = env.evader_positions[evader_id]
    nearest_pursuer = _nearest(position, env.pursuer_positions)
    threat = env.pursuer_positions[nearest_pursuer]
    return _best_flee_action(position, threat, env.pursuer_positions, env.grid_size)


def _best_flee_action(
    position: Position,
    threat: Position,
    pursuer_positions: Mapping[str, Position],
    grid_size: tuple[int, int],
) -> str:
    best_action = greedy_action_away_from(position, threat, grid_size)
    best_distance = _min_distance_to_pursuers(move_position(position, best_action, grid_size), pursuer_positions)
    for action in ACTIONS:
        candidate = move_position(position, action, grid_size)
        distance = _min_distance_to_pursuers(candidate, pursuer_positions)
        if distance > best_distance:
            best_action = action
            best_distance = distance
    return best_action


def _nearest(source: Position, targets: Mapping[str, Position]) -> str:
    if not targets:
        raise ValueError("targets must be non-empty")
    return min(targets, key=lambda agent_id: (manhattan(source, targets[agent_id]), agent_id))


def _min_distance_to_pursuers(position: Position, pursuer_positions: Mapping[str, Position]) -> int:
    return min(manhattan(position, pursuer_pos) for pursuer_pos in pursuer_positions.values())
