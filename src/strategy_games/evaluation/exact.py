"""Exact finite-horizon game-theoretic evaluation for Gridworld v2.

The game is simultaneous-move, zero-sum, and small enough for backward
induction. ``exploitability`` here is Nash exploitability, not the sampled
response gap.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import numpy as np

from strategy_games.envs.gridworld import ACTION_TO_DELTA
from strategy_games.envs.gridworld_v2 import GridworldV2Config, resolve_v2_outcome
from strategy_games.envs.pathfinding import apply_move, is_free
from strategy_games.evaluation.best_response import attacker_v2_heuristic_action
from strategy_games.strategies.controllers import parametric_attacker_action

AttackerPolicy = Callable[[Any], int]


class _StateView:
    """Minimal env-like view so existing controllers can be queried."""

    def __init__(
        self,
        config: GridworldV2Config,
        attacker_pos: tuple[int, int],
        defender_pos: tuple[int, int],
        steps: int,
    ) -> None:
        self.config = config
        self.attacker_pos = attacker_pos
        self.defender_pos = defender_pos
        self.steps = steps
        self.goal_positions = config.goal_positions
        self.obstacles = config.obstacles
        self.goal_pos = config.goal_pos


def _free_cells(config: GridworldV2Config) -> tuple[tuple[int, int], ...]:
    cells = [
        (y, x)
        for y in range(config.grid_size)
        for x in range(config.grid_size)
        if is_free((y, x), config.grid_size, config.obstacles)
    ]
    return tuple(cells)


def solve_matrix_nash(payoff: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (value, row_mixed, col_mixed) of a zero-sum matrix game."""

    if payoff.ndim != 2:
        raise ValueError("payoff must be 2D")
    n_row, n_col = payoff.shape
    # Pure Nash: row i maximizes the column's minimizing reply.
    for i in range(n_row):
        for j in range(n_col):
            if payoff[i, j] + 1e-9 >= payoff[:, j].max() and payoff[i, j] - 1e-9 <= payoff[i, :].min():
                row = np.zeros(n_row, dtype=np.float64)
                col = np.zeros(n_col, dtype=np.float64)
                row[i] = 1.0
                col[j] = 1.0
                return float(payoff[i, j]), row, col

    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required for mixed Nash solves") from exc

    # max v s.t. p^T A[:,j] >= v, sum p = 1, p >= 0
    c = np.zeros(n_row + 1, dtype=np.float64)
    c[-1] = -1.0
    a_ub = np.zeros((n_col, n_row + 1), dtype=np.float64)
    for j in range(n_col):
        a_ub[j, :n_row] = -payoff[:, j]
        a_ub[j, -1] = 1.0
    b_ub = np.zeros(n_col, dtype=np.float64)
    a_eq = np.zeros((1, n_row + 1), dtype=np.float64)
    a_eq[0, :n_row] = 1.0
    bounds = [(0.0, 1.0)] * n_row + [(None, None)]
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=np.array([1.0]), bounds=bounds, method="highs")
    if not result.success:
        # Conservative fallback: maximizer security level.
        security = float(payoff.min(axis=1).max())
        row = np.zeros(n_row, dtype=np.float64)
        row[int(payoff.min(axis=1).argmax())] = 1.0
        col = np.zeros(n_col, dtype=np.float64)
        col[int(payoff[int(row.argmax())].argmin())] = 1.0
        return security, row, col
    probs = np.clip(result.x[:n_row], 0.0, 1.0)
    if probs.sum() <= 0:
        probs = np.ones(n_row) / n_row
    else:
        probs = probs / probs.sum()
    value = float(result.x[-1])
    # Defender mix from the dual-ish best reply to p.
    expected = probs @ payoff
    col = np.zeros(n_col, dtype=np.float64)
    col[int(expected.argmin())] = 1.0
    return value, probs, col


@lru_cache(maxsize=8)
def nash_value(config: GridworldV2Config) -> float:
    """Attacker Nash value from the initial state, cached per config."""

    table = _nash_table(config)
    cells = _free_cells(config)
    index = {cell: i for i, cell in enumerate(cells)}
    return float(table[0, index[config.attacker_start], index[config.defender_start]])


def _nash_table(config: GridworldV2Config) -> np.ndarray:
    cells = _free_cells(config)
    n = len(cells)
    actions = tuple(ACTION_TO_DELTA)
    horizon = config.max_steps
    values = np.zeros((horizon + 1, n, n), dtype=np.float64)
    for t in range(horizon, -1, -1):
        for ia, att in enumerate(cells):
            for id_, deff in enumerate(cells):
                reward, done = resolve_v2_outcome(att, deff, t, config)[:2]
                if t > 0 and done:
                    # States after a terminal transition are not queried; keep 0.
                    values[t, ia, id_] = 0.0
                    continue
                if t == horizon:
                    values[t, ia, id_] = float(reward) if done else float(config.timeout_reward + config.step_penalty)
                    continue
                if t == 0:
                    # Initial state is not yet terminal unless spawn is illegal.
                    pass
                payoff = np.zeros((len(actions), len(actions)), dtype=np.float64)
                for ai, a_act in enumerate(actions):
                    for di, d_act in enumerate(actions):
                        payoff[ai, di] = _transition_value(att, deff, t, a_act, d_act, config, cells, values)
                values[t, ia, id_], _, _ = solve_matrix_nash(payoff)
    return values


def _transition_value(
    att: tuple[int, int],
    deff: tuple[int, int],
    t: int,
    a_act: int,
    d_act: int,
    config: GridworldV2Config,
    cells: tuple[tuple[int, int], ...],
    values: np.ndarray,
) -> float:
    index = getattr(_transition_value, "_index_cache", None)
    cache_key = id(cells)
    if index is None or _transition_value._cells_id != cache_key:  # type: ignore[attr-defined]
        _transition_value._index_cache = {cell: i for i, cell in enumerate(cells)}  # type: ignore[attr-defined]
        _transition_value._cells_id = cache_key  # type: ignore[attr-defined]
        index = _transition_value._index_cache  # type: ignore[attr-defined]

    next_att = apply_move(att, a_act, config.grid_size, config.obstacles)
    next_def = apply_move(deff, d_act, config.grid_size, config.obstacles)
    next_t = t + 1
    reward, done = resolve_v2_outcome(next_att, next_def, next_t, config)[:2]
    if done:
        return float(reward)
    return float(reward) + float(values[next_t, index[next_att], index[next_def]])


def best_response_value(config: GridworldV2Config, attacker_policy: AttackerPolicy) -> float:
    """Attacker value against an exact minimizing defender."""

    cells = _free_cells(config)
    index = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)
    horizon = config.max_steps
    values = np.zeros((horizon + 1, n, n), dtype=np.float64)
    actions = tuple(ACTION_TO_DELTA)
    for t in range(horizon, -1, -1):
        for ia, att in enumerate(cells):
            for id_, deff in enumerate(cells):
                if t == horizon:
                    reward, done = resolve_v2_outcome(att, deff, t, config)[:2]
                    values[t, ia, id_] = float(reward) if done else float(config.timeout_reward + config.step_penalty)
                    continue
                view = _StateView(config, att, deff, t)
                a_act = int(attacker_policy(view))
                if a_act not in ACTION_TO_DELTA:
                    a_act = 0
                best = float("inf")
                for d_act in actions:
                    val = _transition_value(att, deff, t, a_act, d_act, config, cells, values)
                    if val < best:
                        best = val
                values[t, ia, id_] = float(best)
    return float(values[0, index[config.attacker_start], index[config.defender_start]])


def exact_exploitability(config: GridworldV2Config, attacker_policy: AttackerPolicy) -> float:
    """Return ``V_nash - V(sigma, BR(sigma))`` from the attacker's perspective."""

    return float(nash_value(config) - best_response_value(config, attacker_policy))


def policy_from_strategy(strategy: Any, label: str | None = None, policy: Any | None = None) -> AttackerPolicy:
    """Build an attacker policy callable from a latent and optional label."""

    def _act(view: Any) -> int:
        if policy is not None:
            import torch

            from strategy_games.envs.gridworld_v2 import AttackerDefenderGridworldV2

            env = AttackerDefenderGridworldV2(view.config)
            env.attacker_pos = view.attacker_pos
            env.defender_pos = view.defender_pos
            env.steps = view.steps
            device = next(policy.parameters()).device
            state = torch.as_tensor(env.observe(), dtype=torch.float32, device=device)
            strategy_device = strategy.detach().float().to(device)
            return int(policy.act(state, strategy_device, deterministic=True))
        if label is not None:
            return attacker_v2_heuristic_action(view, label)
        return parametric_attacker_action(view, strategy)

    return _act


def exact_metrics_for_strategy(
    config: GridworldV2Config,
    strategy: Any,
    label: str | None = None,
    policy: Any | None = None,
) -> dict[str, float]:
    """Return exact Nash value, BR value, and exploitability for one strategy."""

    attacker_policy = policy_from_strategy(strategy, label=label, policy=policy)
    value_vs_br = best_response_value(config, attacker_policy)
    game_value = nash_value(config)
    return {
        "exact_value_vs_br": float(value_vs_br),
        "exact_nash_value": float(game_value),
        "exploitability": float(game_value - value_vs_br),
    }


def clear_exact_caches() -> None:
    """Drop cached Nash tables. Useful in tests that construct many configs."""

    nash_value.cache_clear()
