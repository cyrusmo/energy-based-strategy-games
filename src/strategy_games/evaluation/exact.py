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
from strategy_games.envs.gridworld_v2 import GridworldV2Config, defender_acts_this_step, resolve_v2_outcome
from strategy_games.envs.pathfinding import apply_move, is_free
from strategy_games.evaluation.best_response import attacker_v2_heuristic_action
from strategy_games.strategies.controllers import parametric_attacker_action

AttackerPolicy = Callable[[Any], int | np.ndarray]


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

    values, _ = _nash_solution(config)
    cells = _free_cells(config)
    index = {cell: i for i, cell in enumerate(cells)}
    return float(values[0, index[config.attacker_start], index[config.defender_start]])


@lru_cache(maxsize=8)
def _nash_solution(config: GridworldV2Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, row_mixes) of the finite-horizon Nash DP."""

    cells = _free_cells(config)
    n = len(cells)
    actions = tuple(ACTION_TO_DELTA)
    n_actions = len(actions)
    horizon = config.max_steps
    values = np.zeros((horizon + 1, n, n), dtype=np.float64)
    mixes = np.zeros((horizon + 1, n, n, n_actions), dtype=np.float64)
    mixes[..., :] = 1.0 / n_actions
    for t in range(horizon, -1, -1):
        for ia, att in enumerate(cells):
            for id_, deff in enumerate(cells):
                reward, done = resolve_v2_outcome(att, deff, t, config)[:2]
                if t > 0 and done:
                    values[t, ia, id_] = 0.0
                    continue
                if t == horizon:
                    values[t, ia, id_] = float(reward) if done else float(config.timeout_reward + config.step_penalty)
                    continue
                payoff = np.zeros((n_actions, n_actions), dtype=np.float64)
                for ai, a_act in enumerate(actions):
                    for di, d_act in enumerate(actions):
                        payoff[ai, di] = _transition_value(att, deff, t, a_act, d_act, config, cells, values)
                value, row, _ = solve_matrix_nash(payoff)
                values[t, ia, id_] = float(value)
                mixes[t, ia, id_] = row
    return values, mixes


def nash_attacker_table(config: GridworldV2Config) -> np.ndarray:
    """Per-state Nash row mix. Exploitability of this table is about 0."""

    _, mixes = _nash_solution(config)
    return mixes


def uniform_random_table(config: GridworldV2Config) -> np.ndarray:
    """Uniform 1/5 action mix at every joint state."""

    cells = _free_cells(config)
    n = len(cells)
    n_actions = len(ACTION_TO_DELTA)
    table = np.full((config.max_steps + 1, n, n, n_actions), 1.0 / n_actions, dtype=np.float64)
    return table


def action_probs_table(config: GridworldV2Config, attacker_policy: AttackerPolicy) -> np.ndarray:
    """Evaluate a policy once per joint state.

    Deterministic policies become one-hot rows; stochastic policies keep their
    probability vectors. Shape is ``[T+1, n_cells, n_cells, n_actions]``.
    """

    cells = _free_cells(config)
    n = len(cells)
    actions = tuple(ACTION_TO_DELTA)
    table = np.zeros((config.max_steps + 1, n, n, len(actions)), dtype=np.float64)
    for t in range(config.max_steps + 1):
        for ia, att in enumerate(cells):
            for id_, deff in enumerate(cells):
                view = _StateView(config, att, deff, t)
                table[t, ia, id_] = _attacker_action_probs(attacker_policy, view, actions)
    return table


def mixture_table(tables: list[np.ndarray], weights: np.ndarray | list[float]) -> np.ndarray:
    """Weighted sum of component tables.

    This is a Markov behavioral mixture: at each state the attacker mixes
    independently. A draw-once-per-episode mixture would need a belief-state DP.
    """

    if not tables:
        raise ValueError("tables must be non-empty")
    probs = np.asarray(weights, dtype=np.float64).reshape(-1)
    if probs.size != len(tables):
        raise ValueError("weights must match the number of tables")
    probs = np.clip(probs, 0.0, None)
    if float(probs.sum()) <= 0.0:
        probs = np.ones(len(tables), dtype=np.float64) / len(tables)
    else:
        probs = probs / probs.sum()
    mixed = np.zeros_like(tables[0], dtype=np.float64)
    for weight, table in zip(probs, tables):
        mixed += float(weight) * np.asarray(table, dtype=np.float64)
    totals = mixed.sum(axis=-1, keepdims=True)
    totals = np.where(totals <= 0.0, 1.0, totals)
    return mixed / totals


def best_response_value_from_table(config: GridworldV2Config, table: np.ndarray) -> float:
    """Attacker value against an exact minimizing defender, given a mix table."""

    cells = _free_cells(config)
    index = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)
    horizon = config.max_steps
    actions = tuple(ACTION_TO_DELTA)
    if table.shape != (horizon + 1, n, n, len(actions)):
        raise ValueError(f"table shape {table.shape} does not match config")
    values = np.zeros((horizon + 1, n, n), dtype=np.float64)
    for t in range(horizon, -1, -1):
        for ia, att in enumerate(cells):
            for id_, deff in enumerate(cells):
                if t == horizon:
                    reward, done = resolve_v2_outcome(att, deff, t, config)[:2]
                    values[t, ia, id_] = float(reward) if done else float(config.timeout_reward + config.step_penalty)
                    continue
                action_probs = table[t, ia, id_]
                best = float("inf")
                for d_act in actions:
                    val = 0.0
                    for ai, a_act in enumerate(actions):
                        weight = float(action_probs[ai])
                        if weight <= 0.0:
                            continue
                        val += weight * _transition_value(att, deff, t, a_act, d_act, config, cells, values)
                    if val < best:
                        best = val
                values[t, ia, id_] = float(best)
    return float(values[0, index[config.attacker_start], index[config.defender_start]])


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
    applied_def = d_act if defender_acts_this_step(t, config.defender_period) else 0
    next_def = apply_move(deff, applied_def, config.grid_size, config.obstacles)
    next_t = t + 1
    reward, done = resolve_v2_outcome(next_att, next_def, next_t, config)[:2]
    if done:
        return float(reward)
    return float(reward) + float(values[next_t, index[next_att], index[next_def]])


def _attacker_action_probs(attacker_policy: AttackerPolicy, view: Any, actions: tuple[int, ...]) -> np.ndarray:
    raw = attacker_policy(view)
    n_actions = len(actions)
    if isinstance(raw, (int, np.integer)):
        probs = np.zeros(n_actions, dtype=np.float64)
        action = int(raw)
        if action not in ACTION_TO_DELTA:
            action = 0
        probs[actions.index(action)] = 1.0
        return probs
    probs = np.asarray(raw, dtype=np.float64).reshape(-1)
    if probs.size != n_actions:
        raise ValueError(f"attacker policy must return an action or {n_actions} probabilities")
    probs = np.clip(probs, 0.0, None)
    if float(probs.sum()) <= 0.0:
        return np.ones(n_actions, dtype=np.float64) / n_actions
    return probs / probs.sum()


def best_response_value(config: GridworldV2Config, attacker_policy: AttackerPolicy) -> float:
    """Attacker value against an exact minimizing defender.

    ``attacker_policy`` may return a deterministic action or a vector of action
    probabilities. Mixtures (PSRO meta-Nash) use the latter.
    """

    return best_response_value_from_table(config, action_probs_table(config, attacker_policy))


def exact_exploitability(config: GridworldV2Config, attacker_policy: AttackerPolicy) -> float:
    """Return ``V_nash - V(sigma, BR(sigma))`` from the attacker's perspective."""

    return float(nash_value(config) - best_response_value(config, attacker_policy))


def _actor_critic_state(policy: Any, view: Any):
    import torch

    from strategy_games.envs.gridworld_v2 import AttackerDefenderGridworldV2

    env = AttackerDefenderGridworldV2(view.config)
    env.attacker_pos = view.attacker_pos
    env.defender_pos = view.defender_pos
    env.steps = view.steps
    device = next(policy.parameters()).device
    state = torch.as_tensor(env.observe(), dtype=torch.float32, device=device)
    return state, device


def policy_from_actor_critic(policy: Any) -> AttackerPolicy:
    """Adapt a PPO ``ActorCriticPolicy`` to a deterministic (argmax) attacker."""

    def _act(view: Any) -> int:
        state, _device = _actor_critic_state(policy, view)
        return int(policy.act(state, deterministic=True))

    return _act


def stochastic_policy_from_actor_critic(policy: Any) -> AttackerPolicy:
    """Adapt a PPO ``ActorCriticPolicy`` to a softmax action mixture."""

    def _act(view: Any) -> np.ndarray:
        import torch

        state, _device = _actor_critic_state(policy, view)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        logits, _ = policy(state)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
        return probs

    return _act


def policy_from_mixture(mix: dict[str, float]) -> AttackerPolicy:
    """Return a stochastic attacker that samples from a labeled mixture."""

    actions = tuple(ACTION_TO_DELTA)

    def _act(view: Any) -> np.ndarray:
        probs = np.zeros(len(actions), dtype=np.float64)
        for label, weight in mix.items():
            if weight <= 0.0:
                continue
            action = attacker_v2_heuristic_action(view, label)
            if action not in ACTION_TO_DELTA:
                action = 0
            probs[actions.index(action)] += float(weight)
        if float(probs.sum()) <= 0.0:
            return np.ones(len(actions), dtype=np.float64) / len(actions)
        return probs / probs.sum()

    return _act


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


def exact_metrics_for_policy(
    config: GridworldV2Config,
    attacker_policy: AttackerPolicy | None = None,
    table: np.ndarray | None = None,
) -> dict[str, float]:
    """Return exact Nash value, BR value, and exploitability for a policy or table."""

    if table is None:
        if attacker_policy is None:
            raise ValueError("exact_metrics_for_policy requires attacker_policy or table")
        table = action_probs_table(config, attacker_policy)
    value_vs_br = best_response_value_from_table(config, table)
    game_value = nash_value(config)
    return {
        "exact_value_vs_br": float(value_vs_br),
        "exact_nash_value": float(game_value),
        "exploitability": float(game_value - value_vs_br),
    }


def exact_metrics_for_strategy(
    config: GridworldV2Config,
    strategy: Any,
    label: str | None = None,
    policy: Any | None = None,
    attacker_policy: AttackerPolicy | None = None,
) -> dict[str, float]:
    """Return exact Nash value, BR value, and exploitability for one strategy."""

    resolved = attacker_policy or policy_from_strategy(strategy, label=label, policy=policy)
    return exact_metrics_for_policy(config, resolved)


def clear_exact_caches() -> None:
    """Drop cached Nash tables. Useful in tests that construct many configs."""

    nash_value.cache_clear()
    _nash_solution.cache_clear()
