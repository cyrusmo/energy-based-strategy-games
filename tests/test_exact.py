import numpy as np

from strategy_games.envs.gridworld_v2 import GridworldV2Config
from strategy_games.evaluation.exact import (
    best_response_value,
    exact_exploitability,
    exact_metrics_for_strategy,
    nash_value,
    solve_matrix_nash,
)
from strategy_games.strategies.embeddings import named_strategy_embedding


def _tiny_config() -> GridworldV2Config:
    return GridworldV2Config(
        grid_size=3,
        max_steps=4,
        attacker_start=(0, 0),
        defender_start=(2, 2),
        goal_positions=((0, 2), (2, 0)),
        obstacles=frozenset(),
        catch_radius=0,
        slip_prob=0.0,
    )


def test_solve_matrix_nash_matching_pennies() -> None:
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])
    value, row, col = solve_matrix_nash(payoff)
    assert abs(value) < 1e-6
    assert abs(row[0] - 0.5) < 1e-5
    assert abs(row.sum() - 1.0) < 1e-8
    assert col.sum() == 1.0


def test_solve_matrix_nash_pure() -> None:
    payoff = np.array([[2.0, 1.0], [0.0, -1.0]])
    value, row, col = solve_matrix_nash(payoff)
    assert value == 1.0
    assert row[0] == 1.0
    assert col[1] == 1.0


def test_tiny_nash_and_br_are_finite() -> None:
    config = _tiny_config()
    value = nash_value(config)
    assert np.isfinite(value)
    strategy = named_strategy_embedding("direct_goal", 8).vector
    metrics = exact_metrics_for_strategy(config, strategy, label="direct_goal")
    assert "exploitability" in metrics
    assert metrics["exact_nash_value"] == value
    assert metrics["exploitability"] == value - metrics["exact_value_vs_br"]


def test_heuristic_exploitability_is_non_negative() -> None:
    config = _tiny_config()
    def _policy(view):
        from strategy_games.evaluation.best_response import attacker_v2_heuristic_action

        return attacker_v2_heuristic_action(view, "patient")

    exploit = exact_exploitability(config, _policy)
    assert exploit >= -1e-6
    assert np.isfinite(best_response_value(config, _policy))
