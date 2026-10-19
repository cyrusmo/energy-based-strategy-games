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


def test_stochastic_attacker_best_response_is_finite() -> None:
    config = _tiny_config()

    def _mix(_view):
        return np.ones(5, dtype=np.float64) / 5.0

    value = best_response_value(config, _mix)
    assert np.isfinite(value)


def test_policy_from_actor_critic_returns_valid_action() -> None:
    from strategy_games.evaluation.exact import _StateView, policy_from_actor_critic
    from strategy_games.training.ppo_baseline import ActorCriticPolicy

    config = _tiny_config()
    policy = ActorCriticPolicy(state_dim=11, action_dim=5, hidden_dim=8)
    attacker = policy_from_actor_critic(policy)
    action = attacker(_StateView(config, config.attacker_start, config.defender_start, 0))
    assert action in {0, 1, 2, 3, 4}


def test_nash_oracle_exploitability_is_near_zero() -> None:
    from strategy_games.evaluation.exact import exact_metrics_for_policy, nash_attacker_table

    config = _tiny_config()
    metrics = exact_metrics_for_policy(config, table=nash_attacker_table(config))
    assert abs(metrics["exploitability"]) < 1e-6


def test_uniform_random_exploitability_is_finite() -> None:
    from strategy_games.evaluation.exact import exact_metrics_for_policy, uniform_random_table

    config = _tiny_config()
    metrics = exact_metrics_for_policy(config, table=uniform_random_table(config))
    assert np.isfinite(metrics["exploitability"])
    assert metrics["exploitability"] >= -1e-6


def test_mixture_table_rows_sum_to_one() -> None:
    from strategy_games.evaluation.exact import (
        action_probs_table,
        exact_metrics_for_policy,
        mixture_table,
        policy_from_strategy,
        uniform_random_table,
    )

    config = _tiny_config()
    strategy = named_strategy_embedding("direct_goal", 8).vector
    first = action_probs_table(config, policy_from_strategy(strategy, label="direct_goal"))
    second = uniform_random_table(config)
    mixed = mixture_table([first, second], [0.7, 0.3])
    sums = mixed.sum(axis=-1)
    assert np.allclose(sums, 1.0)
    metrics = exact_metrics_for_policy(config, table=mixed)
    assert np.isfinite(metrics["exploitability"])


def test_stochastic_policy_from_actor_critic_is_a_distribution() -> None:
    from strategy_games.evaluation.exact import _StateView, stochastic_policy_from_actor_critic
    from strategy_games.training.ppo_baseline import ActorCriticPolicy

    config = _tiny_config()
    policy = ActorCriticPolicy(state_dim=11, action_dim=5, hidden_dim=8)
    attacker = stochastic_policy_from_actor_critic(policy)
    probs = attacker(_StateView(config, config.attacker_start, config.defender_start, 0))
    assert probs.shape == (5,)
    assert abs(float(np.sum(probs)) - 1.0) < 1e-6
    assert np.all(probs >= 0.0)
