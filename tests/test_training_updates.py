import math

import torch

from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig
from strategy_games.evaluation.best_response import GameTheoreticEvaluator
from strategy_games.models.policy import StrategyConditionedPolicy
from strategy_games.strategies.buffer import StrategyRecord
from strategy_games.strategies.embeddings import named_strategy_embedding
from strategy_games.strategies.sampler import langevin_sample
from strategy_games.experiments.runner import run_from_config
from strategy_games.training.train_loop import (
    TrainingConfig,
    collect_policy_rollout,
    gridworld_potential,
    heuristic_action_from_state,
    selection_score_from_metrics,
    run_training_loop,
)


def test_training_loop_reports_update_metrics() -> None:
    config = TrainingConfig(
        seed=321,
        iterations=1,
        candidate_strategies=3,
        strategy_dim=5,
        policy_hidden_dim=8,
        ebm_hidden_dim=8,
        world_model_hidden_dim=8,
        langevin_steps=1,
        ebm_batch_size=2,
        episodes_per_opponent=1,
        env=GridworldConfig(grid_size=5, max_steps=8, defender_start=(4, 4), goal_pos=(4, 0)),
    )
    result = run_training_loop(config)
    updates = result["history"][0]["updates"]
    expected = {
        "policy_loss",
        "policy_grad_norm",
        "policy_entropy",
        "policy_value_loss",
        "world_model_loss",
        "world_model_grad_norm",
        "ebm_loss",
        "ebm_grad_norm",
        "positive_energy",
        "negative_energy",
    }
    assert expected.issubset(updates)
    for key in expected:
        assert math.isfinite(updates[key])


def test_runner_loads_day2_config() -> None:
    result = run_from_config("configs/gridworld_day2.yaml")
    assert result["buffer_size"] == 4
    assert len(result["history"]) == 4
    assert "updates" in result["history"][0]


def test_strategy_policy_has_value_head() -> None:
    policy = StrategyConditionedPolicy(state_dim=9, strategy_dim=5, action_dim=5, hidden_dim=8)
    states = torch.zeros(3, 9)
    strategies = torch.zeros(3, 5)
    logits, values = policy.evaluate(states, strategies)
    assert logits.shape == (3, 5)
    assert values.shape == (3,)
    assert torch.isfinite(values).all()


def test_multiple_episodes_per_update_are_aggregated() -> None:
    config = TrainingConfig(
        seed=11,
        iterations=1,
        episodes_per_update=3,
        candidate_strategies=2,
        strategy_dim=4,
        policy_hidden_dim=8,
        ebm_hidden_dim=8,
        world_model_hidden_dim=8,
        langevin_steps=1,
        ebm_batch_size=2,
        env=GridworldConfig(grid_size=5, max_steps=5, defender_start=(4, 4), goal_pos=(4, 0)),
    )
    result = run_training_loop(config)
    rollout = result["history"][0]["rollout"]
    assert rollout["episodes"] == 3
    assert "policy_value_loss" in result["history"][0]["updates"]


def test_potential_shaping_rewards_goal_progress() -> None:
    env_config = GridworldConfig(grid_size=5, max_steps=8, defender_start=(4, 4), goal_pos=(4, 0))
    env = AttackerDefenderGridworld(env_config)
    before = gridworld_potential(env)
    env.step(2)
    after = gridworld_potential(env)
    assert after > before


def test_evaluator_can_score_learned_policy_path() -> None:
    env_config = GridworldConfig(grid_size=5, max_steps=5, defender_start=(4, 4), goal_pos=(4, 0))
    policy = StrategyConditionedPolicy(state_dim=9, strategy_dim=4, action_dim=5, hidden_dim=8)
    strategy = named_strategy_embedding("direct_goal", 4).vector
    evaluator = GameTheoreticEvaluator(env_config=env_config, opponent_labels=("direct_goal",), strategy_dim=4, seed=3)
    metrics = evaluator.evaluate_strategy(strategy, label="direct_goal", policy=policy)
    assert "average_case_value" in metrics
    assert "goal_rate" in metrics


def test_selection_score_prefers_goal_reaching_candidate() -> None:
    config = TrainingConfig(goal_rate_weight=2.0, robustness_weight=0.0, exploitability_weight=0.0)
    losing = {
        "average_case_value": -0.3,
        "robustness_score": -0.3,
        "exploitability_proxy": 0.0,
        "goal_rate": 0.0,
    }
    sometimes_wins = {
        "average_case_value": -0.4,
        "robustness_score": -0.7,
        "exploitability_proxy": 0.2,
        "goal_rate": 0.5,
    }
    assert selection_score_from_metrics(sometimes_wins, config) > selection_score_from_metrics(losing, config)


def test_buffer_score_prefers_goal_reaching_record() -> None:
    losing = StrategyRecord(
        embedding=torch.zeros(4),
        episode_return=-0.3,
        robustness_score=-0.3,
        exploitability_proxy=0.0,
        average_case_value=-0.3,
        goal_rate=0.0,
        iteration=0,
    )
    winner = StrategyRecord(
        embedding=torch.ones(4),
        episode_return=-0.4,
        robustness_score=-0.7,
        exploitability_proxy=0.2,
        average_case_value=-0.4,
        goal_rate=1.0,
        iteration=0,
    )
    assert winner.score() > losing.score()


def test_langevin_samples_lower_energy_than_random_for_quadratic_energy() -> None:
    class QuadraticEnergy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.square().sum(dim=-1)

    init = torch.ones(8, 4) * 3.0
    samples = langevin_sample(
        QuadraticEnergy(),
        num_samples=8,
        strategy_dim=4,
        init=init,
        steps=20,
        step_size=0.1,
        noise_scale=0.0,
    )
    assert samples.square().sum(dim=-1).mean() < init.square().sum(dim=-1).mean()


def test_behavior_clone_target_from_state_matches_direct_goal() -> None:
    env_config = GridworldConfig(grid_size=5, max_steps=8, defender_start=(4, 4), goal_pos=(4, 0))
    policy = StrategyConditionedPolicy(state_dim=9, strategy_dim=4, action_dim=5, hidden_dim=8)
    strategy = named_strategy_embedding("direct_goal", 4).vector
    trajectory = collect_policy_rollout(
        AttackerDefenderGridworld(env_config),
        policy,
        strategy,
        config=TrainingConfig(env=env_config),
    )
    assert heuristic_action_from_state(trajectory.states[0].cpu(), strategy, "direct_goal", env_config) == 2
