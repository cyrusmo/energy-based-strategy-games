import math

from strategy_games.envs.gridworld import GridworldConfig
from strategy_games.experiments.runner import run_from_config
from strategy_games.training.train_loop import TrainingConfig, run_training_loop


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
