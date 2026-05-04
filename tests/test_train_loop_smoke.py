from strategy_games.envs.gridworld import GridworldConfig
from strategy_games.training.train_loop import TrainingConfig, run_training_loop


def test_train_loop_smoke() -> None:
    config = TrainingConfig(
        seed=123,
        iterations=1,
        candidate_strategies=2,
        strategy_dim=4,
        ebm_hidden_dim=8,
        langevin_steps=1,
        episodes_per_opponent=1,
        env=GridworldConfig(grid_size=5, max_steps=8, defender_start=(4, 4), goal_pos=(4, 0)),
    )
    result = run_training_loop(config)
    assert result["buffer_size"] == 1
    assert len(result["history"]) == 1
    assert result["final_selected_label"] is not None
