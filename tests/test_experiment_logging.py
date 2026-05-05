import json

from strategy_games.envs.gridworld import GridworldConfig
from strategy_games.experiments.logging import ExperimentLogger, summarize_training_result
from strategy_games.training.train_loop import TrainingConfig, run_training_loop


def test_experiment_logger_outputs_expected_files(tmp_path) -> None:
    config = TrainingConfig(
        seed=5,
        iterations=2,
        candidate_strategies=2,
        strategy_dim=4,
        policy_hidden_dim=8,
        ebm_hidden_dim=8,
        world_model_hidden_dim=8,
        langevin_steps=1,
        ebm_batch_size=2,
        env=GridworldConfig(grid_size=5, max_steps=8, defender_start=(4, 4), goal_pos=(4, 0)),
    )
    result = run_training_loop(config)
    logger = ExperimentLogger(tmp_path, run_name="logger_test")
    artifacts = logger.log_run({"seed": 5, "logging": {"enabled": True}}, result)

    assert logger.iterations_path.exists()
    assert logger.metrics_path.exists()
    assert logger.config_path.exists()
    assert artifacts["metrics_json"] == str(logger.metrics_path)

    lines = logger.iterations_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["iteration"] in {0, 1} for line in lines)

    metrics = json.loads(logger.metrics_path.read_text(encoding="utf-8"))
    required = {
        "iterations",
        "final_selected_label",
        "buffer_size",
        "buffer_diversity",
        "mean_episode_return",
        "mean_win_rate",
        "mean_goal_rate",
        "mean_catch_rate",
        "final_update_losses",
    }
    assert required.issubset(metrics)
    assert metrics == summarize_training_result(result)
