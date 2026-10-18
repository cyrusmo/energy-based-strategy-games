from strategy_games.envs.layouts_v2 import get_layout_v2
from strategy_games.training.train_loop import TrainingConfig, run_training_loop


def test_v2_strategy_loop_one_iteration() -> None:
    result = run_training_loop(
        TrainingConfig(
            seed=0,
            iterations=1,
            candidate_strategies=6,
            max_heuristic_candidates=1,
            env=get_layout_v2("corridor_split"),
            evaluator_action_source="heuristic",
            executor_action_source="parametric",
            train_policy=False,
            train_world_model=False,
        )
    )
    assert result["history"]
    assert "strategy_entropy" in result
    assert result["env_steps"] > 0
    assert result["history"][0]["rollout"]["steps"] >= 1
