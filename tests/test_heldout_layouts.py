from pathlib import Path

import yaml

from strategy_games.envs.layouts import HELD_OUT_LAYOUT_NAMES, TRAIN_LAYOUT_NAME, get_layout, layout_names
from strategy_games.experiments.heldout_layouts import (
    HeldoutLayoutStudyConfig,
    run_heldout_layout_study,
    summarize_heldout_rows,
)
from strategy_games.training.ppo_baseline import run_direct_goal_baseline


def test_layout_registry_includes_default_and_held_out() -> None:
    names = layout_names()
    assert TRAIN_LAYOUT_NAME in names
    assert set(HELD_OUT_LAYOUT_NAMES).issubset(names)
    default = get_layout("default")
    intercept = get_layout("intercept_path")
    assert default.defender_start != intercept.defender_start


def test_direct_goal_wins_default_and_fails_intercept_path() -> None:
    default = run_direct_goal_baseline(episodes=3, env_config=get_layout("default"))
    intercept = run_direct_goal_baseline(episodes=3, env_config=get_layout("intercept_path"))
    assert default["goal_rate"] == 1.0
    assert intercept["goal_rate"] == 0.0


def test_heldout_summary_flags_direct_goal_overfitting() -> None:
    summary = summarize_heldout_rows(
        [
            {
                "layout": "default",
                "held_out": False,
                "baseline": "direct_goal_heuristic",
                "goal_rate": 1.0,
            },
            {
                "layout": "intercept_path",
                "held_out": True,
                "baseline": "direct_goal_heuristic",
                "goal_rate": 0.0,
            },
        ]
    )
    assert summary["schema_version"] == "gridworld_heldout_layouts/v1"
    assert summary["direct_goal_overfitting_visible"] is True
    assert summary["baselines"][0]["goal_rate_drop"] == 1.0


def test_heldout_layout_study_smoke(tmp_path: Path) -> None:
    strategy_config = tmp_path / "strategy.yaml"
    strategy_config.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "env": {
                    "grid_size": 5,
                    "max_steps": 6,
                    "attacker_start": [0, 0],
                    "defender_start": [4, 4],
                    "goal_pos": [4, 0],
                },
                "training": {"iterations": 1, "candidate_strategies": 2, "strategy_dim": 4},
                "policy": {"hidden_dim": 8},
                "ebm": {"hidden_dim": 8, "langevin_steps": 1},
                "world_model": {"hidden_dim": 8},
                "evaluator": {"episodes_per_opponent": 1, "action_source": "heuristic"},
                "executor": {"action_source": "aligned"},
                "updates": {"ebm_batch_size": 2, "train_world_model": False},
                "logging": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    result = run_heldout_layout_study(
        HeldoutLayoutStudyConfig(
            layouts=("default", "intercept_path"),
            episodes=1,
            seed=0,
            strategy_config_path=strategy_config,
            output_dir=tmp_path / "heldout",
            include_strategy_loop=True,
            include_ppo=False,
            strategy_iterations=1,
        )
    )

    assert (tmp_path / "heldout" / "summary.json").exists()
    assert (tmp_path / "heldout" / "rows.csv").exists()
    baselines = {row["baseline"] for row in result["rows"]}
    assert {"random_policy", "direct_goal_heuristic", "strategy_loop"} <= baselines
    assert result["summary"]["direct_goal_overfitting_visible"] is True
