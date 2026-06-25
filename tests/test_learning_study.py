import csv
import json
from pathlib import Path

import yaml

from strategy_games.experiments.learning_study import (
    LEARNING_CURVE_FIELDS,
    LEARNING_RUN_FIELDS,
    TRIAGE_FIELDS,
    LearningStudyConfig,
    configure_strategy_loop,
    failed_run_row,
    learning_study_config_from_mapping,
    run_learning_study,
    strategy_curve_rows,
    summarize_learning_runs,
    triage_row,
    write_learning_study_outputs,
)


def test_learning_study_config_from_mapping() -> None:
    config = learning_study_config_from_mapping(
        {
            "study": {
                "seeds": [3, 4],
                "strategy_config": "custom_strategy.yaml",
                "ppo_config": "custom_ppo.yaml",
                "output_dir": "outputs/public/custom_learning",
                "include_ppo": False,
                "ppo_total_steps": 128,
                "strategy_iterations": 2,
                "task_success_metric": "goal_rate",
                "high_variance_threshold": 0.2,
                "min_completed_seeds": 2,
            }
        }
    )

    assert config.seeds == (3, 4)
    assert config.strategy_config_path == Path("custom_strategy.yaml")
    assert config.ppo_config_path == Path("custom_ppo.yaml")
    assert config.include_ppo is False
    assert config.ppo_total_steps == 128
    assert config.strategy_iterations == 2
    assert config.task_success_metric == "goal_rate"
    assert config.high_variance_threshold == 0.2
    assert config.min_completed_seeds == 2


def test_learning_study_summary_schema_has_confidence_fields() -> None:
    summary = summarize_learning_runs(
        [
            {
                "method": "strategy_loop",
                "seed": 0,
                "status": "completed",
                "episode_return": -1.0,
                "win_rate": 0.0,
                "goal_rate": 0.0,
                "catch_rate": 1.0,
                "timeout_rate": 0.0,
                "episode_length": 6.0,
                "final_selected_label": "direct_goal",
                "strategy_diversity": 0.1,
            },
            {
                "method": "strategy_loop",
                "seed": 1,
                "status": "completed",
                "episode_return": 1.0,
                "win_rate": 1.0,
                "goal_rate": 1.0,
                "catch_rate": 0.0,
                "timeout_rate": 0.0,
                "episode_length": 4.0,
                "final_selected_label": "evasive",
                "strategy_diversity": 0.2,
            },
        ]
    )

    method = summary["methods"][0]
    assert summary["schema_version"] == "gridworld_learning_study/v1"
    assert method["method"] == "strategy_loop"
    assert method["episode_return_mean"] == 0.0
    assert method["completed_seed_count"] == 2
    assert method["episode_length_mean"] == 5.0
    assert method["selected_strategy_distribution"] == {"direct_goal": 0.5, "evasive": 0.5}
    assert "episode_return_ci_low" in method
    assert "goal_rate_ci_high" in method
    assert summary["triage_table"][0]["diagnostic_confidence"] == "low"


def test_learning_study_summary_records_failed_seeds_without_aggregating_them() -> None:
    failed = failed_run_row("strategy_loop", 1, RuntimeError("boom"))
    summary = summarize_learning_runs(
        [
            {
                "method": "strategy_loop",
                "seed": 0,
                "status": "completed",
                "episode_return": 2.0,
                "win_rate": 1.0,
                "goal_rate": 1.0,
                "catch_rate": 0.0,
                "timeout_rate": 0.0,
                "episode_length": 3.0,
                "final_selected_label": "direct_goal",
                "strategy_diversity": 0.1,
            },
            failed,
        ],
        min_completed_seeds=1,
    )

    method = summary["methods"][0]
    triage = summary["triage_table"][0]
    assert failed["status"] == "failed"
    assert method["attempted_seeds"] == [0, 1]
    assert method["failed_seeds"] == [1]
    assert method["completed_seed_count"] == 1
    assert method["episode_return_mean"] == 2.0
    assert triage["main_failure_mode"] == "incomplete_runs"
    assert triage["completed_seed_count"] == 1


def test_triage_row_marks_too_few_completed_seeds() -> None:
    row = triage_row(
        {
            "method": "ppo_baseline",
            "completed_seed_count": 0,
            "failed_seed_count": 2,
            "success_rate_mean": 0.0,
            "success_rate_std": 0.0,
            "episode_return_mean": 0.0,
        },
        task_success_metric="goal_rate",
        high_variance_threshold=0.25,
        min_completed_seeds=1,
    )

    assert row["main_failure_mode"] == "all_runs_failed"
    assert row["diagnostic_confidence"] == "high"


def test_strategy_curve_rows_extract_iteration_metrics() -> None:
    rows = strategy_curve_rows(
        0,
        {
            "history": [
                {
                    "iteration": 0,
                    "selected_label": "direct_goal",
                    "rollout": {"episode_return": 0.9, "win_rate": 1.0, "goal_rate": 1.0},
                    "selection_metrics": {
                        "average_case_value": 0.5,
                        "robustness_score": 0.1,
                        "exploitability_proxy": 0.2,
                    },
                    "candidate_diversity": 0.3,
                    "updates": {"policy_loss": 0.4, "policy_value_loss": 0.5, "policy_entropy": 1.0},
                }
            ]
        },
    )

    assert rows[0]["method"] == "strategy_loop"
    assert rows[0]["seed"] == 0
    assert rows[0]["selected_label"] == "direct_goal"
    assert rows[0]["episode_return"] == 0.9
    assert rows[0]["exploitability_proxy"] == 0.2


def test_write_learning_study_outputs(tmp_path: Path) -> None:
    runs = [
        {
            "method": "strategy_loop",
            "seed": 0,
            "episode_return": -1.0,
            "win_rate": 0.0,
            "goal_rate": 0.0,
            "catch_rate": 1.0,
            "timeout_rate": 0.0,
            "episode_length": 6.0,
            "iterations": 1,
            "updates": 0,
            "total_steps": 0,
            "final_selected_label": "direct_goal",
            "buffer_diversity": 0.0,
            "status": "completed",
            "error_type": "",
            "error_message": "",
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "strategy_diversity": 0.0,
            "robustness_score": 0.0,
            "exploitability_proxy": 0.0,
        }
    ]
    summary = summarize_learning_runs(runs)
    curves = [
        {
            "method": "strategy_loop",
            "seed": 0,
            "iteration": 0,
            "step": 0,
            "episode_return": -1.0,
        }
    ]
    paths = write_learning_study_outputs(tmp_path, runs, summary, curves)

    assert set(paths) == {"runs_json", "summary_json", "learning_curves_csv", "triage_table_csv"}
    assert json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))[0]["method"] == "strategy_loop"
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["schema_version"]
    with (tmp_path / "learning_curves.csv").open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert tuple(row) == LEARNING_CURVE_FIELDS
    with (tmp_path / "triage_table.csv").open("r", encoding="utf-8", newline="") as handle:
        triage = next(csv.DictReader(handle))
    assert tuple(triage) == TRIAGE_FIELDS
    assert triage["method"] == "strategy_loop"
    assert set(LEARNING_RUN_FIELDS).issubset(runs[0])


def test_learning_study_strategy_loop_smoke(tmp_path: Path) -> None:
    base_config = tmp_path / "strategy.yaml"
    base_config.write_text(
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
                "evaluator": {"episodes_per_opponent": 1},
                "updates": {"ebm_batch_size": 2, "train_world_model": False},
                "logging": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    result = run_learning_study(
        LearningStudyConfig(
            seeds=(0,),
            strategy_config_path=base_config,
            output_dir=tmp_path / "learning",
            include_ppo=False,
            strategy_iterations=1,
        )
    )

    assert result["summary"]["schema_version"] == "gridworld_learning_study/v1"
    assert (tmp_path / "learning" / "runs.json").exists()
    assert (tmp_path / "learning" / "summary.json").exists()
    assert (tmp_path / "learning" / "learning_curves.csv").exists()
    assert (tmp_path / "learning" / "triage_table.csv").exists()
    assert result["runs"][0]["method"] == "strategy_loop"
    assert result["triage_table"][0]["completed_seed_count"] == 1


def test_configure_strategy_loop_overrides_iterations_and_logging(tmp_path: Path) -> None:
    config = configure_strategy_loop(
        {"training": {"iterations": 10}, "logging": {"enabled": False}},
        seed=5,
        output_dir=tmp_path,
        strategy_iterations=2,
    )

    assert config["seed"] == 5
    assert config["training"]["iterations"] == 2
    assert config["logging"]["enabled"] is True
    assert config["logging"]["run_name"] == "strategy_seed5"
