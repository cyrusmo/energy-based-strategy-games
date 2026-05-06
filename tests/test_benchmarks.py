import json
import math

import pytest
import yaml

from strategy_games.benchmarks.registry import available_benchmarks, make_benchmark_adapter
from strategy_games.benchmarks.runner import run_benchmark_from_config, run_benchmark_suite

REQUIRED_RESULT_KEYS = {
    "env_id",
    "baseline",
    "seed",
    "episode_return",
    "win_rate",
    "goal_rate",
    "catch_rate",
    "timeout_rate",
    "survival_or_capture_rate",
    "steps",
    "strategy_label",
    "wall_clock_seconds",
    "average_case_value",
    "worst_case_value",
    "exploitability_proxy",
    "strategy_diversity",
}


def test_benchmark_registry_resolves_custom_gridworld() -> None:
    assert "custom_gridworld_v0" in available_benchmarks()
    adapter = make_benchmark_adapter(
        "custom_gridworld_v0",
        {
            "env": {
                "grid_size": 5,
                "max_steps": 8,
                "defender_start": [4, 4],
                "goal_pos": [4, 0],
            }
        },
    )
    assert adapter.env_id == "custom_gridworld_v0"


def test_custom_gridworld_adapter_result_schema() -> None:
    adapter = make_benchmark_adapter(
        "custom_gridworld_v0",
        {
            "env": {
                "grid_size": 5,
                "max_steps": 8,
                "defender_start": [4, 4],
                "goal_pos": [4, 0],
            }
        },
    )
    row = adapter.rollout("random_policy", seed=0).to_dict()
    assert REQUIRED_RESULT_KEYS.issubset(row)
    for key in ("episode_return", "win_rate", "goal_rate", "catch_rate", "survival_or_capture_rate"):
        assert math.isfinite(float(row[key]))
    assert row["env_id"] == "custom_gridworld_v0"
    assert row["baseline"] == "random_policy"


def test_benchmark_runner_tiny_config_writes_artifacts(tmp_path) -> None:
    config_path = tmp_path / "benchmark.yaml"
    output_dir = tmp_path / "outputs"
    config = {
        "suite": {
            "name": "tiny_suite",
            "output_dir": str(output_dir),
            "run_name": "tiny_run",
            "write_artifacts": True,
        },
        "benchmarks": [
            {
                "env_id": "custom_gridworld_v0",
                "seeds": [0, 1],
                "baselines": ["random_policy", "direct_goal_heuristic", "strategy_loop"],
                "env": {
                    "grid_size": 5,
                    "max_steps": 8,
                    "attacker_start": [0, 0],
                    "defender_start": [4, 4],
                    "goal_pos": [4, 0],
                    "catch_radius": 0,
                },
                "training": {
                    "iterations": 1,
                    "candidate_strategies": 2,
                    "strategy_dim": 4,
                },
                "ebm": {
                    "hidden_dim": 8,
                    "langevin_steps": 1,
                    "langevin_step_size": 0.02,
                },
                "evaluator": {"episodes_per_opponent": 1},
                "updates": {"ebm_batch_size": 2},
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_benchmark_from_config(config_path)
    assert len(result["results"]) == 6
    assert not result["skipped"]
    artifacts = result["artifacts"]
    results_path = output_dir / "tiny_run" / "results.jsonl"
    summary_path = output_dir / "tiny_run" / "summary.json"
    assert artifacts["results_jsonl"] == str(results_path)
    assert results_path.exists()
    assert summary_path.exists()
    assert len(results_path.read_text(encoding="utf-8").strip().splitlines()) == 6
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "by_env_and_baseline" in summary


def test_benchmark_suite_summary_contains_expected_metrics() -> None:
    result = run_benchmark_suite(
        {
            "suite": {"name": "summary_test", "write_artifacts": False},
            "benchmarks": [
                {
                    "env_id": "custom_gridworld_v0",
                    "seeds": [0],
                    "baselines": ["direct_goal_heuristic"],
                    "env": {
                        "grid_size": 5,
                        "max_steps": 8,
                        "defender_start": [4, 4],
                        "goal_pos": [4, 0],
                    },
                }
            ],
        }
    )
    summary_rows = result["summary"]["by_env_and_baseline"]
    assert len(summary_rows) == 1
    row = summary_rows[0]
    assert row["env_id"] == "custom_gridworld_v0"
    assert row["baseline"] == "direct_goal_heuristic"
    assert "mean_episode_return" in row
    assert "std_episode_return" in row


def test_pettingzoo_adapter_optional_dependency() -> None:
    pytest.importorskip("pettingzoo.sisl")
    adapter = make_benchmark_adapter(
        "pettingzoo_pursuit_v4",
        {"env": {"max_cycles": 2, "x_size": 6, "y_size": 6, "n_evaders": 2, "n_pursuers": 2}},
    )
    row = adapter.rollout("random_policy", seed=0).to_dict()
    assert REQUIRED_RESULT_KEYS.issubset(row)
    assert row["env_id"] == "pettingzoo_pursuit_v4"
