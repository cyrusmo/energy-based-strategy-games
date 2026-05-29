import json

from strategy_games.viewers.dashboard_data import convergence_rows, load_dashboard_data, quality_rows, resource_rows


def test_resource_rows_explain_recommendations() -> None:
    rows = resource_rows(
        [{"job": "langevin_sampling", "cpu_ms": 2.0, "mps_ms": 1.0, "speedup": 2.0, "recommended_device": "mps"}]
    )
    assert rows[0]["recommended_device"] == "mps"
    assert "recommended" in rows[0]["explanation"]


def test_convergence_rows_flattens_history() -> None:
    history = [
        {
            "iteration": 0,
            "rollout": {"episode_return": 1.0, "goal_rate": 1.0, "win_rate": 1.0},
            "updates": {"positive_energy": 0.5, "negative_energy": 1.0, "policy_loss": 0.1},
        }
    ]
    payload = convergence_rows(history)
    assert payload["curves"][0]["ebm_energy_gap"] == 0.5
    assert payload["badges"]["goal_rate"]["converged"] is True


def test_quality_rows_merges_multiseed_summary() -> None:
    rows = quality_rows(
        {"baselines": [{"baseline": "strategy_loop", "episode_return": -1.0, "win_rate": 0.0}]},
        {
            "baselines": [
                {
                    "baseline": "strategy_loop",
                    "episode_return_mean": 0.1,
                    "episode_return_ci_low": 0.0,
                    "episode_return_ci_high": 0.2,
                    "win_rate_mean": 0.5,
                    "win_rate_ci_low": 0.25,
                    "win_rate_ci_high": 0.75,
                    "runs": 2,
                }
            ]
        },
    )
    assert rows[0]["episode_return"] == 0.1
    assert rows[0]["win_rate_ci"] == [0.25, 0.75]


def test_load_dashboard_data_reads_artifacts(tmp_path) -> None:
    (tmp_path / "baselines").mkdir()
    (tmp_path / "multiseed" / "strategy_runs" / "strategy_seed0").mkdir(parents=True)
    (tmp_path / "multiseed").mkdir(exist_ok=True)
    (tmp_path / "device_calibration.json").write_text(
        json.dumps({"jobs": [{"job": "ppo_update", "cpu_ms": 1.0, "recommended_device": "cpu"}]}),
        encoding="utf-8",
    )
    (tmp_path / "baselines" / "metrics.json").write_text(
        json.dumps({"baselines": [{"baseline": "random_policy", "episode_return": -1.0}]}),
        encoding="utf-8",
    )
    (tmp_path / "multiseed" / "summary.json").write_text(json.dumps({"baselines": []}), encoding="utf-8")
    (tmp_path / "multiseed" / "strategy_runs" / "strategy_seed0" / "iterations.jsonl").write_text(
        json.dumps({"iteration": 0, "rollout": {"goal_rate": 0.0}}) + "\n",
        encoding="utf-8",
    )

    data = load_dashboard_data(tmp_path)
    assert not data["missing"]
    assert data["resource"][0]["job"] == "ppo_update"
    assert data["convergence"]["curves"][0]["iteration"] == 0
