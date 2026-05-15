import math

import yaml

from strategy_games.experiments.baselines import BASELINE_FIELDS, compare_baselines, format_baseline_table


def test_baseline_comparison_schema(tmp_path) -> None:
    ppo_config_path = tmp_path / "ppo.yaml"
    ppo_config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 5,
                "env": {
                    "grid_size": 5,
                    "max_steps": 8,
                    "attacker_start": [0, 0],
                    "defender_start": [4, 4],
                    "goal_pos": [4, 0],
                    "catch_radius": 0,
                },
                "ppo": {
                    "total_steps": 16,
                    "rollout_steps": 8,
                    "update_epochs": 1,
                    "minibatch_size": 4,
                    "hidden_dim": 8,
                    "eval_episodes": 1,
                },
                "logging": {"enabled": True, "output_dir": str(tmp_path / "ppo_outputs")},
            }
        ),
        encoding="utf-8",
    )

    rows = compare_baselines(
        config_path="configs/gridworld_debug.yaml",
        episodes=1,
        include_ppo=True,
        ppo_config_path=ppo_config_path,
    )
    assert [row["baseline"] for row in rows] == [
        "random_policy",
        "direct_goal_heuristic",
        "day2_strategy_loop",
        "ppo_baseline",
    ]
    for row in rows:
        assert tuple(row.keys()) == BASELINE_FIELDS
        for field in BASELINE_FIELDS[1:]:
            assert math.isfinite(float(row[field]))

    table = format_baseline_table(rows)
    assert "baseline" in table
    assert "day2_strategy_loop" in table
    assert "ppo_baseline" in table
