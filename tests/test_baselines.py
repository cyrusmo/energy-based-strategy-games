import math

from strategy_games.experiments.baselines import BASELINE_FIELDS, compare_baselines, format_baseline_table


def test_baseline_comparison_schema() -> None:
    rows = compare_baselines(config_path="configs/gridworld_debug.yaml", episodes=1)
    assert [row["baseline"] for row in rows] == [
        "random_policy",
        "direct_goal_heuristic",
        "day2_strategy_loop",
    ]
    for row in rows:
        assert tuple(row.keys()) == BASELINE_FIELDS
        for field in BASELINE_FIELDS[1:]:
            assert math.isfinite(float(row[field]))

    table = format_baseline_table(rows)
    assert "baseline" in table
    assert "day2_strategy_loop" in table
