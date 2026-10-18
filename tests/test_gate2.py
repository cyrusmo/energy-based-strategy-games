from strategy_games.experiments.gate2 import summarize_gate2


def test_gate2_summary_marks_diagnostic_when_method_does_not_transfer() -> None:
    rows = []
    for seed in (0, 1, 2, 3, 4):
        rows.append({"method": "strategy_loop", "seed": seed, "train_return": 0.5, "heldout_goal_rate": 0.2})
        rows.append({"method": "ppo_baseline", "seed": seed, "train_return": 0.4, "heldout_goal_rate": 0.6})
        rows.append({"method": "diversity_only", "seed": seed, "train_return": 0.1, "heldout_goal_rate": 0.1})
        rows.append({"method": "psro_lite", "seed": seed, "train_return": 0.3, "heldout_goal_rate": 0.3})
    summary = summarize_gate2(rows)
    assert summary["passed"] is False
    assert summary["framing"] == "diagnostic"
    assert summary["beats_ppo"] is False


def test_gate2_summary_marks_positive_when_cis_separate() -> None:
    rows = []
    for seed in (0, 1, 2, 3, 4):
        rows.append({"method": "strategy_loop", "seed": seed, "train_return": 0.9, "heldout_goal_rate": 0.9})
        rows.append({"method": "ppo_baseline", "seed": seed, "train_return": 0.2, "heldout_goal_rate": 0.1})
        rows.append({"method": "diversity_only", "seed": seed, "train_return": 0.2, "heldout_goal_rate": 0.1})
    summary = summarize_gate2(rows)
    assert summary["passed"] is True
    assert summary["framing"] == "positive"
