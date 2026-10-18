from strategy_games.experiments.gate2 import summarize_gate2


def _row(method: str, seed: int, exploitability: float, heldout_goal_rate: float = 0.0) -> dict[str, float | str | int]:
    return {
        "method": method,
        "seed": seed,
        "train_return": 0.0,
        "train_exploitability": exploitability,
        "heldout_exploitability": exploitability,
        "exploitability": exploitability,
        "exact_value_vs_br": -exploitability,
        "heldout_goal_rate": heldout_goal_rate,
        "strategy_entropy": 0.5,
        "env_steps": 100,
    }


def test_gate2_summary_marks_diagnostic_when_method_is_more_exploitable() -> None:
    rows = []
    for seed in (0, 1, 2, 3, 4):
        rows.append(_row("strategy_loop", seed, 0.80, heldout_goal_rate=0.2))
        rows.append(_row("ppo_baseline", seed, 0.20, heldout_goal_rate=0.6))
        rows.append(_row("diversity_only", seed, 0.25, heldout_goal_rate=0.1))
        rows.append(_row("psro_lite", seed, 0.30, heldout_goal_rate=0.3))
    summary = summarize_gate2(rows)
    assert summary["passed"] is False
    assert summary["framing"] == "diagnostic"
    assert summary["beats_ppo"] is False
    assert summary["schema_version"] == "gate2/v2"


def test_gate2_summary_marks_positive_when_exploitability_cis_separate() -> None:
    rows = []
    for seed in (0, 1, 2, 3, 4):
        rows.append(_row("strategy_loop", seed, 0.05, heldout_goal_rate=0.4))
        rows.append(_row("ppo_baseline", seed, 0.60, heldout_goal_rate=0.1))
        rows.append(_row("diversity_only", seed, 0.55, heldout_goal_rate=0.1))
    summary = summarize_gate2(rows)
    assert summary["passed"] is True
    assert summary["framing"] == "positive"
    assert summary["beats_ppo"] is True
    assert summary["beats_diversity_only"] is True
