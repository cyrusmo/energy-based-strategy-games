from strategy_games.experiments.gate2 import summarize_gate2


def _row(method: str, seed: int, exploitability: float, heldout_goal_rate: float = 0.0) -> dict[str, float | str | int]:
    return {
        "method": method,
        "seed": seed,
        "attacker_kind": method,
        "train_return": 0.0,
        "train_exploitability": exploitability,
        "heldout_exploitability": exploitability,
        "exploitability": exploitability,
        "exact_value_vs_br": -exploitability,
        "heldout_goal_rate": heldout_goal_rate,
        "strategy_entropy": 0.5,
        "mixture_support": 1.0,
        "env_steps": 100,
    }


def test_gate2_summary_marks_diagnostic_when_method_is_more_exploitable() -> None:
    rows = []
    for seed in (0, 1, 2, 3, 4):
        rows.append(_row("strategy_loop", seed, 0.80, heldout_goal_rate=0.2))
        rows.append(_row("ppo_baseline", seed, 0.20, heldout_goal_rate=0.6))
        rows.append(_row("diversity_only", seed, 0.25, heldout_goal_rate=0.1))
        rows.append(_row("psro_lite", seed, 0.30, heldout_goal_rate=0.3))
        rows.append(_row("nash_oracle", seed, 0.0, heldout_goal_rate=0.8))
        rows.append(_row("uniform_random", seed, 1.5, heldout_goal_rate=0.0))
    summary = summarize_gate2(rows)
    assert summary["passed"] is False
    assert summary["framing"] == "diagnostic"
    assert summary["beats_ppo"] is False
    assert summary["schema_version"] == "gate2/v3"


def test_gate2_summary_marks_positive_when_exploitability_cis_separate() -> None:
    rows = []
    for seed in (0, 1, 2, 3, 4):
        rows.append(_row("strategy_loop", seed, 0.05, heldout_goal_rate=0.4))
        rows.append(_row("ppo_baseline", seed, 0.60, heldout_goal_rate=0.1))
        rows.append(_row("diversity_only", seed, 0.55, heldout_goal_rate=0.1))
        rows.append(_row("nash_oracle", seed, 0.0, heldout_goal_rate=0.9))
        rows.append(_row("uniform_random", seed, 1.4, heldout_goal_rate=0.0))
    summary = summarize_gate2(rows)
    assert summary["passed"] is True
    assert summary["framing"] == "positive"
    assert summary["beats_ppo"] is True
    assert summary["beats_diversity_only"] is True


def test_gate2_pass_rule_ignores_reference_rows() -> None:
    rows = []
    for seed in (0, 1, 2, 3, 4):
        rows.append(_row("strategy_loop", seed, 0.80))
        rows.append(_row("ppo_baseline", seed, 0.20))
        rows.append(_row("diversity_only", seed, 0.25))
        rows.append(_row("nash_oracle", seed, 0.0))
    summary = summarize_gate2(rows)
    assert summary["passed"] is False
    nash = next(item for item in summary["methods"] if item["method"] == "nash_oracle")
    assert nash["reference"] is True
