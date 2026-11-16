import numpy as np
import pytest
import torch

from strategy_games.envs.gridworld_v2 import GridworldV2Config
from strategy_games.evaluation import best_response
from strategy_games.evaluation.exact import action_probs_table
from strategy_games.experiments import gate2
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



def test_gate2_mixture_rolls_out_the_exact_attacker_table(monkeypatch: pytest.MonkeyPatch) -> None:
    train = GridworldV2Config(
        grid_size=3, max_steps=2, attacker_start=(0, 0), defender_start=(2, 2),
        goal_positions=((1, 1),),
    )
    heldout = [("test", train)]
    down = action_probs_table(train, lambda state: 2)
    right = action_probs_table(train, lambda state: 4)
    tables = {2: down, 4: right}
    components = [{"embedding": torch.tensor([float(action)])} for action in tables]
    weights = np.array([0.5, 0.5])
    mixed = gate2.mixture_table([down, right], weights)

    monkeypatch.setattr(gate2, "available_heuristic_strategies", lambda: ("stay",))
    monkeypatch.setattr(best_response, "available_heuristic_strategies", lambda: ("stay",))
    monkeypatch.setattr(gate2, "defender_heuristic_action", lambda env, opponent: 0)
    monkeypatch.setattr(best_response, "defender_heuristic_action", lambda env, opponent: 0)
    monkeypatch.setattr(best_response, "attacker_heuristic_action", lambda env, embedding, label: int(embedding.item()))
    monkeypatch.setattr(gate2, "_component_table", lambda env, component, cache: tables[int(component["embedding"].item())])

    exact_tables = []
    rollout_tables = []
    table_goal_rate = gate2._table_goal_rate

    def exact_metrics(env, *, table):
        exact_tables.append(table)
        return {"exploitability": 0.0, "exact_value_vs_br": 0.0, "exact_nash_value": 0.0}

    def goal_rate(env, table, *, seed):
        rollout_tables.append(table)
        return table_goal_rate(env, table, seed=seed)

    monkeypatch.setattr(gate2, "exact_metrics_for_policy", exact_metrics)
    monkeypatch.setattr(gate2, "_table_goal_rate", goal_rate)

    def evaluate(seed):
        row = gate2._row_from_table(
            "strategy_loop", seed, {}, train, heldout, mixed, "ebm_mixture",
            components=components, weights=weights,
        )
        assert exact_tables[-1] is rollout_tables[-1]
        np.testing.assert_array_equal(rollout_tables[-1], mixed)
        return row

    # Either fixed component misses the diagonal goal. Seed 0 draws right then down.
    first = evaluate(0)
    assert first["heldout_table_goal_rate"] == 1.0
    assert first["heldout_draw_once_goal_rate"] == 0.0
    assert "heldout_goal_rate" not in first
    assert evaluate(1)["heldout_table_goal_rate"] == 0.0
    assert evaluate(0) == first

    summary = summarize_gate2([first])
    method = summary["methods"][0]
    assert summary["schema_version"] == "gate2/v4"
    assert method["heldout_table_goal_rate"]["mean"] == 1.0
    assert method["heldout_draw_once_goal_rate"]["mean"] == 0.0


@pytest.mark.parametrize("same_row", [False, True])
def test_gate2_summary_rejects_mixed_goal_rate_definitions(same_row: bool) -> None:
    legacy = _row("strategy_loop", 0, 0.5, heldout_goal_rate=0.25)
    corrected = _row("strategy_loop", 1, 0.5)
    corrected["heldout_table_goal_rate"] = 0.5
    if not same_row:
        del corrected["heldout_goal_rate"]
    with pytest.raises(ValueError, match="single goal-rate protocol"):
        summarize_gate2([legacy, corrected])
