from strategy_games.envs.layouts_v2 import get_layout_v2
from strategy_games.evaluation.best_response import attacker_v2_heuristic_action
from strategy_games.envs.gridworld_v2 import AttackerDefenderGridworldV2
from strategy_games.experiments.gate1 import evaluate_gate1


def test_v2_attacker_heuristics_are_behaviorally_distinct() -> None:
    traces = {}
    for label in ("direct_goal", "evasive", "feint", "patient", "aggressive"):
        env = AttackerDefenderGridworldV2(get_layout_v2("corridor_split"))
        env.reset()
        path = [env.attacker_pos]
        for _ in range(6):
            action = attacker_v2_heuristic_action(env, label)
            env.step(action, defender_action=0)
            path.append(env.attacker_pos)
        traces[label] = tuple(path)
    assert traces["patient"][1] == traces["patient"][0]
    assert len({traces[label] for label in traces}) >= 3


def test_gate1_payoff_matrix_is_non_degenerate() -> None:
    report = evaluate_gate1()
    train = report["train"]["summary"]
    assert train["distinct_best_response_count"] >= 3, train
    assert train["weakly_dominating_rows"] == [], train
    for label, item in report["unique_best"].items():
        assert item["summary"]["unique_best_strategy"] == label, (label, item["summary"])
    assert report["nash_winnable"], report
    assert report["nash_has_negative_heldout"], report
    assert report["train_nash_value"] > report["train_timeout_value"]
    assert report["passed"]
