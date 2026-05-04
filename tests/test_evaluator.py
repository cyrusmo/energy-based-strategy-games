from strategy_games.evaluation.best_response import GameTheoreticEvaluator
from strategy_games.strategies.embeddings import named_strategy_embedding


def test_evaluator_returns_required_metrics() -> None:
    evaluator = GameTheoreticEvaluator(strategy_dim=8, episodes_per_opponent=1)
    strategy = named_strategy_embedding("direct_goal", 8).vector
    metrics = evaluator.evaluate_strategy(strategy, label="direct_goal")
    required = {
        "average_case_value",
        "worst_case_value",
        "robustness_score",
        "exploitability_proxy",
        "goal_rate",
        "catch_rate",
        "timeout_rate",
        "win_rate",
        "best_response_label",
    }
    assert required.issubset(metrics)
    assert isinstance(metrics["best_response_label"], str)
