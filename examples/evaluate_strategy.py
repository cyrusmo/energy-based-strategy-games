"""Evaluate named strategies with approximate sampled-response metrics."""

from __future__ import annotations

import json

from strategy_games.evaluation.best_response import GameTheoreticEvaluator
from strategy_games.strategies.embeddings import available_heuristic_strategies, named_strategy_embedding


def main() -> None:
    strategy_dim = 8
    evaluator = GameTheoreticEvaluator(strategy_dim=strategy_dim, episodes_per_opponent=1)
    results = {}
    for label in available_heuristic_strategies():
        strategy = named_strategy_embedding(label, strategy_dim).vector
        results[label] = evaluator.evaluate_strategy(strategy, label=label)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
