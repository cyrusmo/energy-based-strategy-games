"""Payoff matrix utilities for named heuristic strategies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from strategy_games.evaluation.best_response import GameTheoreticEvaluator
from strategy_games.strategies.embeddings import available_heuristic_strategies, named_strategy_embedding


def compute_payoff_matrix(
    strategy_labels: list[str] | tuple[str, ...] | None = None,
    opponent_labels: list[str] | tuple[str, ...] | None = None,
    strategy_dim: int = 8,
    episodes_per_opponent: int = 1,
) -> dict[str, object]:
    """Compute average attacker reward for strategies against opponent heuristics."""

    strategies = list(strategy_labels or available_heuristic_strategies())
    opponents = list(opponent_labels or available_heuristic_strategies())
    matrix: list[list[float]] = []
    best_response_labels: dict[str, str] = {}

    for strategy_label in strategies:
        strategy = named_strategy_embedding(strategy_label, strategy_dim).vector
        row: list[float] = []
        for opponent_label in opponents:
            evaluator = GameTheoreticEvaluator(
                opponent_labels=(opponent_label,),
                episodes_per_opponent=episodes_per_opponent,
                strategy_dim=strategy_dim,
            )
            metrics = evaluator.evaluate_strategy(strategy, label=strategy_label)
            row.append(float(metrics["average_case_value"]))
        matrix.append(row)
        worst_idx = int(np.argmin(row))
        best_response_labels[strategy_label] = opponents[worst_idx]

    return {
        "strategy_labels": strategies,
        "opponent_labels": opponents,
        "average_reward_matrix": matrix,
        "best_response_labels": best_response_labels,
    }


def format_payoff_matrix(matrix_result: dict[str, object]) -> str:
    """Format a payoff matrix as aligned text."""

    strategies = list(matrix_result["strategy_labels"])
    opponents = list(matrix_result["opponent_labels"])
    matrix = matrix_result["average_reward_matrix"]
    label_width = max(len("strategy"), *(len(str(label)) for label in strategies))
    col_widths = [max(len(str(label)), 8) for label in opponents]

    header = "strategy".ljust(label_width) + "  " + "  ".join(
        str(label).rjust(width) for label, width in zip(opponents, col_widths)
    )
    lines = [header, "-" * len(header)]
    for strategy_label, row in zip(strategies, matrix):
        values = "  ".join(f"{float(value):>{width}.3f}" for value, width in zip(row, col_widths))
        lines.append(str(strategy_label).ljust(label_width) + "  " + values)
    return "\n".join(lines) + "\n"


def save_payoff_matrix(matrix_result: dict[str, object], path: str | Path) -> Path:
    """Save a payoff matrix result as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(matrix_result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path
