"""Gate 1 checks: the v2 payoff matrix must be a real game, not a dominant heuristic."""

from __future__ import annotations

from typing import Any

import numpy as np

from strategy_games.envs.layouts_v2 import (
    GATE1_UNIQUE_BEST_LAYOUTS,
    TRAIN_LAYOUT_V2_NAME,
    get_layout_v2,
)
from strategy_games.experiments.payoff import compute_payoff_matrix
from strategy_games.strategies.embeddings import available_heuristic_strategies


def summarize_payoff_matrix(matrix_result: dict[str, Any]) -> dict[str, Any]:
    """Return dominance / best-response diagnostics for a payoff matrix."""

    strategies = list(matrix_result["strategy_labels"])
    matrix = np.asarray(matrix_result["average_reward_matrix"], dtype=np.float64)
    best_responses = dict(matrix_result["best_response_labels"])
    row_means = matrix.mean(axis=1)
    unique_br = sorted(set(best_responses.values()))
    dominating = _weakly_dominating_rows(matrix, strategies)
    best_idx = int(np.argmax(row_means))
    ties = [strategies[i] for i, mean in enumerate(row_means) if abs(mean - row_means[best_idx]) < 1e-9]
    return {
        "strategy_labels": strategies,
        "row_means": {label: float(mean) for label, mean in zip(strategies, row_means)},
        "best_response_labels": best_responses,
        "distinct_best_response_count": len(unique_br),
        "distinct_best_response_labels": unique_br,
        "weakly_dominating_rows": dominating,
        "unique_best_strategy": ties[0] if len(ties) == 1 else None,
        "tied_best_strategies": ties,
    }


def evaluate_gate1(strategy_dim: int = 8) -> dict[str, Any]:
    """Run Gate 1 on the train layout plus the unique-best held-out maps."""

    train = compute_payoff_matrix(env_config=get_layout_v2(TRAIN_LAYOUT_V2_NAME), strategy_dim=strategy_dim)
    train_summary = summarize_payoff_matrix(train)
    unique_best: dict[str, Any] = {}
    for label, layout_name in GATE1_UNIQUE_BEST_LAYOUTS.items():
        result = compute_payoff_matrix(env_config=get_layout_v2(layout_name), strategy_dim=strategy_dim)
        unique_best[label] = {
            "layout": layout_name,
            "summary": summarize_payoff_matrix(result),
            "matrix": result,
        }
    passed = (
        train_summary["distinct_best_response_count"] >= 3
        and not train_summary["weakly_dominating_rows"]
        and all(item["summary"]["unique_best_strategy"] == label for label, item in unique_best.items())
    )
    return {
        "passed": passed,
        "train_layout": TRAIN_LAYOUT_V2_NAME,
        "train": {"matrix": train, "summary": train_summary},
        "unique_best": unique_best,
        "heuristic_strategies": list(available_heuristic_strategies()),
    }


def _weakly_dominating_rows(matrix: np.ndarray, strategies: list[str]) -> list[str]:
    dominating: list[str] = []
    for i, label in enumerate(strategies):
        others = [j for j in range(len(strategies)) if j != i]
        if not others:
            continue
        if all(np.all(matrix[i] + 1e-9 >= matrix[j]) for j in others) and any(
            np.any(matrix[i] > matrix[j] + 1e-9) for j in others
        ):
            dominating.append(label)
        if all(np.allclose(matrix[i], matrix[j]) for j in others):
            dominating.append(label)
    return dominating
