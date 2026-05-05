from strategy_games.experiments.payoff import compute_payoff_matrix, format_payoff_matrix


def test_payoff_matrix_shape_and_best_responses() -> None:
    strategies = ("direct_goal", "patient")
    opponents = ("aggressive", "direct_goal", "patient")
    result = compute_payoff_matrix(
        strategy_labels=strategies,
        opponent_labels=opponents,
        strategy_dim=8,
        episodes_per_opponent=1,
    )

    assert result["strategy_labels"] == list(strategies)
    assert result["opponent_labels"] == list(opponents)
    matrix = result["average_reward_matrix"]
    assert len(matrix) == len(strategies)
    assert all(len(row) == len(opponents) for row in matrix)
    assert set(result["best_response_labels"]) == set(strategies)
    assert set(result["best_response_labels"].values()).issubset(opponents)

    text = format_payoff_matrix(result)
    assert "strategy" in text
    assert "direct_goal" in text
