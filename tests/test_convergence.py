import pytest

from strategy_games.experiments.convergence import detect_convergence, energy_gap, multiseed_confidence, summarize_convergence


def test_detect_convergence_reaches_target_with_patience() -> None:
    history = [
        {"rollout": {"goal_rate": 0.0}},
        {"rollout": {"goal_rate": 1.0}},
        {"rollout": {"goal_rate": 1.0}},
    ]
    result = detect_convergence(history, metric="goal_rate", target=1.0, patience=2, min_iter=1)
    assert result["converged"] is True
    assert result["iteration_reached"] == 1


def test_detect_convergence_reports_not_yet() -> None:
    history = [{"rollout": {"episode_return": -1.0}}, {"rollout": {"episode_return": 0.2}}]
    result = detect_convergence(history, metric="episode_return", target=0.9)
    assert result["converged"] is False
    assert result["iteration_reached"] == -1


def test_multiseed_confidence_returns_interval() -> None:
    result = multiseed_confidence([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert float(result["ci_low"]) < 2.0 < float(result["ci_high"])


def test_energy_gap_and_summary() -> None:
    updates = {"positive_energy": 1.0, "negative_energy": 2.5}
    assert energy_gap(updates) == pytest.approx(1.5)
    summary = summarize_convergence([{"rollout": {"goal_rate": 1.0, "episode_return": 1.0}, "updates": updates}])
    assert "goal_rate" in summary
    assert "ebm_energy_gap" in summary
