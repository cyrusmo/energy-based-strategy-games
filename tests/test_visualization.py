from strategy_games.envs.gridworld import GridworldConfig
from strategy_games.experiments.visualization import collect_heuristic_trace, format_trace_text, plot_trace, save_trace_text


def test_rollout_trace_text_and_plot(tmp_path) -> None:
    trace = collect_heuristic_trace(
        strategy_label="direct_goal",
        opponent_label="aggressive",
        env_config=GridworldConfig(grid_size=5, max_steps=8, defender_start=(4, 4), goal_pos=(4, 0)),
    )

    assert trace.attacker_positions[0] == (0, 0)
    assert len(trace.attacker_positions) == trace.steps + 1
    assert len(trace.defender_positions) == trace.steps + 1
    assert len(trace.attacker_actions) == trace.steps
    assert len(trace.rewards) == trace.steps
    assert trace.outcome in {"goal", "caught", "timeout"}

    text = format_trace_text(trace)
    assert "step=00" in text
    assert "outcome=" in text
    assert "total_return=" in text

    text_path = save_trace_text(trace, tmp_path / "trace.txt")
    plot_path = plot_trace(trace, tmp_path / "trajectory.png", grid_size=5)
    assert text_path.read_text(encoding="utf-8") == text
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
