"""Save a public text and PNG visualization for one heuristic rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

from strategy_games.envs.gridworld import GridworldConfig
from strategy_games.experiments.visualization import collect_heuristic_trace, plot_trace, save_trace_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a deterministic attacker-defender rollout.")
    parser.add_argument("--strategy", default="direct_goal", help="Named attacker strategy.")
    parser.add_argument("--opponent", default="aggressive", help="Named defender response.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public/rollout_demo"))
    args = parser.parse_args()

    env_config = GridworldConfig()
    trace = collect_heuristic_trace(
        strategy_label=args.strategy,
        opponent_label=args.opponent,
        env_config=env_config,
    )
    text_path = save_trace_text(trace, args.output_dir / "trace.txt")
    plot_path = plot_trace(trace, args.output_dir / "trajectory.png", grid_size=env_config.grid_size)
    print(f"trace_txt={text_path}")
    print(f"trajectory_png={plot_path}")


if __name__ == "__main__":
    main()
