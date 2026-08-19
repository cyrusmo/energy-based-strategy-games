"""Run held-out gridworld layouts to expose direct-goal overfitting."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from strategy_games.experiments.heldout_layouts import (
    heldout_layout_config_from_mapping,
    run_heldout_layout_study,
)
from strategy_games.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baselines on default vs held-out grid layouts.")
    parser.add_argument("--config", type=Path, default=Path("configs/gridworld_heldout_layouts.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--strategy-iterations", type=int, default=None)
    parser.add_argument("--no-strategy-loop", action="store_true")
    parser.add_argument("--include-ppo", action="store_true")
    parser.add_argument("--no-ppo", action="store_true", help="Explicitly skip PPO (default).")
    args = parser.parse_args()

    config = heldout_layout_config_from_mapping(load_config(args.config))
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.episodes is not None:
        config = replace(config, episodes=args.episodes)
    if args.strategy_iterations is not None:
        config = replace(config, strategy_iterations=args.strategy_iterations)
    if args.no_strategy_loop:
        config = replace(config, include_strategy_loop=False)
    if args.include_ppo:
        config = replace(config, include_ppo=True)
    if args.no_ppo:
        config = replace(config, include_ppo=False)

    result = run_heldout_layout_study(config)
    summary = result["summary"]
    print(f"summary_json={config.output_dir / 'summary.json'}")
    print(f"rows_csv={config.output_dir / 'rows.csv'}")
    print(f"direct_goal_overfitting_visible={summary['direct_goal_overfitting_visible']}")
    for row in summary["baselines"]:
        print(
            f"{row['baseline']}: default_goal={row['default_goal_rate']:.3f} "
            f"heldout_goal={row['heldout_goal_rate']:.3f} drop={row['goal_rate_drop']:.3f}"
        )


if __name__ == "__main__":
    main()
