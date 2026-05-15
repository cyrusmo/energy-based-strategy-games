"""Compare random, heuristic, and Day 2 strategy-loop baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from strategy_games.experiments.baselines import compare_baselines, format_baseline_table, save_baseline_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare public gridworld baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/gridworld_day2.yaml"))
    parser.add_argument("--ppo-config", type=Path, default=Path("configs/gridworld_ppo_baseline.yaml"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/public/baselines/metrics.json"))
    parser.add_argument("--no-ppo", action="store_true", help="Skip the PPO-lite baseline row.")
    parser.add_argument("--no-save", action="store_true", help="Print only; do not write JSON.")
    args = parser.parse_args()

    rows = compare_baselines(
        config_path=args.config,
        episodes=args.episodes,
        include_ppo=not args.no_ppo,
        ppo_config_path=args.ppo_config,
    )
    print(format_baseline_table(rows), end="")
    if not args.no_save:
        path = save_baseline_metrics(rows, args.output)
        print(f"metrics_json={path}")


if __name__ == "__main__":
    main()
