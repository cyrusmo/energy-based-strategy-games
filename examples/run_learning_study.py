"""Run the gridworld learning-quality study."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from strategy_games.experiments.learning_study import (
    learning_study_config_from_mapping,
    run_learning_study,
)
from strategy_games.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gridworld strategy-loop/PPO learning diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("configs/gridworld_learning_study.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--strategy-iterations", type=int, default=None)
    parser.add_argument("--ppo-total-steps", type=int, default=None)
    parser.add_argument("--no-ppo", action="store_true", help="Skip PPO for a fast strategy-loop-only run.")
    args = parser.parse_args()

    config = learning_study_config_from_mapping(load_config(args.config))
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.seeds is not None:
        config = replace(config, seeds=tuple(args.seeds))
    if args.strategy_iterations is not None:
        config = replace(config, strategy_iterations=args.strategy_iterations)
    if args.ppo_total_steps is not None:
        config = replace(config, ppo_total_steps=args.ppo_total_steps)
    if args.no_ppo:
        config = replace(config, include_ppo=False)

    result = run_learning_study(config)
    summary_path = config.output_dir / "summary.json"
    curves_path = config.output_dir / "learning_curves.csv"
    triage_path = config.output_dir / "triage_table.csv"
    print(f"summary_json={summary_path}")
    print(f"learning_curves_csv={curves_path}")
    print(f"triage_table_csv={triage_path}")
    print(f"methods={','.join(str(row['method']) for row in result['summary']['methods'])}")


if __name__ == "__main__":
    main()
