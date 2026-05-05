"""Run the config-driven Generate -> Evaluate -> Execute -> Update loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strategy_games.experiments.runner import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small strategy-games training loop.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gridworld_day3.yaml"),
        help="Path to a YAML experiment config.",
    )
    args = parser.parse_args()
    result = run_from_config(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
