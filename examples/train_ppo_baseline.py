"""Train the public PPO-lite baseline on the custom gridworld."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strategy_games.training.ppo_baseline import train_ppo_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO-lite attacker baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gridworld_ppo_baseline.yaml"),
        help="PPO baseline YAML config.",
    )
    args = parser.parse_args()

    result = train_ppo_from_config(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
