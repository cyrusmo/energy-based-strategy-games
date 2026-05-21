"""Train the tiny PPO-lite pursuer baseline for the multi-evader pursuit game."""

from __future__ import annotations

import argparse
from pathlib import Path

from strategy_games.training.ppo_pursuit import (
    pursuit_ppo_config_from_mapping,
    train_ppo_pursuer,
)
from strategy_games.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny PPO pursuer smoke baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo/ppo_pursuer_smoke.yaml"))
    args = parser.parse_args()

    config = pursuit_ppo_config_from_mapping(load_config(args.config))
    artifact = train_ppo_pursuer(config)
    eval_metrics = artifact["eval_metrics"]

    print("ppo_pursuer_training=complete")
    print(f"training_run_id={artifact['training_run_id']}")
    print(f"policy_id={artifact['policy_id']}")
    print(f"metrics_json={config.output_dir / 'metrics.json'}")
    print(f"config_json={config.output_dir / 'config.json'}")
    print(f"checkpoint_written={artifact['checkpoint_written']}")
    print(
        "eval "
        f"capture_rate={float(eval_metrics['capture_rate']):.3f} "
        f"survival_rate={float(eval_metrics['survival_rate']):.3f} "
        f"mean_pursuer_return={float(eval_metrics['mean_pursuer_return']):.3f}"
    )


if __name__ == "__main__":
    main()
