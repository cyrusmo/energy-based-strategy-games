"""Compare pursuit policies in the custom multi-evader pursuit environment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from strategy_games.experiments.pursuit_comparison import (
    compute_pursuit_policy_comparison,
    format_policy_comparison_summary,
    pursuit_policy_comparison_config_from_mapping,
    save_policy_comparison_csv,
    save_policy_comparison_json,
)
from strategy_games.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pursuit policy comparison diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo/pursuit_policy_comparison.yaml"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument(
        "--include-learned-pursuer",
        type=Path,
        default=None,
        help="Optional private PPO checkpoint to include as an additional learned pursuer row.",
    )
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    raw = load_config(args.config)
    if args.num_seeds is not None:
        raw.pop("seeds", None)
        raw["num_seeds"] = args.num_seeds
    if args.eta is not None:
        raw.setdefault("empirical_game", {})["eta"] = args.eta

    config = pursuit_policy_comparison_config_from_mapping(raw, config_path=str(args.config))
    if args.output_json is not None:
        config = replace(config, output_json=args.output_json)
    if args.output_csv is not None:
        config = replace(config, output_csv=args.output_csv)
    if args.no_csv:
        config = replace(config, save_csv=False)
    if args.include_learned_pursuer is not None:
        config = replace(config, learned_pursuer_checkpoint=args.include_learned_pursuer)

    result = compute_pursuit_policy_comparison(config)
    json_path = save_policy_comparison_json(result, config.output_json)
    print(format_policy_comparison_summary(result), end="")
    print(f"policy_comparison_json={json_path}")
    if config.save_csv:
        csv_path = save_policy_comparison_csv(result, config.output_csv)
        print(f"policy_comparison_csv={csv_path}")


if __name__ == "__main__":
    main()
