"""Run the paper-facing multi-seed baseline protocol."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from strategy_games.experiments.baselines import BASELINE_FIELDS, compare_baselines
from strategy_games.experiments.convergence import multiseed_confidence
from strategy_games.experiments.logging import to_jsonable
from strategy_games.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed gridworld baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/gridworld_day3.yaml"))
    parser.add_argument("--ppo-config", type=Path, default=Path("configs/gridworld_ppo_baseline.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public/multiseed"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--ppo-total-steps", type=int, default=2048)
    parser.add_argument("--no-ppo", action="store_true", help="Skip PPO for faster diagnostic runs.")
    args = parser.parse_args()

    config_dir = args.output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        strategy_config = configure_strategy_loop(load_config(args.config), seed, args.output_dir)
        strategy_path = config_dir / f"strategy_seed{seed}.yaml"
        write_yaml(strategy_config, strategy_path)

        ppo_path = args.ppo_config
        if not args.no_ppo:
            ppo_config = configure_ppo(load_config(args.ppo_config), seed, args.ppo_total_steps, args.output_dir)
            ppo_path = config_dir / f"ppo_seed{seed}.yaml"
            write_yaml(ppo_config, ppo_path)

        for row in compare_baselines(
            config_path=strategy_path,
            episodes=args.episodes,
            seed=seed,
            include_ppo=not args.no_ppo,
            ppo_config_path=ppo_path,
        ):
            rows.append({"seed": seed, **row})

    summary = summarize_by_baseline(rows)
    write_outputs(args.output_dir, rows, summary)
    print(f"summary_json={args.output_dir / 'summary.json'}")
    print(f"runs_csv={args.output_dir / 'runs.csv'}")


def configure_strategy_loop(raw: Mapping[str, Any], seed: int, output_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(dict(raw))
    config["seed"] = int(seed)
    logging = dict(config.get("logging", {}))
    logging.update(
        {
            "enabled": True,
            "output_dir": str(output_dir / "strategy_runs"),
            "run_name": f"strategy_seed{seed}",
        }
    )
    config["logging"] = logging
    return config


def configure_ppo(raw: Mapping[str, Any], seed: int, total_steps: int, output_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(dict(raw))
    config["seed"] = int(seed)
    ppo = dict(config.get("ppo", {}))
    ppo["total_steps"] = int(total_steps)
    config["ppo"] = ppo
    logging = dict(config.get("logging", {}))
    logging.update({"enabled": True, "output_dir": str(output_dir / "ppo_runs" / f"seed{seed}")})
    config["logging"] = logging
    return config


def summarize_by_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["baseline"]), []).append(row)

    summary_rows = []
    for baseline, items in grouped.items():
        result: dict[str, Any] = {
            "baseline": baseline,
            "seeds": [int(item["seed"]) for item in items],
            "runs": len(items),
        }
        for field in BASELINE_FIELDS[1:]:
            values = [float(item[field]) for item in items]
            confidence = multiseed_confidence(values)
            result[f"{field}_mean"] = float(confidence["mean"])
            result[f"{field}_std"] = float(confidence["std"])
            result[f"{field}_ci_low"] = float(confidence["ci_low"])
            result[f"{field}_ci_high"] = float(confidence["ci_high"])
        summary_rows.append(result)
    return {
        "baselines": sorted(summary_rows, key=lambda item: str(item["baseline"])),
        "notes": [
            "PPO uses an extended training budget relative to the May 15 smoke run.",
            "The direct-goal heuristic is intentionally included as a strong sanity-check baseline.",
        ],
    }


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "runs.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(rows), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    fieldnames = ["seed", *BASELINE_FIELDS]
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(to_jsonable(rows))

    with (output_dir / "summary_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrr}\n")
        handle.write("\\toprule\n")
        handle.write("Baseline & Return & Win rate \\\\\n")
        handle.write("\\midrule\n")
        for row in summary["baselines"]:
            handle.write(
                f"{row['baseline']} & "
                f"{float(row['episode_return_mean']):.3f} $\\pm$ {float(row['episode_return_std']):.3f} & "
                f"{float(row['win_rate_mean']):.3f} $\\pm$ {float(row['win_rate_std']):.3f} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def write_yaml(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_jsonable(config), handle, sort_keys=False)


if __name__ == "__main__":
    main()
