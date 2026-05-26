"""Run paper-facing ablations for the gridworld strategy loop."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from strategy_games.experiments.logging import summarize_training_result, to_jsonable
from strategy_games.experiments.runner import run_from_config
from strategy_games.utils.config import load_config


ABLATIONS: dict[str, dict[str, Any]] = {
    "ebm_langevin": {},
    "gaussian_sampler": {"sampler": {"type": "gaussian", "scale": 1.0}},
    "no_buffer_positives": {"updates": {"use_buffer_positives": False}},
    "average_value_selection": {"selection": {"robustness_aware": False}},
    "candidate_count_4": {"training": {"candidate_strategies": 4}},
    "candidate_count_12": {"training": {"candidate_strategies": 12}},
    "opponent_samples_2": {"evaluator": {"episodes_per_opponent": 2}},
    "no_world_model": {"updates": {"train_world_model": False}},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gridworld ablation suite.")
    parser.add_argument("--base-config", type=Path, default=Path("configs/gridworld_day3.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public/ablations"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds to run for each ablation.")
    parser.add_argument("--only", nargs="+", choices=sorted(ABLATIONS), help="Optional subset of ablations.")
    args = parser.parse_args()

    raw_base = load_config(args.base_config)
    selected = args.only or list(ABLATIONS)
    config_dir = args.output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for ablation_name in selected:
        for seed in args.seeds:
            config = build_config(raw_base, ABLATIONS[ablation_name], seed, args.output_dir, ablation_name)
            config_path = config_dir / f"{ablation_name}_seed{seed}.yaml"
            write_yaml(config, config_path)
            result = run_from_config(config_path)
            metrics = summarize_training_result(result)
            rows.append(
                {
                    "ablation": ablation_name,
                    "seed": seed,
                    **metrics,
                }
            )

    summary = summarize_rows(rows)
    write_outputs(args.output_dir, rows, summary)
    print(f"summary_json={args.output_dir / 'summary.json'}")
    print(f"runs_csv={args.output_dir / 'runs.csv'}")


def build_config(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    seed: int,
    output_dir: Path,
    ablation_name: str,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    deep_update(config, override)
    config["seed"] = int(seed)
    logging = dict(config.get("logging", {}))
    logging.update(
        {
            "enabled": True,
            "output_dir": str(output_dir / "runs"),
            "run_name": f"{ablation_name}_seed{seed}",
        }
    )
    config["logging"] = logging
    return config


def deep_update(target: dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["ablation"]), []).append(row)

    summary_rows = []
    for ablation, items in grouped.items():
        metric_values = {
            "mean_episode_return": [float(item.get("mean_episode_return", 0.0)) for item in items],
            "mean_win_rate": [float(item.get("mean_win_rate", 0.0)) for item in items],
            "buffer_diversity": [float(item.get("buffer_diversity", 0.0)) for item in items],
        }
        summary_rows.append(
            {
                "ablation": ablation,
                "seeds": [int(item["seed"]) for item in items],
                "runs": len(items),
                "mean_episode_return": mean(metric_values["mean_episode_return"]),
                "std_episode_return": std(metric_values["mean_episode_return"]),
                "mean_win_rate": mean(metric_values["mean_win_rate"]),
                "std_win_rate": std(metric_values["mean_win_rate"]),
                "mean_buffer_diversity": mean(metric_values["buffer_diversity"]),
                "std_buffer_diversity": std(metric_values["buffer_diversity"]),
            }
        )
    return {
        "runs": sorted(summary_rows, key=lambda item: str(item["ablation"])),
        "notes": [
            "Metrics are diagnostics from sampled-response strategy-loop runs.",
            "exploitability_proxy is approximate and is not Nash exploitability.",
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

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(to_jsonable(rows))

    with (output_dir / "summary_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrr}\n")
        handle.write("\\toprule\n")
        handle.write("Ablation & Return & Win rate & Diversity \\\\\n")
        handle.write("\\midrule\n")
        for row in summary["runs"]:
            handle.write(
                f"{row['ablation']} & "
                f"{float(row['mean_episode_return']):.3f} $\\pm$ {float(row['std_episode_return']):.3f} & "
                f"{float(row['mean_win_rate']):.3f} $\\pm$ {float(row['std_win_rate']):.3f} & "
                f"{float(row['mean_buffer_diversity']):.3f} $\\pm$ {float(row['std_buffer_diversity']):.3f} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def write_yaml(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_jsonable(config), handle, sort_keys=False)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


if __name__ == "__main__":
    main()
