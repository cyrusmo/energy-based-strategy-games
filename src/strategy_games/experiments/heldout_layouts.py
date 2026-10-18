"""Held-out layout protocol for exposing direct-goal overfitting."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from strategy_games.envs.layouts import (
    HELD_OUT_LAYOUT_NAMES,
    TRAIN_LAYOUT_NAME,
    get_layout,
    layout_names,
)
from strategy_games.envs.layouts_v2 import HELD_OUT_LAYOUTS_V2, get_layout_v2
from strategy_games.experiments.logging import to_jsonable
from strategy_games.experiments.runner import run_from_config
from strategy_games.training.ppo_baseline import (
    ppo_config_from_mapping,
    run_direct_goal_baseline,
    run_random_policy_baseline,
    train_ppo_baseline,
)
from strategy_games.utils.config import load_config

HELDOUT_ROW_FIELDS = (
    "layout",
    "held_out",
    "baseline",
    "episode_return",
    "win_rate",
    "goal_rate",
    "catch_rate",
    "timeout_rate",
    "final_selected_label",
)


@dataclass(frozen=True)
class HeldoutLayoutStudyConfig:
    """Config for comparing baselines on default vs held-out layouts."""

    layouts: tuple[str, ...] = layout_names()
    episodes: int = 8
    seed: int = 0
    strategy_config_path: Path = Path("configs/gridworld_day3.yaml")
    ppo_config_path: Path = Path("configs/gridworld_ppo_baseline.yaml")
    output_dir: Path = Path("outputs/public/heldout_layouts")
    include_strategy_loop: bool = True
    include_ppo: bool = False
    strategy_iterations: int = 4
    ppo_total_steps: int = 256


def run_heldout_layout_study(config: HeldoutLayoutStudyConfig | None = None) -> dict[str, Any]:
    """Evaluate baselines on named layouts and write public artifacts."""

    config = config or HeldoutLayoutStudyConfig()
    if not config.layouts:
        raise ValueError("layouts must be non-empty")
    if config.episodes < 1:
        raise ValueError("episodes must be positive")

    rows: list[dict[str, Any]] = []
    for layout_name in config.layouts:
        env_config = get_layout_v2(layout_name) if layout_name in HELD_OUT_LAYOUTS_V2 else get_layout(layout_name)
        held_out = layout_name != TRAIN_LAYOUT_NAME
        rows.append(_baseline_row("random_policy", layout_name, held_out, run_random_policy_baseline(
            episodes=config.episodes, seed=config.seed, env_config=env_config
        )))
        rows.append(_baseline_row("direct_goal_heuristic", layout_name, held_out, run_direct_goal_baseline(
            episodes=config.episodes, env_config=env_config
        )))
        if config.include_strategy_loop:
            strategy_result = _run_strategy_loop(config, layout_name, env_config)
            rows.append(_strategy_row(layout_name, held_out, strategy_result))
        if config.include_ppo:
            ppo_raw = load_config(config.ppo_config_path)
            ppo_cfg = ppo_config_from_mapping(ppo_raw)
            ppo_cfg = _ppo_config_for_layout(ppo_cfg, env_config, config)
            ppo_result = train_ppo_baseline(ppo_cfg)
            rows.append(_baseline_row("ppo_baseline", layout_name, held_out, ppo_result))

    summary = summarize_heldout_rows(rows)
    artifacts = write_heldout_outputs(config.output_dir, rows, summary)
    return {"rows": rows, "summary": summary, "artifacts": artifacts}


def heldout_layout_config_from_mapping(raw: Mapping[str, Any]) -> HeldoutLayoutStudyConfig:
    """Build a held-out layout study config from YAML-style data."""

    study_raw = raw.get("study", raw)
    if not isinstance(study_raw, Mapping):
        study_raw = {}
    layouts = study_raw.get("layouts", layout_names())
    return HeldoutLayoutStudyConfig(
        layouts=tuple(str(name) for name in layouts),
        episodes=int(study_raw.get("episodes", 8)),
        seed=int(study_raw.get("seed", raw.get("seed", 0))),
        strategy_config_path=Path(str(study_raw.get("strategy_config", "configs/gridworld_day3.yaml"))),
        ppo_config_path=Path(str(study_raw.get("ppo_config", "configs/gridworld_ppo_baseline.yaml"))),
        output_dir=Path(str(study_raw.get("output_dir", "outputs/public/heldout_layouts"))),
        include_strategy_loop=bool(study_raw.get("include_strategy_loop", True)),
        include_ppo=bool(study_raw.get("include_ppo", False)),
        strategy_iterations=int(study_raw.get("strategy_iterations", 4)),
        ppo_total_steps=int(study_raw.get("ppo_total_steps", 256)),
    )


def summarize_heldout_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize default vs held-out gaps, especially for direct-goal."""

    by_baseline: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        baseline = str(row["baseline"])
        split = "held_out" if bool(row["held_out"]) else "default"
        by_baseline.setdefault(baseline, {"default": [], "held_out": []})[split].append(float(row["goal_rate"]))

    baselines = []
    for baseline, splits in sorted(by_baseline.items()):
        default_mean = _mean(splits["default"])
        heldout_mean = _mean(splits["held_out"])
        baselines.append(
            {
                "baseline": baseline,
                "default_goal_rate": default_mean,
                "heldout_goal_rate": heldout_mean,
                "goal_rate_drop": default_mean - heldout_mean,
            }
        )
    direct = next((item for item in baselines if item["baseline"] == "direct_goal_heuristic"), None)
    failed_layouts = sorted(
        {
            str(row["layout"])
            for row in rows
            if str(row["baseline"]) == "direct_goal_heuristic"
            and bool(row["held_out"])
            and float(row["goal_rate"]) < 0.5
        }
    )
    overfitting_visible = bool(direct is not None and direct["goal_rate_drop"] >= 0.25)
    return {
        "schema_version": "gridworld_heldout_layouts/v1",
        "train_layout": TRAIN_LAYOUT_NAME,
        "held_out_layouts": list(HELD_OUT_LAYOUT_NAMES),
        "baselines": baselines,
        "direct_goal_failed_layouts": failed_layouts,
        "direct_goal_overfitting_visible": overfitting_visible,
        "notes": [
            "Held-out layouts change start, goal, or defender placement relative to the default map.",
            "A large direct-goal goal-rate drop is evidence of heuristic overfitting, not of a solved transfer task.",
            "exploitability_proxy is not computed in this protocol.",
        ],
    }


def write_heldout_outputs(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> dict[str, str]:
    """Write held-out layout artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.csv"
    summary_path = output_dir / "summary.json"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HELDOUT_ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(to_jsonable(list(rows)))
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"rows_csv": str(rows_path), "summary_json": str(summary_path)}


def _run_strategy_loop(config: HeldoutLayoutStudyConfig, layout_name: str, env_config: Any) -> dict[str, Any]:
    raw = copy.deepcopy(dict(load_config(config.strategy_config_path)))
    raw["seed"] = int(config.seed)
    raw["env"] = _env_mapping(env_config)
    training = dict(raw.get("training", {}))
    training["iterations"] = int(config.strategy_iterations)
    raw["training"] = training
    logging = dict(raw.get("logging", {}))
    logging.update(
        {
            "enabled": True,
            "output_dir": str(config.output_dir / "strategy_runs"),
            "run_name": f"{layout_name}_seed{config.seed}",
        }
    )
    raw["logging"] = logging
    config_dir = config.output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"strategy_{layout_name}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_jsonable(raw), handle, sort_keys=False)
    return dict(run_from_config(path))


def _ppo_config_for_layout(ppo_cfg: Any, env_config: Any, config: HeldoutLayoutStudyConfig) -> Any:
    return replace(
        ppo_cfg,
        seed=config.seed,
        total_steps=config.ppo_total_steps,
        env=env_config,
        eval_episodes=max(1, min(config.episodes, 4)),
        eval_every_updates=0,
    )


def _baseline_row(name: str, layout_name: str, held_out: bool, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "layout": layout_name,
        "held_out": bool(held_out),
        "baseline": name,
        "episode_return": float(metrics.get("episode_return", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "goal_rate": float(metrics.get("goal_rate", 0.0)),
        "catch_rate": float(metrics.get("catch_rate", 0.0)),
        "timeout_rate": float(metrics.get("timeout_rate", 0.0)),
        "final_selected_label": "",
    }


def _strategy_row(layout_name: str, held_out: bool, result: Mapping[str, Any]) -> dict[str, Any]:
    history = result.get("history", [])
    rollouts = [item.get("rollout", {}) for item in history if isinstance(item, Mapping)]
    def _mean_metric(key: str) -> float:
        values = [float(item[key]) for item in rollouts if isinstance(item, Mapping) and key in item]
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "layout": layout_name,
        "held_out": bool(held_out),
        "baseline": "strategy_loop",
        "episode_return": _mean_metric("episode_return"),
        "win_rate": _mean_metric("win_rate"),
        "goal_rate": _mean_metric("goal_rate"),
        "catch_rate": _mean_metric("catch_rate"),
        "timeout_rate": _mean_metric("timeout_rate"),
        "final_selected_label": str(result.get("final_selected_label", "")),
    }


def _env_mapping(env_config: Any) -> dict[str, Any]:
    payload = {
        "grid_size": env_config.grid_size,
        "max_steps": env_config.max_steps,
        "attacker_start": list(env_config.attacker_start),
        "defender_start": list(env_config.defender_start),
        "catch_radius": env_config.catch_radius,
    }
    if hasattr(env_config, "goal_positions"):
        payload["version"] = "v2"
        payload["goal_positions"] = [list(goal) for goal in env_config.goal_positions]
        payload["obstacles"] = [list(item) for item in sorted(env_config.obstacles)]
        payload["slip_prob"] = float(getattr(env_config, "slip_prob", 0.0))
    else:
        payload["goal_pos"] = list(env_config.goal_pos)
    return payload


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
