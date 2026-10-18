"""Gate 2: does the full method beat PPO and diversity-only off the train map?"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from strategy_games.envs.layouts_v2 import (
    HELD_OUT_LAYOUT_V2_NAMES,
    TRAIN_LAYOUT_V2_NAME,
    get_layout_v2,
)
from strategy_games.experiments.convergence import multiseed_confidence
from strategy_games.experiments.diversity_baseline import run_diversity_baseline
from strategy_games.experiments.logging import to_jsonable
from strategy_games.training.ppo_baseline import PPOConfig, train_ppo_baseline
from strategy_games.training.psro import PSROConfig, run_psro
from strategy_games.training.train_loop import TrainingConfig, run_training_loop


@dataclass(frozen=True)
class Gate2Config:
    """Cheap but 5-seed comparison used to decide paper framing."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    strategy_iterations: int = 3
    ppo_total_steps: int = 256
    psro_iterations: int = 2
    candidate_strategies: int = 8
    output_dir: Path = Path("outputs/public/gate2")


def run_gate2(config: Gate2Config | None = None) -> dict[str, Any]:
    """Compare full method, PPO, diversity-only, and PSRO on train + held-out maps."""

    config = config or Gate2Config()
    rows: list[dict[str, Any]] = []
    train = get_layout_v2(TRAIN_LAYOUT_V2_NAME)
    heldout = [get_layout_v2(name) for name in HELD_OUT_LAYOUT_V2_NAMES[:3]]

    for seed in config.seeds:
        rows.extend(_seed_rows(config, seed, train, heldout))

    summary = summarize_gate2(rows)
    artifacts = _write(config.output_dir, rows, summary)
    return {"rows": rows, "summary": summary, "artifacts": artifacts}


def summarize_gate2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/CI by method on held-out goal rate and train return."""

    methods = sorted({str(row["method"]) for row in rows})
    by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        held = [float(row["heldout_goal_rate"]) for row in rows if row["method"] == method]
        train = [float(row["train_return"]) for row in rows if row["method"] == method]
        by_method[method] = {
            "method": method,
            "heldout_goal_rate": _ci(held),
            "train_return": _ci(train),
            "n": len(held),
        }
    full = by_method.get("strategy_loop")
    ppo = by_method.get("ppo_baseline")
    diversity = by_method.get("diversity_only")
    beats_ppo = _beats(full, ppo)
    beats_div = _beats(full, diversity)
    passed = bool(beats_ppo and beats_div)
    return {
        "schema_version": "gate2/v1",
        "methods": list(by_method.values()),
        "beats_ppo": beats_ppo,
        "beats_diversity_only": beats_div,
        "passed": passed,
        "framing": "positive" if passed else "diagnostic",
        "notes": [
            "Held-out goal rate is the transfer metric. Exact Nash exploitability is computed in the full study when budgets allow.",
            "Non-overlapping 95% CIs on held-out goal rate are required to claim a positive result.",
        ],
    }


def _seed_rows(config: Gate2Config, seed: int, train: Any, heldout: list[Any]) -> list[dict[str, Any]]:
    strategy = run_training_loop(
        TrainingConfig(
            seed=seed,
            iterations=config.strategy_iterations,
            candidate_strategies=config.candidate_strategies,
            max_heuristic_candidates=1,
            sampler_type="langevin",
            evaluator_action_source="heuristic",
            executor_action_source="parametric",
            robustness_aware_selection=True,
            use_robustness_g=True,
            env=train,
            train_policy=False,
            train_world_model=False,
        )
    )
    diversity = run_diversity_baseline(env=train, seed=seed, iterations=config.strategy_iterations)
    ppo = train_ppo_baseline(
        PPOConfig(seed=seed, total_steps=config.ppo_total_steps, env=train, eval_episodes=2, eval_every_updates=0)
    )
    psro = run_psro(PSROConfig(env=train, iterations=config.psro_iterations, seed=seed))
    return [
        _row("strategy_loop", seed, strategy, train, heldout),
        _row("diversity_only", seed, diversity, train, heldout),
        _row("ppo_baseline", seed, ppo, train, heldout),
        _row("psro_lite", seed, psro, train, heldout),
    ]


def _row(method: str, seed: int, result: dict[str, Any], train: Any, heldout: list[Any]) -> dict[str, Any]:
    from strategy_games.evaluation.best_response import GameTheoreticEvaluator

    train_return = float(result.get("episode_return", _history_return(result)))
    label = str(result.get("final_selected_label") or "direct_goal")
    strategy = _selected_strategy(result, label)
    eval_label = None if label == "latent" else label
    held_rates = []
    for env in heldout:
        evaluator = GameTheoreticEvaluator(env_config=env, episodes_per_opponent=1, strategy_dim=8)
        metrics = evaluator.evaluate_strategy(strategy, label=eval_label)
        held_rates.append(float(metrics["goal_rate"]))
    return {
        "method": method,
        "seed": seed,
        "train_return": train_return,
        "heldout_goal_rate": float(np.mean(held_rates) if held_rates else 0.0),
        "final_selected_label": label,
        "env_steps": int(result.get("env_steps", result.get("total_steps", 0) or 0)),
    }


def _selected_strategy(result: dict[str, Any], label: str):
    import torch

    from strategy_games.strategies.embeddings import named_strategy_embedding

    history = result.get("history", [])
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict) and last.get("selected_strategy") is not None:
            return torch.tensor(last["selected_strategy"], dtype=torch.float32)
    if label in {"aggressive", "evasive", "feint", "patient", "direct_goal"}:
        return named_strategy_embedding(label, 8).vector
    return named_strategy_embedding("direct_goal", 8).vector


def _history_return(result: dict[str, Any]) -> float:
    history = result.get("history", [])
    if not isinstance(history, list) or not history:
        return float(result.get("meta_nash_value", 0.0))
    rollouts = [item.get("rollout", {}) for item in history if isinstance(item, dict)]
    values = [float(item["episode_return"]) for item in rollouts if isinstance(item, dict) and "episode_return" in item]
    return float(np.mean(values)) if values else 0.0


def _ci(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    stats = multiseed_confidence(values)
    return {
        "mean": float(stats.get("mean", float(np.mean(values)))),
        "ci_low": float(stats.get("ci_low", float(np.min(values)))),
        "ci_high": float(stats.get("ci_high", float(np.max(values)))),
    }


def _beats(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None) -> bool:
    if candidate is None or baseline is None:
        return False
    cand = candidate["heldout_goal_rate"]
    base = baseline["heldout_goal_rate"]
    return bool(cand["mean"] > base["mean"] and cand["ci_low"] > base["ci_high"])


def _write(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.json"
    summary_path = output_dir / "summary.json"
    rows_path.write_text(json.dumps(to_jsonable(rows), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    return {"rows_json": str(rows_path), "summary_json": str(summary_path)}
