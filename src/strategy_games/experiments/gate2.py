"""Gate 2: does the full method have lower exact exploitability than PPO and diversity-only?"""

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
from strategy_games.evaluation.best_response import GameTheoreticEvaluator
from strategy_games.evaluation.exact import (
    exact_metrics_for_policy,
    policy_from_actor_critic,
    policy_from_mixture,
    policy_from_strategy,
)
from strategy_games.experiments.convergence import multiseed_confidence
from strategy_games.experiments.diversity_baseline import run_diversity_baseline
from strategy_games.experiments.logging import to_jsonable
from strategy_games.training.ppo_baseline import PPOConfig, train_ppo_baseline
from strategy_games.training.psro import PSROConfig, run_psro
from strategy_games.training.train_loop import TrainingConfig, run_training_loop


@dataclass(frozen=True)
class Gate2Config:
    """5-seed comparison used to decide paper framing."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    strategy_iterations: int = 8
    ppo_total_steps: int = 4096
    psro_iterations: int = 2
    candidate_strategies: int = 16
    output_dir: Path = Path("outputs/public/gate2")


def run_gate2(config: Gate2Config | None = None) -> dict[str, Any]:
    """Compare full method, PPO, diversity-only, and PSRO on train + held-out maps."""

    config = config or Gate2Config()
    rows: list[dict[str, Any]] = []
    train = get_layout_v2(TRAIN_LAYOUT_V2_NAME)
    heldout = [(name, get_layout_v2(name)) for name in HELD_OUT_LAYOUT_V2_NAMES[:3]]

    for seed in config.seeds:
        rows.extend(_seed_rows(config, seed, train, heldout))

    summary = summarize_gate2(rows)
    artifacts = _write(config.output_dir, rows, summary)
    return {"rows": rows, "summary": summary, "artifacts": artifacts}


def summarize_gate2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/CI by method on exact exploitability (lower is better)."""

    methods = sorted({str(row["method"]) for row in rows})
    by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "method": method,
            "exploitability": _ci([float(row["exploitability"]) for row in method_rows]),
            "train_exploitability": _ci([float(row["train_exploitability"]) for row in method_rows]),
            "heldout_exploitability": _ci([float(row["heldout_exploitability"]) for row in method_rows]),
            "exact_value_vs_br": _ci([float(row["exact_value_vs_br"]) for row in method_rows]),
            "heldout_goal_rate": _ci([float(row["heldout_goal_rate"]) for row in method_rows]),
            "strategy_entropy": _ci([float(row.get("strategy_entropy", 0.0)) for row in method_rows]),
            "env_steps": _ci([float(row.get("env_steps", 0.0)) for row in method_rows]),
            "n": len(method_rows),
        }
    full = by_method.get("strategy_loop")
    ppo = by_method.get("ppo_baseline")
    diversity = by_method.get("diversity_only")
    beats_ppo = _beats_lower(full, ppo, "exploitability")
    beats_div = _beats_lower(full, diversity, "exploitability")
    passed = bool(beats_ppo and beats_div)
    return {
        "schema_version": "gate2/v2",
        "methods": list(by_method.values()),
        "beats_ppo": beats_ppo,
        "beats_diversity_only": beats_div,
        "passed": passed,
        "framing": "positive" if passed else "diagnostic",
        "notes": [
            "Primary metric is exact Nash exploitability (lower is better), averaged over the train layout and held-out layouts.",
            "Pass requires the strategy-loop 95% CI to lie entirely below both PPO and diversity-only.",
        ],
    }


def _seed_rows(
    config: Gate2Config,
    seed: int,
    train: Any,
    heldout: list[tuple[str, Any]],
) -> list[dict[str, Any]]:
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
            compute_exact=True,
            timeout_penalty_weight=2.0,
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


def _row(
    method: str,
    seed: int,
    result: dict[str, Any],
    train: Any,
    heldout: list[tuple[str, Any]],
) -> dict[str, Any]:
    attacker = _attacker_from_result(method, result)
    train_exact = exact_metrics_for_policy(train, attacker)
    held_exact = []
    held_goal = []
    held_by_layout: dict[str, float] = {}
    for name, env in heldout:
        metrics = exact_metrics_for_policy(env, attacker)
        held_exact.append(float(metrics["exploitability"]))
        held_by_layout[name] = float(metrics["exploitability"])
        evaluator = GameTheoreticEvaluator(env_config=env, episodes_per_opponent=1, strategy_dim=8)
        sampled = evaluator.evaluate_strategy(_selected_strategy(result, str(result.get("final_selected_label") or "direct_goal")))
        held_goal.append(float(sampled["goal_rate"]))
    all_exploit = [float(train_exact["exploitability"]), *held_exact]
    return {
        "method": method,
        "seed": seed,
        "train_return": float(result.get("episode_return", _history_return(result))),
        "train_exploitability": float(train_exact["exploitability"]),
        "heldout_exploitability": float(np.mean(held_exact) if held_exact else 0.0),
        "heldout_exploitability_by_layout": held_by_layout,
        "exploitability": float(np.mean(all_exploit)),
        "exact_value_vs_br": float(train_exact["exact_value_vs_br"]),
        "exact_nash_value": float(train_exact["exact_nash_value"]),
        "heldout_goal_rate": float(np.mean(held_goal) if held_goal else 0.0),
        "final_selected_label": result.get("final_selected_label"),
        "strategy_entropy": float(result.get("strategy_entropy", 0.0)),
        "env_steps": int(result.get("env_steps", result.get("total_steps", 0) or 0)),
    }


def _attacker_from_result(method: str, result: dict[str, Any]):
    if method == "ppo_baseline" and result.get("_policy") is not None:
        return policy_from_actor_critic(result["_policy"])
    if method == "psro_lite" and result.get("attacker_mix"):
        return policy_from_mixture({str(key): float(value) for key, value in result["attacker_mix"].items()})
    label = str(result.get("final_selected_label") or "direct_goal")
    strategy = _selected_strategy(result, label)
    eval_label = None if label == "latent" else label
    return policy_from_strategy(strategy, label=eval_label)


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


def _beats_lower(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None, metric: str) -> bool:
    if candidate is None or baseline is None:
        return False
    cand = candidate[metric]
    base = baseline[metric]
    return bool(cand["mean"] < base["mean"] and cand["ci_high"] < base["ci_low"])


def _write(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.json"
    summary_path = output_dir / "summary.json"
    rows_path.write_text(json.dumps(to_jsonable(rows), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    return {"rows_json": str(rows_path), "summary_json": str(summary_path)}
