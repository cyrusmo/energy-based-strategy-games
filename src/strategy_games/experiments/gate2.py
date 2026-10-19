"""Gate 2: does the EBM mixture have lower exact exploitability than PPO and diversity-only?"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from strategy_games.envs.gridworld import ACTION_TO_DELTA
from strategy_games.envs.gridworld_v2 import make_gridworld
from strategy_games.envs.layouts_v2 import (
    HELD_OUT_LAYOUT_V2_NAMES,
    TRAIN_LAYOUT_V2_NAME,
    get_layout_v2,
)
from strategy_games.envs.pathfinding import is_free
from strategy_games.evaluation.best_response import GameTheoreticEvaluator, defender_heuristic_action
from strategy_games.evaluation.exact import (
    action_probs_table,
    exact_metrics_for_policy,
    mixture_table,
    nash_attacker_table,
    policy_from_actor_critic,
    policy_from_mixture,
    policy_from_strategy,
    stochastic_policy_from_actor_critic,
    uniform_random_table,
)
from strategy_games.experiments.convergence import multiseed_confidence
from strategy_games.experiments.diversity_baseline import run_diversity_baseline
from strategy_games.experiments.logging import to_jsonable
from strategy_games.strategies.embeddings import available_heuristic_strategies
from strategy_games.strategies.sampler import langevin_sample
from strategy_games.training.ppo_baseline import PPOConfig, train_ppo_baseline
from strategy_games.training.psro import PSROConfig, run_psro
from strategy_games.training.train_loop import TrainingConfig, run_training_loop

REFERENCE_METHODS = frozenset({"nash_oracle", "uniform_random"})


@dataclass(frozen=True)
class Gate2Config:
    """5-seed comparison used to decide paper framing."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    strategy_iterations: int = 8
    ppo_total_steps: int = 4096
    psro_iterations: int = 2
    candidate_strategies: int = 16
    mixture_k: int = 16
    mixture_goal_samples: int = 8
    output_dir: Path = Path("outputs/public/gate2")


def run_gate2(config: Gate2Config | None = None) -> dict[str, Any]:
    """Compare mixed attackers on train + held-out maps."""

    config = config or Gate2Config()
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    train = get_layout_v2(TRAIN_LAYOUT_V2_NAME)
    heldout = [(name, get_layout_v2(name)) for name in HELD_OUT_LAYOUT_V2_NAMES[:3]]

    for seed in config.seeds:
        seed_rows, seed_curves = _seed_rows(config, seed, train, heldout)
        rows.extend(seed_rows)
        curves.extend(seed_curves)

    summary = summarize_gate2(rows)
    artifacts = _write(config.output_dir, rows, summary, curves)
    return {"rows": rows, "summary": summary, "curves": curves, "artifacts": artifacts}


def summarize_gate2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/CI by method on exact exploitability (lower is better)."""

    methods = sorted({str(row["method"]) for row in rows})
    by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "method": method,
            "attacker_kind": method_rows[0].get("attacker_kind", "unknown") if method_rows else "unknown",
            "reference": method in REFERENCE_METHODS,
            "exploitability": _ci([float(row["exploitability"]) for row in method_rows]),
            "train_exploitability": _ci([float(row["train_exploitability"]) for row in method_rows]),
            "heldout_exploitability": _ci([float(row["heldout_exploitability"]) for row in method_rows]),
            "exact_value_vs_br": _ci([float(row["exact_value_vs_br"]) for row in method_rows]),
            "heldout_goal_rate": _ci([float(row["heldout_goal_rate"]) for row in method_rows]),
            "strategy_entropy": _ci([float(row.get("strategy_entropy", 0.0)) for row in method_rows]),
            "mixture_support": _ci([float(row.get("mixture_support", 0.0)) for row in method_rows]),
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
        "schema_version": "gate2/v3",
        "methods": list(by_method.values()),
        "beats_ppo": beats_ppo,
        "beats_diversity_only": beats_div,
        "passed": passed,
        "framing": "positive" if passed else "diagnostic",
        "notes": [
            "Primary metric is exact Nash exploitability of a Markov behavioral mixture (lower is better).",
            "Pass requires the strategy-loop 95% CI to lie entirely below both PPO and diversity-only.",
            "nash_oracle and uniform_random are reference rows and are excluded from the pass test.",
        ],
    }


def _seed_rows(
    config: Gate2Config,
    seed: int,
    train: Any,
    heldout: list[tuple[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache: dict[tuple[int, str], np.ndarray] = {}
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

    ebm_mix = _ebm_mixture(strategy, train, k=config.mixture_k, cache=cache)
    buffer_mix = _buffer_mixture(strategy, train, cache=cache)
    diversity_mix = _buffer_mixture(diversity, train, cache=cache)
    ppo_stoch = _policy_table(train, stochastic_policy_from_actor_critic(ppo["_policy"]), cache, "ppo_stoch")
    ppo_argmax = _policy_table(train, policy_from_actor_critic(ppo["_policy"]), cache, "ppo_argmax")
    psro_table = _policy_table(
        train,
        policy_from_mixture({str(key): float(value) for key, value in psro["attacker_mix"].items()}),
        cache,
        "psro",
    )
    nash_table = nash_attacker_table(train)
    random_table = uniform_random_table(train)

    rows = [
        _row_from_table(
            "strategy_loop",
            seed,
            strategy,
            train,
            heldout,
            ebm_mix["table"],
            attacker_kind="ebm_mixture",
            mixture_support=ebm_mix["support"],
            components=ebm_mix["components"],
            weights=ebm_mix["weights"],
            goal_samples=config.mixture_goal_samples,
            cache=cache,
        ),
        _row_from_table(
            "strategy_loop_buffer",
            seed,
            strategy,
            train,
            heldout,
            buffer_mix["table"],
            attacker_kind="buffer_mixture",
            mixture_support=buffer_mix["support"],
            components=buffer_mix["components"],
            weights=buffer_mix["weights"],
            goal_samples=config.mixture_goal_samples,
            cache=cache,
        ),
        _row_from_table(
            "diversity_only",
            seed,
            diversity,
            train,
            heldout,
            diversity_mix["table"],
            attacker_kind="buffer_mixture",
            mixture_support=diversity_mix["support"],
            components=diversity_mix["components"],
            weights=diversity_mix["weights"],
            goal_samples=config.mixture_goal_samples,
            cache=cache,
        ),
        _row_from_table(
            "ppo_baseline",
            seed,
            ppo,
            train,
            heldout,
            ppo_stoch,
            attacker_kind="ppo_softmax",
            cache=cache,
        ),
        _row_from_table(
            "ppo_baseline_argmax",
            seed,
            ppo,
            train,
            heldout,
            ppo_argmax,
            attacker_kind="ppo_argmax",
            cache=cache,
        ),
        _row_from_table(
            "psro_lite",
            seed,
            psro,
            train,
            heldout,
            psro_table,
            attacker_kind="label_mixture",
            mixture_support=_mixture_support(list(psro["attacker_mix"].values())),
            cache=cache,
        ),
        _row_from_table(
            "nash_oracle",
            seed,
            {"env_steps": 0, "strategy_entropy": 0.0, "final_selected_label": "nash"},
            train,
            heldout,
            nash_table,
            attacker_kind="nash_oracle",
            mixture_support=1.0,
            cache=cache,
        ),
        _row_from_table(
            "uniform_random",
            seed,
            {"env_steps": 0, "strategy_entropy": 0.0, "final_selected_label": "uniform"},
            train,
            heldout,
            random_table,
            attacker_kind="uniform_random",
            mixture_support=5.0,
            cache=cache,
        ),
    ]
    curves = _buffer_curves(strategy, train, seed, cache)
    return rows, curves


def _row_from_table(
    method: str,
    seed: int,
    result: dict[str, Any],
    train: Any,
    heldout: list[tuple[str, Any]],
    table: np.ndarray,
    attacker_kind: str,
    mixture_support: float = 1.0,
    components: list[dict[str, Any]] | None = None,
    weights: np.ndarray | None = None,
    goal_samples: int = 8,
    cache: dict[tuple[int, str], np.ndarray] | None = None,
) -> dict[str, Any]:
    train_exact = exact_metrics_for_policy(train, table=table)
    held_exact = []
    held_goal = []
    held_by_layout: dict[str, float] = {}
    for name, env in heldout:
        held_table = _heldout_table(method, result, env, components, weights, cache)
        metrics = exact_metrics_for_policy(env, table=held_table)
        held_exact.append(float(metrics["exploitability"]))
        held_by_layout[name] = float(metrics["exploitability"])
        if components and weights is not None:
            held_goal.append(_mixture_goal_rate(env, components, weights, n_samples=goal_samples, seed=seed))
        else:
            held_goal.append(_table_goal_rate(env, held_table, seed=seed))
    all_exploit = [float(train_exact["exploitability"]), *held_exact]
    return {
        "method": method,
        "seed": seed,
        "attacker_kind": attacker_kind,
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
        "mixture_support": float(mixture_support),
        "env_steps": int(result.get("env_steps", result.get("total_steps", 0) or 0)),
    }


def _heldout_table(
    method: str,
    result: dict[str, Any],
    env: Any,
    components: list[dict[str, Any]] | None,
    weights: np.ndarray | None,
    cache: dict[tuple[int, str], np.ndarray] | None,
) -> np.ndarray:
    if method == "nash_oracle":
        return nash_attacker_table(env)
    if method == "uniform_random":
        return uniform_random_table(env)
    if components and weights is not None:
        tables = [_component_table(env, item, cache) for item in components]
        return mixture_table(tables, weights)
    if method == "ppo_baseline" and result.get("_policy") is not None:
        return _policy_table(env, stochastic_policy_from_actor_critic(result["_policy"]), cache, "ppo_stoch")
    if method == "ppo_baseline_argmax" and result.get("_policy") is not None:
        return _policy_table(env, policy_from_actor_critic(result["_policy"]), cache, "ppo_argmax")
    if method == "psro_lite" and result.get("attacker_mix"):
        mix = {str(key): float(value) for key, value in result["attacker_mix"].items()}
        return _policy_table(env, policy_from_mixture(mix), cache, "psro")
    return uniform_random_table(env)


def _ebm_mixture(
    result: dict[str, Any],
    config: Any,
    k: int,
    cache: dict[tuple[int, str], np.ndarray],
) -> dict[str, Any]:
    ebm = result.get("_ebm")
    if ebm is None:
        return _buffer_mixture(result, config, cache=cache)
    device = next(ebm.parameters()).device
    samples = langevin_sample(
        ebm,
        num_samples=k,
        strategy_dim=samples_dim(ebm),
        steps=10,
        step_size=0.02,
        noise_scale=0.25,
        device=device,
    )
    energies = ebm(samples).detach().cpu().numpy()
    weights = _softmax(-energies)
    components = []
    tables = []
    for sample in samples:
        embedding = sample.detach().cpu()
        components.append({"embedding": embedding, "label": None, "score": 0.0, "iteration": -1})
        tables.append(_component_table(config, components[-1], cache))
    return {
        "table": mixture_table(tables, weights),
        "weights": weights,
        "components": components,
        "support": _mixture_support(weights),
    }


def samples_dim(ebm: Any) -> int:
    first = next(ebm.parameters())
    # EnergyMLP first layer is Linear(strategy_dim, hidden)
    return int(first.shape[1]) if first.ndim == 2 else 8


def _buffer_mixture(
    result: dict[str, Any],
    config: Any,
    cache: dict[tuple[int, str], np.ndarray],
    prefix_iter: int | None = None,
    tau: float = 1.0,
) -> dict[str, Any]:
    records = list(result.get("_buffer_records") or [])
    if prefix_iter is not None:
        records = [item for item in records if int(item.get("iteration", 0)) <= prefix_iter]
    if not records:
        table = uniform_random_table(config)
        return {"table": table, "weights": np.array([1.0]), "components": [], "support": 1.0}
    scores = [float(item.get("score", 0.0)) for item in records]
    weights = _softmax(scores, tau=tau)
    tables = [_component_table(config, item, cache) for item in records]
    return {
        "table": mixture_table(tables, weights),
        "weights": weights,
        "components": records,
        "support": _mixture_support(weights),
    }


def _buffer_curves(
    result: dict[str, Any],
    train: Any,
    seed: int,
    cache: dict[tuple[int, str], np.ndarray],
) -> list[dict[str, Any]]:
    history = result.get("history", [])
    if not isinstance(history, list) or not history:
        return []
    total_steps = float(result.get("env_steps", 0) or 0)
    n_iter = max(1, len(history))
    curves: list[dict[str, Any]] = []
    for item in history:
        iteration = int(item.get("iteration", 0))
        mix = _buffer_mixture(result, train, cache=cache, prefix_iter=iteration)
        metrics = exact_metrics_for_policy(train, table=mix["table"])
        curves.append(
            {
                "method": "strategy_loop_buffer",
                "seed": seed,
                "iteration": iteration,
                "env_steps": float(total_steps * (iteration + 1) / n_iter),
                "exploitability": float(metrics["exploitability"]),
            }
        )
    return curves


def _component_table(
    config: Any,
    component: dict[str, Any],
    cache: dict[tuple[int, str], np.ndarray] | None,
) -> np.ndarray:
    embedding = component["embedding"]
    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu()
        key_vec = tuple(round(float(x), 5) for x in embedding.tolist())
    else:
        embedding = torch.as_tensor(embedding, dtype=torch.float32)
        key_vec = tuple(round(float(x), 5) for x in embedding.tolist())
    label = component.get("label")
    eval_label = None if label in {None, "latent"} else str(label)
    cache_key = (id(config), f"{eval_label}:{key_vec}")
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    table = action_probs_table(config, policy_from_strategy(embedding, label=eval_label))
    if cache is not None:
        cache[cache_key] = table
    return table


def _policy_table(
    config: Any,
    policy: Any,
    cache: dict[tuple[int, str], np.ndarray] | None,
    name: str,
) -> np.ndarray:
    cache_key = (id(config), name)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    table = action_probs_table(config, policy)
    if cache is not None:
        cache[cache_key] = table
    return table


def _mixture_goal_rate(
    env: Any,
    components: list[dict[str, Any]],
    weights: np.ndarray,
    n_samples: int = 8,
    seed: int = 0,
) -> float:
    if not components:
        return 0.0
    probs = np.asarray(weights, dtype=np.float64).reshape(-1)
    probs = np.clip(probs, 0.0, None)
    if float(probs.sum()) <= 0.0:
        probs = np.ones(len(components), dtype=np.float64) / len(components)
    else:
        probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    n_draw = min(n_samples, len(components))
    indices = rng.choice(len(components), size=n_draw, replace=True, p=probs)
    evaluator = GameTheoreticEvaluator(env_config=env, episodes_per_opponent=1, strategy_dim=8)
    rates = []
    for idx in indices:
        item = components[int(idx)]
        embedding = item["embedding"]
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().float()
        else:
            embedding = torch.as_tensor(embedding, dtype=torch.float32)
        label = item.get("label")
        eval_label = None if label in {None, "latent"} else str(label)
        metrics = evaluator.evaluate_strategy(embedding, label=eval_label)
        rates.append(float(metrics["goal_rate"]))
    return float(np.mean(rates)) if rates else 0.0


def _table_goal_rate(config: Any, table: np.ndarray, seed: int = 0, episodes_per_opponent: int = 1) -> float:
    cells = [
        (y, x)
        for y in range(config.grid_size)
        for x in range(config.grid_size)
        if is_free((y, x), config.grid_size, config.obstacles)
    ]
    index = {cell: i for i, cell in enumerate(cells)}
    actions = tuple(ACTION_TO_DELTA)
    rng = np.random.default_rng(seed)
    outcomes: list[str] = []
    for opponent in available_heuristic_strategies():
        for offset in range(episodes_per_opponent):
            env = make_gridworld(config)
            env.reset(seed=int(seed * 17 + offset))
            done = False
            info: dict[str, object] = {"outcome": "running"}
            while not done:
                ia = index[env.attacker_pos]
                id_ = index[env.defender_pos]
                probs = np.asarray(table[env.steps, ia, id_], dtype=np.float64)
                probs = np.clip(probs, 0.0, None)
                if float(probs.sum()) <= 0.0:
                    probs = np.ones(len(actions), dtype=np.float64) / len(actions)
                else:
                    probs = probs / probs.sum()
                attacker_action = int(rng.choice(len(actions), p=probs))
                defender_action = defender_heuristic_action(env, opponent)
                result = env.step(actions[attacker_action], defender_action)
                done = result.done
                info = result.info
            outcomes.append(str(info.get("outcome", "running")))
    return float(outcomes.count("goal") / len(outcomes)) if outcomes else 0.0


def _softmax(values: list[float] | np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1) / max(float(tau), 1e-8)
    x = x - float(x.max())
    exp = np.exp(x)
    return exp / exp.sum()


def _mixture_support(weights: list[float] | np.ndarray) -> float:
    probs = np.asarray(weights, dtype=np.float64).reshape(-1)
    probs = np.clip(probs, 0.0, None)
    if float(probs.sum()) <= 0.0:
        return 0.0
    probs = probs / probs.sum()
    probs = probs[probs > 1e-12]
    entropy = float(-np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


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


def _write(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    curves: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.json"
    summary_path = output_dir / "summary.json"
    curves_path = output_dir / "curves.json"
    rows_path.write_text(json.dumps(to_jsonable(rows), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    curves_path.write_text(json.dumps(to_jsonable(curves), indent=2) + "\n", encoding="utf-8")
    return {"rows_json": str(rows_path), "summary_json": str(summary_path), "curves_json": str(curves_path)}
