"""Energy-guided weight-space repair pilot on Gridworld v2.

A strategy selector maps a position (layout, attacker cell, defender cell) to
a mix over the named attacker strategies. A fault is planted in its weights,
and a fixed sparse coordinate search edits the weights back using only a
guidance energy over (position, strategy) pairs:

- high decision energy flags the positions to repair;
- the search keeps energy from rising on the unflagged (low-energy) positions.

Returns are exact: every strategy is rolled out to termination from every
position against each scripted defender, so the value of a selector at a
position is a dot product with a precomputed table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from strategy_games.envs.gridworld_v2 import AttackerDefenderGridworldV2, GridworldV2Config
from strategy_games.envs.layouts_v2 import get_layout_v2, layout_v2_names
from strategy_games.envs.pathfinding import is_free
from strategy_games.evaluation.best_response import attacker_v2_heuristic_action, defender_v2_heuristic_action
from strategy_games.models.ebm import EnergyMLP
from strategy_games.strategies.embeddings import available_heuristic_strategies, named_strategy_embedding

STRATEGIES: tuple[str, ...] = available_heuristic_strategies()
SCHEMA_VERSION = "energy_repair/v1"
METHODS: tuple[str, ...] = ("energy_guided", "repair_only", "position_blind", "oracle_regret")
FAULT_SCOPES: tuple[str, ...] = ("blanket", "selective")


@dataclass(frozen=True)
class RepairConfig:
    """Fixed protocol. Nothing here is tuned per seed, fault, or method."""

    hidden_dim: int = 48
    strategy_dim: int = 8
    selector_temperature: float = 0.03
    energy_temperature: float = 0.1
    selector_steps: int = 4000
    energy_steps: int = 1500
    fault_steps: int = 400
    fault_keep_weight: float = 4.0
    fit_frac: float = 0.7
    evidence_frac: float = 0.1
    material_regret: float = 0.10
    flag_frac: float = 0.25
    search_steps: int = 40
    coords_per_step: int = 32
    preserve_weight: float = 3.0
    step_grid: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)


@dataclass(frozen=True)
class PositionTable:
    """All evaluated positions with exact per-strategy returns."""

    layout_names: tuple[str, ...]
    layout_ids: np.ndarray  # [N]
    features: np.ndarray  # [N, F]
    returns: np.ndarray  # [N, K], mean over scripted defenders


# ---------------------------------------------------------------------------
# Exact per-position returns
# ---------------------------------------------------------------------------


def commit_return(
    config: GridworldV2Config,
    attacker: tuple[int, int],
    defender: tuple[int, int],
    strategy: str,
    defender_label: str,
) -> float:
    """Return of committing to ``strategy`` from a position until termination."""

    env = AttackerDefenderGridworldV2(config)
    env.reset(attacker_pos=attacker, defender_pos=defender)
    total = 0.0
    while True:
        result = env.step(
            attacker_v2_heuristic_action(env, strategy),
            defender_v2_heuristic_action(env, defender_label),
        )
        total += result.reward
        if result.done:
            return total


def layout_positions(
    layout_name: str,
    layout_index: int,
    num_layouts: int,
    max_positions: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, returns) for the non-terminal positions of one layout."""

    config = get_layout_v2(layout_name)
    size = config.grid_size
    cells = [(y, x) for y in range(size) for x in range(size) if is_free((y, x), size, config.obstacles)]
    positions = [(a, d) for a in cells for d in cells if a != d and a not in config.goal_positions]
    if max_positions is not None and max_positions < len(positions):
        keep = np.random.default_rng(seed).choice(len(positions), max_positions, replace=False)
        positions = [positions[i] for i in sorted(keep)]
    env = AttackerDefenderGridworldV2(config)
    one_hot = np.zeros(num_layouts, dtype=np.float32)
    one_hot[layout_index] = 1.0
    features, returns = [], []
    for attacker, defender in positions:
        observation = env.reset(attacker_pos=attacker, defender_pos=defender)
        features.append(np.concatenate([observation, one_hot]))
        returns.append(
            [
                float(np.mean([commit_return(config, attacker, defender, s, d) for d in STRATEGIES]))
                for s in STRATEGIES
            ]
        )
    return np.asarray(features, dtype=np.float32), np.asarray(returns, dtype=np.float32)


def _layout_positions_job(args: tuple[str, int, int, int | None, int]) -> tuple[np.ndarray, np.ndarray]:
    return layout_positions(*args)


def build_position_table(
    layout_names: tuple[str, ...] | None = None,
    max_positions: int | None = None,
    seed: int = 0,
    workers: int = 1,
) -> PositionTable:
    """Roll out every strategy from every position on the given layouts."""

    names = tuple(layout_names or layout_v2_names())
    jobs = [(name, i, len(names), max_positions, seed) for i, name in enumerate(names)]
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(_layout_positions_job, jobs)
    else:
        parts = [_layout_positions_job(job) for job in jobs]
    return PositionTable(
        layout_names=names,
        layout_ids=np.concatenate([np.full(len(f), i, dtype=np.int64) for i, (f, _) in enumerate(parts)]),
        features=np.concatenate([f for f, _ in parts]),
        returns=np.concatenate([r for _, r in parts]),
    )


def save_position_table(table: PositionTable, path: str) -> None:
    np.savez_compressed(
        path,
        layout_names=np.asarray(table.layout_names),
        layout_ids=table.layout_ids,
        features=table.features,
        returns=table.returns,
    )


def load_position_table(path: str) -> PositionTable:
    raw = np.load(path)
    return PositionTable(
        layout_names=tuple(str(n) for n in raw["layout_names"]),
        layout_ids=raw["layout_ids"],
        features=raw["features"],
        returns=raw["returns"],
    )


# ---------------------------------------------------------------------------
# Selector (flat weight vector) and state-conditioned energy
# ---------------------------------------------------------------------------


def selector_shapes(in_dim: int, hidden: int, out_dim: int) -> list[tuple[int, ...]]:
    return [(hidden, in_dim), (hidden,), (hidden, hidden), (hidden,), (out_dim, hidden), (out_dim,)]


def init_selector(shapes: list[tuple[int, ...]], generator: torch.Generator) -> Tensor:
    """Flat weight vector with the usual fan-in uniform initialisation."""

    chunks = []
    for i in range(0, len(shapes), 2):
        bound = 1.0 / float(shapes[i][1]) ** 0.5
        for shape in shapes[i : i + 2]:
            chunks.append(((torch.rand(shape, generator=generator) * 2 - 1) * bound).reshape(-1))
    return torch.cat(chunks)


def selector_logits(theta: Tensor, x: Tensor, shapes: list[tuple[int, ...]]) -> Tensor:
    """Two-hidden-layer tanh MLP evaluated from a flat weight vector."""

    params, offset = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        params.append(theta[offset : offset + size].reshape(shape))
        offset += size
    h = torch.tanh(x @ params[0].T + params[1])
    h = torch.tanh(h @ params[2].T + params[3])
    return h @ params[4].T + params[5]


def selector_probs(theta: Tensor, x: Tensor, shapes: list[tuple[int, ...]]) -> Tensor:
    return torch.softmax(selector_logits(theta, x, shapes), dim=-1)


def _adam(theta: Tensor, loss_fn, steps: int, lr: float) -> Tensor:
    theta = theta.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss_fn(theta).backward()
        optimizer.step()
    return theta.detach()


def soft_targets(returns: Tensor, temperature: float) -> Tensor:
    return torch.softmax(returns / temperature, dim=-1)


def train_selector(x: Tensor, returns: Tensor, shapes: list[tuple[int, ...]], cfg: RepairConfig, seed: int) -> Tensor:
    """Distil the soft-optimal strategy choice into selector weights."""

    target = soft_targets(returns, cfg.selector_temperature)
    theta = init_selector(shapes, torch.Generator().manual_seed(seed))

    def loss(theta: Tensor) -> Tensor:
        return -(target * torch.log_softmax(selector_logits(theta, x, shapes), dim=-1)).sum(-1).mean()

    return _adam(theta, loss, cfg.selector_steps, lr=3e-3)


def energy_table(model: EnergyMLP, x: Tensor, strategy_dim: int) -> Tensor:
    """Excess energy ``E(s, k) - min_k E(s, k)`` for every (position, strategy).

    Zero for the strategy the model likes best at a position, so equally good
    strategies are not charged for sharing probability mass.
    """

    embeddings = torch.stack([named_strategy_embedding(s, strategy_dim).vector for s in STRATEGIES])
    n, k = x.shape[0], embeddings.shape[0]
    pairs = torch.cat([x.unsqueeze(1).expand(n, k, -1), embeddings.unsqueeze(0).expand(n, k, -1)], dim=-1)
    raw = model(pairs.reshape(n * k, -1)).reshape(n, k)
    return raw - raw.min(dim=-1, keepdim=True).values


def train_energy(x: Tensor, returns: Tensor, cfg: RepairConfig, seed: int) -> EnergyMLP:
    """Fit E(position, strategy) so low energy means a high-return strategy."""

    torch.manual_seed(seed)
    model = EnergyMLP(strategy_dim=x.shape[1] + cfg.strategy_dim, hidden_dim=64, num_layers=2)
    target = soft_targets(returns, cfg.energy_temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(cfg.energy_steps):
        optimizer.zero_grad()
        excess = energy_table(model, x, cfg.strategy_dim)
        log_probs = torch.log_softmax(-excess, dim=-1)
        (-(target * log_probs).sum(-1).mean()).backward()
        optimizer.step()
    return model


# ---------------------------------------------------------------------------
# Fault planting
# ---------------------------------------------------------------------------


def plant_fault(
    theta: Tensor,
    x: Tensor,
    fault_mask: Tensor,
    strategy_index: int,
    shapes: list[tuple[int, ...]],
    cfg: RepairConfig,
) -> Tensor:
    """Fine-tune all weights to overuse one strategy on ``fault_mask`` positions.

    Behaviour on the remaining positions is held near the clean policy, so the
    damage is confined in behaviour but spread across the weights.
    """

    clean = selector_probs(theta, x, shapes).detach()
    keep = ~fault_mask

    def loss(theta: Tensor) -> Tensor:
        log_probs = torch.log_softmax(selector_logits(theta, x, shapes), dim=-1)
        fault = -log_probs[fault_mask, strategy_index].mean()
        drift = (clean[keep] * (torch.log(clean[keep] + 1e-9) - log_probs[keep])).sum(-1).mean()
        return fault + cfg.fault_keep_weight * drift

    return _adam(theta, loss, cfg.fault_steps, lr=1e-3)


# ---------------------------------------------------------------------------
# Fixed sparse coordinate search
# ---------------------------------------------------------------------------


def decision_energy(theta: Tensor, x: Tensor, guidance: Tensor, shapes: list[tuple[int, ...]]) -> Tensor:
    """Expected guidance energy of the selector's decision at each position."""

    return (selector_probs(theta, x, shapes) * guidance).sum(-1)


def random_rotation(dim: int, seed: int) -> Tensor:
    """Random orthogonal change of weight coordinates."""

    gaussian = torch.randn(dim, dim, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    return (q * torch.sign(torch.diagonal(r))).float()


def repair_search(
    theta0: Tensor,
    x: Tensor,
    guidance: Tensor,
    shapes: list[tuple[int, ...]],
    cfg: RepairConfig,
    basis: Tensor | None = None,
    preserve: bool = True,
) -> list[Tensor]:
    """Greedy sparse coordinate descent on decision energy. Returns the path.

    ``basis`` columns are the editable coordinate directions (identity when
    None). Each step edits ``coords_per_step`` coordinates. With ``preserve``
    the coordinates are ranked by repair gain per unit sensitivity of the
    unflagged positions, and the step is charged (quadratically) for any energy
    it adds there.
    """

    energy0 = decision_energy(theta0, x, guidance, shapes).detach()
    flagged = torch.zeros_like(energy0, dtype=torch.bool)
    flagged[torch.topk(energy0, max(1, int(cfg.flag_frac * energy0.numel()))).indices] = True
    calm = ~flagged

    def to_coords(vector: Tensor) -> Tensor:
        return vector if basis is None else basis.T @ vector

    def objective(theta: Tensor) -> Tensor:
        energy = decision_energy(theta, x, guidance, shapes)
        value = energy[flagged].mean()
        if preserve and bool(calm.any()):
            value = value + cfg.preserve_weight * torch.relu(energy[calm] - energy0[calm]).square().mean()
        return value

    sensitivity = None
    if preserve and bool(calm.any()):
        per_position = torch.func.vmap(
            torch.func.grad(lambda t, xi, gi: decision_energy(t, xi[None], gi[None], shapes)[0]), in_dims=(None, 0, 0)
        )(theta0, x[calm], guidance[calm])
        if basis is not None:
            per_position = per_position @ basis
        sensitivity = per_position.square().mean(0)
        sensitivity = sensitivity + 1e-3 * sensitivity.mean() + 1e-12

    theta = theta0.clone()
    path = [theta.clone()]
    for _ in range(cfg.search_steps):
        probe = theta.clone().requires_grad_(True)
        value = objective(probe)
        grad = to_coords(torch.autograd.grad(value, probe)[0])
        score = grad.square() if sensitivity is None else grad.square() / sensitivity
        chosen = torch.topk(score, min(cfg.coords_per_step, score.numel())).indices
        direction = torch.zeros_like(grad)
        direction[chosen] = -grad[chosen] / (grad[chosen].norm() + 1e-12)
        if basis is not None:
            direction = basis @ direction
        best_theta, best_value = theta, float(value.detach())
        for step in cfg.step_grid:
            candidate = theta + step * direction
            candidate_value = float(objective(candidate))
            if candidate_value < best_value:
                best_theta, best_value = candidate, candidate_value
        theta = best_theta.detach()
        path.append(theta.clone())
    return path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _top_fraction_precision(score: np.ndarray, label: np.ndarray, frac: float) -> float:
    top = np.argsort(-score, kind="stable")[: max(1, int(frac * score.size))]
    return float(label[top].mean())


def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def path_metrics(
    path: list[Tensor],
    x_eval: Tensor,
    returns_eval: Tensor,
    on_fault_layout: Tensor,
    clean_theta: Tensor,
    shapes: list[tuple[int, ...]],
    material: float,
) -> dict[str, float]:
    """Recovery on the faulted layout and collateral harm on the others."""

    def value(theta: Tensor) -> Tensor:
        return (selector_probs(theta, x_eval, shapes) * returns_eval).sum(-1)

    damaged, clean = value(path[0]), value(clean_theta)
    other = ~on_fault_layout
    lost = float((clean[on_fault_layout] - damaged[on_fault_layout]).mean())
    harms = [float(((damaged[other] - value(theta)[other]) > material).float().mean()) for theta in path]
    final = value(path[-1])
    gained = float((final[on_fault_layout] - damaged[on_fault_layout]).mean())
    return {
        "lost_return": lost,
        "gained_return": gained,
        "recovery": gained / lost if lost > 1e-6 else float("nan"),
        "peak_harm": max(harms),
        "final_harm": harms[-1],
        "other_layout_return_change": float((final[other] - damaged[other]).mean()),
    }


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


def run_seed(
    table: PositionTable,
    seed: int,
    cfg: RepairConfig | None = None,
    fault_strategies: tuple[str, ...] = ("aggressive", "feint"),
    fault_scopes: tuple[str, ...] = FAULT_SCOPES,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, Any]:
    """Train selector and energy, plant each fault, run every repair arm."""

    cfg = cfg or RepairConfig()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = table.features.shape[0]
    order = rng.permutation(n)
    n_fit, n_evidence = int(cfg.fit_frac * n), int(cfg.evidence_frac * n)
    fit, evidence, held_out = order[:n_fit], order[n_fit : n_fit + n_evidence], order[n_fit + n_evidence :]

    x = torch.from_numpy(table.features)
    returns = torch.from_numpy(table.returns)
    layout_ids = torch.from_numpy(table.layout_ids)
    regret = returns.max(-1, keepdim=True).values - returns
    shapes = selector_shapes(x.shape[1], cfg.hidden_dim, len(STRATEGIES))

    clean_theta = train_selector(x[fit], returns[fit], shapes, cfg, seed)
    energy_model = train_energy(x[fit], returns[fit], cfg, seed + 10_000)
    with torch.no_grad():
        learned_energy = energy_table(energy_model, x, cfg.strategy_dim)
    blind_energy = learned_energy[evidence].mean(0, keepdim=True).expand_as(learned_energy)
    guidance = {
        "energy_guided": learned_energy,
        "repair_only": learned_energy,
        "position_blind": blind_energy,
        "oracle_regret": regret,
    }
    rotation = random_rotation(clean_theta.numel(), seed + 20_000)
    fault_layout = seed % len(table.layout_names)
    on_fault_layout = layout_ids == fault_layout

    clean_probs = selector_probs(clean_theta, x[held_out], shapes)
    result: dict[str, Any] = {
        "seed": seed,
        "fault_layout": table.layout_names[fault_layout],
        "num_weights": int(clean_theta.numel()),
        "num_held_out": int(held_out.size),
        "clean_return": float((clean_probs * returns[held_out]).sum(-1).mean()),
        "best_return": float(returns[held_out].max(-1).values.mean()),
        "faults": {},
    }

    for strategy in fault_strategies:
        k = STRATEGIES.index(strategy)
        for scope in fault_scopes:
            fault_mask = on_fault_layout[fit]
            if scope == "selective":
                fault_mask = fault_mask & (regret[fit, k] > cfg.material_regret)
            damaged = plant_fault(clean_theta, x[fit], fault_mask, k, shapes, cfg)

            probs = selector_probs(damaged, x[held_out], shapes)
            decision_regret = (probs * regret[held_out]).sum(-1)
            suboptimal = (decision_regret > cfg.material_regret).numpy()
            learned_score = (probs * learned_energy[held_out]).sum(-1).numpy()
            blind_score = (probs * blind_energy[held_out]).sum(-1).numpy()
            entry: dict[str, Any] = {
                "detection": {
                    "base_rate": float(suboptimal.mean()),
                    "energy_top_quarter": _top_fraction_precision(learned_score, suboptimal, cfg.flag_frac),
                    "blind_top_quarter": _top_fraction_precision(blind_score, suboptimal, cfg.flag_frac),
                    "energy_auroc": _auroc(learned_score, suboptimal),
                    "blind_auroc": _auroc(blind_score, suboptimal),
                },
                "fault_strategy_rate_on_fault_layout": float(probs[on_fault_layout[held_out], k].mean()),
                "arms": {},
            }
            for coords, basis in (("original", None), ("rotated", rotation)):
                for method in methods:
                    path = repair_search(
                        damaged,
                        x[evidence],
                        guidance[method][evidence],
                        shapes,
                        cfg,
                        basis=basis,
                        preserve=method != "repair_only",
                    )
                    entry["arms"][f"{coords}/{method}"] = path_metrics(
                        path,
                        x[held_out],
                        returns[held_out],
                        on_fault_layout[held_out],
                        clean_theta,
                        shapes,
                        cfg.material_regret,
                    )
            result["faults"][f"{strategy}/{scope}"] = entry
    return result


# ---------------------------------------------------------------------------
# Aggregation and report
# ---------------------------------------------------------------------------


def _mean_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "ci95": float("nan")}
    half = 1.96 * float(array.std(ddof=1)) / float(np.sqrt(array.size)) if array.size > 1 else 0.0
    return {"mean": float(array.mean()), "ci95": half}


def aggregate(seed_results: list[dict[str, Any]], cfg: RepairConfig | None = None) -> dict[str, Any]:
    """Summarise per-seed results. Recovery is pooled: total gained / total lost."""

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "num_seeds": len(seed_results),
        "config": asdict(cfg or RepairConfig()),
        "clean_return": _mean_ci([r["clean_return"] for r in seed_results]),
        "best_return": _mean_ci([r["best_return"] for r in seed_results]),
        "faults": {},
    }
    for fault in seed_results[0]["faults"]:
        entries = [r["faults"][fault] for r in seed_results]
        arms = {}
        for arm in entries[0]["arms"]:
            rows = [e["arms"][arm] for e in entries]
            lost = sum(row["lost_return"] for row in rows)
            arms[arm] = {
                "recovery_pooled": sum(row["gained_return"] for row in rows) / lost if lost > 1e-9 else float("nan"),
                "recovery": _mean_ci([row["recovery"] for row in rows]),
                "peak_harm": _mean_ci([row["peak_harm"] for row in rows]),
                "final_harm": _mean_ci([row["final_harm"] for row in rows]),
                "other_layout_return_change": _mean_ci([row["other_layout_return_change"] for row in rows]),
            }
        summary["faults"][fault] = {
            "lost_return": _mean_ci([e["arms"][next(iter(e["arms"]))]["lost_return"] for e in entries]),
            "fault_strategy_rate_on_fault_layout": _mean_ci(
                [e["fault_strategy_rate_on_fault_layout"] for e in entries]
            ),
            "detection": {key: _mean_ci([e["detection"][key] for e in entries]) for key in entries[0]["detection"]},
            "arms": arms,
        }
    return summary


def summary_markdown(summary: dict[str, Any]) -> str:
    """Render the aggregate as markdown tables."""

    def pct(stat: dict[str, float]) -> str:
        return f"{100 * stat['mean']:.2f} ± {100 * stat['ci95']:.2f}"

    lines = [
        f"Seeds: {summary['num_seeds']}. Clean selector return {summary['clean_return']['mean']:.3f}; "
        f"per-position best {summary['best_return']['mean']:.3f}. ± is a 95% CI half-width over seeds.",
        "",
    ]
    for fault, entry in summary["faults"].items():
        det = entry["detection"]
        lines += [
            f"### Fault `{fault}`",
            "",
            f"Return lost on the faulted layout: {entry['lost_return']['mean']:.3f} ± {entry['lost_return']['ci95']:.3f}. "
            f"Materially suboptimal decisions: {pct(det['base_rate'])}% of held-out positions; "
            f"{pct(det['energy_top_quarter'])}% in the top energy quarter "
            f"(position-blind: {pct(det['blind_top_quarter'])}%). "
            f"AUROC {det['energy_auroc']['mean']:.3f} vs {det['blind_auroc']['mean']:.3f}.",
            "",
            "| Coordinates | Method | Recovery % (pooled) | Peak harm % | Final harm % | Other-layout return change |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for arm, stats in entry["arms"].items():
            coords, method = arm.split("/")
            change = stats["other_layout_return_change"]
            lines.append(
                f"| {coords} | `{method}` | {100 * stats['recovery_pooled']:.1f} | {pct(stats['peak_harm'])} | "
                f"{pct(stats['final_harm'])} | {change['mean']:+.4f} ± {change['ci95']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)
