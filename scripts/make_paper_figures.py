"""Generate paper-facing figures from public experiment APIs/artifacts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path("outputs/public/.matplotlib").resolve()
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_XDG_CACHE_HOME = Path("outputs/public/.cache").resolve()
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from strategy_games.experiments.baselines import compare_baselines, save_baseline_metrics
from strategy_games.experiments.payoff import compute_payoff_matrix, save_payoff_matrix
from strategy_games.strategies.embeddings import available_heuristic_strategies, named_strategy_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate public paper figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/public/paper/figures"))
    parser.add_argument("--baseline-json", type=Path, default=Path("outputs/public/baselines/metrics.json"))
    parser.add_argument("--payoff-json", type=Path, default=Path("outputs/public/payoff_matrix/matrix.json"))
    parser.add_argument("--ablation-json", type=Path, default=Path("outputs/public/ablations/summary.json"))
    parser.add_argument("--no-compute", action="store_true", help="Use existing JSON only; do not run experiments.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_architecture(args.output_dir / "architecture_loop.png")
    draw_baseline_bars(args.baseline_json, args.output_dir / "baseline_comparison.png", compute=not args.no_compute)
    draw_payoff_heatmap(args.payoff_json, args.output_dir / "payoff_matrix.png", compute=not args.no_compute)
    draw_embedding_projection(args.output_dir / "strategy_embedding_projection.png")
    draw_ablation_summary(args.ablation_json, args.output_dir / "ablation_summary.png", compute=not args.no_compute)
    write_manifest(
        args.output_dir,
        source_artifacts=[args.baseline_json, args.payoff_json, args.ablation_json],
        command=["python", "scripts/make_paper_figures.py", *sys.argv[1:]],
        is_smoke=args.no_compute,
    )


def draw_architecture(path: Path) -> None:
    """Draw a compact Generate-Evaluate-Execute-Update diagram."""

    labels = [
        "Strategy\nBuffer",
        "EnergyMLP\nGenerator",
        "Langevin\nCandidates",
        "Sampled-Response\nEvaluator",
        "Strategy-Conditioned\nPolicy",
        "Gridworld\nRollout",
        "Policy / EBM /\nWorld Model Updates",
    ]
    positions = np.array(
        [
            [0.05, 0.55],
            [0.20, 0.55],
            [0.36, 0.55],
            [0.53, 0.55],
            [0.72, 0.55],
            [0.89, 0.55],
            [0.53, 0.18],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    for label, (x_pos, y_pos) in zip(labels, positions):
        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black"},
            fontsize=9,
            transform=ax.transAxes,
        )
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (5, 0)]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=positions[end],
            xytext=positions[start],
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.3},
        )
    ax.set_title("Generate -> Evaluate -> Execute -> Update", fontsize=12)
    save_figure(fig, path)


def draw_baseline_bars(json_path: Path, figure_path: Path, compute: bool) -> None:
    rows = load_json(json_path)
    if rows is None and compute:
        rows = compare_baselines()
        save_baseline_metrics(rows, json_path)
    if not isinstance(rows, list):
        raise FileNotFoundError(f"Missing baseline metrics: {json_path}")

    names = [str(row["baseline"]) for row in rows]
    returns = [float(row["episode_return"]) for row in rows]
    wins = [float(row["win_rate"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    axes[0].bar(names, returns)
    axes[0].set_title("Episode return")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].bar(names, wins)
    axes[1].set_title("Win rate")
    axes[1].set_ylim(0.0, 1.05)
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=30)
        axis.set_ylabel("Metric value")
    fig.suptitle("Gridworld baseline diagnostics")
    fig.tight_layout()
    save_figure(fig, figure_path)


def draw_payoff_heatmap(json_path: Path, figure_path: Path, compute: bool) -> None:
    matrix_result = load_json(json_path)
    if matrix_result is None and compute:
        matrix_result = compute_payoff_matrix()
        save_payoff_matrix(matrix_result, json_path)
    if not isinstance(matrix_result, dict):
        raise FileNotFoundError(f"Missing payoff matrix: {json_path}")

    strategies = list(matrix_result["strategy_labels"])
    opponents = list(matrix_result["opponent_labels"])
    matrix = np.asarray(matrix_result["average_reward_matrix"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(opponents)), labels=opponents, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(strategies)), labels=strategies)
    ax.set_xlabel("Opponent response")
    ax.set_ylabel("Candidate strategy")
    ax.set_title("Sampled-response payoff matrix")
    fig.colorbar(image, ax=ax, label="Average attacker reward")
    fig.tight_layout()
    save_figure(fig, figure_path)


def draw_embedding_projection(figure_path: Path) -> None:
    labels = list(available_heuristic_strategies())
    embeddings = torch.stack([named_strategy_embedding(label, 8).vector for label in labels]).numpy()
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(projected[:, 0], projected[:, 1])
    for label, (x_pos, y_pos) in zip(labels, projected):
        ax.annotate(label, (x_pos, y_pos), textcoords="offset points", xytext=(4, 4), fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Named strategy embedding projection")
    fig.tight_layout()
    save_figure(fig, figure_path)


def draw_ablation_summary(json_path: Path, figure_path: Path, compute: bool = True) -> None:
    summary = load_json(json_path)
    if not isinstance(summary, dict) or not isinstance(summary.get("runs"), list):
        if not compute:
            raise FileNotFoundError(f"Missing ablation summary: {json_path}")
        summary = {
            "runs": [
                {"ablation": "ebm_langevin", "mean_episode_return": 0.0},
                {"ablation": "gaussian_sampler", "mean_episode_return": 0.0},
                {"ablation": "no_buffer_positives", "mean_episode_return": 0.0},
                {"ablation": "average_value_selection", "mean_episode_return": 0.0},
                {"ablation": "no_world_model", "mean_episode_return": 0.0},
            ],
            "placeholder": True,
        }
    runs = list(summary["runs"])
    names = [str(run["ablation"]) for run in runs]
    returns = [float(run.get("mean_episode_return", 0.0)) for run in runs]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, returns)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean episode return")
    ax.set_title("Ablation diagnostics")
    ax.tick_params(axis="x", labelrotation=30)
    if summary.get("placeholder"):
        ax.text(0.5, 0.9, "Placeholder until ablation suite is run", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    save_figure(fig, figure_path)


def write_manifest(
    output_dir: Path,
    source_artifacts: list[Path] | None = None,
    command: list[str] | None = None,
    is_smoke: bool = False,
) -> None:
    """Write provenance metadata for generated paper figures."""

    source_artifacts = source_artifacts or []
    manifest = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git_commit": git_commit(),
        "commands": [" ".join(command)] if command else [],
        "source_artifacts": [str(path) for path in source_artifacts],
        "is_smoke": bool(is_smoke),
        "figures": [
            "architecture_loop.png",
            "baseline_comparison.png",
            "payoff_matrix.png",
            "strategy_embedding_projection.png",
            "ablation_summary.png",
        ],
        "note": "Regenerate after running examples/run_ablation_suite.py for final ablation values.",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def git_commit() -> str | None:
    """Return the short git commit when available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
