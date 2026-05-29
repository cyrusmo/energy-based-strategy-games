"""Unified Streamlit dashboard for resource, convergence, and quality diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from strategy_games.viewers.dashboard_data import load_dashboard_data


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit("Streamlit is optional. Install with `pip install -e '.[dev,demo]'`.") from exc

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=Path("outputs/public"))
    args, _ = parser.parse_known_args()
    data = load_dashboard_data(args.root)

    st.set_page_config(page_title="Energy Strategy Performance Kit", layout="wide")
    st.title("Energy Strategy Performance Kit")
    st.caption(
        "A plain-English view of which processor is fastest, whether training is meeting goals, "
        "and how the strategy model compares with baselines."
    )
    if data["missing"]:
        st.warning("Some artifacts are missing. Run the suggested commands below to fill the dashboard.")
        st.code(
            "python examples/calibrate_device.py\n"
            "python examples/compare_baselines.py\n"
            "python examples/run_multiseed_protocol.py --seeds 0 1 --episodes 2 --no-ppo",
            language="bash",
        )

    resource_tab, convergence_tab, quality_tab = st.tabs(["Resource", "Convergence", "Quality"])
    with resource_tab:
        _render_resource(st, data["resource"])
    with convergence_tab:
        _render_convergence(st, data["convergence"])
    with quality_tab:
        _render_quality(st, data["quality"])


def _render_resource(st: Any, rows: list[dict[str, Any]]) -> None:
    st.header("Right Processor For Each Job")
    st.write("Lower wall-clock time is better. The recommendation is empirical, not assumed from hardware specs.")
    if not rows:
        st.info("No device calibration found yet. Run `python examples/calibrate_device.py`.")
        return
    cols = st.columns(min(4, len(rows)))
    for idx, row in enumerate(rows):
        with cols[idx % len(cols)]:
            st.metric(row["job"], row["recommended_device"].upper(), f"{float(row['speedup']):.2f}x MPS/CPU")
            st.caption(row["explanation"])
    st.pyplot(_bar_figure(rows, label_key="job", value_keys=("cpu_ms", "mps_ms"), title="Milliseconds per job"))
    st.json(rows, expanded=False)


def _render_convergence(st: Any, payload: dict[str, Any]) -> None:
    st.header("Convergence Toward Goals")
    st.write("The goal badge shows whether a metric reached the target with patience, not just one lucky episode.")
    curves = payload.get("curves", [])
    badges = payload.get("badges", {})
    if not curves:
        st.info("No training iterations found yet. Run a logged strategy loop or the multiseed protocol.")
        return
    badge_cols = st.columns(3)
    for idx, key in enumerate(("goal_rate", "episode_return", "ebm_energy_gap")):
        badge = badges.get(key, {})
        label = "Reached" if badge.get("converged") else "Not yet"
        with badge_cols[idx]:
            st.metric(key, label, f"final={float(badge.get('final_value', 0.0)):.3f}")
            st.caption(f"Target: {float(badge.get('target', 0.0)):.3f}")
    st.pyplot(_line_figure(curves, "iteration", ("episode_return", "goal_rate", "win_rate"), "Outcome curves"))
    st.pyplot(_line_figure(curves, "iteration", ("policy_loss", "world_model_loss", "ebm_loss"), "Loss curves"))
    st.pyplot(_line_figure(curves, "iteration", ("ebm_energy_gap",), "EBM negative-minus-positive energy gap"))


def _render_quality(st: Any, rows: list[dict[str, Any]]) -> None:
    st.header("Model Quality Vs Baselines")
    st.write(
        "This panel is intentionally honest: the direct-goal heuristic can be a stronger sanity-check baseline than "
        "the current research scaffold."
    )
    if not rows:
        st.info("No baseline metrics found yet. Run `python examples/compare_baselines.py`.")
        return
    st.pyplot(_bar_figure(rows, label_key="baseline", value_keys=("episode_return", "win_rate"), title="Quality metrics"))
    st.json(rows, expanded=False)


def _bar_figure(rows: list[dict[str, Any]], label_key: str, value_keys: tuple[str, ...], title: str) -> plt.Figure:
    labels = [str(row[label_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    width = 0.8 / max(1, len(value_keys))
    offsets = [idx - (len(value_keys) - 1) / 2 for idx in range(len(value_keys))]
    for offset, key in zip(offsets, value_keys, strict=True):
        values = [float(row[key]) if row.get(key) is not None else 0.0 for row in rows]
        positions = [idx + offset * width for idx in range(len(labels))]
        ax.bar(positions, values, width=width, label=key)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def _line_figure(rows: list[dict[str, Any]], x_key: str, y_keys: tuple[str, ...], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [float(row[x_key]) for row in rows]
    for key in y_keys:
        y = [row.get(key) for row in rows]
        if all(value is None for value in y):
            continue
        ax.plot(x, [float(value) if value is not None else float("nan") for value in y], marker="o", label=key)
    ax.set_xlabel(x_key)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
