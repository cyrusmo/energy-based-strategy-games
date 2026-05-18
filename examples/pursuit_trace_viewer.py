"""Optional Streamlit viewer for pursuit/evasion trace artifacts."""

from __future__ import annotations

from pathlib import Path

from strategy_games.rollouts import PursuitRolloutConfig, run_scripted_pursuit_rollout
from strategy_games.traces import load_pursuit_trace
from strategy_games.viewers import agent_status_rows, grid_frame, summary_metrics, trace_table_rows


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit("Streamlit is optional. Install with `pip install -e '.[dev,demo]'`.") from exc

    st.set_page_config(page_title="Pursuit Trace Viewer", layout="wide")
    st.title("Pursuit Trace Viewer")
    st.caption(
        "This viewer is intended for inspecting environment dynamics, scripted policy behavior, "
        "and trace-level metrics. It does not demonstrate learned robustness, optimality, "
        "or exact game-theoretic guarantees."
    )

    default_path = Path("examples/fixtures/pursuit_trace_2_evader_9x9.json")
    trace_path = Path(st.sidebar.text_input("Trace JSON", value=str(default_path)))
    use_live_rollout = st.sidebar.checkbox("Run fresh custom rollout", value=False)

    if use_live_rollout:
        seed = st.sidebar.number_input("Seed", min_value=0, value=7, step=1)
        trace = run_scripted_pursuit_rollout(PursuitRolloutConfig(seed=int(seed)))
    else:
        trace = load_pursuit_trace(trace_path)

    metrics = summary_metrics(trace)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Outcome", str(metrics["outcome"]))
    metric_cols[1].metric("Capture Rate", f"{float(metrics['capture_rate']):.2f}")
    metric_cols[2].metric("Survival Rate", f"{float(metrics['survival_rate']):.2f}")
    metric_cols[3].metric("Steps", str(metrics["total_steps"]))
    metric_cols[4].metric("Evaders", str(metrics["num_evaders"]))

    frame_index = st.slider("Frame", min_value=0, max_value=len(trace.steps), value=0)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Grid")
        st.dataframe(grid_frame(trace, frame_index), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Agent Status")
        st.dataframe(agent_status_rows(trace), use_container_width=True, hide_index=True)

    st.subheader("Trace Steps")
    st.dataframe(trace_table_rows(trace), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
