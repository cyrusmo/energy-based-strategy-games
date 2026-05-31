"""Optional Streamlit viewer for pursuit/evasion trace artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from strategy_games.policies.scripted_pursuit import EVADER_POLICIES, PURSUER_POLICIES
from strategy_games.rollouts import run_scripted_pursuit_rollout
from strategy_games.viewers.pursuit_trace import (
    active_config_from_trace as _active_config_from_trace,
    default_draft_config as _default_draft_config,
    rollout_config_from_draft as _rollout_config_from_draft,
)
from strategy_games.traces import load_pursuit_trace
from strategy_games.viewers import (
    COMPARISON_GENERATION_COMMAND,
    agent_status_rows,
    clamp_frame_index,
    comparison_diagnostics_summary,
    evader_effectiveness_summary,
    load_policy_comparison_state,
    pursuer_effectiveness_summary,
    step_frame_index,
    styled_grid_html,
    trace_metadata,
    trace_table_rows,
    transition_context,
)

DEFAULT_TRACE_PATH = "examples/fixtures/pursuit_trace_2_evader_9x9.json"
DEFAULT_COMPARISON_PATH = "outputs/public/pursuit_demo/policy_comparison.json"


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit("Streamlit is optional. Install with `pip install -e '.[dev,demo]'`.") from exc

    st.set_page_config(page_title="Pursuit Trace Viewer", layout="wide")
    _initialize_state(st)

    st.title("Pursuit Trace Viewer")
    st.caption(
        "This viewer is intended for inspecting environment dynamics, scripted policy behavior, "
        "and trace-level metrics. It does not demonstrate learned robustness, optimality, "
        "or exact game-theoretic guarantees."
    )

    _render_sidebar(st)
    _render_trace_source_controls(st)
    active_trace = st.session_state.active_trace
    if active_trace is None:
        st.info("No trace loaded. Load a saved trace or commit the draft config with Run Rollout.")
    else:
        st.session_state.frame_index = clamp_frame_index(active_trace, st.session_state.frame_index)
        _render_trace_metadata(st, active_trace)
        _render_role_summaries(st, active_trace)
        _render_frame_view(st, active_trace)
        _render_trace_tables(st, active_trace)
    _render_head_to_head(st)


def _initialize_state(st: Any) -> None:
    if "active_trace" not in st.session_state:
        st.session_state.active_trace = None
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0
    if "viewer_mode" not in st.session_state:
        st.session_state.viewer_mode = "saved_trace"
    if "trace_path" not in st.session_state:
        st.session_state.trace_path = DEFAULT_TRACE_PATH
    if "comparison_path" not in st.session_state:
        st.session_state.comparison_path = DEFAULT_COMPARISON_PATH
    if "comparison_artifact" not in st.session_state:
        st.session_state.comparison_artifact = None
    if "comparison_error" not in st.session_state:
        st.session_state.comparison_error = None
    if "comparison_loaded_once" not in st.session_state:
        state = load_policy_comparison_state(st.session_state.comparison_path)
        st.session_state.comparison_path = state.comparison_path or st.session_state.comparison_path
        st.session_state.comparison_artifact = state.comparison_artifact
        st.session_state.comparison_error = state.comparison_error
        st.session_state.comparison_loaded_once = True
    if "draft_config" not in st.session_state:
        st.session_state.draft_config = _default_draft_config()
    if "active_trace_config" not in st.session_state:
        st.session_state.active_trace_config = None
    if "trace_error" not in st.session_state:
        st.session_state.trace_error = None


def _render_sidebar(st: Any) -> None:
    st.sidebar.header("Trace Source")
    st.session_state.viewer_mode = st.sidebar.radio(
        "Mode",
        options=("saved_trace", "live_rollout"),
        format_func=lambda value: "Saved trace" if value == "saved_trace" else "Live rollout",
        horizontal=False,
    )
    st.session_state.trace_path = st.sidebar.text_input("Trace JSON", value=st.session_state.trace_path)
    st.session_state.comparison_path = st.sidebar.text_input(
        "Comparison JSON",
        value=st.session_state.comparison_path,
    )
    if st.sidebar.button("Load Comparison", use_container_width=True):
        _load_comparison(st)

    st.sidebar.divider()
    st.sidebar.header("Draft Config")
    draft = dict(st.session_state.draft_config)
    draft["seed"] = int(st.sidebar.number_input("Seed", min_value=0, value=int(draft["seed"]), step=1))
    draft["grid_size"] = [
        int(st.sidebar.number_input("Grid height", min_value=2, value=int(draft["grid_size"][0]), step=1)),
        int(st.sidebar.number_input("Grid width", min_value=2, value=int(draft["grid_size"][1]), step=1)),
    ]
    draft["num_evaders"] = int(st.sidebar.number_input("Evaders", min_value=1, value=int(draft["num_evaders"]), step=1))
    draft["num_pursuers"] = int(
        st.sidebar.number_input("Pursuers", min_value=1, value=int(draft["num_pursuers"]), step=1)
    )
    draft["max_steps"] = int(st.sidebar.number_input("Max steps", min_value=1, value=int(draft["max_steps"]), step=1))
    draft["catch_radius"] = int(
        st.sidebar.number_input("Catch radius", min_value=0, value=int(draft["catch_radius"]), step=1)
    )
    draft["pursuer_policy"] = st.sidebar.selectbox(
        "Pursuer policy",
        options=PURSUER_POLICIES,
        index=PURSUER_POLICIES.index(str(draft["pursuer_policy"])),
    )
    draft["evader_policy"] = st.sidebar.selectbox(
        "Evader policy",
        options=EVADER_POLICIES,
        index=EVADER_POLICIES.index(str(draft["evader_policy"])),
    )
    draft["feint_steps"] = int(
        st.sidebar.number_input("Feint steps", min_value=0, value=int(draft["feint_steps"]), step=1)
    )
    st.session_state.draft_config = draft


def _render_trace_source_controls(st: Any) -> None:
    source_tab, config_tab = st.tabs(["Trace Controls", "Config State"])
    with source_tab:
        cols = st.columns(4)
        if cols[0].button("Load Trace", use_container_width=True):
            _load_trace(st)
        if cols[1].button("Run Rollout", use_container_width=True):
            _run_rollout(st)
        if cols[2].button("Reset Frame", use_container_width=True):
            st.session_state.frame_index = 0
        if cols[3].button("Reload Comparison", use_container_width=True):
            _load_comparison(st)
        if st.session_state.trace_error:
            st.error(st.session_state.trace_error)
    with config_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Draft Config")
            st.json(st.session_state.draft_config)
        with right:
            st.subheader("Active Trace Config")
            if st.session_state.active_trace_config is None:
                st.info("No active trace config yet. Run a rollout or load a trace.")
            else:
                st.json(st.session_state.active_trace_config)


def _render_trace_metadata(st: Any, trace: Any) -> None:
    metadata = trace_metadata(trace)
    st.subheader("Trace Metadata")
    cols = st.columns(5)
    fields = [
        ("Env", metadata["env_id"]),
        ("Episode", metadata["episode_id"]),
        ("Seed", metadata["seed"]),
        ("Grid", metadata["grid_size"]),
        ("Agents", f"{metadata['num_pursuers']} pursuer / {metadata['num_evaders']} evader"),
        ("Max Steps", metadata["max_steps"]),
        ("Catch Radius", metadata["catch_radius"]),
        ("Outcome", metadata["outcome"]),
        ("Reason", metadata["terminated_reason"]),
    ]
    for idx, (label, value) in enumerate(fields):
        cols[idx % len(cols)].metric(label, str(value))


def _render_role_summaries(st: Any, trace: Any) -> None:
    pursuer = pursuer_effectiveness_summary(trace)
    evader = evader_effectiveness_summary(trace)
    left, right = st.columns(2)
    with left:
        st.subheader("Pursuer Effectiveness")
        cols = st.columns(4)
        cols[0].metric("Capture Rate", f"{float(pursuer['capture_rate']):.2f}")
        cols[1].metric("All Captured", str(pursuer["all_evaders_captured"]))
        cols[2].metric("Mean Return", f"{float(pursuer['mean_pursuer_return']):.2f}")
        cols[3].metric("Episode Steps", str(pursuer["average_episode_steps"]))
        st.caption(str(pursuer["interpretation"]))
    with right:
        st.subheader("Evader Effectiveness")
        cols = st.columns(4)
        cols[0].metric("Survival Rate", f"{float(evader['survival_rate']):.2f}")
        cols[1].metric("Timeout/Survival", str(evader["timeout_or_survival_outcome"]))
        cols[2].metric("Mean Return", f"{float(evader['mean_evader_return']):.2f}")
        cols[3].metric("Episode Steps", str(evader["average_episode_steps"]))
        st.caption(str(evader["interpretation"]))


def _render_frame_view(st: Any, trace: Any) -> None:
    st.subheader("Frame Inspector")
    max_frame = len(trace.steps)
    controls = st.columns([1, 1, 1, 4])
    if controls[0].button("Previous", use_container_width=True):
        st.session_state.frame_index = step_frame_index(trace, st.session_state.frame_index, -1)
    if controls[1].button("Next", use_container_width=True):
        st.session_state.frame_index = step_frame_index(trace, st.session_state.frame_index, 1)
    if controls[2].button("Reset", use_container_width=True):
        st.session_state.frame_index = 0
    st.session_state.frame_index = controls[3].slider(
        "Frame",
        min_value=0,
        max_value=max_frame,
        value=clamp_frame_index(trace, st.session_state.frame_index),
        help="Frame 0 is the initial state; later frames show post-transition positions.",
    )

    context = transition_context(trace, st.session_state.frame_index)
    grid_col, context_col = st.columns([1.1, 1])
    with grid_col:
        st.markdown(styled_grid_html(trace, st.session_state.frame_index), unsafe_allow_html=True)
    with context_col:
        st.subheader("Transition Context")
        st.json(context)


def _render_trace_tables(st: Any, trace: Any) -> None:
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Agent Status")
        st.dataframe(agent_status_rows(trace), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Trace Steps")
        st.dataframe(trace_table_rows(trace), use_container_width=True, hide_index=True)


def _render_head_to_head(st: Any) -> None:
    st.subheader("Head-To-Head Diagnostics")
    state = _comparison_state_from_session(st)
    summary = comparison_diagnostics_summary(state)
    if not summary["loaded"]:
        st.info(str(summary["error"]))
        st.code(COMPARISON_GENERATION_COMMAND)
        return

    st.caption(f"Loaded from {summary['source_path']}")
    cols = st.columns(4)
    cols[0].metric("Payoff Metric", str(summary["payoff_metric"]))
    cols[1].metric("Row Player", str(summary["row_player"]))
    cols[2].metric("Column Player", str(summary["column_player"]))
    cols[3].metric("Maximin Policy", str(summary["maximin_policy"]))
    st.write(str(summary["payoff_orientation"]))

    diag_cols = st.columns(3)
    with diag_cols[0]:
        st.write("Row payoff vs uniform column mixture")
        st.json(summary["row_payoff_vs_uniform_column_mixture"])
    with diag_cols[1]:
        st.write("Empirical regret vs uniform column mixture")
        st.json(summary["empirical_regret_vs_uniform_column_mixture"])
    with diag_cols[2]:
        st.write("Payoff-weighted row-policy ranking distribution")
        st.json(summary["ranking_distribution"])
    st.caption(str(summary["notes"]))


def _load_trace(st: Any) -> None:
    try:
        trace = load_pursuit_trace(Path(st.session_state.trace_path))
    except (OSError, TypeError, ValueError) as exc:
        st.session_state.trace_error = f"Could not load trace: {exc}"
        return
    st.session_state.active_trace = trace
    st.session_state.active_trace_config = _active_config_from_trace(trace)
    st.session_state.frame_index = 0
    st.session_state.trace_error = None


def _run_rollout(st: Any) -> None:
    try:
        config = _rollout_config_from_draft(st.session_state.draft_config)
        trace = run_scripted_pursuit_rollout(config)
    except (TypeError, ValueError) as exc:
        st.session_state.trace_error = f"Could not run rollout: {exc}"
        return
    st.session_state.active_trace = trace
    st.session_state.active_trace_config = dict(st.session_state.draft_config)
    st.session_state.frame_index = 0
    st.session_state.trace_error = None


def _load_comparison(st: Any) -> None:
    state = load_policy_comparison_state(st.session_state.comparison_path)
    st.session_state.comparison_path = state.comparison_path or st.session_state.comparison_path
    st.session_state.comparison_artifact = state.comparison_artifact
    st.session_state.comparison_error = state.comparison_error


def _comparison_state_from_session(st: Any) -> Any:
    from strategy_games.viewers import ComparisonState

    return ComparisonState(
        comparison_path=st.session_state.comparison_path,
        comparison_artifact=st.session_state.comparison_artifact,
        comparison_error=st.session_state.comparison_error,
    )


if __name__ == "__main__":
    main()
