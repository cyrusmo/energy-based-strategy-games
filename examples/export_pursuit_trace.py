"""Export a validated custom pursuit/evasion trace for public demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strategy_games.rollouts import pursuit_rollout_config_from_mapping, run_scripted_pursuit_rollout
from strategy_games.traces import pursuit_summary_to_dict, save_pursuit_summary, save_pursuit_trace
from strategy_games.utils.config import load_config
from strategy_games.viewers import plot_pursuit_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a validated pursuit/evasion trace.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo/custom_2_evader_9x9.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Skip trajectory PNG export.")
    args = parser.parse_args()

    raw = load_config(args.config)
    config = pursuit_rollout_config_from_mapping(raw, config_path=str(args.config))
    trace = run_scripted_pursuit_rollout(config)

    output_raw = raw.get("output", {})
    output_dir = args.output_dir or Path(output_raw.get("dir", "outputs/public/pursuit_demo"))
    trace_path = output_dir / str(output_raw.get("trace_filename", "trace.json"))
    summary_path = output_dir / str(output_raw.get("summary_filename", "summary.json"))
    plot_path = output_dir / str(output_raw.get("trajectory_filename", "trajectory.png"))

    save_pursuit_trace(trace, trace_path)
    save_pursuit_summary(trace.summary, summary_path)
    if not args.no_plot:
        plot_pursuit_trace(trace, plot_path)

    print(f"trace_json={trace_path}")
    print(f"summary_json={summary_path}")
    if not args.no_plot:
        print(f"trajectory_png={plot_path}")
    print(json.dumps(pursuit_summary_to_dict(trace.summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
