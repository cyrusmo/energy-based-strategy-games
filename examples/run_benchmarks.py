"""Run public benchmark suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strategy_games.benchmarks.runner import run_benchmark_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a public benchmark suite.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/debug_suite.yaml"),
        help="Benchmark YAML config.",
    )
    args = parser.parse_args()
    result = run_benchmark_from_config(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
