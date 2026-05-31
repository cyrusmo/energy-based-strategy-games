"""Calibrate CPU vs Apple MPS for the small experiment jobs in this repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strategy_games.utils.calibration import calibrate_devices
from strategy_games.utils.device import DEFAULT_CALIBRATION_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU and MPS for repo workloads.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    result = calibrate_devices(warmup=args.warmup, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"calibration_json={args.output}")


if __name__ == "__main__":
    main()
