"""Compute and save a named-strategy payoff matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from strategy_games.experiments.payoff import compute_payoff_matrix, format_payoff_matrix, save_payoff_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute approximate strategy-vs-opponent payoff matrix.")
    parser.add_argument("--strategy-dim", type=int, default=8)
    parser.add_argument("--episodes-per-opponent", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/public/payoff_matrix/matrix.json"))
    parser.add_argument("--no-save", action="store_true", help="Print only; do not write JSON.")
    args = parser.parse_args()

    matrix = compute_payoff_matrix(
        strategy_dim=args.strategy_dim,
        episodes_per_opponent=args.episodes_per_opponent,
    )
    print(format_payoff_matrix(matrix), end="")
    if not args.no_save:
        path = save_payoff_matrix(matrix, args.output)
        print(f"matrix_json={path}")


if __name__ == "__main__":
    main()
