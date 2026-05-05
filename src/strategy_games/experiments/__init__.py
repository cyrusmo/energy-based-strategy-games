"""Experiment runner and metrics helpers."""

from strategy_games.experiments.baselines import compare_baselines, format_baseline_table
from strategy_games.experiments.logging import ExperimentLogger
from strategy_games.experiments.metrics import compute_strategy_diversity
from strategy_games.experiments.payoff import compute_payoff_matrix, format_payoff_matrix
from strategy_games.experiments.runner import run_from_config
from strategy_games.experiments.visualization import RolloutTrace, collect_heuristic_trace, format_trace_text, plot_trace

__all__ = [
    "ExperimentLogger",
    "RolloutTrace",
    "collect_heuristic_trace",
    "compare_baselines",
    "compute_payoff_matrix",
    "compute_strategy_diversity",
    "format_baseline_table",
    "format_payoff_matrix",
    "format_trace_text",
    "plot_trace",
    "run_from_config",
]
