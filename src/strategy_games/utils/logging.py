"""Logging setup for examples and experiments."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a concise root logger and return the project logger."""

    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger("strategy_games")
