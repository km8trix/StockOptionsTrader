"""
Structured logging configuration for the trading system.

Usage:
    from utils.logging_config import setup_logging
    setup_logging()  # level read from LOG_LEVEL env var, default INFO

Individual modules should then create their own logger:
    import logging
    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import os

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging for the application.

    Args:
        level: explicit logging level (name like "DEBUG" or numeric constant).
            When None, the level is read from the LOG_LEVEL environment
            variable, defaulting to INFO. Unknown level names fall back to
            INFO rather than raising.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.strip().upper(), logging.INFO)
        if not isinstance(level, int):
            level = logging.INFO

    logging.basicConfig(level=level, format=LOG_FORMAT)
