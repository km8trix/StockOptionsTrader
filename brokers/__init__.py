"""Broker implementations and the ExecutionBroker interface."""

from __future__ import annotations

import logging

from brokers.base import ExecutionBroker
from brokers.paper_trader import PaperTrader

logger = logging.getLogger(__name__)

try:
    from brokers.live_trader import LiveEtradeBroker
except ImportError as e:
    # Live trading deps (requests-oauthlib) may be missing; fail loudly in the
    # logs but keep paper trading importable.
    logger.warning("LiveEtradeBroker unavailable: %s", e)
    LiveEtradeBroker = None  # type: ignore[assignment, misc]

__all__ = ['ExecutionBroker', 'PaperTrader', 'LiveEtradeBroker']
