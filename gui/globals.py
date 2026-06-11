# gui/globals.py
"""Shared singletons for the GUI layer.

The TradingDatabase used to be instantiated at import time, which wrote
``trading_data.db`` into whatever CWD imported this module (including test
runs). It is now created lazily via :func:`get_db` on first use and honors
the ``TRADING_DB_PATH`` environment variable.
"""
from __future__ import annotations

import os

from data.market_data import MarketDataHandler
from utils.alerts import AlertManager
from portfolio.risk_manager import RiskManager

# These constructors are cheap and side-effect free (no files, no network),
# so they keep their import-time initialization and existing
# ``from gui.globals import market_data`` callers work unchanged.
market_data = MarketDataHandler()
alert_manager = AlertManager()
risk_manager = RiskManager()

_db = None


def get_db():
    """Return the shared TradingDatabase, creating it on first use.

    Honors the ``TRADING_DB_PATH`` environment variable so deployments and
    tests can redirect the SQLite file away from the process CWD. The value
    is passed explicitly so this works whether or not TradingDatabase's own
    default reads the same variable.
    """
    global _db
    if _db is None:
        from utils.database import TradingDatabase

        db_path = os.environ.get('TRADING_DB_PATH')
        _db = TradingDatabase(db_path) if db_path else TradingDatabase()
    return _db


def __getattr__(name: str):
    """Back-compat: ``from gui.globals import db`` resolves lazily."""
    if name == 'db':
        return get_db()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
